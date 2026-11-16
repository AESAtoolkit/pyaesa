import importlib
from pathlib import Path
import runpy
import shutil
import sys
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa.process.mrios.utils.raw_corrections.basis import positive_factor_inputs_basis_from_frame
from pyaesa.process.mrios.utils.raw_corrections.maintainer_cli import (
    main as build_corrected_values_main,
    parse_build_corrected_values_args,
)
from pyaesa.process.mrios.utils.raw_corrections.exio_corrected_values import (
    _build_forest_rows,
)
from pyaesa.process.mrios.utils.raw_corrections.runtime import (
    _apply_correction_rows_to_iosys,
    _format_year_ranges,
    _format_log_rows,
    _stressor_family_label,
    _require_extension_frame,
    apply_raw_corrected_values,
    load_raw_corrected_value_rows,
    summarize_correction_rows,
    summarize_correction_scopes,
    write_applied_correction_log,
)
from pyaesa.process.mrios.utils.io.paths import _get_mrio_raw_corrected_values_log_path
from pyaesa.process.mrios.utils.raw_corrections.workspace_data import _WorkspaceData
from tests.package.helpers.raw_corrections import (
    _build_workspace,
    _fwu_stressors,
    _read_extension_frame,
    _rewrite_extension_frame,
)
from tests.package.helpers.data_processing_dummy import DummyExtension, build_dummy_iosystem


@pytest.fixture(scope="module")
def ixi_workspace_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the reusable EXIOBASE raw correction archive scaffold once."""
    return _build_workspace(tmp_path_factory.mktemp("raw_corrections_template"), "ixi")


@pytest.fixture
def ixi_workspace(tmp_path: Path, ixi_workspace_template: Path) -> Path:
    """Return one isolated copy of the reusable EXIOBASE raw correction scaffold."""
    workspace_root = tmp_path / "workspace"
    shutil.copytree(ixi_workspace_template, workspace_root)
    return workspace_root


def _minimal_corrected_values_frame(*, fwu_stressors: list[str]) -> pd.DataFrame:
    """Return minimal corrected value rows for runtime application tests."""
    return pd.DataFrame(
        [
            {
                "source": "exiobase_3102_ixi",
                "extension": "land",
                "stressor": "Forest",
                "region": "TW",
                "sector": "S_donor",
                "year": 1995,
                "original_value": 0.0,
                "corrected_value": 30.0,
                "correction_method": "donor_sector_intensity",
                "fit_window": "",
                "correction_reason": "TW correction: missing extension data recorded at 0.",
                "replaced_nonzero_source": False,
                "prediction_clipped_to_zero": False,
            },
            {
                "source": "exiobase_3102_ixi",
                "extension": "land",
                "stressor": "Forest",
                "region": "MT",
                "sector": "S_donor",
                "year": 1995,
                "original_value": 0.0,
                "corrected_value": 8.0,
                "correction_method": "donor_sector_intensity",
                "fit_window": "",
                "correction_reason": "MT correction: donor reconstruction.",
                "replaced_nonzero_source": False,
                "prediction_clipped_to_zero": False,
            },
            {
                "source": "exiobase_3102_ixi",
                "extension": "nutrients",
                "stressor": "P - agriculture - water",
                "region": "CH",
                "sector": "S_ch",
                "year": 1995,
                "original_value": 0.0,
                "corrected_value": 62.0,
                "correction_method": "ols_level",
                "fit_window": "1995-2020",
                "correction_reason": "CH correction: extension data are incoherently too small.",
                "replaced_nonzero_source": False,
                "prediction_clipped_to_zero": False,
            },
            {
                "source": "exiobase_3102_ixi",
                "extension": "water",
                "stressor": fwu_stressors[0],
                "region": "LU",
                "sector": "S_lu",
                "year": 1995,
                "original_value": 0.0,
                "corrected_value": 120.0,
                "correction_method": "ols_level",
                "fit_window": "1995-2022",
                "correction_reason": "LU correction: extension data are incoherently too small.",
                "replaced_nonzero_source": False,
                "prediction_clipped_to_zero": False,
            },
        ]
    )


def test_positive_factor_inputs_basis_from_frame_uses_only_positive_rows() -> None:
    columns = pd.MultiIndex.from_tuples([("R1", "S1"), ("R1", "S2")], names=["region", "sector"])
    frame = pd.DataFrame(
        [[4.0, -1.0], [-3.0, 2.0]],
        index=pd.Index(["a", "b"], name="stressor"),
        columns=columns,
    )
    result = positive_factor_inputs_basis_from_frame(frame)
    assert result.loc[("R1", "S1")] == 4.0
    assert result.loc[("R1", "S2")] == 2.0


def test_build_corrected_values_module_parses_and_runs(
    tmp_path: Path,
    ixi_workspace: Path,
) -> None:
    workspace_root = ixi_workspace
    out_dir = tmp_path / "module_out"
    args = parse_build_corrected_values_args(
        ["--workspace-root", str(workspace_root), "--source", "exiobase_3102_ixi"]
    )
    assert Path(args.workspace_root) == workspace_root
    assert args.source == ["exiobase_3102_ixi"]

    emitted: list[str] = []
    exit_code = build_corrected_values_main(
        [
            "--workspace-root",
            str(workspace_root),
            "--source",
            "exiobase_3102_ixi",
            "--out-dir",
            str(out_dir),
        ],
        emit=emitted.append,
    )
    assert exit_code == 0
    assert emitted
    assert any(str(workspace_root) in line for line in emitted)
    assert any("exiobase_3102_ixi" in line for line in emitted)
    corrected_values = pd.read_csv(out_dir / "exiobase_3102_ixi_raw_corrected_values.csv")

    tw_row = corrected_values.loc[
        (corrected_values["extension"] == "land")
        & (corrected_values["region"] == "TW")
        & (corrected_values["sector"] == "S_donor")
        & (corrected_values["year"] == 1995)
    ].iloc[0]
    assert tw_row["corrected_value"] == pytest.approx(30.0)

    mt_row = corrected_values.loc[
        (corrected_values["extension"] == "land")
        & (corrected_values["region"] == "MT")
        & (corrected_values["sector"] == "S_donor")
        & (corrected_values["year"] == 1995)
    ].iloc[0]
    assert mt_row["corrected_value"] == pytest.approx(8.0)

    ch_row = corrected_values.loc[
        (corrected_values["extension"] == "nutrients")
        & (corrected_values["stressor"] == "P - agriculture - water")
        & (corrected_values["region"] == "CH")
        & (corrected_values["sector"] == "S_ch")
        & (corrected_values["year"] == 2021)
    ].iloc[0]
    assert ch_row["corrected_value"] == pytest.approx(62.0)

    lu_row = corrected_values.loc[
        (corrected_values["extension"] == "water")
        & (corrected_values["region"] == "LU")
        & (corrected_values["sector"] == "S_lu")
        & (corrected_values["year"] == 2020)
    ].iloc[0]
    assert lu_row["corrected_value"] == pytest.approx(120.0)


def test_build_corrected_values_module_main_dunder_path() -> None:
    module_name = "pyaesa.process.mrios.utils.raw_corrections.build_corrected_values"
    sys.modules.pop(module_name, None)
    old_argv = sys.argv[:]
    sys.argv = ["build_corrected_values", "--help"]
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module(
                module_name,
                run_name="__main__",
            )
    finally:
        sys.argv = old_argv
    assert exc.value.code == 0
    assert importlib.import_module(module_name).main is build_corrected_values_main


@pytest.mark.parametrize("system", ["ixi", "pxp"])
def test_exiobase_3112_generates_and_applies_only_forest_corrections(
    tmp_path: Path, system: str
) -> None:
    source = f"exiobase_3112_{system}"
    workspace_root = _build_workspace(tmp_path, system, version="3112")
    out_dir = tmp_path / "corrected_values"
    assert (
        build_corrected_values_main(
            [
                "--workspace-root",
                str(workspace_root),
                "--source",
                source,
                "--out-dir",
                str(out_dir),
            ],
            emit=lambda _message: None,
        )
        == 0
    )
    frame = pd.read_csv(out_dir / f"{source}_raw_corrected_values.csv")
    assert frame.extension.unique().tolist() == ["land"]
    assert sorted(frame.year.unique().tolist()) == list(range(1995, 2024))
    assert sorted(frame.stressor.unique().tolist()) == [
        "Naturally generating forest - Non primary forest",
        "Planted forest",
    ]
    tw = frame.loc[frame.region.eq("TW") & frame.year.eq(1995)].set_index("stressor")
    assert tw.corrected_value.to_dict() == pytest.approx(
        {"Naturally generating forest - Non primary forest": 18.0, "Planted forest": 12.0}
    )
    assert load_raw_corrected_value_rows(
        source=source, year=2024, corrected_values_root=out_dir
    ).empty

    data = _WorkspaceData(workspace_root=workspace_root, source=source)
    land = data.load_extension_frame(year=1995, extension="land").copy()
    direct = pd.DataFrame(3.0, index=land.index, columns=["households"])
    iosys = build_dummy_iosystem(
        include_factor_inputs=False,
        include_satellite_accounts=False,
        extra_extensions={"land": DummyExtension(name="land", F=land, F_Y=direct.copy())},
    )
    summary = apply_raw_corrected_values(
        iosys=iosys, source=source, year=1995, corrected_values_root=out_dir
    )
    assert summary is not None
    assert summary.row_count == len(frame.loc[frame.year.eq(1995)])
    applied = getattr(cast(Any, iosys), "land")
    assert applied.F.loc["Planted forest", ("TW", "S_donor")] == pytest.approx(12.0)
    pd.testing.assert_frame_equal(applied.F_Y, direct)
    assert (
        apply_raw_corrected_values(
            iosys=iosys, source=source, year=2024, corrected_values_root=out_dir
        )
        is None
    )


def test_write_outputs_and_runtime_application_use_raw_corrected_values(
    tmp_path: Path,
    project_repo: Path,
) -> None:
    del project_repo
    corrected_values_root = tmp_path / "corrected_values"
    corrected_values_root.mkdir()
    corrected_values = _minimal_corrected_values_frame(fwu_stressors=_fwu_stressors())
    corrected_values_path = corrected_values_root / "exiobase_3102_ixi_raw_corrected_values.csv"
    corrected_values.to_csv(corrected_values_path, index=False)
    assert "diagnostic_type" not in corrected_values.columns
    assert "correction_reason" in corrected_values.columns

    rows = load_raw_corrected_value_rows(
        source="exiobase_3102_ixi",
        year=1995,
        corrected_values_root=corrected_values_root,
    )
    assert not rows.empty
    assert load_raw_corrected_value_rows(
        source="oecd_v2025", year=1995, corrected_values_root=corrected_values_root
    ).empty

    products = pd.MultiIndex.from_tuples(
        [("TW", "S_donor"), ("MT", "S_donor"), ("CH", "S_ch"), ("LU", "S_lu")],
        names=["region", "sector"],
    )
    land = DummyExtension(
        name="land",
        F=pd.DataFrame(
            [[0.0, 0.0, 0.0, 0.0]], index=pd.Index(["Forest"], name="stressor"), columns=products
        ),
        unit=pd.DataFrame({"unit": ["ha"]}, index=pd.Index(["Forest"], name="stressor")),
    )
    nutrients = DummyExtension(
        name="nutrients",
        F=pd.DataFrame(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            index=pd.Index(["P - agriculture - water", "P - waste - water"], name="stressor"),
            columns=products,
        ),
        unit=pd.DataFrame(
            {"unit": ["kg", "kg"]},
            index=pd.Index(["P - agriculture - water", "P - waste - water"], name="stressor"),
        ),
    )
    water = DummyExtension(
        name="water",
        F=pd.DataFrame(
            0.0,
            index=pd.Index(_fwu_stressors(), name="stressor"),
            columns=products,
        ),
        unit=pd.DataFrame(
            {"unit": ["m3"] * len(_fwu_stressors())},
            index=pd.Index(_fwu_stressors(), name="stressor"),
        ),
    )
    iosys = build_dummy_iosystem(
        include_factor_inputs=False,
        include_satellite_accounts=False,
        extra_extensions={"land": land, "nutrients": nutrients, "water": water},
    )

    summary = apply_raw_corrected_values(
        iosys=iosys,
        source="exiobase_3102_ixi",
        year=1995,
        corrected_values_root=corrected_values_root,
    )
    assert summary is not None
    assert summary.row_count == len(rows)
    assert getattr(cast(Any, iosys), "land").F.loc["Forest", ("TW", "S_donor")] == pytest.approx(
        30.0
    )

    log_path = write_applied_correction_log(
        source_key="exiobase_3102_ixi",
        matrix_version=None,
        saved_dir=tmp_path / "saved",
        year_rows=rows,
    )
    assert log_path is not None and log_path.exists()
    assert log_path == _get_mrio_raw_corrected_values_log_path("exiobase_3102_ixi")
    logged = pd.read_csv(log_path)
    assert "method_detail" in logged.columns
    assert "stressor_family" not in logged.columns
    assert "scope_summary" not in logged.columns
    assert "source" not in logged.columns
    assert "basis_label" not in logged.columns
    assert "prediction_clipped_to_zero" not in logged.columns
    assert "replaced_nonzero_source" not in logged.columns
    log_path.unlink()
    assert (
        write_applied_correction_log(
            source_key="exiobase_3102_ixi",
            matrix_version=None,
            saved_dir=tmp_path / "saved_empty",
            year_rows=pd.DataFrame(),
        )
        is None
    )

    with pytest.raises(FileNotFoundError):
        load_raw_corrected_value_rows(
            source="exiobase_3102_pxp",
            year=1995,
            corrected_values_root=tmp_path / "missing",
        )


def test_raw_correction_helpers_cover_error_and_skip_paths(
    tmp_path: Path,
    ixi_workspace: Path,
) -> None:
    workspace_root = ixi_workspace
    land_1995 = _read_extension_frame(
        workspace_root=workspace_root,
        system="ixi",
        year=1995,
        extension="land",
    )
    land_1995.loc["Forest", ("CN", "S_skip")] = 7.0
    _rewrite_extension_frame(
        workspace_root=workspace_root,
        system="ixi",
        year=1995,
        extension="land",
        frame=land_1995,
    )
    factor_1995 = _read_extension_frame(
        workspace_root=workspace_root,
        system="ixi",
        year=1995,
        extension="factor_inputs",
    )
    factor_1995.loc["fi_pos", ("TW", "S_skip")] = 0.0
    _rewrite_extension_frame(
        workspace_root=workspace_root,
        system="ixi",
        year=1995,
        extension="factor_inputs",
        frame=factor_1995,
    )

    rows, diagnostic_rows = _build_forest_rows(
        _WorkspaceData(workspace_root=workspace_root, source="exiobase_3102_ixi"),
        source="exiobase_3102_ixi",
        region="TW",
    )
    review = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    assert review.loc[
        (review["extension"] == "land")
        & (review["region"] == "TW")
        & (review["sector"] == "S_skip")
        & (review["year"] == 1995)
    ].empty
    tw_skip = diagnostics.loc[
        (diagnostics["extension"] == "land")
        & (diagnostics["region"] == "TW")
        & (diagnostics["sector"] == "S_skip")
        & (diagnostics["year"] == 1995)
        & (diagnostics["diagnostic_type"] == "skipped_target_predictor")
    ]
    assert len(tw_skip) == 1

    factor_1995.loc["fi_pos", ("CN", "S_donor")] = 0.0
    _rewrite_extension_frame(
        workspace_root=workspace_root,
        system="ixi",
        year=1995,
        extension="factor_inputs",
        frame=factor_1995,
    )
    with pytest.raises(ValueError):
        _build_forest_rows(
            _WorkspaceData(workspace_root=workspace_root, source="exiobase_3102_ixi"),
            source="exiobase_3102_ixi",
            region="TW",
        )

    empty_corrected_values_root = tmp_path / "empty_corrected_values"
    empty_corrected_values_root.mkdir()
    pd.DataFrame(columns=["year"]).to_csv(
        empty_corrected_values_root / "exiobase_3102_ixi_raw_corrected_values.csv",
        index=False,
    )
    assert load_raw_corrected_value_rows(
        source="exiobase_3102_ixi",
        year=1995,
        corrected_values_root=empty_corrected_values_root,
    ).empty

    assert summarize_correction_rows(year_rows=pd.DataFrame()) == []
    assert summarize_correction_scopes(year_rows=pd.DataFrame()) == []
    assert _format_log_rows(year_rows=pd.DataFrame()).empty
    assert (
        write_applied_correction_log(
            source_key="exiobase_3102_ixi",
            matrix_version=None,
            saved_dir=tmp_path / "saved_empty_2",
            year_rows=pd.DataFrame(),
        )
        is None
    )

    summary_rows = pd.DataFrame(
        [
            {
                "year": 2020,
                "region": "LU",
                "extension": "water",
                "stressor": "Water Consumption Blue - Livestock - geese",
                "sector": "S_lu",
                "correction_method": "ols_level",
                "correction_reason": (
                    "LU correction: 2020 extension data are incoherently too small."
                ),
                "corrected_value": 1.0,
            },
            {
                "year": 2018,
                "region": "MT",
                "extension": "land",
                "stressor": "Forest",
                "sector": "S_mt",
                "correction_method": "donor_sector_intensity",
                "correction_reason": "MT correction: donor reconstruction.",
                "corrected_value": 2.0,
            },
            {
                "year": 2016,
                "region": "ZZ",
                "extension": "misc",
                "stressor": "Other",
                "sector": "S_zz",
                "correction_method": "custom_method",
                "correction_reason": "Custom correction.",
                "corrected_value": 3.0,
            },
        ]
    )
    summaries = summarize_correction_rows(year_rows=summary_rows)
    assert _format_year_ranges([]) == "[]"
    assert _format_year_ranges([2018, 2020, 2021]) == "2018, 2020-2021"
    assert _stressor_family_label(extension="nutrients", region="CH", stressor="P total") == "P"
    assert len(summaries) == 3
    assert all(summaries)
    scopes = summarize_correction_scopes(year_rows=summary_rows)
    assert len(scopes) == 3

    iosys = build_dummy_iosystem(include_factor_inputs=False, include_satellite_accounts=False)
    with pytest.raises(ValueError):
        _require_extension_frame(iosys, "land")
    iosys = build_dummy_iosystem(
        include_factor_inputs=False,
        include_satellite_accounts=False,
        extra_extensions={"land": DummyExtension(name="land")},
    )
    with pytest.raises(ValueError):
        _require_extension_frame(iosys, "land")

    valid_land = DummyExtension(
        name="land",
        F=pd.DataFrame(
            [[0]],
            index=pd.Index(["Forest"], name="stressor"),
            columns=pd.MultiIndex.from_tuples([("TW", "S_donor")], names=["region", "sector"]),
        ),
        unit=pd.DataFrame({"unit": ["ha"]}, index=pd.Index(["Forest"], name="stressor")),
    )
    iosys = build_dummy_iosystem(
        include_factor_inputs=False,
        include_satellite_accounts=False,
        extra_extensions={"land": valid_land},
    )
    _apply_correction_rows_to_iosys(iosys=iosys, rows=pd.DataFrame())
    _apply_correction_rows_to_iosys(
        iosys=iosys,
        rows=pd.DataFrame(
            [
                {
                    "extension": "land",
                    "stressor": "Forest",
                    "region": "TW",
                    "sector": "S_donor",
                    "corrected_value": 0.25,
                }
            ]
        ),
    )
    assert getattr(cast(Any, iosys), "land").F.loc["Forest", ("TW", "S_donor")] == pytest.approx(
        0.25
    )
    assert str(getattr(cast(Any, iosys), "land").F.dtypes.iloc[0]) == "float64"
    with pytest.raises(ValueError):
        _apply_correction_rows_to_iosys(
            iosys=iosys,
            rows=pd.DataFrame(
                [
                    {
                        "extension": "land",
                        "stressor": "Missing",
                        "region": "TW",
                        "sector": "S_donor",
                        "corrected_value": 1.0,
                    }
                ]
            ),
        )
    with pytest.raises(ValueError):
        _apply_correction_rows_to_iosys(
            iosys=iosys,
            rows=pd.DataFrame(
                [
                    {
                        "extension": "land",
                        "stressor": "Forest",
                        "region": "TW",
                        "sector": "Missing",
                        "corrected_value": 1.0,
                    }
                ]
            ),
        )
