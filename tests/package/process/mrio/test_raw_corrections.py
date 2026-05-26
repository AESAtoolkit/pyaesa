import importlib
from pathlib import Path
import runpy
import shutil
import sys
import zipfile
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa.process.mrios.utils.raw_corrections.basis import positive_factor_inputs_basis_from_frame
from pyaesa.process.mrios.utils.raw_corrections.maintainer_cli import (
    main as build_corrected_values_main,
    parse_build_corrected_values_args,
)
from pyaesa.process.mrios.utils.raw_corrections.exio_3102_corrected_values import (
    FULL_YEAR_RANGE,
    _WorkspaceData,
    _build_tw_rows,
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
from tests.package.helpers.data_processing_dummy import DummyExtension, build_dummy_iosystem


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _eu27_regions() -> list[str]:
    frame = pd.read_csv(
        _repo_root()
        / "pyaesa"
        / "workspace_initialisation"
        / "prerequisites"
        / "mrio"
        / "exiobase_3"
        / "aggregation"
        / "agg_reg_eu27.csv",
        encoding="latin1",
    )
    return sorted(
        {
            str(value).strip()
            for value in frame.loc[frame["aggregated_mrio"].eq("EU27"), "original_classification"]
            if str(value).strip() and str(value).strip() != "MT"
        }
    )


def _fwu_stressors() -> list[str]:
    frame = pd.read_csv(
        _repo_root()
        / "pyaesa"
        / "workspace_initialisation"
        / "prerequisites"
        / "mrio"
        / "exiobase_3"
        / "lcia"
        / "characterization_factors_matrices"
        / "pb_lcia.csv",
        encoding="latin1",
    )
    mask = frame["extension"].astype(str).str.strip().eq("water") & frame["impact_parent"].astype(
        str
    ).str.strip().eq("FWU")
    return sorted(
        {str(value).strip() for value in frame.loc[mask, "stressor"] if str(value).strip()}
    )


def _multiindex_columns(regions: list[str], sectors: list[str]) -> pd.MultiIndex:
    tuples = [(region, sector) for region in regions for sector in sectors]
    return pd.MultiIndex.from_tuples(tuples, names=["region", "sector"]).sort_values()


def _build_workspace(tmp_path: Path, system: str) -> Path:
    workspace_root = tmp_path / "workspace"
    raw_dir = (
        workspace_root
        / "pyaesa"
        / "data_raw"
        / "mrio"
        / "exiobase_3"
        / "exiobase_3102"
        / f"full_{system}"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    eu27_regions = _eu27_regions()
    regions = list(dict.fromkeys(["CN", "TW", "MT", "CH", "LU", *eu27_regions]))
    sectors = ["S_donor", "S_skip", "S_ch", "S_lu"]
    columns = _multiindex_columns(regions, sectors)
    fwu_stressors = _fwu_stressors()
    donor_eu27_regions = [region for region in eu27_regions if region not in {"CH", "LU"}]

    for year in FULL_YEAR_RANGE:
        land = pd.DataFrame(0.0, index=pd.Index(["Forest"], name="stressor"), columns=columns)
        nutrients = pd.DataFrame(
            0.0,
            index=pd.Index(["P - agriculture - water", "P - waste - water"], name="stressor"),
            columns=columns,
        )
        water = pd.DataFrame(
            0.0,
            index=pd.Index(fwu_stressors, name="stressor"),
            columns=columns,
        )
        factor_inputs = pd.DataFrame(
            0.0,
            index=pd.Index(["fi_pos", "fi_neg"], name="stressor"),
            columns=columns,
        )

        land.loc["Forest", ("CN", "S_donor")] = 10.0
        land.loc["Forest", ("TW", "S_donor")] = 0.0
        for region in donor_eu27_regions:
            land.loc["Forest", (region, "S_donor")] = 8.0
            land.loc["Forest", (region, "S_skip")] = 5.0
        land.loc["Forest", ("MT", "S_donor")] = 1.0 if year in (2010, 2011) else 0.0

        factor_inputs.loc["fi_pos", ("CN", "S_donor")] = 2.0
        factor_inputs.loc["fi_pos", ("TW", "S_donor")] = 6.0
        factor_inputs.loc["fi_pos", ("MT", "S_donor")] = 4.0
        factor_inputs.loc["fi_neg", ("MT", "S_skip")] = -3.0
        for region in donor_eu27_regions:
            factor_inputs.loc["fi_pos", (region, "S_donor")] = 4.0
            factor_inputs.loc["fi_pos", (region, "S_skip")] = 3.0

        ch_x = float(year - 1990)
        factor_inputs.loc["fi_pos", ("CH", "S_ch")] = ch_x
        nutrients.loc["P - agriculture - water", ("CH", "S_ch")] = 2.0 * ch_x
        nutrients.loc["P - waste - water", ("CH", "S_ch")] = 3.0 * ch_x
        if year in (2021, 2022):
            nutrients.loc["P - agriculture - water", ("CH", "S_ch")] = 0.0
            nutrients.loc["P - waste - water", ("CH", "S_ch")] = 0.0

        lu_x = float(year - 1990)
        factor_inputs.loc["fi_pos", ("LU", "S_lu")] = lu_x
        for stressor in fwu_stressors:
            water.loc[stressor, ("LU", "S_lu")] = 4.0 * lu_x
        if year == 2020:
            for stressor in fwu_stressors:
                water.loc[stressor, ("LU", "S_lu")] = 0.0

        zip_path = raw_dir / f"IOT_{year}_{system}.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for extension_name, frame in {
                "land": land,
                "nutrients": nutrients,
                "water": water,
                "factor_inputs": factor_inputs,
            }.items():
                archive.writestr(f"{extension_name}/F.txt", frame.to_csv(sep="\t"))
    return workspace_root


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


def _zip_path(*, workspace_root: Path, system: str, year: int) -> Path:
    return (
        workspace_root
        / "pyaesa"
        / "data_raw"
        / "mrio"
        / "exiobase_3"
        / "exiobase_3102"
        / f"full_{system}"
        / f"IOT_{year}_{system}.zip"
    )


def _read_extension_frame(
    *, workspace_root: Path, system: str, year: int, extension: str
) -> pd.DataFrame:
    with zipfile.ZipFile(
        _zip_path(workspace_root=workspace_root, system=system, year=year)
    ) as archive:
        with archive.open(f"{extension}/F.txt") as raw:
            return pd.read_csv(
                raw,
                sep="\t",
                index_col=0,
                header=[0, 1],
                encoding="latin1",
            )


def _rewrite_extension_frame(
    *,
    workspace_root: Path,
    system: str,
    year: int,
    extension: str,
    frame: pd.DataFrame,
) -> None:
    zip_path = _zip_path(workspace_root=workspace_root, system=system, year=year)
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if name != f"{extension}/F.txt":
                entries[name] = archive.read(name)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr(f"{extension}/F.txt", frame.to_csv(sep="\t"))


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

    rows, diagnostic_rows = _build_tw_rows(
        _WorkspaceData(workspace_root=workspace_root, source="exiobase_3102_ixi"),
        source="exiobase_3102_ixi",
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
        _build_tw_rows(
            _WorkspaceData(workspace_root=workspace_root, source="exiobase_3102_ixi"),
            source="exiobase_3102_ixi",
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
    setattr(iosys, "land", object())
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
