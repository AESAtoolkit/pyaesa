from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from pyaesa.process.mrios.utils.parsers.exio_characterization import (
    _build_characterization_validation,
    _build_impact_units,
    _calc_characterized_extensions_minimal,
    _characterize_exiobase_io_core,
    _collect_extensions,
    _direct_characterize_extensions,
    _find_missing_characterization_extensions as _find_missing_core,
    _normalize_label,
    _project_extension_characterization,
    _replace_na_in_zero_output_columns,
    _requested_extensions_from_matrix as _requested_extensions_from_matrix_core,
    _retain_extension_instances as _retain_core,
)
from pyaesa.process.mrios.utils.parsers.exio_parser import (
    _build_characterization_jobs,
    _build_characterization_options,
    _characterize_exiobase_io,
    _ensure_validation_success,
    _find_missing_characterization_extensions,
    _load_characterization_matrix,
    _parse_exio_year,
    _requested_extensions_from_matrix,
    _retain_extension_instances,
    _safe_filename_token,
    _write_validation_mismatch_log,
)
from pyaesa.process.mrios.utils.parsers.oecd_parser import _parse_oecd_year
from pyaesa.process.mrios.utils.pipeline.lcia_tracking import (
    build_lcia_status_payload,
    build_year_entry_payload,
    extract_lcia_units_from_char_matrix,
    extract_lcia_units_from_jobs,
    merge_lcia_method_lists,
    normalize_lcia_method_list,
    resolve_lcia_units_for_methods,
    year_entry_lcia_methods,
    year_entry_unavailable_lcia_methods,
)
from tests.package.helpers.data_processing_dummy import (
    DummyExtension,
    build_dummy_iosystem,
    build_fd_columns,
    build_product_index,
    write_characterization_matrix,
)


def test_parser_wrappers_and_characterization_job_builders(project_repo: Path) -> None:
    full_dir = project_repo / "raw"
    full_dir.mkdir(parents=True, exist_ok=True)
    oecd_csv = full_dir / "ICIO2025_2019.csv"
    oecd_csv.write_text("csv", encoding="utf-8")
    exio_zip = full_dir / "IOT_2019_ixi.zip"
    exio_zip.write_bytes(b"zip")

    oecd_obj = object()
    exio_obj = object()
    parsed_oecd = cast(Any, _parse_oecd_year(full_dir, 2019, parser=lambda path: (path, oecd_obj)))
    assert parsed_oecd[1] is oecd_obj
    assert (
        cast(
            Any,
            _parse_exio_year(
                full_dir,
                2019,
                system="ixi",
                parser=lambda path: (path, exio_obj),
            ),
        )[1]
        is exio_obj
    )
    with pytest.raises(FileNotFoundError):
        _parse_oecd_year(full_dir, 2020, parser=lambda path: path)
    with pytest.raises(FileNotFoundError):
        _parse_exio_year(
            full_dir,
            2020,
            system="ixi",
            parser=lambda path: path,
        )

    assert (
        _build_characterization_options(
            source_key="exiobase_396_ixi",
            requested_lcia_method=" ",
        )
        is None
    )

    matrix_path = write_characterization_matrix(
        project_repo,
        source_key="exiobase_396_ixi",
        method_name="pb_lcia",
    )
    matrix = _load_characterization_matrix(matrix_path)
    assert matrix["extension"].tolist() == ["satellite_accounts"]
    assert _requested_extensions_from_matrix(matrix) == ["satellite_accounts"]
    assert _requested_extensions_from_matrix_core(matrix) == ["satellite_accounts"]

    bad_matrix = matrix_path.parent / "bad.csv"
    pd.DataFrame({"impact": ["x"]}).to_csv(bad_matrix, index=False)
    with pytest.raises(ValueError):
        _load_characterization_matrix(bad_matrix)
    missing_required_matrix = matrix_path.parent / "missing_required.csv"
    pd.DataFrame(
        {
            "extension": ["satellite_accounts"],
            "stressor": ["co2"],
            "impact": ["x"],
        }
    ).to_csv(missing_required_matrix, index=False)
    with pytest.raises(ValueError):
        _load_characterization_matrix(missing_required_matrix)
    empty_required_matrix = matrix_path.parent / "empty_required.csv"
    pd.DataFrame(
        {
            "extension": [" "],
            "stressor": ["co2"],
            "impact": ["x"],
            "factor": [1.0],
            "impact_unit": ["kg"],
        }
    ).to_csv(empty_required_matrix, index=False)
    with pytest.raises(ValueError):
        _load_characterization_matrix(empty_required_matrix)
    with pytest.raises(FileNotFoundError):
        _load_characterization_matrix(matrix_path.parent / "missing.csv")

    jobs = _build_characterization_jobs(
        source_key="exiobase_396_ixi",
        lcia_methods=["pb_lcia", "pb_lcia", " "],
    )
    assert list(jobs) == ["pb_lcia"]
    assert jobs["pb_lcia"].requested_extensions == ["satellite_accounts"]
    assert (
        _build_characterization_jobs(
            source_key="exiobase_396_ixi",
            lcia_methods=None,
        )
        == {}
    )


def test_characterization_validation_logging_and_wrapper_contracts(project_repo: Path) -> None:
    validation = pd.DataFrame(
        {
            "stressor": ["co2"],
            "extension": ["satellite_accounts"],
            "reason": ["unit mismatch"],
            "error_extension": [False],
            "error_stressor": [False],
            "error_unit_stressor": [True],
        }
    )
    log_path = _write_validation_mismatch_log(
        validation_df=validation,
        mask=cast(pd.Series, validation["error_unit_stressor"]),
        source_key="exiobase_396_ixi",
        year=2019,
        lcia_method="pb lcia",
    )
    assert log_path.exists()
    assert _safe_filename_token("pb lcia/test") == "pb_lcia_test"
    assert _safe_filename_token(" ! ") == "unknown"

    _ensure_validation_success(pd.DataFrame({"error_extension": [False], "stressor": ["co2"]}))
    with pytest.raises(RuntimeError):
        _ensure_validation_success(
            validation,
            source_key="exiobase_396_ixi",
            year=2019,
            lcia_method="pb_lcia",
        )
    with pytest.raises(RuntimeError):
        _ensure_validation_success(
            validation,
            source_key="exiobase_396_ixi",
            year=2019,
        )

    iosys = build_dummy_iosystem()
    char_matrix = pd.DataFrame(
        {
            "extension": ["satellite_accounts"],
            "stressor": ["co2"],
            "factor": [2.0],
            "impact": ["impact_1"],
            "impact_parent": ["impact_parent_1"],
            "impact_unit": ["kg CO2-eq"],
        }
    )
    with pytest.raises(ValueError):
        _characterize_exiobase_io(
            iosys,
            char_matrix=char_matrix,
            new_extension_name="not valid",
            retain_instances=["factor_inputs"],
        )

    summary = _characterize_exiobase_io(
        iosys,
        char_matrix=char_matrix,
        new_extension_name="pb_lcia",
        retain_instances=["factor_inputs"],
        prune=False,
    )
    assert summary.keep_instances == ["factor_inputs", "pb_lcia"]
    assert summary.requested_extensions == ["satellite_accounts"]
    assert _find_missing_characterization_extensions(iosys, ["missing"]) == ["missing"]
    assert _retain_extension_instances(iosys, ["factor_inputs", "pb_lcia"]) == [
        "satellite_accounts"
    ]


def test_exio_characterization_core_contracts_cover_success_and_failure_paths() -> None:
    iosys = build_dummy_iosystem()
    satellite = iosys.satellite_accounts
    assert satellite.F is not None
    char_matrix = pd.DataFrame(
        {
            "extension": ["satellite_accounts", "missing_ext", "satellite_accounts"],
            "stressor": ["co2", "co2", "missing_stressor"],
            "stressor_unit": ["kg", "kg", "t"],
            "factor": [2.0, 1.0, 3.0],
            "impact": ["impact_1", "impact_2", "impact_3"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq", "kg CO2-eq"],
        }
    )
    assert _normalize_label(" Satellite_Accounts ") == "satellite accounts"
    resolved, missing = _collect_extensions(iosys, ["satellite accounts", "missing"])
    assert [ext.name for ext in resolved] == ["satellite_accounts"]
    assert missing == ["missing"]

    validation = _build_characterization_validation(
        char_matrix=char_matrix,
        extensions=[satellite],
    )
    assert validation["error_extension"].tolist() == [False, True, False]
    assert validation["error_stressor"].tolist() == [False, False, True]
    assert validation["reason"].tolist() == [
        "",
        "missing extension",
        "missing stressor",
    ]

    _build_characterization_validation(
        char_matrix=pd.DataFrame(
            {
                "extension": ["satellite_accounts"],
                "stressor": ["co2"],
                "stressor_unit": ["kg"],
            }
        ),
        extensions=[
            satellite,
            DummyExtension(name="unused_extension", F=satellite.F.copy()),
        ],
    )

    validation_series_unit = _build_characterization_validation(
        char_matrix=pd.DataFrame(
            {
                "extension": ["series_unit_ext"],
                "stressor": ["co2"],
                "stressor_unit": ["kg"],
            }
        ),
        extensions=[
            DummyExtension(
                name="series_unit_ext",
                F=satellite.F.copy(),
                unit=pd.Series(["kg"], index=pd.Index(["co2"], name="stressor")),
            )
        ],
    )
    assert validation_series_unit["error_unit_stressor"].tolist() == [False]

    validation_scalar_unit = _build_characterization_validation(
        char_matrix=pd.DataFrame(
            {
                "extension": ["scalar_unit_ext"],
                "stressor": ["co2"],
                "stressor_unit": ["kg"],
            }
        ),
        extensions=[DummyExtension(name="scalar_unit_ext", F=satellite.F.copy(), unit="kg")],
    )
    assert validation_scalar_unit["error_unit_stressor"].tolist() == [False]

    missing_f_extension = DummyExtension(name="no_f", F=None, unit=pd.Series(["kg"], index=["co2"]))
    validation_missing_f = _build_characterization_validation(
        char_matrix=pd.DataFrame(
            {
                "extension": ["no_f"],
                "stressor": ["co2"],
                "stressor_unit": ["kg"],
            }
        ),
        extensions=[missing_f_extension],
    )
    assert validation_missing_f["error_stressor"].tolist() == [True]

    with pytest.raises(ValueError):
        _build_impact_units(
            pd.DataFrame(
                {
                    "impact": ["x", "x"],
                    "impact_unit": ["kg", "t"],
                }
            )
        )
    assert (
        _build_impact_units(
            pd.DataFrame(
                {
                    "impact": ["impact_1"],
                    "impact_unit": ["kg CO2-eq"],
                }
            )
        ).loc["impact_1", "unit"]
        == "kg CO2-eq"
    )

    f_char, fy_char = _project_extension_characterization(
        extension=satellite,
        factors=cast(
            pd.DataFrame,
            char_matrix[char_matrix["extension"] == "satellite_accounts"],
        ),
    )
    assert isinstance(f_char, pd.DataFrame)
    assert fy_char is not None
    f_char_only, fy_none = _project_extension_characterization(
        extension=DummyExtension(name="satellite_accounts", F=satellite.F, F_Y=None),
        factors=cast(
            pd.DataFrame,
            char_matrix[char_matrix["extension"] == "satellite_accounts"],
        ),
    )
    assert isinstance(f_char_only, pd.DataFrame)
    assert fy_none is None

    direct = _direct_characterize_extensions(
        extensions=[satellite],
        char_matrix=cast(
            pd.DataFrame,
            char_matrix[char_matrix["extension"] == "satellite_accounts"].iloc[:1],
        ),
        new_extension_name="pb_lcia",
    )
    assert direct.name == "pb_lcia"
    direct_no_fy = _direct_characterize_extensions(
        extensions=[DummyExtension(name="satellite_accounts", F=satellite.F.copy(), F_Y=None)],
        char_matrix=cast(
            pd.DataFrame,
            char_matrix[char_matrix["extension"] == "satellite_accounts"].iloc[:1],
        ),
        new_extension_name="pb_lcia_no_fy",
    )
    assert direct_no_fy.F_Y is None

    captured: list[pd.DataFrame] = []
    keep_instances, requested_extensions = _characterize_exiobase_io_core(
        iosys,
        char_matrix=cast(
            pd.DataFrame,
            char_matrix[char_matrix["extension"] == "satellite_accounts"].iloc[:1],
        ),
        new_extension_name="pb_lcia",
        retain_instances=["factor_inputs"],
        prune=True,
        validate=lambda frame: captured.append(frame),
    )
    assert keep_instances == ["factor_inputs", "pb_lcia"]
    assert requested_extensions == ["satellite_accounts"]
    assert len(captured) == 1
    assert _find_missing_core(iosys, []) == []
    assert _retain_core(iosys, ["factor_inputs", "pb_lcia"]) == []


def test_replace_na_in_zero_output_columns_only_changes_zero_output_targets() -> None:
    frame = pd.DataFrame(
        {
            ("R1", "S1"): [np.nan, 2.0],
            ("R1", "S2"): [np.nan, 4.0],
        },
        index=["stress_1", "stress_2"],
    )
    x_series = pd.Series(
        [0.0, 5.0],
        index=pd.Index([("R1", "S1"), ("R1", "S2")]),
    )

    sanitized = _replace_na_in_zero_output_columns(frame, x_series)

    assert sanitized.loc["stress_1", ("R1", "S1")] == 0.0
    assert pd.isna(sanitized.loc["stress_1", ("R1", "S2")])
    assert sanitized.loc["stress_2", ("R1", "S1")] == 2.0


def test_calc_characterized_extensions_minimal_covers_edge_cases() -> None:
    iosys = build_dummy_iosystem(include_satellite_accounts=False)
    _calc_characterized_extensions_minimal(iosys, [], keep_direct_intensities=False)

    products = build_product_index()
    fd_columns = build_fd_columns()
    lcia_ext = DummyExtension(
        name="pb_lcia",
        F=pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=["impact_1"], columns=products),
        F_Y=pd.DataFrame([[0.1, 0.2]], index=["impact_1"], columns=fd_columns),
    )
    lcia_iosys = type(
        "LciaIO",
        (),
        {
            "x": pd.DataFrame({"indout": [10.0, 20.0, 30.0, 40.0]}, index=products),
            "L": pd.DataFrame(
                pd.DataFrame([[1.0, 0.0, 0.0, 0.0]] * 4, index=products, columns=products)
            ),
            "pb_lcia": lcia_ext,
        },
    )()
    _calc_characterized_extensions_minimal(
        cast(Any, lcia_iosys),
        ["pb_lcia"],
        keep_direct_intensities=True,
    )
    assert cast(Any, lcia_iosys).pb_lcia.D_pba_reg is not None
    assert cast(Any, lcia_iosys).pb_lcia.S is not None

    no_fy_iosys = type(
        "NoFinalDemandExtensionIO",
        (),
        {
            "x": pd.DataFrame({"indout": [10.0, 20.0, 30.0, 40.0]}, index=products),
            "L": cast(Any, lcia_iosys).L,
            "pb_lcia": DummyExtension(
                name="pb_lcia",
                F=pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=["impact_1"], columns=products),
                F_Y=None,
            ),
        },
    )()
    _calc_characterized_extensions_minimal(
        cast(Any, no_fy_iosys),
        ["pb_lcia"],
        keep_direct_intensities=False,
    )
    assert cast(Any, no_fy_iosys).pb_lcia.D_pba_reg is not None
    assert cast(Any, no_fy_iosys).pb_lcia.S is None
    assert cast(Any, no_fy_iosys).pb_lcia.F is None


def test_lcia_tracking_contracts_cover_status_units_and_year_entry_payload(
    project_repo: Path,
) -> None:
    write_characterization_matrix(
        project_repo,
        source_key="exiobase_396_ixi",
        method_name="pb_lcia",
    )
    assert normalize_lcia_method_list(None) == []
    assert normalize_lcia_method_list(["pb_lcia", " ", "gwp100_lcia"]) == [
        "pb_lcia",
        "gwp100_lcia",
    ]
    assert merge_lcia_method_lists(["pb_lcia"], ["pb_lcia", "gwp100_lcia"]) == [
        "pb_lcia",
        "gwp100_lcia",
    ]

    assert year_entry_lcia_methods(None) == []
    assert year_entry_lcia_methods({"enacting_metrics": {}}) == []
    assert year_entry_lcia_methods({"enacting_metrics": {"lcia_methods": ["pb_lcia"]}}) == [
        "pb_lcia"
    ]
    assert year_entry_unavailable_lcia_methods(None) == set()
    assert year_entry_unavailable_lcia_methods({}) == set()
    assert year_entry_unavailable_lcia_methods(
        {
            "lcia_status": {
                "pb_lcia": {"available": False},
                " ": {"available": False},
            }
        }
    ) == {"pb_lcia"}

    with pytest.raises(ValueError):
        extract_lcia_units_from_char_matrix(
            lcia_method="pb_lcia",
            char_matrix=pd.DataFrame({"impact_parent": ["x"]}),
        )
    with pytest.raises(ValueError):
        extract_lcia_units_from_char_matrix(
            lcia_method="pb_lcia",
            char_matrix=pd.DataFrame(
                {
                    "extension": ["satellite_accounts"],
                    "impact_unit": ["kg CO2-eq"],
                }
            ),
        )
    with pytest.raises(ValueError):
        extract_lcia_units_from_char_matrix(
            lcia_method="pb_lcia",
            char_matrix=pd.DataFrame(
                {
                    "extension": ["satellite_accounts", "satellite_accounts"],
                    "impact_parent": [" ", "nan"],
                    "impact_unit": [" ", "kg"],
                }
            ),
        )
    with pytest.raises(ValueError):
        extract_lcia_units_from_char_matrix(
            lcia_method="pb_lcia",
            char_matrix=pd.DataFrame(
                {
                    "extension": ["satellite_accounts", "satellite_accounts"],
                    "impact_parent": ["climate_parent", "climate_parent"],
                    "impact_unit": ["kg CO2-eq", "t CO2-eq"],
                }
            ),
        )
    assert extract_lcia_units_from_char_matrix(
        lcia_method="pb_lcia",
        char_matrix=pd.DataFrame(
            {
                "extension": ["satellite_accounts"],
                "impact_parent": ["climate_parent"],
                "impact_unit": ["kg CO2-eq"],
            }
        ),
    ) == {"climate_parent": "kg CO2-eq"}

    jobs = _build_characterization_jobs(
        source_key="exiobase_396_ixi",
        lcia_methods=["pb_lcia"],
    )
    assert extract_lcia_units_from_jobs(jobs) == {"pb_lcia": {"climate_parent": "kg CO2-eq"}}
    options = _build_characterization_options(
        source_key="exiobase_396_ixi",
        requested_lcia_method="pb_lcia",
    )
    assert options is not None
    assert extract_lcia_units_from_jobs({"pb_lcia": options}) == {
        "pb_lcia": {"climate_parent": "kg CO2-eq"}
    }
    assert resolve_lcia_units_for_methods(
        lcia_method_names=["pb_lcia"],
        units_by_method={"pb_lcia": {"climate_parent": "kg CO2-eq"}},
    ) == {"pb_lcia": {"climate_parent": "kg CO2-eq"}}

    status = build_lcia_status_payload(
        lcia_method_names=["pb_lcia", "gwp100_lcia", "unused"],
        applied_methods=["pb_lcia"],
        missing_by_method={"gwp100_lcia": ["satellite_accounts"]},
    )
    assert status == {
        "pb_lcia": {"available": True},
        "gwp100_lcia": {"available": False, "missing": ["satellite_accounts"]},
        "unused": {"available": False, "missing": []},
    }
    year_entry = build_year_entry_payload(
        saved_dir_name="IOT_2019_ixi_calc",
        core_matrices=["A", "L"],
        extension_payload={"pb_lcia": {"path": "pb_lcia", "available": True}},
        updated_iso="2026-03-12T00:00:00+00:00",
        uncasext_only=False,
        preclip_core_matrices=["A", "L"],
        preclip_extension_payload={"pb_lcia": ["M", "S", "unit"]},
        pymrio_calc_all=True,
        enacting_metric_units={"mrio_default_monetary": "M EUR"},
        applied_methods=["pb_lcia"],
        is_exio=True,
        requires_characterization=True,
        year_char_jobs=jobs,
        missing_by_method={"gwp100_lcia": ["satellite_accounts"]},
        runtime_env={"python": "3.12"},
        raw_correction_payload={"row_count": 1, "log_path": "log.csv"},
    )
    assert year_entry["saved_dir"] == "IOT_2019_ixi_calc"
    assert year_entry["preclip"]["pymrio_calc_all"] is True
    assert year_entry["enacting_metrics"]["lcia_methods"] == ["pb_lcia"]
    assert year_entry["lcia_status"] == {"pb_lcia": {"available": True}}
    assert year_entry["raw_corrected_values"] == {"row_count": 1, "log_path": "log.csv"}
