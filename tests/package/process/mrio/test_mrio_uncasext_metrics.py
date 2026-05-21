import json
from pathlib import Path
from typing import Any, cast
import pickle

import pandas as pd
import pytest

from pyaesa.process.mrios.utils.parsers.exio_characterization import (
    _calc_characterized_extensions_minimal,
)
from pyaesa.process.mrios.utils.uncasext_metrics.common import (
    _all_exist,
    _build_prepared_uncasext_inputs,
    _clip_nonnegative,
    _get_prepared_uncasext_inputs,
    _normalize_x_series,
    _require_dataframe,
    _resolve_single_mrio_unit,
    _set_column_names,
    _set_index_names,
    _write_pickle,
)
from pyaesa.process.mrios.utils.io.paths import (
    _get_mrio_calc_log_path,
    _get_mrio_clipping_log_columns_explanation_path,
    _get_mrio_clipping_log_path,
)
from pyaesa.process.mrios.utils.uncasext_metrics.enacting_metric import (
    _build_enacting_metric_units_payload,
    _precompute_enacting_metrics_uncasext,
    _require_lcia_attr,
    _require_lcia_attr_df,
    _resolve_single_mrio_unit as _resolve_enacting_metric_unit,
    _write_enacting_metric_units_json,
)
from pyaesa.process.mrios.utils.uncasext_metrics.enacting_metric_clip_log import (
    _expand_labels,
    _resolve_clip_log_context,
    clipping_log_columns_explanation_text,
    write_distribution_normalization_log,
    write_clipping_log,
)
from pyaesa.process.mrios.utils.uncasext_metrics.utility_propagation_metrics import (
    _precompute_utility_propag_uncasext,
    _utility_log_name,
    _write_diagnostic_log,
    _write_error_log,
)
from tests.package.helpers.data_processing_dummy import DummyExtension, build_dummy_iosystem


def test_uncasext_common_contracts_cover_shape_validation_and_serialization(tmp_path: Path) -> None:
    iosys_for_products = build_dummy_iosystem()
    assert iosys_for_products.Z is not None

    frame = pd.DataFrame([[1.0]], index=pd.Index(["x"], name="old"), columns=["col"])
    series = pd.Series([1.0], index=pd.Index(["x"], name="old"))
    assert _require_dataframe(frame, label="frame") is frame
    with pytest.raises(TypeError):
        _require_dataframe(series, label="frame")

    renamed_series = _set_index_names(series, ["new"])
    renamed_frame = _set_index_names(frame, ["new"])
    assert renamed_series.index.name == "new"
    assert renamed_frame.index.name == "new"
    with pytest.raises(ValueError):
        _set_index_names(series, ["a", "b"])
    multi_series = pd.Series(
        [1.0], index=pd.MultiIndex.from_tuples([("R1", "S1")], names=["r", "s"])
    )
    with pytest.raises(ValueError):
        _set_index_names(multi_series, ["r"])

    renamed_cols = _set_column_names(frame, ["new_col"])
    assert renamed_cols.columns.name == "new_col"
    with pytest.raises(ValueError):
        _set_column_names(frame, ["a", "b"])
    multi_col_frame = pd.DataFrame(
        [[1.0]],
        columns=pd.MultiIndex.from_tuples([("r1", "c1")], names=["r", "c"]),
    )
    with pytest.raises(ValueError):
        _set_column_names(multi_col_frame, ["only_one"])

    assert _normalize_x_series(pd.Series([1.0], index=["x"])).tolist() == [1.0]
    assert _normalize_x_series(pd.DataFrame({"indout": [1.0]}, index=["x"])).tolist() == [1.0]
    with pytest.raises(ValueError):
        _normalize_x_series(pd.DataFrame([[1.0, 2.0]], columns=["a", "b"]))
    with pytest.raises(TypeError):
        _normalize_x_series(1.0)

    assert _resolve_single_mrio_unit(type("NoUnit", (), {"unit": None})()) is None
    assert _resolve_single_mrio_unit(type("ScalarUnit", (), {"unit": "M EUR"})()) == "M EUR"
    assert (
        _resolve_single_mrio_unit(
            type("SeriesUnit", (), {"unit": pd.Series(["M EUR"], index=["x"])})()
        )
        == "M EUR"
    )
    assert (
        _resolve_single_mrio_unit(
            type(
                "MixedUnit",
                (),
                {"unit": pd.DataFrame({"unit": ["M EUR", "USD"]}, index=["x", "y"])},
            )()
        )
        is None
    )

    pickle_path = tmp_path / "data.pickle"
    _write_pickle(pickle_path, {"x": 1})
    with pickle_path.open("rb") as handle:
        assert pickle.load(handle) == {"x": 1}
    assert _all_exist([pickle_path]) is True
    assert _all_exist([pickle_path, tmp_path / "missing.pickle"]) is False

    clipped_series = _clip_nonnegative(pd.Series([-1.0, 2.0], index=["a", "b"]))
    clipped_frame = _clip_nonnegative(pd.DataFrame([[-1.0, 2.0]], columns=["a", "b"]))
    assert clipped_series.tolist() == [0.0, 2.0]
    assert clipped_frame.iloc[0].tolist() == [0.0, 2.0]


def test_prepared_uncasext_inputs_and_clipping_logs_cover_success_and_failures(
    project_repo: Path,
) -> None:
    iosys = build_dummy_iosystem(negative_y=True, negative_factor_inputs=True)
    prepared = _build_prepared_uncasext_inputs(iosys)
    assert prepared.x_vec.index.names == ["r_p", "s_p"]
    assert prepared.y_fd_raw.columns.name == "r_f"
    assert prepared.z_reg.columns.name == "r_c"
    assert prepared.gva_by_prod.index.names == ["r_p", "s_p"]
    assert prepared.clipping_unit == "M EUR"

    series_factor_iosys = build_dummy_iosystem()
    assert series_factor_iosys.factor_inputs.F is not None
    series_factor_iosys.factor_inputs.F = series_factor_iosys.factor_inputs.F.iloc[0]
    prepared_series = _build_prepared_uncasext_inputs(series_factor_iosys)
    assert prepared_series.gva_by_prod.index.names == ["r_p", "s_p"]

    saved_dir = (
        project_repo
        / "data_processed"
        / "mrio"
        / "oecd_v2025"
        / "original_classification"
        / "ICIO2025_2019_calc"
    )
    saved_dir.mkdir(parents=True, exist_ok=True)
    prepared_once = _get_prepared_uncasext_inputs(
        iosys,
        source_key="oecd_v2025",
        matrix_version=None,
        saved_dir=saved_dir,
    )
    prepared_twice = _get_prepared_uncasext_inputs(
        iosys,
        source_key="oecd_v2025",
        matrix_version=None,
        saved_dir=saved_dir,
    )
    assert prepared_once is prepared_twice
    clip_log = _get_mrio_clipping_log_path(
        "oecd_v2025",
        matrix_version=None,
    )
    clip_rows = pd.read_csv(clip_log)
    assert set(clip_rows["matrix"]) == {"f_factor_inputs_column", "y_fd"}

    with pytest.raises(TypeError):
        _build_prepared_uncasext_inputs(
            type(
                "BadZ",
                (),
                {"Z": None, "Y": iosys.Y, "x": iosys.x, "factor_inputs": iosys.factor_inputs},
            )()
        )
    with pytest.raises(ValueError):
        _build_prepared_uncasext_inputs(
            type(
                "NoFactorF",
                (),
                {"Z": iosys.Z, "Y": iosys.Y, "x": iosys.x, "factor_inputs": object()},
            )()
        )
    with pytest.raises(TypeError):
        _build_prepared_uncasext_inputs(
            type(
                "BadFactorF",
                (),
                {
                    "Z": iosys.Z,
                    "Y": iosys.Y,
                    "x": iosys.x,
                    "factor_inputs": type("Factor", (), {"F": 1.0})(),
                },
            )()
        )


def test_enacting_metric_clip_log_contracts_cover_context_and_append_paths(
    project_repo: Path,
) -> None:
    saved_dir = project_repo / "saved_2019_demo_2020"
    saved_dir.mkdir(parents=True, exist_ok=True)
    log_path, version_label, log_year = _resolve_clip_log_context(
        source_key="oecd_v2025",
        matrix_version="demo version",
        saved_dir=saved_dir,
    )
    assert log_path.name == "oecd_v2025_demo_version_clipping_log.csv"
    assert version_label == "demo version"
    assert log_year == 2020

    data = pd.DataFrame(
        [[-1.0, 2.0], [3.0, -4.0]],
        index=pd.MultiIndex.from_tuples([("R1", "S1"), ("R2", "S2")], names=["r_p", "s_p"]),
        columns=pd.Index(["R1", "R2"], name="r_f"),
    )
    write_clipping_log(
        before=data,
        matrix_name="y_fd",
        unit="M EUR",
        source_key="oecd_v2025",
        matrix_version=None,
        saved_dir=saved_dir,
    )
    write_clipping_log(
        before=pd.Series(
            [-2.0], index=pd.MultiIndex.from_tuples([("R1", "S1")], names=["r_p", "s_p"])
        ),
        matrix_name="factor_inputs",
        unit=None,
        source_key="oecd_v2025",
        matrix_version=None,
        saved_dir=saved_dir,
    )
    rows = pd.read_csv(
        _get_mrio_clipping_log_path(
            "oecd_v2025",
            matrix_version=None,
        )
    )
    expected_log_columns = [
        "source",
        "matrix_version",
        "matrix",
        "event_type",
        "event_detail",
        "year",
        "unit",
        "r_p",
        "s_p",
        "r_f",
        "r_u",
        "distribution_axis",
        "original_value",
        "clipped_value",
        "original_sum",
        "adjusted_sum",
        "expected_sum",
        "processed_output_abs",
        "processed_input_side_value_added_abs",
        "processed_intermediate_input_total_abs",
        "processed_output_side_value_added_abs",
    ]
    assert rows.columns.tolist() == expected_log_columns
    assert set(rows["matrix"]) == {"factor_inputs", "y_fd"}
    explanation_path = _get_mrio_clipping_log_columns_explanation_path(
        "oecd_v2025",
        matrix_version=None,
    )
    assert explanation_path.exists()
    assert explanation_path.read_text(encoding="utf-8") == clipping_log_columns_explanation_text()
    assert all(len(line) <= 100 for line in clipping_log_columns_explanation_text().splitlines())

    write_clipping_log(
        before=pd.Series([1.0], index=pd.Index(["x"])),
        matrix_name="noop",
        unit="M EUR",
        source_key="oecd_v2025",
        matrix_version=None,
        saved_dir=saved_dir,
    )
    rows_after = pd.read_csv(
        _get_mrio_clipping_log_path(
            "oecd_v2025",
            matrix_version=None,
        )
    )
    assert rows_after.shape == rows.shape

    explanation_path.write_text("stale", encoding="utf-8")
    write_clipping_log(
        before=pd.Series([-3.0], index=pd.Index(["y"])),
        matrix_name="stale_rewrite",
        unit="M EUR",
        source_key="oecd_v2025",
        matrix_version=None,
        saved_dir=saved_dir,
    )
    assert explanation_path.read_text(encoding="utf-8") == clipping_log_columns_explanation_text()
    rows_after_stale = pd.read_csv(
        _get_mrio_clipping_log_path(
            "oecd_v2025",
            matrix_version=None,
        )
    )

    empty_labels = _expand_labels(pd.Index([], dtype=object))
    assert empty_labels.empty
    assert empty_labels.columns.tolist() == ["r_p"]
    one_level = _expand_labels(pd.Index(["R1"], name="r_p"))
    assert one_level.columns.tolist() == ["r_p"]
    assert one_level.iloc[0, 0] == "R1"
    three_level = _expand_labels(
        pd.MultiIndex.from_tuples([("R1", "S1", "FD")], names=["r_p", "s_p", "extra"])
    )
    assert three_level.columns.tolist() == ["label_0", "label_1", "label_2"]

    same_before = pd.DataFrame(
        [[0.5, 0.5]],
        index=pd.Index(["R1"], name="r_u"),
        columns=pd.MultiIndex.from_tuples([("R1", "S1"), ("R2", "S2")], names=["r_p", "s_p"]),
    )
    write_distribution_normalization_log(
        before=same_before,
        after=same_before.copy(),
        matrix_name="omega_reg",
        distribution_axis="r_u",
        unit="M EUR",
        source_key="oecd_v2025",
        matrix_version=None,
        saved_dir=saved_dir,
        expected_sum=1.0,
    )
    rows_same = pd.read_csv(
        _get_mrio_clipping_log_path(
            "oecd_v2025",
            matrix_version=None,
        )
    )
    assert rows_same.shape == rows_after_stale.shape

    changed_before = pd.DataFrame(
        [[0.2], [0.2]],
        index=pd.Index(["R1", "R2"], name="r_u"),
        columns=pd.MultiIndex.from_tuples([("R1", "S1")], names=["r_p", "s_p"]),
    )
    changed_after = pd.DataFrame(
        [[0.5], [0.5]],
        index=changed_before.index,
        columns=changed_before.columns,
    )
    write_distribution_normalization_log(
        before=changed_before,
        after=changed_after,
        matrix_name="omega_reg",
        distribution_axis="r_u",
        unit="M EUR",
        source_key="oecd_v2025",
        matrix_version=None,
        saved_dir=saved_dir,
        expected_sum=1.0,
    )
    rows_norm = pd.read_csv(
        _get_mrio_clipping_log_path(
            "oecd_v2025",
            matrix_version=None,
        )
    )
    norm_rows = rows_norm.loc[rows_norm["event_type"] == "normalize_distribution"]
    assert norm_rows.shape[0] == 1
    assert norm_rows["matrix"].tolist() == ["omega_reg"]
    assert norm_rows["distribution_axis"].tolist() == ["r_u"]
    assert norm_rows["original_sum"].tolist() == [0.4]
    assert norm_rows["adjusted_sum"].tolist() == [1.0]
    assert norm_rows["expected_sum"].tolist() == [1.0]
    assert norm_rows["processed_output_abs"].isna().all()
    assert norm_rows["processed_input_side_value_added_abs"].isna().all()
    assert norm_rows["processed_intermediate_input_total_abs"].isna().all()
    assert norm_rows["processed_output_side_value_added_abs"].isna().all()
    assert rows_norm.columns.tolist() == expected_log_columns
    event_detail = norm_rows["event_detail"].iloc[0]
    assert "alpha^T L" in event_detail
    assert "1^T(I-A)" in event_detail
    assert "processed_* absolute columns" in event_detail


def test_utility_propagation_contracts_cover_logging_short_circuit_and_failure_paths(
    project_repo: Path,
) -> None:
    saved_dir = (
        project_repo
        / "data_processed"
        / "mrio"
        / "oecd_v2025"
        / "original_classification"
        / "ICIO2025_2019_calc"
    )
    saved_dir.mkdir(parents=True, exist_ok=True)
    iosys = build_dummy_iosystem()

    assert (
        _utility_log_name(saved_dir)
        == "oecd_v2025__original_classification__ICIO2025_2019_calc_utility_propag_"
        "uncasext_error.log"
    )
    _write_diagnostic_log(saved_dir, "hello")
    _write_error_log(saved_dir, ValueError("boom"))
    utility_log = _get_mrio_calc_log_path(
        _utility_log_name(saved_dir),
        source_key="oecd_v2025",
        matrix_version="original_classification",
    )
    log_text = utility_log.read_text(encoding="utf-8")
    assert "hello" in log_text
    assert "boom" in log_text

    _precompute_utility_propag_uncasext(
        iosys=iosys,
        saved_dir=saved_dir,
        refresh=True,
        source_key="oecd_v2025",
        matrix_version=None,
    )
    outdir = saved_dir / "utility_propag_uncasext"
    assert (outdir / "x_to_rc.pickle").exists()
    assert (outdir / "kappa.pickle").exists()
    assert (outdir / "omega_reg.pickle").exists()

    _precompute_utility_propag_uncasext(
        iosys=iosys,
        saved_dir=saved_dir,
        refresh=False,
        source_key="oecd_v2025",
        matrix_version=None,
    )

    bad_l_iosys = build_dummy_iosystem()
    assert bad_l_iosys.L is not None
    bad_l_iosys.L = pd.DataFrame(
        0.0,
        index=cast(pd.DataFrame, bad_l_iosys.L).index,
        columns=cast(pd.DataFrame, bad_l_iosys.L).columns,
    )
    bad_l_saved_dir = saved_dir.parent / "ICIO2025_2021_calc"
    bad_l_saved_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError):
        _precompute_utility_propag_uncasext(
            iosys=bad_l_iosys,
            saved_dir=bad_l_saved_dir,
            refresh=True,
            source_key="oecd_v2025",
            matrix_version=None,
        )


def test_enacting_metric_contracts_cover_units_and_output_persistence(project_repo: Path) -> None:
    iosys = build_dummy_iosystem()
    assert iosys.satellite_accounts.F is not None
    assert iosys.satellite_accounts.F_Y is not None
    assert iosys.satellite_accounts.unit is not None
    iosys.pb_lcia = DummyExtension(
        name="pb_lcia",
        F=iosys.satellite_accounts.F.copy(),
        F_Y=iosys.satellite_accounts.F_Y.copy(),
        unit=iosys.satellite_accounts.unit.copy(),
    )
    _calc_characterized_extensions_minimal(
        iosys,
        ["pb_lcia"],
        keep_direct_intensities=True,
    )

    with pytest.raises(ValueError):
        _require_lcia_attr(DummyExtension(name="pb_lcia", F=None), "F")
    assert _require_lcia_attr(iosys.pb_lcia, "F").equals(iosys.pb_lcia.F)
    with pytest.raises(TypeError):
        _require_lcia_attr_df(cast(Any, DummyExtension(name="pb_lcia", F=1.0)), "F")
    assert _require_lcia_attr_df(iosys.pb_lcia, "F").equals(iosys.pb_lcia.F)

    assert _resolve_enacting_metric_unit(type("ScalarUnit", (), {"unit": "M EUR"})()) == "M EUR"
    with pytest.raises(ValueError):
        _resolve_enacting_metric_unit(type("NoUnit", (), {"unit": None})())
    with pytest.raises(ValueError):
        _resolve_enacting_metric_unit(
            type("BlankUnit", (), {"unit": pd.Series([" "], index=["x"])})()
        )
    with pytest.raises(ValueError):
        _resolve_enacting_metric_unit(
            type(
                "MixedUnit",
                (),
                {"unit": pd.DataFrame({"unit": ["M EUR", "USD"]}, index=["x", "y"])},
            )()
        )

    units_payload = _build_enacting_metric_units_payload(
        iosys=iosys,
        lcia_method_specs=["pb_lcia"],
        lcia_units_by_method={"pb_lcia": {"climate_parent": "kg CO2-eq", " ": " "}},
    )
    assert units_payload["mrio_default_monetary"] == "M EUR"
    assert units_payload["lcia_by_method"] == {"pb_lcia": {"climate_parent": "kg CO2-eq"}}
    units_path = project_repo / "units.json"
    _write_enacting_metric_units_json(path=units_path, payload=units_payload)
    assert json.loads(units_path.read_text(encoding="utf-8"))["mrio_default_monetary"] == "M EUR"

    non_exio_saved = (
        project_repo
        / "data_processed"
        / "mrio"
        / "oecd_v2025"
        / "original_classification"
        / "ICIO2025_2019_calc"
    )
    non_exio_saved.mkdir(parents=True, exist_ok=True)
    non_exio_units = _precompute_enacting_metrics_uncasext(
        iosys=build_dummy_iosystem(),
        saved_dir=non_exio_saved,
        source_key="oecd_v2025",
        refresh=True,
        lcia_methods=None,
        matrix_version=None,
        lcia_units_by_method=None,
    )
    assert non_exio_units["mrio_default_monetary"] == "M EUR"
    assert (non_exio_saved / "enacting_metrics" / "level_1" / "fd_rf.pickle").exists()
    assert (non_exio_saved / "enacting_metrics" / "units.json").exists()

    non_exio_units_cached = _precompute_enacting_metrics_uncasext(
        iosys=build_dummy_iosystem(),
        saved_dir=non_exio_saved,
        source_key="oecd_v2025",
        refresh=False,
        lcia_methods=None,
        matrix_version=None,
        lcia_units_by_method=None,
    )
    assert non_exio_units_cached["mrio_default_monetary"] == "M EUR"

    exio_saved = (
        project_repo
        / "data_processed"
        / "mrio"
        / "exiobase_396_ixi"
        / "original_classification"
        / "IOT_2019_ixi_calc"
    )
    exio_saved.mkdir(parents=True, exist_ok=True)
    exio_units = _precompute_enacting_metrics_uncasext(
        iosys=iosys,
        saved_dir=exio_saved,
        source_key="exiobase_396_ixi",
        refresh=True,
        lcia_methods=["pb_lcia"],
        matrix_version=None,
        lcia_units_by_method={"pb_lcia": {"climate_parent": "kg CO2-eq"}},
    )
    assert exio_units["lcia_by_method"] == {"pb_lcia": {"climate_parent": "kg CO2-eq"}}
    assert (exio_saved / "enacting_metrics" / "level_1" / "pb_lcia" / "e_pba_reg.pickle").exists()
    assert (
        exio_saved / "enacting_metrics" / "level_2" / "pb_lcia" / "e_cba_td_rc_sp.pickle"
    ).exists()

    cached_exio_units = _precompute_enacting_metrics_uncasext(
        iosys=iosys,
        saved_dir=exio_saved,
        source_key="exiobase_396_ixi",
        refresh=False,
        lcia_methods=["pb_lcia"],
        matrix_version=None,
        lcia_units_by_method={"pb_lcia": {"climate_parent": "kg CO2-eq"}},
    )
    assert cached_exio_units["lcia_by_method"] == {"pb_lcia": {"climate_parent": "kg CO2-eq"}}

    with pytest.raises(ValueError):
        _precompute_enacting_metrics_uncasext(
            iosys=build_dummy_iosystem(),
            saved_dir=exio_saved / "missing_lcia",
            source_key="exiobase_396_ixi",
            refresh=True,
            lcia_methods=["pb_lcia"],
            matrix_version=None,
            lcia_units_by_method={"pb_lcia": {"climate_parent": "kg CO2-eq"}},
        )
