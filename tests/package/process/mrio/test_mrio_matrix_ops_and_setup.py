from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa.process.mrios.utils.pipeline.contracts import ProcessReportMRIO
from pyaesa.process.mrios.utils.aggregation.aggregation import build_aggregation_spec, read_agg_map
from pyaesa.process.mrios.utils.pipeline.matrix_ops import (
    _aggregate_columns,
    _aggregate_iosys_fast,
    _aggregate_rows,
    _build_fd_agg_key,
    _build_product_agg_key,
    _calc_aggregated_full_system_after_fast_aggregation,
    _calc_core_system_minimal,
    _calc_x_from_clipped_fd,
    _clear_extension_derived_accounts,
    _agg_final_demand_by_region,
    _labels_from_product_index,
    _map_values,
    _normalize_mrio_axes,
)
from pyaesa.process.mrios.utils.pipeline.process_setup import (
    _normalize_lcia_methods,
    _resolve_aggregation_inputs,
    _resolve_year_characterization_jobs,
    _update_report_clipping_stats,
)
from pyaesa.process.mrios.utils.pipeline.weighted_aggregation import _normal_final_demand_axis
from tests.package.helpers.data_processing_dummy import (
    DummyExtension,
    DummyIOSystem,
    build_dummy_iosystem,
    build_fd_columns,
    build_product_index,
    write_characterization_matrix,
)


def test_matrix_ops_cover_aggregation_and_minimal_core_paths(project_repo: Path) -> None:
    products = build_product_index()
    unnamed_products = build_product_index(named_levels=False)
    fd_columns = build_fd_columns()

    assert _map_values(pd.Index(["R1", "R2"]), {"R1": "EU"}) == ["EU", "R2"]
    unnamed_iosys = build_dummy_iosystem(named_levels=False)
    _normalize_mrio_axes(unnamed_iosys)
    assert unnamed_iosys.Z is not None
    assert unnamed_iosys.Y is not None
    assert unnamed_iosys.Z.index.names == ["region", "sector"]
    assert unnamed_iosys.Y.columns.names == ["region", "final_demand"]

    product_key = _build_product_agg_key(
        products,
        region_map={"R1": "EU", "R2": "ROW"},
        sector_map={"S1": "Energy", "S2": "Other"},
    )
    assert product_key is not None
    assert list(product_key.names) == ["region", "sector"]
    fallback_key = _build_product_agg_key(
        unnamed_products,
        region_map={"R1": "EU", "R2": "ROW"},
        sector_map={"S1": "Energy", "S2": "Other"},
    )
    assert fallback_key is not None
    assert list(fallback_key.names) == ["region", "sector"]

    fd_key = _build_fd_agg_key(fd_columns, region_map={"R1": "EU", "R2": "ROW"})
    assert isinstance(fd_key, pd.MultiIndex)
    assert list(fd_key.names) == ["region", "final_demand"]
    unnamed_fd_key = _build_fd_agg_key(
        build_fd_columns(named_levels=False),
        region_map={"R1": "EU", "R2": "ROW"},
    )
    assert isinstance(unnamed_fd_key, pd.MultiIndex)
    assert list(unnamed_fd_key.names) == ["region", "final_demand"]
    assert _build_fd_agg_key(pd.Index(["R1", "R2"]), region_map={"R1": "EU"}).tolist() == [
        "EU",
        "R2",
    ]
    weighted_fd_axis = _normal_final_demand_axis(
        pd.MultiIndex.from_tuples(
            [("R1", "FD")],
            names=["region", None],
        )
    )
    assert weighted_fd_axis.names == ["region", "final_demand"]

    row_df = pd.DataFrame([[1, 2], [3, 4]], index=products[:2], columns=["a", "b"])
    aggregated_rows = _aggregate_rows(row_df, cast(pd.MultiIndex, product_key[:2]))
    assert isinstance(aggregated_rows.index, pd.MultiIndex)
    simple_rows = _aggregate_rows(row_df, pd.Index(["same", "same"], name="region"))
    assert list(simple_rows.index) == ["same"]
    aggregated_cols = _aggregate_columns(
        pd.DataFrame([[1, 2], [3, 4]], index=["x", "y"], columns=products[:2]),
        cast(pd.MultiIndex, product_key[:2]),
    )
    assert isinstance(aggregated_cols.columns, pd.MultiIndex)
    simple_cols = _aggregate_columns(
        pd.DataFrame([[1, 2], [3, 4]], index=["x", "y"], columns=["c1", "c2"]),
        pd.Index(["same", "same"], name="region"),
    )
    assert list(simple_cols.columns) == ["same"]
    ext = DummyExtension(name="factor_inputs", F=pd.DataFrame())
    ext.S = pd.DataFrame()
    ext.M = pd.DataFrame()
    ext.D_cba = pd.DataFrame()
    _clear_extension_derived_accounts(ext)
    assert ext.S is None
    assert ext.M is None
    assert ext.D_cba is None

    iosys = build_dummy_iosystem()
    assert iosys.Z is not None
    assert iosys.Y is not None
    assert iosys.unit is not None
    assert iosys.factor_inputs.F is not None
    assert iosys.satellite_accounts.F_Y is not None
    _aggregate_iosys_fast(
        iosys=iosys,
        agg_reg=True,
        region_map={"R1": "EU", "R2": "ROW"},
        sector_map={"S1": "Energy", "S2": "Other"},
    )
    assert iosys.Z.index.names == ["region", "sector"]
    assert iosys.Y.columns.names == ["region", "final_demand"]
    assert iosys.unit.index.names == ["region", "sector"]
    assert iosys.factor_inputs.F.columns.names == ["region", "sector"]
    assert iosys.satellite_accounts.F_Y.columns.names == ["region", "final_demand"]
    assert iosys.x is None
    assert iosys.A is None
    assert iosys.L is None
    assert iosys.G is None

    calc_all_aggregated_iosys = build_dummy_iosystem()
    _aggregate_iosys_fast(
        iosys=calc_all_aggregated_iosys,
        agg_reg=True,
        region_map={"R1": "EU", "R2": "ROW"},
        sector_map={"S1": "Energy", "S2": "Other"},
    )
    _calc_aggregated_full_system_after_fast_aggregation(iosys=calc_all_aggregated_iosys)
    assert calc_all_aggregated_iosys.G is not None
    assert calc_all_aggregated_iosys.factor_inputs.S is not None
    assert cast(pd.DataFrame, calc_all_aggregated_iosys.G).index.tolist() == [
        ("EU", "Energy"),
        ("EU", "Other"),
        ("ROW", "Energy"),
        ("ROW", "Other"),
    ]

    no_ext_iosys = DummyIOSystem(
        Z=iosys.Z.copy(),
        Y=iosys.Y.copy(),
        unit=iosys.unit.copy(),
        extensions={},
    )
    _aggregate_iosys_fast(
        iosys=no_ext_iosys,
        agg_reg=False,
        region_map={"EU": "EU", "ROW": "ROW"},
        sector_map={"Energy": "Energy", "Other": "Other"},
    )

    agg_reg_false_iosys = build_dummy_iosystem()
    _aggregate_iosys_fast(
        iosys=agg_reg_false_iosys,
        agg_reg=False,
        region_map={"R1": "EU", "R2": "ROW"},
        sector_map={"S1": "Energy", "S2": "Other"},
    )

    series_unit_iosys = build_dummy_iosystem()
    assert series_unit_iosys.unit is not None
    series_unit_iosys.unit = series_unit_iosys.unit["unit"]
    series_unit_iosys.satellite_accounts.F = None
    series_unit_iosys.satellite_accounts.F_Y = None
    _aggregate_iosys_fast(
        iosys=series_unit_iosys,
        agg_reg=True,
        region_map={"R1": "EU", "R2": "ROW"},
        sector_map={"S1": "Energy", "S2": "Other"},
    )
    assert series_unit_iosys.unit.index.names == ["region", "sector"]

    no_get_extensions = type(
        "NoGetExtensions",
        (),
        {
            "Z": cast(pd.DataFrame, build_dummy_iosystem().Z).copy(),
            "Y": cast(pd.DataFrame, build_dummy_iosystem().Y).copy(),
            "unit": cast(pd.DataFrame, build_dummy_iosystem().unit).copy(),
        },
    )()
    _aggregate_iosys_fast(
        iosys=cast(Any, no_get_extensions),
        agg_reg=False,
        region_map={"R1": "R1", "R2": "R2"},
        sector_map={"S1": "S1", "S2": "S2"},
    )

    unitless_iosys = build_dummy_iosystem()
    unitless_iosys.unit = None
    _aggregate_iosys_fast(
        iosys=unitless_iosys,
        agg_reg=False,
        region_map={"R1": "R1", "R2": "R2"},
        sector_map={"S1": "S1", "S2": "S2"},
    )
    assert unitless_iosys.unit is None

    aggregated_ysrc = build_dummy_iosystem()
    assert aggregated_ysrc.Y is not None
    aggregated_y = _agg_final_demand_by_region(aggregated_ysrc.Y)
    assert aggregated_y.columns.name == "region"
    aggregated_y_fallback = _agg_final_demand_by_region(
        pd.DataFrame(
            [[1.0, 2.0]],
            index=build_product_index(named_levels=False)[:1],
            columns=build_fd_columns(named_levels=False),
        )
    )
    assert aggregated_y_fallback.shape == (1, 2)

    regions, sectors = _labels_from_product_index(products)
    assert regions == ["R1", "R2"]
    assert sectors == ["S1", "S2"]
    regions_fallback, sectors_fallback = _labels_from_product_index(unnamed_products)
    assert regions_fallback == ["R1", "R2"]
    assert sectors_fallback == ["S1", "S2"]

    iosys_for_core = build_dummy_iosystem(negative_y=True)
    assert iosys_for_core.Z is not None
    assert iosys_for_core.Y is not None
    x_frame = _calc_x_from_clipped_fd(z=iosys_for_core.Z, y=iosys_for_core.Y)
    assert list(x_frame.columns) == ["indout"]
    _calc_core_system_minimal(
        iosys=iosys_for_core,
        include_ghosh=True,
    )
    assert iosys_for_core.A is not None
    assert iosys_for_core.L is not None
    assert iosys_for_core.G is not None

    iosys_without_ghosh = build_dummy_iosystem(negative_y=True)
    _calc_core_system_minimal(
        iosys=iosys_without_ghosh,
        include_ghosh=False,
    )
    assert iosys_without_ghosh.A is None
    assert iosys_without_ghosh.L is not None
    assert iosys_without_ghosh.G is None


def test_weighted_aggregation_uses_pymrio_concordance_semantics(tmp_path: Path) -> None:
    iosys = build_dummy_iosystem()
    assert iosys.Z is not None
    assert iosys.Y is not None
    assert iosys.factor_inputs.F is not None
    assert iosys.satellite_accounts.F_Y is not None
    iosys.Z.loc[:, :] = 0.0
    iosys.Z.loc[("R1", "S1"), ("R1", "S1")] = 100.0
    z_total = float(iosys.Z.to_numpy(dtype=float).sum())
    y_total = float(iosys.Y.to_numpy(dtype=float).sum())
    f_total = float(iosys.factor_inputs.F.to_numpy(dtype=float).sum())
    fy_total = float(iosys.satellite_accounts.F_Y.to_numpy(dtype=float).sum())

    reg_path = tmp_path / "agg_reg_weighted.csv"
    sec_path = tmp_path / "agg_sec_weighted.csv"
    pd.DataFrame(
        {
            "original_classification": ["R1", "R2"],
            "aggregated_mrio": ["EU", "ROW"],
            "weight": [1.0, 1.0],
        }
    ).to_csv(reg_path, index=False)
    pd.DataFrame(
        {
            "original_classification": ["S1", "S1", "S2"],
            "aggregated_mrio": ["Energy", "Other", "Other"],
            "weight": [0.85, 0.15, 1.0],
        }
    ).to_csv(sec_path, index=False)
    reg_spec = build_aggregation_spec(
        ["R1", "R2"],
        read_agg_map(reg_path),
        label_kind="region",
        csv_path=reg_path,
    )
    sec_spec = build_aggregation_spec(
        ["S1", "S2"],
        read_agg_map(sec_path),
        label_kind="sector",
        csv_path=sec_path,
    )

    _aggregate_iosys_fast(
        iosys=iosys,
        agg_reg=True,
        region_spec=reg_spec,
        sector_spec=sec_spec,
    )

    assert iosys.Z is not None
    assert iosys.Y is not None
    assert iosys.factor_inputs.F is not None
    assert iosys.satellite_accounts.F_Y is not None
    assert float(iosys.Z.to_numpy(dtype=float).sum()) == pytest.approx(z_total)
    assert float(iosys.Y.to_numpy(dtype=float).sum()) == pytest.approx(y_total)
    assert float(iosys.factor_inputs.F.to_numpy(dtype=float).sum()) == pytest.approx(f_total)
    assert float(iosys.satellite_accounts.F_Y.to_numpy(dtype=float).sum()) == pytest.approx(
        fy_total
    )
    assert iosys.Z.loc[("EU", "Energy"), ("EU", "Energy")] == pytest.approx(72.25)
    assert iosys.Z.loc[("EU", "Energy"), ("EU", "Other")] == pytest.approx(12.75)
    assert iosys.Z.loc[("EU", "Other"), ("EU", "Energy")] == pytest.approx(12.75)
    assert iosys.Z.loc[("EU", "Other"), ("EU", "Other")] == pytest.approx(2.25)
    assert iosys.satellite_accounts.F_Y.columns.names == ["region", "final_demand"]
    assert iosys.x is None
    assert iosys.A is None
    assert iosys.L is None
    assert iosys.G is None


def test_weighted_aggregation_covers_final_demand_axis_and_optional_extensions(
    tmp_path: Path,
) -> None:
    iosys = build_dummy_iosystem()
    assert iosys.Z is not None
    assert iosys.Y is not None
    assert iosys.unit is not None
    iosys.Y.columns = pd.Index(["R1", "R2"], name="region")
    iosys.unit = iosys.unit["unit"]
    iosys.satellite_accounts.F = None

    reg_path = tmp_path / "agg_reg_weighted.csv"
    sec_path = tmp_path / "agg_sec_weighted.csv"
    pd.DataFrame(
        {
            "original_classification": ["R1", "R1", "R2"],
            "aggregated_mrio": ["EU", "ROW", "ROW"],
            "weight": [0.8, 0.2, 1.0],
        }
    ).to_csv(reg_path, index=False)
    pd.DataFrame(
        {
            "original_classification": ["S1", "S2"],
            "aggregated_mrio": ["Energy", "Other"],
            "weight": [1.0, 1.0],
        }
    ).to_csv(sec_path, index=False)
    reg_spec = build_aggregation_spec(
        ["R1", "R2"],
        read_agg_map(reg_path),
        label_kind="region",
        csv_path=reg_path,
    )
    sec_spec = build_aggregation_spec(
        ["S1", "S2"],
        read_agg_map(sec_path),
        label_kind="sector",
        csv_path=sec_path,
    )

    _aggregate_iosys_fast(
        iosys=iosys,
        agg_reg=True,
        region_spec=reg_spec,
        sector_spec=sec_spec,
    )

    assert isinstance(iosys.unit, pd.Series)
    assert iosys.Y.columns.tolist() == ["EU", "ROW"]
    assert iosys.satellite_accounts.F is None
    assert iosys.satellite_accounts.F_Y is not None
    assert iosys.satellite_accounts.F_Y.columns.tolist() == ["EU", "ROW"]

    detailed_fd_iosys = build_dummy_iosystem()
    assert detailed_fd_iosys.Y is not None
    detailed_fd_iosys.Y.columns = pd.MultiIndex.from_tuples(
        [("R1", "Households", "domestic"), ("R2", "Households", "domestic")],
        names=["region", None, "origin"],
    )
    _aggregate_iosys_fast(
        iosys=detailed_fd_iosys,
        agg_reg=True,
        region_spec=reg_spec,
        sector_spec=sec_spec,
    )
    assert detailed_fd_iosys.Y.columns.names == ["region", "final_demand", "origin"]

    sector_only_iosys = build_dummy_iosystem()
    _aggregate_iosys_fast(
        iosys=sector_only_iosys,
        agg_reg=False,
        region_spec=reg_spec,
        sector_spec=sec_spec,
    )
    assert sector_only_iosys.factor_inputs.F is not None
    assert sector_only_iosys.satellite_accounts.F_Y is not None
    assert sector_only_iosys.satellite_accounts.F_Y.columns.names == [
        "region",
        "final_demand",
    ]

    no_ext_iosys = type(
        "NoWeightedExtensions",
        (),
        {
            "Z": cast(pd.DataFrame, build_dummy_iosystem().Z).copy(),
            "Y": cast(pd.DataFrame, build_dummy_iosystem().Y).copy(),
            "unit": None,
        },
    )()
    _aggregate_iosys_fast(
        iosys=cast(Any, no_ext_iosys),
        agg_reg=False,
        region_spec=reg_spec,
        sector_spec=sec_spec,
    )


def test_process_setup_contracts_cover_aggregation_jobs_and_clipping(project_repo: Path) -> None:
    assert _normalize_lcia_methods(None) is None
    assert _normalize_lcia_methods("pb_lcia") == ["pb_lcia"]
    assert _normalize_lcia_methods([]) is None
    assert _normalize_lcia_methods(["pb_lcia", " ", "gwp100_lcia"]) == [
        "pb_lcia",
        "gwp100_lcia",
    ]

    empty_aggregation = _resolve_aggregation_inputs(
        source="oecd_v2025",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
    )
    assert empty_aggregation[-1] == {
        "agg_reg": False,
        "agg_sec": False,
        "agg_version": None,
        "agg_reg_file": None,
        "agg_sec_file": None,
        "agg_reg_weighted": False,
        "agg_sec_weighted": False,
        "agg_reg_fingerprint": None,
        "agg_sec_fingerprint": None,
    }

    reg_path = (
        project_repo / "data_raw" / "mrio" / "oecd_v2025" / "aggregation" / "agg_reg_demo.csv"
    )
    sec_path = (
        project_repo / "data_raw" / "mrio" / "oecd_v2025" / "aggregation" / "agg_sec_demo.csv"
    )
    with pytest.raises(FileNotFoundError):
        _resolve_aggregation_inputs(
            source="oecd_v2025",
            agg_reg=True,
            agg_sec=False,
            agg_version="demo",
        )
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "original_classification": ["R1", "R2"],
            "aggregated_mrio": ["EU", "ROW"],
        }
    ).to_csv(reg_path, index=False)
    with pytest.raises(FileNotFoundError):
        _resolve_aggregation_inputs(
            source="oecd_v2025",
            agg_reg=True,
            agg_sec=True,
            agg_version="demo",
        )
    pd.DataFrame(
        {
            "original_classification": ["S1", "S2"],
            "aggregated_mrio": ["Energy", "Other"],
        }
    ).to_csv(sec_path, index=False)
    agg_reg_path, agg_sec_path, agg_reg_df, agg_sec_df, payload = _resolve_aggregation_inputs(
        source="oecd_v2025",
        agg_reg=True,
        agg_sec=True,
        agg_version="demo",
    )
    assert agg_reg_df is not None
    assert agg_sec_df is not None
    assert agg_reg_path == reg_path
    assert agg_sec_path == sec_path
    assert list(agg_reg_df["aggregated_mrio"]) == ["EU", "ROW"]
    assert list(agg_sec_df["aggregated_mrio"]) == ["Energy", "Other"]
    assert payload["agg_version"] == "demo"
    assert payload["agg_reg_weighted"] is False
    assert payload["agg_sec_weighted"] is False
    assert isinstance(payload["agg_reg_fingerprint"], str)
    assert isinstance(payload["agg_sec_fingerprint"], str)

    write_characterization_matrix(
        project_repo,
        source_key="exiobase_396_ixi",
        method_name="pb_lcia",
    )
    year_jobs, units = _resolve_year_characterization_jobs(
        source="exiobase_396_ixi",
        year_lcia_methods=["pb_lcia"],
        char_jobs_cache={},
    )
    assert list(year_jobs) == ["pb_lcia"]
    assert units == {"pb_lcia": {"climate_parent": "kg CO2-eq"}}

    cached_jobs, cached_units = _resolve_year_characterization_jobs(
        source="exiobase_396_ixi",
        year_lcia_methods=["pb_lcia"],
        char_jobs_cache=year_jobs.copy(),
    )
    assert list(cached_jobs) == ["pb_lcia"]
    assert cached_units == {"pb_lcia": {"climate_parent": "kg CO2-eq"}}

    report = ProcessReportMRIO(source="oecd_v2025", requested=[2019])
    iosys = build_dummy_iosystem(negative_y=True, negative_factor_inputs=True)
    _update_report_clipping_stats(report, iosys)
    assert report.y_clip_count > 0
    assert report.y_clip_abs_sum > 0
    assert report.y_clip_abs_max > 0
    assert report.f_clip_count > 0
    assert report.f_clip_abs_sum > 0
    assert report.f_clip_abs_max > 0

    clean_report = ProcessReportMRIO(source="oecd_v2025", requested=[2019])
    clean_iosys = build_dummy_iosystem()
    _update_report_clipping_stats(clean_report, clean_iosys)
    assert clean_report.y_clip_count == 0
    assert clean_report.f_clip_count == 0


def test_process_report_formats_multiple_raw_correction_logs() -> None:
    report = ProcessReportMRIO(
        source="exiobase_3102_ixi",
        requested=[2019, 2020],
        raw_corrected_value_scopes=[
            {
                "year": 2019,
                "region": "LU",
                "extension": "water",
                "stressor_family": "water consumption",
                "correction_method": "ols_level",
                "correction_reason": "too small",
            }
        ],
        raw_corrected_value_log_paths=[Path("log_a.csv"), Path("log_b.csv")],
    )

    text = str(report)

    assert text.strip()
    assert "log_a.csv" in text
    assert "log_b.csv" in text
