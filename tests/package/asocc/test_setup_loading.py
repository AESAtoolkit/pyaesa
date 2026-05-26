import importlib

import pandas as pd
import pytest

mod = importlib.import_module("pyaesa.asocc.orchestration.setup.loading.loading")


def test_year_columns_and_normalize_selector() -> None:
    df = pd.DataFrame(columns=["x", "2000", "2001", 2001])
    assert mod._year_columns(df) == [2000, 2001]

    assert mod._normalize_years_selector(None) is None
    assert mod._normalize_years_selector([]) is None
    assert mod._normalize_years_selector([2001, 2000, 2000]) == [2000, 2001]
    assert mod._normalize_years_selector(range(2000, 2002)) == [2000, 2001]


def test_resolve_years_real_paths(allocation_dummy_repo) -> None:
    out = mod._resolve_years(
        years=None,
        source="oecd_v2025",
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
    )
    assert out.resolved_years == [2005, 2006]
    assert out.historical_years == [2005, 2006]
    assert out.out_of_range_years == []

    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version="empty",
        sectors_used=["D"],
        regions_used=["FR", "US"],
        years=[],
    )
    with pytest.raises(ValueError):
        mod._resolve_years(
            years=None,
            source="oecd_v2025",
            agg_version="empty",
            agg_reg=False,
            agg_sec=False,
        )

    with pytest.raises(ValueError):
        mod._resolve_years(
            years=range(2005, 2005),
            source="oecd_v2025",
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
        )

    with pytest.raises(ValueError):
        mod._resolve_years(
            years=[1994],
            source="oecd_v2025",
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
        )

    with pytest.raises(ValueError):
        mod._resolve_years(
            years=[2007],
            source="oecd_v2025",
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
            upstream_analysis=True,
        )

    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version="gap",
        sectors_used=["D"],
        regions_used=["FR", "US"],
        years=[2005, 2007],
    )
    with pytest.raises(ValueError):
        mod._resolve_years(
            years=None,
            source="oecd_v2025",
            agg_version="gap",
            agg_reg=False,
            agg_sec=False,
        )

    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version="sparse_future",
        sectors_used=["D"],
        regions_used=["FR", "US"],
        years=[2020, 2025],
    )
    out_sparse = mod._resolve_years(
        years=[2024],
        source="oecd_v2025",
        agg_version="sparse_future",
        agg_reg=False,
        agg_sec=False,
    )
    assert out_sparse.historical_years == [2020]
    assert out_sparse.out_of_range_years == [2024]


def test_resolve_years_iso3_and_pop_gdp_coverage(allocation_dummy_repo) -> None:
    wb, ssp, _, _ = mod._load_source_tables(source="iso3")

    out = mod._resolve_years_iso3(years=None, wb_df=wb, ssp_df=ssp)
    assert out.resolved_years == [2005, 2006, 2030]
    assert out.max_year == 2030
    out_selected = mod._resolve_years_iso3(years=[2005], wb_df=wb, ssp_df=ssp)
    assert out_selected.resolved_years == [2005]

    mod._validate_pop_gdp_year_coverage(years=[2005, 2030], wb_df=wb, ssp_df=ssp)

    with pytest.raises(ValueError):
        mod._validate_pop_gdp_year_coverage(
            years=[2031],
            wb_df=wb,
            ssp_df=ssp,
        )


def test_validate_region_filter_labels_real_paths(allocation_dummy_repo) -> None:
    wb, ssp, _, _ = mod._load_source_tables(source="oecd_v2025")

    mod._validate_region_filter_labels(
        source=mod.ISO3_SOURCE_KEY,
        agg_version=None,
        agg_reg=False,
        filters={"r_p": ["X"], "r_c": None, "r_f": None},
        wb_df=wb,
        ssp_df=ssp,
    )
    mod._validate_region_filter_labels(
        source="oecd_v2025",
        agg_version=None,
        agg_reg=False,
        filters={"r_p": None, "r_c": None, "r_f": None},
        wb_df=wb,
        ssp_df=ssp,
    )
    mod._validate_region_filter_labels(
        source="oecd_v2025",
        agg_version="demo_reg",
        agg_reg=True,
        filters={"r_p": ["EU"], "r_c": None, "r_f": None},
        wb_df=wb,
        ssp_df=ssp,
    )

    with pytest.raises(ValueError):
        mod._validate_region_filter_labels(
            source="oecd_v2025",
            agg_version=None,
            agg_reg=False,
            filters={"r_p": ["ZZZ"], "r_c": None, "r_f": None},
            wb_df=wb,
            ssp_df=ssp,
        )


def test_resolve_reference_years(
    allocation_dummy_repo,
) -> None:
    assert (
        mod._resolve_reference_years(
            reference_years=None,
            historical_years=[2005, 2006],
            source="oecd_v2025",
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
        )
        is None
    )
    assert mod._resolve_reference_years(
        reference_years=[2005],
        historical_years=[2005, 2006],
        source="oecd_v2025",
        agg_version=None,
        agg_reg=False,
        agg_sec=False,
    ) == [2005]

    with pytest.raises(ValueError):
        mod._resolve_reference_years(
            reference_years=[2007],
            historical_years=[2005, 2006],
            source="oecd_v2025",
            agg_version=None,
            agg_reg=False,
            agg_sec=False,
        )


def test_aggregate_pop_gdp_to_source_regions_and_load_source_tables(
    allocation_dummy_repo,
) -> None:
    df = pd.DataFrame(
        {
            "oecd_code": ["FR", "FR"],
            "variable": ["Population", "Population"],
            "ssp_scenario": ["SSP2", "SSP2"],
            "2005": [1.0, 2.0],
        }
    )
    out = mod._aggregate_pop_gdp_to_source_regions(df=df, source="oecd_v2025")
    assert float(out.loc[0, "2005"]) == 3.0

    wb_base, ssp_base, wb_raw, ssp_raw = mod._load_source_tables(source="oecd_v2025")
    assert "oecd_code" in wb_base.columns
    assert "oecd_code" in ssp_base.columns
    assert "wb_full_name" in wb_raw.columns
    assert "ssp_scenario" in ssp_raw.columns
