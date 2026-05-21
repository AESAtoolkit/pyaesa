from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from pyaesa.download.pop_gdp.contracts import (
    FUTURE_YEARS,
    GDP_SSP_INDICATOR,
    PAST_YEAR_MIN,
    POP_SSP_INDICATOR,
)
from pyaesa.download.pop_gdp.download_wb import (
    GDP_WB_INDICATOR,
    POP_WB_INDICATOR,
)
from pyaesa.process.pop_gdp.process_pop_gdp import (
    _load_matching,
    _load_raw_frame,
    process_pop_gdp,
)
from pyaesa.process.pop_gdp.io.metadata import _meta_covers, _read_meta, _write_meta
from pyaesa.process.pop_gdp.io.paths import (
    _clear_processed_dataset_scope,
    _get_log_path,
    _get_metadata_path,
    _get_processed_output_path,
    _get_ssp_matching_path,
    _get_wb_matching_path,
    _log_dir,
)
from pyaesa.process.pop_gdp.pipeline.parent_aggregation import (
    apply_parent_aggregation,
)
from pyaesa.process.pop_gdp.sources.ssp import (
    _interpolate_years,
    _name_to_iso3,
    _process_ssp_dataset,
)
from pyaesa.process.pop_gdp.sources.wb import (
    _add_imf_taiwan_and_adjust_china,
    _fill_missing_edges_loglin,
    _process_wb_dataset,
    _us_price_level_ratio_2017_over_2021,
)
from pyaesa.process.pop_gdp.pipeline.tabular import (
    coerce_finite_float,
)
from tests.package.helpers.data_processing_dummy import (
    build_pop_gdp_matching_frame,
    build_ssp_raw_frame,
    build_wb_raw_frames,
    write_pop_gdp_matching_files,
    write_pop_gdp_raw_files,
)


def test_processed_pop_gdp_paths_and_metadata_cover_real_files(project_repo: Path) -> None:
    assert _log_dir() == project_repo / "data_processed" / "pop_gdp" / "logs"
    assert (
        _get_processed_output_path("wb")
        == project_repo / "data_processed" / "pop_gdp" / "wb_processed.csv"
    )
    assert _get_log_path("wb_fill_log.csv") == _log_dir() / "wb_fill_log.csv"
    assert _get_metadata_path("wb_processed").name == "wb_processed_meta.json"
    assert _get_wb_matching_path("exiobase_396_ixi").name == "wb_exiobase_3_matching.csv"
    assert _get_ssp_matching_path("exiobase_3102_pxp").name == "ssp_exiobase_3_matching.csv"
    with pytest.raises(ValueError):
        _get_wb_matching_path("unknown")
    _clear_processed_dataset_scope("wb")

    assert _read_meta("wb_processed") is None
    _write_meta("wb_processed", 2000, 2005)
    meta = _read_meta("wb_processed")
    assert meta is not None
    assert meta["begin_year"] == 2000
    assert meta["end_year"] == 2005
    assert "timestamp" in meta
    assert _meta_covers(meta, 2001, 2004) is True
    assert _meta_covers(meta, 1999, 2004) is False
    assert _meta_covers(meta, 2001, 2006) is False


def test_process_wb_utils_cover_transformations_and_dataset_processing() -> None:
    assert coerce_finite_float("1.5") == 1.5
    assert coerce_finite_float("bad") is None
    assert coerce_finite_float(float("inf")) is None
    assert 0.5 < _us_price_level_ratio_2017_over_2021() < 1.0

    year_cols = ["2000", "2001"]
    wb_raw, imf_raw = build_wb_raw_frames(years=[2000, 2001])
    adjusted = _add_imf_taiwan_and_adjust_china(
        pd.concat([wb_raw, imf_raw], ignore_index=True), year_cols
    )
    adjusted_chn_pop = adjusted.loc[
        (adjusted["iso3_code"] == "CHN") & (adjusted["variable"] == POP_WB_INDICATOR),
        "2000",
    ].iloc[0]
    assert adjusted_chn_pop == 1377.0

    no_twn = _add_imf_taiwan_and_adjust_china(wb_raw.copy(), year_cols)
    assert pd.isna(
        no_twn.loc[
            (no_twn["iso3_code"] == "CHN") & (no_twn["variable"] == POP_WB_INDICATOR),
            "2000",
        ].iloc[0]
    )

    with pytest.raises(ValueError):
        _add_imf_taiwan_and_adjust_china(
            pd.DataFrame(
                [
                    {
                        "wb_full_name": "China",
                        "iso3_code": "CHN",
                        "variable": POP_WB_INDICATOR,
                        "unit": "Persons",
                        "2000": 1.0,
                    },
                    {
                        "wb_full_name": "Taiwan",
                        "iso3_code": "TWN",
                        "variable": POP_WB_INDICATOR,
                        "unit": "Persons",
                        "2000": 2.0,
                    },
                ]
            ),
            ["2000"],
        )
    with pytest.raises(ValueError):
        _add_imf_taiwan_and_adjust_china(
            pd.DataFrame(
                [
                    {
                        "wb_full_name": "China",
                        "iso3_code": "CHN",
                        "variable": GDP_WB_INDICATOR,
                        "unit": "USD_2021/yr",
                        "2000": 1.0,
                    },
                    {
                        "wb_full_name": "Taiwan",
                        "iso3_code": "TWN",
                        "variable": GDP_WB_INDICATOR,
                        "unit": "USD_2021/yr",
                        "2000": 2.0,
                    },
                ]
            ),
            ["2000"],
        )

    parent_mapping = build_pop_gdp_matching_frame()
    wb_subset = cast(
        pd.DataFrame,
        wb_raw[["wb_full_name", "iso3_code", "variable", "unit", "2000", "2001"]],
    )
    aggregated = apply_parent_aggregation(
        wb_subset,
        year_cols,
        parent_mapping,
        name_column="wb_full_name",
        group_columns=["variable", "unit"],
    )
    assert "EUR" in aggregated["iso3_code"].values
    assert "FRA" not in aggregated["iso3_code"].values
    assert apply_parent_aggregation(
        wb_raw,
        year_cols,
        pd.DataFrame({"iso3_code": ["FRA"]}),
        name_column="wb_full_name",
        group_columns=["variable", "unit"],
    ).equals(wb_raw)
    assert apply_parent_aggregation(
        wb_subset,
        year_cols,
        parent_mapping.assign(group_parent="NO"),
        name_column="wb_full_name",
        group_columns=["variable", "unit"],
    ).equals(wb_subset)
    assert apply_parent_aggregation(
        wb_subset,
        year_cols,
        pd.DataFrame([{"iso3_code": "FRA", "group_parent": "YES", "parent_iso3_code": ""}]),
        name_column="wb_full_name",
        group_columns=["variable", "unit"],
    ).equals(wb_subset)
    assert apply_parent_aggregation(
        wb_subset,
        year_cols,
        pd.DataFrame([{"iso3_code": "ZZZ", "group_parent": "YES", "parent_iso3_code": "EUR"}]),
        name_column="wb_full_name",
        group_columns=["variable", "unit"],
    ).equals(wb_subset)
    existing_parent = apply_parent_aggregation(
        pd.DataFrame(
            [
                {
                    "wb_full_name": "Europe",
                    "iso3_code": "EUR",
                    "variable": POP_WB_INDICATOR,
                    "unit": "Persons",
                    "2000": 1.0,
                    "2001": 1.0,
                },
                {
                    "wb_full_name": "France",
                    "iso3_code": "FRA",
                    "variable": POP_WB_INDICATOR,
                    "unit": "Persons",
                    "2000": 10.0,
                    "2001": 11.0,
                },
            ]
        ),
        year_cols,
        pd.DataFrame([{"iso3_code": "FRA", "group_parent": "YES", "parent_iso3_code": "EUR"}]),
        name_column="wb_full_name",
        group_columns=["variable", "unit"],
    )
    assert existing_parent.loc[existing_parent["iso3_code"] == "EUR", "2000"].iloc[0] == 11.0

    fill_source = pd.DataFrame(
        [
            {
                "wb_full_name": "France",
                "iso3_code": "FRA",
                "variable": POP_WB_INDICATOR,
                "unit": "Persons",
                "2000": pd.NA,
                "2001": 10.0,
                "2002": 20.0,
            },
            {
                "wb_full_name": "Nowhere",
                "iso3_code": "NUL",
                "variable": POP_WB_INDICATOR,
                "unit": "Persons",
                "2000": pd.NA,
                "2001": pd.NA,
                "2002": pd.NA,
            },
        ]
    )
    filled, fill_log = _fill_missing_edges_loglin(fill_source, ["2000", "2001", "2002"])
    assert filled.loc[0, "2000"] == pytest.approx(5.0)
    assert fill_log["fill_method"].tolist() == ["loglin_leading"]
    trailing_source = pd.DataFrame(
        [
            {
                "wb_full_name": "France",
                "iso3_code": "FRA",
                "variable": POP_WB_INDICATOR,
                "unit": "Persons",
                "2000": 10.0,
                "2001": 20.0,
                "2002": pd.NA,
            }
        ]
    )
    trailing_filled, trailing_log = _fill_missing_edges_loglin(
        trailing_source,
        ["2000", "2001", "2002"],
    )
    assert trailing_filled.loc[0, "2002"] == pytest.approx(40.0)
    assert trailing_log["fill_method"].tolist() == ["loglin_trailing"]
    no_year_fill, no_year_log = _fill_missing_edges_loglin(fill_source, [])
    assert no_year_fill.equals(fill_source)
    assert no_year_log.empty is True

    processed, wb_log = _process_wb_dataset(
        wb_raw,
        imf_raw,
        [2000, 2001],
        parent_mapping,
        parent_mapping,
    )
    assert wb_log.empty is True
    assert set(processed["variable"]) == {GDP_WB_INDICATOR, POP_WB_INDICATOR}
    assert set(processed["iso3_code"]) >= {"EUR", "USA", "CHN", "TWN"}
    assert processed.loc[processed["variable"] == GDP_WB_INDICATOR, "unit"].eq("USD_2017/yr").all()

    pop_only_processed, _ = _process_wb_dataset(
        cast(pd.DataFrame, wb_raw[wb_raw["variable"] == POP_WB_INDICATOR]),
        cast(pd.DataFrame, imf_raw[imf_raw["variable"] == POP_WB_INDICATOR]),
        [2000, 2001],
        parent_mapping,
        parent_mapping,
    )
    assert set(pop_only_processed["variable"]) == {POP_WB_INDICATOR}
    assert GDP_WB_INDICATOR not in pop_only_processed["variable"].values


def test_process_ssp_utils_cover_transformations_and_dataset_processing() -> None:
    assert coerce_finite_float("1.5") == 1.5
    assert coerce_finite_float("bad") is None
    assert coerce_finite_float(float("inf")) is None

    long_df = pd.DataFrame(
        [
            {
                "model": "IIASA-WiC POP",
                "ssp_scenario": "SSP2",
                "ssp_full_name": "France",
                "iso3_code": "FRA",
                "variable": POP_SSP_INDICATOR,
                "unit": "Persons",
                "year": 2025,
                "value": 10.0,
            },
            {
                "model": "IIASA-WiC POP",
                "ssp_scenario": "SSP2",
                "ssp_full_name": "France",
                "iso3_code": "FRA",
                "variable": POP_SSP_INDICATOR,
                "unit": "Persons",
                "year": 2027,
                "value": 14.0,
            },
        ]
    )
    interpolated = _interpolate_years(long_df, [2025, 2026, 2027])
    assert interpolated.loc[interpolated["year"] == 2026, "value"].iloc[0] == 12.0

    matching = build_pop_gdp_matching_frame()
    ssp_raw = build_ssp_raw_frame(years=[2025, 2030])
    aggregated = apply_parent_aggregation(
        pd.DataFrame(
            [
                {
                    "model": "IIASA-WiC POP",
                    "ssp_scenario": "SSP2",
                    "ssp_full_name": "France",
                    "iso3_code": "FRA",
                    "variable": POP_SSP_INDICATOR,
                    "unit": "Persons",
                    "2025": 10.0,
                },
                {
                    "model": "IIASA-WiC POP",
                    "ssp_scenario": "SSP2",
                    "ssp_full_name": "Germany",
                    "iso3_code": "DEU",
                    "variable": POP_SSP_INDICATOR,
                    "unit": "Persons",
                    "2025": 20.0,
                },
            ]
        ),
        ["2025"],
        matching,
        name_column="ssp_full_name",
        group_columns=["model", "ssp_scenario", "variable", "unit"],
    )
    assert "EUR" in aggregated["iso3_code"].values
    assert apply_parent_aggregation(
        ssp_raw,
        ["2025", "2030"],
        pd.DataFrame({"iso3_code": ["FRA"]}),
        name_column="ssp_full_name",
        group_columns=["model", "ssp_scenario", "variable", "unit"],
    ).equals(ssp_raw)
    assert apply_parent_aggregation(
        ssp_raw,
        ["2025", "2030"],
        matching.assign(group_parent="NO"),
        name_column="ssp_full_name",
        group_columns=["model", "ssp_scenario", "variable", "unit"],
    ).equals(ssp_raw)
    ssp_with_iso = pd.DataFrame(
        [
            {
                "model": "IIASA-WiC POP",
                "ssp_scenario": "SSP2",
                "ssp_full_name": "France",
                "iso3_code": "FRA",
                "variable": POP_SSP_INDICATOR,
                "unit": "Persons",
                "2025": 10.0,
                "2030": 11.0,
            }
        ]
    )
    assert apply_parent_aggregation(
        ssp_with_iso,
        ["2025", "2030"],
        pd.DataFrame([{"iso3_code": "FRA", "group_parent": "YES", "parent_iso3_code": ""}]),
        name_column="ssp_full_name",
        group_columns=["model", "ssp_scenario", "variable", "unit"],
    ).equals(ssp_with_iso)
    assert apply_parent_aggregation(
        ssp_with_iso,
        ["2025", "2030"],
        pd.DataFrame([{"iso3_code": "ZZZ", "group_parent": "YES", "parent_iso3_code": "EUR"}]),
        name_column="ssp_full_name",
        group_columns=["model", "ssp_scenario", "variable", "unit"],
    ).equals(ssp_with_iso)
    existing_parent_ssp = apply_parent_aggregation(
        pd.DataFrame(
            [
                {
                    "model": "IIASA-WiC POP",
                    "ssp_scenario": "SSP2",
                    "ssp_full_name": "Europe",
                    "iso3_code": "EUR",
                    "variable": POP_SSP_INDICATOR,
                    "unit": "Persons",
                    "2025": 1.0,
                },
                {
                    "model": "IIASA-WiC POP",
                    "ssp_scenario": "SSP2",
                    "ssp_full_name": "France",
                    "iso3_code": "FRA",
                    "variable": POP_SSP_INDICATOR,
                    "unit": "Persons",
                    "2025": 10.0,
                },
            ]
        ),
        ["2025"],
        pd.DataFrame([{"iso3_code": "FRA", "group_parent": "YES", "parent_iso3_code": "EUR"}]),
        name_column="ssp_full_name",
        group_columns=["model", "ssp_scenario", "variable", "unit"],
    )
    assert (
        existing_parent_ssp.loc[existing_parent_ssp["iso3_code"] == "EUR", "2025"].iloc[0] == 11.0
    )

    assert _name_to_iso3("") is None
    assert _name_to_iso3("Turkey") == "TUR"
    assert _name_to_iso3("France") == "FRA"
    assert _name_to_iso3("Not a Country") is None

    processed = _process_ssp_dataset(
        ssp_raw,
        [2025, 2026, 2030],
        matching,
        matching,
    )
    assert set(processed["variable"]) == {GDP_SSP_INDICATOR, POP_SSP_INDICATOR}
    assert "Not a Country" not in processed["ssp_full_name"].values
    assert processed.loc[processed["variable"] == POP_SSP_INDICATOR, "unit"].eq("Persons").all()
    assert processed.loc[processed["variable"] == GDP_SSP_INDICATOR, "unit"].eq("USD_2017/yr").all()
    assert "EUR" in processed["iso3_code"].values

    sparse_processed = _process_ssp_dataset(
        build_ssp_raw_frame(years=[2025]),
        [2025, 2026],
        matching,
        matching,
    )
    assert set(sparse_processed["variable"]) == {GDP_SSP_INDICATOR, POP_SSP_INDICATOR}
    assert "2026" in sparse_processed.columns
    sparse_year_2026 = cast(pd.Series, sparse_processed["2026"])
    assert bool(sparse_year_2026.isna().to_numpy().any())

    canonical_processed = _process_ssp_dataset(
        pd.DataFrame(
            [
                {
                    "model": "IIASA-WiC POP",
                    "ssp_scenario": "SSP2",
                    "ssp_full_name": "France",
                    "variable": POP_SSP_INDICATOR,
                    "unit": "Persons",
                    "2025": 65.0,
                },
                {
                    "model": "IIASA-WiC POP",
                    "ssp_scenario": "SSP2",
                    "ssp_full_name": "France",
                    "variable": GDP_SSP_INDICATOR,
                    "unit": "USD_2017/yr",
                    "2025": 400.0,
                },
            ]
        ),
        [2025],
        matching,
        matching,
    )
    assert set(canonical_processed["variable"]) == {GDP_SSP_INDICATOR, POP_SSP_INDICATOR}

    with pytest.raises(ValueError):
        _process_ssp_dataset(
            pd.DataFrame(
                [
                    {
                        "model": "IIASA-WiC POP",
                        "ssp_full_name": "France",
                        "variable": POP_SSP_INDICATOR,
                        "unit": "Persons",
                        "2025": 65.0,
                    }
                ]
            ),
            [2025],
            matching,
            matching,
        )


def test_process_pop_gdp_wrappers_use_real_files_and_dispatch_flags(project_repo: Path) -> None:
    wb_years = [PAST_YEAR_MIN, 2026]
    ssp_years = [min(FUTURE_YEARS), max(FUTURE_YEARS)]
    _, _, _, matching = write_pop_gdp_raw_files(
        project_repo,
        wb_years=wb_years,
        ssp_years=ssp_years,
    )
    write_pop_gdp_matching_files(project_repo, matching=matching)

    assert _load_raw_frame("wb").shape[0] > 0
    with pytest.raises(RuntimeError):
        _load_raw_frame("missing")
    assert _load_matching(_get_wb_matching_path("oecd_v2025")).shape[0] > 0
    missing_match = _get_wb_matching_path("oecd_v2025")
    missing_match.unlink()
    with pytest.raises(RuntimeError):
        _load_matching(_get_wb_matching_path("oecd_v2025"))
    write_pop_gdp_matching_files(project_repo, matching=matching)

    process_pop_gdp(past_years=True, future_years=False, refresh=True)
    wb_out = _get_processed_output_path("wb")
    assert wb_out.exists()
    assert "2026" in pd.read_csv(wb_out, nrows=0).columns
    assert _get_log_path("wb_fill_log.csv").exists()
    assert _read_meta("wb_processed") is not None
    process_pop_gdp(past_years=True, future_years=False, refresh=False)
    _write_meta("wb_processed", PAST_YEAR_MIN + 1, 2025)
    process_pop_gdp(past_years=True, future_years=False, refresh=False)
    wb_meta = _read_meta("wb_processed")
    assert wb_meta is not None
    assert wb_meta["begin_year"] == PAST_YEAR_MIN
    stale_wb_log = _get_log_path("wb_fill_log.csv")
    stale_wb_log.write_text("stale", encoding="utf-8")
    _get_metadata_path("wb_processed").write_text("{bad", encoding="utf-8")
    process_pop_gdp(past_years=True, future_years=False, refresh=True)
    assert stale_wb_log.read_text(encoding="utf-8") != "stale"

    process_pop_gdp(past_years=False, future_years=True, refresh=True)
    ssp_out = _get_processed_output_path("ssp")
    assert ssp_out.exists()
    assert _read_meta("ssp_processed") is not None
    process_pop_gdp(past_years=False, future_years=True, refresh=False)
    _write_meta("ssp_processed", min(FUTURE_YEARS) + 1, max(FUTURE_YEARS) - 1)
    process_pop_gdp(past_years=False, future_years=True, refresh=False)
    ssp_meta = _read_meta("ssp_processed")
    assert ssp_meta is not None
    assert ssp_meta["begin_year"] == min(FUTURE_YEARS)
    _get_metadata_path("ssp_processed").write_text("{bad", encoding="utf-8")
    process_pop_gdp(past_years=False, future_years=True, refresh=True)

    process_pop_gdp(
        past_years=True,
        future_years=True,
        refresh=True,
    )
    assert _get_processed_output_path("wb").exists()
    assert _get_processed_output_path("ssp").exists()
    process_pop_gdp(
        past_years=False,
        future_years=True,
        refresh=False,
    )
    assert _get_processed_output_path("ssp").exists()

    process_pop_gdp(past_years=False, future_years=False, refresh=False)
