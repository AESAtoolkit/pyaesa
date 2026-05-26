"""Data loading and year resolution for setup orchestration."""

from typing import cast

import pandas as pd

from pyaesa.asocc.orchestration.setup.formatting.formatting import (
    _format_year_ranges,
    _process_mrio_hint,
)
from pyaesa.asocc.orchestration.setup.request.types import _YearBundle
from pyaesa.process.mrios.utils.aggregation.aggregation import read_agg_map
from pyaesa.process.mrios.utils.io.metadata import _read_metadata
from pyaesa.process.mrios.utils.io.paths import _get_agg_map_path, _get_metadata_path

from ....data.load_mrio import _years_from_metadata
from ....data.load_pop_gdp import _load_processed_table
from ....data.paths import _get_processed_pop_gdp_table_path
from ....data.source_schema import (
    ISO3_SOURCE_KEY,
    max_modeled_year_for_source,
    min_modeled_year_for_source,
    region_code_column_for_source,
)
from ...method_scope import _max_historical_mrio_year
from ...projection.config.config import _normalize_year_selector


def _year_columns(df: pd.DataFrame) -> list[int]:
    """Return sorted integer year columns from a processed pop/gdp table."""
    years = [int(str(c)) for c in df.columns if str(c).isdigit()]
    return sorted(set(years))


def _normalize_years_selector(
    value: int | list[int] | range | None,
) -> list[int] | None:
    """Normalize year selectors, keeping None for default behaviors."""
    if value is None:
        return None
    if isinstance(value, list) and not value:
        return None
    return _normalize_year_selector(value=value, name="years")


def _resolve_years(
    *,
    years: int | list[int] | range | None,
    source: str,
    agg_version: str | None,
    agg_reg: bool | None = None,
    agg_sec: bool | None = None,
    historical_year_cap: int | None = None,
    upstream_analysis: bool = False,
):
    """Resolve studied and historical years from metadata."""
    available_all_years = _years_from_metadata(source, agg_version)
    if not available_all_years:
        raise ValueError("No processed historical MRIO years are available for this source/domain.")
    coverage_min = int(min(available_all_years))
    coverage_max = int(max(available_all_years))
    coverage_label = f"{coverage_min}-{coverage_max}"
    years_norm = _normalize_years_selector(years)
    if years_norm is not None:
        requested_years = list(dict.fromkeys(int(y) for y in years_norm))
    else:
        requested_years = list(available_all_years)
    if not requested_years:
        raise ValueError("No years available for allocation.")
    available_years_set = set(available_all_years)
    missing_years = sorted(y for y in requested_years if y not in available_years_set)
    hard_min_year = cast(int, min_modeled_year_for_source(source))
    hard_max_year = cast(int, max_modeled_year_for_source(source))
    blocking_before_modeled = [y for y in missing_years if y < hard_min_year]
    if blocking_before_modeled:
        raise ValueError(
            "Requested MRIO years are before the source modeled historical start. "
            f"Coverage: {coverage_label}. "
            f"Invalid years: {_format_year_ranges(blocking_before_modeled)}."
        )
    # Missing years inside modeled horizon are treated as hard data gaps.
    blocking_missing = [y for y in missing_years if hard_min_year <= y <= hard_max_year]
    if blocking_missing:
        process_hint = _process_mrio_hint(
            source=source,
            years=blocking_missing,
            agg_version=agg_version,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
            keep_intermediate_uncasext=upstream_analysis,
        )
        raise ValueError(
            "Requested MRIO years are missing from processed MRIO outputs inside the "
            "source modeled historical coverage. "
            f"Coverage: {coverage_label}. "
            f"Missing years: {_format_year_ranges(blocking_missing)}. "
            f"Run: {process_hint}"
        )
    resolved_years = requested_years
    max_year = max(resolved_years)
    # Historical years include all available years up to max requested year
    # because AR/PR-HR windows can depend on earlier periods.
    historical_cap = (
        max_year if historical_year_cap is None else min(max_year, int(historical_year_cap))
    )
    historical_years = sorted(y for y in available_all_years if y <= historical_cap)
    expected_historical = set(range(min(historical_years), max(historical_years) + 1))
    missing_historical = sorted(expected_historical - set(historical_years))
    if missing_historical:
        process_hint = _process_mrio_hint(
            source=source,
            years=missing_historical,
            agg_version=agg_version,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
            keep_intermediate_uncasext=upstream_analysis,
        )
        raise ValueError(
            "Historical MRIO years are not consecutive for the requested run. "
            f"Coverage: {coverage_label}. "
            f"Missing years: {_format_year_ranges(missing_historical)}. "
            f"Run: {process_hint}"
        )
    return _YearBundle(
        resolved_years=resolved_years,
        historical_years=historical_years,
        max_year=max_year,
        out_of_range_years=missing_years,
    )


def _resolve_years_iso3(
    *,
    years: int | list[int] | range | None,
    wb_df: pd.DataFrame,
    ssp_df: pd.DataFrame,
) -> _YearBundle:
    """Resolve studied years for ISO3-only mode from WB/SSP tables."""
    available_years = sorted(set(_year_columns(wb_df)) | set(_year_columns(ssp_df)))
    years_norm = _normalize_years_selector(years)
    if years_norm is not None:
        requested_years = list(dict.fromkeys(int(y) for y in years_norm))
    else:
        requested_years = list(available_years)
    _validate_pop_gdp_year_coverage(
        years=requested_years,
        wb_df=wb_df,
        ssp_df=ssp_df,
    )
    max_year = max(requested_years)
    historical_years = sorted(y for y in available_years if y <= max_year)
    return _YearBundle(
        resolved_years=requested_years,
        historical_years=historical_years,
        max_year=max_year,
        out_of_range_years=[],
    )


def _validate_pop_gdp_year_coverage(
    *,
    years: list[int],
    wb_df: pd.DataFrame,
    ssp_df: pd.DataFrame,
) -> None:
    """Fail fast when requested years exceed WB/SSP processed coverage."""
    available_years = sorted(set(_year_columns(wb_df)) | set(_year_columns(ssp_df)))
    missing_years = sorted(int(year) for year in years if int(year) not in set(available_years))
    if not missing_years:
        return
    raise ValueError(
        "Requested years are missing in processed pop/gdp tables. "
        f"Missing years: {_format_year_ranges(missing_years)}."
    )


def _validate_region_filter_labels(
    *,
    source: str,
    agg_version: str | None,
    agg_reg: bool | None,
    filters: dict[str, list[str] | None],
    wb_df: pd.DataFrame,
    ssp_df: pd.DataFrame,
) -> None:
    """Validate region filter labels against source domain region codes."""
    if source == ISO3_SOURCE_KEY:
        return

    requested = set()
    for key in ("r_p", "r_c", "r_f"):
        values = filters.get(key)
        if values:
            requested.update(str(v).strip() for v in values if str(v).strip())
    if not requested:
        return

    region_col = region_code_column_for_source(source)
    allowed = set(wb_df[region_col].dropna().astype(str))
    allowed.update(ssp_df[region_col].dropna().astype(str))
    if agg_reg and agg_version:
        map_path = _get_agg_map_path(
            source,
            kind="reg",
            agg_version=agg_version,
        )
        map_df = read_agg_map(map_path)
        allowed.update(map_df["aggregated_mrio"].dropna().astype(str))

    invalid = sorted(v for v in requested if v not in allowed)
    if not invalid:
        return
    raise ValueError(
        f"Region filter labels are not valid for source '{source}'. "
        f"Invalid labels: {invalid[:20]}. Expected '{region_col}' labels."
    )


def _validate_sector_filter_labels(
    *,
    source: str,
    agg_version: str | None,
    filters: dict[str, list[str] | None],
) -> None:
    """Validate sector filter labels against processed MRIO metadata labels."""
    if source == ISO3_SOURCE_KEY:
        return

    values = filters.get("s_p")
    if not values:
        return
    requested = sorted({str(value).strip() for value in values if str(value).strip()})
    if not requested:
        return

    metadata_path = _get_metadata_path(source, matrix_version=agg_version)
    domain_label = f"source='{source}', matrix_version='{agg_version or 'original_classification'}'"
    metadata = _read_metadata(source, matrix_version=agg_version)
    sectors_used = metadata["labels"]["sectors_used"]
    allowed = {str(value).strip() for value in sectors_used if str(value).strip()}
    invalid = sorted(value for value in requested if value not in allowed)
    if not invalid:
        return
    raise ValueError(
        f"Sector filter labels are not valid for {domain_label}. "
        f"Invalid labels: {invalid[:20]}. Expected labels from processed MRIO "
        "metadata field 'labels.sectors_used'. "
        f"Metadata file: '{metadata_path}'."
    )


def _resolve_reference_years(
    *,
    reference_years: int | list[int] | range | None,
    historical_years: list[int],
    source: str,
    agg_version: str | None,
    agg_reg: bool | None,
    agg_sec: bool | None,
) -> list[int] | None:
    """Normalize and validate reference years against historical MRIO years."""
    if reference_years is None:
        return None
    reference_years = _normalize_year_selector(
        value=reference_years,
        name="reference_years",
    )
    historical_set = set(historical_years)
    missing_ref = sorted(y for y in reference_years if y not in historical_set)
    if missing_ref:
        hist_min = int(min(historical_years))
        hist_max = _max_historical_mrio_year(historical_years=historical_years)
        # Reference years must exist as historical MRIO years in the active domain.
        process_hint = _process_mrio_hint(
            source=source,
            years=missing_ref,
            agg_version=agg_version,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
        )
        raise ValueError(
            "Requested reference_years are not available in historical "
            f"MRIO coverage {hist_min}-{hist_max}: {_format_year_ranges(missing_ref)}. "
            f"Run: {process_hint}"
        )
    return reference_years


def _load_source_tables(
    *,
    source: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load processed WB/SSP tables for allocation."""
    wb_raw = _load_processed_table(_get_processed_pop_gdp_table_path(dataset="wb"))
    ssp_raw = _load_processed_table(_get_processed_pop_gdp_table_path(dataset="ssp"))
    wb_base = _aggregate_pop_gdp_to_source_regions(df=wb_raw, source=source)
    ssp_base = _aggregate_pop_gdp_to_source_regions(df=ssp_raw, source=source)
    return wb_base, ssp_base, wb_raw, ssp_raw


def _aggregate_pop_gdp_to_source_regions(*, df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Aggregate ISO level processed pop/gdp rows to the active source region code."""
    region_col = region_code_column_for_source(source)
    year_cols = [str(col) for col in df.columns if str(col).isdigit()]
    key_cols = ["variable", region_col]
    if "ssp_scenario" in df.columns:
        key_cols = ["ssp_scenario", *key_cols]
    work = df[key_cols + year_cols].copy()
    aggregated = work.groupby(key_cols, as_index=False, dropna=False)[year_cols].sum(min_count=1)
    return cast(pd.DataFrame, aggregated)
