"""Year resolution ownership shared by deterministic_io_lca and its figure generation."""

from typing import cast

from pyaesa.asocc.orchestration.common_formatting import format_year_ranges
from pyaesa.asocc.orchestration.projection.config.config import (
    _normalize_year_selector,
)
from pyaesa.asocc.data.load_mrio import _years_from_metadata
from pyaesa.asocc.data.source_schema import (
    max_modeled_year_for_source,
    min_modeled_year_for_source,
)
from pyaesa.asocc.orchestration.setup.formatting.formatting import _process_mrio_hint


def resolve_years_strict(
    *,
    years: int | list[int] | range | None,
    source: str,
    agg_version: str | None,
    agg_reg: bool,
    agg_sec: bool,
    upstream_analysis: bool = False,
) -> list[int]:
    """Resolve studied years against the processed IO-LCA source coverage.

    Args:
        years: API year selector.
        source: Source key.
        agg_version: Aggregation version or ``None``.
        agg_reg: Region aggregation flag.
        agg_sec: Sector MRIO aggregation and disaggregation flag.
        upstream_analysis: Whether the caller also requires staged upstream
            diagnostics. This affects only the missing-year recovery hint.

    Returns:
        Sorted resolved year list.

    Raises:
        ValueError: If requested years are outside processed historical
            coverage or missing from the processed MRIO source scope.
    """
    available_years = _years_from_metadata(source, agg_version)
    if not available_years:
        process_hint = _process_mrio_hint(
            source=source,
            years=[],
            agg_version=agg_version,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
        )
        raise ValueError(
            "No processed historical MRIO years are available for IO-LCA. "
            f"source='{source}', agg_version={agg_version!r}. Run: {process_hint}"
        )
    years_norm = _normalize_year_selector(value=years, name="years")
    resolved_years = (
        sorted({int(year) for year in years_norm})
        if years_norm
        else sorted({int(year) for year in available_years})
    )
    available_years_set = {int(year) for year in available_years}
    coverage_min = int(min(available_years))
    coverage_max = int(max(available_years))
    missing_years = sorted(year for year in resolved_years if year not in available_years_set)
    if not missing_years:
        return resolved_years

    hard_min = min_modeled_year_for_source(source)
    hard_max = max_modeled_year_for_source(source)
    hard_min_year = int(cast(int, hard_min))
    hard_max_year = int(cast(int, hard_max))
    blocking_before_modeled = [year for year in missing_years if year < hard_min_year]
    if blocking_before_modeled:
        raise ValueError(
            "Requested MRIO years are before the source modeled historical start. "
            f"Coverage: {coverage_min}-{coverage_max}. "
            f"Invalid years: {format_year_ranges(blocking_before_modeled)}."
        )

    blocking_missing = [year for year in missing_years if hard_min_year <= year <= hard_max_year]
    if blocking_missing:
        hint = _process_mrio_hint(
            source=source,
            years=blocking_missing,
            agg_version=agg_version,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
            keep_intermediate_uncasext=upstream_analysis,
        )
        raise ValueError(
            "Requested IO-LCA years are inside the source modeled historical coverage "
            "but are missing from processed MRIO outputs for this source/domain. "
            f"Processed coverage: {coverage_min}-{coverage_max}. "
            f"Missing years: {format_year_ranges(blocking_missing)}. "
            f"Run: {hint}"
        )

    missing_processed_years = sorted(
        int(year) for year in missing_years if int(year) not in available_years_set
    )
    raise ValueError(
        "Requested IO-LCA years are outside processed historical MRIO coverage. "
        f"Coverage: {coverage_min}-{coverage_max}. "
        f"Invalid years: {format_year_ranges(missing_processed_years)}."
    )


def resolve_subset_years(
    *,
    years: int | list[int] | range | None,
    universe: list[int],
    label: str,
) -> list[int]:
    """Resolve optional year selector and enforce subset relation.

    Args:
        years: Optional selector value.
        universe: Parent year list that defines allowed values.
        label: Argument name used in error messages.

    Returns:
        Sorted resolved subset years. If selector is ``None``, returns
        ``universe`` sorted.

    Raises:
        ValueError: If selector contains values outside ``universe``.
    """
    if years is None:
        return sorted({int(year) for year in universe})
    resolved = _normalize_year_selector(value=years, name=label)
    allowed = {int(year) for year in universe}
    missing = sorted({int(year) for year in resolved if int(year) not in allowed})
    if missing:
        raise ValueError(
            f"{label} must be a subset of resolved years. "
            f"Out-of-scope years: {format_year_ranges(missing)}."
        )
    return sorted({int(year) for year in resolved})
