"""Historical coverage validation for setup orchestration."""

from typing import cast

from ....data.source_schema import is_exio_source, min_modeled_year_for_source
from ....methods.registry.registry import REGISTRY
from pyaesa.asocc.orchestration.setup.formatting.formatting import (
    _format_year_ranges,
    _process_mrio_hint,
)


def _requires_ar_or_prhr_history(*, selection, fu_code: str) -> bool:
    """Return whether selected methods require strict historical continuity."""
    if any(
        REGISTRY.method_requires_contiguous_history(name, level="L1")
        for name in selection.selected_l1
    ):
        return True
    if any(
        REGISTRY.method_requires_contiguous_history(pair[1], level="L1")
        for pair in selection.combined
    ):
        return True
    if any(
        REGISTRY.method_requires_contiguous_history(name, level="L2", fu_code=fu_code)
        for name in selection.selected_l2_one_step
    ):
        return True
    if any(
        REGISTRY.method_requires_contiguous_history(pair[0], level="L2", fu_code=fu_code)
        for pair in selection.combined
    ):
        return True
    return False


def _validate_history_since_baseline(
    *,
    source: str,
    group_version: str | None,
    group_reg: bool | None,
    group_sec: bool | None,
    historical_years: list[int],
    selection,
    fu_code: str,
) -> None:
    """Require contiguous MRIO history from source baseline for AR/PR-HR runs."""
    if not is_exio_source(source):
        return
    # Only AR/PR-HR families require strict continuity from baseline year.
    if not _requires_ar_or_prhr_history(selection=selection, fu_code=fu_code):
        return
    baseline = cast(int, min_modeled_year_for_source(source))
    years = sorted(int(y) for y in historical_years)
    expected = set(range(int(baseline), max(years) + 1))
    available = set(years)
    missing = sorted(expected - available)
    if missing:
        process_hint = _process_mrio_hint(
            source=source,
            years=missing,
            group_version=group_version,
            group_reg=group_reg,
            group_sec=group_sec,
        )
        raise ValueError(
            "AR/PR-HR methods require contiguous MRIO years since "
            f"{baseline}. Missing years: {_format_year_ranges(missing)}. "
            f"Run: {process_hint}"
        )
