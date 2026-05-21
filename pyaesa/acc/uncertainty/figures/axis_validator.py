"""Figure axis validation for aCC uncertainty outputs."""

import pandas as pd

from pyaesa.acc.uncertainty.figures.scope_planner import FigureContext
from pyaesa.acc.uncertainty.sources.source_keys import (
    ASOCC_PROJECTION_SOURCE,
    ASOCC_REFERENCE_YEAR_SOURCE,
)
from pyaesa.shared.tabular.scalars import is_display_missing


def validate_inactive_axes_for_figures(*, identity: pd.DataFrame, context: FigureContext) -> None:
    """Reject figure requests that would need inactive uncertainty axis compression."""
    active = set(context.active_sources)
    if ASOCC_REFERENCE_YEAR_SOURCE not in active:
        _require_single_active_axis_value(
            identity=identity,
            column="reference_year",
            label="reference year",
        )
    if ASOCC_PROJECTION_SOURCE not in active:
        _require_single_active_axis_value(
            identity=identity,
            column="l2_reuse_year",
            label="L2 reuse year",
        )


def _require_single_active_axis_value(
    *,
    identity: pd.DataFrame,
    column: str,
    label: str,
) -> None:
    if column not in identity.columns:
        return
    visible = {
        str(value).strip()
        for value in identity[column].tolist()
        if not is_display_missing(value) and str(value).strip()
    }
    if len(visible) <= 1:
        return
    raise ValueError(
        f"aCC uncertainty figures cannot be generated with more than one {label} when "
        f"the corresponding uncertainty source is inactive. Observed values: {sorted(visible)}."
    )
