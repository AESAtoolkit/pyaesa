"""Compute orchestration for L2 methods."""

from .l2_compute_combined import _compute_combined_methods
from .l2_compute_one_step import _compute_one_step_methods
from .l2_types import _L2RunContext


def _compute_l2_for_year(
    *,
    run: _L2RunContext,
) -> None:
    """Compute L2 results for a single year."""
    _compute_combined_methods(run)
    _compute_one_step_methods(run)
