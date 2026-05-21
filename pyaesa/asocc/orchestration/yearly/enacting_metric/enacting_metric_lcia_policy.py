"""Policy rules for enacting metric LCIA-derived metrics."""

from ....io.metadata import RunContext
from ....methods.registry.registry import REGISTRY
from .enacting_metric_common import _l1_kinds_for_selected_method


def _required_lcia_percap_kinds(*, context: RunContext) -> set[str]:
    """Return LCIA boundary kinds needed for per capita enacting metric outputs."""
    kinds: set[str] = set()
    for l1_method in context.selected_l1:
        if not REGISTRY.method_requires_lcia_percap(l1_method, level="L1"):
            continue
        kinds.update(_l1_kinds_for_selected_method(context=context, l1_method=l1_method))
    return kinds


def _required_pr_hr_cumulative_kinds(*, context: RunContext) -> set[str]:
    """Return LCIA boundary kinds needed for PR-HR cumulative per capita outputs."""
    kinds: set[str] = set()
    paired_l1_methods = [pair[1] for pair in context.combined]
    for l1_method in [*context.selected_l1, *paired_l1_methods]:
        if not REGISTRY.method_requires_pr_hr_cumulative(l1_method, level="L1"):
            continue
        kinds.update(_l1_kinds_for_selected_method(context=context, l1_method=l1_method))
    return kinds
