"""LCIA per-capita enacting metric recording."""

import pandas as pd

from ....io.metadata import RunContext, RunState
from .enacting_metric_lcia_passes import (
    _record_direct_lcia_percap_pass,
    _record_pr_hr_cumulative_pass,
)
from .enacting_metric_lcia_policy import (
    _required_lcia_percap_kinds,
    _required_pr_hr_cumulative_kinds,
)
from .enacting_metric_lcia_selection import _resolve_enacting_metric_output_scenario
from .enacting_metric_lcia_routing import (
    _iter_lcia_percap_metric_pairs,
)


def record_lcia_percap_enacting_metrics(
    *,
    context: RunContext,
    state: RunState,
    year: int,
    ssp_scenario: str | None,
    lcia_by_method: dict[str, dict] | None,
    pop_series: pd.Series,
    use_original_domain: bool = False,
    lcia_effective_year_by_method: dict[str, int] | None = None,
) -> None:
    """Record LCIA per capita level-1 enacting metrics."""
    scenario_key = _resolve_enacting_metric_output_scenario(
        context=context,
        year=int(year),
        ssp_scenario=ssp_scenario,
    )

    percap_kinds = _required_lcia_percap_kinds(context=context)
    if percap_kinds and lcia_by_method:
        pairs = _iter_lcia_percap_metric_pairs(required_kinds=percap_kinds)
        _record_direct_lcia_percap_pass(
            context=context,
            state=state,
            year=int(year),
            scenario_key=scenario_key,
            lcia_by_method=lcia_by_method,
            pop_series=pop_series,
            pairs=pairs,
            use_original_domain=bool(use_original_domain),
            lcia_effective_year_by_method=lcia_effective_year_by_method,
        )

    cumulative_kinds = _required_pr_hr_cumulative_kinds(context=context)
    if not cumulative_kinds:
        return
    _record_pr_hr_cumulative_pass(
        context=context,
        state=state,
        year=int(year),
        ssp_scenario=ssp_scenario,
        scenario_key=scenario_key,
        lcia_by_method=lcia_by_method,
        cumulative_kinds=cumulative_kinds,
        use_original_domain=bool(use_original_domain),
        lcia_effective_year_by_method=lcia_effective_year_by_method,
    )
