"""aSoCC Sobol output scope selection."""

from dataclasses import replace

import pandas as pd

from pyaesa.asocc.uncertainty.sources.inter_mrio import InterMrioPlan
from pyaesa.asocc.uncertainty.sources.projection import build_projection_plan
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan


def selected_sobol_years(
    *,
    plan: SobolPlan,
    requested_years: tuple[int, ...],
) -> tuple[int, ...]:
    """Return selected aSoCC output years evaluated by Sobol analysis."""
    studied = tuple(sorted({int(year) for year in requested_years}))
    if plan.sobol_years is not None:
        requested = set(plan.sobol_years)
        missing = sorted(requested - set(studied))
        if missing:
            raise ValueError(
                "sobol_parameters.sobol_years must be selected from the studied aSoCC years. "
                f"Unsupported year(s): {missing}."
            )
        return tuple(year for year in studied if year in requested)
    return tuple(dict.fromkeys((studied[0], studied[-1])))


def loaded_for_sobol_years(*, loaded, selected_years: tuple[int, ...]):
    """Return loaded aSoCC rows restricted to selected Sobol years."""
    year = pd.Series(
        pd.to_numeric(loaded.rows.loc[:, "year"], errors="raise"), index=loaded.rows.index
    )
    rows = loaded.rows.loc[year.isin(selected_years)].reset_index(drop=True)
    return replace(
        loaded,
        base_asocc_args={**loaded.base_asocc_args, "years": list(selected_years)},
        requested_years=list(selected_years),
        rows=rows,
    )


def inter_mrio_plan_for_sobol_years(
    *,
    plan: InterMrioPlan,
    selected_years: tuple[int, ...],
    projection_active: bool,
) -> InterMrioPlan:
    """Return an inter-MRIO plan restricted to selected Sobol years."""
    alternate_loaded = loaded_for_sobol_years(
        loaded=plan.alternate_loaded,
        selected_years=selected_years,
    )
    return replace(
        plan,
        alternate_loaded=alternate_loaded,
        alternate_projection_plan=build_projection_plan(loaded=alternate_loaded)
        if projection_active
        else None,
    )
