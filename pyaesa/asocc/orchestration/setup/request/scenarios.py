"""Scenario planning for setup orchestration."""

import pandas as pd


def _assert_unique_scenarios(
    *,
    scenarios: list[str | None],
    where: str,
) -> None:
    """Fail fast when a scenario list contains duplicates."""
    unique = set(scenarios)
    if len(unique) == len(scenarios):
        return
    seen: set[str | None] = set()
    duplicates: list[str | None] = []
    for scenario in scenarios:
        if scenario in seen and scenario not in duplicates:
            duplicates.append(scenario)
        seen.add(scenario)
    raise ValueError(f"{where}: duplicate scenarios are not allowed. Duplicates={duplicates}")


def _wb_year_set(*, wb_df: pd.DataFrame) -> set[int]:
    """Return WB backed year set from processed WB table columns."""
    return {int(str(col)) for col in wb_df.columns if str(col).isdigit()}


def build_scenario_plan_by_year(
    *,
    years: list[int],
    wb_df: pd.DataFrame,
    ssp_scenarios: list[str | None],
) -> dict[int, list[str | None]]:
    """Build deterministic per year scenario execution plan.

    WB backed years are scenario agnostic and execute once with ``None``.
    SSP backed years execute for each selected SSP scenario.
    """
    _assert_unique_scenarios(
        scenarios=list(ssp_scenarios),
        where="build_scenario_plan_by_year input",
    )
    wb_years = _wb_year_set(wb_df=wb_df)
    ssp_only = [str(s) for s in ssp_scenarios if s is not None]
    plan: dict[int, list[str | None]] = {}
    for year in years:
        if int(year) in wb_years:
            plan[int(year)] = [None]
            continue
        plan[int(year)] = list(ssp_only) if ssp_only else [None]
    return plan


def scenario_state_options_from_plan(
    *,
    scenario_plan_by_year: dict[int, list[str | None]],
) -> list[str | None]:
    """Return stable scenario key list required for state dictionaries."""
    for year, scenarios in scenario_plan_by_year.items():
        _assert_unique_scenarios(
            scenarios=list(scenarios),
            where=f"scenario_plan_by_year[{int(year)}]",
        )
    values: set[str | None] = set()
    for scenarios in scenario_plan_by_year.values():
        values.update(scenarios)
    ordered_non_null = sorted(str(s) for s in values if s is not None)
    if None in values:
        return [None, *ordered_non_null]
    return [value for value in ordered_non_null]
