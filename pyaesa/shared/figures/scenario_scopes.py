"""Shared scenario scope planning for figure renderers."""

import pandas as pd

from pyaesa.shared.runtime.scenario.columns import ASOCC_TIME_ROUTE_PUBLIC_COLUMN
from pyaesa.shared.runtime.scenario.scoped_rows import scenario_target_rows
from pyaesa.shared.tabular.scalars import is_display_missing

_SCENARIO_SCOPE_VALUE_COLUMNS = frozenset(
    {
        "public_row_id",
        "run_index",
        "asocc",
        "acc",
        "asr",
        "frequency_of_no_transgression",
        "cumulative_asr",
        "cumulative_no_transgression",
        "cumulative_frequency_of_no_transgression",
        "mean",
        "std",
        "min",
        "p5",
        "p25",
        "median",
        "p75",
        "p95",
        "max",
        "__values",
        "__budget_values",
        "__cumulative_values",
        "__run_indices",
        "__component_value",
        "__source_path",
        "__external_method",
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
    }
)


def repeat_invariant_rows_into_scenarios(
    frame: pd.DataFrame,
    *,
    scenario_column: str,
    scope_column: str,
    requested_scenarios: tuple[str, ...] = (),
    identity_excluded_columns: set[str] | None = None,
) -> list[pd.DataFrame]:
    """Return figure scopes with invariant rows repeated into scenario scopes.

    Args:
        frame: Figure rows for one comparison scope.
        scenario_column: Column carrying row owned scenario labels.
        scope_column: Column carrying the final figure scenario scope.
        requested_scenarios: Optional normalized scenario labels requested by the caller.

    Returns:
        One frame per final scenario scope. If the input has no scenario owned rows,
        a single invariant scope is returned.
    """
    work = frame.copy()
    if scenario_column not in work.columns:
        work[scope_column] = pd.NA
        return [work]
    scenario = pd.Series(work[scenario_column], copy=False)
    scenario_mask = ~scenario.map(is_display_missing)
    if not bool(scenario_mask.any()):
        work[scope_column] = pd.NA
        return [work]
    scenario_rows = work.loc[scenario_mask].copy()
    scopes = requested_visible_scenarios(
        visible_scenarios=visible_scenario_values(
            scenario_rows,
            scenario_column=scenario_column,
        ),
        requested_scenarios=requested_scenarios,
    )
    slices: list[pd.DataFrame] = []
    identity_columns = _scenario_scope_identity_columns(
        work,
        scenario_column=scenario_column,
        scope_column=scope_column,
        identity_excluded_columns=identity_excluded_columns,
    )
    for scope in scopes:
        scoped = scenario_target_rows(
            work,
            target={scenario_column: scope},
            scenario_columns=(scenario_column,),
            identity_columns=identity_columns,
        )
        scoped[scope_column] = scope
        slices.append(scoped)
    return slices


def preplanned_scenario_scope_slices(
    frame: pd.DataFrame,
    *,
    scenario_column: str,
    scope_column: str,
    identity_excluded_columns: set[str] | None = None,
) -> list[pd.DataFrame]:
    """Return scopes from a frame that already carries final scenario scopes."""
    if scope_column not in frame.columns:
        return [frame.copy()]
    if not visible_scenario_values(frame, scenario_column=scenario_column):
        invariant = frame.copy()
        invariant[scope_column] = pd.NA
        return [invariant]
    slices: list[pd.DataFrame] = []
    identity_columns = _scenario_scope_identity_columns(
        frame,
        scenario_column=scenario_column,
        scope_column=scope_column,
        identity_excluded_columns=identity_excluded_columns,
    )
    for value, subset in frame.groupby(scope_column, dropna=False, sort=True):
        scoped = scenario_target_rows(
            subset,
            target={scenario_column: value},
            scenario_columns=(scenario_column,),
            identity_columns=identity_columns,
        )
        scoped[scope_column] = value
        slices.append(scoped)
    return slices


def visible_scenario_values(frame: pd.DataFrame, *, scenario_column: str) -> list[str]:
    """Return sorted normalized visible scenario labels from one frame."""
    if scenario_column not in frame.columns:
        return []
    return sorted(
        {
            str(value).strip().upper()
            for value in frame[scenario_column].tolist()
            if not is_display_missing(value) and str(value).strip()
        }
    )


def requested_visible_scenarios(
    *,
    visible_scenarios: list[str],
    requested_scenarios: tuple[str, ...],
) -> list[str]:
    """Return visible scenarios filtered to an optional request."""
    if not requested_scenarios:
        return visible_scenarios
    requested = {str(value).strip().upper() for value in requested_scenarios if str(value).strip()}
    return [scenario for scenario in visible_scenarios if scenario in requested]


def _scenario_scope_identity_columns(
    frame: pd.DataFrame,
    *,
    scenario_column: str,
    scope_column: str,
    identity_excluded_columns: set[str] | None,
) -> list[str]:
    excluded = {
        scenario_column,
        scope_column,
        *_SCENARIO_SCOPE_VALUE_COLUMNS,
        *(identity_excluded_columns or set()),
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and not str(column).startswith("__figure")
    ]
