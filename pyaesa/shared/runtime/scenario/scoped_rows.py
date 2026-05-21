"""Scenario scoped row ownership helpers."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from pyaesa.shared.tabular.scalars import is_display_missing


def scenario_target_rows(
    frame: pd.DataFrame,
    *,
    target: Mapping[str, object],
    scenario_columns: Sequence[str],
    identity_columns: Sequence[str],
) -> pd.DataFrame:
    """Select rows visible for one scenario target.

    The selector compares only scenario columns that exist in ``frame``. A row
    is eligible when every available scenario column is either missing
    (scenario invariant) or equals the requested target value. The selector then
    resolves overlap inside each caller supplied identity group by counting how
    many available scenario columns match the target. Rows with the highest
    count are retained for that identity, so target specific rows replace
    invariant rows for the same identity, while identities that have no target
    specific row keep their invariant route rows. When ``identity_columns`` is
    empty, all eligible rows are resolved as one identity group.
    """
    available_scenarios = [column for column in scenario_columns if column in frame.columns]
    if not available_scenarios:
        return frame.copy()
    mask = _target_match_mask(frame=frame, target=target, scenario_columns=available_scenarios)
    matched = frame.loc[mask].copy()
    if matched.empty:
        return matched
    specificity_column = "__scenario_specificity"
    matched[specificity_column] = _target_specificity(
        frame=matched,
        target=target,
        scenario_columns=available_scenarios,
    )
    identity_group_column = "__scenario_identity_group"
    matched[identity_group_column] = 0
    columns = [column for column in identity_columns if column in matched.columns]
    max_by_identity = matched.groupby(columns + [identity_group_column], dropna=False, sort=False)[
        specificity_column
    ].transform("max")
    selected = matched.loc[matched[specificity_column].eq(max_by_identity)].copy()
    return selected.drop(columns=[specificity_column, identity_group_column]).reset_index(drop=True)


def _target_match_mask(
    *,
    frame: pd.DataFrame,
    target: Mapping[str, object],
    scenario_columns: Sequence[str],
) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in scenario_columns:
        value = target.get(column, pd.NA)
        column_missing = frame[column].map(is_display_missing)
        target_text = _scenario_text(value)
        if not target_text:
            mask &= column_missing
        else:
            text = frame[column].astype("string").str.strip().str.upper()
            mask &= column_missing | text.eq(target_text).fillna(False)
    return mask


def _target_specificity(
    *,
    frame: pd.DataFrame,
    target: Mapping[str, object],
    scenario_columns: Sequence[str],
) -> np.ndarray:
    score = np.zeros(len(frame), dtype=np.int64)
    for column in scenario_columns:
        target_text = _scenario_text(target.get(column, pd.NA))
        if not target_text:
            continue
        text = frame[column].astype("string").str.strip().str.upper()
        score += text.eq(target_text).fillna(False).to_numpy(dtype=np.int64)
    return score


def _scenario_text(value: object) -> str:
    if is_display_missing(value):
        return ""
    return str(value).strip().upper()
