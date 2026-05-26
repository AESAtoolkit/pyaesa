"""Scenario aware public row grouping for uncertainty summaries."""

from typing import cast

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
    EXT_LCA_SSP_SCENARIO_COLUMN,
)
from pyaesa.shared.tabular.scalars import is_display_missing

SCENARIO_COLUMNS = (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    EXT_LCA_SSP_SCENARIO_COLUMN,
)


def scenario_identity_groups_from_excluded_columns(
    *,
    identity: pd.DataFrame,
    excluded_columns: set[str],
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    """Return identity groups with invariant rows repeated into scenario scopes."""
    work = identity.reset_index(drop=True)
    scenario_columns = [column for column in SCENARIO_COLUMNS if column in identity.columns]
    excluded = set(excluded_columns)
    identity_columns = [
        column
        for column in work.columns
        if column != "public_row_id" and column not in excluded and column not in scenario_columns
    ]
    grouped = (
        [((), work)]
        if not identity_columns
        else work.groupby(identity_columns, dropna=False, sort=False)
    )
    records: list[dict[str, object]] = []
    public_row_groups: list[tuple[str, ...]] = []
    ownership_identity_columns = _scenario_ownership_identity_columns(
        identity=work,
        identity_columns=identity_columns,
        excluded_columns=excluded,
        scenario_columns=scenario_columns,
    )
    scenario_texts = {
        column: _scenario_text_array(cast(pd.Series, work[column])) for column in scenario_columns
    }
    ownership_key = (
        np.zeros(len(work), dtype=np.int64)
        if not ownership_identity_columns
        else work.groupby(ownership_identity_columns, dropna=False, sort=False)
        .ngroup()
        .to_numpy(dtype=np.int64)
    )
    public_ids = work["public_row_id"].to_numpy(dtype=np.int64)
    for _key, group in grouped:
        positions = group.index.to_numpy(dtype=np.int64)
        for target in _scenario_targets(group=group, columns=scenario_columns):
            selected_positions = _scenario_target_positions(
                positions=positions,
                target=target,
                scenario_columns=scenario_columns,
                scenario_texts=scenario_texts,
                ownership_key=ownership_key,
            )
            row = pd.Series(work.iloc[int(selected_positions[0])], copy=False)
            payload: dict[str, object] = {column: row[column] for column in identity_columns}
            payload.update(target)
            records.append(payload)
            public_row_groups.append(
                tuple(str(int(public_ids[position])) for position in selected_positions)
            )
    grouped_identity = pd.DataFrame.from_records(records)
    grouped_identity.insert(0, "public_row_id", range(len(grouped_identity)))
    return grouped_identity.reset_index(drop=True), tuple(public_row_groups)


def _scenario_ownership_identity_columns(
    *,
    identity: pd.DataFrame,
    identity_columns: list[str],
    excluded_columns: set[str],
    scenario_columns: list[str],
) -> list[str]:
    columns = list(identity_columns)
    for column in excluded_columns:
        if (
            column in identity.columns
            and column not in columns
            and column not in scenario_columns
            and column != ASOCC_TIME_ROUTE_PUBLIC_COLUMN
        ):
            columns.append(column)
    return columns


def _scenario_text_array(series: pd.Series) -> np.ndarray:
    """Return normalized scenario labels with display missing values as empty text."""
    text = series.astype("string").str.strip().str.upper()
    missing = series.map(is_display_missing) | text.isna() | text.eq("")
    values = text.astype(object)
    values.loc[missing] = ""
    return values.to_numpy()


def _scenario_target_positions(
    *,
    positions: np.ndarray,
    target: dict[str, object],
    scenario_columns: list[str],
    scenario_texts: dict[str, np.ndarray],
    ownership_key: np.ndarray,
) -> np.ndarray:
    """Select rows owned by one final scenario scope for each identity."""
    matched = np.ones(len(positions), dtype=bool)
    specificity = np.zeros(len(positions), dtype=np.int64)
    for column in scenario_columns:
        text = scenario_texts[column][positions]
        target_value = target.get(column, pd.NA)
        target_text = "" if is_display_missing(target_value) else str(target_value).strip().upper()
        if not target_text:
            matched &= text == ""
            continue
        exact = text == target_text
        matched &= (text == "") | exact
        specificity += exact.astype(np.int64)
    selected_positions = positions[matched]
    selected_specificity = specificity[matched]
    selected_keys = ownership_key[selected_positions]
    max_specificity = (
        pd.Series(selected_specificity)
        .groupby(selected_keys, sort=False)
        .transform("max")
        .to_numpy()
    )
    return selected_positions[selected_specificity == max_specificity]


def _scenario_targets(*, group: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    values_by_column = {
        column: _visible_values(cast(pd.Series, group[column])) for column in columns
    }
    tokens = _final_scenario_tokens(values_by_column)
    if not tokens:
        return [{column: pd.NA for column in columns}]
    return [
        {column: token if token in values_by_column[column] else pd.NA for column in columns}
        for token in tokens
    ]


def _final_scenario_tokens(values_by_column: dict[str, list[str]]) -> list[str]:
    for column in SCENARIO_COLUMNS:
        values = values_by_column.get(column, [])
        if values:
            return values
    return []


def _visible_values(series: pd.Series) -> list[str]:
    values = [
        str(value).strip().upper()
        for value in series.tolist()
        if not is_display_missing(value) and str(value).strip()
    ]
    return list(dict.fromkeys(values))
