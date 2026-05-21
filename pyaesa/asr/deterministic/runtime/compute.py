"""Deterministic ASR row matching and ratio computation."""

from typing import cast

import numpy as np
import pandas as pd

from pyaesa.io_lca.data.contracts import IO_LCA_FAMILY
from pyaesa.shared.figures.contracts import SELECTOR_COLUMNS
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    EXT_LCA_SSP_SCENARIO_COLUMN,
)

from pyaesa.shared.lcia.units import try_unit_conversion

_DENOMINATOR_SCENARIO_COLUMNS = (AR6_CC_SSP_SCENARIO_COLUMN, ASOCC_SSP_SCENARIO_COLUMN)
_MATCH_HELPER_PREFIXES = ("_asr_key_",)
_MATCH_INTERNAL_COLUMNS = {
    "_acc_position",
    "_asr_eval_id",
    "_asr_lca_ssp",
    "_asr_output_id",
    "_asr_row_id",
    "_lca_position",
    "acc_impact_unit",
    "acc_value",
    "asr_value",
    "lca_converted_value",
    "lca_impact_unit",
    "lca_value",
}


def required_match_selectors(acc_df: pd.DataFrame) -> list[str]:
    """Return selector axes that define the ASR numerator denominator match."""
    return [
        column
        for column in SELECTOR_COLUMNS
        if column in acc_df.columns and _has_non_empty_values(cast(pd.Series, acc_df[column]))
    ]


def _has_non_empty_values(series: pd.Series) -> bool:
    """Return whether a series carries any non empty values."""
    return bool((~_blank_text_mask(series)).any())


def _blank_text_mask(series: pd.Series) -> pd.Series:
    """Return rows where a matching text axis is absent or blank."""
    raw = pd.Series(series, copy=False)
    text = raw.astype("string").str.strip()
    return raw.isna() | text.isna() | text.eq("")


def _normalized_match_series(series: pd.Series) -> pd.Series:
    """Return one normalized text key series for exact ASR merges."""
    raw = pd.Series(series, copy=False)
    text = raw.astype("string").str.strip()
    out = text.astype(object)
    out.loc[_blank_text_mask(raw)] = ""
    return pd.Series(out, index=series.index, dtype="object")


def _resolve_unit_factor(*, lca_impact_unit: str, acc_impact_unit: str | None) -> float:
    if str(lca_impact_unit).strip().lower() == str(acc_impact_unit).strip().lower():
        return 1.0
    factor = try_unit_conversion(str(lca_impact_unit), str(acc_impact_unit))
    if factor is None:
        raise ValueError(
            "Unsupported LCA/aCC impact_unit combination: "
            f"lca='{lca_impact_unit}', aCC='{acc_impact_unit}'."
        )
    return float(factor)


def unit_factors_for_matches(matches: pd.DataFrame) -> np.ndarray:
    """Return resolved LCA to aCC unit factors for matched long rows."""
    unit_pairs = matches.loc[:, ["lca_impact_unit", "acc_impact_unit"]].drop_duplicates()
    factors: dict[tuple[str, str | None], float] = {}
    for lca_unit, acc_unit in unit_pairs.itertuples(index=False, name=None):
        acc_text = None if pd.isna(acc_unit) else str(acc_unit)
        factors[(str(lca_unit), acc_text)] = _resolve_unit_factor(
            lca_impact_unit=str(lca_unit),
            acc_impact_unit=acc_text,
        )
    return np.array(
        [
            factors[
                (
                    str(lca_unit),
                    None if pd.isna(acc_unit) else str(acc_unit),
                )
            ]
            for lca_unit, acc_unit in matches.loc[
                :, ["lca_impact_unit", "acc_impact_unit"]
            ].itertuples(index=False, name=None)
        ],
        dtype=np.float64,
    )


def _melt_acc_values(*, acc_df: pd.DataFrame, year_cols: list[str]) -> pd.DataFrame:
    """Return one long aCC table for ASR matching without row loops."""
    acc = acc_df.reset_index(drop=True).copy()
    acc["_asr_row_id"] = np.arange(len(acc), dtype=np.int64)
    values = acc.loc[:, year_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    row_positions = np.tile(np.arange(len(acc), dtype=np.int64), len(year_cols))
    year_positions = np.repeat(np.arange(len(year_cols), dtype=np.int64), len(acc))
    metadata = acc.drop(columns=year_cols).iloc[row_positions].reset_index(drop=True)
    metadata["year"] = np.asarray(year_cols, dtype=object)[year_positions]
    metadata["acc_value"] = values.T.reshape(-1)
    metadata["acc_impact_unit"] = metadata["impact_unit"]
    return metadata


def _normalized_lca_rows(
    *,
    impact_code: str,
    lca_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return normalized LCA rows for one ASR impact branch."""
    lca = lca_rows.loc[lca_rows["impact"].astype(str).str.strip().eq(str(impact_code))].copy()
    lca["year"] = lca["year"].astype(int).astype(str)
    lca["impact"] = lca["impact"].astype(str).str.strip()
    lca["lca_value"] = pd.to_numeric(lca["lca_value"], errors="raise")
    lca["lca_impact_unit"] = lca["impact_unit"].astype(str)
    return lca


def _add_match_keys(
    *,
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Add normalized match key columns for exact tabular merges."""
    out = frame.copy()
    for column in columns:
        out[f"_asr_key_{column}"] = _normalized_match_series(cast(pd.Series, out[column]))
    return out


def _lca_merge_frame(*, lca: pd.DataFrame, match_columns: list[str]) -> pd.DataFrame:
    """Return LCA columns needed by ASR matching without duplicate identity axes."""
    owned_columns = ("_lca_position", "lca_value", "lca_impact_unit", EXT_LCA_SSP_SCENARIO_COLUMN)
    columns = [column for column in [*match_columns, *owned_columns] if column in lca.columns]
    return lca.loc[:, columns].copy()


def _denominator_scenario_key(acc_long: pd.DataFrame) -> pd.Series:
    """Return one denominator SSP key for external LCA matching."""
    for column in _DENOMINATOR_SCENARIO_COLUMNS:
        if column not in acc_long.columns or not _has_non_empty_values(
            cast(pd.Series, acc_long[column])
        ):
            continue
        return _normalized_match_series(cast(pd.Series, acc_long[column]))
    return pd.Series([""] * len(acc_long), index=acc_long.index, dtype="object")


def merge_lca_acc_rows(
    *,
    acc_long: pd.DataFrame,
    lca: pd.DataFrame,
    selectors: list[str],
    lca_type: str,
) -> pd.DataFrame:
    """Return exact long ASR matches between aCC rows and LCA rows."""
    match_columns = ["impact", "year", *selectors]
    acc_keyed = _add_match_keys(frame=acc_long, columns=match_columns)
    lca_keyed = _add_match_keys(
        frame=_lca_merge_frame(lca=lca, match_columns=match_columns),
        columns=match_columns,
    )
    lca_keyed = lca_keyed.drop(columns=[column for column in match_columns if column in lca_keyed])
    key_columns = [f"_asr_key_{column}" for column in match_columns]
    if lca_type != IO_LCA_FAMILY and EXT_LCA_SSP_SCENARIO_COLUMN in lca_keyed.columns:
        acc_keyed["_asr_lca_ssp"] = _denominator_scenario_key(acc_keyed)
        lca_keyed["_asr_lca_ssp"] = _normalized_match_series(
            cast(pd.Series, lca_keyed[EXT_LCA_SSP_SCENARIO_COLUMN])
        )
        acc_invariant_mask = _blank_text_mask(cast(pd.Series, acc_keyed["_asr_lca_ssp"]))
        invariant_mask = _blank_text_mask(cast(pd.Series, lca_keyed[EXT_LCA_SSP_SCENARIO_COLUMN]))
        invariant = lca_keyed.loc[invariant_mask].drop(columns=["_asr_lca_ssp"])
        specific = lca_keyed.loc[~invariant_mask]
        pieces = []
        if not invariant.empty:
            pieces.append(acc_keyed.merge(invariant, on=key_columns, how="inner"))
        if not specific.empty:
            invariant_acc = acc_keyed.loc[acc_invariant_mask].drop(columns=["_asr_lca_ssp"])
            specific_acc = acc_keyed.loc[~acc_invariant_mask]
            if not invariant_acc.empty:
                pieces.append(
                    invariant_acc.merge(
                        specific.drop(columns=["_asr_lca_ssp"]),
                        on=key_columns,
                        how="inner",
                    )
                )
            pieces.append(
                specific_acc.merge(
                    specific,
                    on=[*key_columns, "_asr_lca_ssp"],
                    how="inner",
                )
            )
        if not pieces:
            matched = pd.DataFrame(columns=acc_keyed.columns)
        else:
            matched = pd.concat(pieces, ignore_index=True)
    else:
        matched = acc_keyed.merge(lca_keyed, on=key_columns, how="inner")
    _validate_asr_matches(
        acc_long=acc_long,
        matched=matched,
        match_columns=match_columns,
    )
    return matched


def _validate_asr_matches(
    *,
    acc_long: pd.DataFrame,
    matched: pd.DataFrame,
    match_columns: list[str],
) -> None:
    """Fail when ASR alignment is missing or duplicates a public output row."""
    counts = matched["_asr_eval_id"].value_counts(sort=False)
    missing = sorted(set(acc_long["_asr_eval_id"].tolist()) - set(counts.index.tolist()))
    if missing:
        sample_columns = [column for column in match_columns if column in acc_long.columns]
        sample = acc_long.loc[acc_long["_asr_eval_id"].isin(missing[:5]), sample_columns]
        raise ValueError(
            "ASR could not match every aCC denominator row to exactly one LCA numerator row. "
            f"Match columns: {match_columns}. "
            f"First missing scopes: {sample.to_dict(orient='records')}."
        )
    identity_columns = _match_identity_columns(matches=matched, include_year=True)
    duplicated = matched.duplicated(subset=identity_columns, keep=False)
    if bool(duplicated.any()):
        sample_columns = [
            column
            for column in [
                *match_columns,
                AR6_CC_SSP_SCENARIO_COLUMN,
                ASOCC_SSP_SCENARIO_COLUMN,
                EXT_LCA_SSP_SCENARIO_COLUMN,
            ]
            if column in matched.columns
        ]
        sample = matched.loc[duplicated, sample_columns].head(5)
        raise ValueError(
            "ASR matched multiple LCA numerator rows to the same public ASR output row. "
            f"Match columns: {match_columns}. "
            f"First duplicated scopes: {sample.to_dict(orient='records')}."
        )


def _pivot_values(
    *,
    matches: pd.DataFrame,
    row_count: int,
    year_cols: list[str],
    value_column: str,
    output_prefix: str = "",
) -> pd.DataFrame:
    """Pivot one long value column back to deterministic wide row order."""
    index_column = "_asr_output_id" if "_asr_output_id" in matches.columns else "_asr_row_id"
    wide = matches.pivot(index=index_column, columns="year", values=value_column)
    wide = wide.reindex(index=np.arange(row_count, dtype=np.int64), columns=year_cols)
    if output_prefix:
        wide = wide.rename(columns={column: f"{output_prefix}{column}" for column in year_cols})
    return wide.reset_index(drop=True)


def _match_identity_columns(*, matches: pd.DataFrame, include_year: bool) -> list[str]:
    """Return columns that define one public ASR output row."""
    excluded = set(_MATCH_INTERNAL_COLUMNS)
    if not include_year:
        excluded.add("year")
    return [
        column
        for column in matches.columns
        if column not in excluded
        and not any(column.startswith(prefix) for prefix in _MATCH_HELPER_PREFIXES)
    ]


def _expanded_lca_scenario_matches(*, matches: pd.DataFrame) -> pd.DataFrame:
    """Repeat scenario invariant LCA years into scenario specific ASR rows."""
    if EXT_LCA_SSP_SCENARIO_COLUMN not in matches.columns:
        return matches
    scenario = _normalized_match_series(cast(pd.Series, matches[EXT_LCA_SSP_SCENARIO_COLUMN]))
    if not _has_non_empty_values(scenario):
        return matches
    base_columns = [
        column
        for column in _match_identity_columns(matches=matches, include_year=False)
        if column != EXT_LCA_SSP_SCENARIO_COLUMN
    ]
    scenario_map = (
        matches.loc[~_blank_text_mask(scenario), [*base_columns, EXT_LCA_SSP_SCENARIO_COLUMN]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    invariant = matches.loc[_blank_text_mask(scenario)]
    if invariant.empty or scenario_map.empty:
        return matches
    expanded = invariant.drop(columns=[EXT_LCA_SSP_SCENARIO_COLUMN]).merge(
        scenario_map,
        on=base_columns,
        how="inner",
    )
    retained_invariant = invariant.merge(
        scenario_map.loc[:, base_columns].drop_duplicates(),
        on=base_columns,
        how="left",
        indicator=True,
    ).loc[lambda frame: frame["_merge"].eq("left_only")]
    retained_invariant = retained_invariant.drop(columns=["_merge"])
    specific = matches.loc[~_blank_text_mask(scenario)]
    return pd.concat([specific, expanded, retained_invariant], ignore_index=True)


def _deterministic_identity_from_matches(*, matches: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic ASR identity row per wide output row."""
    identity_columns = _match_identity_columns(matches=matches, include_year=False)
    identity = matches.loc[:, identity_columns].drop_duplicates().reset_index(drop=True)
    if EXT_LCA_SSP_SCENARIO_COLUMN in identity.columns and not _has_non_empty_values(
        cast(pd.Series, identity[EXT_LCA_SSP_SCENARIO_COLUMN])
    ):
        identity = identity.drop(columns=[EXT_LCA_SSP_SCENARIO_COLUMN])
    return identity


def _matches_with_output_positions(*, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach deterministic ASR output row positions to long matches."""
    identity = _deterministic_identity_from_matches(matches=matches)
    match_key_columns = _match_identity_columns(matches=matches, include_year=False)
    key_identity = matches.loc[:, match_key_columns].drop_duplicates().reset_index(drop=True)
    key_identity["_asr_output_id"] = np.arange(len(key_identity), dtype=np.int64)
    keyed = matches.merge(key_identity, on=match_key_columns, how="inner")
    return keyed, identity


def position_deterministic_asr_matches(
    *,
    matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long ASR matches and identity rows in deterministic output order."""
    return _matches_with_output_positions(matches=_expanded_lca_scenario_matches(matches=matches))


def _compute_long_matches(
    *,
    acc_df: pd.DataFrame,
    year_cols: list[str],
    impact_code: str,
    lca_rows: pd.DataFrame,
    lca_type: str,
) -> pd.DataFrame:
    """Return long ASR matches with converted numerator and ASR values."""
    acc_long = _melt_acc_values(acc_df=acc_df, year_cols=year_cols)
    return compute_deterministic_asr_long_matches(
        acc_long=acc_long,
        selector_frame=acc_df,
        impact_code=impact_code,
        lca_rows=lca_rows,
        lca_type=lca_type,
    )


def compute_deterministic_asr_long_matches(
    *,
    acc_long: pd.DataFrame,
    selector_frame: pd.DataFrame,
    impact_code: str,
    lca_rows: pd.DataFrame,
    lca_type: str,
) -> pd.DataFrame:
    """Return long deterministic ASR matches with converted LCA numerator values.

    Args:
        acc_long: Long aCC denominator rows with normalized year, value, row
            identity, and unit columns.
        selector_frame: Source frame used to resolve selector columns.
        impact_code: Impact code represented by the denominator rows.
        lca_rows: Deterministic LCA numerator rows.
        lca_type: LCA family used for selector and scenario matching.
    Returns:
        Long matched rows with converted LCA numerator and ASR ratio values.
    """
    acc_long["_asr_eval_id"] = np.arange(len(acc_long), dtype=np.int64)
    lca = _normalized_lca_rows(impact_code=impact_code, lca_rows=lca_rows)
    selectors = required_match_selectors(selector_frame)
    matches = merge_lca_acc_rows(
        acc_long=acc_long,
        lca=lca,
        selectors=selectors,
        lca_type=lca_type,
    )
    factors = unit_factors_for_matches(matches)
    raw_lca_values = matches["lca_value"].to_numpy(dtype=np.float64)
    acc_values = matches["acc_value"].to_numpy(dtype=np.float64)
    converted_lca = raw_lca_values * factors
    matches["lca_converted_value"] = converted_lca
    matches["asr_value"] = np.divide(
        converted_lca,
        acc_values,
        out=np.full_like(converted_lca, np.nan, dtype=np.float64),
        where=acc_values != 0,
    )
    return matches


def deterministic_asr_for_acc_file(
    *,
    acc_df: pd.DataFrame,
    year_cols: list[str],
    impact_code: str,
    lca_rows: pd.DataFrame,
    lca_type: str,
) -> pd.DataFrame:
    """Return one deterministic ASR table aligned with one persisted aCC file."""
    result = build_deterministic_asr_component_frame(
        acc_df=acc_df,
        year_cols=year_cols,
        impact_code=impact_code,
        lca_rows=lca_rows,
        lca_type=lca_type,
    )
    helper_columns = [f"lca_{year}" for year in year_cols] + [f"acc_{year}" for year in year_cols]
    return result.drop(columns=helper_columns)


def build_deterministic_asr_component_frame(
    *,
    acc_df: pd.DataFrame,
    year_cols: list[str],
    impact_code: str,
    lca_rows: pd.DataFrame,
    lca_type: str,
) -> pd.DataFrame:
    """Return one deterministic ASR frame with internal yearly component columns."""
    matches = _compute_long_matches(
        acc_df=acc_df.reset_index(drop=True),
        year_cols=year_cols,
        impact_code=impact_code,
        lca_rows=lca_rows,
        lca_type=lca_type,
    )
    matches, result = position_deterministic_asr_matches(matches=matches)
    asr_wide = _pivot_values(
        matches=matches,
        row_count=len(result),
        year_cols=year_cols,
        value_column="asr_value",
    )
    lca_wide = _pivot_values(
        matches=matches,
        row_count=len(result),
        year_cols=year_cols,
        value_column="lca_converted_value",
        output_prefix="lca_",
    )
    acc_wide = _pivot_values(
        matches=matches,
        row_count=len(result),
        year_cols=year_cols,
        value_column="acc_value",
        output_prefix="acc_",
    )
    for year in year_cols:
        result[year] = asr_wide[year].to_numpy(dtype=np.float64)
        result[f"lca_{year}"] = lca_wide[f"lca_{year}"].to_numpy(dtype=np.float64)
        result[f"acc_{year}"] = acc_wide[f"acc_{year}"].to_numpy(dtype=np.float64)
    return result
