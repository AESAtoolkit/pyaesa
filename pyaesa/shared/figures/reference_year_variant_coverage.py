"""Reference year eligibility for deterministic figure variant compression."""

import pandas as pd

from pyaesa.shared.tabular.scalars import display_scalar, is_display_missing


def eligible_full_window_reference_tokens(
    frame: pd.DataFrame,
    *,
    required_years: list[int],
    group_columns: list[str],
    reference_column: str = "reference_year",
    year_column: str = "year",
) -> set[str] | None:
    """Return reference year tokens that cover every required year in each group."""
    if reference_column not in frame.columns or year_column not in frame.columns:
        return None
    required = {int(year) for year in required_years}
    grouped = (
        frame.groupby(group_columns, dropna=False, sort=False) if group_columns else [(None, frame)]
    )
    valid_by_group: list[set[str]] = []
    for _key, group in grouped:
        ref_mask = group[reference_column].map(lambda value: not is_display_missing(value))
        if not bool(ref_mask.any()):
            continue
        valid = _valid_reference_tokens_for_group(
            group.loc[ref_mask],
            required_years=required,
            reference_column=reference_column,
            year_column=year_column,
        )
        if not valid:
            _raise_no_full_window_reference(required_years=required)
        valid_by_group.append(valid)
    eligible = set.intersection(*valid_by_group)
    if not eligible:
        _raise_no_full_window_reference(required_years=required)
    return eligible


def filter_full_window_reference_years(
    frame: pd.DataFrame,
    *,
    eligible_tokens: set[str] | None,
    reference_column: str = "reference_year",
) -> pd.DataFrame:
    """Keep missing reference rows and reference years eligible for the full window."""
    if eligible_tokens is None or reference_column not in frame.columns:
        return frame.copy()
    ref_tokens = frame[reference_column].map(_reference_token)
    return frame.loc[ref_tokens.isna() | ref_tokens.isin(list(eligible_tokens))].copy()


def _valid_reference_tokens_for_group(
    frame: pd.DataFrame,
    *,
    required_years: set[int],
    reference_column: str,
    year_column: str,
) -> set[str]:
    years = pd.Series(pd.to_numeric(frame[year_column], errors="raise"), copy=False).astype(int)
    work = frame.loc[:, [reference_column]].copy()
    work[year_column] = years
    work["__reference_token"] = work[reference_column].map(_reference_token)
    valid: set[str] = set()
    for token, token_rows in work.groupby("__reference_token", dropna=True, sort=False):
        if required_years.issubset(set(token_rows[year_column].astype(int).tolist())):
            valid.add(str(token))
    return valid


def _reference_token(value: object) -> str | None:
    if is_display_missing(value):
        return None
    return display_scalar(value)


def _raise_no_full_window_reference(*, required_years: set[int]) -> None:
    years = sorted(int(year) for year in required_years)
    raise ValueError(
        "Deterministic figure variant compression cannot select a reference_year because "
        "no reference year applies to the full study window. Reference years retained by "
        f"variant compression must have rows for every plotted year: {years}."
    )
