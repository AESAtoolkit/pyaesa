"""Deterministic AR6 CC row preparation for uncertainty sampling."""

import pandas as pd


def post_study_years(rows: pd.DataFrame) -> list[int]:
    """Return integer year columns from one deterministic post study CC table."""
    return sorted(
        int(column)
        for column in rows.columns
        if isinstance(column, int) or (isinstance(column, str) and column.isdigit())
    )


def combine_study_post_rows(
    *,
    study_rows: pd.DataFrame,
    post_study_rows: pd.DataFrame | None,
    post_study_years: list[int],
) -> pd.DataFrame:
    """Return one deterministic AR6 CC table spanning study and post study years."""
    if post_study_rows is None or not post_study_years:
        return study_rows
    study = _normalize_year_columns(study_rows)
    post = _normalize_year_columns(post_study_rows)
    identity_columns = [
        "cc_model",
        "cc_scenario",
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
    ]
    return study.merge(
        post.loc[:, [*identity_columns, *post_study_years]],
        on=identity_columns,
        how="left",
        validate="one_to_one",
    )


def _normalize_year_columns(rows: pd.DataFrame) -> pd.DataFrame:
    renamed = rows.copy()
    renamed.columns = [
        int(column) if isinstance(column, str) and column.isdigit() else column
        for column in renamed.columns
    ]
    return renamed
