"""Validation helpers for deterministic wide output tables."""

from typing import cast

import pandas as pd

from ....runtime.output.contracts import IdentifierSchema


def assert_no_duplicate_columns(df: pd.DataFrame, *, where: str) -> None:
    """Fail fast when a DataFrame contains duplicate column labels."""
    if not df.columns.duplicated().any():
        return
    dup = [str(c) for c in df.columns[df.columns.duplicated()].tolist()]
    raise ValueError(f"{where}: duplicate columns are not allowed. Duplicates={dup[:20]}")


def validate_wide_frame(
    frame: pd.DataFrame,
    schema: IdentifierSchema,
    *,
    enforce_year_contract: bool = True,
) -> pd.DataFrame:
    """Validate and normalize a strict wide output frame."""
    id_cols = list(schema.columns)
    missing_ids = [col for col in id_cols if col not in frame.columns]
    if missing_ids:
        raise ValueError(
            "Wide output frame is missing required identifier columns: "
            f"{missing_ids}. Required={id_cols}, got={list(frame.columns)}"
        )

    out = frame.copy()
    rename_map: dict[object, str] = {}
    for col in out.columns:
        if isinstance(col, int):
            rename_map[col] = str(col)
        elif isinstance(col, float) and col.is_integer():
            rename_map[col] = str(int(col))
    if rename_map:
        out = out.rename(columns=rename_map)

    year_cols = [str(c) for c in out.columns if str(c) not in id_cols]
    invalid_years = [c for c in year_cols if not c.isdigit()]
    if invalid_years:
        raise ValueError(f"Non canonical year labels in wide output. Sample={invalid_years[:10]}")
    if schema.year_columns and enforce_year_contract:
        invalid_allowed = [c for c in year_cols if c not in schema.year_columns]
        if invalid_allowed:
            raise ValueError(
                "Year values are outside schema contract. "
                f"Allowed={list(schema.year_columns)}, sample_invalid={invalid_allowed[:10]}"
            )
    year_sorted = sorted(year_cols, key=int)

    dup = out.duplicated(subset=id_cols, keep=False)
    if dup.any():
        sample = out.loc[dup, id_cols].head(10)
        raise ValueError(
            "Duplicate primary key rows in wide output batch. "
            f"Identifier columns={id_cols}. Sample=\n{sample}"
        )

    return cast(pd.DataFrame, out[id_cols + year_sorted].reset_index(drop=True))


def normalize_existing_wide_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize one persisted wide table without imposing the incoming schema."""
    out = frame.copy()
    rename_map: dict[object, str] = {}
    for col in out.columns:
        if isinstance(col, int):
            rename_map[col] = str(col)
        elif isinstance(col, float) and col.is_integer():
            rename_map[col] = str(int(col))
    if rename_map:
        out = out.rename(columns=rename_map)

    identifier_cols = [str(col) for col in out.columns if not str(col).isdigit()]
    year_cols = [str(col) for col in out.columns if str(col).isdigit()]
    if not year_cols:
        raise ValueError("Persisted wide output table must contain at least one year column.")
    year_sorted = sorted(dict.fromkeys(year_cols), key=int)
    return cast(pd.DataFrame, out[identifier_cols + year_sorted].reset_index(drop=True))
