"""Final output-shape ownership for processed pop/gdp tables."""

from typing import Sequence, cast

import pandas as pd


def attach_mrio_codes(
    df: pd.DataFrame,
    *,
    exio_mapping: pd.DataFrame,
    oecd_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy with canonical EXIOBASE and OECD code attachments."""
    return cast(
        pd.DataFrame,
        df.merge(
            exio_mapping[["iso3_code", "exio_code"]].drop_duplicates(),
            on="iso3_code",
            how="left",
        ).merge(
            oecd_mapping[["iso3_code", "oecd_code"]].drop_duplicates(),
            on="iso3_code",
            how="left",
        ),
    )


def finalize_processed_pop_gdp_rows(
    df: pd.DataFrame,
    *,
    leading_columns: Sequence[str],
    year_cols: Sequence[str],
) -> pd.DataFrame:
    """Return processed pop/gdp rows in the canonical family-local wide shape."""
    return cast(pd.DataFrame, df[list(leading_columns) + list(year_cols)].copy())
