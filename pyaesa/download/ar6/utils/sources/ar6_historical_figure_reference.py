"""Validation ownership for the AR6 historical figure reference CSV."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("Model", "Scenario", "Region", "Variable", "Unit")
REQUIRED_EDGAR_VARIABLES = (
    "Emissions|Kyoto Gases (AR6-GWP100)",
    "Emissions|Kyoto Gases (AR6-GWP100)|Lower",
    "Emissions|Kyoto Gases (AR6-GWP100)|Upper",
    "Emissions|CO2",
    "Emissions|CO2|Lower",
    "Emissions|CO2|Upper",
)


def read_ar6_historical_figure_reference(csv_file: Path) -> pd.DataFrame:
    """Read and validate the AR6 historical figure reference CSV.

    The AR6 historical figure reference file is expected to contain one EDGAR
    historical World row for each central/lower/upper series used by the GHG
    and CO2 panels, plus an optional RCMIP comparison row.

    Args:
        csv_file: Local CSV path.

    Returns:
        The parsed dataframe.

    Raises:
        RuntimeError: If the file is missing required columns or does not match
            the expected EDGAR row contract.
    """
    overlay_df = pd.read_csv(csv_file)
    missing_cols = sorted(set(REQUIRED_COLUMNS).difference(overlay_df.columns))
    if missing_cols:
        raise RuntimeError(
            "The AR6 historical figure reference CSV is missing required columns. "
            f"File={csv_file}. Missing columns={missing_cols}."
        )
    for variable in REQUIRED_EDGAR_VARIABLES:
        row_df = overlay_df.loc[
            (overlay_df["Model"] == "EDGAR")
            & (overlay_df["Scenario"] == "historical")
            & (overlay_df["Region"] == "World")
            & (overlay_df["Variable"] == variable),
            :,
        ]
        if len(row_df) != 1:
            raise RuntimeError(
                "The AR6 historical figure reference file did not contain exactly one "
                "EDGAR historical World row. "
                f"File={csv_file}. Variable='{variable}'. Observed row count={len(row_df)}."
            )
    return overlay_df
