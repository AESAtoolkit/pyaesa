"""Static carrying capacity table loading and validation."""

from pathlib import Path

import pandas as pd


def read_static_cc(cc_csv_path: Path) -> pd.DataFrame:
    """Read one static carrying capacity prerequisite CSV."""
    if not cc_csv_path.exists():
        raise FileNotFoundError(
            f"Static CC CSV not found at {cc_csv_path}. "
            "Check that the static carrying capacity prerequisite exists for the "
            "requested LCIA method."
        )
    df = pd.read_csv(cc_csv_path)
    required = {"impact", "impact_unit", "min_cc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Static CC CSV {cc_csv_path} is missing columns: {sorted(missing)}.")
    min_numeric = pd.Series(pd.to_numeric(df["min_cc"], errors="coerce"), copy=False)
    invalid_min_rows = df.loc[pd.Series(min_numeric.isna(), copy=False), "impact"].astype(str)
    if not invalid_min_rows.empty:
        raise ValueError(
            f"Static CC CSV {cc_csv_path} requires numeric min_cc values for every impact. "
            f"Observed invalid impacts: {sorted(invalid_min_rows.tolist())}."
        )
    if "max_cc" in df.columns:
        max_numeric = pd.Series(pd.to_numeric(df["max_cc"], errors="coerce"), copy=False)
        invalid_max_rows = df.loc[
            pd.Series(df["max_cc"].notna() & max_numeric.isna(), copy=False),
            "impact",
        ].astype(str)
        if not invalid_max_rows.empty:
            raise ValueError(
                f"Static CC CSV {cc_csv_path} requires numeric max_cc values "
                "when max_cc is provided. "
                f"Observed invalid impacts: {sorted(invalid_max_rows.tolist())}."
            )
    return df


def require_static_cc_bounds_available(
    *,
    cc_df: pd.DataFrame,
    requested_bounds: list[str],
    context: str,
) -> None:
    """Validate that the requested static CC bounds are present in one CSV table."""
    if "max_cc" not in requested_bounds:
        return
    if "max_cc" not in cc_df.columns:
        raise ValueError(
            f"{context} requires a 'max_cc' column in the static carrying capacity CSV."
        )
    max_numeric = pd.Series(pd.to_numeric(cc_df["max_cc"], errors="coerce"), copy=False)
    missing_max = cc_df.loc[pd.Series(max_numeric.isna(), copy=False), "impact"].astype(str)
    if not missing_max.empty:
        raise ValueError(
            f"{context} requires numeric max_cc values for every impact. "
            f"Observed missing or invalid impacts: {sorted(missing_max.tolist())}."
        )
