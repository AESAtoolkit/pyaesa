"""Shared plot ownership for AR6 figure generation."""

from pathlib import Path

import numpy as np
import pandas as pd

CATEGORY_COLORS = {
    "C1": "blue",
    "C2": "violet",
    "C3": "orange",
    "C4": "red",
}
FIGURE_MODEL_LABEL = "ALL"
WARMING_METADATA_COLUMN = "Median warming in 2100 (MAGICCv7.5.3)"
MT_TO_GT = 1e-3


def _require_row_series(data_df: pd.DataFrame, row_label: str, *, context: str) -> pd.Series:
    """Return exactly one indexed row and fail on duplicate labels."""
    row = data_df.loc[row_label]
    if isinstance(row, pd.DataFrame):
        raise RuntimeError(f"{context} requires a unique row for '{row_label}'.")
    return row


def _metadata_scalar_for_index(
    metadata_s: pd.Series,
    row_label: tuple[str, str, str],
    *,
    field_name: str,
) -> object:
    """Return one stable metadata value for one possibly duplicated sampled row label."""
    value = metadata_s.loc[row_label]
    if isinstance(value, pd.Series):
        non_na = value.dropna()
        if non_na.empty:
            return pd.NA
        unique = pd.unique(non_na)
        if len(unique) != 1:
            raise RuntimeError(
                "AR6 figure drop diagnostics found conflicting metadata values for "
                f"field '{field_name}' and row {row_label}."
            )
        return unique[0]
    return value


def numeric_year_columns(df: pd.DataFrame) -> list[int]:
    """Return sorted numeric year columns from ``df``."""
    years = []
    for col in df.columns:
        if isinstance(col, int):
            years.append(col)
            continue
        col_s = str(col)
        if col_s.isdigit():
            years.append(int(col_s))
    return sorted(set(years))


def year_slice(df: pd.DataFrame, start_year: int, end_year: int) -> list[int]:
    """Return inclusive year columns between ``start_year`` and ``end_year``."""
    return [year for year in numeric_year_columns(df) if int(start_year) <= year <= int(end_year)]


def year_slice_exclusive_end(df: pd.DataFrame, start_year: int, end_year: int) -> list[int]:
    """Return year columns from ``start_year`` up to but excluding ``end_year``."""
    return [year for year in numeric_year_columns(df) if int(start_year) <= year < int(end_year)]


def max_year(df: pd.DataFrame) -> int | None:
    """Return the maximum numeric year column from ``df``."""
    years = numeric_year_columns(df)
    if not years:
        return None
    return int(max(years))


def var_df(data_df: pd.DataFrame, var_selected: str) -> pd.DataFrame:
    """Return one variable slice from the harmonized/original table."""
    if var_selected not in data_df.index.get_level_values("variable"):
        raise RuntimeError(
            f"Required AR6 variable '{var_selected}' is missing from the figure input table."
        )
    return data_df.loc[(slice(None), slice(None), var_selected), :]


def scenario_df_from_harmonized(harmonized_data: pd.DataFrame) -> pd.DataFrame:
    """Return unique harmonized scenario rows indexed by model-scenario."""
    if harmonized_data.empty:
        raise RuntimeError(
            "AR6 figure generation requires at least one retained harmonized pathway row."
        )
    scenario_df = harmonized_data.reset_index().loc[
        :,
        ["model", "scenario", "Category", "Ssp_family"],
    ]
    scenario_df = scenario_df.drop_duplicates(subset=["model", "scenario"], keep="first")
    return scenario_df.set_index(["model", "scenario"]).sort_index()


def remaining_budget_end_year(data_df: pd.DataFrame) -> int:
    """Return the end year used for remaining budget panels."""
    data_max = max_year(data_df)
    if data_max is None:
        raise RuntimeError("AR6 figure inputs did not contain any numeric year columns.")
    return int(data_max)


def append_remaining_budget_drop_records(
    *,
    drop_records: list[dict] | None,
    data_all_cats_df: pd.DataFrame,
    var_data_df: pd.DataFrame,
    remaining_budget_end_year_value: int | None,
    figure_name: str | None,
    subset_name: str | None,
    study_period: list[int],
) -> pd.Series:
    """Return an aligned boolean mask for remaining budget rows.

    A boolean mask is used instead of a label list because the SRS/LHS
    sampling figures intentionally contain duplicate MultiIndex rows. Reusing
    a duplicate label list with ``.loc[...]`` causes a cartesian style
    expansion on duplicate labels and can allocate enormous intermediate
    arrays. Filtering with an aligned boolean mask preserves the sampled
    duplicates without creating that expansion.
    """
    if (
        remaining_budget_end_year_value is None
        or remaining_budget_end_year_value not in var_data_df.columns
    ):
        remaining_ok = pd.Series(False, index=var_data_df.index, dtype=bool)
    else:
        remaining_values = pd.Series(
            pd.to_numeric(var_data_df[remaining_budget_end_year_value], errors="raise"),
            index=var_data_df.index,
            dtype=float,
        )
        remaining_ok = pd.Series(
            np.isfinite(remaining_values.to_numpy(dtype=float)),
            index=var_data_df.index,
            dtype=bool,
        )
    if drop_records is not None:
        dropped_idx = [
            idx
            for idx, keep in zip(
                list(var_data_df.index),
                remaining_ok.to_numpy(dtype=bool).tolist(),
                strict=True,
            )
            if not keep
        ]
        cat_series = pd.Series(data_all_cats_df["Category"], copy=False)
        ssp_series = pd.Series(data_all_cats_df["Ssp_family"], copy=False)
        for mod, scen, var in dropped_idx:
            drop_records.append(
                {
                    "model": mod,
                    "scenario": scen,
                    "variable": var,
                    "category": _metadata_scalar_for_index(
                        cat_series,
                        (mod, scen, var),
                        field_name="Category",
                    ),
                    "ssp_family": _metadata_scalar_for_index(
                        ssp_series,
                        (mod, scen, var),
                        field_name="Ssp_family",
                    ),
                    "figure": figure_name,
                    "subset": subset_name,
                    "figure_component": "remaining_budget_panel",
                    "study_start_year": int(study_period[0]),
                    "study_end_year": int(study_period[1]),
                    "remaining_budget_end_year": remaining_budget_end_year_value,
                    "drop_reason": (
                        "missing_value_at_remaining_budget_end_year_"
                        f"{remaining_budget_end_year_value}"
                    ),
                }
            )
    return remaining_ok


def write_drop_csv(output_dir: Path, stem: str, drop_records: list[dict]) -> str | None:
    """Write figure only drop diagnostics and return the CSV path."""
    if not drop_records:
        return None
    drop_df = pd.DataFrame(drop_records)
    drop_df = drop_df.sort_values(
        by=["figure", "subset", "variable", "category", "ssp_family", "model", "scenario"],
        kind="stable",
    )
    csv_path = output_dir / f"{stem}-dropped_rows.csv"
    drop_df.to_csv(csv_path, index=False)
    return str(csv_path)


def plot_violin(parts, facecolor: str, alpha: float) -> None:
    """Apply a consistent style to violin plot bodies."""
    for body in parts["bodies"]:
        body.set_facecolor(facecolor)
        body.set_edgecolor("black")
        body.set_alpha(alpha)


def historical_series(
    historical_data: pd.DataFrame,
    variable: str,
    start_year: int,
    end_year_exclusive: int | None = None,
) -> pd.Series:
    """Return one historical series in Gt units for plotting."""
    if variable not in historical_data.index:
        raise RuntimeError(
            f"Required historical AR6 variable '{variable}' is missing from the processed input."
        )
    max_hist_year = max_year(historical_data)
    if max_hist_year is None:
        raise RuntimeError("Historical AR6 data does not contain any numeric year columns.")
    effective_end = (
        max_hist_year if end_year_exclusive is None else min(max_hist_year, end_year_exclusive)
    )
    years = year_slice_exclusive_end(historical_data, start_year, effective_end)
    if not years:
        raise RuntimeError(
            f"Historical AR6 data does not cover the requested plotting window for '{variable}'."
        )
    # Processed historical series are stored in Mt/yr. The figures display
    # annual pathways and cumulative budgets in Gt units.
    history_row = _require_row_series(
        historical_data,
        variable,
        context="Historical AR6 plotting",
    )
    out = (
        pd.Series(
            pd.to_numeric(history_row.loc[years], errors="raise"),
            index=pd.Index(years),
            dtype=float,
        )
        * MT_TO_GT
    )
    out.index = out.index.astype(int)
    return out
