"""Fill-log table ownership for processed pop/gdp datasets."""

from typing import Any

import pandas as pd


FILL_LOG_COLUMNS = [
    "wb_full_name",
    "iso3_code",
    "variable",
    "year",
    "fill_method",
    "source_years",
    "n_obs",
    "filled_value",
    "reg_slope",
    "reg_intercept",
    "reg_r2",
    "reg_pvalue",
    "reg_stderr",
]


def empty_fill_log_frame() -> pd.DataFrame:
    """Return an empty fill-log table in the canonical family-local shape."""
    return pd.DataFrame(columns=FILL_LOG_COLUMNS)


def build_fill_log_row(
    *,
    info: dict[str, Any],
    year_missing: int,
    fill_method: str,
    fit: dict[str, Any],
    value: float,
) -> dict[str, Any]:
    """Return one canonical fill-log row."""
    src_years = fit["source_years"]
    src_range = f"{min(src_years)}-{max(src_years)}" if src_years else ""
    return {
        "wb_full_name": info.get("wb_full_name"),
        "iso3_code": info.get("iso3_code"),
        "variable": info.get("variable"),
        "year": int(year_missing),
        "fill_method": fill_method,
        "source_years": src_range,
        "n_obs": int(fit["nobs"]),
        "filled_value": float(value),
        "reg_slope": float(fit["slope"]),
        "reg_intercept": float(fit["intercept"]),
        "reg_r2": float(fit["r2"]),
        "reg_pvalue": float(fit["pvalue"]),
        "reg_stderr": float(fit["stderr"]),
    }
