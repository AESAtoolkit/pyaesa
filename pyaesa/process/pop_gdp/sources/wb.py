"""World Bank specific processing ownership for pop/gdp.

Transforms World Bank and IMF (Taiwan) into the processed schema by
merging the tables, ensuring China/Taiwan splits are applied, filling missing
years via log linear regression, matching to MRIO regions, and aggregating
parent regions when requested by MRIO data.
"""

import math
from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pyaesa.download.pop_gdp.contracts import GDP_WB_INDICATOR, POP_WB_INDICATOR

from pyaesa.process.pop_gdp.pipeline.fill_log import (
    build_fill_log_row,
    empty_fill_log_frame,
)
from pyaesa.process.pop_gdp.pipeline.finalize import (
    attach_mrio_codes,
    finalize_processed_pop_gdp_rows,
)
from pyaesa.process.pop_gdp.pipeline.parent_aggregation import apply_parent_aggregation
from pyaesa.process.pop_gdp.pipeline.tabular import (
    coerce_finite_float,
    ensure_requested_year_columns,
    row_year_series,
    requested_year_columns,
    wide_year_column_positions,
    write_row_year_series,
)

# Fixed rebasing factor from USD_2021 to USD_2017.
# Built from World Bank indicator NY.GDP.DEFL.KD.ZG (US, years 2018-2021):
# P_US(2017)/P_US(2021) = 1 / prod(1 + growth_y/100) for y in 2018..2021.
_US_PRICE_LEVEL_RATIO_2017_OVER_2021 = 0.9155917545795447


def _us_price_level_ratio_2017_over_2021() -> float:
    """Return fixed P_US(2017) / P_US(2021) rebasing factor."""
    return _US_PRICE_LEVEL_RATIO_2017_OVER_2021


def _series_scalar_or_none(series: pd.Series) -> float | None:
    """Return one scalar numeric value from a year slice when present."""
    numeric = cast(pd.Series, pd.to_numeric(series, errors="raise")).dropna()
    if numeric.empty:
        return None
    return coerce_finite_float(numeric.iloc[0])


def _adjust_china_indicator_column(
    df: pd.DataFrame,
    *,
    china_mask: pd.Series,
    taiwan_mask: pd.Series,
    year_col: str,
    label: str,
) -> None:
    """Adjust one China indicator column by subtracting Taiwan in place."""
    china_value = _series_scalar_or_none(df.loc[china_mask, year_col])
    taiwan_value = _series_scalar_or_none(df.loc[taiwan_mask, year_col])
    if china_value is None or taiwan_value is None:
        df.loc[china_mask, year_col] = pd.NA
        return

    adjusted_value = china_value - taiwan_value
    if adjusted_value <= 0:
        raise ValueError(
            f"Adjusted China {label} for {year_col} must be greater than 0. "
            f"China value={china_value}, Taiwan value={taiwan_value}, "
            f"adjusted value={adjusted_value}."
        )
    df.loc[china_mask, year_col] = adjusted_value


def _fit_log_linear_series(observed_pairs: list[tuple[int, float]]) -> dict[str, Any]:
    """Return one canonical log linear fit payload for observed year/value pairs."""
    observed_years = [pair[0] for pair in observed_pairs]
    values = np.asarray([pair[1] for pair in observed_pairs], dtype=float)
    x = np.asarray(observed_years, dtype=float)
    y_log = np.log(values)

    design_matrix_years = sm.add_constant(x)
    model = sm.OLS(y_log, design_matrix_years)
    res = model.fit()
    has_variation = len(y_log) > 1 and not bool(np.allclose(y_log, y_log[0], equal_nan=True))
    return {
        "slope": float(res.params[1]) if len(res.params) > 1 else 0.0,
        "intercept": float(res.params[0]),
        "r2": float(res.rsquared) if has_variation else float("nan"),
        "pvalue": float(res.pvalues[1]) if len(res.pvalues) > 1 else float("nan"),
        "stderr": float(res.bse[1]) if len(res.bse) > 1 else float("nan"),
        "nobs": len(x),
        "source_years": observed_years,
    }


def _fill_missing_edge_years(
    *,
    series: pd.Series,
    year_ints: list[int],
    fit: dict[str, Any],
    append_log,
    id_info: dict[str, Any],
) -> None:
    """Fill leading and trailing missing years for one row in place."""
    observed_years = cast(list[int], fit["source_years"])
    first_non_missing = observed_years[0]
    last_non_missing = observed_years[-1]
    leading_missing = [yy for yy in year_ints if yy < first_non_missing]
    trailing_missing = [yy for yy in year_ints if yy > last_non_missing]

    for year_missing in leading_missing:
        log_pred = float(fit["slope"]) * float(year_missing) + float(fit["intercept"])
        pred = math.exp(log_pred)
        series.at[year_missing] = pred
        append_log(id_info, year_missing, "loglin_leading", fit, pred)

    for year_missing in trailing_missing:
        log_pred = float(fit["slope"]) * float(year_missing) + float(fit["intercept"])
        pred = math.exp(log_pred)
        series.at[year_missing] = pred
        append_log(id_info, year_missing, "loglin_trailing", fit, pred)


def _fill_wb_row_and_collect_logs(
    *,
    row: pd.Series,
    year_cols: Sequence[str],
    year_ints: list[int],
    append_log,
) -> pd.Series:
    """Return one filled WB wide-row year series and append diagnostics."""
    series = row_year_series(row, year_cols)
    id_info = {
        column: row[column] if column in row else None
        for column in ("wb_full_name", "iso3_code", "variable")
    }
    observed_pairs: list[tuple[int, float]] = []
    for yy in year_ints:
        value_float = coerce_finite_float(series.at[yy])
        if value_float is None:
            continue
        observed_pairs.append((yy, value_float))
    if not observed_pairs:
        return series

    fit_res = _fit_log_linear_series(observed_pairs)
    _fill_missing_edge_years(
        series=series,
        year_ints=year_ints,
        fit=fit_res,
        append_log=append_log,
        id_info=id_info,
    )
    return series


def _add_imf_taiwan_and_adjust_china(
    df: pd.DataFrame,
    year_cols: Sequence[str],
) -> pd.DataFrame:
    """Integrate IMF Taiwan rows and subtract them from China totals.

    The World Bank series embeds Taiwan within China; the IMF download
    provides standalone Taiwan rows which must be removed from China's totals.
    Taiwan rows are always preserved even if China is missing for the same
    year. China retains a value only when both China and Taiwan have data for
    the corresponding variable; otherwise the China entry becomes ``pd.NA``.

    Args:
        df (pandas.DataFrame): Combined World Bank and IMF data.
        year_cols (Sequence[str]): Columns that need to be adjusted.

    Returns:
        pandas.DataFrame: Copy of ``df`` with China adjusted and Taiwan kept.

    Raises:
        ValueError: If the resulting China population or GDP becomes <= 0.
    """
    df = df.copy()
    chn_gdp = (df["iso3_code"] == "CHN") & (df["variable"] == GDP_WB_INDICATOR)
    chn_pop = (df["iso3_code"] == "CHN") & (df["variable"] == POP_WB_INDICATOR)
    twn_pop = (df["iso3_code"] == "TWN") & (df["variable"] == POP_WB_INDICATOR)
    twn_gdp = (df["iso3_code"] == "TWN") & (df["variable"] == GDP_WB_INDICATOR)

    for col in year_cols:
        _adjust_china_indicator_column(
            df,
            china_mask=chn_pop,
            taiwan_mask=twn_pop,
            year_col=col,
            label="population",
        )
        _adjust_china_indicator_column(
            df,
            china_mask=chn_gdp,
            taiwan_mask=twn_gdp,
            year_col=col,
            label="GDP",
        )

    return df


def _fill_missing_edges_loglin(
    df_wide: pd.DataFrame,
    year_cols: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extrapolate edge years using log linear regression.

    Args:
        df_wide (pandas.DataFrame): Wide WB table ready for filling.
        year_cols (Sequence[str]): Ordered list of year columns.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: The filled table plus a log
        DataFrame containing the diagnostics for each extrapolated value.
    """
    if not year_cols:
        return df_wide.copy(), empty_fill_log_frame()

    year_ints = [int(y) for y in year_cols]
    filled = df_wide.copy()
    log_rows: List[Dict[str, Any]] = []

    col_positions = wide_year_column_positions(filled, year_cols)

    def _append_log(
        info: Dict[str, Any],
        year_missing: int,
        fill_method: str,
        fit: Dict[str, Any],
        value: float,
    ) -> None:
        log_rows.append(
            build_fill_log_row(
                info=info,
                year_missing=year_missing,
                fill_method=fill_method,
                fit=fit,
                value=value,
            )
        )

    for row_idx in range(len(filled)):
        row = cast(pd.Series, filled.iloc[row_idx])
        series = _fill_wb_row_and_collect_logs(
            row=row,
            year_cols=year_cols,
            year_ints=year_ints,
            append_log=_append_log,
        )
        write_row_year_series(
            filled,
            row_idx=row_idx,
            year_cols=year_cols,
            year_ints=year_ints,
            series=series,
            column_positions=col_positions,
        )

    log_df = pd.DataFrame(log_rows, columns=empty_fill_log_frame().columns)
    return filled, log_df


def _prepare_wb_processing_input(
    *,
    wb_df: pd.DataFrame,
    imf_df: pd.DataFrame,
    year_cols: Sequence[str],
) -> pd.DataFrame:
    """Return the canonical WB wide input frame before extrapolation."""
    combined = pd.concat([wb_df, imf_df], ignore_index=True, sort=False)
    combined = ensure_requested_year_columns(combined, year_cols)
    combined[year_cols] = combined[year_cols].apply(pd.to_numeric, errors="raise")

    combined = _add_imf_taiwan_and_adjust_china(combined, year_cols)
    gdp_mask = combined["variable"] == GDP_WB_INDICATOR
    if gdp_mask.any():
        factor = _us_price_level_ratio_2017_over_2021()
        combined.loc[gdp_mask, year_cols] = combined.loc[gdp_mask, year_cols] * factor
        combined.loc[gdp_mask, "unit"] = "USD_2017/yr"

    return cast(
        pd.DataFrame,
        combined[
            [
                "wb_full_name",
                "iso3_code",
                "variable",
                "unit",
            ]
            + list(year_cols)
        ].copy(),
    )


def _finalize_wb_processed_output(
    *,
    filled: pd.DataFrame,
    year_cols: Sequence[str],
    exio_mapping: pd.DataFrame,
    oecd_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Return the canonical processed WB output after fill completion."""
    filled = apply_parent_aggregation(
        filled,
        year_cols,
        exio_mapping,
        name_column="wb_full_name",
        group_columns=["variable", "unit"],
    )
    filled = attach_mrio_codes(
        filled,
        exio_mapping=exio_mapping,
        oecd_mapping=oecd_mapping,
    )
    return finalize_processed_pop_gdp_rows(
        filled,
        leading_columns=[
            "wb_full_name",
            "iso3_code",
            "exio_code",
            "oecd_code",
            "variable",
            "unit",
        ],
        year_cols=year_cols,
    )


def _process_wb_dataset(
    wb_df: pd.DataFrame,
    imf_df: pd.DataFrame,
    years: Sequence[int],
    exio_mapping: pd.DataFrame,
    oecd_mapping: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return processed WB table and the log of filled values.

    Args:
        wb_df (pandas.DataFrame): World Bank raw download.
        imf_df (pandas.DataFrame): IMF Taiwan supplement used to carve Taiwan
            out of China totals.
        years (Sequence[int]): Year span to guarantee in the output.
        exio_mapping (pandas.DataFrame): Country to EXIOBASE mapping table.
        oecd_mapping (pandas.DataFrame): Country to OECD mapping table.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame]: Processed WB table
        containing only GDP and population rows, plus a log DataFrame
        capturing regression diagnostics for filled years.
    """
    year_cols = requested_year_columns(years)
    combined = _prepare_wb_processing_input(
        wb_df=wb_df,
        imf_df=imf_df,
        year_cols=year_cols,
    )

    # Fill missing edge years and collect diagnostics.
    filled, log_df = _fill_missing_edges_loglin(combined, year_cols)

    filled = _finalize_wb_processed_output(
        filled=filled,
        year_cols=year_cols,
        exio_mapping=exio_mapping,
        oecd_mapping=oecd_mapping,
    )

    return filled, log_df
