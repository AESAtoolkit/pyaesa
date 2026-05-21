"""Unit resolution for enacting metric output writes."""

from typing import cast

import pandas as pd

from pyaesa.download.pop_gdp.contracts import (
    GDP_SSP_INDICATOR,
    POP_SSP_INDICATOR,
)
from pyaesa.download.pop_gdp.contracts import (
    GDP_WB_INDICATOR,
    POP_WB_INDICATOR,
)

from ....io.metadata import EnactingMetricKey


def _column_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return one frame column as a Series."""
    return pd.Series(frame.loc[:, column], copy=False)


def _units_for_variable(
    *,
    df: pd.DataFrame,
    variable: str,
    ssp_scenario: str | None,
) -> set[str]:
    """Collect non null unit labels for one variable in a source table."""
    required = {"variable", "unit"}
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Processed pop/gdp table missing required unit columns: {missing}")
    rows = pd.DataFrame(df.loc[df["variable"] == variable].copy(), copy=False)
    if ssp_scenario is not None and "ssp_scenario" in df.columns:
        scoped = pd.DataFrame(rows.loc[rows["ssp_scenario"] == ssp_scenario].copy(), copy=False)
        if not scoped.empty:
            rows = scoped
    unit_series = _column_series(rows, "unit")
    return {str(v).strip() for v in unit_series.dropna().tolist() if str(v).strip()}


def _single_unit(units: set[str], *, metric: str, key: EnactingMetricKey) -> str:
    """Return a single canonical unit or fail fast on ambiguity."""
    if not units:
        raise ValueError(f"Missing unit metadata for enacting metric '{metric}' and key={key}.")
    if len(units) != 1:
        raise ValueError(
            f"Inconsistent units for enacting metric '{metric}' and key={key}: {sorted(units)}"
        )
    return next(iter(units))


def resolve_enacting_metric_unit(
    *,
    context,
    key: EnactingMetricKey,
    year_map: dict[int, pd.Series],
    mrio_default_monetary_unit: str | None,
    mrio_units: dict[str, str],
    lcia_units: dict[str, pd.Series],
    df: pd.DataFrame,
) -> str | pd.Series:
    """Resolve output unit for one enacting metric."""
    if key.lcia_method:
        unit_map = lcia_units[key.lcia_method]
        if "impact" not in df.columns:
            unit_values = sorted({str(v) for v in unit_map.dropna().astype(str)})
            if len(unit_values) == 1:
                unit = unit_values[0]
                metric_lower = key.metric.lower()
                if metric_lower.endswith("_cap") or metric_lower.endswith("_cap_cum"):
                    return f"{unit}/cap"
                return unit
            raise ValueError(
                f"LCIA enacting metric '{key.metric}' is missing 'impact' column, "
                f"cannot resolve per-impact units for method '{key.lcia_method}'."
            )
        impacts = _column_series(df, "impact").astype(str)
        unit_lookup = {
            str(impact): str(unit) for impact, unit in unit_map.dropna().astype(str).items()
        }
        units_series = impacts.map(unit_lookup)
        missing_mask = units_series.isna()
        if bool(missing_mask.any()):
            missing = (
                pd.Series(impacts.loc[missing_mask], copy=False).drop_duplicates().tolist()[:10]
            )
            source_csv = unit_map.attrs.get("source_csv")
            source_hint = f" CSV: {source_csv}" if source_csv else ""
            raise ValueError(
                "Missing LCIA unit mapping for impacts in enacting metric output. "
                f"lcia_method='{key.lcia_method}', missing impacts (sample)={missing}.{source_hint}"
            )
        metric_lower = key.metric.lower()
        if metric_lower.endswith("_cap") or metric_lower.endswith("_cap_cum"):
            units_series = pd.Series(units_series.astype(str) + "/cap", copy=False)
        return units_series

    years = [int(y) for y in year_map]
    wb_year_cols = {str(col) for col in context.wb_df.columns if str(col).isdigit()}
    use_wb = any(str(y) in wb_year_cols for y in years)
    use_ssp = any(str(y) not in wb_year_cols for y in years)

    if key.metric == "population":
        pop_units: set[str] = set()
        if use_wb:
            pop_units |= _units_for_variable(
                df=context.wb_df_raw,
                variable=POP_WB_INDICATOR,
                ssp_scenario=None,
            )
        if use_ssp:
            pop_units |= _units_for_variable(
                df=context.ssp_df_raw,
                variable=POP_SSP_INDICATOR,
                ssp_scenario=key.ssp_scenario,
            )
        return _single_unit(pop_units, metric=key.metric, key=key)

    if key.metric == "gdp_capita":
        gdp_units: set[str] = set()
        if use_wb:
            gdp_units |= _units_for_variable(
                df=context.wb_df_raw,
                variable=GDP_WB_INDICATOR,
                ssp_scenario=None,
            )
        if use_ssp:
            gdp_units |= _units_for_variable(
                df=context.ssp_df_raw,
                variable=GDP_SSP_INDICATOR,
                ssp_scenario=key.ssp_scenario,
            )
        gdp_unit = _single_unit(gdp_units, metric=key.metric, key=key)
        return f"{gdp_unit}/cap"

    metric = key.metric.lower()
    explicit_unit = mrio_units.get(metric)
    if explicit_unit is not None:
        return explicit_unit
    if metric.startswith(("fd_", "gva_", "fda_", "gvaa_", "x_")):
        return cast(str, mrio_default_monetary_unit)

    raise ValueError(
        "Unknown enacting metric unit. "
        "Metric="
        f"'{key.metric}', lcia_method='{key.lcia_method}', "
        f"ssp_scenario='{key.ssp_scenario}'."
    )
