"""SSP specific processing ownership for pop/gdp.

Transforms SSP projections into the processed schema by interpolating years,
matching to MRIO regions, and aggregating parent regions when requested by
MRIO data.
"""

from typing import Optional, Sequence, cast

import pandas as pd
import pycountry

from pyaesa.download.pop_gdp.contracts import (
    GDP_SSP_INDICATOR,
    POP_SSP_INDICATOR,
)
from pyaesa.shared.tabular.wide_tables import melt_requested_year_value_rows

from pyaesa.process.pop_gdp.pipeline.finalize import (
    attach_mrio_codes,
    finalize_processed_pop_gdp_rows,
)
from pyaesa.process.pop_gdp.pipeline.parent_aggregation import apply_parent_aggregation
from pyaesa.process.pop_gdp.pipeline.tabular import (
    ensure_requested_year_columns,
    requested_year_columns,
)

SSP_ID_COLUMNS = [
    "model",
    "ssp_scenario",
    "ssp_full_name",
    "iso3_code",
    "variable",
    "unit",
]


MANUAL_ISO_OVERRIDES = {
    "Democratic Republic of the Congo": "COD",
    "Micronesia": "FSM",
    "Turkey": "TUR",
    "United States Virgin Islands": "VIR",
    "Palestine": "PSE",
}


def _interpolate_years(df: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """Interpolate missing yearly values for each SSP scenario/variable.

    Args:
        df (pandas.DataFrame): Long form SSP data with ``year``/``value`` cols.
        years (Sequence[int]): Target years that must be present.

    Returns:
        pandas.DataFrame: Long form DataFrame with interpolated values.
    """
    idx = pd.Index(sorted(dict.fromkeys(int(y) for y in years)), name="year")

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        series = cast(pd.Series, group.set_index("year")["value"].reindex(idx))
        series = series.interpolate(method="linear", limit_area="inside")
        out = series.reset_index(name="value")
        keys = group.name if isinstance(group.name, tuple) else (group.name,)
        for col, val in zip(SSP_ID_COLUMNS, keys):
            out[col] = val
        return out

    interpolated = cast(
        pd.DataFrame,
        df.groupby(SSP_ID_COLUMNS, group_keys=False)[["year", "value"]].apply(_apply),
    )
    return cast(pd.DataFrame, interpolated.reset_index(drop=True))


def _pivot_interpolated_ssp_rows(
    interpolated: pd.DataFrame,
    *,
    year_cols: Sequence[str],
) -> pd.DataFrame:
    """Return interpolated SSP rows back in the canonical wide family shape."""
    wide = interpolated.pivot_table(
        index=SSP_ID_COLUMNS,
        columns="year",
        values="value",
        aggfunc="first",  # type: ignore[arg type]
    ).reset_index()
    wide.columns = [
        str(column) if isinstance(column, (int, float)) else column for column in wide.columns
    ]
    return ensure_requested_year_columns(cast(pd.DataFrame, wide), year_cols)


def _name_to_iso3(name: str) -> Optional[str]:
    """Map SSP region names to ISO3 codes.

    Args:
        name (str): SSP full region name.

    Returns:
        Optional[str]: ISO3 code or ``None`` when no match exists.
    """
    name = (name or "").strip()
    if not name:
        return None
    if name in MANUAL_ISO_OVERRIDES:
        return MANUAL_ISO_OVERRIDES[name]
    try:
        return pycountry.countries.lookup(name).alpha_3
    except LookupError:
        return None


def _process_ssp_dataset(
    ssp_df: pd.DataFrame,
    years: Sequence[int],
    exio_mapping: pd.DataFrame,
    oecd_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """Return a processed SSP table ready for downstream use.

    Args:
        ssp_df (pandas.DataFrame): Raw SSP wide format table.
        years (Sequence[int]): Desired year horizon.
        exio_mapping (pandas.DataFrame): ISO to EXIOBASE matcher.
        oecd_mapping (pandas.DataFrame): ISO to OECD matcher.

    Returns:
        pandas.DataFrame: Processed SSP dataset containing only GDP and
        population rows.
    """
    year_cols = requested_year_columns(years)
    df = ssp_df.copy()
    if "ssp_scenario" not in df.columns:
        raise ValueError("Raw SSP data is missing required column 'ssp_scenario'.")

    # Keep only IASA SSP models to align Pop and GDP projections per SSP
    df = df[df["model"] != "OECD ENV-Growth 2023"].copy()

    # Map SSP names to ISO3 and drop unmapped rows before reshaping; this
    # avoids interpolating/aggregating entries that are not countries
    # (region groups).
    ssp_full_name = cast(pd.Series, df["ssp_full_name"])
    df["iso3_code"] = ssp_full_name.apply(_name_to_iso3)
    iso3_code = cast(pd.Series, df["iso3_code"])
    df = cast(pd.DataFrame, df[cast(pd.Series, iso3_code.notna())].copy())

    # Reshape the wide SSP file to a long form to ease interpolation.
    df = cast(
        pd.DataFrame,
        melt_requested_year_value_rows(
            df,
            requested_years=years,
            year_name="year",
            value_name="value",
        ),
    )
    df["year"] = df["year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="raise")

    # Interpolate missing years, pivot back to wide, and ensure every requested
    # year column exists before parent aggregations.
    interpolated = _interpolate_years(df, years)
    wide = _pivot_interpolated_ssp_rows(interpolated, year_cols=year_cols)

    # Canonical unit conversions: ensure GDP rows are `USD_2017/yr` and
    # Population rows are `Persons`.
    pop_mask = (wide["variable"] == POP_SSP_INDICATOR) & (wide["unit"] != "Persons")
    if pop_mask.any():
        wide.loc[pop_mask, year_cols] = (
            wide.loc[pop_mask, year_cols].apply(pd.to_numeric, errors="raise") * 1e6
        )
        wide.loc[pop_mask, "unit"] = "Persons"

    gdp_mask = (wide["variable"] == GDP_SSP_INDICATOR) & (wide["unit"] != "USD_2017/yr")
    if gdp_mask.any():
        wide.loc[gdp_mask, year_cols] = (
            wide.loc[gdp_mask, year_cols].apply(pd.to_numeric, errors="raise") * 1e9
        )
        wide.loc[gdp_mask, "unit"] = "USD_2017/yr"

    # Aggregate parents where requested and attach MRIO mappings.
    wide = apply_parent_aggregation(
        wide,
        year_cols,
        exio_mapping,
        name_column="ssp_full_name",
        group_columns=["model", "ssp_scenario", "variable", "unit"],
    )
    wide = attach_mrio_codes(
        wide,
        exio_mapping=exio_mapping,
        oecd_mapping=oecd_mapping,
    )
    return finalize_processed_pop_gdp_rows(
        wide,
        leading_columns=[
            "model",
            "ssp_scenario",
            "ssp_full_name",
            "iso3_code",
            "exio_code",
            "oecd_code",
            "variable",
            "unit",
        ],
        year_cols=year_cols,
    )
