"""International Monetary Fund Taiwan downloader for Population and GDP (PPP).

This module fetches IMF time series for Taiwan and produces a WB style
wide CSV with Population and GDP (PPP) totals. Output CSV is written to
``data_raw/pop_gdp/imf_twn_raw.csv`` and metadata to
``data_raw/logs/imf_twn_raw_meta.json``.
"""

from typing import Any, Callable, Dict, List, cast
from pathlib import Path
import requests
import pandas as pd

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.text import print_user_text_line
from pyaesa.download.pop_gdp.contracts import (
    GDP_WB_INDICATOR,
    GDP_WB_UNIT,
    PAST_YEAR_MIN,
    POP_WB_INDICATOR,
    POP_WB_UNIT,
    resolve_historical_years_from_frame,
)
from pyaesa.download.pop_gdp.raw_paths import _clear_raw_output_scope, _get_output_path
from pyaesa.download.pop_gdp.metadata import (
    _meta_covers,
    _read_meta,
    _write_meta,
)

OUTPUT_FILENAME = "imf_twn"

BASE = "https://www.imf.org/external/datamapper/api/v1"

IMF_TWN_ISO3 = "TWN"
IMF_TWN_NAME = "Taiwan"

IMF_INDICATORS = {
    "gdp_ppp_current": "PPPGDP",  # billions of international dollars
    "population": "LP",  # millions of people
    "real_gdp_growth": "NGDP_RPCH",  # percent
}

# Base year used to chain GDP PPP to a constant price level.
# Has to be equibalent to WB constant price level year.
BASE_YEAR = 2021


def _resolve_imf_historical_years_from_wb_raw() -> list[int]:
    """Return the canonical historical year horizon owned by WB raw data."""
    wb_raw_path = _get_output_path("wb")
    if not wb_raw_path.exists():
        raise RuntimeError(
            "World Bank raw dataset is missing. Run the WB download before IMF Taiwan."
        )
    wb_raw = pd.read_csv(wb_raw_path, nrows=0)
    return resolve_historical_years_from_frame(wb_raw, minimum_year=PAST_YEAR_MIN)


def _coerce_float(value: Any) -> float | None:
    """Return ``value`` as a finite float when possible."""
    numeric = cast(pd.Series, pd.to_numeric(pd.Series([value]), errors="raise"))
    candidate = numeric.iloc[0]
    if pd.isna(candidate):
        return None
    return float(candidate)


def _fetch_imf_series(
    code: str,
    iso3: str,
    years: list[int],
    *,
    base_url: str = BASE,
    http_get: Callable[..., Any] = requests.get,
) -> pd.Series:
    """Fetch a single IMF indicator series for the given ISO3 and years.

    Args:
        code (str): IMF series code to request.
        iso3 (str): ISO3 country code.
        years (list[int]): Years to request.

    Returns:
        pandas.Series: Series indexed by integer years (nullable Int64 index)
            with float values and possibly missing entries as ``pd.NA``.
    """
    url = f"{base_url}/{code}/{iso3}?periods={','.join(str(y) for y in years)}"
    resp = http_get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    data = (payload or {}).get("values", {}).get(code, {}).get(iso3, {})
    # Build a float Series from the JSON mapping. The IMF API returns period
    # keys as strings (e.g. '2020'), so convert the index to integer like
    # values and use pandas' nullable `Int64` Index. The nullable integer
    # dtype preserves missing years as `pd.NA` while allowing integer
    # operations (sorting, comparisons) required by the chaining logic.
    s = pd.Series(data, dtype="float")
    _idx = cast(pd.Series, pd.to_numeric(pd.Series(list(s.index)), errors="raise"))
    _arr = pd.array(_idx.tolist(), dtype=pd.Int64Dtype())
    s.index = pd.Index(_arr)
    s = s.sort_index()
    return s


def _chain_to_constant_base(
    df: pd.DataFrame,
    value_col: str,
    growth_col: str,
    base_year: int,
) -> pd.Series:
    """Chain a value series to a constant base year using growth rates.

    The function uses available real growth rates to forward- and
    backward chain a value series so that the series is expressed in
    the chosen ``base_year`` price level.

    Args:
        df (pandas.DataFrame): DataFrame indexed by years with columns
            containing the value and growth series.
        value_col (str): Column name containing the value to chain.
        growth_col (str): Column name containing growth rates (percent).
        base_year (int): Year to use as chaining base. Must be present in
            ``df.index``.

    Returns:
        pandas.Series: Chained series indexed like ``df`` with float
            values and ``pd.NA`` for missing entries.

    Raises:
        ValueError: If ``base_year`` is not present in ``df.index``.
    """
    if base_year not in df.index:
        raise ValueError(f"Base year {base_year} missing in IMF data (TWN)")

    chained: pd.Series = pd.Series(index=df.index, dtype="float64")

    # Coerce stored values to numeric using the helper
    base_val = pd.to_numeric(df.at[base_year, value_col], errors="raise")
    chained.at[base_year] = base_val

    index_numeric = cast(pd.Series, pd.to_numeric(pd.Series(list(df.index)), errors="raise"))
    index_ints = [int(value) for value in index_numeric.dropna().tolist()]

    # Forward chain: compute chained[y] for years > base_year using growth at y
    # Requirement: to compute chained[y] we need chained[y-1] (previous chained
    # value) and the growth rate at year y. If either is missing, chained[y]
    # remains `pd.NA`.
    max_year = max(index_ints) if index_ints else base_year
    for y in range(base_year + 1, int(max_year) + 1):
        # If the year isn't present in the fetched data index, we can't compute
        # the value so leave as missing.
        if y not in df.index:
            chained.at[y] = pd.NA
            continue

        # Coerce growth rate for this year and the previous chained value to
        # numeric types. `pd.to_numeric(..., errors='raise')` returns `NaN`
        # for non numeric inputs which we treat as missing.
        growth_y = _coerce_float(df.at[y, growth_col])
        prev_chained = _coerce_float(chained.at[y - 1])

        # If either growth or the previous chained value is missing, we cannot
        # derive the current year's chained value.
        if growth_y is None or prev_chained is None:
            chained.at[y] = pd.NA
            continue

        # Otherwise apply growth: chained[y] = chained[y-1] * (1 + growth_y/100)
        chained.at[y] = float(prev_chained) * (1.0 + float(growth_y) / 100.0)

    # Backward chain: compute chained[y] for years < base_year using growth at y+1
    # Requirement: to compute chained[y] we need chained[y+1] (next chained
    # value) and the growth rate at year y+1. If either is missing, chained[y]
    # remains `pd.NA`.
    min_year = min(index_ints) if index_ints else base_year
    for y in range(base_year - 1, int(min_year) - 1, -1):
        if y not in df.index:
            chained.at[y] = pd.NA
            continue

        # Growth for the following year (y+1) is needed to invert the growth
        # step. If the following year isn't available, treat growth as missing.
        growth_next = _coerce_float(df.at[y + 1, growth_col]) if y + 1 in df.index else None
        next_chained = _coerce_float(chained.at[y + 1])
        if growth_next is None or next_chained is None:
            chained.at[y] = pd.NA
            continue

        # Invert the growth step: chained[y] = chained[y+1] / (1 + growth_next/100)
        chained.at[y] = float(next_chained) / (1.0 + float(growth_next) / 100.0)

    return chained


def _get_imf_twn_pop_gdp(
    years: list[int],
    *,
    fetch_series: Callable[[str, str, list[int]], pd.Series] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with GDP PPP (constant ``BASE_YEAR``) and Population.

    Args:
        years (list[int]): Years to include as output columns.

    Returns:
        pandas.DataFrame: Two row DataFrame (GDP|PPP and Population)
            with columns: ``wb_full_name``, ``iso3_code``, ``variable``,
            ``unit`` and the requested year columns.
    """
    years_clean = sorted(dict.fromkeys(int(y) for y in years))
    fetch_years = sorted(dict.fromkeys(years_clean + [BASE_YEAR]))

    effective_fetch_series = _fetch_imf_series if fetch_series is None else fetch_series
    frames: Dict[str, pd.Series] = {
        name: effective_fetch_series(code, IMF_TWN_ISO3, fetch_years)
        for name, code in IMF_INDICATORS.items()
    }
    df = pd.concat(list(frames.values()), axis=1)
    df.columns = list(frames.keys())

    # Ensure integer like index (nullable Int64) for downstream chaining
    _df_idx = cast(pd.Series, pd.to_numeric(pd.Series(list(df.index)), errors="raise"))
    _df_arr = pd.array(_df_idx.tolist(), dtype=pd.Int64Dtype())
    df.index = pd.Index(_df_arr)
    df = df.sort_index()

    # IMF source notes and column naming
    # - `gdp_ppp_current`: IMF returns GDP (PPP) in *billions* of international
    #    dollars for the period. Multiply by 1e9 to obtain absolute USD values
    #    and store in `gdp_ppp_usd`.
    # - `population`: IMF returns population in *millions*; multiply by 1e6
    #    to get persons and store in `population_persons`.
    df["gdp_ppp_usd"] = df["gdp_ppp_current"] * 1_000_000_000
    df["population_persons"] = df["population"] * 1_000_000

    # Chain the current price GDP series to the chosen constant price
    # base year using the reported real GDP growth rates. The result is a
    # GDP series in `USD_Base_year` saved as `gdp_ppp_usd_const`.
    df["gdp_ppp_usd_const"] = _chain_to_constant_base(
        df,
        value_col="gdp_ppp_usd",
        growth_col="real_gdp_growth",
        base_year=BASE_YEAR,
    )

    year_cols = [str(y) for y in years_clean]

    def _build_row(variable: str, unit: str, source_col: str) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "wb_full_name": IMF_TWN_NAME,
            "iso3_code": IMF_TWN_ISO3,
            "variable": variable,
            "unit": unit,
        }
        for y in year_cols:
            y_int = int(y)
            if y_int in df.index:
                val = _coerce_float(df.at[y_int, source_col])
                row[y] = pd.NA if val is None else float(val)
            else:
                row[y] = pd.NA
        return row

    # Return GDP  (constant USD_Base_year) and Population only.
    rows: List[Dict[str, Any]] = [
        _build_row(GDP_WB_INDICATOR, GDP_WB_UNIT, "gdp_ppp_usd_const"),
        _build_row(POP_WB_INDICATOR, POP_WB_UNIT, "population_persons"),
    ]
    out_df = pd.DataFrame(
        rows, columns=["wb_full_name", "iso3_code", "variable", "unit"] + year_cols
    )
    return out_df


def _generate_imf_twn_raw(
    *,
    refresh: bool = False,
    pop_gdp_loader: Callable[[list[int]], pd.DataFrame] | None = None,
    historical_years_loader: Callable[[], list[int]] | None = None,
) -> Path:
    """Generate IMF Taiwan raw CSV and update metadata.

    Args:
        refresh (bool): If True, delete and recreate the IMF Taiwan raw CSV
            and metadata.

    Returns:
        pathlib.Path: Path to the written CSV.
    """
    out = _get_output_path(OUTPUT_FILENAME)
    out = ensure_file_parent(out)
    if refresh:
        _clear_raw_output_scope(OUTPUT_FILENAME)
    effective_historical_years_loader = (
        _resolve_imf_historical_years_from_wb_raw
        if historical_years_loader is None
        else historical_years_loader
    )
    years = effective_historical_years_loader()

    indicators = [POP_WB_INDICATOR, GDP_WB_INDICATOR]

    # If saved output already exists and the saved metadata indicates the
    # requested years/variables are covered, return the saved CSV immediately.
    meta = _read_meta(OUTPUT_FILENAME)
    if (
        out.exists()
        and (not refresh)
        and meta
        and _meta_covers(meta, years[0], years[-1], indicators)
    ):
        return out

    # Not saved (or refresh requested): announce a download/generation start
    print_user_text_line(
        "Downloading IMF Taiwan Population and GDP PPP data for years "
        f"{int(min(years))}-{int(max(years))}"
    )

    # get the IMF helper output (it provides per capita and population). We'll compute totals.
    effective_pop_gdp_loader = _get_imf_twn_pop_gdp if pop_gdp_loader is None else pop_gdp_loader
    imf_df = effective_pop_gdp_loader(years)

    # imf_df columns: wb_full_name, iso3_code, variable, unit, <years...>
    year_cols = [str(y) for y in years]

    # find GDP total and population rows returned by the helper
    total_row = imf_df[imf_df["variable"] == "GDP|PPP"]
    pop_row = imf_df[imf_df["variable"] == "Population"]

    rows = []
    # population (copy)
    if not pop_row.empty:
        rows.append(dict(pop_row.iloc[0]))

    # GDP total (copy)
    if not total_row.empty:
        rows.append(dict(total_row.iloc[0]))

    out_df = pd.DataFrame(rows)
    # ensure columns order
    cols = ["wb_full_name", "iso3_code", "variable", "unit"] + year_cols
    out_df = out_df[cols]
    out_df.to_csv(out, index=False)
    _write_meta(OUTPUT_FILENAME, years[0], years[-1], indicators)
    print_user_text_line(f"Downloaded IMF Taiwan raw CSV to: {out}")
    return out
