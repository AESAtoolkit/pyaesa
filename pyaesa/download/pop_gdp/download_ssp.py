"""SSP downloader for Population and GDP (PPP).

This module fetches SSP scenario data directly from the IIASA Scenario
Explorer legacy API and writes a wide CSV to
``data_raw/pop_gdp/ssp_raw.csv``.

Why no ``pyam`` dependency:
    The rest of this package requires ``pymrio``. Recent ``pyam`` versions
    conflict with ``pymrio`` through incompatible ``openpyxl`` constraints,
    which blocks modern Python environments. Using direct HTTP queries keeps
    SSP download behavior while avoiding that hard dependency conflict.
"""

import json
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import pandas as pd
import requests

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.text import print_user_text_line
from pyaesa.download.pop_gdp.contracts import (
    DEFAULT_SSP_INDICATORS,
    FUTURE_YEARS,
)
from pyaesa.download.pop_gdp.metadata import (
    _meta_covers,
    _read_meta,
    _write_meta,
)
from pyaesa.download.pop_gdp.raw_paths import _clear_raw_output_scope, _get_output_path

_AUTH_URL = "https://api.manager.ece.iiasa.ac.at"
_SSP_ENV_ALIAS = "ssp"
_OUTPUT_FILENAME = "ssp"
_HTTP_TIMEOUT = 60
_HTTP_TIMEOUT_BULK = 300


def _ensure_year_columns(df: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """Ensure every year in ``years`` exists as a column on ``df``."""
    for y in years:
        cy = str(y)
        if cy not in df.columns:
            df[cy] = pd.NA
    return df


def _http_json_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    http_get: Callable[..., Any] = requests.get,
) -> list | dict:
    """GET JSON payload and raise clear runtime errors on API failures."""
    response = http_get(url, headers=headers, timeout=_HTTP_TIMEOUT)
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"IIASA API GET failed: {url}") from exc
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"IIASA API returned invalid JSON: {url}") from exc


def _get_anonymous_auth_header(
    *,
    auth_url: str = _AUTH_URL,
    http_get: Callable[..., Any] = requests.get,
) -> dict[str, str]:
    """Return bearer auth header for anonymous IIASA API access."""
    payload = _http_json_get(
        f"{auth_url}/legacy/anonym/",
        http_get=http_get,
    )
    if not isinstance(payload, str) or not payload.strip():
        raise RuntimeError("IIASA anonymous auth returned an invalid token payload.")
    return {"Authorization": f"Bearer {payload}"}


def _resolve_ssp_base_url(
    auth_header: dict[str, str],
    *,
    auth_url: str = _AUTH_URL,
    http_get: Callable[..., Any] = requests.get,
) -> str:
    """Resolve the base URL of the SSP Scenario Explorer API instance."""
    apps_payload = _http_json_get(
        f"{auth_url}/legacy/applications",
        headers=auth_header,
        http_get=http_get,
    )
    if not isinstance(apps_payload, list):
        raise RuntimeError("IIASA applications endpoint returned an invalid payload.")

    app_name: str | None = None
    for app in apps_payload:
        if not isinstance(app, dict):
            continue
        config = app.get("config")
        if not isinstance(config, list):
            continue
        env = next(
            (
                item.get("value")
                for item in config
                if isinstance(item, dict) and item.get("path") == "env"
            ),
            None,
        )
        if env == _SSP_ENV_ALIAS:
            candidate = app.get("name")
            if isinstance(candidate, str) and candidate.strip():
                app_name = candidate
                break

    if app_name is None:
        raise RuntimeError("Unable to resolve IIASA SSP application entry (env='ssp').")

    config_payload = _http_json_get(
        f"{auth_url}/legacy/applications/{app_name}/config",
        headers=auth_header,
        http_get=http_get,
    )
    if not isinstance(config_payload, list):
        raise RuntimeError("IIASA SSP config endpoint returned an invalid payload.")

    base_url = next(
        (
            item.get("value")
            for item in config_payload
            if isinstance(item, dict) and item.get("path") == "baseUrl"
        ),
        None,
    )
    if not isinstance(base_url, str) or not base_url.strip():
        raise RuntimeError("Unable to resolve SSP baseUrl from IIASA config payload.")
    return base_url.rstrip("/")


def _fetch_default_run_ids(
    *,
    base_url: str,
    auth_header: dict[str, str],
    http_get: Callable[..., Any] = requests.get,
    read_json: Callable[..., pd.DataFrame] = pd.read_json,
) -> list[int]:
    """Return run IDs of default SSP runs."""
    url = f"{base_url}/runs?getOnlyDefaultRuns=true&includeMetadata=false"
    response = http_get(url, headers=auth_header, timeout=_HTTP_TIMEOUT)
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"IIASA SSP runs query failed: {url}") from exc

    runs = read_json(StringIO(response.text), orient="records")
    if runs.empty or "run_id" not in runs.columns:
        raise RuntimeError("IIASA SSP runs query returned no usable default runs.")

    run_id_series = cast(pd.Series, runs["run_id"])
    run_ids = cast(pd.Series, pd.to_numeric(run_id_series, errors="coerce")).dropna().astype(int)
    unique = sorted(set(run_ids.tolist()))
    if not unique:
        raise RuntimeError("IIASA SSP runs query returned no valid run IDs.")
    return unique


def _fetch_ssp_timeseries(
    *,
    variables: Sequence[str],
    auth_url: str = _AUTH_URL,
    http_get: Callable[..., Any] = requests.get,
    http_post: Callable[..., Any] = requests.post,
    read_json: Callable[..., pd.DataFrame] = pd.read_json,
) -> pd.DataFrame:
    """Query SSP timeseries from IIASA API and return long form rows."""
    auth_header = _get_anonymous_auth_header(auth_url=auth_url, http_get=http_get)
    base_url = _resolve_ssp_base_url(
        auth_header,
        auth_url=auth_url,
        http_get=http_get,
    )
    run_ids = _fetch_default_run_ids(
        base_url=base_url,
        auth_header=auth_header,
        http_get=http_get,
        read_json=read_json,
    )

    payload = {
        "filters": {
            "regions": [],
            "variables": list(variables),
            "runs": run_ids,
            "years": [],
            "units": [],
            "timeslices": [],
        }
    }
    headers = {**auth_header, "Content-Type": "application/json"}
    url = f"{base_url}/runs/bulk/ts"
    response = http_post(
        url,
        headers=headers,
        data=json.dumps(payload),
        timeout=_HTTP_TIMEOUT_BULK,
    )
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"IIASA SSP bulk timeseries query failed: {url}") from exc

    data = read_json(StringIO(response.text), orient="records")
    required = {"model", "scenario", "region", "variable", "unit", "year", "value"}
    missing = sorted(col for col in required if col not in data.columns)
    if missing:
        raise RuntimeError(f"IIASA SSP bulk payload missing required columns: {missing}")
    if data.empty:
        raise RuntimeError("IIASA SSP bulk timeseries query returned no rows.")

    if "subannual" in data.columns:
        year_like = data["subannual"].isna() | data["subannual"].isin(["Year", -1])
        data = data[year_like].copy()

    data["year"] = pd.to_numeric(data["year"], errors="raise")
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    year_series = cast(pd.Series, data["year"])
    value_series = cast(pd.Series, data["value"])
    valid_rows = cast(pd.Series, year_series.notna() & value_series.notna())
    data = cast(pd.DataFrame, data.loc[valid_rows].copy())
    data["year"] = data["year"].astype(int)

    present_vars = set(data["variable"].astype(str).unique().tolist())
    missing_vars = sorted(set(variables) - present_vars)
    if missing_vars:
        raise RuntimeError(f"IIASA SSP data query is missing requested variable(s): {missing_vars}")

    data = cast(pd.DataFrame, data.rename(columns={"scenario": "ssp_scenario"}))
    return cast(
        pd.DataFrame,
        data[["model", "ssp_scenario", "region", "variable", "unit", "year", "value"]].copy(),
    )


def _generate_ssp_raw(
    *,
    refresh: bool = False,
    timeseries_loader: Callable[[Sequence[str]], pd.DataFrame] | None = None,
    fetch_timeseries_func: Callable[..., pd.DataFrame] = _fetch_ssp_timeseries,
) -> Path:
    """Generate SSP wide table and write CSV + metadata."""
    years = list(FUTURE_YEARS)
    indicators = list(DEFAULT_SSP_INDICATORS)
    out = _get_output_path(_OUTPUT_FILENAME)
    out = ensure_file_parent(out)
    if refresh:
        _clear_raw_output_scope(_OUTPUT_FILENAME)

    meta = _read_meta(_OUTPUT_FILENAME)
    if (
        out.exists()
        and (not refresh)
        and meta
        and _meta_covers(meta, int(min(years)), int(max(years)), indicators)
    ):
        return out

    print_user_text_line(
        f"Downloading SSP Population and GDP PPP data for years {int(min(years))}-{int(max(years))}"
    )

    if timeseries_loader is None:
        df = fetch_timeseries_func(variables=indicators)
    else:
        df = timeseries_loader(indicators)
    df = df[df["year"].isin(FUTURE_YEARS)].copy()

    wide = df.pivot_table(
        index=["model", "ssp_scenario", "region", "variable", "unit"],
        columns="year",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide["ssp_full_name"] = wide["region"]
    wide = wide.drop(columns=["region"])
    wide.columns = [str(c) if isinstance(c, (int, float)) else c for c in wide.columns]

    _ensure_year_columns(wide, years)
    year_cols = [str(y) for y in years]
    base_cols = ["model", "ssp_scenario", "ssp_full_name", "variable", "unit"]
    wide = cast(pd.DataFrame, wide[base_cols + year_cols].copy())
    wide = cast(pd.DataFrame, wide.sort_values(base_cols).reset_index(drop=True))

    wide.to_csv(out, index=False)
    _write_meta(_OUTPUT_FILENAME, int(min(years)), int(max(years)), indicators)
    print_user_text_line(f"Downloaded SSP raw CSV to: {out}")
    return out
