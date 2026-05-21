"""World Bank downloader for Population and GDP (PPP).

This module downloads the World Bank World Development Indicators bulk ZIP
published by the World Bank Data Catalog service and extracts the two
historical indicators required by the population GDP processing pipeline.
It writes a wide CSV under ``data_raw/pop_gdp/wb_raw.csv`` and metadata
under ``data_raw/logs/wb_raw_meta.json``.
"""

import json
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence, TypeVar, cast

import pandas as pd
import requests

from pyaesa.shared.runtime.text import print_user_text_line
from pyaesa.download.pop_gdp.contracts import (
    GDP_WB_INDICATOR,
    GDP_WB_UNIT,
    PAST_YEAR_MIN,
    POP_WB_INDICATOR,
    POP_WB_UNIT,
    resolve_historical_years_from_frame,
)
from pyaesa.download.pop_gdp.metadata import (
    _meta_covers,
    _read_meta,
    _write_meta,
)
from pyaesa.download.pop_gdp.raw_paths import _clear_raw_output_scope, _get_output_path
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

POP_WB_CODE = "SP.POP.TOTL"
GDP_WB_CODE = "NY.GDP.MKTP.PP.KD"
DEFAULT_WB_CODES: Sequence[str] = (POP_WB_CODE, GDP_WB_CODE)
OUTPUT_FILENAME = "wb"

CODES_TO_INDICATORS = {
    GDP_WB_CODE: GDP_WB_INDICATOR,
    POP_WB_CODE: POP_WB_INDICATOR,
}

INDICATORS_TO_UNITS = {
    GDP_WB_INDICATOR: GDP_WB_UNIT,
    POP_WB_INDICATOR: POP_WB_UNIT,
}

_WDI_BULK_URL = "https://ddh-openapi.worldbank.org/resources/DR0095335"
_WDI_MAIN_MEMBER = "WDICSV.csv"
_WDI_COUNTRY_MEMBER = "WDICountry.csv"
_WB_MAX_ATTEMPTS = 3
_WB_RETRY_DELAY_SECONDS = 2.0
_WB_DOWNLOAD_TIMEOUT_SECONDS = 300
_WB_DOWNLOAD_CHUNK_SIZE = 1024 * 1024

T = TypeVar("T")


class _StreamingResponseLike(Protocol):
    def __enter__(self) -> "_StreamingResponseLike": ...

    def __exit__(self, *args: object) -> None: ...

    def raise_for_status(self) -> None: ...

    def iter_content(self, chunk_size: int) -> Iterable[bytes]: ...


def _run_wb_request_with_retry(
    request_loader: Callable[[], T],
    *,
    operation: str,
    max_attempts: int = _WB_MAX_ATTEMPTS,
    retry_delay_seconds: float = _WB_RETRY_DELAY_SECONDS,
) -> T:
    """Run one World Bank source operation with bounded retries.

    Args:
        request_loader (Callable[[], T]): Loader executing one download or
            parsing operation.
        operation (str): Human readable description of the operation.
        max_attempts (int): Maximum number of attempts before failing.
        retry_delay_seconds (float): Delay between retry attempts.

    Returns:
        T: Value returned by ``request_loader``.

    Raises:
        RuntimeError: If the operation still fails after retry exhaustion.
    """
    last_error: Exception | None = None
    retriable_errors = (
        requests.RequestException,
        OSError,
        zipfile.BadZipFile,
    )
    for attempt in range(1, max_attempts + 1):
        try:
            return request_loader()
        except retriable_errors as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            print_user_text_line(
                "World Bank source operation failed during "
                f"{operation} (attempt {attempt}/{max_attempts}). "
                f"Retrying in {retry_delay_seconds:.0f}s."
            )
            time.sleep(retry_delay_seconds)

    raise RuntimeError(
        f"World Bank source operation failed during {operation} after {max_attempts} attempts."
    ) from last_error


def _ensure_year_columns(df: pd.DataFrame, years: Sequence[int]) -> pd.DataFrame:
    """Ensure every year in ``years`` exists as a column on ``df``."""
    for y in years:
        year_col = str(y)
        if year_col not in df.columns:
            df[year_col] = pd.NA
    return df


def _drop_trailing_empty_historical_years(
    df: pd.DataFrame,
    *,
    minimum_year: int = PAST_YEAR_MIN,
) -> tuple[pd.DataFrame, list[int]]:
    """Return ``df`` without trailing historical year columns that are fully empty.

    World Bank files can expose a newly added historical year column before
    any country values are published. That placeholder year must not be carried
    into ``wb_raw.csv`` because downstream processing treats retained year
    columns as part of the valid historical horizon.

    Args:
        df (pandas.DataFrame): Wide World Bank frame containing historical year
            columns.
        minimum_year (int): Inclusive lower bound for historical year columns.

    Returns:
        tuple[pandas.DataFrame, list[int]]: A copy of ``df`` with trailing fully
        empty historical years removed plus the retained year labels.

    Raises:
        RuntimeError: If no historical year with at least one value remains.
    """
    retained_years = resolve_historical_years_from_frame(df, minimum_year=minimum_year)
    if df.empty:
        return df.copy(), retained_years

    dropped_years: list[int] = []
    while retained_years:
        last_year = retained_years[-1]
        last_year_values = pd.Series(df.loc[:, str(last_year)], copy=False)
        if bool(last_year_values.notna().any()):
            break
        dropped_years.append(retained_years.pop())

    if not retained_years:
        raise RuntimeError(
            "World Bank data did not expose any historical year columns with values "
            f"at or above {int(minimum_year)}."
        )

    if not dropped_years:
        return df.copy(), retained_years

    dropped_columns = {str(year) for year in dropped_years}
    kept_columns = [column for column in df.columns if str(column) not in dropped_columns]
    trimmed = cast(pd.DataFrame, df.loc[:, kept_columns].copy())
    return trimmed, retained_years


def _resolve_wdi_bulk_download_url(
    *,
    metadata_url: str,
    request_get: Callable[..., _StreamingResponseLike] = requests.get,
) -> str:
    """Return the current WDI CSV bulk ZIP URL from Data Catalog metadata."""
    with request_get(
        metadata_url,
        timeout=_WB_DOWNLOAD_TIMEOUT_SECONDS,
        stream=True,
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        payload = b"".join(response.iter_content(chunk_size=_WB_DOWNLOAD_CHUNK_SIZE))
    metadata = json.loads(payload.decode("utf-8-sig"))
    return str(metadata["distribution"]["url"]).strip()


def _download_wdi_bulk_zip(
    archive_path: Path,
    *,
    bulk_url: str = _WDI_BULK_URL,
    request_get: Callable[..., _StreamingResponseLike] = requests.get,
) -> Path:
    """Download the official World Bank WDI bulk ZIP to ``archive_path``."""
    archive_path = ensure_file_parent(archive_path)
    temp_path = archive_path.with_suffix(f"{archive_path.suffix}.part")
    temp_path.unlink(missing_ok=True)

    def _download_once() -> Path:
        download_url = _resolve_wdi_bulk_download_url(
            metadata_url=bulk_url,
            request_get=request_get,
        )
        with request_get(
            download_url,
            timeout=_WB_DOWNLOAD_TIMEOUT_SECONDS,
            stream=True,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_WB_DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        handle.write(chunk)
        with zipfile.ZipFile(temp_path) as archive:
            members = set(archive.namelist())
        required = {_WDI_MAIN_MEMBER, _WDI_COUNTRY_MEMBER}
        missing = sorted(required - members)
        if missing:
            raise RuntimeError(f"World Bank bulk ZIP is missing required member(s): {missing}")
        temp_path.replace(archive_path)
        return archive_path

    try:
        return _run_wb_request_with_retry(_download_once, operation="WDI bulk ZIP download")
    finally:
        temp_path.unlink(missing_ok=True)


def _load_wdi_country_frame(archive: zipfile.ZipFile) -> pd.DataFrame:
    """Return the real country table from ``WDICountry.csv``."""
    with archive.open(_WDI_COUNTRY_MEMBER) as handle:
        frame = pd.read_csv(
            handle,
            usecols=["Country Code", "Short Name", "Region"],
            dtype={
                "Country Code": "string",
                "Short Name": "string",
                "Region": "string",
            },
        )
    real_countries = cast(pd.DataFrame, frame.loc[frame["Region"].notna()].copy())
    if real_countries.empty:
        raise RuntimeError("World Bank country metadata did not expose any real countries.")
    return cast(
        pd.DataFrame,
        real_countries.rename(
            columns={
                "Country Code": "iso3_code",
                "Short Name": "wb_full_name",
            }
        )[["iso3_code", "wb_full_name"]],
    )


def _load_wdi_indicator_frame(
    archive: zipfile.ZipFile,
    *,
    indicators: Sequence[str],
) -> pd.DataFrame:
    """Load the requested indicators and years from ``WDICSV.csv``."""
    with archive.open(_WDI_MAIN_MEMBER) as handle:
        header = pd.read_csv(handle, nrows=0)
    years = resolve_historical_years_from_frame(header, minimum_year=PAST_YEAR_MIN)
    usecols = ["Country Code", "Indicator Code"] + [str(year) for year in years]
    with archive.open(_WDI_MAIN_MEMBER) as handle:
        frame = pd.read_csv(
            handle,
            usecols=usecols,
            dtype={
                "Country Code": "string",
                "Indicator Code": "string",
            },
        )
    filtered = cast(pd.DataFrame, frame.loc[frame["Indicator Code"].isin(indicators)].copy())
    if filtered.empty:
        raise RuntimeError(
            f"World Bank bulk data did not contain requested indicators {list(indicators)}."
        )
    return cast(
        pd.DataFrame,
        filtered.rename(
            columns={
                "Country Code": "iso3_code",
                "Indicator Code": "series_code",
            }
        ),
    )


def _load_wdi_bulk_frame_from_archive(
    archive_path: Path,
    indicators: Sequence[str],
) -> pd.DataFrame:
    """Load the requested World Bank indicators from a downloaded bulk ZIP."""
    with zipfile.ZipFile(archive_path) as archive:
        country_frame = _load_wdi_country_frame(archive)
        indicator_frame = _load_wdi_indicator_frame(
            archive,
            indicators=indicators,
        )

    merged = cast(
        pd.DataFrame,
        indicator_frame.merge(
            country_frame,
            on="iso3_code",
            how="inner",
            validate="many_to_one",
        ),
    )
    if merged.empty:
        raise RuntimeError("World Bank bulk data did not contain any real country indicator rows.")

    series_code = cast(pd.Series, merged["series_code"])
    merged["variable"] = cast(
        pd.Series,
        series_code.map(lambda value: CODES_TO_INDICATORS.get(str(value), str(value))),
    )
    variable = cast(pd.Series, merged["variable"])
    merged["unit"] = cast(
        pd.Series,
        variable.map(lambda value: INDICATORS_TO_UNITS.get(str(value), str(value))),
    )
    years = resolve_historical_years_from_frame(merged, minimum_year=PAST_YEAR_MIN)
    _ensure_year_columns(merged, years)
    ordered_cols = ["wb_full_name", "iso3_code", "variable", "unit"] + [str(year) for year in years]
    ordered = cast(pd.DataFrame, merged.loc[:, ordered_cols].copy())
    sorted_ordered = cast(
        pd.DataFrame,
        ordered.sort_values(by=["iso3_code", "variable"]).reset_index(drop=True),
    )
    trimmed, _retained_years = _drop_trailing_empty_historical_years(
        sorted_ordered,
        minimum_year=PAST_YEAR_MIN,
    )
    return trimmed


def _generate_wb_raw(
    *,
    refresh: bool = False,
    bulk_archive_downloader: Callable[[Path], Path] | None = None,
    bulk_frame_loader: Callable[[Path, Sequence[str]], pd.DataFrame] | None = None,
) -> Path:
    """Generate the World Bank wide table and write CSV plus metadata.

    Args:
        refresh (bool): If ``True``, delete and recreate the World Bank raw
            CSV and metadata.
        bulk_archive_downloader: Optional injected downloader used by package
            tests to provide a deterministic local bulk ZIP.
        bulk_frame_loader: Optional injected loader used by package tests to
            supply deterministic parsed bulk data.

    Returns:
        pathlib.Path: Path to the written CSV.
    """
    indicators = [CODES_TO_INDICATORS.get(code, code) for code in DEFAULT_WB_CODES]
    out = ensure_file_parent(_get_output_path(OUTPUT_FILENAME))
    if refresh:
        _clear_raw_output_scope(OUTPUT_FILENAME)

    meta = _read_meta(OUTPUT_FILENAME)
    if out.exists() and (not refresh) and meta:
        cached_years = resolve_historical_years_from_frame(
            pd.read_csv(out, nrows=0),
            minimum_year=PAST_YEAR_MIN,
        )
        if _meta_covers(meta, int(min(cached_years)), int(max(cached_years)), indicators):
            return out

    print_user_text_line(
        "Downloading WB Population and GDP PPP data for the available historical year range"
    )

    effective_downloader = (
        _download_wdi_bulk_zip if bulk_archive_downloader is None else bulk_archive_downloader
    )
    effective_loader = (
        _load_wdi_bulk_frame_from_archive if bulk_frame_loader is None else bulk_frame_loader
    )
    with tempfile.TemporaryDirectory(prefix="pyaesa_wdi_") as temp_dir:
        archive_path = Path(temp_dir) / "WDI_CSV.zip"
        effective_downloader(archive_path)
        out_df = effective_loader(archive_path, DEFAULT_WB_CODES)

    years = resolve_historical_years_from_frame(out_df, minimum_year=PAST_YEAR_MIN)
    out_df.to_csv(out, index=False)
    _write_meta(OUTPUT_FILENAME, int(min(years)), int(max(years)), indicators)
    print_user_text_line(f"Downloaded WB raw CSV to: {out}")
    return out
