"""Process population and GDP datasets for downstream use.

Raw CSVs produced by the download layer are loaded, harmonised with MRIO
mappings, completed for the requested year ranges, and written to
``data_processed`` as GDP and population tables. Metadata coverage is tracked
so repeated calls can skip work when possible.
"""

from pathlib import Path

import pandas as pd

from pyaesa.download.pop_gdp.contracts import (
    FUTURE_YEARS,
    PAST_YEAR_MIN,
    resolve_historical_years_from_frame,
)
from pyaesa.download.pop_gdp.raw_paths import _get_output_path
from pyaesa.process.pop_gdp.io.metadata import (
    _meta_covers,
    _read_meta,
    _write_meta,
)
from pyaesa.process.pop_gdp.io.paths import (
    _clear_processed_dataset_scope,
    _get_log_path,
    _get_processed_output_path,
    _get_ssp_matching_path,
    _get_wb_matching_path,
)
from pyaesa.process.pop_gdp.sources.ssp import _process_ssp_dataset
from pyaesa.process.pop_gdp.sources.wb import _process_wb_dataset
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.text import print_user_text_line

_EXIO_MATCHING_SOURCE = "exiobase_3102_ixi"


def _load_raw_frame(frame_key: str) -> pd.DataFrame:
    """Load a raw frame from disk.

    Args:
        frame_key (str): Key identifying the frame (``'wb'``, ``'ssp'``, etc.).

    Returns:
        pandas.DataFrame: Requested raw dataset loaded from disk.

    Raises:
        RuntimeError: If the dataset is unavailable on disk.
    """
    path = _get_output_path(frame_key)
    if not path.exists():
        raise RuntimeError(
            f"Raw dataset '{frame_key}' not found at {path}. "
            "Run download_pop_gdp(...) before process_pop_gdp(...)."
        )
    return pd.read_csv(path)


def _load_matching(path: Path) -> pd.DataFrame:
    """Return MRIO matching rows from one canonical matching CSV path.

    Args:
        path: Matching CSV path resolved by the pop GDP path owner.

    Returns:
        pandas.DataFrame: Matching table read from disk.

    Raises:
        RuntimeError: If the expected CSV is absent.
    """
    if not path.exists():
        raise RuntimeError(
            "The active workspace is missing a packaged MRIO matching CSV required by "
            f"process_pop_gdp(...). Missing file: {path}. Run set_workspace(...) for "
            "the workspace before processing population and GDP data."
        )
    return pd.read_csv(path, dtype=str)


def _process_wb(*, refresh: bool) -> Path:
    """Return processed WB/IMF tables and output path.

    Args:
        refresh (bool): When True, clear and rebuild the WB processed scope.

    Returns:
        pathlib.Path: Output path to the processed CSV.
    """
    wb_raw = _load_raw_frame("wb")
    years = resolve_historical_years_from_frame(wb_raw, minimum_year=PAST_YEAR_MIN)
    begin_year = min(years)
    end_year = max(years)
    out_path = _get_processed_output_path("wb")
    log_path = _get_log_path("wb_fill_log.csv")
    meta_name = "wb_processed"
    if refresh:
        _clear_processed_dataset_scope("wb")

    if out_path.exists() and not refresh:
        meta = _read_meta(meta_name)
        if meta and _meta_covers(meta, begin_year, end_year):
            return out_path

    print_user_text_line(f"Processing WB population and GDP data for years {begin_year}-{end_year}")
    imf_raw = _load_raw_frame("imf_twn")
    exio_mapping = _load_matching(_get_wb_matching_path(_EXIO_MATCHING_SOURCE))
    oecd_mapping = _load_matching(_get_wb_matching_path("oecd_v2025"))

    processed, fill_log = _process_wb_dataset(
        wb_raw,
        imf_raw,
        years,
        exio_mapping,
        oecd_mapping,
    )

    out_path = ensure_file_parent(out_path)
    processed.to_csv(out_path, index=False)
    log_path = ensure_file_parent(log_path)
    fill_log.to_csv(log_path, index=False)
    _write_meta(meta_name, begin_year, end_year)
    print_user_text_line(f"Wrote processed WB CSV to {out_path}")
    print_user_text_line(f"Wrote WB fill log to {log_path}")
    del processed
    del fill_log
    return out_path


def _process_ssp(*, refresh: bool) -> Path:
    """Return the processed SSP table and output path.

    Args:
        refresh (bool): When True, clear and rebuild the SSP processed scope.

    Returns:
        pathlib.Path: Output path to the processed CSV.
    """
    years = list(FUTURE_YEARS)
    begin_year = min(years)
    end_year = max(years)
    out_path = _get_processed_output_path("ssp")
    meta_name = "ssp_processed"
    if refresh:
        _clear_processed_dataset_scope("ssp")

    if out_path.exists() and not refresh:
        meta = _read_meta(meta_name)
        if meta and _meta_covers(meta, begin_year, end_year):
            return out_path

    print_user_text_line(
        f"Processing SSP population and GDP data for years {begin_year}-{end_year}"
    )
    ssp_raw = _load_raw_frame("ssp")
    exio_mapping = _load_matching(_get_ssp_matching_path(_EXIO_MATCHING_SOURCE))
    oecd_mapping = _load_matching(_get_ssp_matching_path("oecd_v2025"))
    processed = _process_ssp_dataset(
        ssp_raw,
        years,
        exio_mapping,
        oecd_mapping,
    )

    out_path = ensure_file_parent(out_path)
    processed.to_csv(out_path, index=False)
    _write_meta(meta_name, begin_year, end_year)
    print_user_text_line(f"Wrote processed SSP CSV to {out_path}")
    del processed
    return out_path


def _run_process_pop_gdp(
    *,
    past_years: bool = True,
    future_years: bool = True,
    refresh: bool = False,
) -> None:
    """Run the processed population/GDP pipeline."""
    if past_years:
        _process_wb(refresh=refresh)

    if future_years:
        _process_ssp(refresh=refresh)


def process_pop_gdp(
    *,
    past_years: bool = True,
    future_years: bool = True,
    refresh: bool = False,
) -> None:
    """Process raw population/GDP data into harmonised analysis tables.

    Raw population/GDP files produced by ``download_pop_gdp(...)``
    must already exist on disk in the active workspace. This function reads those
    files and does not download missing population or GDP data.
    Omit arguments to use their default.

    Args:
        past_years: If ``True``, build the historical (World Bank + IMF)
            processed output. Default ``True`` includes the historical branch.
        future_years: If ``True``, build the SSP processed output.
            Default ``True`` includes the prospective branch.
        refresh: If ``True``, clear and recompute only the selected processed
            population and GDP tables under ``data_processed/pop_gdp``.
            ``past_years=True`` refreshes ``wb_processed.csv``, its metadata,
            and the World Bank fill log. ``future_years=True`` refreshes
            ``ssp_processed.csv`` and its metadata. Raw downloads and project
            outputs are not refreshed. Defaults to ``False``.

    Returns:
        None.

    Raises:
        RuntimeError: If a required raw population, GDP, or packaged matching
            CSV is missing from the active workspace.
        OSError: If writing a selected processed output, log, or metadata file
            fails.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function. Run ``download_pop_gdp(...)`` before processing when the raw
        population, GDP, and SSP files are not already present. The processed
        outputs contain GDP and population rows for the requested year ranges.

    Example:
        Process historical and SSP population/GDP inputs::

            from pyaesa import process_pop_gdp

            process_pop_gdp()
    """
    _run_process_pop_gdp(
        past_years=past_years,
        future_years=future_years,
        refresh=refresh,
    )
