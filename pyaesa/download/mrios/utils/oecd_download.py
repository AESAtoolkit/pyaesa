"""Utilities to download OECD ICIO v2025 bundles.

This module contains utilities to map years to the official OECD
bundles used in the OECD ICIO v2025 release, construct expected
archive names, and download + unpack the official zip bundles.

Behavioral notes:
- The OECD releases are distributed as multi-year ZIP bundles (for example
    ``1995-2000``). The downloader extracts all files and renames those that
    include the marker ``_SML`` into canonical names like ``ICIO2025_1995.csv``
    so downstream parsing code can find them by year.
- The function ``download_oecd_bundle`` performs I/O network activity and
    can raise network related exceptions (requests) or I/O errors.
"""

from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, Callable, List, Mapping
import zipfile
from requests import api as request

from pyaesa.download.mrios.utils.archive_validation import _assert_valid_zip
from pyaesa.download.mrios.utils.paths import (
    _get_oecd_bundle_temp_dir,
    _get_oecd_bundle_temp_zip_path,
    _get_oecd_csv_path,
)
from pyaesa.shared.runtime.io.filesystem import ensure_dir

if TYPE_CHECKING:
    from pyaesa.download.mrios.download_mrio import DownloadReportMRIO

_OECD_VERSION = "v2025"

# Year bundles available for OECD ICIO 2025
OECD_BUNDLES = [
    "1995-2000",
    "2001-2005",
    "2006-2010",
    "2011-2015",
    "2016-2022",
]


def _year_to_oecd_bundle(year):
    """Return the OECD bundle string that includes ``year``.

    Args:
        year: Four digit year.

    Returns:
        The bundle string (for example ``"1995-2000"``) which contains
        the requested year.

    Raises:
        ValueError: If the year is not covered by any known bundle.
    """
    year = int(year)
    for bundle in OECD_BUNDLES:
        start, end = bundle.split("-")
        if int(start) <= year <= int(end):
            return bundle
    raise ValueError(f"Year {year} is not covered by any OECD ICIO v2025 bundle")


def _oecd_bundle_years(bundle: str) -> list[int]:
    """Return the covered years for one OECD bundle label."""
    start, end = bundle.split("-")
    return list(range(int(start), int(end) + 1))


def _coalesce_oecd_bundle_years(pending_years: list[int]) -> list[int]:
    """Return one representative year per OECD bundle in deterministic order."""
    representative_years: list[int] = []
    seen_bundles: set[str] = set()
    for year in pending_years:
        bundle = _year_to_oecd_bundle(year)
        if bundle in seen_bundles:
            continue
        seen_bundles.add(bundle)
        representative_years.append(int(year))
    return representative_years


def _validate_oecd_bundle_outputs(temp_dir: Path, bundle: str) -> None:
    """Fail fast when a staged OECD bundle is missing expected CSV outputs."""
    missing = [
        year
        for year in _oecd_bundle_years(bundle)
        if not _get_oecd_csv_path(temp_dir, year).exists()
    ]
    if missing:
        raise RuntimeError(
            f"OECD ICIO bundle {bundle} did not extract all expected yearly CSVs. "
            f"Missing years: {missing}."
        )


# pylint: disable=C0301
OECD_URLS_V2025 = {
    "1995-2000": "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=f337e03f-697a-4495-a772-78e8963da2d0",
    "2001-2005": "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=873eb327-a175-449e-9608-677d1b9ebf83",
    "2006-2010": "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=dac9b40c-a3ab-4689-83c7-a9015b289dc1",
    "2011-2015": "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=5cb62314-8367-4a1f-a1bf-ae7a8064fd41",
    "2016-2022": "https://stats.oecd.org/wbos/fileview2.aspx?IDFile=7af46a6a-c5a6-4ba1-91d8-5c09ea436cd9",
}
# pylint: enable=C0301


def _download_oecd_bundle(
    target_dir,
    bundle,
    *,
    request_get: Callable[..., Any] = request.get,
    urls_by_bundle: Mapping[str, str] | None = None,
    refresh: bool = False,
    archive_validator: Callable[..., None] = _assert_valid_zip,
):
    """Download one OECD ICIO v2025 bundle and rename extracted files.

    The function performs the following steps:

    1. Download the ZIP file for the requested ``bundle`` into ``target_dir``.
    2. Extract all files from the ZIP into ``target_dir`` and delete the ZIP.
    3. Rename files containing ``_SML`` into the canonical
       ``ICIO2025_<year>.csv`` targets expected downstream.

    Args:
        target_dir: Destination directory where the ZIP is saved and extracted.
            The directory is created lazily when missing.
        bundle: Bundle string such as ``"1995-2000"``.
    Raises:
        ValueError: If ``bundle`` is unsupported.
        requests.exceptions.RequestException: If the download fails.
        OSError: For filesystem errors while writing, extracting, or renaming.
    """
    target_dir = ensure_dir(Path(target_dir))

    effective_urls = OECD_URLS_V2025 if urls_by_bundle is None else dict(urls_by_bundle)

    temp_dir = _get_oecd_bundle_temp_dir(target_dir, bundle, _OECD_VERSION)
    zip_path = _get_oecd_bundle_temp_zip_path(target_dir, bundle, _OECD_VERSION)
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir = ensure_dir(temp_dir)

    try:
        resp = request_get(effective_urls[bundle], stream=True, timeout=30)
        resp.raise_for_status()

        # Stream the zip to disk. Chunked writing minimizes memory use for
        # large bundles.
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(1024 * 5):
                if not chunk:
                    continue
                fh.write(chunk)

        archive_validator(zip_path, artifact_label=f"OECD ICIO bundle {bundle}")

        # Extract everything into a temporary bundle directory and only move
        # validated yearly CSVs into the final folder after extraction
        # succeeds.
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(temp_dir)
        zip_path.unlink()

        desired_prefix = "ICIO2025_"
        for staged_file in (p for p in temp_dir.iterdir() if p.is_file() and "_SML" in p.name):
            new_basename = staged_file.name.replace("_SML", "")
            canonical_name = f"{desired_prefix}{new_basename}"
            staged_file.rename(temp_dir / canonical_name)

        _validate_oecd_bundle_outputs(temp_dir, bundle)

        for year in _oecd_bundle_years(bundle):
            staged_csv = _get_oecd_csv_path(temp_dir, year)
            target = _get_oecd_csv_path(target_dir, year)
            if target.exists() and not refresh:
                staged_csv.unlink()
                continue
            staged_csv.replace(target)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _bundle_download_unit(bundle: str) -> str:
    """Return the run local deduplication token for one OECD bundle."""
    return f"oecd_bundle:{bundle}"


class OECDMRIOHandler:
    """Handler owning OECD specific download behavior."""

    key = "oecd_v2025"

    def full_exists(self, full_dir: Path, year: int) -> bool:
        """Return True when the OECD CSV for ``year`` exists.

        Args:
            full_dir: Directory where CSV files are stored.
            year: Year to check.

        Returns:
            True when the CSV exists on disk.
        """
        return _get_oecd_csv_path(full_dir, year).exists()

    def coalesce_pending_years(self, pending_years: list[int]) -> list[int]:
        """Return one representative year per pending OECD bundle."""
        return _coalesce_oecd_bundle_years(pending_years)

    def progress_label(self, year: int) -> str:
        """Return the OECD bundle label used in progress rendering."""
        return _year_to_oecd_bundle(year)

    def list_existing_years(self, full_dir: Path) -> set[int]:
        """Return OECD years already present on disk.

        Args:
            full_dir: Directory to scan for OECD CSV files.

        Returns:
            A set of integer years detected from CSV filenames in ``full_dir``.
        """
        years: set[int] = set()
        for csv_path in full_dir.glob("ICIO2025_*.csv"):
            stem = csv_path.stem
            parts = stem.split("_")
            try:
                years.add(int(parts[1]))
            except (IndexError, TypeError, ValueError):
                raise ValueError(
                    f"Malformed OECD ICIO CSV filename in '{full_dir}': {csv_path.name}."
                ) from None
        return years

    def download_year(
        self,
        *,
        year: int,
        full_dir: Path,
        requested_years: List[int],
        report: "DownloadReportMRIO",
        completed_download_units: set[str],
        refresh: bool = False,
        bundle_downloader=_download_oecd_bundle,
    ) -> None:
        """Download the OECD bundle covering ``year`` and record new CSVs.

        The method downloads the entire bundle (which may contain multiple
        years), extracts and normalises filenames and appends any newly
        created CSV years to ``report.downloaded``.

        Args:
            year: Year for which the covering bundle will be downloaded.
            full_dir: Directory where CSV files should be stored.
            requested_years: Years requested by the caller. The handler uses
                this list to avoid reporting unrequested years from a downloaded
                multi-year bundle.
            report: The run report object which will be updated with any
                newly downloaded years.
        """
        bundle = _year_to_oecd_bundle(year)
        bundle_unit = _bundle_download_unit(bundle)
        if bundle_unit in completed_download_units:
            return
        years_in_bundle = _oecd_bundle_years(bundle)
        requested = requested_years
        preexisting = (
            set()
            if refresh
            else {y for y in years_in_bundle if _get_oecd_csv_path(full_dir, y).exists()}
        )
        bundle_downloader(full_dir, bundle, refresh=refresh)
        completed_download_units.add(bundle_unit)
        for y in years_in_bundle:
            csv_path = _get_oecd_csv_path(full_dir, y)
            if (
                y in requested
                and y not in preexisting
                and csv_path.exists()
                and y not in report.downloaded
            ):
                report.downloaded.append(y)


OECD_HANDLER = OECDMRIOHandler()
