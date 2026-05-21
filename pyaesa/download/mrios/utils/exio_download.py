"""Utilities to download EXIOBASE archives for all supported versions."""

from contextlib import contextmanager
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, Callable, Iterator, List, Union

import pymrio
import pymrio.tools.iodownloader as pymrio_iodownloader

from pyaesa.download.mrios.utils.archive_validation import _assert_valid_zip
from pyaesa.download.mrios.utils.paths import (
    _get_exio_archive_path,
    _get_exio_archive_temp_dir,
    _get_exio_archive_temp_path,
)
from pyaesa.shared.runtime.io.filesystem import ensure_dir
from pyaesa.download.mrios.utils.logging import (
    suppress_pymrio_logging,
)
from pyaesa.download.mrios.utils.source_registry import iter_mrio_entries

if TYPE_CHECKING:
    from pyaesa.download.mrios.download_mrio import DownloadReportMRIO


@contextmanager
def _temporary_pymrio_headers(headers: dict[str, str]) -> Iterator[None]:
    """Temporarily replace pymrio downloader headers and restore afterwards."""
    original_headers = dict(pymrio_iodownloader.HEADERS)
    try:
        pymrio_iodownloader.HEADERS.clear()
        pymrio_iodownloader.HEADERS.update(headers)
        yield
    finally:
        pymrio_iodownloader.HEADERS.clear()
        pymrio_iodownloader.HEADERS.update(original_headers)


def _download_with_headers(
    *,
    downloader: Callable[..., Any],
    target_dir: Path,
    system_clean: str,
    year: int,
    doi: str,
    headers: dict[str, str],
) -> None:
    """Run one EXIO download call with explicit pymrio request headers."""
    with _temporary_pymrio_headers(headers):
        downloader(
            storage_folder=str(target_dir),
            system=system_clean,
            years=[year],
            doi=doi,
        )


def _ensure_archive_exists(*, archive_path: Path, year: int, system_clean: str, doi: str) -> None:
    """Fail fast when an EXIO archive is still missing after a download attempt."""
    if archive_path.exists():
        return
    raise RuntimeError(
        f"EXIOBASE archive was not downloaded for year={year}, system={system_clean}, doi={doi}. "
        "The DOI page was reachable but no matching archive URL could be resolved."
    )


def _download_exiobase_year(
    target_dir: Union[Path, str],
    year: int,
    *,
    system: str,
    doi: str,
    downloader: Callable[..., Any] = pymrio.download_exiobase3,
    archive_validator: Callable[..., None] = _assert_valid_zip,
) -> None:
    """Download a single EXIOBASE year into ``target_dir``.

    Args:
        target_dir: Directory where the downloaded archive (zip) is stored.
            The directory is created lazily when missing.
        year: Four digit year to download (for example ``1995``).
        system: EXIO accounting system (``"ixi"`` or ``"pxp"``).
        doi: DOI used by :func:`pymrio.download_exiobase3` to resolve the
            dataset.

    Raises:
        Exception: Propagates any exception raised by
            :func:`pymrio.download_exiobase3` so callers can react to network
            or filesystem issues.
    """
    target_dir = ensure_dir(Path(target_dir))
    system_clean = system
    archive_path = _get_exio_archive_path(target_dir, year, system=system_clean)
    temp_dir = _get_exio_archive_temp_dir(target_dir, year, system=system_clean)
    temp_archive_path = _get_exio_archive_temp_path(target_dir, year, system=system_clean)

    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir = ensure_dir(temp_dir)

    try:
        # External INFO logs are silenced only for the duration
        # of the download call.
        # Always use neutral headers because DOI page parsing may fail with
        # the pymrio default User Agent in some environments.
        with suppress_pymrio_logging():
            _download_with_headers(
                downloader=downloader,
                target_dir=temp_dir,
                system_clean=system_clean,
                year=year,
                doi=doi,
                # Some environments receive a short DOI intermediary page when
                # pymrio's default User Agent is sent; using empty headers keeps
                # Zenodo file link discovery reliable for this EXIO DOI flow.
                headers={},
            )

        _ensure_archive_exists(
            archive_path=temp_archive_path,
            year=year,
            system_clean=system_clean,
            doi=doi,
        )
        archive_validator(
            temp_archive_path,
            artifact_label=f"EXIOBASE archive for year={year}, system={system_clean}, doi={doi}",
        )
        temp_archive_path.replace(archive_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class ExiobaseMRIOHandler:
    """Handler owning EXIOBASE specific download behavior."""

    def __init__(self, *, key: str, system: str, doi: str) -> None:
        """Initialize one EXIOBASE handler bound to a source key/system."""
        self.key = str(key).strip().lower()
        self.system = system
        self.doi = str(doi).strip()

    def full_exists(self, full_dir: Path, year: int) -> bool:
        """Return True when the EXIO archive for ``year`` exists."""
        return _get_exio_archive_path(full_dir, year, system=self.system).exists()

    def list_existing_years(self, full_dir: Path) -> set[int]:
        """Return EXIOBASE years already present on disk (metadata aware)."""
        years: set[int] = set()
        pattern = f"IOT_*_{self.system}.zip"
        for archive in full_dir.glob(pattern):
            stem = archive.stem
            parts = stem.split("_")
            try:
                years.add(int(parts[1]))
            except (IndexError, TypeError, ValueError):
                raise ValueError(
                    f"Malformed EXIOBASE archive filename in '{full_dir}': {archive.name}."
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
        download_func=_download_exiobase_year,
    ) -> None:
        """Download ``year`` if needed and mark new artefacts in ``report``."""
        download_unit = f"{self.key}:{int(year)}"
        if download_unit in completed_download_units:
            return
        archive_path = _get_exio_archive_path(full_dir, year, system=self.system)
        preexisting = archive_path.exists()
        download_func(full_dir, year, system=self.system, doi=self.doi)
        completed_download_units.add(download_unit)
        if (refresh or not preexisting) and archive_path.exists() and year not in report.downloaded:
            report.downloaded.append(year)


EXIOBASE_HANDLERS = tuple(
    ExiobaseMRIOHandler(
        key=entry.source_key,
        system=str(entry.system),
        doi=str(entry.doi),
    )
    for entry in iter_mrio_entries()
    if entry.family == "exiobase"
)
