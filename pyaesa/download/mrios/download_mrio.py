"""Download MRIO archives and track coverage via metadata.

This module orchestrates downloads for the supported MRIO sources:
EXIOBASE 3.9.6, EXIOBASE 3.10.2, and OECD ICIO v2025. It exposes a
registry describing how to download each source and records which years
exist locally after each run. If archives already exist on disk (even
when metadata is missing) they are detected and added to the metadata
record.

Key concepts:

* ``SOURCE_HANDLERS`` maps short keys (for example ``"exiobase_396_ixi"``) to the
  handler objects required to obtain and identify archives.
* :func:`download_mrio` iterates year by year, checks both metadata
  and the filesystem, and tracks outcomes in :class:`DownloadReportMRIO` so
  callers can inspect downloaded/skipped_already_saved/failed years.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from pyaesa.download.mrios.utils.exio_download import (
    EXIOBASE_HANDLERS,
)
from pyaesa.download.mrios.utils.metadata import (
    _read_meta,
    _write_meta,
)
from pyaesa.download.mrios.utils.oecd_download import (
    OECD_HANDLER,
)
from pyaesa.download.mrios.utils.paths import _get_full_dir
from pyaesa.download.mrios.utils.source_registry import (
    normalize_mrio_source_key,
)
from pyaesa.download.mrios.utils.year_selection import (
    YearSelection,
    normalize_mrio_years,
)
from pyaesa.shared.runtime.io.filesystem import ensure_dir
from pyaesa.shared.runtime.reporting.progress import YearProgressPrinter

# MRIO source handlers exposing per source download ownership.
SOURCE_HANDLERS = {
    handler.key: handler
    for handler in (
        *EXIOBASE_HANDLERS,
        OECD_HANDLER,
    )
}


@dataclass
class DownloadReportMRIO:
    """Outcome container for a MRIO download run.

    Attributes:
        source: Source identifier (for example ``"exiobase_396_ixi"``).
        requested: Requested units (years) for the run.
        downloaded: Units freshly downloaded during the run.
        skipped_already_saved: Units skipped because archives already exist.
        errors: Mapping from unit to error message.
    """

    source: str
    requested: List[Any]
    download_root: Path | None = None
    downloaded: List[Any] = field(default_factory=list)
    skipped_already_saved: List[Any] = field(default_factory=list)
    errors: Dict[Any, str] = field(default_factory=dict)

    def missing(self) -> List[Any]:
        """Return requested years that could not be satisfied."""
        handled = set(self.downloaded) | set(self.skipped_already_saved)
        return [x for x in self.requested if x not in handled]

    def _format_year_ranges(self, values: List[Any]) -> str:
        """Format year like values as compact ranges when possible."""
        if not values:
            return "[]"
        years: List[int] = []
        for value in values:
            try:
                years.append(int(value))
            except (TypeError, ValueError):
                labels = sorted({str(item) for item in values})
                return ", ".join(labels)
        years = sorted({int(year) for year in years})
        ranges: List[str] = []
        start = years[0]
        prev = years[0]
        for year in years[1:]:
            if year == prev + 1:
                prev = year
                continue
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = year
            prev = year
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ", ".join(ranges)

    def _format_year_ranges_with_count(self, values: List[Any]) -> str:
        """Format year like values and append a count."""
        unique_count = len({str(value) for value in values})
        return f"{self._format_year_ranges(values)} ({unique_count} year(s))"

    def __str__(self) -> str:
        """Return a multi line summary."""
        lines: List[str] = [
            f"[{self.source}] Summary:",
            f"  Requested: {self._format_year_ranges_with_count(self.requested)}",
            f"  Downloaded: {self._format_year_ranges_with_count(self.downloaded)}",
        ]
        if self.skipped_already_saved:
            lines.append(
                "  Skipped existing files: "
                f"{self._format_year_ranges_with_count(self.skipped_already_saved)}"
            )
        if self.errors:
            failed = sorted(self.errors.keys())
            lines.append(f"  Errors: {self._format_year_ranges_with_count(failed)}")
            lines.append("  See report.errors for details.")
        if self.download_root is not None:
            lines.append(f"  Download root: {self.download_root}")
        return "\n".join(lines)

    __repr__ = __str__


def _download_year_with_report(
    *,
    handler,
    full_dir: Path,
    units: List[int],
    year: int,
    report: DownloadReportMRIO,
    disk_years: set[int],
    completed_download_units: set[str],
    refresh: bool,
) -> None:
    """Download one year and update ``report``/``disk_years``."""
    try:
        handler.download_year(
            year=year,
            full_dir=full_dir,
            requested_years=units,
            report=report,
            completed_download_units=completed_download_units,
            refresh=refresh,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        report.errors[year] = f"{exc}"
    else:
        disk_years.update(handler.list_existing_years(full_dir))


def _coalesce_pending_download_years(*, handler, pending_years: List[int]) -> List[int]:
    """Return the effective download loop units for one source handler."""
    coalesce = getattr(handler, "coalesce_pending_years", None)
    if coalesce is None:
        return list(pending_years)
    return list(coalesce(pending_years))


def _progress_label_for_unit(*, handler, year: int) -> int | str:
    """Return the display label for one progress unit."""
    progress_label = getattr(handler, "progress_label", None)
    if progress_label is None:
        return year
    return progress_label(year)


def download_mrio(
    source: str,
    years: YearSelection = None,
    *,
    refresh: bool = False,
) -> DownloadReportMRIO | None:
    """Download missing MRIO archives for ``source``.

    The function inspects archives under the source download folder and
    downloads selected years that are missing. When ``refresh=True`` the
    selected archive scope is downloaded again to replace existing files.
    Omit arguments to use their default.

    Args:
        source: MRIO source key (``"exiobase_396_ixi"``,
            ``"exiobase_396_pxp"``,
            ``"exiobase_3102_ixi"``, ``"exiobase_3102_pxp"``,
            or ``"oecd_v2025"``).
        years: Optional year selection. Accepted forms are ``None``, one
            integer year, a ``range``, or a sequence of integer years. Defaults
            to ``None``, which selects all supported years for ``source``:
            EXIOBASE 3.9.6 uses 1995 to 2022, EXIOBASE 3.10.2 uses 1995 to
            2024, and OECD ICIO v2025 uses 1995 to 2022.
        refresh: If ``True``, download the selected raw MRIO archive scope
            again to replace previous downloads. For EXIOBASE, the scope is
            each requested year archive under the selected source and system
            raw folder. For OECD ICIO, the scope is the OECD bundle containing
            each requested year, so refreshing one year can replace every
            yearly CSV extracted from that bundle. Processed MRIO outputs and
            project outputs are not refreshed. Defaults to ``False``.

    Returns:
        DownloadReportMRIO capturing requested, downloaded or refreshed,
        skipped_already_saved, and error states, or ``None`` when nothing was
        downloaded and no errors occurred.

    Raises:
        ValueError: If ``source`` or ``years`` falls outside the supported
            MRIO selection contract.
        OSError: If raw download directory creation or metadata writing fails.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function. The source download folder is created lazily when missing.

    Example:
        Download EXIOBASE 3.10.2 ixi archives with default years::

            from pyaesa import download_mrio

            download_mrio("exiobase_3102_ixi")
    """
    source_key = normalize_mrio_source_key(source)
    handler = SOURCE_HANDLERS[source_key]
    full_dir = _get_full_dir(source_key)
    return _download_mrio_with_handler(
        source_key=source_key,
        handler=handler,
        full_dir=full_dir,
        years=years,
        refresh=refresh,
    )


def _download_mrio_with_handler(
    *,
    source_key: str,
    handler,
    full_dir: Path,
    years: YearSelection,
    refresh: bool,
) -> DownloadReportMRIO | None:
    """Download missing MRIO archives for one normalized source.

    The function inspects archives under the source download folder and
    downloads selected years that are missing. When ``refresh=True`` the
    selected archive scope is downloaded again and recorded as work performed.
    Results for the current run are recorded in :class:`DownloadReportMRIO`.

    Args:
        source_key: Source identifier associated with ``handler``.
        handler: Download handler implementing the MRIO download contract.
        full_dir: Directory where raw archives/CSVs are stored.
        years: Year selection (``None`` for defaults, ``int``,
            ``range``, or a sequence of years).
        refresh: If ``True``, download the selected raw MRIO archive scope
            again to replace previous downloads. For EXIOBASE, the scope is
            each requested year archive under the selected source and system
            raw folder. For OECD ICIO, the scope is the OECD bundle containing
            each requested year, so refreshing one year can replace every
            yearly CSV extracted from that bundle. Processed MRIO outputs and
            project outputs are not refreshed. Defaults to ``False``.

    Returns:
        DownloadReportMRIO capturing requested, downloaded or refreshed,
        skipped_already_saved, and error states, or ``None`` when nothing was
        downloaded and no errors occurred.

    Notes:
        The repository must first be created via :func:`set_workspace`. This
        function creates the source download folder lazily when missing.
    """
    full_dir = ensure_dir(full_dir)

    years_list = normalize_mrio_years(years, source_key=source_key)
    disk_years = handler.list_existing_years(full_dir)

    # iterate per year for both sources;
    units = years_list

    # Prepare a report object to record which years were downloaded,
    # which were skipped because the full archive already exists, and any
    # errors encountered during the run.
    report = DownloadReportMRIO(
        source=source_key,
        requested=units,
    )
    report.download_root = full_dir
    completed_download_units: set[str] = set()
    pending_years: list[int] = []
    if refresh:
        pending_years = [int(year) for year in units]
    else:
        for unit in units:
            year = int(unit)
            full_exists = handler.full_exists(full_dir, year)
            if year in disk_years or full_exists:
                report.skipped_already_saved.append(year)
                continue
            pending_years.append(year)
    progress_years = _coalesce_pending_download_years(
        handler=handler,
        pending_years=pending_years,
    )
    progress = YearProgressPrinter(
        source=source_key,
        action="downloading",
        total=len(progress_years),
    )

    try:
        for year in progress_years:
            # Attempt download. Any exceptions are recorded on report.errors.
            progress_label = _progress_label_for_unit(handler=handler, year=year)
            progress.begin_year(progress_label)
            _download_year_with_report(
                handler=handler,
                full_dir=full_dir,
                units=units,
                year=year,
                report=report,
                disk_years=disk_years,
                completed_download_units=completed_download_units,
                refresh=refresh,
            )
            progress.complete_year(progress_label)
    finally:
        progress.finish()

    # Build a list of available years observed during this run: both newly
    # downloaded years and those that were skipped because the archive
    # already existed.
    available_years = sorted(
        set(report.downloaded) | set(report.skipped_already_saved) | set(disk_years)
    )

    # Read existing metadata (if any) and merge the years observed previously
    # with the years available after this run.
    existing_meta = _read_meta(source_key)
    existing_years = (
        {int(year) for year in existing_meta["years"]} if existing_meta is not None else set()
    )

    # Merge previously known years with newly available ones and persist the
    # updated coverage back to the metadata JSON so future runs can detect
    # known coverage.
    merged = sorted(existing_years | set(available_years))
    _write_meta(source_key, merged)

    # Return the run report to the caller so they can inspect what happened
    # (which years were downloaded/skipped and any errors encountered).
    if report.downloaded or report.errors:
        return report
    return None
