import importlib
import io
import logging
import struct
import warnings
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

from pyaesa.download.mrios.utils.archive_validation import _assert_valid_zip
from pyaesa.download.mrios.utils.year_selection import normalize_mrio_years
from pyaesa.download.mrios.utils import exio_download as exio_mod
from pyaesa.download.mrios.utils import metadata as metadata_mod
from pyaesa.download.mrios.utils import oecd_download as oecd_mod
from pyaesa.download.mrios.utils.logging import suppress_pymrio_logging
from pyaesa.download.mrios.utils.paths import (
    _get_exio_archive_path,
    _get_exio_archive_temp_dir,
    _get_exio_archive_temp_path,
    _get_oecd_bundle_temp_dir,
    _get_oecd_bundle_temp_zip_path,
    _get_oecd_csv_path,
    _resolve_source_layout,
)
from pyaesa.download.mrios.utils.source_registry import default_years_for_source
from pyaesa.download.mrios.utils.source_registry import get_mrio_entry
from pyaesa.download.mrios.utils.source_registry import is_exio_mrio_source
from pyaesa.download.mrios.utils.source_registry import list_mrio_source_keys

mrio_mod = importlib.import_module("pyaesa.download.mrios.download_mrio")


class _DummyHandler:
    def __init__(self, *, failing_years: set[int] | None = None) -> None:
        self.failing_years = set() if failing_years is None else set(failing_years)

    def _path(self, full_dir: Path, year: int) -> Path:
        return full_dir / f"{int(year)}.bin"

    def full_exists(self, full_dir: Path, year: int) -> bool:
        return self._path(full_dir, year).exists()

    def list_existing_years(self, full_dir: Path) -> set[int]:
        years: set[int] = set()
        for path in full_dir.glob("*.bin"):
            years.add(int(path.stem))
        return years

    def download_year(
        self,
        *,
        year: int,
        full_dir: Path,
        requested_years: list[int],
        report,
        completed_download_units: set[str],
        refresh: bool = False,
    ) -> None:
        del refresh
        del requested_years
        if f"dummy:{year}" in completed_download_units:
            return
        if year in self.failing_years:
            raise RuntimeError(f"cannot download {year}")
        self._path(full_dir, year).write_text(f"year={year}", encoding="utf-8")
        completed_download_units.add(f"dummy:{year}")
        report.downloaded.append(year)


class _BundledDummyHandler(_DummyHandler):
    def coalesce_pending_years(self, pending_years: list[int]) -> list[int]:
        return [pending_years[0]] if pending_years else []

    def progress_label(self, year: int) -> str:
        return f"bundle-{int(year)}"

    def download_year(
        self,
        *,
        year: int,
        full_dir: Path,
        requested_years: list[int],
        report,
        completed_download_units: set[str],
        refresh: bool = False,
    ) -> None:
        del refresh
        if f"bundle:{year}" in completed_download_units:
            return
        for requested_year in requested_years:
            self._path(full_dir, requested_year).write_text(
                f"year={requested_year}",
                encoding="utf-8",
            )
            report.downloaded.append(int(requested_year))
        completed_download_units.add(f"bundle:{year}")


class _BytesResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for index in range(0, len(self._payload), chunk_size):
            yield self._payload[index : index + chunk_size]


class _ExternalPymrioWarning(Warning):
    pass


def test_archive_validation_contract(tmp_path: Path) -> None:
    valid_zip = tmp_path / "valid.zip"
    with zipfile.ZipFile(valid_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("data.txt", "payload")
    _assert_valid_zip(valid_zip, artifact_label="valid zip")

    invalid_zip = tmp_path / "invalid.zip"
    invalid_zip.write_text("not a zip", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _assert_valid_zip(invalid_zip, artifact_label="invalid zip")

    corrupted_zip = tmp_path / "corrupted.zip"
    with zipfile.ZipFile(corrupted_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("data.txt", "payload")
    with zipfile.ZipFile(corrupted_zip, "r") as archive:
        info = archive.getinfo("data.txt")
    payload = bytearray(corrupted_zip.read_bytes())
    filename_len = struct.unpack_from("<H", payload, info.header_offset + 26)[0]
    extra_len = struct.unpack_from("<H", payload, info.header_offset + 28)[0]
    data_offset = info.header_offset + 30 + filename_len + extra_len
    payload[data_offset] ^= 0x01
    corrupted_zip.write_bytes(payload)
    with pytest.raises(RuntimeError):
        _assert_valid_zip(corrupted_zip, artifact_label="corrupted zip")


def test_download_report_contracts_and_year_normalization() -> None:
    assert get_mrio_entry("exiobase_3102_ixi").shared_prereq_root == "exiobase_3"
    assert list_mrio_source_keys() == (
        "exiobase_396_ixi",
        "exiobase_396_pxp",
        "exiobase_3102_ixi",
        "exiobase_3102_pxp",
        "oecd_v2025",
    )
    assert normalize_mrio_years(None, source_key="exiobase_3102_ixi")[0] == 1995
    assert normalize_mrio_years(None, source_key="exiobase_3102_ixi")[-1] == 2024
    assert normalize_mrio_years(2001, source_key="oecd_v2025") == [2001]
    assert normalize_mrio_years(range(2000, 2002), source_key="oecd_v2025") == [
        2000,
        2001,
    ]
    assert normalize_mrio_years([2002, 2001, 2001], source_key="oecd_v2025") == [
        2001,
        2002,
    ]

    report = mrio_mod.DownloadReportMRIO(
        "dummy",
        requested=[2000, 2001, "alpha"],
        download_root=Path("C:/tmp/dummy"),
    )
    report.downloaded.extend([2000, 2001])
    report.skipped_already_saved.append(2002)
    report.errors["alpha"] = "bad"
    assert report.missing() == ["alpha"]
    text = str(report)
    assert text

    assert report._format_year_ranges([]) == "[]"
    assert report._format_year_ranges([2000, 2002, 2003]) == "2000, 2002-2003"
    assert report._format_year_ranges(["beta", "alpha"]) == "alpha, beta"

    with pytest.raises(
        ValueError,
    ):
        normalize_mrio_years(cast(Any, "2001"), source_key="oecd_v2025")

    minimal_text = str(mrio_mod.DownloadReportMRIO("dummy", requested=[2000], downloaded=[2000]))
    assert minimal_text
    assert _resolve_source_layout("oecd_v2025") == ("oecd_v2025", "full")
    assert is_exio_mrio_source("exiobase_3102_ixi") is True
    assert is_exio_mrio_source("oecd_v2025") is False
    assert default_years_for_source("oecd_v2025")[0] == 1995
    assert default_years_for_source("oecd_v2025")[-1] == 2022


def test_download_mrio_metadata_contracts_cover_missing_and_subset_logic(
    project_repo: Path,
) -> None:
    del project_repo
    assert metadata_mod._read_meta("oecd_v2025") is None

    metadata_mod._write_meta("oecd_v2025", [2005, 2004, 2005])
    payload = metadata_mod._read_meta("oecd_v2025")
    assert payload is not None
    assert payload["source"] == "oecd_v2025"
    assert payload["years"] == [2004, 2005]


def test_download_year_with_report_and_contract_pipeline(
    tmp_path: Path,
    project_repo: Path,
) -> None:
    del project_repo
    report = mrio_mod.DownloadReportMRIO("dummy", requested=[2000, 2001])
    handler = _DummyHandler(failing_years={2001})
    disk_years: set[int] = set()
    completed_download_units: set[str] = set()

    mrio_mod._download_year_with_report(
        handler=handler,
        full_dir=tmp_path,
        units=[2000, 2001],
        year=2000,
        report=report,
        disk_years=disk_years,
        completed_download_units=completed_download_units,
        refresh=False,
    )
    mrio_mod._download_year_with_report(
        handler=handler,
        full_dir=tmp_path,
        units=[2000, 2001],
        year=2001,
        report=report,
        disk_years=disk_years,
        completed_download_units=completed_download_units,
        refresh=False,
    )

    assert report.downloaded == [2000]
    assert report.errors == {2001: "cannot download 2001"}
    assert disk_years == {2000}
    assert completed_download_units == {"dummy:2000"}

    assert mrio_mod._coalesce_pending_download_years(
        handler=_BundledDummyHandler(),
        pending_years=[2003, 2004],
    ) == [2003]
    assert (
        mrio_mod._progress_label_for_unit(
            handler=_BundledDummyHandler(),
            year=2003,
        )
        == "bundle-2003"
    )

    metadata_mod._get_metadata_path("oecd_v2025").unlink(missing_ok=True)
    result = mrio_mod._download_mrio_with_handler(
        source_key="oecd_v2025",
        handler=_DummyHandler(),
        full_dir=tmp_path / "downloads",
        years=[2003, 2004],
        refresh=False,
    )
    assert result is not None
    assert result.downloaded == [2003, 2004]
    metadata = metadata_mod._read_meta("oecd_v2025")
    assert metadata is not None
    assert metadata["years"] == [2003, 2004]

    bundled_result = mrio_mod._download_mrio_with_handler(
        source_key="oecd_v2025",
        handler=_BundledDummyHandler(),
        full_dir=tmp_path / "bundled_downloads",
        years=[2005, 2006],
        refresh=False,
    )
    assert bundled_result is not None
    assert bundled_result.downloaded == [2005, 2006]
    metadata = metadata_mod._read_meta("oecd_v2025")
    assert metadata is not None
    assert metadata["years"] == [2003, 2004, 2005, 2006]


def test_download_mrio_with_handler_refresh_skip(
    tmp_path: Path,
    project_repo: Path,
) -> None:
    del project_repo
    metadata_path = metadata_mod._get_metadata_path("oecd_v2025")
    metadata_path.unlink(missing_ok=True)
    full_dir = tmp_path / "downloads"
    full_dir.mkdir(parents=True, exist_ok=True)
    (full_dir / "2000.bin").write_text("saved", encoding="utf-8")

    skipped = mrio_mod._download_mrio_with_handler(
        source_key="oecd_v2025",
        handler=_DummyHandler(),
        full_dir=full_dir,
        years=[2000],
        refresh=False,
    )
    assert skipped is None
    metadata = metadata_mod._read_meta("oecd_v2025")
    assert metadata is not None
    assert metadata["years"] == [2000]

    refreshed = mrio_mod._download_mrio_with_handler(
        source_key="oecd_v2025",
        handler=_DummyHandler(),
        full_dir=full_dir,
        years=[2000],
        refresh=True,
    )
    assert refreshed is not None
    assert refreshed.downloaded == [2000]


def test_download_mrio_merges_existing_metadata_and_uses_current_paths(project_repo: Path) -> None:
    full_dir = mrio_mod._get_full_dir("exiobase_3102_ixi")
    full_dir.mkdir(parents=True, exist_ok=True)
    archive = full_dir / "IOT_2001_ixi.zip"
    archive.write_text("saved", encoding="utf-8")

    assert (
        full_dir == project_repo / "data_raw" / "mrio" / "exiobase_3" / "exiobase_3102" / "full_ixi"
    )
    assert mrio_mod.download_mrio("exiobase_3102_ixi", years=2001, refresh=False) is None
    with pytest.raises(ValueError):
        mrio_mod.download_mrio("missing", years=2001)


def test_download_mrio_metadata_contracts_cover_roundtrip(
    project_repo: Path,
) -> None:
    del project_repo
    source_key = "oecd_v2025"
    metadata_path = metadata_mod._get_metadata_path(source_key)
    metadata_path.unlink(missing_ok=True)

    assert metadata_mod._read_meta(source_key) is None

    metadata_mod._write_meta(source_key, [2003, 2001, 2001])
    payload = metadata_mod._read_meta(source_key)
    assert payload is not None
    assert payload["source"] == source_key
    assert payload["years"] == [2001, 2003]


def test_exio_download_contracts_and_handler(tmp_path: Path) -> None:
    entry = get_mrio_entry("exiobase_3102_ixi")
    calls: list[tuple[str, str, list[int], str]] = []
    observed_headers: list[dict[str, str]] = []

    def downloader(*, storage_folder: str, system: str, years: list[int], doi: str) -> None:
        observed_headers.append(dict(exio_mod.pymrio_iodownloader.HEADERS))
        calls.append((storage_folder, system, years, doi))
        Path(storage_folder, f"IOT_{years[0]}_{system}.zip").write_text("archive", encoding="utf-8")

    target_dir = tmp_path / "exio"
    original_headers = dict(exio_mod.pymrio_iodownloader.HEADERS)
    exio_mod.pymrio_iodownloader.HEADERS.clear()
    exio_mod.pymrio_iodownloader.HEADERS.update({"User-Agent": "test-agent"})
    exio_mod._download_exiobase_year(
        target_dir,
        2000,
        system="ixi",
        doi=str(entry.doi),
        downloader=downloader,
        archive_validator=lambda *_args, **_kwargs: None,
    )
    assert observed_headers == [{}]
    assert exio_mod.pymrio_iodownloader.HEADERS == {"User-Agent": "test-agent"}
    exio_mod.pymrio_iodownloader.HEADERS.clear()
    exio_mod.pymrio_iodownloader.HEADERS.update(original_headers)
    assert calls == [
        (
            str(_get_exio_archive_temp_dir(target_dir, 2000, system="ixi")),
            "ixi",
            [2000],
            str(entry.doi),
        )
    ]
    assert (target_dir / "IOT_2000_ixi.zip").exists()
    assert not _get_exio_archive_temp_dir(target_dir, 2000, system="ixi").exists()

    missing_dir = tmp_path / "exio_missing"

    def downloader_without_archive(
        *,
        storage_folder: str,
        system: str,
        years: list[int],
        doi: str,
    ) -> None:
        calls.append((storage_folder, system, years, doi))

    with pytest.raises(RuntimeError):
        exio_mod._download_exiobase_year(
            missing_dir,
            2002,
            system="ixi",
            doi=str(entry.doi),
            downloader=downloader_without_archive,
            archive_validator=lambda *_args, **_kwargs: None,
        )

    stale_temp_dir = _get_exio_archive_temp_dir(target_dir, 2005, system="ixi")
    stale_temp_dir.mkdir(parents=True, exist_ok=True)
    _get_exio_archive_temp_path(target_dir, 2005, system="ixi").write_text(
        "partial",
        encoding="utf-8",
    )

    handler = exio_mod.ExiobaseMRIOHandler(
        key="exiobase_3102_ixi",
        system="ixi",
        doi=str(entry.doi),
    )
    assert handler.full_exists(target_dir, 2005) is False
    assert 2005 not in handler.list_existing_years(target_dir)
    report = mrio_mod.DownloadReportMRIO("exiobase_3102_ixi", requested=[2001])
    malformed_exio = target_dir / "IOT_bad_ixi.zip"
    malformed_exio.write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError):
        handler.list_existing_years(target_dir)
    malformed_exio.unlink()

    def download_func(target_dir: Path, year: int, *, system: str, doi: str) -> None:
        downloader(storage_folder=str(target_dir), system=system, years=[year], doi=doi)

    completed_download_units: set[str] = set()
    handler.download_year(
        year=2001,
        full_dir=target_dir,
        requested_years=[2001],
        report=report,
        completed_download_units=completed_download_units,
        download_func=download_func,
    )
    assert handler.full_exists(target_dir, 2001) is True
    assert handler.list_existing_years(target_dir) == {2000, 2001}
    assert _get_exio_archive_path(target_dir, 2000, system="ixi").name == "IOT_2000_ixi.zip"
    assert report.downloaded == [2001]
    assert completed_download_units == {"exiobase_3102_ixi:2001"}

    report_deduped = mrio_mod.DownloadReportMRIO("exiobase_3102_ixi", requested=[2001])
    handler.download_year(
        year=2001,
        full_dir=target_dir,
        requested_years=[2001],
        report=report_deduped,
        completed_download_units={"exiobase_3102_ixi:2001"},
        download_func=download_func,
    )
    assert report_deduped.downloaded == []

    report_preexisting = mrio_mod.DownloadReportMRIO("exiobase_3102_ixi", requested=[2001])
    handler.download_year(
        year=2001,
        full_dir=target_dir,
        requested_years=[2001],
        report=report_preexisting,
        completed_download_units=set(),
        download_func=download_func,
    )
    assert report_preexisting.downloaded == []

    report_refreshed = mrio_mod.DownloadReportMRIO("exiobase_3102_ixi", requested=[2001])
    handler.download_year(
        year=2001,
        full_dir=target_dir,
        requested_years=[2001],
        report=report_refreshed,
        completed_download_units=set(),
        refresh=True,
        download_func=download_func,
    )
    assert report_refreshed.downloaded == [2001]


def test_oecd_bundle_download_and_handler(tmp_path: Path) -> None:
    assert oecd_mod._year_to_oecd_bundle(1995) == "1995-2000"
    assert oecd_mod._oecd_bundle_years("1995-2000") == [1995, 1996, 1997, 1998, 1999, 2000]
    assert oecd_mod._coalesce_oecd_bundle_years([1995, 1996, 2001, 2002]) == [1995, 2001]
    with pytest.raises(ValueError):
        oecd_mod._year_to_oecd_bundle(1900)

    missing_output_dir = tmp_path / "oecd_missing"
    missing_output_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(RuntimeError):
        oecd_mod._validate_oecd_bundle_outputs(missing_output_dir, "1995-2000")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for year, value in zip(range(1995, 2001), "abcdef"):
            archive.writestr(f"{year}_SML.csv", f"{value},1\n")

    class _ChunkyResponse(_BytesResponse):
        def iter_content(self, chunk_size: int):
            del chunk_size
            yield b""
            yield self._payload

    response = _ChunkyResponse(buffer.getvalue())
    target_dir = tmp_path / "oecd"
    target_dir.mkdir(parents=True, exist_ok=True)
    existing_target = target_dir / "ICIO2025_1995.csv"
    existing_target.write_text("keep", encoding="utf-8")

    oecd_mod._download_oecd_bundle(
        target_dir,
        "1995-2000",
        request_get=lambda *_args, **_kwargs: response,
        urls_by_bundle={"1995-2000": "https://example/bundle.zip"},
    )
    assert existing_target.read_text(encoding="utf-8") == "keep"
    assert (target_dir / "ICIO2025_1996.csv").read_text(encoding="utf-8") == "b,1\n"
    assert not _get_oecd_bundle_temp_dir(target_dir, "1995-2000").exists()
    assert not _get_oecd_bundle_temp_zip_path(target_dir, "1995-2000").exists()
    assert oecd_mod._bundle_download_unit("1995-2000") == "oecd_bundle:1995-2000"

    refreshed_target_dir = tmp_path / "oecd_refresh"
    refreshed_target_dir.mkdir(parents=True, exist_ok=True)
    refreshed_existing = refreshed_target_dir / "ICIO2025_1995.csv"
    refreshed_existing.write_text("old", encoding="utf-8")
    oecd_mod._download_oecd_bundle(
        refreshed_target_dir,
        "1995-2000",
        request_get=lambda *_args, **_kwargs: response,
        urls_by_bundle={"1995-2000": "https://example/bundle.zip"},
        refresh=True,
    )
    assert refreshed_existing.read_text(encoding="utf-8") == "a,1\n"

    handler = oecd_mod.OECDMRIOHandler()
    handler_dir = tmp_path / "oecd_handler"
    handler_dir.mkdir(parents=True, exist_ok=True)
    malformed_oecd = handler_dir / "ICIO2025_bad.csv"
    malformed_oecd.write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError):
        handler.list_existing_years(handler_dir)
    malformed_oecd.unlink()
    stale_oecd_temp_dir = _get_oecd_bundle_temp_dir(handler_dir, "2001-2005")
    stale_oecd_temp_dir.mkdir(parents=True, exist_ok=True)
    _get_oecd_bundle_temp_zip_path(handler_dir, "2001-2005").write_text(
        "partial",
        encoding="utf-8",
    )
    bundle_calls: list[str] = []

    def bundle_downloader(
        full_dir: Path,
        bundle: str,
        refresh: bool = False,
    ) -> None:
        bundle_calls.append(bundle)
        assert isinstance(refresh, bool)
        for year in range(1995, 2001):
            (full_dir / f"ICIO2025_{year}.csv").write_text(str(year), encoding="utf-8")

    report = mrio_mod.DownloadReportMRIO("oecd_v2025", requested=[1995, 1996])
    completed_download_units: set[str] = set()
    handler.download_year(
        year=1995,
        full_dir=handler_dir,
        requested_years=[1995, 1996],
        report=report,
        completed_download_units=completed_download_units,
        bundle_downloader=bundle_downloader,
    )
    handler.download_year(
        year=1996,
        full_dir=handler_dir,
        requested_years=[1995, 1996],
        report=report,
        completed_download_units=completed_download_units,
        bundle_downloader=bundle_downloader,
    )
    assert bundle_calls == ["1995-2000"]
    assert _get_oecd_csv_path(handler_dir, 1995).name == "ICIO2025_1995.csv"
    assert handler.full_exists(handler_dir, 1995) is True
    assert handler.full_exists(handler_dir, 2001) is False
    assert handler.coalesce_pending_years([1995, 1996, 2001]) == [1995, 2001]
    assert handler.list_existing_years(handler_dir) == {1995, 1996, 1997, 1998, 1999, 2000}
    assert handler.progress_label(1996) == "1995-2000"
    assert report.downloaded == [1995, 1996]
    assert completed_download_units == {"oecd_bundle:1995-2000"}

    preexisting_report = mrio_mod.DownloadReportMRIO("oecd_v2025", requested=[1995])
    handler.download_year(
        year=1995,
        full_dir=handler_dir,
        requested_years=[1995],
        report=preexisting_report,
        completed_download_units=set(),
        bundle_downloader=bundle_downloader,
    )
    assert preexisting_report.downloaded == []

    refresh_report = mrio_mod.DownloadReportMRIO("oecd_v2025", requested=[1995, 1996])
    handler.download_year(
        year=1995,
        full_dir=handler_dir,
        requested_years=[1995, 1996],
        report=refresh_report,
        completed_download_units=set(),
        refresh=True,
        bundle_downloader=bundle_downloader,
    )
    assert refresh_report.downloaded == [1995, 1996]


def test_suppress_pymrio_logging_restores_previous_state() -> None:
    previous_disable = logging.root.manager.disable
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with suppress_pymrio_logging():
            assert logging.root.manager.disable == logging.CRITICAL
            warnings.warn_explicit(
                "Starting with pandas version 4.0 all arguments of sum will be keyword-only.",
                category=_ExternalPymrioWarning,
                filename="pymrio/tools/iomath.py",
                lineno=554,
                module="pymrio.tools.iomath",
            )
    assert logging.root.manager.disable == previous_disable
    assert caught == []
