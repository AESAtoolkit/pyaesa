import io
import json
import threading
import zipfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest
import requests

from tests.package.helpers.ar6_imports import (
    collection_download,
    collection_explorer,
    collection_historical,
    collection_iiasa,
    collection_config,
    collection_metadata,
    collection_overlay,
    collection_paths,
    collection_public_archive,
    collection_reports,
)
from tests.package.helpers.ar6_dummy_repo import AR6DummyRepo, build_ar6_explorer_frames

download_ar6 = collection_download.download_ar6
_ar6_public_explorer_metadata = collection_download._ar6_public_explorer_metadata
_download_signature = collection_download._download_signature
_record_historical_downloaded_assets = collection_download._record_historical_downloaded_assets
read_download_metadata = collection_metadata.read_download_metadata
require_metadata_for_existing_output = collection_metadata.require_metadata_for_existing_output
signature_matches = collection_metadata.signature_matches
write_download_metadata = collection_metadata.write_download_metadata
get_citation_txt_path = collection_paths.get_citation_txt_path
get_explorer_csv_path = collection_paths.get_explorer_csv_path
get_logs_dir = collection_paths.get_logs_dir
get_metadata_path = collection_paths.get_metadata_path
get_raw_dir = collection_paths.get_raw_dir
DownloadReportAR6 = collection_reports.DownloadReportAR6
ar6_historical_figure_reference = collection_overlay
historical_sources = collection_historical
_download_bytes = collection_iiasa._download_bytes
_flatten_runs_metadata = collection_iiasa._flatten_runs_metadata
_get_anonymous_headers = collection_iiasa._get_anonymous_headers
_list_public_files = collection_iiasa._list_public_files
_pick_consistent_value = collection_iiasa._pick_consistent_value
_load_ar6_public_archive_from_files = collection_iiasa._load_ar6_public_archive_from_files
_request_json = collection_iiasa._request_json
_resolve_legacy_application = collection_iiasa._resolve_legacy_application
_resolve_run_id_column = collection_iiasa._resolve_run_id_column
_select_public_file = collection_iiasa._select_public_file
download_iiasa_explorer_data = collection_iiasa.download_iiasa_explorer_data
download_iiasa_public_file = collection_iiasa.download_iiasa_public_file
load_ar6_public_archive_data = collection_public_archive.load_ar6_public_archive_data
ExplorerData = collection_explorer.ExplorerData
drop_non_persisted_columns = collection_explorer.drop_non_persisted_columns
read_explorer_csv = collection_explorer.read_explorer_csv
to_wide_with_meta = collection_explorer.to_wide_with_meta
write_explorer_csv = collection_explorer.write_explorer_csv
DEFAULT_META_COLUMNS = collection_config.DEFAULT_META_COLUMNS
DEFAULT_VARIABLES_RELEVANT = collection_config.DEFAULT_VARIABLES_RELEVANT


@dataclass
class FakeResponse:
    status_code: int
    url: str
    json_payload: Any | None = None
    text: str = ""
    headers: dict[str, str] | None = None
    content: bytes = b""

    def json(self) -> Any:
        return self.json_payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(self.text or f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 1024) -> list[bytes]:
        del chunk_size
        return [self.content]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb


class FakeSession:
    def __init__(self, routes: dict[tuple[str, str], list[FakeResponse]]) -> None:
        self._routes = {key: list(value) for key, value in routes.items()}

    def get(self, url: str, **kwargs) -> FakeResponse:
        del kwargs
        return self.request("GET", url)

    def request(self, method: str, url: str, **kwargs) -> FakeResponse:
        del kwargs
        key = (method.upper(), url)
        queue = self._routes[key]
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]


class _IIASATestHandler(BaseHTTPRequestHandler):
    token_calls = 0
    base_url = ""
    zip_bytes = b""
    csv_bytes = b""
    world_archive_bytes = b""
    historical_archive_bytes = b""
    include_world_archive = True
    include_historical_archive = True

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        del format, args

    def _write_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_bytes(self, content: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/legacy/anonym/":
            type(self).token_calls += 1
            self._write_json("token-123")
            return
        if path == "/legacy/applications":
            self._write_json(
                [
                    {"name": "app-name", "config": [{"path": "env", "value": "testdb"}]},
                    {
                        "name": "IXSE_AR6_PUBLIC",
                        "config": [{"path": "env", "value": "ar6-public"}],
                    },
                ]
            )
            return
        if path == "/legacy/applications/app-name/config":
            self._write_json([{"path": "baseUrl", "value": f"{self.base_url}/app"}])
            return
        if path == "/legacy/applications/IXSE_AR6_PUBLIC/config":
            self._write_json([{"path": "baseUrl", "value": f"{self.base_url}/public-files"}])
            return
        if path == "/app/runs":
            self._write_json(
                [
                    {
                        "id": 1,
                        "model": "M1",
                        "scenario": "S1",
                        "metadata": {"Category": "C1", "Ssp_family": 1},
                    },
                    {
                        "id": 2,
                        "model": "M2",
                        "scenario": "S2",
                        "metadata": {"Category": "C2", "Ssp_family": 2},
                    },
                ]
            )
            return
        if path == "/public-files/runs":
            self._write_json([])
            return
        if path == "/public-files/files":
            query = parse_qs(parsed.query)
            assert query["includePublic"] == ["true"]
            assert query["includePrivate"] == ["true"]
            files = [
                {
                    "filename": "test-data.zip",
                    "description": "download-description",
                    "contentType": "application/zip",
                    "fileSize": len(self.zip_bytes),
                    "createAt": 2,
                },
                {
                    "filename": "plain.csv",
                    "description": "plain-description",
                    "contentType": "text/csv",
                    "fileSize": len(self.csv_bytes),
                    "createAt": 1,
                },
            ]
            if self.include_world_archive:
                files.append(
                    {
                        "filename": "1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip",
                        "description": "AR6_Scenarios_Database_World_v1.1",
                        "contentType": "application/zip",
                        "fileSize": len(self.world_archive_bytes),
                        "createAt": 3,
                    }
                )
            if self.include_historical_archive:
                files.append(
                    {
                        "filename": "test-AR6_historical_emissions.zip",
                        "description": "AR6_historical_emissions",
                        "contentType": "application/zip",
                        "fileSize": len(self.historical_archive_bytes),
                        "createAt": 4,
                    }
                )
            self._write_json(files)
            return
        if path == "/public-files/files/test-data.zip":
            self._write_json({"directLink": f"{self.base_url}/downloads/test-data.zip"})
            return
        if path == "/public-files/files/plain.csv":
            self._write_json({"directLink": f"{self.base_url}/downloads/plain.csv"})
            return
        if path == "/public-files/files/1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip":
            self._write_json(
                {
                    "directLink": (
                        f"{self.base_url}/downloads/"
                        "1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip"
                    )
                }
            )
            return
        if path == "/public-files/files/test-AR6_historical_emissions.zip":
            self._write_json(
                {"directLink": f"{self.base_url}/downloads/test-AR6_historical_emissions.zip"}
            )
            return
        if path == "/downloads/test-data.zip":
            self._write_bytes(self.zip_bytes, "application/zip")
            return
        if path == "/downloads/plain.csv":
            self._write_bytes(self.csv_bytes, "text/csv")
            return
        if path == "/downloads/1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip":
            self._write_bytes(self.world_archive_bytes, "application/zip")
            return
        if path == "/downloads/test-AR6_historical_emissions.zip":
            self._write_bytes(self.historical_archive_bytes, "application/zip")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/app/runs/bulk/ts":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        assert payload["filters"]["regions"] == ["World"]
        assert payload["filters"]["variables"] == ["Emissions|CO2"]
        self._write_json(
            [
                {
                    "runId": 1,
                    "variable": "Emissions|CO2",
                    "unit": "MtCO2/yr",
                    "region": "World",
                    "year": 2020,
                    "value": 10.0,
                    "subannual": "Year",
                },
                {
                    "runId": 2,
                    "variable": "Emissions|CO2",
                    "unit": "MtCO2/yr",
                    "region": "World",
                    "year": 2020,
                    "value": 12.0,
                    "subannual": -1,
                },
            ]
        )


class IIASALocalServer:
    def __init__(
        self,
        *,
        include_world_archive: bool = True,
        include_historical_archive: bool = True,
    ) -> None:
        archive_buffer = io.BytesIO()
        with zipfile.ZipFile(archive_buffer, mode="w") as archive:
            archive.writestr("member.csv", "a,b\n1,2\n")
        world_archive_buffer = io.BytesIO()
        world_timeseries_rows = []
        for variable_index, variable in enumerate(DEFAULT_VARIABLES_RELEVANT):
            world_timeseries_rows.append(
                {
                    "Model": "M1",
                    "Scenario": "S1",
                    "Region": "World",
                    "Variable": variable,
                    "Unit": "MtCO2eq/yr" if "Kyoto Gases" in variable else "MtCO2/yr",
                    "2019": 9.0 + variable_index,
                    "2020": 10.0 + variable_index,
                    "2021": 11.0 + variable_index,
                }
            )
        world_timeseries_rows.append(
            {
                "Model": "M2",
                "Scenario": "S2",
                "Region": "Europe",
                "Variable": "Emissions|CO2",
                "Unit": "MtCO2/yr",
                "2019": 1.0,
                "2020": 2.0,
                "2021": 3.0,
            }
        )
        world_timeseries = pd.DataFrame(world_timeseries_rows)
        meta_row: dict[str, object] = {column: pd.NA for column in DEFAULT_META_COLUMNS}
        meta_row.update(
            {
                "Model": "M1",
                "Scenario": "S1",
                "Category": "C1",
                "Category_name": "Category 1",
                "Category_subset": "subset",
                "Ssp_family": 1,
                "Vetting_historical": "Pass",
                "Vetting_future": "Pass",
                "Time horizon": 2100,
                "Peak Emissions|CO2": 10.0,
            }
        )
        world_metadata = pd.DataFrame([meta_row])
        with zipfile.ZipFile(world_archive_buffer, mode="w") as archive:
            archive.writestr(
                "AR6_Scenarios_Database_World_v1.1.csv",
                world_timeseries.to_csv(index=False),
            )
            metadata_buffer = io.BytesIO()
            with pd.ExcelWriter(metadata_buffer, engine="xlsxwriter") as writer:
                world_metadata.to_excel(writer, sheet_name="meta", index=False)
            archive.writestr(
                "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx",
                metadata_buffer.getvalue(),
            )
        historical_archive_buffer = io.BytesIO()
        with zipfile.ZipFile(historical_archive_buffer, mode="w") as archive:
            archive.writestr(
                "AR6_historical_emissions.csv",
                "Model,Scenario,Region,Variable,Unit,2020\nEDGAR,historical,World,Emissions|CO2,"
                "Gt CO2/yr,1\n",
            )
        self._handler = _IIASATestHandler
        self._handler.zip_bytes = archive_buffer.getvalue()
        self._handler.csv_bytes = b"col\nvalue\n"
        self._handler.world_archive_bytes = world_archive_buffer.getvalue()
        self._handler.historical_archive_bytes = historical_archive_buffer.getvalue()
        self._handler.include_world_archive = include_world_archive
        self._handler.include_historical_archive = include_historical_archive
        self._handler.token_calls = 0
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._handler.base_url = self.base_url
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "IIASALocalServer":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)


def test_ar6_collection_exports_paths_metadata_and_reports(
    project_repo: Path,
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    del project_repo
    assert callable(download_ar6)
    assert get_raw_dir() == ar6_dummy_repo.raw_dir
    assert get_logs_dir() == ar6_dummy_repo.raw_logs_dir
    assert get_metadata_path() == ar6_dummy_repo.metadata_path
    assert get_citation_txt_path() == ar6_dummy_repo.citation_txt_path
    assert get_explorer_csv_path("demo-db").name == "demo_db_explorer.csv"

    signature = _download_signature()
    assert signature["variables"] == DEFAULT_VARIABLES_RELEVANT
    assert {"Vetting_historical", "Vetting_future"}.issubset(set(signature["meta_columns"]))
    metadata_path = write_download_metadata({"signature": signature, "function": "download_ar6"})
    assert metadata_path == ar6_dummy_repo.metadata_path
    loaded_metadata = read_download_metadata()
    assert loaded_metadata is not None
    assert signature_matches(loaded_metadata, signature) is True
    assert signature_matches(loaded_metadata, {"signature": "other"}) is False
    assert signature_matches(None, signature) is False

    require_metadata_for_existing_output(
        metadata=loaded_metadata, paths=[ar6_dummy_repo.explorer_csv_path]
    )
    require_metadata_for_existing_output(metadata=None, paths=[ar6_dummy_repo.raw_dir / "missing"])
    ar6_dummy_repo.metadata_path.unlink(missing_ok=True)
    with pytest.raises(RuntimeError):
        require_metadata_for_existing_output(
            metadata=None,
            paths=[ar6_dummy_repo.explorer_csv_path],
        )
    assert read_download_metadata() is None

    report = DownloadReportAR6(
        database="ar6-public",
        raw_root=ar6_dummy_repo.raw_dir,
        logs_dir=ar6_dummy_repo.raw_logs_dir,
        metadata_path=ar6_dummy_repo.metadata_path,
    )
    assert str(report)
    report.errors["download_ar6"] = "boom"
    assert str(report)

    asset_report = DownloadReportAR6(
        database="ar6-public",
        raw_root=ar6_dummy_repo.raw_dir,
        logs_dir=ar6_dummy_repo.raw_logs_dir,
        metadata_path=ar6_dummy_repo.metadata_path,
    )
    assert (
        _record_historical_downloaded_assets(
            asset_report,
            {
                "primap": {"final_file": "x"},
                "gcp": {"file": "x"},
                "ar6_historical_figure_reference": {"file": "x"},
            },
        )
        is True
    )
    assert asset_report.downloaded_assets == [
        "primap_hist_final.csv",
        "primap_hist_final_no_rounding.csv",
        "gcp_national_fossil.xlsx",
        "ar6_historical_figure_reference.csv",
    ]
    empty_asset_report = DownloadReportAR6(
        database="ar6-public",
        raw_root=ar6_dummy_repo.raw_dir,
        logs_dir=ar6_dummy_repo.raw_logs_dir,
        metadata_path=ar6_dummy_repo.metadata_path,
    )
    assert _record_historical_downloaded_assets(empty_asset_report, {}) is False


def test_explorer_csv_roundtrip_and_overlay_validation(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    long_df, meta_df = build_ar6_explorer_frames()
    wide_df = to_wide_with_meta(long_df, meta_df.assign(Category_definition="remove me"))
    assert "Category_definition" not in wide_df.columns
    assert "2020" in wide_df.columns
    csv_path = tmp_path / "explorer.csv"
    write_explorer_csv(csv_file=csv_path, data_df=long_df, meta_df=meta_df)
    loaded = read_explorer_csv(csv_path)
    assert isinstance(loaded, ExplorerData)
    assert "Category_definition" not in loaded.data.columns
    normalized = drop_non_persisted_columns(meta_df.assign(Category_definition="drop me"))
    assert "Category_definition" not in normalized.columns
    normalized_wide = drop_non_persisted_columns(wide_df.assign(Category_definition="drop me"))
    assert "Category_definition" not in normalized_wide.columns

    overlay_df = ar6_historical_figure_reference.read_ar6_historical_figure_reference(
        ar6_dummy_repo.overlay_path
    )
    assert not overlay_df.empty

    bad_overlay = tmp_path / "bad_overlay.csv"
    pd.DataFrame({"Model": ["EDGAR"]}).to_csv(bad_overlay, index=False)
    with pytest.raises(RuntimeError):
        ar6_historical_figure_reference.read_ar6_historical_figure_reference(bad_overlay)


def test_download_iiasa_internal_contracts() -> None:
    session = FakeSession(
        {
            ("GET", "http://manager/legacy/anonym/"): [
                FakeResponse(HTTPStatus.OK, "http://manager/legacy/anonym/", "token")
            ],
            ("GET", "http://api/data"): [
                FakeResponse(HTTPStatus.OK, "http://api/data", {"ok": True})
            ],
            ("GET", "http://api/retry"): [
                FakeResponse(HTTPStatus.UNAUTHORIZED, "http://api/retry", {"detail": "retry"}),
                FakeResponse(HTTPStatus.OK, "http://api/retry", {"ok": "retry"}),
            ],
            ("GET", "http://manager/legacy/applications"): [
                FakeResponse(
                    HTTPStatus.OK,
                    "http://manager/legacy/applications",
                    [
                        {"name": "resolved", "config": [{"path": "env", "value": "alias"}]},
                        {"name": "direct-name", "config": []},
                    ],
                )
            ],
            ("GET", "http://manager/legacy/applications/resolved/config"): [
                FakeResponse(
                    HTTPStatus.OK,
                    "http://manager/legacy/applications/resolved/config",
                    [{"path": "baseUrl", "value": "http://base/app"}],
                )
            ],
            ("GET", "http://base/files"): [
                FakeResponse(HTTPStatus.OK, "http://base/files", [{"filename": "a.csv"}])
            ],
        }
    )

    headers = _get_anonymous_headers(session, manager_url="http://manager")
    assert headers == {"Authorization": "Bearer token"}

    result, returned_headers = _request_json(
        session,
        "GET",
        "http://api/data",
        headers,
    )
    assert result == {"ok": True}
    assert returned_headers == headers

    retried, retried_headers = _request_json(
        session,
        "GET",
        "http://api/retry",
        headers,
        auth_header_loader=lambda current_session: _get_anonymous_headers(
            current_session,
            manager_url="http://manager",
        ),
    )
    assert retried == {"ok": "retry"}
    assert retried_headers == {"Authorization": "Bearer token"}

    base_url, _headers = _resolve_legacy_application(
        session,
        headers,
        "alias",
        manager_url="http://manager",
    )
    assert base_url == "http://base/app"

    files, _headers = _list_public_files(session, headers, "http://base")
    assert files == [{"filename": "a.csv"}]
    assert (
        _select_public_file(
            [
                {"filename": "old.csv", "description": "plain", "createAt": 1},
                {"filename": "new.csv", "description": "plain", "createAt": 2},
            ],
            filename_suffix=".csv",
            description="plain",
        )["filename"]
        == "new.csv"
    )
    with pytest.raises(RuntimeError):
        _select_public_file([], filename_suffix=".csv", description=None)

    assert _flatten_runs_metadata(pd.DataFrame({"run_id": [1]})).equals(
        pd.DataFrame({"run_id": [1]})
    )
    assert _flatten_runs_metadata(pd.DataFrame({"metadata": [pd.NA], "run_id": [1]})).equals(
        pd.DataFrame({"run_id": [1]})
    )
    flattened = _flatten_runs_metadata(
        pd.DataFrame({"run_id": [1], "metadata": [{"Category": "C1"}]})
    )
    assert flattened.loc[0, "Category"] == "C1"

    assert _resolve_run_id_column(pd.DataFrame({"run_id": [1]})) == "run_id"
    assert _resolve_run_id_column(pd.DataFrame({"runId": [1]})) == "runId"
    assert _resolve_run_id_column(pd.DataFrame({"id": [1]})) == "id"
    with pytest.raises(RuntimeError):
        _resolve_run_id_column(pd.DataFrame({"other": [1]}))

    assert _pick_consistent_value(pd.Series([pd.NA, None], dtype=object)) is pd.NA
    assert _pick_consistent_value(pd.Series(["one", "one"], dtype=object)) == "one"
    with pytest.raises(RuntimeError):
        _pick_consistent_value(pd.Series(["one", "two"], dtype=object))

    missing_link_session = FakeSession(
        {
            ("GET", "http://base/files"): [
                FakeResponse(
                    HTTPStatus.OK,
                    "http://base/files",
                    [
                        {
                            "filename": "1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip",
                            "description": "AR6_Scenarios_Database_World_v1.1",
                            "createAt": 1,
                        }
                    ],
                )
            ],
            ("GET", "http://base/files/1668008312256-AR6_Scenarios_Database_World_v1.1.csv.zip"): [
                FakeResponse(HTTPStatus.OK, "http://base/files/world.zip", {"notDirectLink": "x"})
            ],
        }
    )
    with pytest.raises(RuntimeError):
        _load_ar6_public_archive_from_files(
            session=missing_link_session,
            headers=headers,
            base_url="http://base",
            variables=["Emissions|CO2"],
            region="World",
            meta_columns=["Category"],
            auth_header_loader=lambda current_session: _get_anonymous_headers(
                current_session,
                manager_url="http://manager",
            ),
        )


def test_download_iiasa_public_functions_use_local_server(tmp_path: Path) -> None:
    with IIASALocalServer() as server:
        data_df, meta_df = download_iiasa_explorer_data(
            database="testdb",
            variables=["Emissions|CO2"],
            region="World",
            meta_columns=["Category", "Ssp_family", "MissingColumn"],
            manager_url=server.base_url,
        )
        assert sorted(data_df["scenario"].unique().tolist()) == ["S1", "S2"]
        assert meta_df.loc[("M1", "S1"), "Category"] == "C1"
        assert pd.isna(meta_df.loc[("M1", "S1"), "MissingColumn"])

        ar6_public_data_df, ar6_public_meta_df = download_iiasa_explorer_data(
            database="ar6-public",
            variables=["Emissions|CO2"],
            region="World",
            meta_columns=["Category", "Ssp_family", "MissingColumn"],
            manager_url=server.base_url,
        )
        assert ar6_public_data_df["scenario"].unique().tolist() == ["S1"]
        assert set(ar6_public_data_df["variable"]) == {"Emissions|CO2"}
        assert sorted(ar6_public_data_df["year"].unique().tolist()) == [2019, 2020, 2021]
        assert ar6_public_meta_df.loc[("M1", "S1"), "Category"] == "C1"
        assert ar6_public_meta_df.loc[("M1", "S1"), "Ssp_family"] == 1
        assert pd.isna(ar6_public_meta_df.loc[("M1", "S1"), "MissingColumn"])

        zip_target = tmp_path / "downloaded" / "member.csv"
        zip_result = download_iiasa_public_file(
            database="IXSE_AR6_PUBLIC",
            target_path=zip_target,
            filename_suffix=".zip",
            description="download-description",
            archive_member_name="member.csv",
            manager_url=server.base_url,
        )
        assert zip_target.read_text(encoding="utf-8").startswith("a,b")
        assert zip_result["source_filename"] == "test-data.zip"

        plain_target = tmp_path / "downloaded" / "plain.csv"
        plain_result = download_iiasa_public_file(
            database="IXSE_AR6_PUBLIC",
            target_path=plain_target,
            filename_suffix=".csv",
            description="plain-description",
            manager_url=server.base_url,
            download_bytes_func=_download_bytes,
        )
        assert plain_target.read_text(encoding="utf-8").startswith("col")
        assert plain_result["source_filename"] == "plain.csv"

        with pytest.raises(RuntimeError):
            download_iiasa_public_file(
                database="IXSE_AR6_PUBLIC",
                target_path=tmp_path / "missing.csv",
                filename_suffix=".zip",
                description="download-description",
                archive_member_name="missing.csv",
                manager_url=server.base_url,
            )


def test_ar6_public_archive_parser_edge_branches() -> None:
    missing_member_buffer = io.BytesIO()
    with zipfile.ZipFile(missing_member_buffer, mode="w") as archive:
        archive.writestr("other.csv", "a,b\n1,2\n")
    with pytest.raises(RuntimeError):
        load_ar6_public_archive_data(
            missing_member_buffer.getvalue(),
            variables=["Emissions|CO2"],
            region="World",
            meta_columns=["Category"],
        )

    no_match_buffer = io.BytesIO()
    rows = []
    for row_index in range(5001):
        rows.append(
            {
                "Model": f"M{row_index}",
                "Scenario": f"S{row_index}",
                "Region": "Europe",
                "Variable": "Emissions|CO2",
                "Unit": "MtCO2/yr",
                "2020": float(row_index),
            }
        )
    rows.append(
        {
            "Model": "M-final",
            "Scenario": "S-final",
            "Region": "World",
            "Variable": "Emissions|CH4",
            "Unit": "MtCH4/yr",
            "2020": 1.0,
        }
    )
    with zipfile.ZipFile(no_match_buffer, mode="w") as archive:
        archive.writestr(
            "AR6_Scenarios_Database_World_v1.1.csv",
            pd.DataFrame(rows).to_csv(index=False),
        )
        metadata_buffer = io.BytesIO()
        with pd.ExcelWriter(metadata_buffer, engine="xlsxwriter") as writer:
            pd.DataFrame([{"Model": "M-final", "Scenario": "S-final"}]).to_excel(
                writer,
                sheet_name="meta",
                index=False,
            )
        archive.writestr(
            "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx",
            metadata_buffer.getvalue(),
        )
    with pytest.raises(RuntimeError):
        load_ar6_public_archive_data(
            no_match_buffer.getvalue(),
            variables=["Emissions|CO2"],
            region="World",
            meta_columns=["Category"],
        )

    missing_identity_buffer = io.BytesIO()
    with zipfile.ZipFile(missing_identity_buffer, mode="w") as archive:
        archive.writestr(
            "AR6_Scenarios_Database_World_v1.1.csv",
            pd.DataFrame(
                [
                    {
                        "Model": "M1",
                        "Scenario": "S1",
                        "Region": "World",
                        "Variable": "Emissions|CO2",
                        "Unit": "MtCO2/yr",
                        "2020": 10.0,
                    }
                ]
            ).to_csv(index=False),
        )
        metadata_buffer = io.BytesIO()
        with pd.ExcelWriter(metadata_buffer, engine="xlsxwriter") as writer:
            pd.DataFrame([{"Model": "M1"}]).to_excel(writer, sheet_name="meta", index=False)
        archive.writestr(
            "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx",
            metadata_buffer.getvalue(),
        )
    with pytest.raises(RuntimeError):
        load_ar6_public_archive_data(
            missing_identity_buffer.getvalue(),
            variables=["Emissions|CO2"],
            region="World",
            meta_columns=["Category"],
        )


def test_historical_sources_contracts_and_ensure_historical_sources(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    del ar6_dummy_repo
    assert historical_sources._version_tuple("1.2.3") == (1, 2, 3)
    assert historical_sources._parse_primap_sort_key("file_v2.1_demo.csv")[0] == (2, 1)

    fake_response = FakeResponse(
        HTTPStatus.OK,
        "http://download/test.csv",
        headers={"Content-Disposition": 'attachment; filename="remote.csv"'},
    )
    assert (
        historical_sources._extract_filename_from_response(fake_response, "fallback.csv")
        == "remote.csv"
    )
    fake_response.headers = {}
    fake_response.url = "http://download/path/final.csv"
    assert (
        historical_sources._extract_filename_from_response(fake_response, "fallback.csv")
        == "final.csv"
    )
    fake_response.url = "http://download/"
    assert (
        historical_sources._extract_filename_from_response(fake_response, "fallback.csv")
        == "fallback.csv"
    )

    citation = historical_sources._record_to_citation(
        {
            "id": 1,
            "conceptrecid": 2,
            "metadata": {
                "title": "Title",
                "doi": "10.1/demo",
                "publication_date": "2025-01-01",
                "creators": [{"name": "Alice"}, {"name": "Bob"}],
            },
        }
    )
    assert citation["creators"] == ["Alice", "Bob"]
    assert "doi:10.1/demo" in historical_sources._primap_recommended_citation_from_record(
        {
            "metadata": {
                "title": "Title",
                "doi": "10.1/demo",
                "publication_date": "2025-01-01",
                "creators": [{"name": "Alice"}],
            }
        }
    )
    no_doi_text = historical_sources._primap_recommended_citation_from_record(
        {
            "metadata": {
                "title": "Title",
                "publication_date": "2025-01-01",
                "creators": [{"name": "Alice"}],
            }
        }
    )
    assert no_doi_text
    citations_txt = historical_sources.historical_citations_txt(
        {
            "primap_hist": {"recommended_citation": "primap citation"},
            "gcp_national_fossil": {"recommended_citation": "gcp citation"},
            "ar6_historical_figure_reference": {
                "recommended_citation": "edgar citation",
                "secondary_citation": "rcmip citation",
            },
        }
    )
    assert "primap citation" in citations_txt
    non_url_lines = [line for line in citations_txt.splitlines() if "http" not in line]
    assert all(len(line) <= 100 for line in non_url_lines)

    record = {
        "files": [
            {
                "key": "foo-PRIMAP-hist_v1.2_final_01-Jan-2024.csv",
                "links": {"self": "http://x/final"},
            },
            {
                "key": "foo-PRIMAP-hist_v1.2_final_no_rounding_01-Jan-2024.csv",
                "links": {"self": "http://x/final-no-rounding"},
            },
        ]
    }
    selected_final, selected_no_rounding = historical_sources._select_primap_files(record)
    assert selected_final["links"]["self"] == "http://x/final"
    assert selected_no_rounding["links"]["self"] == "http://x/final-no-rounding"
    with pytest.raises(RuntimeError):
        historical_sources._select_primap_files({"files": []})

    raw_dir = tmp_path / "raw"
    primap_payload = {
        "id": 1,
        "conceptrecid": 2,
        "metadata": {
            "title": "PRIMAP title",
            "doi": "10.1/demo",
            "publication_date": "2025-01-01",
            "creators": [{"name": "Alice"}],
        },
    }
    primap_payload.update(record)

    def fake_download_binary(
        url: str, target_path: Path, force_target_name: bool = False
    ) -> dict[str, object]:
        del url, force_target_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("binary payload", encoding="utf-8")
        return {"path": str(target_path), "filename": target_path.name, "url": "http://example"}

    result = historical_sources.ensure_historical_sources(
        raw_dir=raw_dir,
        refresh=True,
        fetch_latest_primap_record_func=lambda: primap_payload,
        select_primap_files_func=historical_sources._select_primap_files,
        download_binary_func=fake_download_binary,
        fetch_latest_gcp_link_func=lambda: ("http://gcp/file.xlsx", "GCP label", 2024),
        extract_gcp_recommended_citation_func=lambda: "GCP citation",
        download_iiasa_public_file_func=lambda **kwargs: {
            "path": str(kwargs["target_path"]),
            "filename": kwargs["target_path"].name,
        },
    )
    assert result["primap"] is not None
    assert result["gcp"] is not None
    assert result["ar6_historical_figure_reference"] is not None
    assert get_citation_txt_path().exists()

    reused = historical_sources.ensure_historical_sources(raw_dir=raw_dir, refresh=False)
    assert reused["primap"] is None
    assert reused["used_local_primap"] is True


def test_download_ar6_runner_and_public_wrapper(
    project_repo: Path,
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    del project_repo
    ar6_dummy_repo.explorer_csv_path.unlink(missing_ok=True)
    ar6_dummy_repo.citation_txt_path.unlink(missing_ok=True)
    ar6_dummy_repo.metadata_path.unlink(missing_ok=True)

    with IIASALocalServer() as server:
        fresh_report = download_ar6(refresh=False, manager_url=server.base_url)
    assert fresh_report is not None
    assert ar6_dummy_repo.explorer_csv_path.exists()
    assert read_download_metadata() is not None
    assert fresh_report.downloaded_assets == [ar6_dummy_repo.explorer_csv_path.name]

    reused_none = download_ar6(refresh=False)
    assert reused_none is None
    assert download_ar6(refresh=False) is None

    ar6_dummy_repo.overlay_path.unlink(missing_ok=True)
    ar6_dummy_repo.citation_txt_path.unlink(missing_ok=True)
    with IIASALocalServer() as server:
        reused_report = download_ar6(refresh=False, manager_url=server.base_url)
    assert reused_report is not None
    assert reused_report.downloaded_assets == ["ar6_historical_figure_reference.csv"]
    assert download_ar6(refresh=False) is None

    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)
    assert not explorer.data.empty
    assert set(explorer.data["variable"]) == set(DEFAULT_VARIABLES_RELEVANT)
    assert {"Vetting_historical", "Vetting_future"}.issubset(set(explorer.data.columns))

    ar6_dummy_repo.explorer_csv_path.write_text("stale", encoding="utf-8")
    ar6_dummy_repo.citation_txt_path.write_text("stale", encoding="utf-8")
    write_download_metadata({"signature": _download_signature()})
    with IIASALocalServer(include_world_archive=False) as server:
        error_report = download_ar6(refresh=True, manager_url=server.base_url)
    assert error_report is not None
    assert "download_ar6" in error_report.errors
    assert ar6_dummy_repo.explorer_csv_path.exists() is False
    assert ar6_dummy_repo.citation_txt_path.exists() is False
    assert ar6_dummy_repo.metadata_path.exists() is False

    ar6_dummy_repo.explorer_csv_path.write_text("model,scenario\n", encoding="utf-8")
    ar6_dummy_repo.metadata_path.unlink(missing_ok=True)
    with pytest.raises(RuntimeError):
        download_ar6(refresh=False)

    assert _ar6_public_explorer_metadata()["source"].startswith("AR6 Scenarios Database")
