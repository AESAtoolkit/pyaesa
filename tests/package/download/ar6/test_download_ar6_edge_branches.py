import json
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import pytest
import requests

from tests.package.helpers.ar6_imports import (
    collection_download,
    collection_config,
    collection_historical,
    collection_iiasa,
    collection_metadata,
    collection_overlay,
    collection_paths,
    collection_reports,
)
from tests.package.helpers.ar6_dummy_repo import AR6DummyRepo

_record_historical_downloaded_assets = collection_download._record_historical_downloaded_assets
require_metadata_for_existing_output = collection_metadata.require_metadata_for_existing_output
ar6_historical_figure_reference = collection_overlay
historical_sources = collection_historical
_get_anonymous_headers = collection_iiasa._get_anonymous_headers
_list_public_files = collection_iiasa._list_public_files
_request_json = collection_iiasa._request_json
_resolve_legacy_application = collection_iiasa._resolve_legacy_application
_select_public_file = collection_iiasa._select_public_file
download_iiasa_explorer_data = collection_iiasa.download_iiasa_explorer_data
download_iiasa_public_file = collection_iiasa.download_iiasa_public_file
citation_txt_path_for_raw_dir = collection_paths.citation_txt_path_for_raw_dir
clear_download_output_scope = collection_paths.clear_download_output_scope
DownloadReportAR6 = collection_reports.DownloadReportAR6
DEFAULT_DATABASE = collection_config.DEFAULT_DATABASE


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


class ChunkedResponse(FakeResponse):
    def iter_content(self, chunk_size: int = 1024) -> list[bytes]:
        del chunk_size
        return [b"", self.content]


class _IIASAVariantHandler(BaseHTTPRequestHandler):
    base_url = ""
    runs_payload: Any = []
    ts_payload: Any = []
    files_payload: Any = []
    link_payloads: dict[str, Any] = {}

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        del format, args

    def _write_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/legacy/anonym/":
            self._write_json("token")
            return
        if path == "/legacy/applications":
            self._write_json([{"name": "app-name", "config": [{"path": "env", "value": "db"}]}])
            return
        if path == "/legacy/applications/app-name/config":
            self._write_json([{"path": "baseUrl", "value": f"{self.base_url}/app"}])
            return
        if path == "/app/runs":
            self._write_json(type(self).runs_payload)
            return
        if path.startswith("/app/files/") or path.startswith("/public-files/files/"):
            filename = path.split("/")[-1]
            self._write_json(type(self).link_payloads.get(filename, {}))
            return
        if path.startswith("/app/files") or path.startswith("/public-files/files"):
            self._write_json(type(self).files_payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path == "/app/runs/bulk/ts":
            length = int(self.headers["Content-Length"])
            self.rfile.read(length)
            self._write_json(type(self).ts_payload)
            return
        self.send_error(HTTPStatus.NOT_FOUND)


class IIASAVariantServer:
    def __init__(
        self,
        *,
        runs_payload: Any,
        ts_payload: Any,
        files_payload: Any | None = None,
        link_payloads: dict[str, Any] | None = None,
    ) -> None:
        handler = _IIASAVariantHandler
        handler.runs_payload = runs_payload
        handler.ts_payload = ts_payload
        handler.files_payload = [] if files_payload is None else files_payload
        handler.link_payloads = {} if link_payloads is None else link_payloads
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        handler.base_url = self.base_url
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "IIASAVariantServer":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)


def test_ar6_collection_metadata_overlay_and_download_runner_edge_branches(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    require_metadata_for_existing_output(metadata=None, paths=[tmp_path / "missing.csv"])

    bad_overlay = tmp_path / "duplicate_overlay.csv"
    overlay_df = pd.read_csv(ar6_dummy_repo.overlay_path)
    pd.concat([overlay_df, overlay_df.iloc[[0]]], ignore_index=True).to_csv(
        bad_overlay, index=False
    )
    with pytest.raises(RuntimeError):
        ar6_historical_figure_reference.read_ar6_historical_figure_reference(bad_overlay)

    report = DownloadReportAR6(
        database=DEFAULT_DATABASE,
        raw_root=ar6_dummy_repo.raw_dir,
        logs_dir=ar6_dummy_repo.raw_logs_dir,
        metadata_path=ar6_dummy_repo.metadata_path,
    )
    assert (
        _record_historical_downloaded_assets(
            report,
            {
                "primap": None,
                "gcp": {"file": "gcp"},
                "ar6_historical_figure_reference": None,
            },
        )
        is True
    )
    assert report.downloaded_assets == ["gcp_national_fossil.xlsx"]

    ar6_dummy_repo.metadata_path.write_text("{}", encoding="utf-8")
    ar6_dummy_repo.citation_txt_path.write_text("old", encoding="utf-8")
    clear_download_output_scope(DEFAULT_DATABASE)
    assert ar6_dummy_repo.explorer_csv_path.exists() is False
    assert ar6_dummy_repo.citation_txt_path.exists() is False
    assert ar6_dummy_repo.metadata_path.exists() is False


def test_ar6_collection_download_iiasa_edge_branches(tmp_path: Path) -> None:
    invalid_token_session = FakeSession(
        {("GET", "http://manager/legacy/anonym/"): [FakeResponse(HTTPStatus.OK, "u", "")]}
    )
    with pytest.raises(RuntimeError):
        _get_anonymous_headers(invalid_token_session, manager_url="http://manager")

    retry_session = FakeSession(
        {
            ("GET", "http://manager/legacy/anonym/"): [
                FakeResponse(HTTPStatus.OK, "u", "token"),
            ],
            ("POST", "http://api/retry"): [
                FakeResponse(HTTPStatus.UNAUTHORIZED, "http://api/retry", {"detail": "retry"}),
                FakeResponse(HTTPStatus.OK, "http://api/retry", {"ok": True}),
            ],
        }
    )
    payload, headers = _request_json(
        retry_session,
        "POST",
        "http://api/retry",
        {"Authorization": "Bearer old"},
        json_body={"demo": True},
        auth_header_loader=lambda session: _get_anonymous_headers(
            session, manager_url="http://manager"
        ),
    )
    assert payload == {"ok": True}
    assert headers == {"Authorization": "Bearer token"}

    unresolved_session = FakeSession(
        {
            ("GET", "http://manager/legacy/applications"): [
                FakeResponse(HTTPStatus.OK, "u", [{"name": "other", "config": []}])
            ]
        }
    )
    with pytest.raises(RuntimeError):
        _resolve_legacy_application(
            unresolved_session,
            {"Authorization": "Bearer token"},
            "missing",
            manager_url="http://manager",
        )

    nobase_session = FakeSession(
        {
            ("GET", "http://manager/legacy/applications"): [
                FakeResponse(
                    HTTPStatus.OK,
                    "u",
                    [{"name": "app-name", "config": [{"path": "env", "value": "db"}]}],
                )
            ],
            ("GET", "http://manager/legacy/applications/app-name/config"): [
                FakeResponse(HTTPStatus.OK, "u", [{"path": "other", "value": "x"}])
            ],
        }
    )
    with pytest.raises(RuntimeError):
        _resolve_legacy_application(
            nobase_session,
            {"Authorization": "Bearer token"},
            "db",
            manager_url="http://manager",
        )

    alias_session = FakeSession(
        {
            ("GET", "http://manager/legacy/applications"): [
                FakeResponse(
                    HTTPStatus.OK,
                    "u",
                    [
                        {
                            "name": "app-name",
                            "config": [
                                {"path": "other", "value": "x"},
                                {"path": "env", "value": "db"},
                            ],
                        }
                    ],
                )
            ],
            ("GET", "http://manager/legacy/applications/app-name/config"): [
                FakeResponse(HTTPStatus.OK, "u", [{"path": "baseUrl", "value": "http://base"}])
            ],
        }
    )
    base_url, headers = _resolve_legacy_application(
        alias_session,
        {"Authorization": "Bearer token"},
        "db",
        manager_url="http://manager",
    )
    assert base_url == "http://base"
    assert headers == {"Authorization": "Bearer token"}

    nonlist_session = FakeSession(
        {("GET", "http://base/files"): [FakeResponse(HTTPStatus.OK, "u", {"not": "a-list"})]}
    )
    with pytest.raises(RuntimeError):
        _list_public_files(nonlist_session, {"Authorization": "Bearer token"}, "http://base")

    with pytest.raises(RuntimeError):
        _select_public_file(
            [{"filename": "a.csv", "description": "other"}],
            filename_suffix=None,
            description="wanted",
        )
    with pytest.raises(RuntimeError):
        _select_public_file(
            [],
            filename_suffix=None,
            description=None,
        )

    with IIASAVariantServer(runs_payload=[], ts_payload=[]) as server:
        with pytest.raises(RuntimeError):
            download_iiasa_explorer_data(
                database="db",
                variables=["Emissions|CO2"],
                region="World",
                meta_columns=["Category"],
                manager_url=server.base_url,
            )

    runs_payload = [{"id": 1, "model": "M1", "scenario": "S1", "metadata": {}}]
    with IIASAVariantServer(runs_payload=runs_payload, ts_payload=[]) as server:
        with pytest.raises(RuntimeError):
            download_iiasa_explorer_data(
                database="db",
                variables=["Emissions|CO2"],
                region="World",
                meta_columns=["Category"],
                manager_url=server.base_url,
            )

    bad_runs_payload = [{"id": 1, "model": "M1", "metadata": {}}]
    ts_payload = [
        {
            "runId": 1,
            "variable": "Emissions|CO2",
            "unit": "MtCO2/yr",
            "region": "World",
            "year": 2020,
            "value": 1.0,
        }
    ]
    with IIASAVariantServer(runs_payload=bad_runs_payload, ts_payload=ts_payload) as server:
        with pytest.raises(RuntimeError):
            download_iiasa_explorer_data(
                database="db",
                variables=["Emissions|CO2"],
                region="World",
                meta_columns=["Category"],
                manager_url=server.base_url,
            )

    missing_ts_cols = [{"runId": 1, "year": 2020, "value": 1.0}]
    with IIASAVariantServer(runs_payload=runs_payload, ts_payload=missing_ts_cols) as server:
        with pytest.raises(RuntimeError):
            download_iiasa_explorer_data(
                database="db",
                variables=["Emissions|CO2"],
                region="World",
                meta_columns=["Category"],
                manager_url=server.base_url,
            )

    ts_with_identity = [
        {
            "run_id": 1,
            "model": "M1",
            "scenario": "S1",
            "variable": "Emissions|CO2",
            "unit": "MtCO2/yr",
            "region": "World",
            "year": 2020,
            "value": 1.0,
        }
    ]
    with IIASAVariantServer(runs_payload=runs_payload, ts_payload=ts_with_identity) as server:
        data_df, meta_df = download_iiasa_explorer_data(
            database="db",
            variables=["Emissions|CO2"],
            region="World",
            meta_columns=["MissingOnly"],
            manager_url=server.base_url,
        )
    assert data_df["scenario"].tolist() == ["S1"]
    assert list(meta_df.columns) == ["MissingOnly"]
    assert pd.isna(meta_df.iloc[0, 0])

    runs_payload_with_run_id = [{"run_id": 2, "model": "M2", "scenario": "S2", "metadata": {}}]
    ts_with_run_id = [
        {
            "run_id": 2,
            "model": "M2",
            "scenario": "S2",
            "variable": "Emissions|CO2",
            "unit": "MtCO2/yr",
            "region": "World",
            "year": 2030,
            "value": 2.0,
        }
    ]
    with IIASAVariantServer(
        runs_payload=runs_payload_with_run_id,
        ts_payload=ts_with_run_id,
    ) as server:
        data_df_run_id, meta_df_run_id = download_iiasa_explorer_data(
            database="db",
            variables=["Emissions|CO2"],
            region="World",
            meta_columns=["MissingOnly"],
            manager_url=server.base_url,
        )
    assert data_df_run_id["scenario"].tolist() == ["S2"]
    assert list(meta_df_run_id.columns) == ["MissingOnly"]

    files_payload = [{"filename": "", "description": "blank"}]
    with IIASAVariantServer(
        runs_payload=[],
        ts_payload=[],
        files_payload=files_payload,
    ) as server:
        with pytest.raises(RuntimeError):
            download_iiasa_public_file(
                database="db",
                target_path=tmp_path / "blank.csv",
                description="blank",
                manager_url=server.base_url,
                download_bytes_func=lambda url: b"x",
            )

    files_payload = [{"filename": "demo.csv", "description": "demo"}]
    with IIASAVariantServer(
        runs_payload=[],
        ts_payload=[],
        files_payload=files_payload,
        link_payloads={"demo.csv": {"notDirectLink": "x"}},
    ) as server:
        with pytest.raises(RuntimeError):
            download_iiasa_public_file(
                database="db",
                target_path=tmp_path / "demo.csv",
                description="demo",
                manager_url=server.base_url,
                download_bytes_func=lambda url: b"x",
            )


def test_ar6_collection_historical_source_edge_branches(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    payload = historical_sources._request_json(
        "http://example",
        request_get=lambda url, timeout=60: FakeResponse(HTTPStatus.OK, url, {"ok": True}),
    )
    assert payload == {"ok": True}

    with pytest.raises(RuntimeError):
        historical_sources._fetch_latest_primap_record(
            request_get=lambda url, **kwargs: FakeResponse(HTTPStatus.OK, url, {"id": 1}),
        )

    def primap_request(url: str, **kwargs):
        if url.endswith("/seed"):
            return FakeResponse(HTTPStatus.OK, url, {"conceptrecid": 7})
        return FakeResponse(HTTPStatus.OK, url, {"hits": {"hits": []}})

    with pytest.raises(RuntimeError):
        historical_sources._fetch_latest_primap_record(
            seed_record_url="http://example/seed",
            records_api_url="http://example/records",
            request_get=primap_request,
        )

    def primap_success_request(url: str, **kwargs):
        if url.endswith("/seed"):
            return FakeResponse(HTTPStatus.OK, url, {"conceptrecid": 7})
        return FakeResponse(HTTPStatus.OK, url, {"hits": {"hits": [{"id": 9}]}})

    assert historical_sources._fetch_latest_primap_record(
        seed_record_url="http://example/seed",
        records_api_url="http://example/records",
        request_get=primap_success_request,
    ) == {"id": 9}

    final_file, no_rounding_file = historical_sources._select_primap_files(
        {
            "files": [
                {"key": "notes.txt"},
                {
                    "key": "x-PRIMAP-hist_v1.0_final_no_rounding_01-Jan-2024.csv",
                    "links": {"self": "http://example/no-rounding"},
                },
                {
                    "key": "x-PRIMAP-hist_v1.0_final_01-Jan-2024.csv",
                    "links": {"self": "http://example/final"},
                },
                {"key": "misc.csv"},
                {"key": "readme.txt"},
            ]
        }
    )
    assert final_file["links"]["self"] == "http://example/final"
    assert no_rounding_file["links"]["self"] == "http://example/no-rounding"

    with pytest.raises(RuntimeError):
        historical_sources._select_primap_files(
            {
                "files": [
                    {
                        "key": "x-PRIMAP-hist_v1.0_final_01-Jan-2024.csv",
                        "links": {"self": "http://example/final"},
                    }
                ]
            }
        )

    download_target = tmp_path / "download" / "target.bin"
    binary_meta = historical_sources._download_binary(
        "http://example/file.bin",
        download_target,
        request_get=lambda url, **kwargs: FakeResponse(
            HTTPStatus.OK,
            "http://example/renamed.bin",
            headers={},
            content=b"payload",
        ),
    )
    assert Path(binary_meta["path"]).read_bytes() == b"payload"
    assert Path(binary_meta["path"]).name == "renamed.bin"

    chunked_meta = historical_sources._download_binary(
        "http://example/chunked.bin",
        tmp_path / "download" / "chunked.bin",
        request_get=lambda url, **kwargs: ChunkedResponse(
            HTTPStatus.OK,
            url,
            headers={},
            content=b"chunked",
        ),
    )
    assert Path(chunked_meta["path"]).read_bytes() == b"chunked"

    with pytest.raises(RuntimeError):
        historical_sources._fetch_latest_gcp_link(
            request_get=lambda url, **kwargs: FakeResponse(HTTPStatus.OK, url, text="<html></html>")
        )

    gcp_link = historical_sources._fetch_latest_gcp_link(
        request_get=lambda url, **kwargs: FakeResponse(
            HTTPStatus.OK,
            url,
            text='<a href="/file.xlsx">National fossil carbon emissions vlatest</a>',
        )
    )
    assert gcp_link[2] is None

    gcp_latest = historical_sources._fetch_latest_gcp_link(
        request_get=lambda url, **kwargs: FakeResponse(
            HTTPStatus.OK,
            url,
            text=(
                '<a href="/ignore.xlsx">Other dataset</a>'
                '<a href="/older.xlsx">National fossil carbon emissions v2023</a>'
                '<a href="/newer.xlsx">National fossil carbon emissions v2024</a>'
            ),
        )
    )
    assert gcp_latest == (
        "https://globalcarbonbudget.org/newer.xlsx",
        "National fossil carbon emissions v2024",
        2024,
    )

    with pytest.raises(RuntimeError):
        historical_sources._extract_gcp_recommended_citation(
            request_get=lambda url, **kwargs: FakeResponse(HTTPStatus.OK, url, text="<html></html>")
        )

    assert (
        historical_sources._extract_gcp_recommended_citation(
            request_get=lambda url, **kwargs: FakeResponse(
                HTTPStatus.OK,
                url,
                text="Citation: Please cite Example et al. (2025).",
            )
        )
        == "Citation: Please cite Example et al. (2025)."
    )

    status_messages: list[str] = []
    historical_sources._emit_status(status_messages.append, "status")
    assert status_messages == ["status"]

    raw_dir = tmp_path / "historical"
    raw_dir.mkdir()
    for source_path in [
        ar6_dummy_repo.primap_final_path,
        ar6_dummy_repo.primap_no_rounding_path,
        ar6_dummy_repo.gcp_path,
        ar6_dummy_repo.overlay_path,
    ]:
        (raw_dir / source_path.name).write_bytes(source_path.read_bytes())
    (raw_dir / "Guetschow_et_al_demo-PRIMAP-hist_v1_final.csv").write_text("old", encoding="utf-8")
    (raw_dir / "National_Fossil_Carbon_Emissions_old.xlsx").write_text("old", encoding="utf-8")
    (raw_dir / "unrelated.csv").write_text("old", encoding="utf-8")
    citation_txt_path_for_raw_dir(raw_dir).write_text("stale citation", encoding="utf-8")

    reuse_result = historical_sources.ensure_historical_sources(raw_dir=raw_dir, refresh=False)
    assert reuse_result["used_local_gcp"] is True
    assert reuse_result["used_local_ar6_historical_figure_reference"] is True

    refreshed = historical_sources.ensure_historical_sources(
        raw_dir=raw_dir,
        refresh=True,
        fetch_latest_primap_record_func=lambda: {
            "id": 1,
            "conceptrecid": 2,
            "metadata": {
                "title": "PRIMAP title",
                "publication_date": "2025-01-01",
                "creators": [{"name": "Alice"}],
            },
            "files": [
                {
                    "key": "x-PRIMAP-hist_v1.0_final_01-Jan-2024.csv",
                    "links": {"self": "http://example/final"},
                },
                {
                    "key": "x-PRIMAP-hist_v1.0_final_no_rounding_01-Jan-2024.csv",
                    "links": {"self": "http://example/no_rounding"},
                },
            ],
        },
        select_primap_files_func=historical_sources._select_primap_files,
        download_binary_func=lambda url, target_path, force_target_name=False: {
            "path": str(target_path),
            "filename": target_path.name,
            "url": url,
        },
        fetch_latest_gcp_link_func=lambda: ("http://example/gcp.xlsx", "label", 2024),
        extract_gcp_recommended_citation_func=lambda: "citation",
        download_iiasa_public_file_func=lambda **kwargs: {
            "path": str(kwargs["target_path"]),
            "filename": kwargs["target_path"].name,
        },
    )
    assert refreshed["primap"] is not None
    assert (raw_dir / "unrelated.csv").read_text(encoding="utf-8") == "old"
    assert citation_txt_path_for_raw_dir(raw_dir).read_text(encoding="utf-8").strip()

    with pytest.raises(RuntimeError):
        historical_sources.ensure_historical_sources(
            raw_dir=tmp_path / "missing-links",
            refresh=True,
            fetch_latest_primap_record_func=lambda: {"id": 1, "conceptrecid": 2, "metadata": {}},
            select_primap_files_func=lambda record: (
                {"key": "final.csv", "links": {}},
                {"key": "no_rounding.csv", "links": {}},
            ),
            download_binary_func=lambda url, target_path, force_target_name=False: {
                "path": str(target_path),
                "filename": target_path.name,
                "url": url,
            },
            fetch_latest_gcp_link_func=lambda: ("http://example/gcp.xlsx", "label", 2024),
            extract_gcp_recommended_citation_func=lambda: "citation",
            download_iiasa_public_file_func=lambda **kwargs: {
                "path": str(kwargs["target_path"]),
                "filename": kwargs["target_path"].name,
            },
        )
