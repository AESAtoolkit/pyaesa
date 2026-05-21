"""IIASA AR6 explorer download ownership."""

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

from .public_archive import (
    AR6_WORLD_ARCHIVE_DESCRIPTION,
    AR6_WORLD_ARCHIVE_FILENAME_SUFFIX,
    load_ar6_public_archive_data,
)

DEFAULT_MANAGER_URL = "https://api.manager.ece.iiasa.ac.at"
AR6_SCENARIO_EXPLORER_URL = "https://data.ece.iiasa.ac.at/ar6/"
AR6_SCENARIO_EXPLORER_ABOUT_URL = "https://data.ece.iiasa.ac.at/ar6/static/About.html"
AR6_SCENARIO_EXPLORER_RECOMMENDED_CITATION = (
    "Edward Byers, Volker Krey, Elmar Kriegler, Keywan Riahi, Roberto Schaeffer, "
    "Jarmo Kikstra, Robin Lamboll, Zebedee Nicholls, Marit Sanstad, Chris Smith, "
    "Kaj-Ivar van def Wijst, Alaa Al Khourdajie, Franck Lecocq, Joana Portugal-Pereira, "
    "Yamina Saheb, Anders Stromann, Harald Winkler, Cornelia Auer, Elina Brutschin, "
    "Matthew Gidden, Philip Hackstock, Mathijs Harmsen, Daniel Huppmann, Peter Kolp, "
    "Claire Lepault, Jared Lewis, Giacomo Marangoni, Eduardo Muller-Casseres, "
    "Ragnhild Skeie, Michaela Werning, Katherine Calvin, Piers Forster, Celine Guivarch, "
    "Tomoko Hasegawa, Malte Meinshausen, Glen Peters, Joeri Rogelj, Bjorn Samset, "
    "Julia Steinberger, Massimo Tavoni, Detlef van Vuuren. AR6 Scenarios Database "
    "hosted by IIASA. International Institute for Applied Systems Analysis, 2022. "
    "doi:10.5281/zenodo.5886911 | url: data.ece.iiasa.ac.at/ar6/."
)


def _get_anonymous_headers(
    session: requests.Session,
    *,
    manager_url: str = DEFAULT_MANAGER_URL,
) -> dict[str, str]:
    response = session.get(f"{manager_url}/legacy/anonym/", timeout=30)
    response.raise_for_status()
    token = response.json()
    if not isinstance(token, str) or not token:
        raise RuntimeError("IIASA anonymous token response was not a non-empty string.")
    return {"Authorization": f"Bearer {token}"}


def _request_json(
    session: requests.Session,
    http_method: str,
    url: str,
    headers: dict[str, str],
    *,
    params: dict | None = None,
    json_body: dict | None = None,
    timeout: int = 120,
    auth_header_loader=None,
) -> tuple[Any, dict[str, str]]:
    request_headers = dict(headers)
    if json_body is not None:
        request_headers["Content-Type"] = "application/json"
    response = session.request(
        method=http_method,
        url=url,
        headers=request_headers,
        params=params,
        data=None if json_body is None else json.dumps(json_body),
        timeout=timeout,
    )
    if response.status_code in {401, 403}:
        loader = _get_anonymous_headers if auth_header_loader is None else auth_header_loader
        headers = loader(session)
        request_headers = dict(headers)
        if json_body is not None:
            request_headers["Content-Type"] = "application/json"
        response = session.request(
            method=http_method,
            url=url,
            headers=request_headers,
            params=params,
            data=None if json_body is None else json.dumps(json_body),
            timeout=timeout,
        )
    response.raise_for_status()
    return response.json(), headers


def _resolve_legacy_application(
    session: requests.Session,
    headers: dict[str, str],
    slug: str,
    *,
    manager_url: str = DEFAULT_MANAGER_URL,
    auth_header_loader=None,
) -> tuple[str, dict[str, str]]:
    apps, headers = _request_json(
        session,
        "GET",
        f"{manager_url}/legacy/applications",
        headers,
        timeout=60,
        auth_header_loader=auth_header_loader,
    )
    env_to_name: dict[str, str] = {}
    name_to_name: dict[str, str] = {}
    for app in apps:
        app_name = app["name"]
        name_to_name[app_name] = app_name
        env_value = None
        for entry in app.get("config", []):
            if entry.get("path") == "env":
                env_value = entry.get("value")
                break
        if env_value:
            env_to_name[env_value] = app_name
    resolved_name = env_to_name.get(slug, name_to_name.get(slug))
    if not resolved_name:
        raise RuntimeError(f"Could not resolve IIASA legacy application slug '{slug}'.")
    config, headers = _request_json(
        session,
        "GET",
        f"{manager_url}/legacy/applications/{resolved_name}/config",
        headers,
        timeout=60,
        auth_header_loader=auth_header_loader,
    )
    config_map = {entry["path"]: entry["value"] for entry in config}
    base_url = config_map.get("baseUrl")
    if not base_url:
        raise RuntimeError(f"Application '{resolved_name}' did not expose a baseUrl.")
    return base_url.rstrip("/"), headers


def _list_public_files(
    session: requests.Session,
    headers: dict[str, str],
    base_url: str,
) -> tuple[list[dict], dict[str, str]]:
    files, headers = _request_json(
        session,
        "GET",
        f"{base_url}/files",
        headers,
        params={"includePublic": "true", "includePrivate": "true"},
        timeout=60,
    )
    if not isinstance(files, list):
        raise RuntimeError("IIASA public files response was not a list.")
    return files, headers


def _select_public_file(
    files: list[dict],
    *,
    filename_suffix: str | None,
    description: str | None,
) -> dict:
    matches: list[dict] = []
    for entry in files:
        filename = str(entry.get("filename", ""))
        entry_description = str(entry.get("description", ""))
        if filename_suffix is not None and not filename.endswith(filename_suffix):
            continue
        if description is not None and entry_description != description:
            continue
        matches.append(entry)
    if not matches:
        criteria = []
        if filename_suffix is not None:
            criteria.append(f"filename suffix '{filename_suffix}'")
        if description is not None:
            criteria.append(f"description '{description}'")
        criteria_s = " and ".join(criteria) if criteria else "the requested criteria"
        raise RuntimeError(f"Could not find an IIASA public file matching {criteria_s}.")
    matches.sort(
        key=lambda entry: (
            int(entry.get("createAt") or -1),
            str(entry.get("filename", "")),
        )
    )
    return matches[-1]


def _download_bytes(url: str, *, timeout: int = 300) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _flatten_runs_metadata(runs_df: pd.DataFrame) -> pd.DataFrame:
    if "metadata" not in runs_df.columns:
        return runs_df
    metadata_series = runs_df.loc[:, "metadata"]
    if bool(pd.isna(metadata_series).all()):
        return runs_df.drop(columns=["metadata"])
    meta_df = pd.DataFrame.from_records(metadata_series.tolist())
    return pd.concat([runs_df.drop(columns=["metadata"]).reset_index(drop=True), meta_df], axis=1)


def _resolve_run_id_column(df: pd.DataFrame) -> str:
    for col in ["run_id", "runId", "id"]:
        if col in df.columns:
            return col
    raise RuntimeError("No run identifier column found in IIASA runs response.")


def _pick_consistent_value(values: pd.Series):
    non_na = values.dropna()
    if non_na.empty:
        return pd.NA
    unique = pd.unique(non_na)
    if len(unique) == 1:
        return unique[0]
    raise RuntimeError(
        "IIASA run metadata contained conflicting values for one model-scenario pair. "
        f"Metadata field='{values.name}'. Observed values={unique.tolist()}. "
        "The AR6 explorer download cannot resolve that ambiguity automatically."
    )


def _load_ar6_public_archive_from_files(
    *,
    session: requests.Session,
    headers: dict[str, str],
    base_url: str,
    variables: list[str],
    region: str,
    meta_columns: list[str],
    auth_header_loader,
    download_bytes_func=_download_bytes,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ``ar6-public`` data from the IIASA public archive files endpoint."""
    files, headers = _list_public_files(session, headers, base_url)
    selected = _select_public_file(
        files,
        filename_suffix=AR6_WORLD_ARCHIVE_FILENAME_SUFFIX,
        description=AR6_WORLD_ARCHIVE_DESCRIPTION,
    )
    source_filename = str(selected.get("filename", ""))
    link_payload, _headers = _request_json(
        session,
        "GET",
        f"{base_url}/files/{source_filename}",
        headers,
        params={"redirect": "false"},
        timeout=60,
        auth_header_loader=auth_header_loader,
    )
    if not isinstance(link_payload, dict) or "directLink" not in link_payload:
        raise RuntimeError(
            f"The AR6 public archive '{source_filename}' did not expose a direct download link."
        )
    archive_bytes = download_bytes_func(str(link_payload["directLink"]))
    return load_ar6_public_archive_data(
        archive_bytes,
        variables=variables,
        region=region,
        meta_columns=meta_columns,
    )


def download_iiasa_explorer_data(
    *,
    database: str,
    variables: list[str],
    region: str,
    meta_columns: list[str],
    manager_url: str = DEFAULT_MANAGER_URL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download AR6 explorer timeseries and scenario metadata from IIASA."""

    def auth_header_loader(session: requests.Session) -> dict[str, str]:
        return _get_anonymous_headers(session, manager_url=manager_url)

    with requests.Session() as session:
        headers = auth_header_loader(session)
        base_url, headers = _resolve_legacy_application(
            session,
            headers,
            database,
            manager_url=manager_url,
            auth_header_loader=auth_header_loader,
        )
        runs_payload, headers = _request_json(
            session,
            "GET",
            f"{base_url}/runs",
            headers,
            params={"getOnlyDefaultRuns": "true", "includeMetadata": "true"},
            timeout=120,
            auth_header_loader=auth_header_loader,
        )
    runs_df = pd.DataFrame.from_records(runs_payload)
    if runs_df.empty:
        if database == "ar6-public":
            with requests.Session() as archive_session:
                archive_headers = auth_header_loader(archive_session)
                archive_base_url, archive_headers = _resolve_legacy_application(
                    archive_session,
                    archive_headers,
                    database,
                    manager_url=manager_url,
                    auth_header_loader=auth_header_loader,
                )
                return _load_ar6_public_archive_from_files(
                    session=archive_session,
                    headers=archive_headers,
                    base_url=archive_base_url,
                    variables=variables,
                    region=region,
                    meta_columns=meta_columns,
                    auth_header_loader=auth_header_loader,
                )
        raise RuntimeError("No IIASA runs returned.")
    runs_df = _flatten_runs_metadata(runs_df)
    run_id_col = _resolve_run_id_column(runs_df)
    runs_df["run_id"] = runs_df[run_id_col]
    if run_id_col != "run_id":
        runs_df = runs_df.drop(columns=[run_id_col])
    query_payload = {
        "filters": {
            "regions": [region],
            "variables": list(variables),
            "runs": runs_df["run_id"].dropna().astype(int).tolist(),
            "years": [],
            "units": [],
            "timeslices": [],
        }
    }
    with requests.Session() as session:
        headers = auth_header_loader(session)
        base_url, headers = _resolve_legacy_application(
            session,
            headers,
            database,
            manager_url=manager_url,
            auth_header_loader=auth_header_loader,
        )
        ts_payload, _headers = _request_json(
            session,
            "POST",
            f"{base_url}/runs/bulk/ts",
            headers,
            json_body=query_payload,
            timeout=300,
            auth_header_loader=auth_header_loader,
        )
    ts_df = pd.DataFrame.from_records(ts_payload)
    if ts_df.empty:
        raise RuntimeError("No timeseries rows returned from IIASA.")
    if "runId" in ts_df.columns:
        ts_df["run_id"] = ts_df["runId"]
        ts_df = ts_df.drop(columns=["runId"])
    if "subannual" in ts_df.columns:
        ts_df = ts_df[ts_df["subannual"].isna() | ts_df["subannual"].isin(["Year", -1])].copy()

    required_run_cols = ["run_id", "model", "scenario"]
    available_meta_cols = [col for col in meta_columns if col in runs_df.columns]
    for req in required_run_cols:
        if req not in runs_df.columns:
            raise RuntimeError(f"Missing required run metadata column '{req}' from IIASA.")
    ts_has_identity = {"model", "scenario"}.issubset(set(ts_df.columns))
    merge_cols = ["run_id"] + available_meta_cols
    if not ts_has_identity:
        merge_cols = required_run_cols + available_meta_cols
    selected_runs = runs_df.loc[:, merge_cols]
    merged = ts_df.merge(
        selected_runs.drop_duplicates(subset=["run_id"], keep="first"),
        on="run_id",
        how="left",
        validate="many_to_one",
    )
    required_data_cols = ["model", "scenario", "variable", "unit", "region", "year", "value"]
    missing_data = [col for col in required_data_cols if col not in merged.columns]
    if missing_data:
        raise RuntimeError(f"Missing required timeseries columns from IIASA: {missing_data}")

    data_df = merged.loc[:, required_data_cols].copy()
    meta_src = merged.loc[:, ["model", "scenario"] + available_meta_cols].copy()
    if available_meta_cols:
        agg_map = {col: _pick_consistent_value for col in available_meta_cols}
        meta_df = meta_src.groupby(["model", "scenario"], dropna=False).agg(agg_map)
    else:
        meta_df = meta_src.drop_duplicates(subset=["model", "scenario"], keep="first").set_index(
            ["model", "scenario"]
        )
    for meta_col in meta_columns:
        if meta_col not in meta_df.columns:
            meta_df[meta_col] = pd.NA
    return data_df, meta_df


def download_iiasa_public_file(
    *,
    database: str,
    target_path: Path,
    filename_suffix: str | None = None,
    description: str | None = None,
    archive_member_name: str | None = None,
    manager_url: str = DEFAULT_MANAGER_URL,
    download_bytes_func=_download_bytes,
) -> dict:
    """Download one public file exposed by the IIASA scenario explorer backend.

    Args:
        database: IIASA legacy application slug.
        target_path: Local output path to write.
        filename_suffix: Optional required filename suffix used to select the
            public file from the IIASA ``/files`` listing.
        description: Optional required description used to select the public
            file from the IIASA ``/files`` listing.
        archive_member_name: Optional member name to extract from a ZIP
            archive instead of saving the archive itself.
        download_bytes_func: Optional byte download used by package tests
            to supply deterministic public file payloads without performing a
            live network download.

    Returns:
        Metadata describing the downloaded local file and the matched IIASA
        public file entry.

    Raises:
        RuntimeError: If the IIASA file listing or selected archive member is
            incompatible with the expected public file contract.
    """

    def auth_header_loader(session: requests.Session) -> dict[str, str]:
        return _get_anonymous_headers(session, manager_url=manager_url)

    with requests.Session() as session:
        headers = auth_header_loader(session)
        base_url, headers = _resolve_legacy_application(
            session,
            headers,
            database,
            manager_url=manager_url,
            auth_header_loader=auth_header_loader,
        )
        files, headers = _list_public_files(session, headers, base_url)
        selected = _select_public_file(
            files,
            filename_suffix=filename_suffix,
            description=description,
        )
        source_filename = str(selected.get("filename", ""))
        if not source_filename:
            raise RuntimeError("The selected IIASA public file did not expose a filename.")
        link_payload, _headers = _request_json(
            session,
            "GET",
            f"{base_url}/files/{source_filename}",
            headers,
            params={"redirect": "false"},
            timeout=60,
            auth_header_loader=auth_header_loader,
        )
    if not isinstance(link_payload, dict) or "directLink" not in link_payload:
        raise RuntimeError(
            f"The IIASA public file '{source_filename}' did not expose a direct download link."
        )
    target_path = ensure_file_parent(target_path)
    content = download_bytes_func(str(link_payload["directLink"]))
    if archive_member_name is None:
        target_path.write_bytes(content)
    else:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive_member_name not in archive.namelist():
                raise RuntimeError(
                    f"The IIASA archive '{source_filename}' did not contain "
                    f"'{archive_member_name}'."
                )
            target_path.write_bytes(archive.read(archive_member_name))
    return {
        "path": str(target_path),
        "filename": target_path.name,
        "source_database": database,
        "source_filename": source_filename,
        "source_description": selected.get("description"),
        "source_content_type": selected.get("contentType"),
        "source_file_size": selected.get("fileSize"),
        "source_created_at": selected.get("createAt"),
        "direct_link": str(link_payload["directLink"]),
        "archive_member_name": archive_member_name,
    }
