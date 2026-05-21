"""Historical raw source download ownership used by the AR6 climate pipeline."""

import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent
from pyaesa.shared.runtime.text import join_user_text_lines

from .download_iiasa import (
    AR6_SCENARIO_EXPLORER_ABOUT_URL,
    AR6_SCENARIO_EXPLORER_RECOMMENDED_CITATION,
    AR6_SCENARIO_EXPLORER_URL,
    DEFAULT_MANAGER_URL,
    download_iiasa_public_file,
)
from ..io.paths import citation_txt_path_for_raw_dir

PRIMAP_SEED_RECORD_URL = "https://zenodo.org/api/records/17090760"
GCP_LATEST_DATA_URL = "https://globalcarbonbudget.org/the-latest-gcb-data/"
AR6_HISTORICAL_FIGURE_REFERENCE_DESCRIPTION = "AR6_historical_emissions"
AR6_HISTORICAL_FIGURE_REFERENCE_ZIP_SUFFIX = "-AR6_historical_emissions.zip"
AR6_HISTORICAL_FIGURE_REFERENCE_ARCHIVE_MEMBER = "AR6_historical_emissions.csv"

PRIMAP_FINAL_LOCAL_NAME = "primap_hist_final.csv"
PRIMAP_FINAL_NO_ROUNDING_LOCAL_NAME = "primap_hist_final_no_rounding.csv"
GCP_NATIONAL_FOSSIL_LOCAL_NAME = "gcp_national_fossil.xlsx"
AR6_HISTORICAL_FIGURE_REFERENCE_LOCAL_NAME = "ar6_historical_figure_reference.csv"
EDGAR_RECOMMENDED_CITATION = (
    "Minx, J. C., Lamb, W. F., Andrew, R. M., Canadell, J. G., Crippa, M., "
    "Doebbeling, N., Forster, P. M., Guizzardi, D., Olivier, J., Peters, G. P., "
    "Pongratz, J., Reisinger, A., Rigby, M., Saunois, M., Smith, S. J., Solazzo, E., "
    "and Tian, H. (2021): A comprehensive and synthetic dataset for global, regional, "
    "and national greenhouse gas emissions by sector 1970-2018 with an extension to 2019. "
    "Earth System Science Data, 13, 5213-5252. https://doi.org/10.5194/essd-13-5213-2021."
)
RCMIP_RECOMMENDED_CITATION = (
    "Nicholls, Z. R. J., Meinshausen, M., Lewis, J., Gieseke, R., Dommenget, D., "
    "Dorheim, K., Fan, C.-S., Fuglestvedt, J. S., Gasser, T., Goluke, U., Goodwin, P., "
    "Hartin, C., Hope, A. P., Kriegler, E., Leach, N. J., Marchegiani, D., McBride, L. A., "
    "Quilcaille, Y., Rogelj, J., Salawitch, R. J., Samset, B. H., Sandstad, M., "
    "Shiklomanov, A. N., Skeie, R. B., Smith, C. J., Smith, S., Tanaka, K., Tsutsui, J., "
    "and Xie, Z. (2020): Reduced Complexity Model Intercomparison Project Phase 1: "
    "introduction and evaluation of global-mean temperature response. Geoscientific "
    "Model Development, 13, 5175-5190. https://doi.org/10.5194/gmd-13-5175-2020."
)


def _version_tuple(version_s: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in str(version_s).split("."):
        token = token.strip()
        parts.append(int(token) if token.isdigit() else 0)
    return tuple(parts)


def _parse_primap_sort_key(filename: str) -> tuple[tuple[int, ...], str]:
    match = re.search(r"_v([0-9][0-9.]*)_", filename)
    version = _version_tuple(match.group(1)) if match else tuple()
    return version, filename


def _request_json(url: str, timeout: int = 60, request_get=requests.get) -> dict:
    response = request_get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _fetch_latest_primap_record(
    *,
    seed_record_url: str = PRIMAP_SEED_RECORD_URL,
    records_api_url: str = "https://zenodo.org/api/records",
    request_get=requests.get,
) -> dict:
    seed = _request_json(seed_record_url, request_get=request_get)
    concept = seed.get("conceptrecid")
    if concept is None:
        raise RuntimeError(
            "The PRIMAP seed Zenodo record did not expose a conceptrecid for latest-version lookup."
        )
    response = request_get(
        records_api_url,
        params={"q": f"conceptrecid:{concept}", "sort": "mostrecent", "size": 1},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    hits = payload.get("hits", {}).get("hits", [])
    if not hits:
        raise RuntimeError(
            f"Could not resolve the latest PRIMAP Zenodo record for conceptrecid {concept}."
        )
    return hits[0]


def _select_primap_files(record: dict) -> tuple[dict, dict]:
    files = record.get("files", [])
    final_candidates: list[dict] = []
    no_rounding_candidates: list[dict] = []
    for entry in files:
        key = entry.get("key", "")
        if not key.endswith(".csv"):
            continue
        if re.search(
            r"-PRIMAP-hist_v[0-9.]+_final_no_rounding_\d{2}-[A-Za-z]{3}-\d{4}\.csv$",
            key,
        ):
            no_rounding_candidates.append(entry)
            continue
        if re.search(
            r"-PRIMAP-hist_v[0-9.]+_final_\d{2}-[A-Za-z]{3}-\d{4}\.csv$",
            key,
        ):
            final_candidates.append(entry)
    if not final_candidates:
        raise RuntimeError("Could not find PRIMAP final CSV in latest Zenodo record.")
    if not no_rounding_candidates:
        raise RuntimeError("Could not find PRIMAP final_no_rounding CSV in latest Zenodo record.")
    final_file = sorted(
        final_candidates,
        key=lambda item: _parse_primap_sort_key(item.get("key", "")),
    )[-1]
    no_rounding_file = sorted(
        no_rounding_candidates,
        key=lambda item: _parse_primap_sort_key(item.get("key", "")),
    )[-1]
    return final_file, no_rounding_file


def _extract_filename_from_response(response: requests.Response, fallback_name: str) -> str:
    content_disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if match:
        return match.group(1)
    parsed = urlparse(response.url)
    candidate = Path(parsed.path).name
    if candidate:
        return candidate
    return fallback_name


def _download_binary(
    url: str,
    target_path: Path,
    force_target_name: bool = False,
    request_get=requests.get,
) -> dict:
    target_path = ensure_file_parent(target_path)
    with request_get(url, timeout=120, stream=True, allow_redirects=True) as response:
        response.raise_for_status()
        filename = (
            target_path.name
            if force_target_name
            else _extract_filename_from_response(response, target_path.name)
        )
        write_path = target_path.with_name(filename)
        with write_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return {"path": str(write_path), "filename": write_path.name, "url": url}


def _fetch_latest_gcp_link(
    *,
    page_url: str = GCP_LATEST_DATA_URL,
    request_get=requests.get,
) -> tuple[str, str, int | None]:
    response = request_get(page_url, timeout=60)
    response.raise_for_status()
    html = response.text
    anchor_matches = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, flags=re.I | re.S)
    candidates: list[tuple[int, str, str]] = []
    for href, label in anchor_matches:
        clean_label = re.sub(r"<[^>]*>", "", label).strip()
        if "national fossil carbon emissions v" not in clean_label.lower():
            continue
        year_match = re.search(r"v(\d{4})", clean_label, flags=re.I)
        year = int(year_match.group(1)) if year_match else None
        abs_href = urljoin(page_url, href)
        candidates.append((year if year is not None else -1, abs_href, clean_label))
    if not candidates:
        raise RuntimeError("Could not find GCP National fossil emissions download link.")
    candidates.sort(key=lambda item: item[0])
    year, href, label = candidates[-1]
    return href, label, (year if year >= 0 else None)


def _extract_gcp_recommended_citation(
    *,
    page_url: str = GCP_LATEST_DATA_URL,
    request_get=requests.get,
) -> str:
    response = request_get(page_url, timeout=60)
    response.raise_for_status()
    html = response.text
    match = re.search(r"Citation:\s*Please cite[^<\n\r]+", html, flags=re.I)
    if match:
        return match.group(0).strip()
    raise RuntimeError(
        "Could not extract the recommended Global Carbon Budget citation from the source page."
    )


def _record_to_citation(record: dict) -> dict:
    metadata = record.get("metadata", {})
    creators = [
        creator.get("name") for creator in metadata.get("creators", []) if creator.get("name")
    ]
    return {
        "title": metadata.get("title"),
        "doi": metadata.get("doi"),
        "publication_date": metadata.get("publication_date"),
        "creators": creators,
        "record_id": record.get("id"),
        "conceptrecid": record.get("conceptrecid"),
    }


def _primap_recommended_citation_from_record(record: dict) -> str:
    citation = _record_to_citation(record)
    creators = citation.get("creators", [])
    creator_s = "; ".join(creators) if creators else "Unknown authors"
    title = citation.get("title") or "PRIMAP-hist dataset"
    doi = citation.get("doi")
    date = citation.get("publication_date") or "n.d."
    if doi:
        return f"{creator_s} ({date}): {title}. Zenodo. doi:{doi}."
    return f"{creator_s} ({date}): {title}. Zenodo."


def historical_citations_txt(citations: dict) -> str:
    """Return the raw citation/source usage TXT content."""
    primap_rec = citations.get("primap_hist", {}).get("recommended_citation", "")
    gcp_rec = citations.get("gcp_national_fossil", {}).get("recommended_citation", "")
    ar6_hist = citations.get("ar6_historical_figure_reference", {})
    primap_classic = (
        "Gutschow, J.; Jeffery, L.; Gieseke, R.; Gebel, R.; Stevens, D.; Krapp, M.; "
        "Rocha, M. (2016): The PRIMAP-hist national historical emissions time series, "
        "Earth Syst. Sci. Data, 8, 571-603, doi:10.5194/essd-8-571-2016"
    )
    return join_user_text_lines(
        [
            "Recommended citations and source usage for AR6 climate raw inputs",
            "",
            "AR6 Scenarios Database hosted by IIASA:",
            AR6_SCENARIO_EXPLORER_RECOMMENDED_CITATION,
            f"Scenario Explorer URL: {AR6_SCENARIO_EXPLORER_URL}",
            f"About page with citation guidance: {AR6_SCENARIO_EXPLORER_ABOUT_URL}",
            "",
            "PRIMAP-hist (used for historical Kyoto Gases and CO2 baselines):",
            primap_rec,
            primap_classic,
            "",
            "Global Carbon Budget national fossil dataset (used to add bunker CO2 emissions):",
            gcp_rec,
            "",
            (
                "AR6 historical comparison file downloaded from the AR6 Scenario Explorer "
                "(used only as the red EDGAR/RCMIP overlay in the historical figure, "
                "not in the harmonization baseline):"
            ),
            ar6_hist.get("recommended_citation", ""),
            ar6_hist.get("secondary_citation", ""),
            "",
            (
                "PRIMAP-hist and Global Carbon Budget are used by process_ar6 when building "
                "the harmonized outputs. The AR6 historical comparison file is used only for the "
                "historical-emissions figure overlay."
            ),
        ]
    )


def _emit_status(status_callback: Callable[[str], None] | None, message: str) -> None:
    """Render one transient download status line when a callback is available."""
    if callable(status_callback):
        status_callback(message)


def ensure_historical_sources(
    raw_dir: Path,
    refresh: bool = False,
    status_callback: Callable[[str], None] | None = None,
    fetch_latest_primap_record_func=_fetch_latest_primap_record,
    select_primap_files_func=_select_primap_files,
    download_binary_func=_download_binary,
    fetch_latest_gcp_link_func=_fetch_latest_gcp_link,
    extract_gcp_recommended_citation_func=_extract_gcp_recommended_citation,
    download_iiasa_public_file_func=download_iiasa_public_file,
    manager_url: str = DEFAULT_MANAGER_URL,
) -> dict:
    """Ensure PRIMAP, GCP, and AR6 historical figure reference files exist."""
    raw_dir = ensure_dir(raw_dir)
    citation_txt_file = citation_txt_path_for_raw_dir(raw_dir)
    primap_final_local = raw_dir / PRIMAP_FINAL_LOCAL_NAME
    primap_no_rounding_local = raw_dir / PRIMAP_FINAL_NO_ROUNDING_LOCAL_NAME
    gcp_local = raw_dir / GCP_NATIONAL_FOSSIL_LOCAL_NAME
    ar6_historical_local = raw_dir / AR6_HISTORICAL_FIGURE_REFERENCE_LOCAL_NAME
    if refresh:
        for path in [
            citation_txt_file,
            primap_final_local,
            primap_no_rounding_local,
            gcp_local,
            ar6_historical_local,
        ]:
            path.unlink(missing_ok=True)
        for path in raw_dir.glob("Guetschow_et_al_*-PRIMAP-hist_v*_final*.csv"):
            path.unlink(missing_ok=True)
        for path in raw_dir.glob("National_Fossil_Carbon_Emissions_*.xlsx"):
            path.unlink(missing_ok=True)

    need_primap = (not primap_final_local.exists()) or (not primap_no_rounding_local.exists())
    need_gcp = not gcp_local.exists()
    need_ar6_historical = not ar6_historical_local.exists()
    primap_result = None
    gcp_result = None
    ar6_historical_result = None
    citations = {
        "updated_at": datetime.now().isoformat(),
        "primap_hist": None,
        "gcp_national_fossil": None,
        "ar6_historical_figure_reference": None,
    }
    if need_primap or refresh:
        _emit_status(status_callback, "Downloading PRIMAP historical emissions data (2/4)")
        primap_record = fetch_latest_primap_record_func()
        primap_final, primap_no_rounding = select_primap_files_func(primap_record)
        primap_final_url = primap_final.get("links", {}).get("self")
        primap_no_rounding_url = primap_no_rounding.get("links", {}).get("self")
        if not primap_final_url or not primap_no_rounding_url:
            raise RuntimeError("PRIMAP latest files did not expose direct download links.")
        primap_final_dl = download_binary_func(
            primap_final_url, primap_final_local, force_target_name=True
        )
        primap_no_rounding_dl = download_binary_func(
            primap_no_rounding_url,
            primap_no_rounding_local,
            force_target_name=True,
        )
        primap_result = {
            "final_file": primap_final_dl,
            "final_no_rounding_file": primap_no_rounding_dl,
            "record": _record_to_citation(primap_record),
            "source_filenames": [primap_final.get("key"), primap_no_rounding.get("key")],
        }
        citations["primap_hist"] = {
            "source": "Zenodo",
            "record": _record_to_citation(primap_record),
            "files": [primap_final_dl, primap_no_rounding_dl],
            "source_filenames": [primap_final.get("key"), primap_no_rounding.get("key")],
            "recommended_citation": _primap_recommended_citation_from_record(primap_record),
        }
    else:
        citations["primap_hist"] = {
            "source": "existing local files",
            "files": [str(primap_final_local), str(primap_no_rounding_local)],
        }
    if need_gcp or refresh:
        _emit_status(
            status_callback,
            "Downloading Global Carbon Project national fossil emissions data (3/4)",
        )
        gcp_url, gcp_label, gcp_year = fetch_latest_gcp_link_func()
        gcp_dl = download_binary_func(gcp_url, gcp_local, force_target_name=True)
        gcp_result = {
            "file": gcp_dl,
            "label": gcp_label,
            "year_in_label": gcp_year,
            "source_page": GCP_LATEST_DATA_URL,
        }
        citations["gcp_national_fossil"] = {
            "source": "Global Carbon Budget",
            "source_page": GCP_LATEST_DATA_URL,
            "dataset_label": gcp_label,
            "file": gcp_dl,
            "recommended_citation": extract_gcp_recommended_citation_func(),
        }
    else:
        citations["gcp_national_fossil"] = {
            "source": "existing local files",
            "files": [str(gcp_local)],
        }
    if need_ar6_historical or refresh:
        _emit_status(
            status_callback,
            "Downloading AR6 historical comparison data for figures (4/4)",
        )
        ar6_historical_dl = download_iiasa_public_file_func(
            database="IXSE_AR6_PUBLIC",
            target_path=ar6_historical_local,
            filename_suffix=AR6_HISTORICAL_FIGURE_REFERENCE_ZIP_SUFFIX,
            description=AR6_HISTORICAL_FIGURE_REFERENCE_DESCRIPTION,
            archive_member_name=AR6_HISTORICAL_FIGURE_REFERENCE_ARCHIVE_MEMBER,
            manager_url=manager_url,
        )
        ar6_historical_result = {
            "file": ar6_historical_dl,
            "source": AR6_SCENARIO_EXPLORER_URL,
        }
        citations["ar6_historical_figure_reference"] = {
            "source": "AR6 Scenarios Database hosted by IIASA",
            "scenario_explorer_url": AR6_SCENARIO_EXPLORER_URL,
            "file": ar6_historical_dl,
            "recommended_citation": EDGAR_RECOMMENDED_CITATION,
            "secondary_citation": RCMIP_RECOMMENDED_CITATION,
            "usage_note": (
                "Used only for the historical comparison overlay in "
                "process_ar6(..., figures=True). This file is the "
                "AR6 Scenario Explorer public download "
                "'AR6_historical_emissions.csv'."
            ),
        }
    else:
        citations["ar6_historical_figure_reference"] = {
            "source": "existing local files",
            "files": [str(ar6_historical_local)],
        }

    citation_txt_file.write_text(historical_citations_txt(citations), encoding="utf-8")
    return {
        "citation_txt_file": str(citation_txt_file),
        "primap": primap_result,
        "gcp": gcp_result,
        "ar6_historical_figure_reference": ar6_historical_result,
        "used_local_primap": not (need_primap or refresh),
        "used_local_gcp": not (need_gcp or refresh),
        "used_local_ar6_historical_figure_reference": not (need_ar6_historical or refresh),
    }
