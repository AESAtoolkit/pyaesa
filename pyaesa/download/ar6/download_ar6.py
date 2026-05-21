"""Public AR6 raw data download entrypoint."""

from pyaesa.shared.runtime.reporting.status import TransientStatusPrinter

from .utils.config import (
    DEFAULT_DATABASE,
    DEFAULT_META_COLUMNS,
    DEFAULT_REGION,
    DEFAULT_VARIABLES_RELEVANT,
)
from .utils.io.metadata import (
    build_download_metadata_payload,
    read_download_metadata,
    require_metadata_for_existing_output,
    signature_matches,
    write_download_metadata,
)
from .utils.io.paths import (
    clear_download_output_scope,
    get_citation_txt_path,
    get_explorer_csv_path,
    get_logs_dir,
    get_metadata_path,
    get_raw_dir,
)
from .utils.io.reports import DownloadReportAR6
from .utils.sources.download_iiasa import (
    AR6_SCENARIO_EXPLORER_ABOUT_URL,
    AR6_SCENARIO_EXPLORER_RECOMMENDED_CITATION,
    AR6_SCENARIO_EXPLORER_URL,
    DEFAULT_MANAGER_URL,
    download_iiasa_explorer_data,
)
from .utils.sources.explorer_csv import (
    drop_non_persisted_columns,
    write_explorer_csv,
)
from .utils.sources.historical_sources import ensure_historical_sources


def _download_signature() -> dict[str, object]:
    return {
        "database": DEFAULT_DATABASE,
        "variables": list(DEFAULT_VARIABLES_RELEVANT),
        "meta_columns": list(DEFAULT_META_COLUMNS),
        "region": DEFAULT_REGION,
    }


def _ar6_public_explorer_metadata() -> dict[str, str]:
    return {
        "source": "AR6 Scenarios Database hosted by IIASA",
        "scenario_explorer_url": AR6_SCENARIO_EXPLORER_URL,
        "citation_guidance_url": AR6_SCENARIO_EXPLORER_ABOUT_URL,
        "recommended_citation": AR6_SCENARIO_EXPLORER_RECOMMENDED_CITATION,
    }


def _record_historical_downloaded_assets(
    report: DownloadReportAR6,
    historical_sources: dict[str, object],
) -> bool:
    """Record downloaded historical source names on ``report``."""
    recorded = False
    if historical_sources.get("primap") is not None:
        report.downloaded_assets.extend(
            ["primap_hist_final.csv", "primap_hist_final_no_rounding.csv"]
        )
        recorded = True
    if historical_sources.get("gcp") is not None:
        report.downloaded_assets.append("gcp_national_fossil.xlsx")
        recorded = True
    if historical_sources.get("ar6_historical_figure_reference") is not None:
        report.downloaded_assets.append("ar6_historical_figure_reference.csv")
        recorded = True
    return recorded


def _run_download_ar6(
    *,
    refresh: bool,
    manager_url: str,
) -> DownloadReportAR6 | None:
    """Run the AR6 raw download workflow."""
    raw_dir = get_raw_dir()
    logs_dir = get_logs_dir()
    explorer_csv_file = get_explorer_csv_path(DEFAULT_DATABASE)
    metadata_path = get_metadata_path()
    citation_path = get_citation_txt_path()
    signature = _download_signature()
    if refresh:
        clear_download_output_scope(DEFAULT_DATABASE)
    metadata = read_download_metadata()
    if not refresh:
        require_metadata_for_existing_output(
            metadata=metadata,
            paths=[explorer_csv_file, citation_path],
        )

    report = DownloadReportAR6(
        database=DEFAULT_DATABASE,
        raw_root=raw_dir,
        logs_dir=logs_dir,
        metadata_path=metadata_path,
    )
    saved_outputs_complete = (
        explorer_csv_file.exists() and signature_matches(metadata, signature) and not refresh
    )
    if saved_outputs_complete:
        status = TransientStatusPrinter("download_ar6")
        try:
            historical_sources = ensure_historical_sources(
                raw_dir=raw_dir,
                refresh=False,
                manager_url=manager_url,
                status_callback=status.show,
            )
        finally:
            status.finish()
        if _record_historical_downloaded_assets(report, historical_sources):
            write_download_metadata(
                build_download_metadata_payload(
                    signature=signature,
                    raw_root=raw_dir,
                    explorer_csv_file=explorer_csv_file,
                    citation_txt_file=citation_path,
                    ar6_public_explorer=_ar6_public_explorer_metadata(),
                    historical_sources=historical_sources,
                )
            )
            return report
        return None

    status = TransientStatusPrinter("download_ar6")
    try:
        status.show("Downloading AR6 public explorer data (1/4)")
        data_df, meta_df = download_iiasa_explorer_data(
            database=DEFAULT_DATABASE,
            variables=list(DEFAULT_VARIABLES_RELEVANT),
            region=DEFAULT_REGION,
            meta_columns=list(DEFAULT_META_COLUMNS),
            manager_url=manager_url,
        )
        meta_df = drop_non_persisted_columns(meta_df)
        write_explorer_csv(csv_file=explorer_csv_file, data_df=data_df, meta_df=meta_df)
        report.downloaded_assets.append(explorer_csv_file.name)

        historical_sources = ensure_historical_sources(
            raw_dir=raw_dir,
            refresh=refresh,
            manager_url=manager_url,
            status_callback=status.show,
        )
        _record_historical_downloaded_assets(report, historical_sources)
        write_download_metadata(
            build_download_metadata_payload(
                signature=signature,
                raw_root=raw_dir,
                explorer_csv_file=explorer_csv_file,
                citation_txt_file=citation_path,
                ar6_public_explorer=_ar6_public_explorer_metadata(),
                historical_sources=historical_sources,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        report.errors["download_ar6"] = f"{exc}"
        return report
    finally:
        status.finish()

    return report


def download_ar6(
    *,
    refresh: bool = False,
    manager_url: str = DEFAULT_MANAGER_URL,
) -> DownloadReportAR6 | None:
    """Download the raw datasets required for dynamic AR6 climate change
     carrying capacity processing.

    The function retrieves AR6 public scenario explorer table together with
    the historical PRIMAP and Global Carbon Budget datasets used to construct
    the historical GHG and CO2 historicalbaselines in ``process_ar6(...)``.
    AR6 categories included: ``C1-C4``; SSPs included: ``SSP1-SSP5``.

    Raw source files are written under
    ``data_raw/carrying_capacities/dynamic_climate_change_ar6``. A single metadata JSON
    describing the raw output signature and source provenance is written under
    ``data_raw/logs``. When all required raw outputs already exist on disk and
    match the stored metadata, the function performs no work and prints
    nothing. Omit arguments to use their default.

    Args:
        refresh: If ``True``, clears only the AR6 raw output scope, then
            downloads it again. Processed AR6 outputs, dynamic carrying
            capacity outputs, and project outputs are not refreshed. Defaults
            to ``False``.
        manager_url: IIASA Scenario Explorer manager endpoint.

    Returns:
        ``DownloadReportAR6`` when the function downloads or refreshes at
        least one raw source, or when the run fails after an error is captured
        on the report. Returns ``None`` when all required raw outputs already
        exist on disk and no work is needed.

    Raises:
        RuntimeError: If existing raw files are inconsistent with missing or
            unreadable metadata.
        ValueError: If remote payloads are structurally invalid.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

    Example:
        Download AR6 climate change pathways::

            from pyaesa import download_ar6

            download_ar6()
    """
    return _run_download_ar6(
        refresh=refresh,
        manager_url=manager_url,
    )
