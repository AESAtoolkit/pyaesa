"""Reuse path ownership for saved AR6 processed outputs."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pyaesa.download.ar6.utils.io.paths import explorer_csv_path_for_raw_dir
from pyaesa.download.ar6.utils.sources.explorer_csv import read_explorer_csv
from pyaesa.shared.runtime.reporting.status import StatusSink, TransientStatusPrinter

from ..figures.figure_guides import ensure_figures_guide
from ..figures.figure_outputs import ensure_figures, load_saved_figure_files
from ..figures.figure_sampling_config import SamplingFigureConfig
from ..io.report_summaries import (
    deserialize_variable_coverage_summary_counts,
)
from ..io.metadata import variable_coverage_summary_payload
from ..io.reports import ProcessReportAR6, build_process_report
from .loaders import scenario_metadata_from_wide
from .raw_inputs import require_ar6_historical_figure_reference
from .runtime_helpers import show_stage


def reuse_processed_outputs(
    *,
    study_period: list[int],
    figures: bool,
    harmonization: bool,
    harmonization_method: str,
    ssps: list[int],
    figure_output_format: str,
    figure_dpi: int,
    sampling_config: SamplingFigureConfig,
    processed_dir: Path,
    logs_dir: Path,
    figures_dir: Path,
    out_file: Path,
    log_file: Path | None,
    dropped_rows_csv_file: Path,
    process_meta_file: Path,
    figures_meta_file: Path,
    process_meta: dict,
    categories: list[str],
    variables_output: list[str],
    database: str,
    raw_data_dir: Path,
    load_saved_figure_files_func=load_saved_figure_files,
    ensure_figures_func=ensure_figures,
    ensure_figures_guide_func=ensure_figures_guide,
    status: StatusSink | None = None,
) -> ProcessReportAR6:
    """Handle the reuse path when processed outputs already exist."""
    if not figures:
        return _build_reused_process_report(
            study_period=study_period,
            categories=categories,
            ssps=ssps,
            harmonization=harmonization,
            harmonization_method=harmonization_method if harmonization else None,
            processed_dir=processed_dir,
            logs_dir=logs_dir,
            figures_dir=None,
            out_file=out_file,
            log_file=log_file,
            dropped_rows_csv_file=dropped_rows_csv_file,
            process_meta_file=process_meta_file,
            figures_meta_file=None,
            process_meta=process_meta,
            figure_files=[],
            figure_guide_file=None,
            reuse_status="reused_exact",
        )

    saved_figure_files = load_saved_figure_files_func(
        figures_metadata_file=figures_meta_file,
        figures_dir=figures_dir,
        study_period=study_period,
        database=database,
        categories=list(categories),
        variables_output=list(variables_output),
        figure_output_format=figure_output_format,
        figure_dpi=figure_dpi,
        harmonization_method=harmonization_method,
        figure_convergence_tol=sampling_config["relative_tolerance"],
        figure_convergence_max_runs=sampling_config["max_runs_per_bucket"],
    )
    if saved_figure_files is not None:
        figure_guide_file, _guide_written = ensure_figures_guide_func(
            figures_dir=figures_dir,
            figure_files=saved_figure_files,
            study_period=study_period,
            global_drop_csv_file=dropped_rows_csv_file,
            rewrite=False,
        )
        return _build_reused_process_report(
            study_period=study_period,
            categories=categories,
            ssps=ssps,
            harmonization=harmonization,
            harmonization_method=harmonization_method if harmonization else None,
            processed_dir=processed_dir,
            logs_dir=logs_dir,
            figures_dir=figures_dir,
            out_file=out_file,
            log_file=log_file,
            dropped_rows_csv_file=dropped_rows_csv_file,
            process_meta_file=process_meta_file,
            figures_meta_file=figures_meta_file,
            process_meta=process_meta,
            figure_files=saved_figure_files,
            figure_guide_file=figure_guide_file,
            reuse_status="reused_exact",
        )

    figure_guide_file: Path | None = None
    with _status_scope(status) as status_owner:
        show_stage(status_owner, "Generating figures")
        require_ar6_historical_figure_reference(raw_data_dir=raw_data_dir)
        explorer_csv_path = explorer_csv_path_for_raw_dir(raw_data_dir, database)
        explorer = read_explorer_csv(explorer_csv_path)
        source_meta = scenario_metadata_from_wide(explorer.data)
        figure_files, _figures_reused = ensure_figures_func(
            out_file=out_file,
            log_file=log_file,
            figures_dir=figures_dir,
            figures_metadata_file=figures_meta_file,
            logs_dir=logs_dir,
            variables_output=list(variables_output),
            study_period=study_period,
            database=database,
            categories=list(categories),
            figure_output_format=figure_output_format,
            figure_dpi=figure_dpi,
            harmonization_method=harmonization_method,
            figure_convergence_tol=sampling_config["relative_tolerance"],
            figure_convergence_max_runs=sampling_config["max_runs_per_bucket"],
            refresh=False,
            source_metadata=source_meta,
            raw_data_dir=raw_data_dir,
            status_callback=lambda message: show_stage(status_owner, message),
        )
        figure_guide_file, _guide_written = ensure_figures_guide_func(
            figures_dir=figures_dir,
            figure_files=figure_files,
            study_period=study_period,
            global_drop_csv_file=dropped_rows_csv_file,
            rewrite=True,
        )
    return _build_reused_process_report(
        study_period=study_period,
        categories=categories,
        ssps=ssps,
        harmonization=harmonization,
        harmonization_method=harmonization_method if harmonization else None,
        processed_dir=processed_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        out_file=out_file,
        log_file=log_file,
        dropped_rows_csv_file=dropped_rows_csv_file,
        process_meta_file=process_meta_file,
        figures_meta_file=figures_meta_file,
        process_meta=process_meta,
        figure_files=figure_files,
        figure_guide_file=figure_guide_file,
        reuse_status="partially_reused",
    )


@contextmanager
def _status_scope(status: StatusSink | None) -> Iterator[StatusSink]:
    """Yield a caller supplied status sink or own a direct run status sink."""
    if status is not None:
        yield status
        return
    owned = TransientStatusPrinter("process_ar6")
    try:
        yield owned
    finally:
        owned.finish()


def _build_reused_process_report(
    *,
    study_period: list[int],
    categories: list[str],
    ssps: list[int],
    harmonization: bool,
    harmonization_method: str | None,
    processed_dir: Path,
    logs_dir: Path,
    figures_dir: Path | None,
    out_file: Path,
    log_file: Path | None,
    dropped_rows_csv_file: Path,
    process_meta_file: Path,
    figures_meta_file: Path | None,
    process_meta: dict,
    figure_files: list[str],
    figure_guide_file: Path | None,
    reuse_status: str,
) -> ProcessReportAR6:
    """Build the public process report from the persisted processed AR6 scope."""
    provenance = process_meta["provenance"]
    return build_process_report(
        study_period=study_period,
        categories=categories,
        ssps=ssps,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        processed_dir=processed_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        output_file=out_file,
        log_file=log_file,
        dropped_rows_csv_file=dropped_rows_csv_file,
        process_meta_file=process_meta_file,
        figures_meta_file=figures_meta_file,
        variable_coverage_summary_counts=deserialize_variable_coverage_summary_counts(
            variable_coverage_summary_payload(process_meta)
        ),
        latest_historical_year=provenance.get("latest_historical_year"),
        harmonization_year_requested=provenance.get("harmonization_year_requested"),
        harmonization_year=provenance.get("harmonization_year"),
        harmonization_year_message=provenance.get("harmonization_year_message"),
        figure_files=figure_files,
        figure_guide_file=figure_guide_file,
        reuse_status=reuse_status,
    )
