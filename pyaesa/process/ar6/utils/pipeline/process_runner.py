"""Internal orchestration runner for ``process_ar6``."""

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from pyaesa.download.ar6.utils.io.paths import explorer_csv_path_for_raw_dir
from pyaesa.download.ar6.utils.sources.explorer_csv import read_explorer_csv
from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent
from pyaesa.shared.runtime.reporting.summary_log import summary_log_path, write_summary_log
from pyaesa.shared.runtime.reporting.status import StatusSink, TransientStatusPrinter

from ..figures.figure_guides import ensure_figures_guide
from ..figures.figure_outputs import ensure_figures, load_saved_figure_files
from ..figures.figure_sampling_config import SamplingFigureConfig
from ..io.contracts import harmonization_log_workbook_name, processed_workbook_name
from ..io.metadata import build_process_metadata_payload, read_json, signature_matches, write_json
from ..io.paths import get_scope_dirs
from ..io.reports import ProcessReportAR6, build_process_report
from ..io.text_outputs import (
    excel_readme_sheet,
    log_columns_explanation_text,
    processing_citation_text,
)
from ..io.writers import (
    build_dropped_rows_df,
    write_harmonization_log_workbook,
    write_model_scenario_template,
    write_processed_workbook,
)
from .loaders import scenario_metadata_from_wide
from .processing_modes import build_pathway_outputs
from .raw_inputs import require_ar6_historical_figure_reference, require_downloaded_ar6_raw_inputs
from .reuse_outputs import reuse_processed_outputs
from .runtime_helpers import show_stage
from .study_period import validate_study_period_in_ar6


def _write_process_report_summary(report: ProcessReportAR6) -> ProcessReportAR6:
    """Persist and return the public process summary for one AR6 scope."""
    write_summary_log(path=summary_log_path(logs_dir=report.logs_dir), summary=str(report))
    return report


def run_process_ar6_workflow(
    *,
    study_period: list[int],
    figures: bool,
    harmonization: bool,
    harmonization_method: str,
    refresh: bool,
    figure_output_format: str,
    figure_dpi: int,
    sampling_config: SamplingFigureConfig,
    signature: dict[str, object],
    categories: list[str],
    ssps: list[int],
    variables_output: list[str],
    database: str,
    raw_data_dir: Path,
    citation_txt_path: Path,
    ensure_figures_func=ensure_figures,
    ensure_figures_guide_func=ensure_figures_guide,
    load_saved_figure_files_func=load_saved_figure_files,
    status: StatusSink | None = None,
) -> ProcessReportAR6:
    """Run the AR6 processing workflow with explicit dependency inputs."""
    processed_dir, logs_dir, figures_dir = get_scope_dirs(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
    )
    out_file = processed_dir / processed_workbook_name(harmonization=harmonization)
    log_file = logs_dir / harmonization_log_workbook_name() if harmonization else None
    dropped_rows_csv_file = logs_dir / "dropped_model_scenario_variable_rows.csv"
    process_meta_file = logs_dir / "scope_manifest.json"
    figures_meta_file = logs_dir / "figure.json"
    process_citation_txt_file = processed_dir / "recommended_citations_data_sources_and_usage.txt"
    logs_columns_txt_file = logs_dir / "log_columns_explanation.txt"

    if refresh:
        for path in [processed_dir, logs_dir, figures_dir]:
            if path.exists():
                shutil.rmtree(path)

    processed_dir = ensure_dir(processed_dir)
    logs_dir = ensure_dir(logs_dir)
    process_meta = read_json(process_meta_file)
    if process_meta is None and (out_file.exists() or (log_file is not None and log_file.exists())):
        existing = [
            str(path) for path in (out_file, log_file) if path is not None and path.exists()
        ]
        raise RuntimeError(
            "Processed AR6 files exist but process metadata is missing. "
            f"Metadata={process_meta_file}. Existing artifacts={existing}. "
            "Rerun process_ar6(..., refresh=True) to rebuild this processed AR6 scope."
        )

    can_reuse_process = (
        out_file.exists() and signature_matches(process_meta, signature) and not refresh
    )
    if can_reuse_process:
        return _write_process_report_summary(
            reuse_processed_outputs(
                study_period=study_period,
                figures=figures,
                harmonization=harmonization,
                harmonization_method=harmonization_method,
                ssps=ssps,
                figure_output_format=figure_output_format,
                figure_dpi=figure_dpi,
                sampling_config=sampling_config,
                processed_dir=processed_dir,
                logs_dir=logs_dir,
                figures_dir=figures_dir,
                out_file=out_file,
                log_file=log_file,
                dropped_rows_csv_file=dropped_rows_csv_file,
                process_meta_file=process_meta_file,
                figures_meta_file=figures_meta_file,
                process_meta=cast(dict, process_meta),
                categories=categories,
                variables_output=variables_output,
                database=database,
                raw_data_dir=raw_data_dir,
                load_saved_figure_files_func=load_saved_figure_files_func,
                ensure_figures_func=ensure_figures_func,
                ensure_figures_guide_func=ensure_figures_guide_func,
                status=status,
            )
        )

    with _status_scope(status) as status_owner:
        show_stage(status_owner, "Loading raw inputs")
        require_downloaded_ar6_raw_inputs(
            raw_data_dir=raw_data_dir,
            citation_txt_path=citation_txt_path,
            database=database,
        )
        if figures:
            require_ar6_historical_figure_reference(raw_data_dir=raw_data_dir)
        explorer_csv_path = explorer_csv_path_for_raw_dir(raw_data_dir, database)
        explorer = read_explorer_csv(explorer_csv_path)
        ar6_years = sorted([int(col) for col in explorer.data.columns if str(col).isdigit()])
        validate_study_period_in_ar6(study_period, ar6_years)
        source_meta = scenario_metadata_from_wide(explorer.data)
        models_relevant_all = sorted(list(set(source_meta.index.get_level_values(0))))
        raw_citation_text = citation_txt_path.read_text(encoding="utf-8")
        processed_citation_text = processing_citation_text(raw_citation_text, harmonization)
        show_stage(
            status_owner,
            "Preparing harmonized scenario pathways"
            if harmonization
            else "Preparing scenario pathways",
        )
        pathway_outputs = build_pathway_outputs(
            explorer=explorer,
            categories=list(categories),
            ssps=list(ssps),
            variables_output=list(variables_output),
            study_period=study_period,
            database_raw_dir=raw_data_dir,
            models_relevant_all=models_relevant_all,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
        )
        final_all = pathway_outputs["final_all"]
        original_all = pathway_outputs["original_all"]
        harmonization_log_all = pathway_outputs["harmonization_log_all"]
        stats_var = pathway_outputs["stats_var"]
        historical_emissions = pathway_outputs["historical_emissions"]
        drop_logs = pathway_outputs["drop_logs"]
        variable_coverage_summary_counts = pathway_outputs["variable_coverage_summary_counts"]

        show_stage(status_owner, "Writing processed outputs")
        write_processed_workbook(
            harmonization=harmonization,
            output_file=out_file,
            readme_df=excel_readme_sheet(harmonization=harmonization),
            citations_text=processed_citation_text,
            final_all=final_all,
            original_all=original_all,
            source_meta=source_meta,
            stats_var=stats_var,
            historical_emissions=historical_emissions if not historical_emissions.empty else None,
        )
        template_csv_path = write_model_scenario_template(
            final_all=final_all,
            processed_dir=processed_dir,
        )
        ensure_file_parent(process_citation_txt_file).write_text(
            processed_citation_text,
            encoding="utf-8",
        )
        dropped_rows_df = build_dropped_rows_df(drop_logs)
        dropped_rows_df.to_csv(dropped_rows_csv_file, index=False)
        if harmonization and log_file is not None and harmonization_log_all is not None:
            write_harmonization_log_workbook(log_file, harmonization_log_all)
            logs_columns_txt_file.write_text(log_columns_explanation_text(), encoding="utf-8")
        write_json(
            process_meta_file,
            build_process_metadata_payload(
                signature=signature,
                categories=categories,
                ssps=ssps,
                harmonization=harmonization,
                harmonization_method=harmonization_method if harmonization else None,
                latest_historical_year=pathway_outputs["latest_historical_year"],
                requested_harmonization_year=pathway_outputs["requested_harmonization_year"],
                harmonization_year=pathway_outputs["harmonization_year"],
                harmonization_message=pathway_outputs["harmonization_message"],
                processed_dir=processed_dir,
                logs_dir=logs_dir,
                figures_dir=figures_dir,
                output_file=out_file,
                log_file=log_file,
                dropped_rows_csv_file=dropped_rows_csv_file,
                variable_coverage_summary_counts=variable_coverage_summary_counts,
            ),
        )

        figure_files: list[str] = []
        figure_guide_file: Path | None = None
        if figures:
            show_stage(status_owner, "Generating figures")
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
                refresh=True,
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
    return _write_process_report_summary(
        build_process_report(
            study_period=study_period,
            categories=categories,
            ssps=ssps,
            harmonization=harmonization,
            harmonization_method=harmonization_method if harmonization else None,
            processed_dir=processed_dir,
            logs_dir=logs_dir,
            figures_dir=figures_dir if figures else None,
            output_file=out_file,
            log_file=log_file,
            dropped_rows_csv_file=dropped_rows_csv_file,
            process_meta_file=process_meta_file,
            figures_meta_file=figures_meta_file if figures else None,
            variable_coverage_summary_counts=variable_coverage_summary_counts,
            latest_historical_year=pathway_outputs["latest_historical_year"],
            harmonization_year_requested=pathway_outputs["requested_harmonization_year"],
            harmonization_year=pathway_outputs["harmonization_year"],
            harmonization_year_message=pathway_outputs["harmonization_message"],
            figure_files=figure_files,
            figure_guide_file=figure_guide_file,
            template_csv_path=template_csv_path,
        )
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
