"""Processed MRIO report integration for raw correction logs."""

from pathlib import Path

from pyaesa.process.mrios.utils.pipeline.contracts import ProcessReportMRIO

from .runtime import (
    AppliedCorrectionSummary,
    load_raw_corrected_value_rows,
    summarize_correction_rows,
    summarize_correction_scopes,
    write_applied_correction_log,
)


def record_raw_correction_payload(
    *,
    iosys: object,
    report: ProcessReportMRIO,
    source_key: str,
    matrix_version: str | None,
    saved_dir: Path,
    year: int,
) -> dict[str, object] | None:
    """Persist raw correction log rows and update the process report."""
    correction_summary = getattr(iosys, "_raw_corrected_values_summary", None)
    if not isinstance(correction_summary, AppliedCorrectionSummary):
        return None
    year_rows = load_raw_corrected_value_rows(source=source_key, year=year)
    log_path = write_applied_correction_log(
        source_key=source_key,
        matrix_version=matrix_version,
        saved_dir=saved_dir,
        year_rows=year_rows,
    )
    report.raw_corrected_value_row_count += int(correction_summary.row_count)
    summary_lines = summarize_correction_rows(year_rows=year_rows)
    for scope in summarize_correction_scopes(year_rows=year_rows):
        report.raw_corrected_value_scopes.append(scope)
    if log_path is not None and log_path not in report.raw_corrected_value_log_paths:
        report.raw_corrected_value_log_paths.append(log_path)
    return {
        "row_count": int(correction_summary.row_count),
        "log_path": str(log_path) if log_path is not None else "",
        "summary_lines": summary_lines,
    }
