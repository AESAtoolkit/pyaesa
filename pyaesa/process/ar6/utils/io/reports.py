"""Report objects for AR6 climate processing."""

from dataclasses import dataclass, field
from collections.abc import Iterator
from pathlib import Path

from pyaesa.shared.runtime.reporting.composite_phase_index import PHASE_B1_AR6_DYNAMIC_CC
from pyaesa.shared.runtime.reporting.ar6_process_coverage import process_ar6_coverage_line
from pyaesa.shared.runtime.reporting.output_inventory import (
    OutputInventoryItem,
    inventory_item,
    inventory_lines,
)
from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    labelled_values_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.reuse_status import public_reuse_status
from pyaesa.shared.runtime.reporting.summary import document, render_summary, section

from .report_summaries import (
    VariableCoverageSummaryAR6,
    summarize_variable_model_scenario_pairs,
)


def build_process_report(
    *,
    study_period: list[int],
    categories: list[str],
    ssps: list[int],
    harmonization: bool,
    harmonization_method: str | None,
    processed_dir: Path,
    logs_dir: Path,
    figures_dir: Path | None,
    output_file: Path,
    log_file: Path | None,
    dropped_rows_csv_file: Path,
    process_meta_file: Path,
    figures_meta_file: Path | None,
    variable_coverage_summary_counts: dict[str, dict[str, object]] | None = None,
    latest_historical_year: int | None = None,
    harmonization_year_requested: int | None = None,
    harmonization_year: int | None = None,
    harmonization_year_message: str | None = None,
    figure_files: list[str] | None = None,
    figure_guide_file: Path | None = None,
    template_csv_path: Path | None = None,
    reuse_status: str = "computed",
) -> "ProcessReportAR6":
    """Build the user facing AR6 processing report."""
    variable_coverage_summaries = summarize_variable_model_scenario_pairs(
        variable_coverage_summary_counts
    )
    return ProcessReportAR6(
        study_period=list(study_period),
        categories=list(categories),
        ssps=[int(value) for value in ssps],
        harmonization=bool(harmonization),
        harmonization_method=None if harmonization_method is None else str(harmonization_method),
        processed_dir=processed_dir,
        logs_dir=logs_dir,
        figures_dir=figures_dir,
        output_file=output_file,
        harmonization_log_file=log_file,
        dropped_rows_csv_file=dropped_rows_csv_file,
        process_metadata_path=process_meta_file,
        figures_metadata_path=figures_meta_file,
        figure_files=[Path(path) for path in (figure_files or [])],
        figure_guide_files=[] if figure_guide_file is None else [figure_guide_file],
        template_csv_path=template_csv_path,
        variable_coverage_summaries=variable_coverage_summaries,
        latest_historical_year=latest_historical_year,
        harmonization_year_requested=harmonization_year_requested,
        harmonization_year=harmonization_year,
        harmonization_year_message=harmonization_year_message,
        reuse_status=reuse_status,
    )


@dataclass
class ProcessReportAR6:
    """Outcome summary for one AR6 processing run."""

    study_period: list[int]
    categories: list[str]
    ssps: list[int]
    harmonization: bool
    harmonization_method: str | None
    processed_dir: Path
    logs_dir: Path
    figures_dir: Path | None = None
    output_file: Path | None = None
    harmonization_log_file: Path | None = None
    dropped_rows_csv_file: Path | None = None
    process_metadata_path: Path | None = None
    figures_metadata_path: Path | None = None
    figure_files: list[Path] = field(default_factory=list)
    figure_guide_files: list[Path] = field(default_factory=list)
    template_csv_path: Path | None = None
    variable_coverage_summaries: list[VariableCoverageSummaryAR6] = field(default_factory=list)
    latest_historical_year: int | None = None
    harmonization_year_requested: int | None = None
    harmonization_year: int | None = None
    harmonization_year_message: str | None = None
    reuse_status: str = "computed"

    def _format_categories(self) -> str:
        categories = [str(value).strip() for value in self.categories if str(value).strip()]
        if categories == ["C1", "C2", "C3", "C4"]:
            return "C1-C4"
        return ", ".join(categories)

    def _format_ssps(self) -> str:
        ssps = sorted({int(value) for value in self.ssps})
        if ssps == [1, 2, 3, 4, 5]:
            return "1-5"
        return ", ".join(str(value) for value in ssps)

    def __str__(self) -> str:
        period = f"{int(self.study_period[0])}-{int(self.study_period[1])}"
        coverage_lines = ["Processed pathway coverage:"]
        if not self.variable_coverage_summaries:
            coverage_lines.append(
                "No variable specific model-scenario pair summaries were recorded."
            )
        for entry in self.variable_coverage_summaries:
            coverage_lines.append(
                process_ar6_coverage_line(
                    variable=entry.variable,
                    retained=entry.retained_model_scenario_pairs,
                )
            )
        output_lines = [
            f"Output folder: {self.processed_dir}",
            output_files_available_line(self._public_output_file_count()),
            *inventory_lines(self._inventory_items()),
        ]
        if self.figure_files:
            output_lines.append(figures_available_line(len(self.figure_files)))
        ssp_values = [f"SSP{value}" for value in sorted({int(value) for value in self.ssps})]
        run_lines = [
            f"Run status: {public_reuse_status(self.reuse_status)}",
            f"Study period: {period}",
            labelled_values_line(
                "AR6 category",
                "AR6 categories",
                tuple(self.categories),
                self._format_categories(),
            ),
            labelled_values_line(
                "SSP scenario",
                "SSP scenarios",
                tuple(ssp_values),
                self._format_ssps(),
            ),
            f"Harmonization: {self.harmonization}",
        ]
        if self.harmonization and self.harmonization_method is not None:
            run_lines.append(f"Harmonization method: {self.harmonization_method}")
        if self.harmonization_year_message:
            run_lines.append(self.harmonization_year_message)
        return render_summary(
            document(
                "process_ar6",
                lines=tuple(run_lines),
                sections=(
                    section(
                        PHASE_B1_AR6_DYNAMIC_CC,
                        children=(
                            section(
                                "process_ar6",
                                lines=(*coverage_lines, *output_lines),
                            ),
                        ),
                    ),
                ),
            )
        )

    def _inventory_items(self) -> Iterator[OutputInventoryItem]:
        """Yield public output inventory items for this process AR6 report."""
        if self.output_file is not None:
            yield inventory_item(folder="results", content="processed AR6 workbook")
        if self.dropped_rows_csv_file is not None:
            yield inventory_item(folder="logs", content="dropped row audit when produced")
        if self.harmonization_log_file is not None:
            yield inventory_item(folder="logs", content="harmonization log")
        if self.template_csv_path is not None:
            yield inventory_item(
                folder="interpretation",
                content="model-scenario subset template",
            )
        yield inventory_item(folder="interpretation", content="README worksheets")
        yield inventory_item(folder="interpretation", content="model-scenario guides")
        if self.figure_guide_files:
            yield inventory_item(folder="interpretation", content="figure guide")

    def _public_output_file_count(self) -> int:
        """Return the number of public files available in the output folder."""
        paths = [
            self.output_file,
            self.harmonization_log_file,
            self.dropped_rows_csv_file,
            self.template_csv_path,
            *self.figure_files,
            *self.figure_guide_files,
        ]
        return sum(1 for path in paths if path is not None)

    __repr__ = __str__
