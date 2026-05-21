"""Report objects for deterministic_ar6_cc."""

from dataclasses import dataclass, field
from pathlib import Path

from pyaesa.shared.runtime.text import extend_user_text_lines
from pyaesa.shared.runtime.reporting.ar6_process_coverage import (
    process_ar6_coverage_lines_from_payload,
)
from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    labelled_values_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.reuse_status import public_reuse_status
from pyaesa.shared.runtime.reporting.summary import (
    SummaryDocument,
    SummarySection,
    SummaryWarning,
    document,
    render_summary,
    section,
    warning,
)
from pyaesa.shared.runtime.reporting.values import (
    as_sequence,
    format_report_value,
    format_ssp_value,
)


@dataclass(frozen=True)
class AR6CCPathwayCount:
    """Retained model-scenario count for one AR6 CC category and SSP."""

    category: str
    ssp_scenario: str
    model_scenario_pairs: int


@dataclass
class ComputeAR6CCReport:
    """Outcome summary for one deterministic_ar6_cc run."""

    study_period: list[int]
    harmonization: bool
    harmonization_method: str
    emission_type: str
    include_afolu: bool
    emissions_mode: str
    variable: str
    categories: list[str]
    ssp_scenarios: list[str]
    subset_version: str | None
    total_model_scenario_pairs: int
    output_file: Path
    process_ar6: dict[str, object]
    pathway_counts: list[AR6CCPathwayCount] = field(default_factory=list)
    missing_pathway_combinations: list[AR6CCPathwayCount] = field(default_factory=list)
    post_study_output_file: Path | None = None
    figure_paths: list[Path] = field(default_factory=list)
    meta_file: Path | None = None
    cc_dir: Path | None = None
    logs_dir: Path | None = None
    reuse_status: str = "computed"

    def __str__(self) -> str:
        """Return a human-readable run summary."""
        return render_summary(_ar6_cc_summary_document(report=self))

    __repr__ = __str__

    def _public_output_file_count(self) -> int:
        paths = [self.output_file, self.post_study_output_file, *self.figure_paths]
        if self.logs_dir is not None:
            paths.append(self.logs_dir / "summary.log")
        return sum(1 for path in paths if path is not None)


def _ar6_cc_summary_document(*, report: ComputeAR6CCReport) -> SummaryDocument:
    return document(
        "deterministic_ar6_cc",
        lines=_ar6_cc_common_lines(report=report),
        sections=(
            section(
                "Phase B.1: Dynamic AR6 CC",
                children=tuple(_ar6_cc_phase_children(report=report)),
            ),
        ),
    )


def _ar6_cc_common_lines(*, report: ComputeAR6CCReport) -> list[str]:
    return [
        f"Run status: {public_reuse_status(report.reuse_status)}",
        f"Emission type: {report.emission_type}",
        f"Includes AFOLU: {report.include_afolu}",
        f"Emissions mode: {report.emissions_mode}",
    ]


def _ar6_cc_phase_children(*, report: ComputeAR6CCReport) -> list[SummarySection]:
    children = [
        section("process_ar6", lines=_process_ar6_summary_lines(report.process_ar6)),
        section(
            "deterministic_ar6_cc",
            lines=_ar6_cc_function_lines(report=report),
            warnings=tuple(_ar6_cc_pathway_warnings(report=report)),
        ),
    ]
    return children


def _ar6_cc_function_lines(*, report: ComputeAR6CCReport) -> list[str]:
    lines = [
        labelled_values_line(
            "AR6 category",
            "AR6 categories",
            tuple(report.categories),
            ", ".join(report.categories),
        ),
        labelled_values_line(
            "SSP scenario",
            "SSP scenarios",
            tuple(report.ssp_scenarios),
            ", ".join(str(s) for s in report.ssp_scenarios),
        ),
    ]
    if report.subset_version is not None:
        lines.append(f"Subset version: {report.subset_version}")
    lines.append(f"Total retained AR6 CC pathways: {report.total_model_scenario_pairs}")
    if report.pathway_counts:
        lines.append("Retained AR6 CC pathways by category and SSP:")
        for item in report.pathway_counts:
            lines.append(
                f"  {item.category} / {item.ssp_scenario}: "
                f"{item.model_scenario_pairs} model-scenario pairs"
            )
    output_dir = report.output_file.parent if report.cc_dir is None else report.cc_dir
    lines.append(f"Output folder: {output_dir}")
    lines.append(output_files_available_line(report._public_output_file_count()))
    if report.figure_paths:
        lines.append(figures_available_line(len(report.figure_paths)))
    lines.extend(_ar6_cc_inventory_lines(report=report))
    return lines


def _ar6_cc_inventory_lines(*, report: ComputeAR6CCReport) -> list[str]:
    inventory = [inventory_item(folder="results", content="deterministic AR6 CC table")]
    if report.post_study_output_file is not None:
        inventory.append(inventory_item(folder="results", content="post study period AR6 CC table"))
    if report.logs_dir is not None:
        inventory.append(inventory_item(folder="logs", content="summary log"))
    return list(inventory_lines(inventory))


def _ar6_cc_pathway_warnings(*, report: ComputeAR6CCReport) -> list[SummaryWarning]:
    start_year = int(report.study_period[0])
    end_year = int(report.study_period[1])
    n_years = end_year - start_year + 1
    period = f"{start_year}-{end_year} ({n_years} year(s))"
    return [
        warning(
            "No AR6 CC model-scenario pair matches the requested "
            f"category {item.category}, SSP {item.ssp_scenario}, emissions mode "
            f"{report.emissions_mode}, variable {report.variable}, and study window {period}."
        )
        for item in report.missing_pathway_combinations
    ]


def _process_ar6_summary_lines(payload: dict[str, object]) -> list[str]:
    """Return structured process_ar6 prerequisite information."""
    lines: list[str] = []
    for label, key in (
        ("Run status", "reuse_status"),
        ("Study period", "study_period"),
        ("Harmonization", "harmonization"),
        ("Harmonization method", "harmonization_method"),
        ("Output folder", "output_root"),
    ):
        value = payload.get(key)
        if value is not None:
            text = (
                public_reuse_status(str(value))
                if key == "reuse_status"
                else format_report_value(value)
            )
            lines.append(f"{label}: {text}")
        if key == "study_period":
            categories = payload.get("categories")
            if categories is not None:
                values = tuple(str(item) for item in as_sequence(categories))
                lines.append(
                    labelled_values_line(
                        "AR6 category",
                        "AR6 categories",
                        values,
                        format_report_value(values),
                    )
                )
    ssps = payload.get("ssps")
    if isinstance(ssps, list | tuple) and ssps:
        values = [format_ssp_value(item) for item in ssps]
        extend_user_text_lines(
            lines,
            labelled_values_line(
                "SSP scenario",
                "SSP scenarios",
                tuple(values),
                format_report_value(values),
            ),
        )
    coverage = payload.get("variable_coverage")
    if isinstance(coverage, list | tuple) and coverage:
        lines.append("Processed pathway coverage:")
        lines.extend(process_ar6_coverage_lines_from_payload(coverage))
    if payload.get("harmonization_year_message"):
        lines.append(str(payload["harmonization_year_message"]))
    figures_available = payload.get("figures_available")
    if figures_available is not None:
        lines.append(figures_available_line(int(str(figures_available))))
    output_files_available = payload.get("output_files_available")
    if output_files_available is not None:
        lines.append(output_files_available_line(int(str(output_files_available))))
    return lines
