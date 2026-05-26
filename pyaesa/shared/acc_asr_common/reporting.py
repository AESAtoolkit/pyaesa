"""Shared summary text helpers for deterministic aCC and ASR reports."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    labelled_values_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.ar6_process_coverage import (
    process_ar6_coverage_lines_from_payload,
)
from pyaesa.shared.runtime.reporting.reuse_status import public_reuse_status
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    public_phase_index_reuse_status,
)
from pyaesa.shared.runtime.reporting.summary import SummarySection, info, section, warning
from pyaesa.shared.runtime.reporting.values import (
    as_sequence,
    format_report_value,
    format_ssp_value,
    format_values,
)
from pyaesa.shared.runtime.reporting.year_ranges import format_year_ranges


@dataclass(frozen=True)
class DynamicAR6PathwayCount:
    """Retained model-scenario count for one dynamic AR6 CC selector."""

    category: str
    ssp_scenario: str
    model_scenario_pairs: int


@dataclass(frozen=True)
class DynamicAR6Summary:
    """Dynamic AR6 CC facts shown by downstream deterministic summaries."""

    categories: list[str]
    ssp_scenarios: list[str]
    subset_version: str | None
    pathway_counts: list[DynamicAR6PathwayCount]
    missing_pathway_combinations: list[DynamicAR6PathwayCount]
    study_period: list[int] | None = None
    harmonization: bool | None = None
    harmonization_method: str | None = None
    emission_type: str | None = None
    include_afolu: bool | None = None
    emissions_mode: str | None = None
    variable: str | None = None
    total_model_scenario_pairs: int | None = None
    process_ar6: dict[str, Any] | None = None


def format_dynamic_ar6_summary_lines(summary: DynamicAR6Summary) -> list[str]:
    """Return user facing summary lines for one dynamic AR6 CC prerequisite."""
    lines = []
    if summary.emission_type is not None:
        lines.append(f"Emission type: {summary.emission_type}")
    if summary.include_afolu is not None:
        lines.append(f"Includes AFOLU: {summary.include_afolu}")
    if summary.emissions_mode is not None:
        lines.append(f"Emissions mode: {summary.emissions_mode}")
    lines.extend(
        [
            _ar6_categories_line(summary.categories),
            _ssp_scenarios_line(summary.ssp_scenarios),
        ]
    )
    if summary.subset_version is not None:
        lines.append(f"Subset version: {summary.subset_version}")
    if summary.total_model_scenario_pairs is not None:
        lines.append(f"Total retained AR6 CC pathways: {summary.total_model_scenario_pairs}")
    if summary.pathway_counts:
        lines.append("Retained AR6 CC pathways by category and SSP:")
        lines.extend(
            "  " + _format_pathway_count(item, suffix="model-scenario pairs")
            for item in summary.pathway_counts
        )
    for item in summary.missing_pathway_combinations:
        lines.append(
            "WARNING: no retained AR6 CC pathway for "
            + _format_pathway_count(item, suffix="matching model-scenario pairs")
        )
    return lines


def format_process_ar6_payload_lines(payload: dict[str, Any]) -> list[str]:
    """Return process AR6 payload lines for downstream phase summaries."""
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
                lines.append(_ar6_categories_line(values))
    ssps = payload.get("ssps")
    if isinstance(ssps, list | tuple) and ssps:
        values = [format_ssp_value(item) for item in ssps]
        lines.append(
            labelled_values_line(
                "SSP scenario",
                "SSP scenarios",
                tuple(values),
                format_report_value(values),
            )
        )
    coverage = payload.get("variable_coverage")
    if isinstance(coverage, list | tuple) and coverage:
        lines.append("Processed pathway coverage:")
        lines.extend(process_ar6_coverage_lines_from_payload(coverage))
    if payload.get("harmonization_year_message"):
        lines.append(str(payload["harmonization_year_message"]))
    figures_available = payload.get("figures_available")
    if figures_available is not None:
        lines.append(figures_available_line(int(figures_available)))
    output_files_available = payload.get("output_files_available")
    if output_files_available is not None:
        lines.append(output_files_available_line(int(output_files_available)))
    return lines


def dynamic_cc_scope_label(branch_cc_type: str, summary: DynamicAR6Summary | None) -> str:
    """Return the public CC scope label for deterministic aCC and ASR summaries."""
    if branch_cc_type != "dynamic_ar6" or summary is None:
        return ""
    return _ar6_categories_line(summary.categories)


def composite_phase_sections(
    *,
    entries: tuple[CompositePhaseIndexEntry, ...],
    dynamic_ar6_summary: DynamicAR6Summary | None = None,
    extra_lines_by_function: Mapping[str, Sequence[str]] | None = None,
) -> tuple[SummarySection, ...]:
    """Return structured phase sections from persisted composite phase entries."""
    sections: list[SummarySection] = []
    active_phase: str | None = None
    active_children: list[SummarySection] = []
    process_ar6_rendered = False
    extras = extra_lines_by_function or {}

    def flush_active_phase() -> None:
        if active_phase is not None:
            sections.append(section(active_phase, children=tuple(active_children)))

    for entry in entries:
        if entry.phase != active_phase:
            flush_active_phase()
            active_phase = entry.phase
            active_children = []
            process_ar6_rendered = False
        if (
            entry.function == "deterministic_ar6_cc"
            and dynamic_ar6_summary is not None
            and dynamic_ar6_summary.process_ar6 is not None
            and not process_ar6_rendered
        ):
            active_children.append(
                section(
                    "process_ar6",
                    lines=format_process_ar6_payload_lines(dynamic_ar6_summary.process_ar6),
                )
            )
            process_ar6_rendered = True
        active_children.append(
            _composite_phase_entry_section(
                entry=entry,
                dynamic_ar6_summary=dynamic_ar6_summary,
                extra_lines=tuple(extras.get(entry.function, ())),
            )
        )
    flush_active_phase()
    return tuple(sections)


def _composite_phase_entry_section(
    *,
    entry: CompositePhaseIndexEntry,
    dynamic_ar6_summary: DynamicAR6Summary | None,
    extra_lines: Sequence[str],
) -> SummarySection:
    lines: list[str] = [f"Run status: {public_phase_index_reuse_status(entry.reuse_status)}"]
    if entry.output_root is not None:
        lines.append(f"Output folder: {entry.output_root}")
    lines.extend(entry.summary_lines)
    lines.extend(extra_lines)
    if entry.function == "deterministic_ar6_cc" and dynamic_ar6_summary is not None:
        lines.extend(format_dynamic_ar6_summary_lines(dynamic_ar6_summary))
    lines.extend(entry.inventory_lines)
    return section(
        entry.function,
        lines=lines,
        infos=tuple(info(message) for message in entry.info_messages),
        warnings=tuple(warning(message) for message in entry.warning_messages),
    )


def build_downstream_common_scope_lines(
    *,
    project_name: str,
    years: list[int],
    lcia_methods: list[str],
    fu_code: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
    group_indices: bool,
    ssp_scenarios: list[str] | None = None,
    lca_route: str | None = None,
) -> list[str]:
    """Return common user scope lines for deterministic downstream composites."""
    lines = [
        f"Project: {project_name}",
        _studied_years_line(years),
        _lcia_methods_line(lcia_methods),
        f"Functional unit: {fu_code}",
    ]
    if ssp_scenarios:
        lines.append(_ssp_scenarios_line(ssp_scenarios))
    if lca_route is not None:
        lines.append(f"LCA route: {lca_route}")
    return lines


def _studied_years_line(years: Sequence[int]) -> str:
    values = [int(year) for year in years]
    return labelled_values_line(
        "Studied year",
        "Studied years",
        values,
        format_year_ranges(values),
    )


def _lcia_methods_line(lcia_methods: Sequence[str]) -> str:
    return labelled_values_line(
        "LCIA method",
        "LCIA methods",
        tuple(lcia_methods),
        format_values(lcia_methods),
    )


def _ssp_scenarios_line(ssp_scenarios: Sequence[str]) -> str:
    return labelled_values_line(
        "SSP scenario",
        "SSP scenarios",
        tuple(ssp_scenarios),
        format_values(ssp_scenarios),
    )


def _ar6_categories_line(categories: Sequence[str]) -> str:
    return labelled_values_line(
        "AR6 category",
        "AR6 categories",
        tuple(categories),
        format_values(categories),
    )


def _format_pathway_count(item: DynamicAR6PathwayCount, *, suffix: str) -> str:
    return f"{item.category} / {item.ssp_scenario}: {int(item.model_scenario_pairs)} {suffix}"
