"""Dynamic AR6 CC subsection rendering for uncertainty run reports."""

from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    labelled_values_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.ar6_process_coverage import (
    process_ar6_coverage_lines_from_payload,
)
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.shared.runtime.reporting.summary import SummarySection, section, warning
from pyaesa.shared.runtime.reporting.values import (
    as_sequence,
    format_report_value,
    format_ssp_value,
    format_summary_value,
    format_values,
)
from pyaesa.shared.uncertainty_assessment.run_state.report_messages import (
    payload_infos,
    payload_warnings,
)


def ar6_cc_dependency_section(*, payload: dict[str, Any]) -> SummarySection:
    """Return the deterministic dynamic AR6 CC dependency subsection."""
    lines = _ar6_cc_detail_lines(payload=payload)
    warnings = tuple(
        warning(
            "Requested category and SSP combination has no retained AR6 CC pathway: "
            + _format_pathway_count(item=item, suffix="matching model-scenario pairs")
        )
        for item in payload.get("missing_pathway_combinations") or ()
    )
    return section(
        "deterministic_ar6_cc",
        lines=lines,
        infos=payload_infos(payload=payload),
        warnings=(*warnings, *payload_warnings(payload=payload)),
    )


def process_ar6_section(*, payload: dict[str, Any]) -> SummarySection:
    """Return the process_ar6 subsection carried by dynamic AR6 CC payloads."""
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
            lines.append(f"{label}: {format_summary_value(key=key, value=value)}")
        if key == "study_period":
            categories = payload.get("categories")
            if categories is not None:
                values = as_sequence(categories)
                lines.append(
                    labelled_values_line(
                        "AR6 category",
                        "AR6 categories",
                        values,
                        format_values(values),
                    )
                )
    ssps = payload.get("ssps")
    if isinstance(ssps, list | tuple) and ssps:
        values = tuple(format_ssp_value(item) for item in ssps)
        lines.append(
            labelled_values_line(
                "SSP scenario",
                "SSP scenarios",
                values,
                format_values(values),
            )
        )
    coverage = payload.get("variable_coverage")
    if coverage:
        lines.append("Processed pathway coverage:")
        lines.extend(process_ar6_coverage_lines_from_payload(coverage))
    if payload.get("harmonization_year_message"):
        lines.append(str(payload["harmonization_year_message"]))
    if payload.get("figures_available") is not None:
        lines.append(figures_available_line(int(payload["figures_available"])))
    if payload.get("output_files_available") is not None:
        lines.append(output_files_available_line(int(payload["output_files_available"])))
    return section(
        "process_ar6",
        lines=lines,
        infos=payload_infos(payload=payload),
        warnings=payload_warnings(payload=payload),
    )


def _ar6_cc_detail_lines(*, payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for label, key in (
        ("Run status", "reuse_status"),
        ("Emission type", "emission_type"),
        ("Includes AFOLU", "include_afolu"),
        ("Emissions mode", "emissions_mode"),
    ):
        value = payload.get(key)
        if value is not None:
            lines.append(f"{label}: {format_summary_value(key=key, value=value)}")
        if key == "reuse_status":
            categories = payload.get("categories")
            if categories is not None:
                values = as_sequence(categories)
                lines.append(
                    labelled_values_line(
                        "AR6 category",
                        "AR6 categories",
                        values,
                        format_values(values),
                    )
                )
    ssp_scenarios = payload.get("ssp_scenarios")
    if ssp_scenarios is not None:
        scenarios = as_sequence(ssp_scenarios)
        lines.append(
            labelled_values_line(
                "SSP scenario",
                "SSP scenarios",
                scenarios,
                format_values(scenarios),
            )
        )
    output_root = _payload_output_root(payload=payload)
    if output_root is not None:
        lines.append(f"Output folder: {output_root}")
    output_count = _payload_public_output_count(payload=payload)
    if output_count:
        lines.append(output_files_available_line(output_count))
    lines.extend(_payload_inventory_lines(payload=payload))
    subset_version = payload.get("subset_version")
    if subset_version is not None:
        lines.append(f"Subset version: {format_report_value(subset_version)}")
    pathway_counts = list(payload.get("pathway_counts") or ())
    if pathway_counts:
        lines.append("Retained AR6 CC pathways by category and SSP:")
        lines.extend(
            "  " + _format_pathway_count(item=item, suffix="model-scenario pairs")
            for item in pathway_counts
        )
    return lines


def _format_pathway_count(*, item: object, suffix: str) -> str:
    payload = dict(item) if isinstance(item, dict) else {}
    return (
        f"{payload.get('category')} / {payload.get('ssp_scenario')}: "
        f"{payload.get('model_scenario_pairs')} {suffix}"
    )


def _payload_output_root(*, payload: dict[str, Any]) -> Path | None:
    for key in ("metadata_path", "scope_manifest"):
        value = payload.get(key)
        if value is not None:
            return public_output_root_from_path(Path(str(value)))
    return None


def _payload_inventory_lines(*, payload: dict[str, Any]) -> list[str]:
    items = []
    if payload.get("output_file") is not None or payload.get("deterministic_paths"):
        items.append(inventory_item(folder="results", content="result tables"))
    if payload.get("post_study_output_file") is not None:
        items.append(inventory_item(folder="results", content="post study period result table"))
    if payload.get("metadata_path") is not None or payload.get("scope_manifest") is not None:
        items.append(inventory_item(folder="logs", content="summary log"))
    return list(inventory_lines(items))


def _payload_public_output_count(*, payload: dict[str, Any]) -> int:
    paths: set[str] = set()
    for key in ("output_file", "post_study_output_file", "deterministic_paths", "figure_paths"):
        value = payload.get(key)
        if isinstance(value, list | tuple):
            paths.update(str(item).strip() for item in value if str(item).strip())
        elif value is not None and str(value).strip():
            paths.add(str(value).strip())
    if payload.get("metadata_path") is not None or payload.get("scope_manifest") is not None:
        paths.add("summary.log")
    return len(paths)
