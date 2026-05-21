"""Self section rendering for uncertainty run reports."""

from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.reuse_status import public_reuse_status
from pyaesa.shared.runtime.reporting.summary import SummarySection, section
from pyaesa.shared.runtime.reporting.values import format_report_value, format_values
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest
from pyaesa.shared.uncertainty_assessment.run_state.report_arguments import scope_arguments
from pyaesa.shared.uncertainty_assessment.run_state.report_roots import (
    uncertainty_manifest_output_root,
)


def self_section(
    *,
    manifest: UncertaintyManifest,
    warnings,
    reuse_status: str | None = None,
) -> SummarySection:
    """Return the subsection for the uncertainty function itself."""
    artifacts = manifest.artifacts
    lines: list[str] = []
    if reuse_status is not None:
        lines.append(f"Run status: {public_reuse_status(reuse_status)}")
    lines.extend(_owner_scope_lines(manifest=manifest))
    lines.append(f"Run id: {manifest.run_id}")
    output_root = uncertainty_manifest_output_root(manifest)
    lines.append(f"Output folder: {output_root}")
    output_file_count = _public_artifact_count(artifacts=artifacts)
    lines.append(output_files_available_line(output_file_count))
    lines.extend(_artifact_inventory_lines(artifacts=artifacts))
    figure_paths = [Path(str(path)) for path in artifacts.get("figure_paths", [])]
    if figure_paths:
        lines.append(figures_available_line(len(figure_paths)))
    return section(
        f"uncertainty_{manifest.family}",
        lines=lines,
        warnings=warnings,
    )


def _owner_scope_lines(*, manifest: UncertaintyManifest) -> list[str]:
    """Return source and MRIO selector lines owned by uncertainty aSoCC or IO-LCA."""
    if manifest.family not in {"asocc", "io_lca"}:
        return []
    arguments = scope_arguments(manifest=manifest)
    if not arguments:
        return []
    lines: list[str] = []
    source = arguments.get("source")
    if source is not None:
        lines.append(f"Source: {format_report_value(source)}")
    lines.append("MRIO scope: " + _format_mrio_scope(arguments=arguments))
    if manifest.family == "asocc":
        alternate_source = _alternate_inter_mrio_source(manifest=manifest)
        if alternate_source is not None:
            lines.append(f"Alternate inter MRIO source: {alternate_source}")
    return lines


def _artifact_inventory_lines(*, artifacts: dict[str, Any]) -> list[str]:
    items = []
    results = _present_labels(
        artifacts=artifacts,
        labels={
            "public_row_identity": "row identity",
            "run_values": "run values",
            "summary_stats_runs": "summary statistics",
            "cumulative_summary_stats_runs": "cumulative summary statistics",
        },
    )
    if results:
        items.append(inventory_item(folder="results", content=format_values(results)))
    guides = _present_labels(
        artifacts=artifacts,
        labels={
            "results_readme": "README",
            "source_methods": "source methods",
        },
    )
    if guides:
        items.append(inventory_item(folder="interpretation", content=format_values(guides)))
    sobol = _present_labels(
        artifacts=artifacts,
        labels={
            "sobol_indices": "indices",
            "sobol_source_summary": "source summary",
            "sobol_readme": "README",
        },
    )
    if sobol:
        items.append(inventory_item(folder="results/sobol", content=format_values(sobol)))
    items.append(inventory_item(folder="logs", content="summary log"))
    return list(inventory_lines(items))


def _public_artifact_count(*, artifacts: dict[str, Any]) -> int:
    counted_keys = (
        "public_row_identity",
        "run_values",
        "summary_stats_runs",
        "cumulative_summary_stats_runs",
        "results_readme",
        "source_methods",
        "sobol_indices",
        "sobol_source_summary",
        "sobol_readme",
        "scope_manifest",
    )
    count = sum(1 for key in counted_keys if artifacts.get(key) is not None)
    return count + len(tuple(artifacts.get("figure_paths") or ()))


def _present_labels(*, artifacts: dict[str, Any], labels: dict[str, str]) -> tuple[str, ...]:
    return tuple(label for key, label in labels.items() if artifacts.get(key) is not None)


def _format_mrio_scope(*, arguments: dict[str, Any]) -> str:
    parts = [
        f"group_reg={bool(arguments.get('group_reg', False))}",
        f"group_sec={bool(arguments.get('group_sec', False))}",
        f"group_version={arguments.get('group_version') or 'none'}",
        f"aggreg_indices={bool(arguments.get('aggreg_indices', False))}",
    ]
    return ", ".join(parts)


def _alternate_inter_mrio_source(*, manifest: UncertaintyManifest) -> str | None:
    source_parameters = manifest.source_parameters or {}
    payload = source_parameters.get("inter_mrio_uncertainty")
    if payload is None:
        return None
    source = dict(payload).get("source")
    if source is None:
        return None
    text = str(source).strip()
    return text or None
