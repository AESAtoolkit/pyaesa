"""Public deterministic aSoCC subsection facts."""

from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.metadata.json import read_json_dict
from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    labelled_values_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.shared.runtime.reporting.values import format_values


def deterministic_asocc_phase_summary_lines(
    *,
    metadata_path: Path,
    output_root: Path | None = None,
    source: str | None = None,
) -> tuple[str, ...]:
    """Return deterministic aSoCC public summary lines from its scope manifest."""
    payload = read_json_dict(metadata_path)
    arguments = dict(payload.get("arguments", {}))
    artifacts = dict(payload.get("artifacts", {}))
    root = output_root or public_output_root_from_path(metadata_path)
    lines = [f"Source: {arguments.get('source', source)}"]
    lcia_method = arguments.get("lcia_method")
    if lcia_method:
        methods = _as_list(lcia_method)
        lines.append(
            labelled_values_line(
                "LCIA method",
                "LCIA methods",
                methods,
                format_values(methods),
            )
        )
    lines.append("MRIO scope: " + _format_mrio_scope(arguments=arguments))
    output_count = deterministic_asocc_public_output_count(
        artifacts=artifacts,
        output_root=root,
    )
    if output_count:
        lines.append(output_files_available_line(output_count))
    figure_count = len(_as_list(artifacts.get("figure_paths")))
    if figure_count:
        lines.append(figures_available_line(figure_count))
    return tuple(lines)


def deterministic_asocc_phase_inventory_lines(
    *,
    metadata_path: Path,
    output_root: Path | None = None,
) -> tuple[str, ...]:
    """Return deterministic aSoCC public output folder inventory lines."""
    payload = read_json_dict(metadata_path)
    arguments = dict(payload.get("arguments", {}))
    provenance = dict(payload.get("provenance", {}))
    selected = dict(provenance.get("selected_methods", {}))
    artifacts = dict(payload.get("artifacts", {}))
    root = output_root or public_output_root_from_path(metadata_path)
    output_paths = tuple(_as_list(artifacts.get("outputs")))
    items = []
    if selected.get("l1"):
        items.append(inventory_item(folder="results/level_1", content="Level 1 shares"))
    if selected.get("l2_in_l1"):
        items.append(
            inventory_item(
                folder="results/level_2/l2_in_l1",
                content="conditional L2 in L1 weights used by two step routes",
            )
        )
    if selected.get("l2_in_l1") or selected.get("l2_vs_global"):
        items.append(
            inventory_item(
                folder="results/level_2/l2_vs_global",
                content=(
                    "final L2 vs global shares, with direct one step outputs "
                    "and two step L1 * L2 in L1 products as selected"
                ),
            )
        )
    if bool(arguments.get("intermediate_outputs")):
        items.append(
            inventory_item(
                folder="enacting_metrics",
                content="intermediate enacting metric inputs",
            )
        )
        if _has_utility_route(selected=selected):
            items.append(
                inventory_item(
                    folder="results/level_2/utility_propagation_contrib",
                    content="utility propagation contributions",
                )
            )
    if any("historical_reuse" in path for path in output_paths):
        items.append(
            inventory_item(
                folder="results/level_2/historical_reuse",
                content="historical reuse L2 route outputs",
            )
        )
    if any("regression_proj" in path for path in output_paths):
        items.append(
            inventory_item(
                folder="results/level_2/regression_proj",
                content="regression projected L2 route outputs",
            )
        )
    if artifacts.get("regression_stats_paths"):
        items.append(
            inventory_item(folder="logs/regression_proj", content="regression diagnostics")
        )
    summary_log = root / "logs" / "summary.log"
    if summary_log.exists() and summary_log.stat().st_size > 0:
        items.append(inventory_item(folder="logs", content="summary log"))
    return tuple(inventory_lines(items))


def deterministic_asocc_summary_record_messages(
    *,
    metadata_path: Path,
    severity: str,
) -> tuple[str, ...]:
    """Return persisted deterministic aSoCC INFO or WARNING messages."""
    payload = read_json_dict(metadata_path)
    messages: list[str] = []
    for record in payload.get("summary_records") or ():
        if not isinstance(record, dict):
            continue
        if str(record.get("severity", "")).strip().upper() != severity:
            continue
        message = str(record.get("message", "")).strip()
        if message:
            messages.append(message)
    return tuple(dict.fromkeys(messages))


def deterministic_asocc_public_output_count(
    *,
    artifacts: dict[str, Any],
    output_root: Path,
) -> int:
    """Return the number of public deterministic aSoCC files available."""
    paths = {str(path).strip() for path in _as_list(artifacts.get("outputs"))}
    for path in _as_list(artifacts.get("figure_paths")):
        paths.add(str(path).strip())
    for path in _as_list(artifacts.get("regression_stats_paths")):
        paths.add(str(path).strip())
    paths.discard("")
    summary_log = output_root / "logs" / "summary.log"
    if summary_log.exists() and summary_log.stat().st_size > 0:
        paths.add(str(summary_log))
    return len(paths)


def _as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _format_mrio_scope(*, arguments: dict[str, Any]) -> str:
    parts = [
        f"agg_reg={bool(arguments.get('agg_reg'))}",
        f"agg_sec={bool(arguments.get('agg_sec'))}",
        f"agg_version={arguments.get('agg_version') or 'none'}",
        f"group_indices={bool(arguments.get('group_indices'))}",
    ]
    return ", ".join(parts)


def _has_utility_route(*, selected: dict[str, Any]) -> bool:
    method_labels = [
        *[str(value) for value in _as_list(selected.get("l2_in_l1"))],
        *[str(value) for value in _as_list(selected.get("l2_vs_global"))],
    ]
    return any("UT(FDa)" in label or "UT(GVAa)" in label for label in method_labels)
