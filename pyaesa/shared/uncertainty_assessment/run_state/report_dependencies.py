"""Dependency subsection rendering for uncertainty run reports."""

import json
from pathlib import Path
from typing import Any, cast

from pyaesa.asocc.runtime.reporting.deterministic_summary import (
    deterministic_asocc_phase_inventory_lines,
    deterministic_asocc_phase_summary_lines,
    deterministic_asocc_summary_record_messages,
)
from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.shared.runtime.reporting.summary import SummarySection, info, section, warning
from pyaesa.shared.runtime.reporting.values import (
    format_report_value,
    format_summary_value,
)
from pyaesa.shared.uncertainty_assessment.run_state.report_ar6 import ar6_cc_dependency_section
from pyaesa.shared.uncertainty_assessment.run_state.report_messages import (
    payload_infos,
    payload_warnings,
)


def payload_source(payload: dict[str, Any]) -> str:
    """Return the public source key for one persisted dependency payload."""
    if payload.get("selection") is not None and payload.get("storage_mode") is not None:
        return f"external_asocc_{payload['storage_mode']}"
    return str(payload.get("base_function_source") or payload.get("type") or payload["source"])


def dependency_section(*, payload: dict[str, Any]) -> SummarySection:
    """Return one dependency subsection for an uncertainty report."""
    source = payload_source(payload)
    if source == "deterministic_ar6_cc":
        return ar6_cc_dependency_section(payload=payload)
    if source == "deterministic_asocc":
        metadata_path = _required_dependency_path(payload=payload)
        output_root = _payload_output_root(payload=payload)
        lines = (
            *deterministic_asocc_phase_summary_lines(
                metadata_path=metadata_path,
                output_root=output_root,
                source=None if payload.get("source") is None else str(payload["source"]),
            ),
            *deterministic_asocc_phase_inventory_lines(
                metadata_path=metadata_path,
                output_root=output_root,
            ),
        )
        return section(
            "deterministic_asocc",
            lines=lines,
            infos=tuple(
                info(message)
                for message in deterministic_asocc_summary_record_messages(
                    metadata_path=metadata_path,
                    severity="INFO",
                )
            ),
            warnings=tuple(
                warning(message)
                for message in deterministic_asocc_summary_record_messages(
                    metadata_path=metadata_path,
                    severity="WARNING",
                )
            ),
            children=(),
        )
    if source.startswith("external_lca_"):
        return section(
            "external_lca",
            lines=_external_lca_lines(payload=payload),
            infos=payload_infos(payload=payload),
            warnings=payload_warnings(payload=payload),
        )
    if source == "io_lca_deterministic":
        return section(
            "deterministic_io_lca",
            lines=_simple_dependency_lines(payload=payload),
            infos=payload_infos(payload=payload),
            warnings=payload_warnings(payload=payload),
        )
    if source in {"external_asocc_deterministic", "external_asocc_monte_carlo"}:
        return section(
            "external_asocc",
            lines=_external_asocc_lines(payload=payload),
            infos=payload_infos(payload=payload),
            warnings=payload_warnings(payload=payload),
        )
    return section(
        _public_dependency_name(source),
        lines=_simple_dependency_lines(payload=payload),
        infos=payload_infos(payload=payload),
        warnings=payload_warnings(payload=payload),
    )


def _public_dependency_name(source: str) -> str:
    names = {
        "uncertainty_asocc": "uncertainty_asocc",
        "uncertainty_ar6_cc": "uncertainty_ar6_cc",
        "uncertainty_acc": "uncertainty_acc",
        "uncertainty_io_lca": "uncertainty_io_lca",
        "uncertainty_asr": "uncertainty_asr",
        "deterministic_asocc": "deterministic_asocc",
        "deterministic_io_lca": "deterministic_io_lca",
        "deterministic_acc": "deterministic_acc",
        "deterministic_asr": "deterministic_asr",
        "external_lca": "external_lca",
    }
    return names[source]


def _simple_dependency_lines(*, payload: dict[str, Any]) -> list[str]:
    details: list[str] = []
    source = payload_source(payload)
    dependency_arguments = _dependency_arguments(payload=payload, source=source)
    details.extend(
        _dependency_owner_scope_lines(
            source=source,
            payload=payload,
            arguments=dependency_arguments,
        )
    )
    for label, key in (
        ("Run status", "reuse_status"),
        ("Version", "version_name"),
        ("Output folder", "output_root"),
    ):
        value = payload.get(key)
        if value is not None:
            details.append(f"{label}: {format_summary_value(key=key, value=value)}")
    output_root = _payload_output_root(payload=payload)
    if output_root is not None and payload.get("output_root") is None:
        details.append(f"Output folder: {output_root}")
    output_count = _payload_public_output_count(payload=payload)
    if output_count:
        details.append(output_files_available_line(output_count))
    details.extend(_payload_inventory_lines(payload=payload))
    return details


def _dependency_owner_scope_lines(
    *,
    source: str,
    payload: dict[str, Any],
    arguments: dict[str, Any],
) -> list[str]:
    if source not in {"deterministic_asocc", "deterministic_io_lca", "io_lca_deterministic"}:
        raw_source = payload.get("source")
        return [] if raw_source is None else [f"Source: {format_report_value(raw_source)}"]
    lines: list[str] = []
    raw_source = arguments.get("source", payload.get("source"))
    if raw_source is not None:
        lines.append(f"Source: {format_report_value(raw_source)}")
    if arguments:
        lines.append("MRIO scope: " + _format_mrio_scope(arguments=arguments))
    return lines


def _dependency_arguments(*, payload: dict[str, Any], source: str) -> dict[str, Any]:
    arguments = payload.get("summary_arguments")
    if arguments is not None:
        return dict(arguments)
    if source not in {"deterministic_asocc", "deterministic_io_lca", "io_lca_deterministic"}:
        return {}
    path = _required_dependency_path(payload=payload)
    raw = _read_dependency_manifest(path=path)
    return dict(raw["arguments"])


def _external_lca_lines(*, payload: dict[str, Any]) -> list[str]:
    lines = []
    source_type = (
        "Monte Carlo"
        if payload_source(payload) == "external_lca_monte_carlo"
        else ("deterministic")
    )
    lines.append(f"Source type: {source_type}")
    for label, key in (
        ("Run status", "reuse_status"),
        ("Version", "version_name"),
    ):
        value = payload.get(key)
        if value is not None:
            lines.append(f"{label}: {format_summary_value(key=key, value=value)}")
    output_root = _external_lca_output_root(payload=payload)
    if output_root is not None:
        lines.append(f"Output folder: {output_root}")
    figures_available = payload.get("figures_available")
    if figures_available is not None and int(figures_available) > 0:
        lines.append(figures_available_line(int(figures_available)))
    return lines


def _external_asocc_lines(*, payload: dict[str, Any]) -> list[str]:
    return [
        f"Selection: {format_report_value(payload['selection'])}",
        f"Storage mode: {format_report_value(payload['storage_mode'])}",
    ]


def _payload_output_root(*, payload: dict[str, Any]) -> Path | None:
    for key in ("metadata_path", "scope_manifest"):
        value = payload.get(key)
        if value is not None:
            return public_output_root_from_path(Path(str(value)))
    return None


def _required_dependency_path(*, payload: dict[str, Any]) -> Path:
    return Path(str(payload.get("metadata_path") or payload["scope_manifest"]))


def _external_lca_output_root(*, payload: dict[str, Any]) -> Path | None:
    value = payload.get("output_root")
    if value is not None:
        return Path(str(value))
    return None


def _payload_inventory_lines(*, payload: dict[str, Any]) -> list[str]:
    source = payload_source(payload)
    if source in {"deterministic_io_lca", "io_lca_deterministic"}:
        return _deterministic_io_lca_inventory_lines(payload=payload)
    items = []
    if payload.get("output_file") is not None or payload.get("deterministic_paths"):
        items.append(inventory_item(folder="results", content="result tables"))
    if payload.get("post_study_output_file") is not None:
        items.append(inventory_item(folder="results", content="post study period result table"))
    if payload.get("metadata_path") is not None or payload.get("scope_manifest") is not None:
        items.append(inventory_item(folder="logs", content="summary log"))
    return list(inventory_lines(items))


def _payload_public_output_count(*, payload: dict[str, Any]) -> int:
    source = payload_source(payload)
    if source in {"deterministic_io_lca", "io_lca_deterministic"}:
        return _deterministic_io_lca_output_count(payload=payload)
    paths: set[str] = set()
    for key in (
        "output_file",
        "post_study_output_file",
        "deterministic_paths",
        "figure_paths",
    ):
        value = payload.get(key)
        if isinstance(value, list | tuple):
            paths.update(str(item).strip() for item in value)
        elif value is not None:
            paths.add(str(value).strip())
    paths.discard("")
    if payload.get("metadata_path") is not None or payload.get("scope_manifest") is not None:
        paths.add("summary.log")
    return len(paths)


def _read_dependency_manifest(*, path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _deterministic_io_lca_output_count(*, payload: dict[str, Any]) -> int:
    metadata_path = _required_dependency_path(payload=payload)
    raw = _read_dependency_manifest(path=metadata_path)
    artifacts = dict(raw["artifacts"])
    paths = {str(path).strip() for path in artifacts.get("paths_written", []) if str(path).strip()}
    for figure_path in _deterministic_io_lca_figure_paths(raw=raw):
        paths.add(str(figure_path))
    summary_log = public_output_root_from_path(metadata_path) / "logs" / "summary.log"
    paths.add(str(summary_log))
    return len(paths)


def _deterministic_io_lca_inventory_lines(*, payload: dict[str, Any]) -> list[str]:
    metadata_path = _required_dependency_path(payload=payload)
    raw = _read_dependency_manifest(path=metadata_path)
    sections = _deterministic_io_lca_sections(raw=raw)
    items = []
    if sections.get("main"):
        items.append(inventory_item(folder="results", content="main LCA tables"))
    if sections.get("origin"):
        items.append(inventory_item(folder="results/origin", content="origin contribution tables"))
    if sections.get("stages"):
        items.append(inventory_item(folder="results/stages", content="stage contribution tables"))
    items.append(inventory_item(folder="logs", content="summary log"))
    return list(inventory_lines(items))


def _deterministic_io_lca_sections(
    *,
    raw: dict[str, Any],
) -> dict[str, Any]:
    execution = cast(dict[str, Any], raw["execution"])
    return cast(dict[str, Any], execution["sections"])


def _deterministic_io_lca_figure_paths(*, raw: dict[str, Any]) -> tuple[str, ...]:
    execution = cast(dict[str, Any], raw["execution"])
    sections = cast(dict[str, Any], execution["sections"])
    figures = sections.get("figures")
    if figures is None:
        return ()
    paths: list[str] = []
    for value in cast(dict[str, Any], figures).values():
        figure_payload = cast(dict[str, Any], value)
        paths.extend(str(path) for path in figure_payload.get("paths", []) if str(path).strip())
    return tuple(paths)


def _format_mrio_scope(*, arguments: dict[str, Any]) -> str:
    parts = [
        f"agg_reg={bool(arguments.get('agg_reg', False))}",
        f"agg_sec={bool(arguments.get('agg_sec', False))}",
        f"agg_version={arguments.get('agg_version') or 'none'}",
        f"group_indices={bool(arguments.get('group_indices', False))}",
    ]
    return ", ".join(parts)
