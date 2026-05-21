"""Phase section assembly for uncertainty run reports."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.reporting.composite_phase_index import (
    required_phase_label_for_report_source,
    required_phase_label_for_uncertainty_family,
)
from pyaesa.shared.runtime.reporting.summary import SummarySection, section
from pyaesa.shared.runtime.reporting.summary import SummaryWarning
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.report_dependencies import (
    dependency_section,
    payload_source,
)
from pyaesa.shared.uncertainty_assessment.run_state.report_ar6 import process_ar6_section
from pyaesa.shared.uncertainty_assessment.run_state.report_self import self_section


def phase_sections(
    *,
    manifest: UncertaintyManifest,
    run_warnings: tuple[SummaryWarning, ...],
    warning_builder: Callable[[UncertaintyManifest], tuple[SummaryWarning, ...]],
) -> tuple[SummarySection, ...]:
    """Return phase sections for one uncertainty report."""
    grouped: dict[str, list[SummarySection]] = {}
    for payload in [*manifest.deterministic_prerequisites, *manifest.external_inputs]:
        for phase, dependency in _dependency_phase_items(
            payload=payload,
            warning_builder=warning_builder,
        ):
            grouped.setdefault(phase, []).append(dependency)
    grouped.setdefault(
        _phase_label_for_family(manifest.family),
        [],
    ).append(self_section(manifest=manifest, warnings=run_warnings))
    return tuple(section(phase, children=children) for phase, children in grouped.items())


def _dependency_phase_items(
    *,
    payload: dict[str, Any],
    warning_builder: Callable[[UncertaintyManifest], tuple[SummaryWarning, ...]],
) -> tuple[tuple[str, SummarySection], ...]:
    source = payload_source(payload)
    nested = _nested_manifest(payload=payload, source=source)
    if nested is not None:
        reuse_status = payload.get("reuse_status")
        return _phase_items_from_manifest(
            manifest=nested,
            reuse_status=None if reuse_status is None else str(reuse_status),
            warning_builder=warning_builder,
        )
    if source == "deterministic_ar6_cc":
        phase = _phase_label_for_source(source)
        return (
            (phase, process_ar6_section(payload=payload["process_ar6"])),
            (phase, dependency_section(payload=payload)),
        )
    return (
        (
            _phase_label_for_source(source),
            dependency_section(payload=payload),
        ),
    )


def _phase_items_from_manifest(
    *,
    manifest: UncertaintyManifest,
    warning_builder: Callable[[UncertaintyManifest], tuple[SummaryWarning, ...]],
    reuse_status: str | None = None,
) -> tuple[tuple[str, SummarySection], ...]:
    items: list[tuple[str, SummarySection]] = []
    for payload in [*manifest.deterministic_prerequisites, *manifest.external_inputs]:
        items.extend(_dependency_phase_items(payload=payload, warning_builder=warning_builder))
    items.append(
        (
            _phase_label_for_family(manifest.family),
            self_section(
                manifest=manifest,
                warnings=warning_builder(manifest),
                reuse_status=reuse_status,
            ),
        )
    )
    return tuple(items)


def _nested_manifest(
    *,
    payload: dict[str, Any],
    source: str,
) -> UncertaintyManifest | None:
    if not source.startswith("uncertainty_"):
        return None
    path = Path(str(payload["scope_manifest"]))
    return read_manifest(path=path)


def _phase_label_for_source(source: str) -> str:
    return required_phase_label_for_report_source(source)


def _phase_label_for_family(family: str) -> str:
    return required_phase_label_for_uncertainty_family(family=family)
