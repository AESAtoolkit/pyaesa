"""Structured phase index helpers for composite public workflows."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.io.filesystem import atomic_write_text
from pyaesa.shared.runtime.reporting.reuse_status import public_reuse_status

# Public composite phase labels form the cross family persisted phase index schema
# for aCC and ASR orchestration reports. The separate function field records
# which public function executed in each phase.
PHASE_A_LCA = "Phase A: LCA"
PHASE_B1_ASOCC = "Phase B.1: aSoCC"
PHASE_B1_AR6_DYNAMIC_CC = "Phase B.1: Dynamic AR6 CC"
PHASE_B2_ACC = "Phase B.2: aCC"
PHASE_C_ASR = "Phase C: ASR"
COMPOSITE_PHASE_ORDER = (
    PHASE_A_LCA,
    PHASE_B1_ASOCC,
    PHASE_B1_AR6_DYNAMIC_CC,
    PHASE_B2_ACC,
    PHASE_C_ASR,
)
PUBLIC_PHASE_REUSE_STATUSES = frozenset({"computed", "updated", "reused"})
_OWNER_PHASES = {
    "deterministic_io_lca": PHASE_A_LCA,
    "uncertainty_io_lca": PHASE_A_LCA,
    "external_lca": PHASE_A_LCA,
    "deterministic_asocc": PHASE_B1_ASOCC,
    "uncertainty_asocc": PHASE_B1_ASOCC,
    "disaggregate_asocc": PHASE_B1_ASOCC,
    "process_ar6": PHASE_B1_AR6_DYNAMIC_CC,
    "deterministic_ar6_cc": PHASE_B1_AR6_DYNAMIC_CC,
    "uncertainty_ar6_cc": PHASE_B1_AR6_DYNAMIC_CC,
    "deterministic_acc": PHASE_B2_ACC,
    "uncertainty_acc": PHASE_B2_ACC,
    "deterministic_asr": PHASE_C_ASR,
    "uncertainty_asr": PHASE_C_ASR,
}
_REPORT_SOURCE_PHASES = {
    **_OWNER_PHASES,
    "io_lca_deterministic": PHASE_A_LCA,
    "external_lca_deterministic": PHASE_A_LCA,
    "external_lca_monte_carlo": PHASE_A_LCA,
    "external_asocc_deterministic": PHASE_B1_ASOCC,
    "external_asocc_monte_carlo": PHASE_B1_ASOCC,
}
_UNCERTAINTY_FAMILY_PHASES = {
    "io_lca": PHASE_A_LCA,
    "asocc": PHASE_B1_ASOCC,
    "ar6_cc": PHASE_B1_AR6_DYNAMIC_CC,
    "acc": PHASE_B2_ACC,
    "asr": PHASE_C_ASR,
}


@dataclass(frozen=True)
class CompositePhaseIndexEntry:
    """One persisted stage entry in a composite phase index.

    ``reuse_status`` is the persisted public phase status: ``computed``,
    ``reused``, or ``updated``.
    """

    phase: str
    function: str
    status: str
    reuse_status: str
    output_root: Path | None
    summary_lines: tuple[str, ...] = ()
    info_messages: tuple[str, ...] = ()
    warning_messages: tuple[str, ...] = ()
    inventory_lines: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        """Return one JSON-serializable phase payload."""
        return {
            "phase": self.phase,
            "function": self.function,
            "status": self.status,
            "reuse_status": public_phase_index_reuse_status(self.reuse_status),
            "output_root": None if self.output_root is None else str(self.output_root),
            "summary_lines": list(self.summary_lines),
            "info_messages": list(self.info_messages),
            "warning_messages": list(self.warning_messages),
            "inventory_lines": list(self.inventory_lines),
        }


def public_phase_reuse_status(*, run_status: str) -> str:
    """Return the public composite phase reuse status for one function run."""
    return public_reuse_status(run_status)


def public_phase_index_reuse_status(status: str) -> str:
    """Return one already public composite phase reuse status."""
    return str(status).strip()


def phase_label_for_owner(owner: str | None) -> str | None:
    """Return the canonical phase label for one public function owner."""
    if owner is None:
        return None
    return _OWNER_PHASES.get(str(owner).strip())


def required_phase_label_for_report_source(source: object) -> str:
    """Return the canonical phase label for a package generated report source."""
    return _REPORT_SOURCE_PHASES[str(source).strip()]


def required_phase_label_for_uncertainty_family(*, family: str) -> str:
    """Return the canonical phase label for a package generated uncertainty family."""
    return _UNCERTAINTY_FAMILY_PHASES[str(family).strip()]


def phase_index_path_for_metadata(*, metadata_path: Path) -> Path:
    """Return the canonical composite phase index path for one top-level scope."""
    return Path(metadata_path).parent / "composite_phase_index.json"


def write_phase_index(
    *,
    metadata_path: Path,
    entries: list[CompositePhaseIndexEntry],
) -> Path:
    """Persist one composite phase index beside the top-level metadata artifact."""
    index_path = phase_index_path_for_metadata(metadata_path=metadata_path)
    payload = [entry.to_payload() for entry in entries]
    atomic_write_text(index_path, text=json.dumps(payload, indent=2))
    return index_path


def read_phase_index(*, metadata_path: Path) -> tuple[CompositePhaseIndexEntry, ...]:
    """Read one persisted composite phase index."""
    index_path = phase_index_path_for_metadata(metadata_path=metadata_path)
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries: list[CompositePhaseIndexEntry] = []
    for item in payload:
        output_root = item.get("output_root")
        entries.append(
            CompositePhaseIndexEntry(
                phase=str(item["phase"]),
                function=str(item["function"]),
                status=str(item["status"]),
                reuse_status=public_phase_index_reuse_status(str(item["reuse_status"])),
                output_root=None if output_root is None else Path(str(output_root)),
                summary_lines=tuple(str(value) for value in item.get("summary_lines", ())),
                info_messages=tuple(str(value) for value in item.get("info_messages", ())),
                warning_messages=tuple(str(value) for value in item.get("warning_messages", ())),
                inventory_lines=tuple(str(value) for value in item.get("inventory_lines", ())),
            )
        )
    return tuple(entries)


def phase_ready_detail(
    *,
    scope_name: str,
    output_root: Path | None = None,
) -> str:
    """Return the canonical completion detail for a resolved phase with work."""
    return _phase_detail(
        prefix=f"{scope_name} scope ready",
        output_root=output_root,
    )


def phase_uncertainty_done_detail(
    *,
    scope_name: str,
    mode: str,
    convergence: dict[str, Any] | None,
    output_root: Path | None = None,
) -> str:
    """Return completion detail for fixed and convergence uncertainty runs."""
    prefix = f"{scope_name} scope ready"
    if str(mode) == "convergence":
        reached = bool((convergence or {}).get("reached"))
        suffix = "ready; convergence reached" if reached else "completed; convergence not reached"
        prefix = f"{scope_name} scope {suffix}"
    return _phase_detail(
        prefix=prefix,
        output_root=output_root,
    )


def phase_reused_detail(
    *,
    scope_name: str,
    output_root: Path | None = None,
) -> str:
    """Return the canonical completion detail for an exact reused phase."""
    return _phase_detail(
        prefix=f"{scope_name} scope reused exactly",
        output_root=output_root,
    )


def _phase_detail(
    *,
    prefix: str,
    output_root: Path | None,
) -> str:
    parts = [prefix]
    if output_root is not None:
        parts.append(f"output folder: {output_root}")
    return " | ".join(parts)
