"""Report dataclasses for deterministic ASR runs."""

from dataclasses import dataclass, field
from pathlib import Path

from pyaesa.asr.deterministic.state.branch_state import DeterministicBranchState
from pyaesa.shared.acc_asr_common.reporting import (
    DynamicAR6Summary,
    composite_phase_sections,
    dynamic_cc_scope_label,
)
from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.composite_phase_index import CompositePhaseIndexEntry
from pyaesa.shared.runtime.reporting.summary import (
    SummaryDocument,
    SummarySection,
    document,
    render_summary,
    section,
)


@dataclass(frozen=True)
class ASRBranchReport:
    """Summary for one written ASR branch."""

    cc_source: str
    cc_type: str
    cc_bounds: list[str]
    lca_type: str
    n_acc_files_matched: int
    n_asr_files_written: int
    impacts_used: list[str]
    figure_paths: list[Path] = field(default_factory=list)
    output_dirs: list[Path] = field(default_factory=list)
    meta_file: Path | None = None
    phase_index_path: Path | None = None
    phase_entries: tuple[CompositePhaseIndexEntry, ...] = ()
    reuse_status: str = "computed"
    dynamic_ar6_summary: DynamicAR6Summary | None = None
    external_lca_summary: dict[str, object] | None = None


@dataclass(frozen=True)
class ComputeASRReport:
    """Aggregate summary for one composite deterministic_asr call."""

    branches: list[ASRBranchReport]
    output_root: Path
    common_lines: list[str]

    def __str__(self) -> str:
        return render_summary(_asr_summary_document(report=self))

    __repr__ = __str__


def build_asr_branch_report(
    *,
    state: DeterministicBranchState,
    lca_type: str,
    n_acc_files_matched: int,
    n_asr_files_written: int,
    external_lca_summary: dict[str, object] | None = None,
    reuse_status: str = "computed",
) -> ASRBranchReport:
    """Build one deterministic ASR branch report from shared branch state."""
    return ASRBranchReport(
        cc_source=state.cc_source,
        cc_type=state.cc_type,
        cc_bounds=state.cc_bounds,
        lca_type=lca_type,
        n_acc_files_matched=n_acc_files_matched,
        n_asr_files_written=n_asr_files_written,
        impacts_used=state.impacts_used,
        figure_paths=state.figure_paths,
        output_dirs=state.output_dirs,
        meta_file=state.meta_file,
        external_lca_summary=external_lca_summary,
        reuse_status=reuse_status,
    )


def _asr_summary_document(*, report: ComputeASRReport) -> SummaryDocument:
    sections: list[SummarySection] = []
    multi_scope = len(report.branches) > 1
    for index, branch in enumerate(report.branches, start=1):
        branch_sections = _asr_branch_sections(branch=branch)
        if multi_scope:
            sections.append(section(f"Scope {index}", children=branch_sections))
        else:
            sections.extend(branch_sections)
    return document("deterministic_asr", lines=report.common_lines, sections=sections)


def _asr_branch_sections(*, branch: ASRBranchReport) -> tuple[SummarySection, ...]:
    external_lca_lines = (
        ()
        if branch.external_lca_summary is None
        else tuple(_external_lca_summary_lines(branch.external_lca_summary))
    )
    return composite_phase_sections(
        entries=branch.phase_entries,
        dynamic_ar6_summary=branch.dynamic_ar6_summary,
        extra_lines_by_function={"external_lca": external_lca_lines},
    )


def asr_phase_summary_lines(*, branch: ASRBranchReport) -> tuple[str, ...]:
    """Return deterministic ASR summary lines for phase index rendering."""
    bounds_str = dynamic_cc_scope_label(
        branch.cc_type, branch.dynamic_ar6_summary
    ) or _format_values(branch.cc_bounds)
    lines = [
        f"CC source: {branch.cc_source}",
        f"CC type: {branch.cc_type}",
        f"CC scope: {bounds_str}",
    ]
    output_count = branch.n_asr_files_written + len(branch.figure_paths) + 1
    lines.append(output_files_available_line(output_count))
    if branch.figure_paths:
        lines.append(figures_available_line(len(branch.figure_paths)))
    return tuple(lines)


def asr_phase_inventory_lines(*, branch: ASRBranchReport) -> tuple[str, ...]:
    """Return deterministic ASR public output folders for phase index rendering."""
    inventory = [
        inventory_item(folder="results", content="deterministic ASR tables"),
        inventory_item(folder="logs", content="summary log"),
    ]
    return tuple(inventory_lines(inventory))


def _external_lca_summary_lines(payload: dict[str, object]) -> list[str]:
    """Return deterministic external LCA summary lines."""
    lines = [
        f"{label}: {value}"
        for label, key in (
            ("Source type", "source_type"),
            ("Version", "version_name"),
        )
        if (value := payload.get(key)) is not None
    ]
    figures_available = payload.get("figures_available")
    if figures_available is not None and int(str(figures_available)) > 0:
        lines.append(figures_available_line(int(str(figures_available))))
    return lines


def _format_values(values: list[str]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"
