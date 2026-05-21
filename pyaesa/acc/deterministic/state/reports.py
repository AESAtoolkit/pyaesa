"""Report dataclasses for deterministic aCC runs."""

from dataclasses import dataclass, field
from pathlib import Path

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
class ACCBranchReport:
    """Summary for one written aCC branch."""

    cc_source: str
    cc_type: str
    cc_bounds: list[str]
    n_share_files_processed: int
    n_acc_files_written: int
    impacts_used: list[str]
    output_dirs: list[Path] = field(default_factory=list)
    figure_paths: list[Path] = field(default_factory=list)
    meta_file: Path | None = None
    phase_index_path: Path | None = None
    phase_entries: tuple[CompositePhaseIndexEntry, ...] = ()
    reuse_status: str = "computed"
    dynamic_ar6_summary: DynamicAR6Summary | None = None


@dataclass(frozen=True)
class ComputeACCReport:
    """Aggregate summary for one composite deterministic_acc call."""

    branches: list[ACCBranchReport]
    output_root: Path
    common_lines: list[str]

    def __str__(self) -> str:
        return render_summary(_acc_summary_document(report=self))

    __repr__ = __str__


def _acc_summary_document(*, report: ComputeACCReport) -> SummaryDocument:
    sections: list[SummarySection] = []
    multi_scope = len(report.branches) > 1
    for index, branch in enumerate(report.branches, start=1):
        branch_sections = _acc_branch_sections(branch=branch)
        if multi_scope:
            sections.append(section(f"Scope {index}", children=branch_sections))
        else:
            sections.extend(branch_sections)
    return document("deterministic_acc", lines=report.common_lines, sections=sections)


def _acc_branch_sections(*, branch: ACCBranchReport) -> tuple[SummarySection, ...]:
    return composite_phase_sections(
        entries=branch.phase_entries,
        dynamic_ar6_summary=branch.dynamic_ar6_summary,
    )


def acc_phase_summary_lines(*, branch: ACCBranchReport) -> tuple[str, ...]:
    """Return deterministic aCC summary lines for phase index rendering."""
    bounds_str = dynamic_cc_scope_label(
        branch.cc_type, branch.dynamic_ar6_summary
    ) or _format_values(branch.cc_bounds)
    lines = [
        f"CC source: {branch.cc_source}",
        f"CC type: {branch.cc_type}",
        f"CC scope: {bounds_str}",
    ]
    output_count = branch.n_acc_files_written + len(branch.figure_paths) + 1
    lines.append(output_files_available_line(output_count))
    if branch.figure_paths:
        lines.append(figures_available_line(len(branch.figure_paths)))
    return tuple(lines)


def acc_phase_inventory_lines(*, branch: ACCBranchReport) -> tuple[str, ...]:
    """Return deterministic aCC public output folders for phase index rendering."""
    inventory = [
        inventory_item(folder="results", content="deterministic aCC tables"),
        inventory_item(folder="logs", content="summary log"),
    ]
    return tuple(inventory_lines(inventory))


def _format_values(values: list[str]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"
