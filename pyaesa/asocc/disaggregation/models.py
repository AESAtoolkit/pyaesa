"""Typed models for disaggregate_asocc runtime."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.reporting.labels import figures_available_line
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines


@dataclass(frozen=True)
class RunSelector:
    """Selector for one prerequisite deterministic_asocc run."""

    source: str
    group_reg: bool
    group_sec: bool
    group_version: str | None
    s_p: list[str]


@dataclass(frozen=True)
class DisaggregationSpec:
    """One grouped to split sector mapping."""

    grouped_sector_label: str
    split_sector_label: str


@dataclass(frozen=True)
class DisaggregationConfigModel:
    """Validated disaggregation configuration payload."""

    target_grouped_run: RunSelector
    ref_grouped_run: RunSelector
    ref_split_run: RunSelector
    disaggregation_specs: list[DisaggregationSpec]
    new_disaggregated_version_name: str


@dataclass(frozen=True)
class ParsedArgs:
    """Validated argument payload for disaggregate_asocc orchestration."""

    disaggregation: DisaggregationConfigModel
    base_allocate_args: dict[str, Any]
    output_format: str
    figures: bool
    figure_options: dict[str, Any]
    figure_format: dict[str, Any]
    figure_external_method: dict[str, list[str]] | None
    refresh: bool


@dataclass(frozen=True)
class MatchedRun:
    """Resolved prerequisite deterministic scope for one selector."""

    selector_name: str
    proj_base: Path
    run_metadata_path: Path
    scope_key: str
    scope_signature: dict[str, Any]
    completed_years: list[int]
    output_source_label: str


@dataclass(frozen=True)
class PreparedBranchContext:
    """Prepared per-branch runtime context for one disaggregated source write."""

    matched_runs: dict[str, MatchedRun]
    requested_years: list[int]
    ssp_scenario_options_by_year: dict[int, list[str | None]]
    disagg_run_signature: dict[str, Any]
    disagg_proj_base: Path
    branch_complete: bool = False


@dataclass(frozen=True)
class DisaggregationRunPlan:
    """Resolved branch planning inputs shared across disaggregation branches."""

    r_p: list[str] | None
    r_c: list[str] | None
    r_f: list[str] | None
    l1_methods: list[str]
    combined_non_lcia: list[tuple[str, str]]
    one_step_non_lcia: list[str]
    selected_l2_methods: list[str]
    ssp_scenarios: list[str] | None
    aggreg_indices: bool
    l1_reg_aggreg: str


@dataclass(frozen=True)
class DisaggregationBranchReport:
    """One branch report block for disaggregate_asocc."""

    l1_reg_aggreg: str
    aggreg_indices: bool
    summaries: list[str]
    disaggregation_audit_path: Path
    metadata_path: Path
    figure_paths: list[Path]
    run_status: str = "computed"

    def format_header(self) -> str:
        """Return human readable branch header."""
        mode = "grouped" if self.aggreg_indices else "ungrouped"
        return (
            f"[disaggregate_asocc] Branch l1_reg_aggreg={self.l1_reg_aggreg}, aggreg_indices={mode}"
        )


@dataclass(frozen=True)
class DisaggregationReport:
    """Aggregate report for all executed disaggregation branches."""

    source_label: str
    branch_reports: list[DisaggregationBranchReport]

    def reuse_status(self) -> str:
        """Return aggregate reuse status for the requested disaggregation run."""
        statuses = {branch.run_status for branch in self.branch_reports}
        if statuses == {"reused_exact"}:
            return "reused_exact"
        return "computed"

    def output_root(self) -> Path:
        """Return the public output root for the first branch report."""
        return self.branch_reports[0].metadata_path.parent.parent

    def __str__(self) -> str:
        blocks: list[str] = []
        for branch in self.branch_reports:
            lines = [branch.format_header(), *branch.summaries]
            if branch.figure_paths:
                lines.append(f"  {figures_available_line(len(branch.figure_paths))}")
            inventory = [inventory_item(folder="logs", content="disaggregation audit")]
            for line in inventory_lines(inventory):
                lines.append(f"  {line}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    __repr__ = __str__
