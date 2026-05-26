"""ASR specific deterministic prerequisite orchestration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyaesa.acc.deterministic_acc import deterministic_acc
from pyaesa.acc.deterministic.runtime.paths import (
    build_acc_path_context,
    get_acc_meta_path,
)
from pyaesa.acc.deterministic.state.metadata import load_recorded_output_files
from pyaesa.io_lca.deterministic_io_lca import deterministic_io_lca
from pyaesa.io_lca.data.paths import (
    io_metadata_path_for_source,
    resolve_io_lca_paths,
)
from pyaesa.external_inputs.lca.paths import external_lca_root
from pyaesa.asocc.runtime.scope.branch_resolution import resolve_allocate_project_base
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.shared.acc_asr_common.scope.composite import (
    base_asocc_kwargs_from_allocate_args,
    build_composite_base_allocate_args,
)
from pyaesa.shared.acc_asr_common.reporting import DynamicAR6Summary
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    PHASE_A_LCA,
    public_phase_reuse_status,
)
from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.shared.runtime.reporting.output_inventory import (
    inventory_item,
    inventory_lines as render_inventory_lines,
)


@dataclass(frozen=True)
class ASRBranchPrerequisites:
    """Resolved upstream artifacts for one deterministic ASR branch."""

    phase_entries: list[CompositePhaseIndexEntry]
    acc_output_files: list[Path]
    dynamic_ar6_summary: DynamicAR6Summary | None


def _io_lca_phase_summary_lines(*, io_report, base_allocate_args: dict[str, Any]) -> list[str]:
    lines = [
        f"Source: {io_report.source}",
        "MRIO scope: " + _format_mrio_scope(arguments=base_allocate_args),
    ]
    output_count = (
        len(io_report.main_result_paths)
        + len(io_report.origin_paths)
        + len(io_report.stage_paths)
        + len(io_report.figure_paths)
        + 1
    )
    lines.append(output_files_available_line(output_count))
    if io_report.figure_paths:
        lines.append(figures_available_line(len(io_report.figure_paths)))
    return lines


def _format_mrio_scope(*, arguments: dict[str, Any]) -> str:
    parts = [
        f"agg_reg={bool(arguments.get('agg_reg'))}",
        f"agg_sec={bool(arguments.get('agg_sec'))}",
        f"agg_version={arguments.get('agg_version') or 'none'}",
        f"group_indices={bool(arguments.get('group_indices'))}",
    ]
    return ", ".join(parts)


def ensure_asr_branch_prerequisites(
    *,
    phase: PhasePrinter,
    project_name: str,
    source: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
    years: int | list[int] | range,
    fu_code: str,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
    r_c: str | list[str] | None,
    r_f: str | list[str] | None,
    group_indices: bool,
    base_asocc_args: dict[str, Any],
    shared_lcia_methods: list[str],
    branch: dict[str, Any],
    external_method: dict[str, Any] | None,
    lca_type: str,
    lca_version_name: str | None,
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any],
    figure_output_format: str,
    figure_dpi: int,
    refresh: bool = False,
) -> ASRBranchPrerequisites:
    """Ensure one ASR branch has all required deterministic upstream outputs."""
    phase_entries: list[CompositePhaseIndexEntry] = []
    base_allocate_args = build_composite_base_allocate_args(
        project_name=project_name,
        years=years,
        lcia_method=[branch["cc_source"]],
        asocc_lcia_methods=shared_lcia_methods,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        source=source,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
        group_indices=group_indices,
        base_asocc_args=base_asocc_args,
    )
    proj_base = resolve_allocate_project_base(
        base_allocate_args=normalize_base_allocate_args(
            base_asocc_kwargs_from_allocate_args(base_allocate_args=base_allocate_args)
        )
    )
    branch_cc_args = {
        "static": {"exclude_max_cc": list(branch["static_cc_bounds"]) == ["min_cc"]}
        if branch["cc_type"] == "static"
        else None,
        "dynamic_ar6": (
            {
                "harmonization": bool(branch["harmonization"]),
                "harmonization_method": str(branch["harmonization_method"]),
                "category": branch["category"],
                "ssp_scenario": branch["ssp_scenario"],
                "emission_type": str(branch["emission_type"]),
                "include_afolu": bool(branch["include_afolu"]),
                "emissions_mode": str(branch["emissions_mode"]),
                "subset_version": branch["subset_version"],
            }
            if branch["cc_type"] == "dynamic_ar6"
            else None
        ),
    }
    acc_report = deterministic_acc(
        project_name=project_name,
        source=source,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version="" if agg_version is None else agg_version,
        years=years,
        lcia_method=[branch["cc_source"]],
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        group_indices=group_indices,
        base_asocc_args=base_asocc_args,
        base_cc_args=branch_cc_args,
        _shared_asocc_lcia_methods=shared_lcia_methods,
        external_method=external_method,
        output_format=output_format,
        figures=figures,
        subfigures=figures,
        figure_options={
            "per_method": figure_options["per_method"],
            "multi_method": figure_options["multi_method"],
        },
        figure_format={
            "format": figure_output_format,
            "dpi": figure_dpi,
        },
        refresh=refresh,
        _phase=phase,
    )
    acc_path_context = build_acc_path_context(
        proj_base=proj_base,
        source_label=str(base_allocate_args["source"]),
        agg_version=base_allocate_args["agg_version"],
        cc_source=str(branch["cc_source"]),
        cc_type=str(branch["cc_type"]),
    )
    acc_metadata_path = get_acc_meta_path(context=acc_path_context)
    acc_output_files = load_recorded_output_files(metadata_path=acc_metadata_path)
    acc_branch = acc_report.branches[0]
    dynamic_ar6_summary = (
        acc_branch.dynamic_ar6_summary if branch["cc_type"] == "dynamic_ar6" else None
    )
    phase_entries.extend(acc_branch.phase_entries)
    if lca_type == "io_lca":
        phase.announce(PHASE_A_LCA, "deterministic_io_lca")
        io_report = deterministic_io_lca(
            project_name=project_name,
            source=base_allocate_args["source"],
            agg_sec=bool(base_allocate_args["agg_sec"]),
            agg_reg=bool(base_allocate_args["agg_reg"]),
            agg_version=(
                ""
                if str(base_allocate_args["agg_version"] or "").strip() == ""
                else str(base_allocate_args["agg_version"])
            ),
            years=base_allocate_args["years"],
            lcia_method=[branch["cc_source"]],
            fu_code=fu_code,
            r_f=r_f,
            r_c=r_c,
            r_p=r_p,
            s_p=s_p,
            group_indices=base_allocate_args["group_indices"],
            output_format=output_format,
            figures=figures,
            figure_format={
                "format": figure_output_format,
                "dpi": figure_dpi,
            },
            refresh=refresh,
            _status=phase,
        )
        io_paths = resolve_io_lca_paths(
            project_name=project_name,
            agg_reg=bool(base_allocate_args["agg_reg"]),
            agg_sec=bool(base_allocate_args["agg_sec"]),
            agg_version=(
                None
                if str(base_allocate_args["agg_version"] or "").strip() == ""
                else str(base_allocate_args["agg_version"])
            ),
        )
        io_metadata_path = io_metadata_path_for_source(
            paths=io_paths,
            source=str(base_allocate_args["source"]),
        )
        io_output_root = public_output_root_from_path(io_metadata_path)
        phase_entries.append(
            CompositePhaseIndexEntry(
                phase=PHASE_A_LCA,
                function="deterministic_io_lca",
                status="complete",
                reuse_status=public_phase_reuse_status(run_status=io_report.reuse_status),
                output_root=io_output_root,
                summary_lines=tuple(
                    _io_lca_phase_summary_lines(
                        io_report=io_report,
                        base_allocate_args=base_allocate_args,
                    )
                ),
                inventory_lines=tuple(
                    render_inventory_lines(
                        [
                            *(
                                [inventory_item(folder="results", content="main LCA tables")]
                                if io_report.main_result_paths
                                else []
                            ),
                            *(
                                [
                                    inventory_item(
                                        folder="results/origin",
                                        content="origin contribution tables",
                                    )
                                ]
                                if io_report.origin_paths
                                else []
                            ),
                            *(
                                [
                                    inventory_item(
                                        folder="results/stages",
                                        content="stage contribution tables",
                                    )
                                ]
                                if io_report.stage_paths
                                else []
                            ),
                            inventory_item(folder="logs", content="summary log"),
                        ]
                    )
                ),
            )
        )
    else:
        lca_output_root = external_lca_root(project_base=proj_base)
        phase_entries.append(
            CompositePhaseIndexEntry(
                phase=PHASE_A_LCA,
                function="external_lca",
                status="complete",
                reuse_status=public_phase_reuse_status(run_status="reused_exact"),
                output_root=lca_output_root,
            )
        )
    return ASRBranchPrerequisites(
        phase_entries=phase_entries,
        acc_output_files=acc_output_files,
        dynamic_ar6_summary=dynamic_ar6_summary,
    )
