"""Shared deterministic prerequisite orchestration for aCC and ASR."""

from dataclasses import dataclass
from typing import Any, cast

from pyaesa.asocc.deterministic_asocc import deterministic_asocc
from pyaesa.asocc.runtime.scope.branch_resolution import (
    allocate_run_metadata_path,
    resolve_allocate_path_scope,
)
from pyaesa.asocc.runtime.reporting.deterministic_summary import (
    deterministic_asocc_phase_inventory_lines,
    deterministic_asocc_phase_summary_lines,
    deterministic_asocc_summary_record_messages,
)
from pyaesa.ar6_cc.deterministic_ar6_cc import deterministic_ar6_cc
from pyaesa.ar6_cc.deterministic.io.paths import (
    get_cc_scope_dir,
)
from pyaesa.process.ar6.utils.pipeline.study_period import resolve_study_period
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    PHASE_B1_AR6_DYNAMIC_CC,
    PHASE_B1_ASOCC,
    phase_ready_detail,
    phase_reused_detail,
    public_phase_reuse_status,
)
from pyaesa.shared.acc_asr_common.reporting import DynamicAR6PathwayCount, DynamicAR6Summary
from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.process.ar6.utils.pipeline import study_period as ar6_study_period


@dataclass(frozen=True)
class ACCBranchPrerequisites:
    """Resolved deterministic prerequisites for one aCC branch."""

    phase_entries: list[CompositePhaseIndexEntry]
    dynamic_ar6_summary: DynamicAR6Summary | None


def _base_asocc_kwargs(*, base_allocate_args: dict[str, Any]) -> dict[str, Any]:
    """Return the public aSoCC kwargs for one shared scope."""
    return {
        "project_name": base_allocate_args["project_name"],
        "source": base_allocate_args["source"],
        "group_reg": base_allocate_args["group_reg"],
        "group_sec": base_allocate_args["group_sec"],
        "group_version": base_allocate_args["group_version"],
        "years": base_allocate_args["years"],
        "fu_code": base_allocate_args["fu_code"],
        "r_p": base_allocate_args["r_p"],
        "s_p": base_allocate_args["s_p"],
        "r_c": base_allocate_args["r_c"],
        "r_f": base_allocate_args["r_f"],
        "aggreg_indices": base_allocate_args["aggreg_indices"],
        "method_plan": base_allocate_args["method_plan"],
        "l1_methods": base_allocate_args["l1_methods"],
        "one_step_methods": base_allocate_args["one_step_methods"],
        "two_step_methods": base_allocate_args["two_step_methods"],
        "l1_l2_pairs": base_allocate_args["l1_l2_pairs"],
        "l1_reg_aggreg": base_allocate_args["l1_reg_aggreg"],
        "lcia_method": base_allocate_args["lcia_method"],
        "reference_years": base_allocate_args["reference_years"],
        "ssp_scenario": base_allocate_args["ssp_scenario"],
        "projection_mode": base_allocate_args["projection_mode"],
        "reg_window": base_allocate_args["reg_window"],
        "l2_reuse_years": base_allocate_args["l2_reuse_years"],
    }


def ensure_acc_branch_prerequisites(
    *,
    phase: PhasePrinter,
    base_allocate_args: dict[str, Any],
    cc_source: str,
    cc_type: str,
    years: int | list[int] | range,
    harmonization: bool,
    harmonization_method: str,
    category: list[str] | None,
    ssp_scenario: list[str] | None,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str,
    subset_version: str | None,
    output_format: str,
    figures: bool = False,
    figure_format: dict[str, Any] | None = None,
    figure_options: dict[str, bool] | None = None,
    refresh: bool = False,
) -> ACCBranchPrerequisites:
    """Ensure one aCC branch has all required deterministic upstream outputs."""
    phase_entries: list[CompositePhaseIndexEntry] = []
    dynamic_ar6_summary: DynamicAR6Summary | None = None
    asocc_scope = resolve_allocate_path_scope(base_allocate_args=base_allocate_args)
    asocc_metadata_path = allocate_run_metadata_path(scope=asocc_scope)
    asocc_output_root = public_output_root_from_path(asocc_metadata_path)
    phase.announce(PHASE_B1_ASOCC, "deterministic_asocc")
    asocc_report = deterministic_asocc(
        **_base_asocc_kwargs(base_allocate_args=base_allocate_args),
        figures=figures,
        figure_format=figure_format or {"format": "png", "dpi": 500},
        figure_options=figure_options or {"per_method": True, "multi_method": True},
        refresh=refresh,
        _phase=phase,
    )
    asocc_detail = (
        phase_reused_detail(
            scope_name="aSoCC",
            output_root=asocc_output_root,
        )
        if asocc_report.reuse_status == "reused_exact"
        else phase_ready_detail(
            scope_name="aSoCC",
            output_root=asocc_output_root,
        )
    )
    phase.complete(asocc_detail)
    phase_entries.append(
        CompositePhaseIndexEntry(
            phase=PHASE_B1_ASOCC,
            function="deterministic_asocc",
            status="complete",
            reuse_status=public_phase_reuse_status(run_status=asocc_report.reuse_status),
            output_root=asocc_output_root,
            summary_lines=tuple(
                deterministic_asocc_phase_summary_lines(
                    metadata_path=asocc_metadata_path,
                    output_root=asocc_output_root,
                    source=str(base_allocate_args["source"]),
                )
            ),
            info_messages=tuple(
                deterministic_asocc_summary_record_messages(
                    metadata_path=asocc_metadata_path,
                    severity="INFO",
                )
            ),
            warning_messages=tuple(
                deterministic_asocc_summary_record_messages(
                    metadata_path=asocc_metadata_path,
                    severity="WARNING",
                )
            ),
            inventory_lines=tuple(
                deterministic_asocc_phase_inventory_lines(
                    metadata_path=asocc_metadata_path,
                    output_root=asocc_output_root,
                )
            ),
        )
    )
    if cc_type == "dynamic_ar6":
        phase.expect_visible(PHASE_B1_AR6_DYNAMIC_CC)
        phase.announce(PHASE_B1_AR6_DYNAMIC_CC, "deterministic_ar6_cc")
        normalized_study_period = ar6_study_period.resolve_study_period(years)
        cc_years = range(int(normalized_study_period[0]), int(normalized_study_period[1]) + 1)
        cc_report = deterministic_ar6_cc(
            years=cc_years,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            category=cast(list[str], category),
            ssp_scenario=cast(list[str], ssp_scenario),
            emission_type=emission_type,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode,
            subset_version=subset_version,
            output_format=output_format,
            figures=figures,
            figure_format=figure_format or {"format": "png", "dpi": 500},
            refresh=refresh,
            _status=phase,
        )
        dynamic_ar6_summary = DynamicAR6Summary(
            categories=list(cc_report.categories),
            ssp_scenarios=list(cc_report.ssp_scenarios),
            subset_version=cc_report.subset_version,
            pathway_counts=[
                DynamicAR6PathwayCount(
                    category=item.category,
                    ssp_scenario=item.ssp_scenario,
                    model_scenario_pairs=int(item.model_scenario_pairs),
                )
                for item in cc_report.pathway_counts
            ],
            missing_pathway_combinations=[
                DynamicAR6PathwayCount(
                    category=item.category,
                    ssp_scenario=item.ssp_scenario,
                    model_scenario_pairs=int(item.model_scenario_pairs),
                )
                for item in cc_report.missing_pathway_combinations
            ],
            study_period=list(cc_report.study_period),
            harmonization=cc_report.harmonization,
            harmonization_method=cc_report.harmonization_method,
            emission_type=cc_report.emission_type,
            include_afolu=cc_report.include_afolu,
            emissions_mode=cc_report.emissions_mode,
            variable=cc_report.variable,
            total_model_scenario_pairs=cc_report.total_model_scenario_pairs,
            process_ar6=cc_report.process_ar6,
        )
        study_period = resolve_study_period(cc_years)
        cc_scope_dir = get_cc_scope_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            emission_type=emission_type,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode,
            subset_version=subset_version,
            category=cast(list[str], category),
            ssp_scenario=cast(list[str], ssp_scenario),
        )
        cc_output_root = public_output_root_from_path(cc_scope_dir)
        cc_detail = (
            phase_reused_detail(
                scope_name="dynamic AR6 CC",
                output_root=cc_output_root,
            )
            if cc_report.reuse_status == "reused_exact"
            else phase_ready_detail(
                scope_name="dynamic AR6 CC",
                output_root=cc_output_root,
            )
        )
        phase.complete(cc_detail, owner="deterministic_ar6_cc")
        phase_entries.append(
            CompositePhaseIndexEntry(
                phase=PHASE_B1_AR6_DYNAMIC_CC,
                function="deterministic_ar6_cc",
                status="complete",
                reuse_status=public_phase_reuse_status(run_status=cc_report.reuse_status),
                output_root=cc_output_root,
            )
        )
    return ACCBranchPrerequisites(
        phase_entries=phase_entries,
        dynamic_ar6_summary=dynamic_ar6_summary,
    )
