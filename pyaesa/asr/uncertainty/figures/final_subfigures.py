"""Render nested aCC and LCA figures after ASR uncertainty runs."""

from typing import Any

from pyaesa.asr.uncertainty.runtime.component_inputs import acc_inventory_report
from pyaesa.asr.uncertainty.runtime.scope import ASRUncertaintyScope
from pyaesa.asr.uncertainty.sources.config import ASRSourceConfig
from pyaesa.asr.uncertainty.sources.lca_inputs import render_lca_subfigures_from_input
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import monte_carlo_run_progress
from pyaesa.shared.runtime.reporting.status import StatusSink


def render_final_asr_subfigures(
    *,
    plan,
    scope: ASRUncertaintyScope,
    source_config: ASRSourceConfig,
    base_cc_args: dict[str, Any],
    output_format: str,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    phase: PhasePrinter,
    status: StatusSink,
    completed_runs: int,
    component_session: Any | None,
    parent_mode: str,
    parent_max_runs: int,
    report_reused_progress: bool,
) -> None:
    """Render final ASR nested aCC and LCA subfigures."""
    args = scope.base_args
    acc_progress = monte_carlo_run_progress(
        source="uncertainty_acc",
        enabled=True,
        status=status,
    )
    acc_inventory_report(
        project_name=str(args["project_name"]),
        years=args["years"],
        shared_methods=scope.shared_methods,
        base_allocate_args=scope.base_allocate_args,
        fu_code=str(args["fu_code"]),
        r_p=args["r_p"],
        s_p=args["s_p"],
        r_c=args["r_c"],
        r_f=args["r_f"],
        mrio_scope=scope.mrio_scope,
        asocc_config=scope.asocc_config,
        base_cc_args=base_cc_args,
        source_config=source_config.acc_config,
        external_method=scope.external_method,
        output_format=output_format,
        phase=phase,
        target_runs=completed_runs,
        parent_mode=parent_mode,
        parent_max_runs=parent_max_runs,
        figures=True,
        figure_options=figure_options,
        figure_format=figure_format,
        subfigures=True,
        show_progress=False,
        show_component_progress=False,
        run_id=plan.acc_manifest.run_id,
        refresh=False,
        progress=acc_progress,
        component_session=component_session,
        finalize_component_inventory=True,
        report_reused_progress=report_reused_progress,
    )
    render_lca_subfigures_from_input(
        lca_input=plan.lca_input,
        base_allocate_args=scope.base_allocate_args,
        lcia_methods=scope.shared_methods,
        lca_version_name=scope.lca_version_name,
        lca_config=source_config.lca_config,
        figure_format=figure_format,
        status=status,
        completed_runs=completed_runs,
        phase=phase,
        complete_phase=True,
    )
