"""Render nested aSoCC and dynamic AR6 CC figures after aCC uncertainty runs."""

from typing import Any

from pyaesa.acc.uncertainty.runtime.component_inputs import deterministic_asocc_input
from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyPlan
from pyaesa.acc.uncertainty.runtime.scope import ACCUncertaintyScope
from pyaesa.acc.uncertainty.sources.dynamic_cc import deterministic_dynamic_cc_input
from pyaesa.ar6_cc.uncertainty.figures.reuse import (
    render_reusable_ar6_cc_figures_if_requested,
)
from pyaesa.asocc.uncertainty.figures.reuse import render_reusable_asocc_figures_if_requested
from pyaesa.shared.acc_asr_common.scope.composite import base_asocc_kwargs_from_allocate_args
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    phase_reused_detail,
    phase_uncertainty_done_detail,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.uncertainty_assessment.orchestration import (
    manifest_output_root,
    progress_complete_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def render_final_acc_subfigures(
    *,
    plan: ACCUncertaintyPlan,
    scope: ACCUncertaintyScope,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    phase: PhasePrinter | NullPhasePrinter,
    status: StatusSink | None,
    report_reused_progress: bool,
) -> None:
    """Render final aCC nested subfigures and visible component completion lines."""
    if plan.asocc_input.manifest is not None:
        progress_complete_manifest(
            source="uncertainty_asocc",
            status=phase,
            manifest=plan.asocc_input.manifest,
            visible=report_reused_progress,
        )
        asocc_manifest = render_reusable_asocc_figures_if_requested(
            manifest=plan.asocc_input.manifest,
            figures=True,
            figure_options=figure_options,
            figure_format=figure_format,
            status=status,
        )
        _complete_visible_nested_uncertainty(
            phase=phase,
            owner="uncertainty_asocc",
            scope_name="aSoCC uncertainty",
            manifest=asocc_manifest,
            reuse_status=plan.asocc_input.reuse_status,
        )
    else:
        deterministic_asocc_input(
            phase=NullPhasePrinter(),
            base_asocc_args=base_asocc_kwargs_from_allocate_args(
                base_allocate_args=scope.base_allocate_args
            ),
            external_lcia_methods=scope.asocc_lcia_methods,
            external_method=scope.base_args["external_method"],
            figures=True,
            figure_options=figure_options,
            figure_format=figure_format,
            refresh=False,
        )
    dynamic_cc_input = plan.dynamic_cc_input
    if dynamic_cc_input is not None and dynamic_cc_input.manifest is not None:
        progress_complete_manifest(
            source="uncertainty_ar6_cc",
            status=phase,
            manifest=dynamic_cc_input.manifest,
            visible=report_reused_progress,
        )
        dynamic_manifest = render_reusable_ar6_cc_figures_if_requested(
            manifest=dynamic_cc_input.manifest,
            figure_options=None,
            figure_format=figure_format,
            status=status,
        )
        _complete_visible_nested_uncertainty(
            phase=phase,
            owner="uncertainty_ar6_cc",
            scope_name="dynamic AR6 CC uncertainty",
            manifest=dynamic_manifest,
            reuse_status=dynamic_cc_input.reuse_status,
        )
    elif (
        dynamic_cc_input is not None
        and dynamic_cc_input.deterministic_manifest_path is not None
        and scope.dynamic_branch is not None
    ):
        deterministic_dynamic_cc_input(
            branch=scope.dynamic_branch,
            years=scope.base_args["years"],
            figures=True,
            figure_format=figure_format,
            status=status,
            refresh=False,
        )


def _complete_visible_nested_uncertainty(
    *,
    phase: PhasePrinter | NullPhasePrinter,
    owner: str,
    scope_name: str,
    manifest: UncertaintyManifest,
    reuse_status: str,
) -> None:
    if str(reuse_status) == "reused_exact":
        detail = phase_reused_detail(
            scope_name=scope_name,
            output_root=manifest_output_root(manifest),
        )
    else:
        detail = phase_uncertainty_done_detail(
            scope_name=scope_name,
            mode=manifest.mode,
            convergence=manifest.convergence,
            output_root=manifest_output_root(manifest),
        )
    phase.complete(detail, owner=owner)
