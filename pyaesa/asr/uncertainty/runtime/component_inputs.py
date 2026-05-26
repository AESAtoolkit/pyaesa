"""Component inputs for uncertainty ASR runs."""

from dataclasses import dataclass
from typing import Any

from pyaesa.acc.uncertainty.runner import run_uncertainty_acc_component
from pyaesa.asr.uncertainty.runtime.models import LCAUncertaintyInput
from pyaesa.asr.uncertainty.runtime.scope import ASRUncertaintyScope
from pyaesa.asr.uncertainty.sources.config import ASRSourceConfig
from pyaesa.asr.uncertainty.sources.lca_inputs import (
    lcia_uncertainty_source_active,
    resolve_lca_uncertainty_component_input,
)
from pyaesa.io_lca.data.contracts import IO_LCA_FAMILY
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import RunProgressPrinter
from pyaesa.shared.runtime.reporting.composite_phase_index import PHASE_A_LCA, PHASE_B2_ACC
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import (
    ComponentRun,
    component_inventory_payload,
    fixed_inventory_mc_parameters,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


@dataclass(frozen=True)
class ASRInitialComponents:
    """Initial pyaesa owned component state for an ASR uncertainty run."""

    acc_manifest: UncertaintyManifest
    acc_reuse_status: str
    acc_session: Any | None
    lca_input: LCAUncertaintyInput
    lca_session: Any | None
    run_id: str


def initial_asr_components(
    *,
    phase: PhasePrinter,
    scope: ASRUncertaintyScope,
    source_config: ASRSourceConfig,
    project_name: str,
    years: int | list[int] | range,
    fu_code: str,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
    r_c: str | list[str] | None,
    r_f: str | list[str] | None,
    base_cc_args: dict[str, Any],
    refresh: bool,
    output_format: str,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    acc_progress: RunProgressPrinter,
    lca_progress: RunProgressPrinter,
    finalize_component_inventory: bool = False,
    run_id: str | None = None,
) -> ASRInitialComponents:
    """Resolve the first ASR aCC and LCA component inventories."""
    acc_run = acc_inventory_report(
        project_name=project_name,
        years=years,
        shared_methods=scope.shared_methods,
        base_allocate_args=scope.base_allocate_args,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        mrio_scope=scope.mrio_scope,
        asocc_config=scope.asocc_config,
        base_cc_args=base_cc_args,
        source_config=source_config.acc_config,
        external_method=scope.external_method,
        output_format=output_format,
        phase=phase,
        target_runs=target_runs,
        parent_mode=parent_mode,
        parent_max_runs=parent_max_runs,
        figures=False,
        figure_options=figure_options,
        figure_format=figure_format,
        subfigures=False,
        show_progress=True,
        show_component_progress=True,
        run_id=run_id,
        refresh=refresh,
        progress=acc_progress,
        component_session=None,
        finalize_component_inventory=finalize_component_inventory,
    )
    phase.announce(PHASE_B2_ACC, "uncertainty_acc")
    acc_manifest = acc_run.report.manifest
    phase.announce(
        PHASE_A_LCA,
        "uncertainty_io_lca" if scope.lca_type == IO_LCA_FAMILY else "external_lca",
    )
    lca_component = resolve_lca_uncertainty_component_input(
        proj_base=scope.proj_base,
        source_label=scope.source_label,
        lca_type=scope.lca_type,
        lca_version_name=scope.lca_version_name,
        base_allocate_args=scope.base_allocate_args,
        lcia_methods=scope.shared_methods,
        uncertainty_config=source_config.lca_config,
        output_format=output_format,
        refresh=refresh,
        phase=phase,
        component_inventory=lca_component_inventory(
            lca_type=scope.lca_type,
            target_runs=target_runs,
            parent_mode=parent_mode,
            parent_max_runs=parent_max_runs,
        ),
        figures=False,
        figure_format=figure_format,
        show_progress=True,
        run_id=acc_manifest.run_id,
        status=lca_progress,
        progress=lca_progress,
        component_session=None,
        finalize_component_inventory=finalize_component_inventory,
        figure_run_count=target_runs,
    )
    return ASRInitialComponents(
        acc_manifest=acc_manifest,
        acc_reuse_status=acc_run.report.reuse_status,
        acc_session=acc_run.session,
        lca_input=lca_component.input,
        lca_session=lca_component.session,
        run_id=acc_manifest.run_id,
    )


def io_lca_progress_enabled(*, scope: ASRUncertaintyScope, source_config: ASRSourceConfig) -> bool:
    """Return whether the ASR managed IO-LCA component has visible run progress."""
    return scope.lca_type == IO_LCA_FAMILY and lcia_uncertainty_source_active(
        source_config.lca_config
    )


def acc_inventory_report(
    *,
    project_name: str,
    years: int | list[int] | range,
    shared_methods: list[str],
    base_allocate_args: dict[str, Any],
    fu_code: str,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
    r_c: str | list[str] | None,
    r_f: str | list[str] | None,
    mrio_scope: dict[str, Any],
    asocc_config: dict[str, Any],
    base_cc_args: dict[str, Any],
    source_config: dict[str, Any],
    external_method: dict[str, Any] | None,
    output_format: str,
    phase,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
    figures: bool,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
    subfigures: bool,
    show_progress: bool,
    show_component_progress: bool,
    run_id: str | None,
    refresh: bool,
    progress: RunProgressPrinter | None = None,
    component_session: Any | None = None,
    finalize_component_inventory: bool = False,
) -> ComponentRun:
    """Run or reuse the aCC component with the ASR aSoCC request."""
    return run_uncertainty_acc_component(
        project_name=project_name,
        source=str(mrio_scope["source"]),
        agg_reg=bool(mrio_scope["agg_reg"]),
        agg_sec=bool(mrio_scope["agg_sec"]),
        agg_version=("" if mrio_scope["agg_version"] is None else str(mrio_scope["agg_version"])),
        years=years,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        group_indices=bool(mrio_scope["group_indices"]),
        lcia_method=shared_methods,
        base_asocc_args=asocc_config,
        external_method=external_method,
        base_cc_args=base_cc_args,
        uncertainty_config={
            **source_config,
            "mc_parameters": fixed_inventory_mc_parameters(target_runs=target_runs),
        },
        sobol_parameters={"active": False},
        output_format=output_format,
        figures=figures,
        figure_options=acc_figure_options_for_asr_component(figure_options),
        figure_format=figure_format,
        subfigures=subfigures,
        refresh=refresh,
        phase=phase,
        component_inventory=component_inventory_payload(
            composite_family="asr",
            component_name="acc",
            target_runs=target_runs,
            parent_mode=parent_mode,
            parent_max_runs=parent_max_runs,
        ),
        run_id=run_id,
        show_progress=show_progress,
        show_component_progress=show_component_progress,
        progress=progress,
        component_session=component_session,
        finalize_component_inventory=finalize_component_inventory,
        asocc_base_allocate_args=base_allocate_args,
    )


def acc_figure_options_for_asr_component(
    figure_options: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the aCC-owned subset of a public ASR figure option request."""
    if figure_options is None:
        return None
    return {
        "per_method": figure_options["per_method"],
        "multi_method": figure_options["multi_method"],
        "inter_method": figure_options["inter_method"],
    }


def lca_component_inventory(
    *,
    lca_type: str,
    target_runs: int,
    parent_mode: str,
    parent_max_runs: int,
) -> dict[str, Any] | None:
    """Return IO-LCA component inventory metadata for ASR managed LCA calls."""
    if lca_type != IO_LCA_FAMILY:
        return None
    return component_inventory_payload(
        composite_family="asr",
        component_name="io_lca",
        target_runs=target_runs,
        parent_mode=parent_mode,
        parent_max_runs=parent_max_runs,
    )
