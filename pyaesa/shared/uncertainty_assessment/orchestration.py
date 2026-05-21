"""Shared orchestration helpers for public uncertainty runners."""

from pathlib import Path
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    public_phase_reuse_status,
    phase_ready_detail,
    phase_reused_detail,
    write_phase_index,
)
from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import (
    RunProgressPrinter,
    monte_carlo_run_drawing_label,
    monte_carlo_run_progress_label,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest
from pyaesa.shared.uncertainty_assessment.run_state.report import UncertaintyRunReport
from pyaesa.shared.uncertainty_assessment.run_state.report_roots import (
    uncertainty_manifest_output_root,
)


def component_checkpoint_figures(*, runtime_mode: str, subfigures: bool) -> bool:
    """Return whether pyaesa owned component figures render during checkpoints."""
    return bool(subfigures) and str(runtime_mode) != "convergence"


def final_component_figures_required(
    *,
    runtime_mode: str,
    subfigures: bool,
    completed_runs: int,
) -> bool:
    """Return whether convergence deferred component figures must render once."""
    return bool(subfigures) and str(runtime_mode) == "convergence" and int(completed_runs) > 0


def required_manifest_path(manifest: UncertaintyManifest) -> Path:
    """Return the scope manifest path recorded by a package-owned uncertainty run."""
    return Path(str(manifest.artifacts["scope_manifest"]))


def manifest_output_root(manifest: UncertaintyManifest) -> Path:
    """Return the public run root recorded by one uncertainty manifest."""
    return uncertainty_manifest_output_root(manifest)


def output_root_from_path(path: Path) -> Path:
    """Return the public output root that owns one package generated path."""
    return public_output_root_from_path(path)


def complete_uncertainty_manifest_phase(
    *,
    phase: PhasePrinter,
    scope_name: str,
    report: UncertaintyRunReport,
) -> None:
    """Print one concise phase completion message for an uncertainty subrun."""
    detail_builder = (
        phase_reused_detail if report.reuse_status == "reused_exact" else phase_ready_detail
    )
    phase.complete(
        detail_builder(
            scope_name=scope_name,
            output_root=manifest_output_root(report.manifest),
        )
    )


def deterministic_phase_index_entry(
    *,
    phase_label: str,
    function_name: str,
    metadata_path: Path,
    reuse_status: str,
) -> CompositePhaseIndexEntry:
    """Return one phase index entry for a deterministic prerequisite."""
    return CompositePhaseIndexEntry(
        phase=phase_label,
        function=function_name,
        status="complete",
        reuse_status=public_phase_reuse_status(run_status=reuse_status),
        output_root=output_root_from_path(metadata_path),
    )


def phase_index_entry(
    *,
    phase_label: str,
    function_name: str,
    reuse_status: str,
    output_root: Path | None,
) -> CompositePhaseIndexEntry:
    """Return one phase index entry for a public phase with a resolved output root."""
    return CompositePhaseIndexEntry(
        phase=phase_label,
        function=function_name,
        status="complete",
        reuse_status=public_phase_reuse_status(run_status=reuse_status),
        output_root=output_root,
    )


def uncertainty_phase_index_entry(
    *,
    phase_label: str,
    function_name: str,
    manifest: UncertaintyManifest,
    reuse_status: str,
) -> CompositePhaseIndexEntry:
    """Return one phase index entry for an uncertainty run."""
    return CompositePhaseIndexEntry(
        phase=phase_label,
        function=function_name,
        status="complete",
        reuse_status=public_phase_reuse_status(run_status=reuse_status),
        output_root=manifest_output_root(manifest),
    )


def write_uncertainty_phase_index(
    *,
    manifest: UncertaintyManifest,
    entries: list[CompositePhaseIndexEntry],
) -> Path:
    """Persist the public phase index beside an uncertainty scope manifest."""
    return write_phase_index(
        metadata_path=required_manifest_path(manifest),
        entries=entries,
    )


def progress_begin(
    *,
    progress: RunProgressPrinter,
    completed: int,
    max_runs: int,
    target_runs: int,
    mode: str = "convergence",
    component: bool = False,
) -> None:
    """Begin or update one Monte Carlo progress line."""
    progress.begin(
        label=monte_carlo_run_drawing_label(
            start=completed,
            stop=target_runs,
            max_runs=max_runs,
            mode=mode,
            component=component,
        )
    )


def progress_complete(
    *,
    progress: RunProgressPrinter,
    completed: int,
    max_runs: int,
    mode: str = "convergence",
    persistent: bool | None = None,
    component: bool = False,
    visible: bool = True,
) -> None:
    """Complete one Monte Carlo progress line."""
    if not visible:
        return
    progress.complete(
        label=monte_carlo_run_progress_label(
            completed=completed,
            max_runs=max_runs,
            mode=mode,
            component=component,
        ),
        persistent=str(mode) == "fixed" if persistent is None else persistent,
    )
