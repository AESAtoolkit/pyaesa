"""Phase reporting for public aSoCC uncertainty runs."""

from pyaesa.asocc.uncertainty.engine.reuse.prerequisites import (
    AsoccDeterministicPrerequisite,
)
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    PHASE_B1_ASOCC,
    phase_ready_detail,
    phase_reused_detail,
    phase_uncertainty_done_detail,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.uncertainty_assessment.orchestration import (
    deterministic_phase_index_entry,
    manifest_output_root,
    output_root_from_path,
    uncertainty_phase_index_entry,
    write_uncertainty_phase_index,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest

AsoccPhasePrinter = PhasePrinter | NullPhasePrinter


def asocc_phase_owner(phase: AsoccPhasePrinter | None) -> AsoccPhasePrinter:
    """Return the direct run phase printer or a caller supplied orchestration printer."""
    return PhasePrinter("uncertainty_asocc") if phase is None else phase


def asocc_deterministic_phase_entries(
    *,
    prerequisite: AsoccDeterministicPrerequisite,
) -> list[CompositePhaseIndexEntry]:
    """Return phase index entries for the deterministic aSoCC prerequisite."""
    return [
        deterministic_phase_index_entry(
            phase_label=PHASE_B1_ASOCC,
            function_name="deterministic_asocc",
            metadata_path=prerequisite.deterministic_manifest_path,
            reuse_status=prerequisite.reuse_status,
        )
    ]


def complete_deterministic_asocc_phase(
    *,
    phase: AsoccPhasePrinter,
    prerequisite: AsoccDeterministicPrerequisite,
) -> None:
    """Print deterministic aSoCC prerequisite completion for aSoCC uncertainty."""
    detail_builder = (
        phase_reused_detail if prerequisite.reuse_status == "reused_exact" else phase_ready_detail
    )
    phase.expect_visible(PHASE_B1_ASOCC)
    phase.complete(
        detail_builder(
            scope_name="aSoCC deterministic",
            output_root=output_root_from_path(prerequisite.deterministic_manifest_path),
        )
    )


def complete_asocc_uncertainty_phase(
    *,
    phase: AsoccPhasePrinter,
    manifest: UncertaintyManifest,
    reuse_status: str,
) -> None:
    """Print aSoCC uncertainty completion for direct and orchestrated calls."""
    if reuse_status == "reused_exact":
        detail = phase_reused_detail(
            scope_name="aSoCC uncertainty",
            output_root=manifest_output_root(manifest),
        )
    else:
        detail = phase_uncertainty_done_detail(
            scope_name="aSoCC uncertainty",
            mode=manifest.mode,
            convergence=manifest.convergence,
            output_root=manifest_output_root(manifest),
        )
    phase.complete(detail)


def write_asocc_phase_index(
    *,
    manifest: UncertaintyManifest,
    phase_entries: list[CompositePhaseIndexEntry],
    reuse_status: str,
) -> None:
    """Persist the aSoCC uncertainty composite phase index."""
    write_uncertainty_phase_index(
        manifest=manifest,
        entries=[
            *phase_entries,
            uncertainty_phase_index_entry(
                phase_label=PHASE_B1_ASOCC,
                function_name="uncertainty_asocc",
                manifest=manifest,
                reuse_status=reuse_status,
            ),
        ],
    )
