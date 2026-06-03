"""aCC uncertainty composite phase index entries."""

from pathlib import Path
from typing import cast

from pyaesa.acc.uncertainty.runtime.models import ACCAsoccInput, ACCDynamicCCInput
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    PHASE_B0_AR6_DYNAMIC_CC,
    PHASE_B1_ASOCC,
)
from pyaesa.shared.uncertainty_assessment.orchestration import (
    deterministic_phase_index_entry,
    uncertainty_phase_index_entry,
)


def asocc_phase_entries(*, asocc_input: ACCAsoccInput) -> list[CompositePhaseIndexEntry]:
    """Return the completed aSoCC component phase entry for one aCC run."""
    if asocc_input.manifest is not None:
        return [
            uncertainty_phase_index_entry(
                phase_label=PHASE_B1_ASOCC,
                function_name="uncertainty_asocc",
                manifest=asocc_input.manifest,
                reuse_status=asocc_input.reuse_status,
            )
        ]
    return [
        deterministic_phase_index_entry(
            phase_label=PHASE_B1_ASOCC,
            function_name="deterministic_asocc",
            metadata_path=cast(Path, asocc_input.deterministic_manifest_path),
            reuse_status=asocc_input.reuse_status,
        )
    ]


def dynamic_cc_phase_entries(
    *,
    dynamic_cc_input: ACCDynamicCCInput | None,
) -> list[CompositePhaseIndexEntry]:
    """Return the completed dynamic AR6 CC component phase entry for one aCC run."""
    if dynamic_cc_input is None:
        return []
    if dynamic_cc_input.manifest is not None:
        return [
            uncertainty_phase_index_entry(
                phase_label=PHASE_B0_AR6_DYNAMIC_CC,
                function_name="uncertainty_ar6_cc",
                manifest=dynamic_cc_input.manifest,
                reuse_status=dynamic_cc_input.reuse_status,
            )
        ]
    return [
        deterministic_phase_index_entry(
            phase_label=PHASE_B0_AR6_DYNAMIC_CC,
            function_name="deterministic_ar6_cc",
            metadata_path=cast(Path, dynamic_cc_input.deterministic_manifest_path),
            reuse_status=dynamic_cc_input.reuse_status,
        )
    ]
