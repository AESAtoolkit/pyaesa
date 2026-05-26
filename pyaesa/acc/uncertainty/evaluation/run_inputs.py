"""Shared upstream run input helpers for aCC uncertainty evaluation."""

from typing import cast

import numpy as np

from pyaesa.acc.uncertainty.runtime.models import (
    ACCAsoccInput,
    ACCDynamicCCInput,
    ACCUncertaintyPlan,
)
from pyaesa.ar6_cc.uncertainty.io.artifacts import (
    ar6_cc_run_layout_from_manifest,
    ar6_cc_run_paths_from_manifest,
)
from pyaesa.asocc.uncertainty.io.artifacts import (
    asocc_run_layout_from_manifest,
    asocc_run_paths_from_manifest,
)
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import iter_compact_run_matrix
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def iter_asocc_compact_values(
    *,
    paths: AsoccUncertaintyRunPaths,
    output_format: str,
    public_row_count: int,
    start_run_index: int = 0,
    stop_run_index: int | None = None,
    max_rows_per_chunk: int | None = None,
):
    """Yield compact aSoCC run matrix chunks."""
    yield from iter_compact_run_matrix(
        path=paths.public_runs,
        output_format=output_format,
        column_count=public_row_count,
        start_run_index=start_run_index,
        stop_run_index=stop_run_index,
        max_rows_per_chunk=max_rows_per_chunk,
    )


def iter_asocc_values(
    *,
    asocc_input: ACCAsoccInput,
    output_format: str,
    public_row_count: int,
    start_run_index: int,
    stop_run_index: int,
    batch_size: int | None = None,
):
    """Yield aSoCC value chunks from fixed deterministic or compact uncertainty rows."""
    if asocc_input.manifest is not None:
        paths = asocc_run_paths_from_manifest(
            manifest=cast(UncertaintyManifest, asocc_input.manifest)
        )
        yield from iter_asocc_compact_values(
            paths=paths,
            output_format=output_format,
            public_row_count=public_row_count,
            start_run_index=start_run_index,
            stop_run_index=stop_run_index,
            max_rows_per_chunk=batch_size,
        )
        return
    fixed_values = cast(np.ndarray, asocc_input.deterministic_values)
    for start in range(start_run_index, stop_run_index, 1024):
        stop = min(start + 1024, stop_run_index)
        run_indices = np.arange(start, stop, dtype=np.int64)
        yield run_indices, np.broadcast_to(fixed_values, (len(run_indices), len(fixed_values)))


def asocc_layout(*, asocc_input: ACCAsoccInput) -> str:
    """Return the upstream aSoCC run layout used by aCC."""
    if asocc_input.manifest is None:
        return "fixed_values"
    return asocc_run_layout_from_manifest(manifest=cast(UncertaintyManifest, asocc_input.manifest))


def asocc_paths(*, asocc_input: ACCAsoccInput):
    """Return aSoCC run paths for an uncertainty backed aSoCC lane."""
    return asocc_run_paths_from_manifest(manifest=cast(UncertaintyManifest, asocc_input.manifest))


def fixed_cc_values_for_runs(
    *,
    run_indices: np.ndarray,
    deterministic_cc_values: np.ndarray | None,
) -> np.ndarray | None:
    """Return fixed carrying capacity rows repeated for selected run indices."""
    if deterministic_cc_values is None:
        return None
    fixed_values = cast(np.ndarray, deterministic_cc_values)
    return np.broadcast_to(fixed_values, (len(run_indices), len(fixed_values)))


def dynamic_cc_layout(*, dynamic_cc_input: ACCDynamicCCInput | None) -> str:
    """Return the upstream dynamic AR6 CC run layout used by aCC."""
    if dynamic_cc_input is None or dynamic_cc_input.manifest is None:
        return "fixed_values"
    return ar6_cc_run_layout_from_manifest(
        manifest=cast(UncertaintyManifest, dynamic_cc_input.manifest)
    )


def dynamic_cc_paths(*, dynamic_cc_input: ACCDynamicCCInput | None):
    """Return dynamic AR6 CC run paths for an uncertainty backed dynamic branch."""
    dynamic_input = cast(ACCDynamicCCInput, dynamic_cc_input)
    return ar6_cc_run_paths_from_manifest(
        manifest=cast(UncertaintyManifest, dynamic_input.manifest)
    )


def asocc_public_row_count(*, plan: ACCUncertaintyPlan) -> int:
    """Return the compact aSoCC public row count required by the ACC plan."""
    return 1 + max(
        int(position)
        for branch in plan.branch_plans
        for position in branch.asocc_positions.tolist()
    )
