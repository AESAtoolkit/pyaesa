"""aSoCC uncertainty source and sampling plan assembly."""

from dataclasses import dataclass, replace
from typing import Any

from pyaesa.asocc.uncertainty.engine.inter_method.execution import (
    build_inter_method_execution_plan,
)
from pyaesa.asocc.uncertainty.engine.monte_carlo.batch_sizing import batch_row_count
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    LoadedAsoccFinalRows,
    validate_single_l2_reuse_year_per_identity,
    validate_single_reference_year_per_identity,
)
from pyaesa.asocc.uncertainty.inputs.external_rows import (
    ExternalAsoccRowsPlan,
    resolve_external_asocc_rows,
)
from pyaesa.asocc.uncertainty.sources.activation import (
    deactivate_inter_mrio_without_targets,
    deactivate_sources_without_row_targets,
)
from pyaesa.asocc.uncertainty.sources.inter_method import build_inter_method_plan
from pyaesa.asocc.uncertainty.sources.inter_mrio import build_inter_mrio_plan
from pyaesa.asocc.uncertainty.sources.lcia import (
    LCIASupportRowCache,
    build_lcia_plan,
    lcia_sampling_memory_row_counts,
)
from pyaesa.asocc.uncertainty.sources.names import (
    INTER_METHOD_SOURCE,
    INTER_MRIO_SOURCE,
    LCIA_SOURCE,
    PROJECTION_SOURCE,
    REFERENCE_YEAR_SOURCE,
)
from pyaesa.asocc.uncertainty.sources.projection import build_projection_plan
from pyaesa.asocc.uncertainty.sources.reference_year import admissible_reference_year_rows
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.request.core import BatchMemoryBlock
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan


@dataclass(frozen=True)
class ASOCCSourceScope:
    """Resolved row universe and active source plans for aSoCC uncertainty."""

    loaded: LoadedAsoccFinalRows
    external_plan: ExternalAsoccRowsPlan
    sources: SourceActivationPlan
    inter_method_plan: Any | None
    inter_mrio_plan: Any | None


@dataclass(frozen=True)
class ASOCCSamplingScope:
    """Resolved draw scale sampling plans for aSoCC uncertainty."""

    runtime: UncertaintyRuntimeRequest
    lcia_plan: Any | None
    projection_plan: Any | None
    inter_method_execution_plan: Any | None


def build_asocc_source_scope(
    *,
    loaded: LoadedAsoccFinalRows,
    external_method: dict[str, Any] | None,
    external_lcia_methods: list[str] | None,
    runtime: UncertaintyRuntimeRequest,
    sources: SourceActivationPlan,
    phase: Any = None,
) -> ASOCCSourceScope:
    """Resolve external rows, active uncertainty sources, and source level plans."""
    loaded, external_plan = resolve_external_asocc_rows(
        loaded=loaded,
        external_method=external_method,
        required_runs=(
            runtime.n_runs
            if runtime.mode == "fixed" and not sources.is_active(INTER_METHOD_SOURCE)
            else None
        ),
        external_lcia_methods=external_lcia_methods,
    )
    sources = deactivate_sources_without_row_targets(
        loaded=loaded,
        sources=sources,
        external_plan=external_plan,
    )
    sampled_identity_columns = (
        ("l1_l2_method", "l1_method", "l2_method") if sources.is_active(INTER_METHOD_SOURCE) else ()
    )
    if sources.names and not sources.is_active(PROJECTION_SOURCE):
        validate_single_l2_reuse_year_per_identity(
            rows=loaded.rows,
            sampled_identity_columns=sampled_identity_columns,
        )
    if sources.is_active(INTER_METHOD_SOURCE) and not sources.is_active(REFERENCE_YEAR_SOURCE):
        validate_single_reference_year_per_identity(
            rows=loaded.rows,
            sampled_identity_columns=sampled_identity_columns,
        )
    if sources.is_active(REFERENCE_YEAR_SOURCE):
        loaded = replace(loaded, rows=admissible_reference_year_rows(frame=loaded.rows))
    inter_method_plan = (
        build_inter_method_plan(
            loaded=loaded,
            parameters=sources.parameters_for(INTER_METHOD_SOURCE),
            external_plan=external_plan,
        )
        if sources.is_active(INTER_METHOD_SOURCE)
        else None
    )
    inter_mrio_plan = (
        build_inter_mrio_plan(
            loaded=loaded,
            parameters=sources.parameters_for(INTER_MRIO_SOURCE),
            projection_active=sources.is_active(PROJECTION_SOURCE),
            reference_year_uncertainty_active=sources.is_active(REFERENCE_YEAR_SOURCE),
            external_method_labels=external_plan.method_labels,
            phase=phase,
        )
        if sources.is_active(INTER_MRIO_SOURCE)
        else None
    )
    sources, inter_mrio_plan = deactivate_inter_mrio_without_targets(
        sources=sources,
        plan=inter_mrio_plan,
    )
    return ASOCCSourceScope(
        loaded=loaded,
        external_plan=external_plan,
        sources=sources,
        inter_method_plan=inter_method_plan,
        inter_mrio_plan=inter_mrio_plan,
    )


def build_asocc_sampling_scope(
    *,
    source_scope: ASOCCSourceScope,
    runtime: UncertaintyRuntimeRequest,
    append_existing: bool,
) -> ASOCCSamplingScope:
    """Resolve draw scale source plans and memory bounded run batch size."""
    lcia_support_cache = LCIASupportRowCache()
    lcia_plan = (
        build_lcia_plan(
            loaded=source_scope.loaded,
            parameters=source_scope.sources.parameters_for(LCIA_SOURCE),
            support_cache=lcia_support_cache,
            include_source_methods=not append_existing,
            external_method_labels=source_scope.external_plan.method_labels,
        )
        if source_scope.sources.is_active(LCIA_SOURCE)
        else None
    )
    projection_plan = (
        build_projection_plan(
            loaded=source_scope.loaded,
            external_method_labels=source_scope.external_plan.method_labels,
        )
        if source_scope.sources.is_active(PROJECTION_SOURCE)
        else None
    )
    inter_method_execution_plan = (
        build_inter_method_execution_plan(
            loaded=source_scope.loaded,
            inter_method_plan=source_scope.inter_method_plan,
            sources=source_scope.sources,
            external_plan=source_scope.external_plan,
            lcia_plan=lcia_plan,
            projection_plan=projection_plan,
        )
        if source_scope.inter_method_plan is not None
        else None
    )
    runtime = replace(
        runtime,
        batch_size=runtime_batch_size(
            runtime=runtime,
            source_scope=source_scope,
            lcia_plan=lcia_plan,
            projection_plan=projection_plan,
            inter_method_execution_plan=inter_method_execution_plan,
        ),
    )
    return ASOCCSamplingScope(
        runtime=runtime,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
        inter_method_execution_plan=inter_method_execution_plan,
    )


def runtime_batch_size(
    *,
    runtime: UncertaintyRuntimeRequest,
    source_scope: ASOCCSourceScope,
    lcia_plan: Any | None,
    projection_plan: Any | None,
    inter_method_execution_plan: Any | None,
) -> int:
    """Return the memory bounded batch size for one aSoCC uncertainty run."""
    from pyaesa.shared.uncertainty_assessment.request.core import memory_bounded_batch_size

    lcia_memory_plan = (
        inter_method_execution_plan.lcia_plan
        if inter_method_execution_plan is not None
        else lcia_plan
    )
    batch_size = memory_bounded_batch_size(
        runtime=runtime,
        row_count=batch_row_count(
            loaded=source_scope.loaded,
            inter_method_execution_plan=inter_method_execution_plan,
            inter_mrio_plan=source_scope.inter_mrio_plan,
            lcia_plan=lcia_plan,
            projection_plan=projection_plan,
            sources=source_scope.sources,
            external_plan=source_scope.external_plan,
        ),
        extra_blocks=_lcia_memory_blocks(lcia_plan=lcia_memory_plan),
    )
    return batch_size


def _lcia_memory_blocks(*, lcia_plan: Any | None) -> tuple[BatchMemoryBlock, ...]:
    shared_keys, sampled_rows = lcia_sampling_memory_row_counts(plan=lcia_plan)
    blocks: list[BatchMemoryBlock] = []
    if shared_keys:
        blocks.append(BatchMemoryBlock("asocc_lcia_shared_u", shared_keys))
    if sampled_rows:
        blocks.append(BatchMemoryBlock("asocc_lcia_support_values", sampled_rows, 2))
    return tuple(blocks)
