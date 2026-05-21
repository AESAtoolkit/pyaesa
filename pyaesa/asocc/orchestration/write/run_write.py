"""Write allocation outputs and metadata."""

from ..run_allocate_support import _has_regression_projection_context
from ...runtime.paths.deterministic import (
    _get_allocate_run_metadata_path,
)
from ...io.metadata import _load_run_metadata, _save_run_metadata
from pyaesa.asocc.orchestration.write.writers.allocations import (
    count_l1_output_targets,
    count_l2_output_targets,
    write_l1_outputs,
    write_l2_outputs,
    write_result_artifact,
)
from pyaesa.asocc.orchestration.write.writers.enacting_metric import (
    _write_enacting_metric_outputs,
)
from pyaesa.asocc.orchestration.write.writers.enacting_metric import (
    count_enacting_metric_output_targets,
)
from pyaesa.asocc.orchestration.write.metadata.payload import build_metadata_payload
from pyaesa.asocc.orchestration.write.regression_stats.write import write_regression_stats
from pyaesa.asocc.orchestration.write.writers.ut_gvaa_identity_closure import (
    write_ut_gvaa_identity_closure_audit,
)


def _context_output_source(*, context) -> str:
    """Return output source label used for source scoped logs and metadata."""
    return context.output_source


def _write_run_metadata(
    *,
    context,
    state,
    output_source: str,
    completed_years: list[int],
    outputs: list[str],
    merge_prior_current_scope: bool,
) -> None:
    """Persist deterministic aSoCC run metadata."""
    metadata_path = _get_allocate_run_metadata_path(
        context.proj_base,
        source=output_source,
        group_version=context.group_version,
    )
    payload = build_metadata_payload(
        context=context,
        state=state,
        completed_years_override=completed_years,
        outputs_override=outputs,
        prior_metadata=_load_run_metadata(metadata_path),
        merge_prior_current_scope=merge_prior_current_scope,
    )
    _save_run_metadata(metadata_path, payload)


def _write_outputs(
    *,
    context,
    state,
    refresh: bool,
    write_metadata: bool = True,
    show_progress: bool = True,
    progress_label: str | None = None,
    progress_prefix: str | None = None,
) -> None:
    """Write allocation outputs and run metadata."""
    output_source = _context_output_source(context=context)
    l1_source = output_source
    refresh_effective = refresh
    if show_progress:
        state.write_progress_total = _count_main_output_targets(
            context=context,
            state=state,
        )
        state.write_progress_current = 0
        state.write_progress_last_width = 0
        state.write_progress_label = (
            str(progress_label).strip() if progress_label is not None else None
        ) or None
        state.write_progress_prefix = (
            str(progress_prefix).strip() if progress_prefix is not None else None
        ) or None
    else:
        state.write_progress_total = 0
        state.write_progress_current = 0
        state.write_progress_last_width = 0
        state.write_progress_label = None
        state.write_progress_prefix = None

    write_l1_outputs(
        context=context,
        state=state,
        refresh_effective=refresh_effective,
        l1_source=l1_source,
    )
    write_l2_outputs(
        context=context,
        state=state,
        refresh_effective=refresh_effective,
    )
    _write_enacting_metric_outputs(
        context=context,
        state=state,
        refresh_effective=refresh_effective,
        l1_source=l1_source,
        write_result_artifact=write_result_artifact,
    )
    write_regression_stats(
        context=context,
        state=state,
    )
    write_ut_gvaa_identity_closure_audit(
        context=context,
        state=state,
        refresh_effective=refresh_effective,
    )

    if not write_metadata:
        return

    metadata_completed_years = getattr(context, "metadata_completed_years", None)
    completed_years = list(getattr(state, "processed_years", []))
    if metadata_completed_years is not None:
        completed_years = sorted(
            {int(year) for year in completed_years}
            | {int(year) for year in metadata_completed_years}
        )
    prior_outputs = getattr(context, "metadata_prior_outputs", None) or []
    outputs = list(dict.fromkeys([*prior_outputs, *list(getattr(state, "outputs_all", []))]))
    _write_run_metadata(
        context=context,
        state=state,
        output_source=output_source,
        completed_years=completed_years,
        outputs=outputs,
        merge_prior_current_scope=bool(metadata_completed_years is not None or prior_outputs),
    )


def _count_main_output_targets(*, context, state) -> int:
    """Return number of deterministic allocation/enacting metric files to write now."""
    l1_targets = count_l1_output_targets(state=state)
    l2_targets = count_l2_output_targets(context=context, state=state)
    enacting_metric_targets = count_enacting_metric_output_targets(
        context=context,
        state=state,
    )
    regression_targets = _count_regression_output_targets(
        context=context,
        state=state,
    )
    closure_targets = int(bool(state.ut_gvaa_identity_closure_rows))
    return int(
        l1_targets + l2_targets + enacting_metric_targets + regression_targets + closure_targets
    )


def _count_regression_output_targets(*, context, state) -> int:
    """Return expected number of regression diagnostics files written now."""
    if not _has_regression_projection_context(context=context):
        return 0
    has_stats_rows = bool(state.regression_stats_rows)
    has_fit_rows = bool(state.regression_fit_inputs_rows)
    if has_stats_rows or has_fit_rows:
        return int(has_stats_rows) + int(has_fit_rows)
    return 0
