"""Run metadata payload construction for allocation output writes."""

from ....io.metadata import (
    _build_run_metadata,
    _normalize_reg_window_for_storage,
)
from pyaesa.asocc.orchestration.reporting_records import deterministic_asocc_info_messages
from pyaesa.asocc.orchestration.write.regression_stats.paths_io import existing_scoped_stats_paths


def _ordered_unique_text(values: list[object]) -> list[str]:
    """Return non-empty strings preserving first occurrence order."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _merge_scope_payload(*, prior_scope: dict, current_scope: dict) -> dict:
    """Return current deterministic scope metadata including earlier completed years."""
    merged = dict(current_scope)
    prior_execution = dict(prior_scope["execution"])
    current_execution = dict(current_scope["execution"])
    current_execution["completed_years"] = sorted(
        {int(year) for year in prior_execution.get("completed_years", [])}
        | {int(year) for year in current_execution.get("completed_years", [])}
    )
    current_artifacts = dict(current_scope["artifacts"])
    prior_artifacts = dict(prior_scope["artifacts"])
    current_artifacts["outputs"] = _ordered_unique_text(
        [*prior_artifacts.get("outputs", []), *current_artifacts.get("outputs", [])]
    )
    merged["execution"] = current_execution
    merged["artifacts"] = current_artifacts
    return merged


def _summary_records(*, context, state) -> list[dict[str, str]]:
    """Return structured summary records owned by deterministic aSoCC metadata."""
    records: list[dict[str, str]] = []
    for message in deterministic_asocc_info_messages(context=context):
        records.append({"severity": "INFO", "message": str(message)})
    for level_raw, message_raw in getattr(state, "startup_notices", []):
        level = str(level_raw).strip().upper()
        message = str(message_raw).strip()
        if level in {"INFO", "WARNING"} and message:
            records.append({"severity": level, "message": message})
    for message_raw in getattr(state, "summary_warnings", []):
        message = str(message_raw).strip()
        if message:
            records.append({"severity": "WARNING", "message": message})
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record["severity"], record["message"])
        if key not in seen:
            unique.append(record)
            seen.add(key)
    return unique


def build_metadata_payload(
    *,
    context,
    state,
    completed_years_override: list[int] | None = None,
    outputs_override: list[str] | None = None,
    prior_metadata: dict | None = None,
    merge_prior_current_scope: bool = False,
) -> dict:
    """Build metadata payload for a completed deterministic run."""
    output_source = context.output_source
    if completed_years_override is None:
        completed_years = sorted({int(year) for year in state.processed_years})
    else:
        completed_years = sorted({int(year) for year in completed_years_override})
    if outputs_override is None:
        outputs_value = list(getattr(state, "outputs_all", []))
    else:
        outputs_value = [str(path).strip() for path in outputs_override if str(path).strip()]
    signature_selected_methods = context.run_signature.get(
        "selected_methods",
        context.selected_methods,
    )
    run_meta = _build_run_metadata(
        requested_years=context.requested_years,
        resolved_years=context.resolved_years,
        selected_methods=signature_selected_methods,
        fu_code=context.fu_code,
        studied_indices_tag=context.studied_indices_tag,
        skipped_years=state.skipped_years,
        outputs=outputs_value,
        signature=context.run_signature,
    )
    run_meta["execution"]["completed_years"] = completed_years
    run_meta["execution"]["empty_reference_years"] = state.empty_ref_years
    run_meta["provenance"]["reference_years"] = context.reference_years or "all"
    run_meta["provenance"]["filters"] = context.filters
    run_meta["provenance"]["ssp_scenarios"] = context.run_signature.get(
        "ssp_scenario_input",
        context.ssp_scenario_options,
    )
    projection_context = context.projection_context
    run_meta["provenance"]["projection"] = {
        "enabled": bool(projection_context.enabled) if projection_context is not None else False,
        "mode": projection_context.mode if projection_context is not None else None,
        "reg_window": (
            _normalize_reg_window_for_storage(projection_context.reg_window)
            if projection_context is not None and projection_context.reg_window is not None
            else None
        ),
        "l2_reuse_years": (
            list(projection_context.l2_reuse_years) if projection_context is not None else []
        ),
    }
    stats_paths = existing_scoped_stats_paths(
        proj_base=context.proj_base,
        output_format=context.output_format,
        source=output_source,
        group_version=context.group_version,
    )
    run_meta["artifacts"]["regression_stats_path"] = (
        str(stats_paths[0]) if len(stats_paths) == 1 else None
    )
    run_meta["artifacts"]["regression_stats_paths"] = [str(path) for path in stats_paths]
    run_meta["summary_records"] = _summary_records(context=context, state=state)
    if prior_metadata and merge_prior_current_scope:
        run_meta = _merge_scope_payload(
            prior_scope=prior_metadata,
            current_scope=run_meta,
        )
    return run_meta
