"""Published-output disaggregation pipeline."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pyaesa.shared.runtime.reporting.labels import output_files_available_line
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.metadata.json import read_optional_json_dict
from pyaesa.shared.runtime.text import wrap_user_text_lines

from ..io.logging import close_loggers_for_scope
from ..orchestration.setup.run_setup import _prepare_context
from ..orchestration.run_allocate_support import format_summary_years
from pyaesa.asocc.runtime.scope.branch_resolution import (
    allocate_run_metadata_path,
    asocc_l2_dir,
    path_scope_from_signature,
)
from pyaesa.asocc.runtime.reporting.family import sync_asocc_branch_figures
from .branch_context import _build_selector_request, validate_region_compatibility
from .completion import (
    is_disaggregation_branch_complete,
    write_branch_metadata,
    write_scope_manifest,
)
from .matching import match_selector_scope
from .models import (
    DisaggregationBranchReport,
    DisaggregationReport,
    MatchedRun,
    ParsedArgs,
    PreparedBranchContext,
)
from pyaesa.asocc.disaggregation.paths import (
    disaggregation_audit_path,
    disaggregation_logs_dir,
    disaggregation_metadata_path,
    disaggregation_source_root,
)
from .published_storage import (
    disaggregate_rows,
    load_partitioned_rows,
    write_partitioned_rows,
)
from .run_plan import build_disaggregation_run_plan


def _disaggregated_context(
    *,
    parsed: ParsedArgs,
    selector,
    mode: str,
    aggreg_indices: bool,
    l1_methods: list[str],
    combined_methods: list[tuple[str, str]],
    one_step_methods: list[str],
) -> PreparedBranchContext:
    """Prepare the published disaggregated branch identity."""
    request = _build_selector_request(
        selector=selector,
        base_allocate_args=parsed.base_allocate_args,
        l1_methods=l1_methods,
        combined_methods=combined_methods,
        one_step_methods=one_step_methods,
        l1_reg_aggreg=mode,
        aggreg_indices=aggreg_indices,
        variant_tag=None,
        output_format=parsed.output_format,
        output_source_label=parsed.disaggregation.new_disaggregated_version_name,
    )
    context, _state, _skipped = _prepare_context(request=request)
    return PreparedBranchContext(
        matched_runs={},
        requested_years=list(context.requested_years),
        ssp_scenario_options_by_year=dict(context.ssp_scenario_options_by_year or {}),
        disagg_run_signature=dict(context.run_signature),
        disagg_proj_base=context.proj_base,
        branch_complete=False,
    )


def _match_runs(
    *,
    parsed: ParsedArgs,
    mode: str,
    aggreg_indices: bool,
    l1_methods: list[str],
    combined_methods: list[tuple[str, str]],
    one_step_methods: list[str],
    requested_years: list[int],
) -> dict[str, MatchedRun]:
    """Match all prerequisite deterministic selector scopes for one branch."""
    validate_region_compatibility(
        target_selector=parsed.disaggregation.target_grouped_run,
        ref_grouped_selector=parsed.disaggregation.ref_grouped_run,
        ref_split_selector=parsed.disaggregation.ref_split_run,
        base_allocate_args=parsed.base_allocate_args,
        combined_methods=combined_methods,
    )
    selectors = {
        "target_grouped_run": parsed.disaggregation.target_grouped_run,
        "ref_grouped_run": parsed.disaggregation.ref_grouped_run,
        "ref_split_run": parsed.disaggregation.ref_split_run,
    }
    matched_runs: dict[str, MatchedRun] = {}
    for name, selector in selectors.items():
        request = _build_selector_request(
            selector=selector,
            base_allocate_args=parsed.base_allocate_args,
            l1_methods=l1_methods,
            combined_methods=combined_methods,
            one_step_methods=one_step_methods,
            l1_reg_aggreg=mode,
            aggreg_indices=aggreg_indices,
            variant_tag=None,
        )
        matched_runs[name] = match_selector_scope(
            selector_name=name,
            request=request,
            requested_years=requested_years,
        )
    return matched_runs


def _scope_for_matched_run(matched_run, *, context_label: str):
    """Return the canonical branch scope for one matched prerequisite run."""
    return path_scope_from_signature(
        proj_base=matched_run.proj_base,
        source_label=matched_run.output_source_label,
        run_signature=matched_run.scope_signature,
        context_label=context_label,
    )


def _run_family(
    *,
    target_scope,
    ref_grouped_scope,
    ref_split_scope,
    disagg_scope,
    grouped_sector_by_split: dict[str, str],
    bucket: str,
    stem_prefix: str,
    requested_years: list[int],
    output_format: str,
) -> tuple[list[Path], pd.DataFrame]:
    """Disaggregate one published output family into the target-owned partition layout."""
    target_rows, target_schemas = load_partitioned_rows(
        root=asocc_l2_dir(scope=target_scope, bucket=bucket, lcia_sub=None),
        stem_prefix=stem_prefix,
        requested_years=requested_years,
        require_requested_coverage=True,
    )
    ref_grouped_rows, _ = load_partitioned_rows(
        root=asocc_l2_dir(scope=ref_grouped_scope, bucket=bucket, lcia_sub=None),
        stem_prefix=stem_prefix,
        requested_years=requested_years,
        require_requested_coverage=True,
    )
    ref_split_rows, _ = load_partitioned_rows(
        root=asocc_l2_dir(scope=ref_split_scope, bucket=bucket, lcia_sub=None),
        stem_prefix=stem_prefix,
        requested_years=requested_years,
        require_requested_coverage=True,
    )
    output_rows, audit = disaggregate_rows(
        target_rows=target_rows,
        ref_grouped_rows=ref_grouped_rows,
        ref_split_rows=ref_split_rows,
        grouped_sector_by_split=grouped_sector_by_split,
    )
    written = write_partitioned_rows(
        rows=output_rows,
        schemas=target_schemas,
        output_root=asocc_l2_dir(scope=disagg_scope, bucket=bucket, lcia_sub=None),
        output_format=output_format,
    )
    audit["bucket"] = bucket
    audit["method_stem"] = stem_prefix
    return written, audit


def _time_route_warning_lines(*, audit_frame: pd.DataFrame) -> list[str]:
    """Return report warning lines for source-specific time-route bridges."""
    bridge_columns = [
        column
        for column in ("ref_grouped_time_route_bridge", "ref_split_time_route_bridge")
        if column in audit_frame.columns
    ]
    if audit_frame.empty or not bridge_columns:
        return []
    bridged = audit_frame.loc[:, bridge_columns].fillna(False).astype(bool).any(axis=1)
    if not bool(bridged.any()):
        return []
    year_values = pd.Series(pd.to_numeric(audit_frame.loc[bridged, "year"], errors="raise"))
    years = sorted({int(year) for year in year_values.tolist()})
    return wrap_user_text_lines(
        [
            "WARNING: The target grouped run required a time-route bridge for selected "
            f"source rows in years {format_summary_years(years)} because one selected MRIO "
            "run is historical while the other is prospective. Disaggregation kept the "
            "target grouped run year and asocc_time_route: regression target rows used "
            "selected source values from the same studied year, and historical_reuse target "
            "rows used selected source values matched on the same l2_reuse_year. These "
            "affected years are automatically skipped for inter-MRIO uncertainty of "
            "allocated shares by downstream uncertainty_asocc, uncertainty_acc, and "
            "uncertainty_asr."
        ]
    )


def _branch_artifact_paths(
    *,
    prepared: PreparedBranchContext,
    source_label: str,
) -> tuple[Path, Path]:
    """Return branch-local disaggregation audit and metadata paths."""
    logs_dir = disaggregation_logs_dir(
        proj_base=prepared.disagg_proj_base,
        source_label=source_label,
    )
    audit_path = disaggregation_audit_path(
        logs_dir=logs_dir,
        mode=str(prepared.disagg_run_signature["l1_reg_aggreg"]),
        aggreg_indices=bool(prepared.disagg_run_signature["aggreg_indices"]),
    )
    metadata_path = disaggregation_metadata_path(
        proj_base=prepared.disagg_proj_base,
        source_label=source_label,
        mode=str(prepared.disagg_run_signature["l1_reg_aggreg"]),
        aggreg_indices=bool(prepared.disagg_run_signature["aggreg_indices"]),
    )
    ensure_file_parent(audit_path)
    ensure_file_parent(metadata_path)
    return audit_path, metadata_path


def _branch_mode_result(
    *,
    parsed: ParsedArgs,
    prepared: PreparedBranchContext,
    skipped: bool,
    output_paths: list[Path] | None = None,
):
    """Return the minimal branch payload needed by figure ownership helpers."""
    if output_paths is None:
        _audit_path, metadata_path = _branch_artifact_paths(
            prepared=prepared,
            source_label=parsed.disaggregation.new_disaggregated_version_name,
        )
        output_paths = [
            Path(str(path))
            for path in read_optional_json_dict(metadata_path).get("final_output_files", [])
        ]
    return SimpleNamespace(
        proj_base=prepared.disagg_proj_base,
        output_source_label=parsed.disaggregation.new_disaggregated_version_name,
        fu_code=parsed.base_allocate_args["fu_code"],
        requested_years=list(prepared.requested_years),
        lcia_methods=None,
        ssp_scenario_options_by_year=dict(prepared.ssp_scenario_options_by_year),
        run_signature=dict(prepared.disagg_run_signature),
        output_paths=[str(path) for path in output_paths],
        skipped=skipped,
    )


def _format_method_progress_message(
    *,
    method_index: int,
    total_methods: int,
    stem_prefix: str,
) -> str:
    """Return one transient status line for method scoped disaggregation progress."""
    return (
        f"[disaggregate_asocc] disaggregating method {method_index}/{total_methods} ({stem_prefix})"
    )


def run_disaggregation(
    parsed: ParsedArgs,
    *,
    phase: PhasePrinter,
) -> DisaggregationReport:
    """Execute published-output disaggregation across all requested branch modes."""
    run_plan = build_disaggregation_run_plan(parsed)
    grouped_sector_by_split = {
        spec.split_sector_label: spec.grouped_sector_label
        for spec in parsed.disaggregation.disaggregation_specs
    }
    branch_reports: list[DisaggregationBranchReport] = []
    initial_context = _disaggregated_context(
        parsed=parsed,
        selector=parsed.disaggregation.ref_split_run,
        mode=run_plan.l1_reg_aggreg,
        aggreg_indices=run_plan.aggreg_indices,
        l1_methods=run_plan.l1_methods,
        combined_methods=run_plan.combined_non_lcia,
        one_step_methods=run_plan.one_step_non_lcia,
    )
    refresh_root = disaggregation_source_root(
        proj_base=initial_context.disagg_proj_base,
        source_label=parsed.disaggregation.new_disaggregated_version_name,
    )
    if parsed.refresh:
        close_loggers_for_scope(refresh_root)
        shutil.rmtree(refresh_root, ignore_errors=True)
    aggreg_indices = run_plan.aggreg_indices
    mode = run_plan.l1_reg_aggreg
    branch_label = (
        f"l1_reg_aggreg={mode}, aggreg_indices={'grouped' if aggreg_indices else 'ungrouped'}"
    )
    prepared = _disaggregated_context(
        parsed=parsed,
        selector=parsed.disaggregation.ref_split_run,
        mode=mode,
        aggreg_indices=aggreg_indices,
        l1_methods=run_plan.l1_methods,
        combined_methods=run_plan.combined_non_lcia,
        one_step_methods=run_plan.one_step_non_lcia,
    )
    matched_runs = _match_runs(
        parsed=parsed,
        mode=mode,
        aggreg_indices=aggreg_indices,
        l1_methods=run_plan.l1_methods,
        combined_methods=run_plan.combined_non_lcia,
        one_step_methods=run_plan.one_step_non_lcia,
        requested_years=prepared.requested_years,
    )
    prepared = PreparedBranchContext(
        matched_runs=matched_runs,
        requested_years=prepared.requested_years,
        ssp_scenario_options_by_year=prepared.ssp_scenario_options_by_year,
        disagg_run_signature=prepared.disagg_run_signature,
        disagg_proj_base=prepared.disagg_proj_base,
        branch_complete=is_disaggregation_branch_complete(
            parsed=parsed,
            proj_base=prepared.disagg_proj_base,
            source_label=parsed.disaggregation.new_disaggregated_version_name,
            run_signature=prepared.disagg_run_signature,
            requested_years=prepared.requested_years,
            matched_runs=matched_runs,
        ),
    )
    if prepared.branch_complete:
        figure_result = sync_asocc_branch_figures(
            mode_result=_branch_mode_result(
                parsed=parsed,
                prepared=prepared,
                skipped=True,
            ),
            figures=parsed.figures,
            refresh=parsed.refresh,
            figure_external_method=parsed.figure_external_method,
            figure_options={"per_method": True, "multi_method": True},
            figure_output_format=parsed.figure_format["format"],
            figure_dpi=int(parsed.figure_format["dpi"]),
            status_source="disaggregate_asocc",
            status=phase,
        )
        audit_path, metadata_path = _branch_artifact_paths(
            prepared=prepared,
            source_label=parsed.disaggregation.new_disaggregated_version_name,
        )
        branch_reports.append(
            DisaggregationBranchReport(
                l1_reg_aggreg=mode,
                aggreg_indices=bool(aggreg_indices),
                summaries=[
                    "Run status: reused exactly.",
                    (f"Requested years: {format_summary_years(prepared.requested_years)}"),
                    output_files_available_line(1 + len(figure_result.figure_paths)),
                    f"Output folder: {metadata_path.parent.parent}",
                ],
                disaggregation_audit_path=audit_path,
                metadata_path=metadata_path,
                figure_paths=figure_result.figure_paths,
                run_status="reused_exact",
            )
        )
        return DisaggregationReport(
            source_label=parsed.disaggregation.new_disaggregated_version_name,
            branch_reports=branch_reports,
        )
    phase.log_message(f"[disaggregate_asocc] Starting branch: {branch_label}", persistent=True)
    phase.status("matching prerequisite scopes", owner="disaggregate_asocc")
    target_scope = _scope_for_matched_run(
        matched_runs["target_grouped_run"],
        context_label="Disaggregation target scope",
    )
    ref_grouped_scope = _scope_for_matched_run(
        matched_runs["ref_grouped_run"],
        context_label="Disaggregation reference-grouped scope",
    )
    ref_split_scope = _scope_for_matched_run(
        matched_runs["ref_split_run"],
        context_label="Disaggregation reference-split scope",
    )
    disagg_scope = path_scope_from_signature(
        proj_base=prepared.disagg_proj_base,
        source_label=parsed.disaggregation.new_disaggregated_version_name,
        run_signature=prepared.disagg_run_signature,
        context_label="Disaggregated output scope",
    )
    written_files: list[Path] = []
    audit_chunks: list[pd.DataFrame] = []
    method_stems = [
        *run_plan.one_step_non_lcia,
        *[f"{l1_method}_{l2_method}" for l2_method, l1_method in run_plan.combined_non_lcia],
    ]
    total_methods = len(method_stems)
    for method_index, l2_method in enumerate(run_plan.one_step_non_lcia, start=1):
        phase.show(
            _format_method_progress_message(
                method_index=method_index,
                total_methods=total_methods,
                stem_prefix=l2_method,
            )
        )
        written, audit = _run_family(
            target_scope=target_scope,
            ref_grouped_scope=ref_grouped_scope,
            ref_split_scope=ref_split_scope,
            disagg_scope=disagg_scope,
            grouped_sector_by_split=grouped_sector_by_split,
            bucket="l2_vs_global",
            stem_prefix=l2_method,
            requested_years=prepared.requested_years,
            output_format=parsed.output_format,
        )
        written_files.extend(written)
        audit_chunks.append(audit)
    combined_start_index = len(run_plan.one_step_non_lcia) + 1
    for method_index, (l2_method, l1_method) in enumerate(
        run_plan.combined_non_lcia,
        start=combined_start_index,
    ):
        stem_prefix = f"{l1_method}_{l2_method}"
        phase.show(
            _format_method_progress_message(
                method_index=method_index,
                total_methods=total_methods,
                stem_prefix=stem_prefix,
            )
        )
        written_final, audit_final = _run_family(
            target_scope=target_scope,
            ref_grouped_scope=ref_grouped_scope,
            ref_split_scope=ref_split_scope,
            disagg_scope=disagg_scope,
            grouped_sector_by_split=grouped_sector_by_split,
            bucket="l2_vs_global",
            stem_prefix=stem_prefix,
            requested_years=prepared.requested_years,
            output_format=parsed.output_format,
        )
        written_files.extend(written_final)
        audit_chunks.append(audit_final)
    audit_path, metadata_path = _branch_artifact_paths(
        prepared=prepared,
        source_label=parsed.disaggregation.new_disaggregated_version_name,
    )
    audit_frame = (
        pd.concat(audit_chunks, ignore_index=True)
        if audit_chunks
        else pd.DataFrame(columns=["bucket", "method_stem", "year", "value"])
    )
    audit_frame.to_csv(audit_path, index=False)
    time_route_warnings = _time_route_warning_lines(audit_frame=audit_frame)
    scope_key = write_scope_manifest(
        manifest_path=allocate_run_metadata_path(scope=disagg_scope),
        parsed=parsed,
        run_signature=prepared.disagg_run_signature,
        requested_years=prepared.requested_years,
        final_output_files=sorted({Path(path) for path in written_files}),
        ssp_scenario_options_by_year=prepared.ssp_scenario_options_by_year,
    )
    write_branch_metadata(
        parsed=parsed,
        proj_base=prepared.disagg_proj_base,
        run_signature=prepared.disagg_run_signature,
        requested_years=prepared.requested_years,
        matched_runs=matched_runs,
        final_output_files=sorted({Path(path) for path in written_files}),
        audit_path=audit_path,
        metadata_path=metadata_path,
        disaggregated_scope_key=scope_key,
    )
    phase.show(f"{branch_label}: finalizing metadata")
    figure_result = sync_asocc_branch_figures(
        mode_result=_branch_mode_result(
            parsed=parsed,
            prepared=prepared,
            skipped=False,
            output_paths=sorted({Path(path) for path in written_files}),
        ),
        figures=parsed.figures,
        refresh=parsed.refresh,
        figure_external_method=parsed.figure_external_method,
        figure_options={"per_method": True, "multi_method": True},
        figure_output_format=parsed.figure_format["format"],
        figure_dpi=int(parsed.figure_format["dpi"]),
        status_source="disaggregate_asocc",
        status=phase,
    )
    branch_reports.append(
        DisaggregationBranchReport(
            l1_reg_aggreg=mode,
            aggreg_indices=bool(aggreg_indices),
            summaries=[
                "Run status: computed.",
                (f"Requested years: {format_summary_years(prepared.requested_years)}"),
                *time_route_warnings,
                output_files_available_line(
                    len(set(written_files)) + 1 + len(figure_result.figure_paths)
                ),
                f"Output folder: {metadata_path.parent.parent}",
            ],
            disaggregation_audit_path=audit_path,
            metadata_path=metadata_path,
            figure_paths=figure_result.figure_paths,
            run_status="computed",
        )
    )
    return DisaggregationReport(
        source_label=parsed.disaggregation.new_disaggregated_version_name,
        branch_reports=branch_reports,
    )
