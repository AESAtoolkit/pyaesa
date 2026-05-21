"""Internal aCC branch runtime."""

from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.shared.runtime.reuse.branch_reuse import (
    cleanup_branch_outputs_for_refresh,
)
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.acc.shared.runtime.paths import public_result_root_name_for_fu_code
from pyaesa.shared.acc_asr_common.deterministic.state.scope_guard import (
    branch_coverage,
    branch_identity_payload,
    branch_reuse_mode_or_raise,
    ensure_recorded_output_files_exist,
    ensure_same_branch_identity_or_raise,
    merged_coverage,
)
from pyaesa.shared.acc_asr_common.deterministic.downstream.shares import (
    build_downstream_asocc_share_context,
)
from pyaesa.shared.selectors.time_selectors import normalize_requested_years

from .state.metadata import (
    build_run_metadata_payload,
    load_run_metadata,
    save_run_metadata,
)
from .figures.render import render_acc_deterministic_figures
from pyaesa.acc.deterministic.runtime.paths import (
    build_acc_path_context,
    build_acc_scope_label,
    get_acc_figure_metadata_path,
    get_acc_meta_path,
)
from .runtime.dynamic import dynamic_cc_coverage, process_dynamic_acc, resolve_dynamic_cc_input
from .runtime.static import process_static_acc
from .state.reports import ACCBranchReport


def run_single_acc(
    *,
    proj_base: Path,
    build_branch_signature,
    public_request_payload: dict[str, Any],
    source_label: str,
    base_allocate_args: dict[str, Any],
    fu_code: str,
    external_method: dict[str, Any] | None,
    cc_source: str,
    cc_type: str,
    years: int | list[int] | range,
    static_cc_bounds: list[str],
    harmonization: bool,
    harmonization_method: str,
    category: list[str] | None,
    ssp_scenario: list[str] | None,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str,
    subset_version: str | None,
    output_format: str,
    figures: bool,
    figure_options: dict[str, bool],
    figure_output_format: str,
    figure_dpi: int,
    refresh: bool,
    status: StatusSink,
) -> ACCBranchReport:
    """Execute one aCC branch."""
    public_result_root_name = public_result_root_name_for_fu_code(fu_code=fu_code)
    requested_years = normalize_requested_years(base_allocate_args["years"])
    path_context = build_acc_path_context(
        proj_base=proj_base,
        source_label=source_label,
        group_version=base_allocate_args["group_version"],
        cc_source=cc_source,
        cc_type=cc_type,
        public_result_root_name=public_result_root_name,
    )
    scope_label = build_acc_scope_label(
        source_label=source_label,
        group_version=base_allocate_args["group_version"],
        cc_source=cc_source,
        cc_type=cc_type,
    )
    meta_path = get_acc_meta_path(context=path_context)
    identity = build_branch_signature(
        public_request_payload=branch_identity_payload(
            public_request_payload=public_request_payload,
            cc_type=path_context.cc_type,
        )
    )
    existing = load_run_metadata(meta_path)
    resolved_dynamic_cc_path = None
    resolved_dynamic_cc_table = None
    if cc_type == "dynamic_ar6":
        resolved_dynamic_cc_path, resolved_dynamic_cc_table = resolve_dynamic_cc_input(
            years=years,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            category=category,
            ssp_scenario=ssp_scenario,
            emission_type=emission_type,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode,
            subset_version=subset_version,
            fmt=output_format,
        )
        requested_coverage = cast(
            dict[str, list[Any]],
            dynamic_cc_coverage(cc_table=resolved_dynamic_cc_table),
        )
        requested_coverage["years"] = [int(year) for year in requested_years]
    else:
        requested_coverage = branch_coverage(
            cc_type="static",
            requested_years=requested_years,
            static_cc_bounds=static_cc_bounds,
            category=None,
            ssp_scenario=None,
        )
    if refresh:
        ensure_same_branch_identity_or_raise(
            existing_metadata=existing,
            requested_identity=identity,
            scope_label=scope_label,
            function_name="deterministic_acc",
        )
        cleanup_branch_outputs_for_refresh(
            existing_metadata=existing,
            meta_path=meta_path,
            artifact_keys=("output_files", "figure_paths"),
            scope_targets=(get_acc_figure_metadata_path(context=path_context),),
        )
        existing = None
    reuse_mode = branch_reuse_mode_or_raise(
        existing_metadata=existing,
        requested_identity=identity,
        requested_coverage=requested_coverage,
        scope_label=scope_label,
        function_name="deterministic_acc",
    )
    if reuse_mode in {"reuse", "append"}:
        ensure_recorded_output_files_exist(
            existing_metadata=cast(dict[str, Any], existing),
            scope_label=scope_label,
            function_name="deterministic_acc",
        )
    if reuse_mode == "reuse" and not refresh:
        metadata = load_run_metadata(meta_path)
        artifacts = metadata["artifacts"]
        provenance = metadata["provenance"]
        execution = metadata["execution"]
        figure_paths: list[Path] = []
        reuse_status = "reused_exact"
        if figures:
            figure_paths, figure_reused = render_acc_deterministic_figures(
                metadata_path=meta_path,
                dpi=figure_dpi,
                output_format=figure_output_format,
                figure_options=figure_options,
                coverage=requested_coverage,
                status=status,
            )
            if not figure_reused:
                reuse_status = "partially_reused"
        return ACCBranchReport(
            cc_source=str(provenance["cc_source"]),
            cc_type=str(provenance["cc_type"]),
            cc_bounds=[str(value) for value in provenance["cc_bounds"]],
            impacts_used=[str(value) for value in provenance["impacts"]],
            output_dirs=[Path(str(path)) for path in artifacts["output_dirs"]],
            meta_file=meta_path,
            n_share_files_processed=int(execution["n_share_files_processed"]),
            n_acc_files_written=int(execution["n_acc_files_written"]),
            reuse_status=reuse_status,
            figure_paths=figure_paths,
        )
    asocc_share_context = build_downstream_asocc_share_context(
        proj_base=proj_base,
        source_label=source_label,
        base_allocate_args=base_allocate_args,
        fu_code=fu_code,
        external_method=external_method,
        years=requested_years,
        lcia_method=cc_source,
        output_source_label=source_label,
        branch_ssp_scenario=ssp_scenario,
    )
    share_transition_meta = asocc_share_context.share_transition_meta
    if cc_type == "static":
        result = process_static_acc(
            path_context=path_context,
            cc_source=cc_source,
            years=requested_years,
            asocc_shares=asocc_share_context.asocc_shares,
            fmt=output_format,
            static_cc_bounds=static_cc_bounds,
            public_result_root_name=public_result_root_name,
            status=status,
        )
    else:
        result = process_dynamic_acc(
            path_context=path_context,
            public_result_root_name=public_result_root_name,
            cc_source=cc_source,
            asocc_shares=asocc_share_context.asocc_shares,
            fmt=output_format,
            lcia_method=cc_source,
            years=years,
            emission_type=emission_type,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode,
            share_transition_meta=share_transition_meta,
            status=status,
            resolved_cc_path=cast(Path, resolved_dynamic_cc_path),
            resolved_cc_table=cast(pd.DataFrame, resolved_dynamic_cc_table),
        )
    n_share, n_written, impacts, output_dirs, output_files, cc_input_path = result
    payload = build_run_metadata_payload(
        arguments=public_request_payload,
        identity_payload=identity,
        coverage=merged_coverage(
            existing_metadata=None if refresh else existing,
            requested_coverage=requested_coverage,
        ),
        scope_label=scope_label,
        cc_source=cc_source,
        cc_type=cc_type,
        cc_bounds=static_cc_bounds,
        n_share_files_processed=n_share,
        n_acc_files_written=n_written,
        impacts=impacts,
        requested_years=requested_years,
        share_transition_meta=share_transition_meta,
        output_dirs=output_dirs,
        output_files=output_files,
        cc_input_path=cc_input_path,
    )
    save_run_metadata(meta_path, payload)
    figure_paths = (
        render_acc_deterministic_figures(
            metadata_path=meta_path,
            dpi=figure_dpi,
            output_format=figure_output_format,
            figure_options=figure_options,
            coverage=requested_coverage,
            status=status,
        )[0]
        if figures
        else []
    )
    return ACCBranchReport(
        cc_source=cc_source,
        cc_type=cc_type,
        cc_bounds=static_cc_bounds,
        impacts_used=impacts,
        output_dirs=output_dirs,
        meta_file=meta_path,
        n_share_files_processed=n_share,
        n_acc_files_written=n_written,
        reuse_status="computed" if reuse_mode == "create" else "partially_reused",
        figure_paths=figure_paths,
    )
