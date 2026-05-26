"""Internal ASR branch runtime."""

from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.shared.figures.persisted_outputs import delete_persisted_figure_paths
from pyaesa.shared.figures.title_contract import selector_scope_request_from_selector_values
from pyaesa.shared.runtime.reuse.branch_reuse import (
    cleanup_branch_outputs_for_refresh,
)
from pyaesa.shared.runtime.reuse.derived_state import request_state_matches
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_A_LCA,
    PHASE_C_ASR,
    phase_ready_detail,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.asr.deterministic.state.branch_state import (
    cached_branch_state,
    written_branch_state,
)
from pyaesa.asr.deterministic.state.metadata import (
    cached_int,
    cached_manifest_value,
    cached_path_list,
    load_figure_metadata,
    load_run_metadata,
    save_figure_metadata,
    save_run_metadata,
    set_figure_state,
)
from pyaesa.shared.runtime.manifest_contract import manifest_digest, path_list
from pyaesa.shared.acc_asr_common.deterministic.state.scope_guard import (
    branch_coverage,
    branch_identity_payload,
    branch_reuse_mode_or_raise,
    coverage_signature_covers,
    ensure_recorded_output_files_exist,
    ensure_same_branch_identity_or_raise,
    merged_coverage,
)
from pyaesa.shared.acc_asr_common.deterministic.downstream.shares import (
    build_downstream_asocc_share_context,
)
from pyaesa.shared.selectors.time_selectors import normalize_requested_years
from pyaesa.external_inputs.lca.figures import (
    render_external_lca_deterministic_figures_from_rows,
)
from pyaesa.external_inputs.lca.paths import external_lca_root

from ..deterministic.figures.component_diagnostics import (
    component_rows_from_runtime_frame,
    write_component_rows_artifact,
)
from ..deterministic.figures.render import render_asr_figures
from ..deterministic.runtime.dynamic import process_dynamic_asr
from ..deterministic.runtime.lca_rows import lca_public_output_root, load_lca_rows
from ..deterministic.runtime.static import process_static_asr
from ..shared.runtime.paths import (
    build_asr_path_context,
    build_asr_scope_label,
    get_asr_dynamic_component_rows_path,
    get_asr_meta_path,
    get_asr_figure_metadata_path,
)
from ..deterministic.state.reports import ASRBranchReport, build_asr_branch_report

_FIGURE_STATE_KEY = "figure_state"


def _build_run_metadata_payload(
    *,
    arguments: dict[str, Any],
    identity_payload: dict[str, Any],
    coverage: dict[str, list[Any]],
    scope_label: str,
    fu_code: str,
    cc_source: str,
    cc_type: str,
    cc_bounds: list[str],
    lca_type: str,
    lca_version_name: str | None,
    n_acc_files_matched: int,
    n_asr_files_written: int,
    impacts: list[str],
    requested_years: list[int],
    share_transition_meta: dict[str, dict[str, object]],
    external_lca_transition: dict[str, Any] | None,
    output_dirs: list[Path],
    output_files: list[Path],
    figure_paths: list[Path],
    dynamic_component_rows_path: Path | None,
    external_lca_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical deterministic ASR scope-manifest payload."""
    return {
        "function": "deterministic_asr",
        "arguments": arguments,
        "execution": {
            "status": "complete",
            "n_acc_files_matched": int(n_acc_files_matched),
            "n_asr_files_written": int(n_asr_files_written),
        },
        "reuse": {
            "identity_key": manifest_digest(identity_payload),
            "coverage": coverage,
        },
        "artifacts": {
            "output_dirs": path_list(output_dirs),
            "output_files": path_list(output_files),
            "figure_paths": path_list(figure_paths),
            "dynamic_component_rows": (
                path_list([dynamic_component_rows_path])
                if dynamic_component_rows_path is not None
                else []
            ),
        },
        "provenance": {
            "scope_label": scope_label,
            "fu_code": fu_code,
            "cc_source": cc_source,
            "cc_type": cc_type,
            "cc_bounds": list(cc_bounds),
            "lca_type": lca_type,
            "lca_version_name": lca_version_name,
            "impacts": list(impacts),
            "requested_years": list(requested_years),
            "share_transition_meta": share_transition_meta,
            "external_lca_transition": external_lca_transition,
            "external_lca_summary": external_lca_summary,
        },
    }


def _external_lca_summary_payload(
    *,
    proj_base: Path,
    lca_type: str,
    lca_version_name: str | None,
    lcia_method: str,
    figure_paths: list[Path],
) -> dict[str, Any] | None:
    """Return deterministic external LCA summary payload when relevant."""
    if lca_type != "external":
        return None
    return {
        "source_type": "deterministic",
        "version_name": lca_version_name,
        "lcia_method": lcia_method,
        "output_root": str(external_lca_root(project_base=proj_base)),
        "figures_available": len(figure_paths) if figure_paths else None,
        "figure_paths": path_list(figure_paths),
    }


def _external_lca_figure_paths_from_summary(
    *,
    metadata: dict[str, Any],
) -> list[Path]:
    """Return persisted deterministic external LCA subfigure paths."""
    summary = cast(dict[str, Any], metadata["provenance"].get("external_lca_summary") or {})
    return [Path(str(path)) for path in summary.get("figure_paths", [])]


def _external_lca_subfigures_complete(
    *,
    metadata: dict[str, Any],
    lca_type: str,
    subfigures: bool,
) -> bool:
    """Return whether requested external LCA subfigures are already complete."""
    if lca_type != "external" or not subfigures:
        return True
    paths = _external_lca_figure_paths_from_summary(metadata=metadata)
    return bool(paths) and all(path.exists() for path in paths)


def _render_missing_external_lca_subfigures(
    *,
    metadata: dict[str, Any],
    proj_base: Path,
    source_label: str,
    lca_type: str,
    lca_version_name: str | None,
    lcia_method: str,
    base_allocate_args: dict[str, Any],
    years: list[int],
    figure_output_format: str,
    figure_dpi: int,
    status: PhasePrinter | NullPhasePrinter,
) -> dict[str, object] | None:
    """Render missing deterministic external LCA subfigures and return summary payload."""
    status.announce(PHASE_A_LCA, "external_lca")
    lca_rows = load_lca_rows(
        proj_base=proj_base,
        source_label=source_label,
        lca_type=lca_type,
        lcia_method=lcia_method,
        lca_version_name=lca_version_name,
        base_allocate_args=base_allocate_args,
        years=years,
    )
    figure_paths = _render_external_lca_subfigures(
        lca_rows=lca_rows,
        proj_base=proj_base,
        lca_version_name=lca_version_name,
        lcia_method=lcia_method,
        figure_output_format=figure_output_format,
        figure_dpi=figure_dpi,
        status=status,
    )
    _complete_lca_phase(
        status=status,
        proj_base=proj_base,
        source_label=source_label,
        lca_type=lca_type,
        base_allocate_args=base_allocate_args,
        owner="external_lca",
    )
    return _external_lca_summary_payload(
        proj_base=proj_base,
        lca_type=lca_type,
        lca_version_name=lca_version_name,
        lcia_method=lcia_method,
        figure_paths=figure_paths,
    )


def resolve_acc_l1_l2_methods(
    *,
    proj_base: Path,
    source_label: str,
    base_allocate_args: dict[str, Any],
    fu_code: str,
    external_method: dict[str, Any] | None,
    cc_source: str,
    years: list[int],
    ssp_scenario: list[str] | None,
) -> tuple[set[str], dict[str, dict[str, object]]]:
    """Return aCC denominator allocation identities and transition metadata."""
    asocc_share_context = build_downstream_asocc_share_context(
        proj_base=proj_base,
        source_label=source_label,
        base_allocate_args=base_allocate_args,
        fu_code=fu_code,
        external_method=external_method,
        years=years,
        lcia_method=cc_source,
        output_source_label=source_label,
        branch_ssp_scenario=ssp_scenario,
        switch_label="aSoCC SSP-dependent switch",
    )
    return (
        asocc_share_context.allowed_l1_l2_methods,
        asocc_share_context.share_transition_meta,
    )


def run_single_asr(
    *,
    proj_base: Path,
    build_branch_signature,
    build_figure_signature,
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
    lca_type: str,
    lca_version_name: str | None,
    acc_output_files: list[Path],
    output_format: str,
    figures: bool,
    figure_options: dict[str, Any],
    figure_output_format: str,
    figure_dpi: int,
    subfigures: bool,
    refresh: bool,
    status: PhasePrinter | NullPhasePrinter,
) -> ASRBranchReport:
    """Execute one ASR branch."""
    normalized_years = normalize_requested_years(years)
    selector_scope_request = selector_scope_request_from_selector_values(
        selector_values=base_allocate_args,
    )
    allowed_l1_l2_methods, share_transition_meta = resolve_acc_l1_l2_methods(
        proj_base=proj_base,
        source_label=source_label,
        base_allocate_args=base_allocate_args,
        fu_code=fu_code,
        external_method=external_method,
        cc_source=cc_source,
        years=normalized_years,
        ssp_scenario=ssp_scenario,
    )
    path_context = build_asr_path_context(
        proj_base=proj_base,
        source_label=source_label,
        agg_version=base_allocate_args["agg_version"],
        fu_code=fu_code,
        lca_type=lca_type,
        cc_source=cc_source,
        cc_type=cc_type,
        lca_version_name=lca_version_name,
    )
    scope_label = build_asr_scope_label(
        source_label=source_label,
        agg_version=base_allocate_args["agg_version"],
        lca_type=lca_type,
        cc_source=cc_source,
        cc_type=cc_type,
        lca_version_name=lca_version_name,
    )
    meta_path = get_asr_meta_path(context=path_context)
    identity_payload = branch_identity_payload(
        public_request_payload=public_request_payload,
        cc_type=cc_type,
    )
    requested_coverage = branch_coverage(
        cc_type=cc_type,
        requested_years=normalized_years,
        static_cc_bounds=static_cc_bounds,
        category=category,
        ssp_scenario=ssp_scenario,
    )
    identity = build_branch_signature(public_request_payload=identity_payload)
    existing = load_run_metadata(meta_path)
    if refresh:
        ensure_same_branch_identity_or_raise(
            existing_metadata=existing,
            requested_identity=identity,
            scope_label=scope_label,
            function_name="deterministic_asr",
        )
        cleanup_branch_outputs_for_refresh(
            existing_metadata=existing,
            meta_path=meta_path,
            artifact_keys=("output_files", "figure_paths", "dynamic_component_rows"),
            scope_targets=(get_asr_figure_metadata_path(context=path_context),),
        )
        existing = None
    reuse_mode = branch_reuse_mode_or_raise(
        existing_metadata=existing,
        requested_identity=identity,
        requested_coverage=requested_coverage,
        scope_label=scope_label,
        function_name="deterministic_asr",
    )
    if reuse_mode in {"reuse", "append"}:
        ensure_recorded_output_files_exist(
            existing_metadata=cast(dict[str, Any], existing),
            scope_label=scope_label,
            function_name="deterministic_asr",
        )
    figure_signature = build_figure_signature(
        figure_format={
            "format": figure_output_format,
            "dpi": figure_dpi,
        },
        figure_options=figure_options,
    )
    figure_compute_signature = {
        "identity_key": manifest_digest(identity),
        "coverage": requested_coverage,
    }
    figure_metadata = load_figure_metadata(meta_path)
    matched_metadata = existing if reuse_mode == "reuse" and not refresh else None
    external_subfigures_complete = (
        matched_metadata is None
        or refresh
        or _external_lca_subfigures_complete(
            metadata=matched_metadata,
            lca_type=lca_type,
            subfigures=subfigures,
        )
    )
    if (
        matched_metadata is not None
        and not refresh
        and not figures
        and external_subfigures_complete
    ):
        return build_asr_branch_report(
            state=cached_branch_state(
                existing_metadata=matched_metadata,
                figure_paths=cached_path_list(
                    existing_metadata=matched_metadata,
                    field_name="figure_paths",
                ),
                meta_path=meta_path,
            ),
            lca_type=lca_type,
            n_acc_files_matched=cached_int(
                existing_metadata=matched_metadata,
                field_name="n_acc_files_matched",
            ),
            n_asr_files_written=cached_int(
                existing_metadata=matched_metadata,
                field_name="n_asr_files_written",
            ),
            external_lca_summary=cast(
                dict[str, Any] | None,
                matched_metadata["provenance"].get("external_lca_summary"),
            ),
            reuse_status="reused_exact",
        )
    if (
        matched_metadata is not None
        and not refresh
        and figures
        and external_subfigures_complete
        and request_state_matches(
            payload=figure_metadata,
            state_key=_FIGURE_STATE_KEY,
            request_signature=figure_signature,
            compute_signature=figure_compute_signature,
            compute_compatible=coverage_signature_covers,
        )
    ):
        return build_asr_branch_report(
            state=cached_branch_state(
                existing_metadata=matched_metadata,
                figure_paths=cached_path_list(
                    existing_metadata=matched_metadata,
                    field_name="figure_paths",
                ),
                meta_path=meta_path,
            ),
            lca_type=lca_type,
            n_acc_files_matched=cached_int(
                existing_metadata=matched_metadata,
                field_name="n_acc_files_matched",
            ),
            n_asr_files_written=cached_int(
                existing_metadata=matched_metadata,
                field_name="n_asr_files_written",
            ),
            external_lca_summary=cast(
                dict[str, Any] | None,
                matched_metadata["provenance"].get("external_lca_summary"),
            ),
            reuse_status="reused_exact",
        )
    if (
        matched_metadata is not None
        and not refresh
        and subfigures
        and lca_type == "external"
        and not external_subfigures_complete
        and (
            not figures
            or request_state_matches(
                payload=figure_metadata,
                state_key=_FIGURE_STATE_KEY,
                request_signature=figure_signature,
                compute_signature=figure_compute_signature,
                compute_compatible=coverage_signature_covers,
            )
        )
    ):
        external_lca_summary = _render_missing_external_lca_subfigures(
            metadata=matched_metadata,
            proj_base=proj_base,
            source_label=source_label,
            lca_type=lca_type,
            lca_version_name=lca_version_name,
            lcia_method=cc_source,
            base_allocate_args=base_allocate_args,
            years=normalized_years,
            figure_output_format=figure_output_format,
            figure_dpi=figure_dpi,
            status=status,
        )
        matched_metadata["provenance"]["external_lca_summary"] = external_lca_summary
        save_run_metadata(meta_path, matched_metadata)
        return build_asr_branch_report(
            state=cached_branch_state(
                existing_metadata=matched_metadata,
                figure_paths=cached_path_list(
                    existing_metadata=matched_metadata,
                    field_name="figure_paths",
                ),
                meta_path=meta_path,
            ),
            lca_type=lca_type,
            n_acc_files_matched=cached_int(
                existing_metadata=matched_metadata,
                field_name="n_acc_files_matched",
            ),
            n_asr_files_written=cached_int(
                existing_metadata=matched_metadata,
                field_name="n_asr_files_written",
            ),
            external_lca_summary=external_lca_summary,
            reuse_status="partially_reused",
        )
    if matched_metadata is not None and not refresh:
        delete_persisted_figure_paths(
            raw_paths=figure_metadata.get(_FIGURE_STATE_KEY, {}).get("paths")
        )
        cached_component_rows = cached_path_list(
            existing_metadata=matched_metadata,
            field_name="dynamic_component_rows",
        )
        figure_paths = render_asr_figures(
            path_context=path_context,
            fu_code=fu_code,
            cc_source=cc_source,
            cc_type=cc_type,
            requested_years=normalized_years,
            share_transition_meta=cast(
                dict[str, dict[str, object]],
                cached_manifest_value(
                    existing_metadata=matched_metadata,
                    field_name="share_transition_meta",
                ),
            ),
            emissions_mode=emissions_mode,
            dpi=figure_dpi,
            output_format=figure_output_format,
            selector_scope_request=selector_scope_request,
            figure_options=figure_options,
            output_paths=cached_path_list(
                existing_metadata=matched_metadata,
                field_name="output_files",
            ),
            acc_output_files=acc_output_files,
            component_rows_path=cached_component_rows[0] if cached_component_rows else None,
            coverage=requested_coverage,
            status=status,
        )
        external_lca_summary = None
        if subfigures and lca_type == "external" and not external_subfigures_complete:
            external_lca_summary = _render_missing_external_lca_subfigures(
                metadata=matched_metadata,
                proj_base=proj_base,
                source_label=source_label,
                lca_type=lca_type,
                lca_version_name=lca_version_name,
                lcia_method=cc_source,
                base_allocate_args=base_allocate_args,
                years=normalized_years,
                figure_output_format=figure_output_format,
                figure_dpi=figure_dpi,
                status=status,
            )
        set_figure_state(
            payload=figure_metadata,
            signature=figure_signature,
            compute_signature=figure_compute_signature,
            paths=figure_paths,
        )
        matched_metadata["artifacts"]["figure_paths"] = path_list(figure_paths)
        if external_lca_summary is not None:
            matched_metadata["provenance"]["external_lca_summary"] = external_lca_summary
        save_figure_metadata(meta_path, figure_metadata)
        save_run_metadata(meta_path, matched_metadata)
        return build_asr_branch_report(
            state=cached_branch_state(
                existing_metadata=matched_metadata,
                figure_paths=figure_paths,
                meta_path=meta_path,
            ),
            lca_type=lca_type,
            n_acc_files_matched=cached_int(
                existing_metadata=matched_metadata,
                field_name="n_acc_files_matched",
            ),
            n_asr_files_written=cached_int(
                existing_metadata=matched_metadata,
                field_name="n_asr_files_written",
            ),
            external_lca_summary=cast(
                dict[str, Any] | None,
                matched_metadata["provenance"].get("external_lca_summary"),
            ),
            reuse_status="partially_reused",
        )
    lca_rows = _load_lca_rows_for_branch(
        proj_base=proj_base,
        source_label=source_label,
        lca_type=lca_type,
        lcia_method=cc_source,
        lca_version_name=lca_version_name,
        base_allocate_args=base_allocate_args,
        years=normalized_years,
        status=status,
        complete_phase=not (subfigures and lca_type == "external"),
    )
    external_lca_figure_paths: list[Path] = []
    if cc_type == "static":
        process_result = process_static_asr(
            proj_base=proj_base,
            fu_code=fu_code,
            source_label=source_label,
            base_allocate_args=base_allocate_args,
            years=normalized_years,
            cc_source=cc_source,
            static_cc_bounds=static_cc_bounds,
            lca_type=lca_type,
            fmt=output_format,
            lca_version_name=lca_version_name,
            acc_output_files=acc_output_files,
            allowed_l1_l2_methods=allowed_l1_l2_methods,
            lca_rows=lca_rows,
            status=status,
            return_lca_rows=lca_type == "external",
        )
    else:
        process_result = process_dynamic_asr(
            proj_base=proj_base,
            fu_code=fu_code,
            source_label=source_label,
            base_allocate_args=base_allocate_args,
            cc_source=cc_source,
            years=normalized_years,
            lca_type=lca_type,
            fmt=output_format,
            lca_version_name=lca_version_name,
            acc_output_files=acc_output_files,
            allowed_l1_l2_methods=allowed_l1_l2_methods,
            share_transition_meta=share_transition_meta,
            lca_rows=lca_rows,
            status=status,
            return_lca_rows=figures or lca_type == "external",
        )
    dynamic_component_rows_path = None
    if cc_type != "static":
        component_rows_path = get_asr_dynamic_component_rows_path(
            context=path_context,
            fmt=output_format,
        )
        write_component_rows_artifact(
            path=component_rows_path,
            rows=component_rows_from_runtime_frame(
                component_frame=cast(pd.DataFrame, process_result.dynamic_component_frame),
                lca_rows=lca_rows,
                acc_output_files=acc_output_files,
            ),
        )
        dynamic_component_rows_path = component_rows_path
    figure_paths = (
        render_asr_figures(
            path_context=path_context,
            fu_code=fu_code,
            cc_source=cc_source,
            cc_type=cc_type,
            requested_years=normalized_years,
            share_transition_meta=share_transition_meta,
            emissions_mode=emissions_mode,
            dpi=figure_dpi,
            output_format=figure_output_format,
            selector_scope_request=selector_scope_request,
            figure_options=figure_options,
            status=status,
            output_paths=process_result.output_files,
            acc_output_files=acc_output_files,
            component_rows_path=dynamic_component_rows_path,
            coverage=requested_coverage,
        )
        if figures
        else []
    )
    if subfigures and lca_type == "external":
        status.announce(PHASE_A_LCA, "external_lca")
        external_lca_figure_paths = _render_external_lca_subfigures(
            lca_rows=lca_rows,
            proj_base=proj_base,
            lca_version_name=lca_version_name,
            lcia_method=cc_source,
            figure_output_format=figure_output_format,
            figure_dpi=figure_dpi,
            status=status,
        )
        _complete_lca_phase(
            status=status,
            proj_base=proj_base,
            source_label=source_label,
            lca_type=lca_type,
            base_allocate_args=base_allocate_args,
            owner="external_lca",
        )
        status.announce(PHASE_C_ASR, "deterministic_asr")
    payload = _build_run_metadata_payload(
        arguments=public_request_payload,
        identity_payload=identity,
        coverage=merged_coverage(
            existing_metadata=None if refresh else existing,
            requested_coverage=requested_coverage,
        ),
        scope_label=scope_label,
        fu_code=fu_code,
        cc_source=cc_source,
        cc_type=cc_type,
        cc_bounds=static_cc_bounds,
        lca_type=lca_type,
        lca_version_name=lca_version_name,
        n_acc_files_matched=process_result.n_matched,
        n_asr_files_written=process_result.n_written,
        impacts=process_result.impacts,
        requested_years=[int(year) for year in normalized_years],
        share_transition_meta=share_transition_meta,
        external_lca_transition=process_result.external_lca_transition,
        output_dirs=process_result.output_dirs,
        output_files=process_result.output_files,
        figure_paths=figure_paths,
        dynamic_component_rows_path=dynamic_component_rows_path,
        external_lca_summary=_external_lca_summary_payload(
            proj_base=proj_base,
            lca_type=lca_type,
            lca_version_name=lca_version_name,
            lcia_method=cc_source,
            figure_paths=external_lca_figure_paths,
        ),
    )
    if figures:
        figure_metadata = {}
        set_figure_state(
            payload=figure_metadata,
            signature=figure_signature,
            compute_signature=figure_compute_signature,
            paths=figure_paths,
        )
        save_figure_metadata(meta_path, figure_metadata)
    save_run_metadata(meta_path, payload)
    return build_asr_branch_report(
        state=written_branch_state(
            cc_source=cc_source,
            cc_type=cc_type,
            cc_bounds=static_cc_bounds,
            impacts_used=process_result.impacts,
            figure_paths=figure_paths,
            output_dirs=process_result.output_dirs,
            meta_path=meta_path,
        ),
        lca_type=lca_type,
        n_acc_files_matched=process_result.n_matched,
        n_asr_files_written=process_result.n_written,
        external_lca_summary=_external_lca_summary_payload(
            proj_base=proj_base,
            lca_type=lca_type,
            lca_version_name=lca_version_name,
            lcia_method=cc_source,
            figure_paths=external_lca_figure_paths,
        ),
    )


def _render_external_lca_subfigures(
    *,
    lca_rows: pd.DataFrame,
    proj_base: Path,
    lca_version_name: str | None,
    lcia_method: str,
    figure_output_format: str,
    figure_dpi: int,
    status: StatusSink | None = None,
) -> list[Path]:
    return render_external_lca_deterministic_figures_from_rows(
        proj_base=proj_base,
        version_name=cast(str, lca_version_name),
        lcia_method=lcia_method,
        rows=lca_rows,
        output_format=figure_output_format,
        dpi=figure_dpi,
        status=status,
    )


def _load_lca_rows_for_branch(
    *,
    proj_base: Path,
    source_label: str,
    lca_type: str,
    lcia_method: str,
    lca_version_name: str | None,
    base_allocate_args: dict[str, Any],
    years: list[int],
    status: PhasePrinter | NullPhasePrinter,
    complete_phase: bool,
) -> pd.DataFrame:
    """Load one ASR numerator source and close the public LCA phase."""
    owner = "deterministic_io_lca" if lca_type == "io_lca" else "external_lca"
    status.show(f"[{owner}] Loading {lca_type} LCA numerator rows")
    rows = load_lca_rows(
        proj_base=proj_base,
        source_label=source_label,
        lca_type=lca_type,
        lcia_method=lcia_method,
        lca_version_name=lca_version_name,
        base_allocate_args=base_allocate_args,
        years=years,
    )
    if complete_phase:
        _complete_lca_phase(
            status=status,
            proj_base=proj_base,
            source_label=source_label,
            lca_type=lca_type,
            base_allocate_args=base_allocate_args,
            owner=owner,
        )
    return rows


def _complete_lca_phase(
    *,
    status: PhasePrinter | NullPhasePrinter,
    proj_base: Path,
    source_label: str,
    lca_type: str,
    base_allocate_args: dict[str, Any],
    owner: str,
) -> None:
    """Print the deterministic ASR LCA phase completion once."""
    status.complete(
        phase_ready_detail(
            scope_name="LCA",
            output_root=lca_public_output_root(
                proj_base=proj_base,
                source_label=source_label,
                lca_type=lca_type,
                base_allocate_args=base_allocate_args,
            ),
        ),
        owner=owner,
    )
