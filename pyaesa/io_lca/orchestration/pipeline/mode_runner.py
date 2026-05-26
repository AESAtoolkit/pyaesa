"""Per-mode deterministic IO-LCA execution runner."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from pyaesa.asocc.io.contracts import mode_label
from pyaesa.shared.runtime.reuse.contracts import (
    normalize_selector_payload,
    normalize_signature_selectors,
)

from ...data.metadata import (
    ensure_scope,
    get_scope,
    get_lcia_method_years,
    iter_scope_entries,
    load_scope_manifest,
    merge_written_paths,
    require_scope_signature,
    save_scope_manifest,
    scope_complete_and_existing,
    set_lcia_method_years,
    set_scope_complete,
)
from ...data.paths import (
    IOLCAPaths,
    deterministic_scope_metadata_paths,
    figures_dir_for_source,
    io_metadata_path_for_source,
    lca_results_dir_for_source,
    origin_dir_for_source,
    resolve_io_lca_paths,
    stages_dir_for_source,
)
from ...data.writers import clear_scope_outputs
from pyaesa.io_lca.orchestration.request.domain_checks import (
    validate_group_indices_requires_multi_selection,
    validate_group_indices_supported,
)
from pyaesa.io_lca.orchestration.pipeline.method_runner import run_io_lca_method
from pyaesa.io_lca.orchestration.pipeline.progress import io_lca_banner, year_progress
from pyaesa.io_lca.orchestration.pipeline.run_signatures import build_io_lca_signature
from pyaesa.shared.runtime.reporting.progress import StatusProgressPrinter
from pyaesa.shared.runtime.reporting.status import StatusSink


def _read_signature_group_indices(signature: dict[str, Any]) -> bool:
    """Return persisted IO-LCAn aggregation identity from deterministic metadata."""
    return bool(signature["group_indices"])


def _ensure_no_conflicting_group_indices_project(
    *,
    paths: IOLCAPaths,
    current_metadata_path: Path,
    log_payload: dict[str, Any],
    group_indices: bool,
) -> None:
    """Raise when one IO-LCA project tree already records the opposite aggregation identity."""

    def _scan_payload(payload: dict[str, Any]) -> None:
        for raw_scope in iter_scope_entries(payload=payload):
            signature = require_scope_signature(scope=raw_scope)
            persisted_group_indices = _read_signature_group_indices(signature)
            if persisted_group_indices == group_indices:
                continue
            raise ValueError(
                "deterministic_io_lca requires a different project_name when "
                "group_indices changes between True and False. Grouped selector "
                "outputs and non aggregated selector outputs must not coexist in one "
                "project tree."
            )

    _scan_payload(log_payload)
    for metadata_path in deterministic_scope_metadata_paths(paths=paths):
        if metadata_path == current_metadata_path:
            continue
        candidate_payload = load_scope_manifest(
            path=metadata_path,
            function_name="deterministic_io_lca",
        )
        _scan_payload(candidate_payload)


def _ensure_no_conflicting_flat_output_scope(
    *,
    log_payload: dict[str, Any],
    fu_code: str,
    filters: dict[str, list[str] | None],
) -> None:
    """Raise when one project scope already records a different flat table identity."""
    requested_selectors = normalize_selector_payload(
        filters,
        context="deterministic_io_lca filters",
    )
    requested_fu_code = str(fu_code).strip()
    for raw_scope in iter_scope_entries(payload=log_payload):
        signature = require_scope_signature(scope=raw_scope)
        candidate_fu_code = str(signature["fu_code"]).strip()
        candidate_selectors = normalize_signature_selectors(signature)
        if candidate_fu_code == requested_fu_code and candidate_selectors == requested_selectors:
            continue
        raise ValueError(
            "deterministic_io_lca does not support mixing different fu_code or selector "
            "scopes in the same project scope because the deterministic IO-LCA tables use "
            "shared canonical filenames. Use refresh=True to replace the existing scope or "
            "choose a different project_name."
        )


@dataclass(frozen=True)
class IOLCAModeExecution:
    """Collected outputs and coverage for one resolved aggregation branch."""

    metadata_path: Path
    output_format: str
    scope_payload: dict
    main_paths: list[Path] = field(default_factory=list)
    origin_paths: list[Path] = field(default_factory=list)
    stage_paths: list[Path] = field(default_factory=list)
    covered_main_years: set[int] = field(default_factory=set)
    covered_origin_years: set[int] = field(default_factory=set)
    covered_stage_years: set[int] = field(default_factory=set)
    skipped_method_years: dict[str, dict[int, str]] = field(default_factory=dict)
    reuse_status: str = "computed"


def _skipped_method_years_from_scope(
    *,
    scope: dict[str, Any],
    methods: list[str],
    mode_tag: str,
) -> dict[str, dict[int, str]]:
    """Return persisted skipped IO-LCA method years for a reused scope."""
    skipped: dict[str, dict[int, str]] = {}
    status = cast(dict[str, Any], scope["status"])
    main_status = cast(dict[str, Any], status["main"])
    for method in methods:
        entry = cast(dict[str, Any], main_status.get(method, {}))
        raw_skipped = entry.get("years_skipped", {})
        years = {
            int(year): str(reason) for year, reason in cast(dict[str, Any], raw_skipped).items()
        }
        if years:
            skipped[f"{method}__{mode_tag}"] = years
    return skipped


def _missing_scope_output_paths(*, scope: dict[str, Any]) -> list[Path]:
    """Return output paths recorded by a complete scope that are missing on disk."""
    return [
        path
        for path in (Path(str(raw_path)) for raw_path in cast(list[Any], scope["paths_written"]))
        if not path.exists()
    ]


def execute_io_lca_mode(
    *,
    project_name: str,
    source: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
    methods: list[str],
    spec,
    filters: dict[str, list[str] | None],
    metadata: dict,
    domain_metadata_path: Path,
    resolved_years: list[int],
    upstream_analysis: bool,
    stages: int,
    stage_outputs_enabled: bool,
    group_indices: bool,
    output_format: str,
    refresh: bool,
    has_multi_indices: bool,
    status: StatusSink | None = None,
) -> IOLCAModeExecution:
    """Execute one deterministic IO-LCAn aggregation branch and persist metadata."""
    validate_group_indices_requires_multi_selection(
        group_indices=group_indices,
        has_multi_indices=has_multi_indices,
    )
    validate_group_indices_supported(spec=spec, group_indices=group_indices)
    mode_tag = mode_label(group_indices=group_indices)
    paths = resolve_io_lca_paths(
        project_name=project_name,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
    )
    io_run_metadata_path = io_metadata_path_for_source(paths=paths, source=source)
    log_payload = load_scope_manifest(
        path=io_run_metadata_path,
        function_name="deterministic_io_lca",
    )
    _ensure_no_conflicting_group_indices_project(
        paths=paths,
        current_metadata_path=io_run_metadata_path,
        log_payload=log_payload,
        group_indices=group_indices,
    )
    if not refresh:
        _ensure_no_conflicting_flat_output_scope(
            log_payload=log_payload,
            fu_code=spec.fu_code,
            filters=filters,
        )
    signature = build_io_lca_signature(
        project_name=project_name,
        source=source,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
        years=resolved_years,
        methods=methods,
        fu_code=spec.fu_code,
        filters={key: value for key, value in filters.items() if key != "studied_indices_tag"},
        upstream_analysis=upstream_analysis,
        upstream_stages=stages,
        group_indices=group_indices,
        output_format=output_format,
    )
    scope_key, scope_existing = get_scope(payload=log_payload, signature=signature)
    if refresh:
        clear_scope_outputs(
            scope_root=lca_results_dir_for_source(
                paths=paths,
                source=source,
            )
        )
        if upstream_analysis:
            clear_scope_outputs(
                scope_root=origin_dir_for_source(
                    paths=paths,
                    source=source,
                )
            )
        if stage_outputs_enabled:
            clear_scope_outputs(
                scope_root=stages_dir_for_source(
                    paths=paths,
                    source=source,
                )
            )
        clear_scope_outputs(
            scope_root=figures_dir_for_source(
                paths=paths,
                source=source,
            )
        )
        io_run_metadata_path.unlink(missing_ok=True)
        log_payload = load_scope_manifest(
            path=io_run_metadata_path,
            function_name="deterministic_io_lca",
        )
        scope_key, scope_existing = get_scope(payload=log_payload, signature=signature)
        scope_existing = None
    scope = ensure_scope(
        payload=log_payload,
        key=scope_key,
        signature=signature,
        function_name="deterministic_io_lca",
    )
    if not refresh and scope_existing is not None:
        if scope_complete_and_existing(scope_existing):
            written_paths = [Path(str(path)) for path in scope_existing["paths_written"]]
            return IOLCAModeExecution(
                metadata_path=io_run_metadata_path,
                output_format=output_format,
                scope_payload=dict(scope_existing),
                main_paths=written_paths,
                covered_main_years={
                    int(year)
                    for method in methods
                    for year in get_lcia_method_years(
                        scope=scope_existing,
                        section="main",
                        lcia_method=method,
                    )
                },
                covered_origin_years={
                    int(year)
                    for method in methods
                    for year in get_lcia_method_years(
                        scope=scope_existing,
                        section="origin",
                        lcia_method=method,
                    )
                },
                covered_stage_years={
                    int(year)
                    for method in methods
                    for year in get_lcia_method_years(
                        scope=scope_existing,
                        section="stages",
                        lcia_method=method,
                    )
                },
                skipped_method_years=_skipped_method_years_from_scope(
                    scope=scope_existing,
                    methods=methods,
                    mode_tag=mode_tag,
                ),
                reuse_status="reused_exact",
            )
        if bool(scope_existing.get("complete")):
            missing = _missing_scope_output_paths(scope=scope_existing)
            raise ValueError(
                "deterministic_io_lca found a complete scope with missing output files. "
                "Run deterministic_io_lca(..., refresh=True) for this scope. "
                f"Missing paths: {[str(path) for path in missing]}."
            )
    io_lca_banner(
        source=source,
        years=resolved_years,
        methods=methods,
        fu_code=spec.fu_code,
        filters=filters,
        upstream_analysis=upstream_analysis,
        upstream_stages=stages,
        status=status,
    )
    progress = (
        StatusProgressPrinter(
            source="deterministic_io_lca",
            action="processing",
            total=0,
            status=status,
        )
        if status is not None
        else year_progress(source="deterministic_io_lca", action="processing", total=0)
    )
    main_paths: list[Path] = []
    origin_paths: list[Path] = []
    stage_paths: list[Path] = []
    covered_main_years: set[int] = set()
    covered_origin_years: set[int] = set()
    covered_stage_years: set[int] = set()
    skipped_method_years: dict[str, dict[int, str]] = {}
    try:
        for lcia_method in methods:
            result = run_io_lca_method(
                lcia_method=lcia_method,
                source=source,
                agg_version=agg_version,
                agg_reg=agg_reg,
                agg_sec=agg_sec,
                spec=spec,
                filters=filters,
                metadata=metadata,
                domain_metadata_path=domain_metadata_path,
                paths=paths,
                scope=scope,
                resolved_years=resolved_years,
                upstream_analysis=upstream_analysis,
                upstream_stages=stages,
                group_indices=group_indices,
                output_format=output_format,
                refresh=refresh,
                method_progress=progress,
            )
            main_paths.extend(result.main_paths)
            origin_paths.extend(result.origin_paths)
            stage_paths.extend(result.stage_paths)
            covered_main_years.update(int(year) for year in result.done_main_years)
            covered_origin_years.update(int(year) for year in result.done_origin_years)
            covered_stage_years.update(int(year) for year in result.done_stage_years)
            if result.skipped_years:
                skipped_method_years[f"{lcia_method}__{mode_tag}"] = result.skipped_years
            set_lcia_method_years(
                scope=scope,
                section="main",
                lcia_method=lcia_method,
                years_done=result.done_main_years,
                skipped_by_year=result.skipped_years if result.skipped_years else None,
            )
            if upstream_analysis:
                set_lcia_method_years(
                    scope=scope,
                    section="origin",
                    lcia_method=lcia_method,
                    years_done=result.done_origin_years,
                    skipped_by_year=result.skipped_years if result.skipped_years else None,
                )
                set_lcia_method_years(
                    scope=scope,
                    section="stages",
                    lcia_method=lcia_method,
                    years_done=result.done_stage_years if stage_outputs_enabled else [],
                    skipped_by_year=result.skipped_years if result.skipped_years else None,
                )
    finally:
        progress.finish()
    required_outputs = [*main_paths, *origin_paths, *stage_paths]
    merge_written_paths(scope=scope, paths=required_outputs)
    set_scope_complete(scope=scope, complete=bool(required_outputs))
    save_scope_manifest(path=io_run_metadata_path, payload=log_payload)
    return IOLCAModeExecution(
        metadata_path=io_run_metadata_path,
        output_format=output_format,
        scope_payload=dict(scope),
        main_paths=main_paths,
        origin_paths=origin_paths,
        stage_paths=stage_paths,
        covered_main_years=covered_main_years,
        covered_origin_years=covered_origin_years,
        covered_stage_years=covered_stage_years,
        skipped_method_years=skipped_method_years,
        reuse_status="computed",
    )
