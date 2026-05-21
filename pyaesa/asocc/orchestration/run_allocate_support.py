"""Shared support for run-level allocation orchestration."""

from dataclasses import dataclass, field
from pathlib import Path

from pyaesa.asocc.io.metadata import _load_run_metadata
from pyaesa.shared.runtime.text import extend_user_text_lines, print_user_text_line
from pyaesa.shared.runtime.reporting.composite_phase_index import PHASE_B1_ASOCC
from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    labelled_values_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.reuse_status import public_reuse_status
from pyaesa.shared.runtime.reporting.summary import document, info, render_summary, section, warning
from pyaesa.asocc.runtime.scope.persisted_scope import load_asocc_persisted_run_catalog
from ..io.contracts import mode_label
from pyaesa.asocc.runtime.paths.deterministic import (
    _get_allocate_regression_fit_inputs_path,
    _get_allocate_regression_stats_path,
    _get_allocate_run_metadata_path,
    _get_allocate_summary_log_path,
    _get_allocate_ut_gvaa_identity_closure_path,
    columns_defs_path_for_stats,
)
from .common_formatting import format_year_ranges
from .reporting_records import deterministic_asocc_info_messages


def format_summary_years(years: list[int]) -> str:
    """Format years as compact ranges for public summaries."""
    return format_year_ranges(years)


def _years_line(
    singular: str,
    plural: str,
    years: list[int],
    formatted_years: str,
) -> str:
    return labelled_values_line(singular, plural, tuple(years), formatted_years)


def _format_mrio_scope(*, context) -> str:
    parts = [
        f"group_reg={bool(context.group_reg)}",
        f"group_sec={bool(context.group_sec)}",
        f"group_version={context.group_version or 'none'}",
        f"aggreg_indices={bool(context.aggreg_indices)}",
    ]
    return ", ".join(parts)


def clear_year_buffers(
    *,
    context,
    state,
    preserve_l2_buckets: set[str] | None = None,
) -> None:
    """Drop per year in memory result buffers after their owner used them."""
    preserved_l2 = {scenario: {} for scenario in context.ssp_scenario_options}
    if preserve_l2_buckets:
        for scenario, results in state.l2_results_by_ssp_scenario.items():
            preserved_l2.setdefault(scenario, {})
            for output_spec, frames in results.items():
                bucket = output_spec.route.bucket or "l2_vs_global"
                if bucket in preserve_l2_buckets:
                    preserved_l2[scenario][output_spec] = list(frames)
    state.l1_results_by_ssp_scenario = {scenario: {} for scenario in context.ssp_scenario_options}
    state.l2_results_by_ssp_scenario = preserved_l2
    state.enacting_metric_inputs = {}
    prune_year_scoped_caches(state=state)


def prune_year_scoped_caches(*, state) -> None:
    """Prune caches that are only useful during the current year loop."""
    processed_years = {int(y) for y in getattr(state, "processed_years", [])}
    if not processed_years:
        return

    for year in list(state.l1_year_invariant_cache.keys()):
        if int(year) in processed_years:
            state.l1_year_invariant_cache.pop(year, None)

    for key in list(state.projection_payload_cache.keys()):
        if int(key[0]) in processed_years:
            state.projection_payload_cache.pop(key, None)

    for scenario_cache in state.preweight_cache_by_ssp_scenario.values():
        for key in list(scenario_cache.keys()):
            if int(key[2]) in processed_years:
                scenario_cache.pop(key, None)

    state.lcia_sliced_payload_cache.clear()
    state.l2_batch_weighting_plan_cache.clear()


def format_indices_label(filters: dict[str, list[str] | None]) -> str:
    """Format filter labels for branch start logging."""
    keys = ("r_p", "s_p", "r_c", "r_f")
    if all(not filters.get(key) for key in keys):
        return "all"
    parts: list[str] = []
    for key in keys:
        values = filters.get(key)
        if not values:
            continue
        joined = "+".join(str(v) for v in values)
        parts.append(f"{key}={joined}")
    return ", ".join(parts)


def has_multi_selected_indices(filters: dict[str, list[str] | None]) -> bool:
    """Return whether at least one studied index has multiple selected values."""
    for key in ("r_p", "s_p", "r_c", "r_f"):
        values = filters.get(key)
        if values and len(values) > 1:
            return True
    return False


def format_branch_label(*, context, mode: str, grouped_mode: bool) -> str:
    """Build concise branch label based on active grouping dimensions."""
    parts: list[str] = []
    if bool(context.group_reg):
        parts.append(f"l1_reg_aggreg={mode}")
    if has_multi_selected_indices(context.filters):
        parts.append(f"aggreg_indices={mode_label(aggreg_indices=grouped_mode)}")
    return ", ".join(parts)


def source_prefix(*, context, show_mode_tag: bool) -> str:
    """Return source scoped prefix used in persisted deterministic summaries."""
    base = f"[{context.source}]"
    if not show_mode_tag:
        return base
    return f"{base} [{context.l1_reg_aggreg}]"


def runtime_prefix(*, context, show_mode_tag: bool) -> str:
    """Return public function scoped prefix used in live deterministic output."""
    base = "[deterministic_asocc]"
    if not show_mode_tag:
        return base
    return f"{base} [{context.l1_reg_aggreg}]"


def _has_regression_projection_context(*, context) -> bool:
    """Return whether regression diagnostics paths apply to the current context."""
    projection_context = getattr(context, "projection_context", None)
    if projection_context is None or getattr(projection_context, "mode", None) != "regression":
        return False
    reg_window = getattr(projection_context, "reg_window", None)
    return reg_window is not None


def _persisted_figure_paths(*, context, source: str) -> list[Path]:
    """Return persisted deterministic aSoCC figure paths from run metadata."""
    metadata_path = _get_allocate_run_metadata_path(
        context.proj_base,
        source=source,
        group_version=context.group_version,
    )
    payload = _load_run_metadata(metadata_path)
    if not load_asocc_persisted_run_catalog(payload=payload).scopes:
        return []
    artifacts = payload["artifacts"]
    paths = []
    for value in artifacts.get("figure_paths", []):
        text = str(value).strip()
        if text:
            paths.append(Path(text))
    return sorted({path for path in paths})


def deterministic_output_root(*, context, source: str) -> Path:
    """Return the deterministic aSoCC output root for one resolved context."""
    scope_manifest = _get_allocate_run_metadata_path(
        context.proj_base,
        source=source,
        group_version=context.group_version,
    )
    return scope_manifest.parent.parent


def _persisted_summary_records(*, context, source: str) -> list[dict[str, str]]:
    """Return persisted deterministic aSoCC summary records for exact reuse."""
    metadata_path = _get_allocate_run_metadata_path(
        context.proj_base,
        source=source,
        group_version=context.group_version,
    )
    if not metadata_path.exists():
        return []
    payload = _load_run_metadata(metadata_path)
    records = payload.get("summary_records") or []
    if not isinstance(records, list):
        return []
    out: list[dict[str, str]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        severity = str(record.get("severity", "")).strip().upper()
        message = str(record.get("message", "")).strip()
        if severity in {"INFO", "WARNING"} and message:
            out.append({"severity": severity, "message": message})
    return out


def emit_runtime_message(*, state, message: str, transient: bool = False) -> None:
    """Emit one runtime message while preserving active progress rendering."""
    progress = getattr(state, "runtime_progress", None)
    log_message = getattr(progress, "log_message", None)
    if callable(log_message):
        log_message(str(message), persistent=not transient)
        return
    print_user_text_line(str(message))


@dataclass(frozen=True)
class AllocateReport:
    """Report container for deterministic_asocc run summaries."""

    source: str
    summaries: list[list[str]]
    figure_paths: list[Path] = field(default_factory=list)
    reuse_status: str = "computed"
    output_root: Path | None = None

    def __str__(self) -> str:
        blocks = ["\n".join(lines) for lines in self.summaries if lines]
        return "\n\n".join(blocks)

    __repr__ = __str__


def build_run_summary_lines(
    *,
    context,
    state,
    show_mode_tag: bool = False,
    figure_paths: list[Path] | None = None,
) -> list[str]:
    """Build end of run summary lines for logging and reporting."""
    resolved_figure_paths = figure_paths or []
    requested = sorted({int(y) for y in context.requested_years})

    requested_msg = format_summary_years(requested)
    computed_requested = {int(y) for y in context.resolved_years}
    skipped_already_saved = sorted(y for y in requested if y not in computed_requested)
    persisted_completed_years = [
        int(y) for y in (getattr(context, "metadata_completed_years", None) or [])
    ]
    processed_source_years = state.processed_years or persisted_completed_years
    processed_years = sorted({int(y) for y in processed_source_years})
    lcia_methods = sorted({str(m) for m in (context.lcia_methods or [])})
    lcia_msg = ", ".join(lcia_methods) if lcia_methods else "none"
    ssp_scenarios = sorted({str(s) for s in context.ssp_scenario_options if s is not None})
    ssp_msg = ", ".join(ssp_scenarios) if ssp_scenarios else "none"
    wb_years = {int(str(c)) for c in context.wb_df.columns if str(c).isdigit()}
    ssp_years = sorted(y for y in requested if y not in wb_years)
    ssp_start_msg = str(ssp_years[0]) if ssp_scenarios and ssp_years else "n/a"
    output_source = context.output_source
    persisted_figure_paths = _persisted_figure_paths(context=context, source=output_source)
    figure_paths = sorted({*resolved_figure_paths, *persisted_figure_paths})
    if not state.processed_years and persisted_completed_years:
        run_status = "reused_exact"
    elif not processed_years:
        run_status = "reused_exact"
    elif skipped_already_saved:
        run_status = "partially_reused"
    else:
        run_status = "computed"

    function_lines = [
        f"Source: {context.output_source}",
        _years_line("Studied year", "Studied years", requested, requested_msg),
        _years_line(
            "Completed year",
            "Completed years",
            processed_years,
            format_summary_years(processed_years),
        ),
        "MRIO scope: " + _format_mrio_scope(context=context),
    ]
    if show_mode_tag:
        function_lines.append(f"L1 regional aggregation mode: {context.l1_reg_aggreg}")
    if skipped_already_saved:
        function_lines.append(
            f"Reused completed years: {format_summary_years(skipped_already_saved)}"
        )
    startup_warnings = [
        str(message).strip()
        for level, message in getattr(state, "startup_notices", [])
        if str(level).strip().upper() == "WARNING" and str(message).strip()
    ]
    startup_infos = [
        str(message).strip()
        for level, message in getattr(state, "startup_notices", [])
        if str(level).strip().upper() == "INFO" and str(message).strip()
    ]
    projection_infos = deterministic_asocc_info_messages(context=context)
    persisted_records = _persisted_summary_records(context=context, source=context.output_source)
    persisted_infos = [
        record["message"] for record in persisted_records if record["severity"] == "INFO"
    ]
    persisted_warnings = [
        record["message"] for record in persisted_records if record["severity"] == "WARNING"
    ]
    family_warnings = [
        str(message).strip()
        for message in getattr(state, "summary_warnings", [])
        if str(message).strip()
    ]
    infos = list(dict.fromkeys([*projection_infos, *startup_infos, *persisted_infos]))
    warnings = list(dict.fromkeys([*startup_warnings, *family_warnings, *persisted_warnings]))
    if lcia_methods:
        function_lines.append(
            labelled_values_line("LCIA method", "LCIA methods", lcia_methods, lcia_msg)
        )
    if ssp_scenarios and ssp_years:
        function_lines.append(
            labelled_values_line("SSP scenario", "SSP scenarios", ssp_scenarios, ssp_msg)
        )
        function_lines.append(f"SSP applies from year: {ssp_start_msg}")
    scope_manifest = _get_allocate_run_metadata_path(
        context.proj_base,
        source=output_source,
        group_version=context.group_version,
    )
    output_root = deterministic_output_root(context=context, source=output_source)
    function_lines.append(f"Output folder: {output_root}")
    recorded_outputs = [
        str(path).strip()
        for path in (getattr(context, "metadata_prior_outputs", None) or [])
        if str(path).strip()
    ]
    log_path = _get_allocate_summary_log_path(
        context.proj_base,
        source=output_source,
        group_version=context.group_version,
    )
    closure_audit_exists = False
    if state.ut_gvaa_identity_closure_rows:
        closure_count = len(state.ut_gvaa_identity_closure_rows)
        closure_path = _get_allocate_ut_gvaa_identity_closure_path(
            proj_base=context.proj_base,
            source=output_source,
            group_version=context.group_version,
        )
        extend_user_text_lines(
            function_lines,
            "UT(GVAa) identity closure adjustments: "
            f"{closure_count} row(s) were set equal to UT(GVA) because "
            "MRIO input-output identity disequilibrium (inputs < outputs) "
            "was detected for the given sector-region pairs (MRIO data quality issue).",
        )
        closure_audit_exists = closure_path.exists()
    inventory = _asocc_inventory_items(
        context=context,
        output_source=output_source,
        scope_manifest=scope_manifest,
        log_path=log_path,
        figure_paths=figure_paths,
        closure_audit_exists=closure_audit_exists,
        has_recorded_outputs=(
            bool(state.output_files_created)
            or bool(state.output_files_updated)
            or bool(getattr(state, "outputs_all", []))
            or bool(recorded_outputs)
        ),
    )
    output_file_count = _asocc_public_output_file_count(
        context=context,
        state=state,
        output_source=output_source,
        recorded_outputs=recorded_outputs,
        log_path=log_path,
        figure_paths=figure_paths,
        closure_audit_exists=closure_audit_exists,
    )
    function_lines.append(output_files_available_line(output_file_count))
    function_lines.extend(inventory_lines(inventory))
    if figure_paths:
        function_lines.append(figures_available_line(len(figure_paths)))
    summary = render_summary(
        document(
            "deterministic_asocc",
            lines=(f"Run status: {public_reuse_status(run_status)}",),
            sections=(
                section(
                    PHASE_B1_ASOCC,
                    children=(
                        section(
                            "deterministic_asocc",
                            lines=function_lines,
                            infos=[info(message) for message in infos],
                            warnings=[warning(message) for message in warnings],
                        ),
                    ),
                ),
            ),
        )
    )
    return summary.splitlines()


def _asocc_inventory_items(
    *,
    context,
    output_source: str,
    scope_manifest: Path,
    log_path: Path,
    figure_paths: list[Path],
    closure_audit_exists: bool,
    has_recorded_outputs: bool,
) -> list:
    """Return deterministic aSoCC folder inventory items for public summaries."""
    inventory = []
    if has_recorded_outputs:
        if getattr(context, "selected_l1", []):
            inventory.append(inventory_item(folder="results/level_1", content="Level 1 shares"))
        if getattr(context, "combined", []):
            inventory.append(
                inventory_item(
                    folder="results/level_2/l2_in_l1",
                    content="conditional L2 in L1 weights used by two step routes",
                )
            )
        if getattr(context, "combined", []) or getattr(context, "selected_l2_one_step", []):
            inventory.append(
                inventory_item(
                    folder="results/level_2/l2_vs_global",
                    content=(
                        "final L2 vs global shares, with direct one step outputs "
                        "and two step L1 * L2 in L1 products as selected"
                    ),
                )
            )
        if bool(getattr(context, "intermediate_outputs", False)):
            inventory.append(
                inventory_item(
                    folder="enacting_metrics",
                    content="intermediate enacting metric inputs",
                )
            )
            if _has_utility_route(context=context):
                inventory.append(
                    inventory_item(
                        folder="results/level_2/utility_propagation_contrib",
                        content="utility propagation contributions",
                    )
                )
        projection_context = getattr(context, "projection_context", None)
        if projection_context is not None and bool(getattr(projection_context, "enabled", False)):
            if _projection_methods_for_route(
                projection_context=projection_context,
                route="historical_reuse",
            ):
                inventory.append(
                    inventory_item(
                        folder="results/level_2/historical_reuse",
                        content="historical reuse L2 route outputs",
                    )
                )
            if _projection_methods_for_route(
                projection_context=projection_context,
                route="regression",
            ):
                inventory.append(
                    inventory_item(
                        folder="results/level_2/regression_proj",
                        content="regression projected L2 route outputs",
                    )
                )
    inventory.append(inventory_item(folder="logs", content="summary log"))
    if _has_regression_projection_context(context=context):
        stats_path = _get_allocate_regression_stats_path(
            proj_base=context.proj_base,
            output_format=context.output_format,
            source=output_source,
            group_version=context.group_version,
        )
        if stats_path.exists():
            inventory.append(
                inventory_item(folder="logs/regression_proj", content="regression diagnostics")
            )
            columns_path = columns_defs_path_for_stats(stats_path=stats_path)
            if columns_path.exists():
                inventory.append(
                    inventory_item(
                        folder="logs/regression_proj",
                        content="regression column definitions",
                    )
                )
        fit_inputs_path = _get_allocate_regression_fit_inputs_path(
            proj_base=context.proj_base,
            output_format=context.output_format,
            source=output_source,
            group_version=context.group_version,
        )
        if fit_inputs_path.exists():
            inventory.append(
                inventory_item(folder="logs/regression_proj", content="regression fit inputs")
            )
    if closure_audit_exists:
        inventory.append(inventory_item(folder="logs", content="UT(GVAa) identity closure audit"))
    return inventory


def _asocc_public_output_file_count(
    *,
    context,
    state,
    output_source: str,
    recorded_outputs: list[str],
    log_path: Path,
    figure_paths: list[Path],
    closure_audit_exists: bool,
) -> int:
    paths: set[str] = set()
    for collection in (
        getattr(state, "output_files_created", []),
        getattr(state, "output_files_updated", []),
        getattr(state, "outputs_all", []),
        recorded_outputs,
        figure_paths,
    ):
        for value in collection:
            paths.add(str(value).strip())
    paths.discard("")
    paths.add(str(log_path))
    if _has_regression_projection_context(context=context):
        stats_path = _get_allocate_regression_stats_path(
            proj_base=context.proj_base,
            output_format=context.output_format,
            source=output_source,
            group_version=context.group_version,
        )
        if stats_path.exists():
            paths.add(str(stats_path))
            columns_path = columns_defs_path_for_stats(stats_path=stats_path)
            if columns_path.exists():
                paths.add(str(columns_path))
        fit_inputs_path = _get_allocate_regression_fit_inputs_path(
            proj_base=context.proj_base,
            output_format=context.output_format,
            source=output_source,
            group_version=context.group_version,
        )
        if fit_inputs_path.exists():
            paths.add(str(fit_inputs_path))
    if closure_audit_exists:
        closure_path = _get_allocate_ut_gvaa_identity_closure_path(
            proj_base=context.proj_base,
            source=output_source,
            group_version=context.group_version,
        )
        paths.add(str(closure_path))
    return len(paths)


def _projection_methods_for_route(*, projection_context, route: str) -> list[str]:
    return [
        str(method)
        for method, method_route in getattr(
            projection_context,
            "l2_method_route_by_name",
            {},
        ).items()
        if str(method_route) == route
    ]


def _has_utility_route(*, context) -> bool:
    l2_methods = [
        *list(getattr(context, "selected_l2_one_step", [])),
        *[l2_method for l2_method, _ in getattr(context, "combined", [])],
    ]
    return any(method in {"UT(FDa)", "UT(GVAa)"} for method in l2_methods)
