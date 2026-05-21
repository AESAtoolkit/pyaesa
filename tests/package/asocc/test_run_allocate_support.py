from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyaesa.asocc.orchestration.run_allocate_support import (
    AllocateReport,
    _has_regression_projection_context,
    _asocc_inventory_items,
    build_run_summary_lines,
    clear_year_buffers,
    emit_runtime_message,
    format_branch_label,
    format_indices_label,
    format_summary_years,
    has_multi_selected_indices,
    prune_year_scoped_caches,
    runtime_prefix,
    source_prefix,
    _persisted_figure_paths,
)
from pyaesa.asocc.io.metadata import _run_scope_key, _save_run_metadata
from pyaesa.asocc.runtime.reporting.family import emit_deduplicated_family_warning
from pyaesa.asocc.runtime.output.contracts import OutputRoute, OutputSpec
from pyaesa.asocc.runtime.paths.deterministic import (
    _get_allocate_regression_fit_inputs_path,
    _get_allocate_regression_stats_path,
    _get_allocate_run_metadata_path,
    _get_allocate_summary_log_path,
)
from pyaesa.asocc.orchestration.write.writers.progress import tick_write_progress


@dataclass
class _RecorderLogger:
    messages: list[str]

    def info(self, message: str) -> None:
        self.messages.append(str(message))


@dataclass
class _RecorderProgress:
    messages: list[str]

    def log_message(self, message: str, persistent: bool = True) -> None:
        self.messages.append(f"{persistent}:{message}")


def _context(
    *,
    proj_base: Path,
    source: str = "oecd_v2025",
    group_reg: bool = False,
    group_sec: bool = False,
    aggreg_indices: bool = False,
    l1_reg_aggreg: str = "post",
    requested_years: list[int] | None = None,
    resolved_years: list[int] | None = None,
    ssp_scenario_options: list[str] | None = None,
    lcia_methods: list[str] | None = None,
    projection_context=None,
    output_source_label: str | None = None,
) -> SimpleNamespace:
    wb_df = SimpleNamespace(columns=[2005, 2010])
    published_source = output_source_label or source
    return SimpleNamespace(
        source=source,
        proj_base=proj_base,
        group_version=None,
        group_reg=group_reg,
        group_sec=group_sec,
        aggreg_indices=aggreg_indices,
        l1_reg_aggreg=l1_reg_aggreg,
        projection_context=projection_context,
        output_format="csv",
        output_source_label=output_source_label,
        output_source=published_source,
        requested_years=[2005, 2030] if requested_years is None else requested_years,
        resolved_years=[2005] if resolved_years is None else resolved_years,
        ssp_scenario_options=(["SSP2"] if ssp_scenario_options is None else ssp_scenario_options),
        lcia_methods=["gwp100_lcia"] if lcia_methods is None else lcia_methods,
        wb_df=wb_df,
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None},
        logger=_RecorderLogger(messages=[]),
    )


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        processed_years=[2005, 2010],
        l1_year_invariant_cache={2005: "drop", 2010: "keep"},
        projection_payload_cache={(2005, "a"): "drop", (2010, "b"): "keep"},
        preweight_cache_by_ssp_scenario={
            "SSP2": {
                ("l1", "l2", 2005): "drop",
                ("l1", "l2", 2010): "keep",
            }
        },
        lcia_sliced_payload_cache={1: 2},
        l2_batch_weighting_plan_cache={("plan",): "drop"},
        l1_results_by_ssp_scenario={},
        l2_results_by_ssp_scenario={},
        enacting_metric_inputs={1: 2},
        outputs_written={"written.csv"},
        output_files_created=["new.csv"],
        output_files_updated=["updated.csv"],
        ut_gvaa_identity_closure_rows=[{"row": 1}],
        runtime_progress=_RecorderProgress(messages=[]),
    )


def _output_spec(bucket: str) -> OutputSpec:
    return OutputSpec(
        l1_l2_method="method",
        l2_method="l2",
        l1_method=None,
        file_stem=f"stem_{bucket}",
        route=OutputRoute(
            level="L2",
            bucket=bucket,
            source=None,
            grouped_mode=False,
            variant_tag=None,
            ssp_scenario=None,
            lcia_method=None,
            projection_subfolder=None,
        ),
        scenario_dependent=False,
        identifier_columns=("r_p",),
    )


def test_format_and_selection_cover_all_branches() -> None:
    assert format_summary_years([2005, 2005, 2010]) == "2005, 2010"
    assert format_indices_label({"r_p": None, "s_p": None, "r_c": None, "r_f": None}) == "all"
    assert format_indices_label({"r_p": ["FR"], "s_p": ["A", "B"], "r_c": None, "r_f": []}) == (
        "r_p=FR, s_p=A+B"
    )
    assert has_multi_selected_indices({"r_p": ["FR"], "s_p": ["A", "B"], "r_c": None, "r_f": []})
    assert not has_multi_selected_indices({"r_p": ["FR"], "s_p": ["A"], "r_c": None, "r_f": []})
    assert (
        format_branch_label(
            context=SimpleNamespace(group_reg=True, filters={"r_p": ["FR", "DE"]}),
            mode="post",
            grouped_mode=True,
        )
        == "l1_reg_aggreg=post, aggreg_indices=grouped"
    )
    assert (
        format_branch_label(
            context=SimpleNamespace(group_reg=False, filters={"r_p": ["FR"], "s_p": ["A", "B"]}),
            mode="post",
            grouped_mode=False,
        )
        == "aggreg_indices=ungrouped"
    )
    assert (
        format_branch_label(
            context=SimpleNamespace(group_reg=False, filters={"r_p": ["FR"], "s_p": ["A"]}),
            mode="post",
            grouped_mode=False,
        )
        == ""
    )
    assert (
        source_prefix(context=SimpleNamespace(source="oecd_v2025"), show_mode_tag=False)
        == "[oecd_v2025]"
    )
    assert (
        source_prefix(
            context=SimpleNamespace(source="oecd_v2025", l1_reg_aggreg="post"),
            show_mode_tag=True,
        )
        == "[oecd_v2025] [post]"
    )
    assert (
        runtime_prefix(context=SimpleNamespace(l1_reg_aggreg="post"), show_mode_tag=False)
        == "[deterministic_asocc]"
    )
    assert (
        runtime_prefix(context=SimpleNamespace(l1_reg_aggreg="post"), show_mode_tag=True)
        == "[deterministic_asocc] [post]"
    )


def test_cache_and_message_cover_remaining_branches(tmp_path: Path) -> None:
    context = _context(proj_base=tmp_path)
    state = _state()
    clear_year_buffers(context=context, state=state)
    assert state.l1_results_by_ssp_scenario == {"SSP2": {}}
    assert state.l2_results_by_ssp_scenario == {"SSP2": {}}
    assert state.enacting_metric_inputs == {}
    assert state.l1_year_invariant_cache == {}
    assert state.projection_payload_cache == {}
    assert state.preweight_cache_by_ssp_scenario["SSP2"] == {}
    assert state.lcia_sliced_payload_cache == {}
    assert state.l2_batch_weighting_plan_cache == {}

    preserve_state = _state()
    keep_spec = _output_spec("l2_in_l1")
    drop_spec = _output_spec("l2_vs_global")
    preserve_state.l2_results_by_ssp_scenario = {
        "SSP2": {
            keep_spec: ["keep"],
            drop_spec: ["drop"],
        }
    }
    clear_year_buffers(
        context=context,
        state=preserve_state,
        preserve_l2_buckets={"l2_in_l1"},
    )
    assert preserve_state.l2_results_by_ssp_scenario == {"SSP2": {keep_spec: ["keep"]}}

    partial_state = SimpleNamespace(
        processed_years=[2005],
        l1_year_invariant_cache={2005: "drop", 2010: "keep"},
        projection_payload_cache={(2005, "a"): "drop", (2010, "b"): "keep"},
        preweight_cache_by_ssp_scenario={
            "SSP2": {
                ("l1", "l2", 2005): "drop",
                ("l1", "l2", 2010): "keep",
            }
        },
        lcia_sliced_payload_cache={1: 2},
        lcia_method_payload_cache={("method",): "drop"},
        l2_batch_weighting_plan_cache={("plan",): "drop"},
    )
    prune_year_scoped_caches(state=partial_state)
    assert partial_state.l1_year_invariant_cache == {2010: "keep"}
    assert partial_state.projection_payload_cache == {(2010, "b"): "keep"}
    assert partial_state.preweight_cache_by_ssp_scenario["SSP2"] == {("l1", "l2", 2010): "keep"}
    assert partial_state.l2_batch_weighting_plan_cache == {}

    empty_state = SimpleNamespace(
        processed_years=[],
        l1_year_invariant_cache={},
        projection_payload_cache={},
        preweight_cache_by_ssp_scenario={},
        lcia_sliced_payload_cache={},
        l2_batch_weighting_plan_cache={("plan",): "keep"},
    )
    prune_year_scoped_caches(state=empty_state)
    assert empty_state.lcia_sliced_payload_cache == {}
    assert empty_state.l2_batch_weighting_plan_cache == {("plan",): "keep"}

    emit_runtime_message(state=state, message="hello", transient=True)
    assert state.runtime_progress.messages == ["False:hello"]

    print_state = SimpleNamespace(runtime_progress=None)
    emit_runtime_message(state=print_state, message="printed", transient=False)

    warning_messages: list[str] = []
    state_with_existing_warnings = SimpleNamespace(
        notices_emitted=set(),
        summary_warnings=[],
    )
    emit_deduplicated_family_warning(
        context=SimpleNamespace(
            logger=SimpleNamespace(warning=lambda message: warning_messages.append(str(message))),
            source="fallback_source",
        ),
        state=state_with_existing_warnings,
        key="demo-warning",
        message="family warning",
    )
    assert len(warning_messages) == 1
    assert state_with_existing_warnings.summary_warnings == warning_messages


def test_family_reporting_warning_cover_summary_warning_and_dedup() -> None:
    warning_messages: list[str] = []
    context = SimpleNamespace(
        logger=SimpleNamespace(warning=lambda message: warning_messages.append(str(message))),
        source="fallback_source",
    )
    state = SimpleNamespace(
        notices_emitted=set(),
    )

    emit_deduplicated_family_warning(
        context=context,
        state=state,
        key="printed-warning",
        message="printed warning",
    )
    emit_deduplicated_family_warning(
        context=context,
        state=state,
        key="printed-warning",
        message="printed warning",
    )

    assert len(warning_messages) == 1
    assert state.summary_warnings == warning_messages


def test_write_progress_uses_runtime_progress_and_print_fallback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = SimpleNamespace(source="oecd_v2025")
    progress = _RecorderProgress(messages=[])
    state = SimpleNamespace(
        write_progress_total=2,
        write_progress_current=0,
        write_progress_prefix="",
        write_progress_label="a very long method label that should be compacted before display",
        runtime_progress=progress,
    )

    tick_write_progress(context=context, state=state)
    assert progress.messages[0]
    assert "oecd_v2025" in progress.messages[0]
    assert state.write_progress_current == 1

    state.write_progress_current = 2
    tick_write_progress(context=context, state=state)
    assert state.write_progress_current == 2
    assert state.write_progress_last_width == 0

    print_state = SimpleNamespace(
        write_progress_total=1,
        write_progress_current=0,
        write_progress_prefix="[custom]",
        write_progress_label=None,
        write_progress_last_width=5,
        runtime_progress=None,
    )
    tick_write_progress(context=context, state=print_state)
    printed = capsys.readouterr().out
    assert printed.strip()
    assert "custom" in printed
    assert print_state.write_progress_last_width == 0

    active_print_state = SimpleNamespace(
        write_progress_total=2,
        write_progress_current=0,
        write_progress_prefix="[custom]",
        write_progress_label=None,
        write_progress_last_width=5,
        runtime_progress=None,
    )
    tick_write_progress(context=context, state=active_print_state)
    assert active_print_state.write_progress_last_width > 0

    skip_state = SimpleNamespace(write_progress_total=0)
    tick_write_progress(context=context, state=skip_state)
    assert getattr(skip_state, "write_progress_current", None) is None


def test_persisted_figure_paths_and_summary_cover_report_paths(
    tmp_path: Path,
) -> None:
    projection_context = SimpleNamespace(mode="regression", reg_window=(2005, 2010))
    context = _context(
        proj_base=tmp_path,
        group_reg=True,
        aggreg_indices=True,
        projection_context=projection_context,
        output_source_label="oecd_v2025",
    )
    state = _state()
    metadata_path = _get_allocate_run_metadata_path(
        tmp_path,
        source="oecd_v2025",
        group_version=None,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    figure_a = tmp_path / "figures" / "figure_a.png"
    figure_b = tmp_path / "figures" / "figure_b.png"
    figure_a.parent.mkdir(parents=True, exist_ok=True)
    scope_key = _run_scope_key(signature={"source": "oecd_v2025"})
    _save_run_metadata(
        metadata_path,
        {
            "function": "deterministic_asocc",
            "arguments": {"source": "oecd_v2025"},
            "execution": {"status": "complete", "completed_years": [2005]},
            "reuse": {"identity_key": scope_key},
            "artifacts": {
                "outputs": [],
                "figure_paths": ["", str(figure_b), str(figure_a), str(figure_a)],
            },
            "provenance": {"ssp_scenarios": ["SSP2"]},
        },
    )
    paths = _persisted_figure_paths(context=context, source="oecd_v2025")
    assert paths == sorted({figure_a, figure_b})
    state.startup_notices = [("WARNING", "LCIA input unavailable for one method")]
    state.summary_warnings = ["scenario route skipped one row"]

    log_path = _get_allocate_summary_log_path(
        tmp_path,
        source="oecd_v2025",
        group_version=None,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("log", encoding="utf-8")
    stats_path = _get_allocate_regression_stats_path(
        proj_base=tmp_path,
        output_format="csv",
        source="oecd_v2025",
        group_version=None,
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text("stats", encoding="utf-8")
    fit_inputs_path = _get_allocate_regression_fit_inputs_path(
        proj_base=tmp_path,
        output_format="csv",
        source="oecd_v2025",
        group_version=None,
    )
    fit_inputs_path.write_text("fit", encoding="utf-8")

    lines = build_run_summary_lines(
        context=context,
        state=state,
        show_mode_tag=True,
        figure_paths=[figure_b],
    )
    assert len(lines) > 1
    stats_path.unlink()
    fit_inputs_path.unlink()

    report = AllocateReport(source="oecd_v2025", summaries=[["alpha"], ["beta"]])
    assert str(report) == "alpha\n\nbeta"
    assert repr(report) == str(report)


def test_has_regression_projection_context_covers_none_and_regression_branch(
    tmp_path: Path,
) -> None:
    no_projection = _context(proj_base=tmp_path, projection_context=None)
    assert _has_regression_projection_context(context=no_projection) is False

    non_regression = _context(
        proj_base=tmp_path,
        projection_context=SimpleNamespace(mode="historical_reuse", reg_window=(2005, 2010)),
    )
    assert _has_regression_projection_context(context=non_regression) is False

    regression = _context(
        proj_base=tmp_path,
        projection_context=SimpleNamespace(mode="regression", reg_window=(2005, 2010)),
    )
    assert _has_regression_projection_context(context=regression) is True


def test_metadata_and_summary_cover_validation_and_false_branches(
    tmp_path: Path,
) -> None:
    context = _context(
        proj_base=tmp_path,
        requested_years=[2005],
        resolved_years=[2005],
        ssp_scenario_options=[],
        lcia_methods=[],
        projection_context=SimpleNamespace(mode="regression", reg_window=None),
    )
    state = SimpleNamespace(
        processed_years=[],
        l1_year_invariant_cache={},
        projection_payload_cache={},
        preweight_cache_by_ssp_scenario={},
        lcia_sliced_payload_cache={},
        l1_results_by_ssp_scenario={},
        l2_results_by_ssp_scenario={},
        enacting_metric_inputs={},
        outputs_written=set(),
        output_files_created=[],
        output_files_updated=[],
        ut_gvaa_identity_closure_rows=[],
        runtime_progress=None,
    )

    lines = build_run_summary_lines(context=context, state=state, figure_paths=None)
    assert len(lines) > 1
    assert _has_regression_projection_context(context=context) is False
    persisted_context = _context(
        proj_base=tmp_path,
        requested_years=[2005],
        resolved_years=[2005],
        ssp_scenario_options=[],
        lcia_methods=[],
        projection_context=None,
    )
    persisted_context.metadata_completed_years = [2005]
    persisted_lines = build_run_summary_lines(
        context=persisted_context,
        state=state,
        figure_paths=None,
    )
    assert any("reused" in line.lower() for line in persisted_lines)

    regression_context = _context(
        proj_base=tmp_path,
        requested_years=[2005],
        resolved_years=[2005],
        ssp_scenario_options=[],
        lcia_methods=[],
        projection_context=SimpleNamespace(mode="regression", reg_window=(2005, 2010)),
    )
    regression_lines = build_run_summary_lines(
        context=regression_context,
        state=state,
        figure_paths=None,
    )
    assert len(regression_lines) > 1

    computed_state = SimpleNamespace(
        processed_years=[2005],
        l1_year_invariant_cache={},
        projection_payload_cache={},
        preweight_cache_by_ssp_scenario={},
        lcia_sliced_payload_cache={},
        l1_results_by_ssp_scenario={},
        l2_results_by_ssp_scenario={},
        enacting_metric_inputs={},
        outputs_written=set(),
        output_files_created=[],
        output_files_updated=[],
        ut_gvaa_identity_closure_rows=[],
        runtime_progress=None,
    )
    computed_lines = build_run_summary_lines(
        context=context,
        state=computed_state,
        figure_paths=None,
    )
    assert computed_lines
    inventory_context = _context(proj_base=tmp_path, projection_context=None)
    inventory_context.intermediate_outputs = True
    inventory_context.selected_l2_one_step = ["UT(FDa)"]
    inventory = _asocc_inventory_items(
        context=inventory_context,
        output_source=inventory_context.output_source,
        scope_manifest=tmp_path / "metadata" / "scope_manifest.json",
        log_path=tmp_path / "logs" / "summary.log",
        figure_paths=[],
        closure_audit_exists=False,
        has_recorded_outputs=True,
    )
    assert any(item.folder == "results/level_2/utility_propagation_contrib" for item in inventory)
    assert any(item.folder == "results/level_2/l2_vs_global" for item in inventory)
    two_step_inventory_context = _context(proj_base=tmp_path, projection_context=None)
    two_step_inventory_context.intermediate_outputs = False
    two_step_inventory_context.combined = [("UT(FD)", "EG(Pop)")]
    two_step_inventory_context.selected_l2_one_step = []
    two_step_inventory = _asocc_inventory_items(
        context=two_step_inventory_context,
        output_source=two_step_inventory_context.output_source,
        scope_manifest=tmp_path / "metadata" / "scope_manifest.json",
        log_path=tmp_path / "logs" / "summary.log",
        figure_paths=[],
        closure_audit_exists=False,
        has_recorded_outputs=True,
    )
    assert any(item.folder == "results/level_2/l2_in_l1" for item in two_step_inventory)
    assert any(item.folder == "results/level_2/l2_vs_global" for item in two_step_inventory)
    no_utility_context = _context(proj_base=tmp_path, projection_context=None)
    no_utility_context.intermediate_outputs = True
    no_utility_context.selected_l2_one_step = ["AR(E^{CBA_FD})"]
    no_utility_inventory = _asocc_inventory_items(
        context=no_utility_context,
        output_source=no_utility_context.output_source,
        scope_manifest=tmp_path / "metadata" / "scope_manifest.json",
        log_path=tmp_path / "logs" / "summary.log",
        figure_paths=[],
        closure_audit_exists=False,
        has_recorded_outputs=True,
    )
    assert not any(
        item.folder == "results/level_2/utility_propagation_contrib"
        for item in no_utility_inventory
    )

    metadata_path = _get_allocate_run_metadata_path(
        tmp_path,
        source=context.output_source,
        group_version=None,
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    base_payload = {
        "function": "deterministic_asocc",
        "arguments": {"source": context.output_source},
        "execution": {"status": "complete", "completed_years": [2005]},
        "reuse": {"identity_key": _run_scope_key(signature={"source": context.output_source})},
        "artifacts": {"outputs": []},
        "provenance": {},
    }
    _save_run_metadata(metadata_path, {**base_payload, "summary_records": {"bad": "shape"}})
    assert build_run_summary_lines(context=context, state=state, figure_paths=None)
    _save_run_metadata(
        metadata_path,
        {
            **base_payload,
            "summary_records": [
                "ignored",
                {"severity": "INFO", "message": " persisted info "},
                {"severity": "ERROR", "message": "ignored"},
            ],
        },
    )
    persisted_info_lines = build_run_summary_lines(context=context, state=state, figure_paths=None)
    assert any("persisted" in line for line in persisted_info_lines)
