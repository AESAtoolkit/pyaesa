from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pyaesa.asocc.io.metadata import _load_run_metadata
from pyaesa.asocc.orchestration.write import run_write as run_write_mod
from pyaesa.asocc.orchestration.write.writers import allocations as allocations_mod
from pyaesa.asocc.orchestration.write.writers import progress as write_progress_mod
from pyaesa.asocc.runtime.paths.deterministic import _get_allocate_run_metadata_path
from pyaesa.asocc.runtime.paths.published import _get_asocc_l2_dir
from pyaesa.asocc.runtime.output.contracts import (
    IdentifierSchema,
    OutputArtifact,
    OutputRoute,
    OutputSpec,
)
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_HISTORICAL,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)


def _artifact(*, value: float) -> OutputArtifact:
    schema = IdentifierSchema(
        columns=("l1_l2_method", "l2_method", "l1_method", "lcia_method", "r_p"),
        year_columns=("2020",),
    )
    frame = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FD)"],
            "l2_method": ["UT(FD)"],
            "l1_method": [None],
            "lcia_method": [None],
            "r_p": ["FR"],
            "2020": [value],
        }
    )
    return OutputArtifact(schema=schema, data_wide=frame)


def _empty_artifact() -> OutputArtifact:
    schema = IdentifierSchema(
        columns=("l1_l2_method", "l2_method", "l1_method", "lcia_method", "r_p"),
        year_columns=("2020",),
    )
    frame = pd.DataFrame(columns=[*schema.columns, *schema.year_columns])
    return OutputArtifact(schema=schema, data_wide=frame)


def _output_spec(*, bucket: str) -> OutputSpec:
    return OutputSpec(
        l1_l2_method="UT(FD)",
        l2_method="UT(FD)",
        l1_method=None,
        file_stem=f"table_{bucket}",
        route=OutputRoute(
            level="level_2",
            bucket=bucket,
            source="oecd_v2025",
            grouped_mode=False,
            variant_tag=None,
            ssp_scenario=None,
            lcia_method=None,
            projection_subfolder=None,
        ),
        scenario_dependent=False,
        identifier_columns=("r_p",),
    )


def _l1_output_spec() -> OutputSpec:
    return OutputSpec(
        l1_l2_method="UT(FD)",
        l2_method="UT(FD)",
        l1_method=None,
        file_stem="table_l1",
        route=OutputRoute(
            level="level_1",
            bucket=None,
            source="oecd_v2025",
            grouped_mode=False,
            variant_tag=None,
            ssp_scenario=None,
            lcia_method=None,
            projection_subfolder=None,
        ),
        scenario_dependent=False,
        identifier_columns=("r_p",),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame({2020: [1.0]}, index=pd.Index(["FR"], name="r_p"))


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        output_files_created=[],
        output_files_updated=[],
        outputs_written=[],
        outputs_all=[],
        l1_results_by_ssp_scenario={},
        l2_results_by_ssp_scenario={},
        enacting_metric_inputs={},
        regression_stats_rows=[],
        regression_fit_inputs_rows=[],
        ut_gvaa_identity_closure_rows=[],
        processed_years=[],
        skipped_years={},
        empty_ref_years=[],
        write_progress_total=0,
        write_progress_current=0,
        write_progress_last_width=0,
        write_progress_label=None,
        write_progress_prefix=None,
    )


def _projection_context(value=None) -> SimpleNamespace:
    if value is None:
        value = SimpleNamespace(enabled=False, mode=None, reg_window=None, l2_reuse_years=[])
    routes = dict(getattr(value, "l2_method_route_by_name", {}))
    route_for_l2_method = getattr(value, "route_for_l2_method", None)
    if route_for_l2_method is None:
        route_for_l2_method = routes.get
    l2_reuse_years = list(getattr(value, "l2_reuse_years", []))
    max_historical_year = getattr(value, "max_historical_year", None)
    return SimpleNamespace(
        enabled=bool(getattr(value, "enabled", getattr(value, "mode", None) is not None)),
        mode=getattr(value, "mode"),
        reg_window=getattr(value, "reg_window"),
        l2_reuse_years=l2_reuse_years,
        future_years=list(getattr(value, "future_years", [])),
        max_historical_year=max_historical_year,
        l2_method_route_by_name=routes,
        route_for_l2_method=route_for_l2_method,
        l2_reuse_years_for=lambda: tuple(sorted({int(year) for year in l2_reuse_years})),
        is_future_year=lambda year: (
            False if max_historical_year is None else int(year) > int(max_historical_year)
        ),
    )


def _context(
    tmp_path: Path,
    *,
    intermediate_outputs: bool = True,
    output_source_label: str | None = "oecd_v2025",
    projection_context=None,
    metadata_completed_years: list[int] | None = None,
    metadata_prior_outputs: list[str] | None = None,
) -> SimpleNamespace:
    normalized_projection = _projection_context(projection_context)
    published_source = output_source_label or "oecd_v2025"
    return SimpleNamespace(
        project_name="write_runtime",
        proj_base=tmp_path,
        source="oecd_v2025",
        output_source_label=output_source_label,
        output_source=published_source,
        group_version=None,
        group_reg=False,
        group_sec=False,
        fu_code="L1.a",
        lcia_method=["pb_lcia"],
        years_input=[2020],
        reference_years_input=None,
        requested_years=[2020],
        wb_df=pd.DataFrame(columns=["2020"]),
        selected_methods=["UT(FD)"],
        selected_l1=[],
        combined=[],
        selected_l2_one_step=[],
        studied_indices_tag="custom",
        reference_years=None,
        ssp_scenario_options=[],
        ssp_scenario=None,
        variant_tag=None,
        run_signature={"source": "oecd_v2025", "years": [2020], "methods": ["UT(FD)"]},
        aggreg_indices=False,
        l1_reg_aggreg="post",
        output_format="csv",
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None},
        persisted_years=[2020],
        resolved_years=[2020],
        intermediate_outputs=intermediate_outputs,
        projection_context=normalized_projection,
        metadata_completed_years=metadata_completed_years,
        metadata_prior_outputs=metadata_prior_outputs,
    )


def test_write_result_artifact_covers_empty_existing_and_repeated_updates(tmp_path: Path) -> None:
    context = _context(tmp_path)
    out_path = tmp_path / "allocation.csv"

    created_state = _state()
    allocations_mod.write_result_artifact(
        context=context,
        artifact=_artifact(value=1.0),
        out_path=out_path,
        refresh_effective=False,
        output_format="csv",
        state=created_state,
    )
    out_path_str = str(out_path)
    assert created_state.output_files_created == [out_path_str]
    assert created_state.output_files_updated == []
    assert created_state.outputs_written == [out_path_str]
    assert created_state.outputs_all == [out_path_str]

    empty_state = _state()
    empty_path = tmp_path / "empty.csv"
    allocations_mod.write_result_artifact(
        context=context,
        artifact=_empty_artifact(),
        out_path=empty_path,
        refresh_effective=False,
        output_format="csv",
        state=empty_state,
    )
    assert not empty_path.exists()
    assert empty_state.outputs_all == []

    unchanged_state = _state()
    allocations_mod.write_result_artifact(
        context=context,
        artifact=_artifact(value=1.0),
        out_path=out_path,
        refresh_effective=False,
        output_format="csv",
        state=unchanged_state,
    )
    assert unchanged_state.output_files_created == []
    assert unchanged_state.output_files_updated == []
    assert unchanged_state.outputs_written == []
    assert unchanged_state.outputs_all == [out_path_str]

    allocations_mod.write_result_artifact(
        context=context,
        artifact=_artifact(value=1.0),
        out_path=out_path,
        refresh_effective=False,
        output_format="csv",
        state=unchanged_state,
    )
    assert unchanged_state.outputs_all == [out_path_str]

    updated_state = _state()
    allocations_mod.write_result_artifact(
        context=context,
        artifact=_artifact(value=2.0),
        out_path=out_path,
        refresh_effective=False,
        output_format="csv",
        state=updated_state,
    )
    assert updated_state.output_files_updated == [out_path_str]
    assert updated_state.outputs_written == [out_path_str]
    assert updated_state.outputs_all == [out_path_str]

    allocations_mod.write_result_artifact(
        context=context,
        artifact=_artifact(value=3.0),
        out_path=out_path,
        refresh_effective=False,
        output_format="csv",
        state=updated_state,
    )
    assert updated_state.output_files_updated == [out_path_str]
    assert updated_state.outputs_written == [out_path_str]
    assert updated_state.outputs_all == [out_path_str]


def test_write_l2_outputs_expand_l2_in_l1_years_for_historical_reuse(tmp_path: Path) -> None:
    context = _context(
        tmp_path,
        projection_context=SimpleNamespace(
            enabled=True,
            mode="regression",
            reg_window=(2008, 2009),
            l2_reuse_years=[2009],
            route_for_l2_method=lambda name: "historical_reuse" if name == "UT(FD)" else None,
        ),
    )
    context.persisted_years = [2030]
    context.resolved_years = [2030]
    spec = _output_spec(bucket="l2_in_l1")
    state = _state()
    state.l2_results_by_ssp_scenario = {
        None: {
            spec: [
                pd.DataFrame(
                    {2008: [1.0], 2009: [2.0]},
                    index=pd.Index(["FR"], name="r_p"),
                )
            ]
        }
    }

    allocations_mod.write_l2_outputs(
        context=context,
        state=state,
        refresh_effective=False,
    )

    out_path = (
        _get_asocc_l2_dir(
            proj_base=context.proj_base,
            source=context.output_source,
            group_version=context.group_version,
            bucket="l2_in_l1",
            lcia_sub=None,
        )
        / "table_l2_in_l1.csv"
    )
    written = pd.read_csv(out_path)
    assert list(written.columns) == [
        "l1_l2_method",
        "l2_method",
        ASOCC_SSP_SCENARIO_COLUMN,
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        "r_p",
        "2008",
        "2009",
    ]
    assert pd.isna(written.loc[0, ASOCC_SSP_SCENARIO_COLUMN])
    assert written.loc[0, ASOCC_TIME_ROUTE_PUBLIC_COLUMN] == ASOCC_TIME_ROUTE_HISTORICAL
    assert float(written.loc[0, "2008"]) == 1.0
    assert float(written.loc[0, "2009"]) == 2.0


def test_l2_in_l1_year_contract_includes_regression_owner_years_for_historical_reuse(
    tmp_path: Path,
) -> None:
    spec = _output_spec(bucket="l2_in_l1")
    context = _context(
        tmp_path,
        projection_context=SimpleNamespace(
            enabled=True,
            mode="regression",
            reg_window=(1995, 1997),
            l2_reuse_years=[1997],
            route_for_l2_method=lambda _name: "historical_reuse",
        ),
    )
    context.persisted_years = [2026]

    assert allocations_mod._output_years_for_output_spec(  # noqa: SLF001
        context=context,
        output_spec=spec,
    ) == [1995, 1996, 1997, 2026]


def test_write_l2_outputs_respects_allowed_buckets(tmp_path: Path) -> None:
    context = _context(tmp_path)
    state = _state()
    state.l2_results_by_ssp_scenario = {
        None: {
            _output_spec(bucket="l2_in_l1"): [_frame()],
            _output_spec(bucket="l2_vs_global"): [_frame()],
        }
    }

    allocations_mod.write_l2_outputs(
        context=context,
        state=state,
        refresh_effective=False,
        allowed_buckets={"l2_in_l1"},
    )

    assert any("table_l2_in_l1.csv" in path for path in state.outputs_all)
    assert not any("table_l2_vs_global.csv" in path for path in state.outputs_all)


def test_l2_in_l1_year_contract_keeps_persisted_years_for_non_reuse_routes(
    tmp_path: Path,
) -> None:
    spec = _output_spec(bucket="l2_in_l1")
    context_without_route_callable = _context(
        tmp_path,
        projection_context=SimpleNamespace(
            enabled=True,
            mode="historical_reuse",
            reg_window=(2008, 2009),
            l2_reuse_years=[2008, 2009],
        ),
    )
    context_without_route_callable.persisted_years = [2030]
    assert allocations_mod._output_years_for_output_spec(  # noqa: SLF001
        context=context_without_route_callable,
        output_spec=spec,
    ) == [2030]

    context_regression_route = _context(
        tmp_path,
        projection_context=SimpleNamespace(
            enabled=True,
            mode="historical_reuse",
            reg_window=(2008, 2009),
            l2_reuse_years=[2008, 2009],
            route_for_l2_method=lambda _name: "regression",
        ),
    )
    context_regression_route.persisted_years = [2030]
    assert allocations_mod._output_years_for_output_spec(  # noqa: SLF001
        context=context_regression_route,
        output_spec=spec,
    ) == [2030]


def test_tick_write_progress_runtime_logger_keeps_incomplete_width_state() -> None:
    messages: list[tuple[str, bool]] = []
    context = SimpleNamespace(source="oecd_v2025")
    state = SimpleNamespace(
        write_progress_total=2,
        write_progress_current=0,
        write_progress_last_width=9,
        write_progress_label="branch",
        write_progress_prefix="[custom]",
        runtime_progress=SimpleNamespace(
            log_message=lambda message, persistent=False: messages.append((message, persistent))
        ),
    )

    write_progress_mod.tick_write_progress(context=context, state=state)

    assert state.write_progress_current == 1
    assert state.write_progress_last_width == 9
    assert len(messages) == 1
    assert messages[0][1] is False


def test_run_write_covers_regression_counts_and_metadata_guards(tmp_path: Path) -> None:
    plain_context = _context(tmp_path, projection_context=None)
    assert run_write_mod._context_output_source(context=plain_context) == "oecd_v2025"  # noqa: SLF001
    regression_context = _context(
        tmp_path,
        projection_context=SimpleNamespace(mode="regression", reg_window=(2018, 2021)),
    )

    regression_state = _state()
    regression_state.regression_stats_rows = [{"row": 1}]
    regression_state.regression_fit_inputs_rows = [{"row": 1}]
    assert (
        run_write_mod._count_regression_output_targets(  # noqa: SLF001
            context=regression_context,
            state=regression_state,
        )
        == 2
    )
    assert (
        run_write_mod._count_regression_output_targets(  # noqa: SLF001
            context=regression_context,
            state=_state(),
        )
        == 0
    )


def test_write_stage_covers_intermediate_skip_and_write_metadata_toggle(
    tmp_path: Path,
) -> None:
    utility_spec = _output_spec(bucket="utility_propagation_contrib")
    normal_spec = _output_spec(bucket="l2_vs_global")
    count_state = _state()
    count_state.l2_results_by_ssp_scenario = {
        None: {
            utility_spec: [_frame()],
            normal_spec: [_frame()],
        }
    }
    assert (
        allocations_mod.count_l2_output_targets(
            context=_context(tmp_path, intermediate_outputs=False),
            state=count_state,
        )
        == 1
    )
    duplicate_scenario_state = _state()
    duplicate_scenario_state.l1_results_by_ssp_scenario = {
        "SSP1": {_l1_output_spec(): [_frame()]},
        "SSP2": {_l1_output_spec(): [_frame()]},
    }
    duplicate_scenario_state.l2_results_by_ssp_scenario = {
        "SSP1": {normal_spec: [_frame()]},
        "SSP2": {normal_spec: [_frame()]},
    }
    assert allocations_mod.count_l1_output_targets(state=duplicate_scenario_state) == 1
    assert (
        allocations_mod.count_l2_output_targets(
            context=_context(tmp_path),
            state=duplicate_scenario_state,
        )
        == 1
    )

    skip_state = _state()
    skip_state.l2_results_by_ssp_scenario = {None: {utility_spec: [_frame()]}}
    allocations_mod.write_l2_outputs(
        context=_context(tmp_path, intermediate_outputs=False),
        state=skip_state,
        refresh_effective=False,
    )
    assert skip_state.outputs_all == []

    metadata_off_state = _state()
    run_write_mod._write_outputs(  # noqa: SLF001
        context=_context(tmp_path),
        state=metadata_off_state,
        refresh=False,
        write_metadata=False,
        show_progress=False,
    )
    metadata_path = _get_allocate_run_metadata_path(
        tmp_path,
        source="oecd_v2025",
        group_version=None,
    )
    assert not metadata_path.exists()


def test_write_allocation_covers_l1_l2_loops_and_metadata_commit(
    tmp_path: Path,
) -> None:
    context = _context(
        tmp_path,
        output_source_label=None,
        projection_context=SimpleNamespace(
            enabled=True,
            mode="regression",
            reg_window=(2018, 2021),
            l2_reuse_years=[],
        ),
    )
    state = _state()
    state.l1_results_by_ssp_scenario = {None: {_l1_output_spec(): [_frame()]}}
    state.l2_results_by_ssp_scenario = {None: {_output_spec(bucket="l2_vs_global"): [_frame()]}}

    allocations_mod.write_l1_outputs(
        context=context,
        state=state,
        refresh_effective=False,
        l1_source=None,
    )
    allocations_mod.write_l2_outputs(
        context=context,
        state=state,
        refresh_effective=False,
    )

    l1_path = next(Path(path) for path in state.outputs_all if "table_l1.csv" in path)
    l2_path = next(Path(path) for path in state.outputs_all if "table_l2_vs_global.csv" in path)
    assert l1_path.exists()
    assert l2_path.exists()

    artifact = allocations_mod.build_l1_artifact(  # noqa: SLF001
        output_spec=_l1_output_spec(),
        df=pd.read_csv(l1_path),
        context=context,
    )
    assert artifact.schema.columns == (
        "l1_l2_method",
        "l2_method",
        ASOCC_SSP_SCENARIO_COLUMN,
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        "r_p",
    )
    assert artifact.schema.year_columns == ("2020",)

    assert (
        run_write_mod._count_regression_output_targets(  # noqa: SLF001
            context=_context(tmp_path, output_source_label=None),
            state=_state(),
        )
        == 0
    )

    prior_output = str(tmp_path / "prior.csv")
    context_with_append_metadata = _context(
        tmp_path,
        metadata_completed_years=[2019],
        metadata_prior_outputs=[prior_output],
    )
    run_state = _state()
    run_state.l1_results_by_ssp_scenario = {None: {_l1_output_spec(): [_frame()]}}
    run_state.l2_results_by_ssp_scenario = {None: {_output_spec(bucket="l2_vs_global"): [_frame()]}}
    run_state.processed_years = [2020]

    run_write_mod._write_outputs(  # noqa: SLF001
        context=context_with_append_metadata,
        state=run_state,
        refresh=True,
        write_metadata=True,
        show_progress=True,
        progress_label=" progress ",
        progress_prefix=" prefix ",
    )

    metadata_path = _get_allocate_run_metadata_path(
        tmp_path,
        source="oecd_v2025",
        group_version=None,
    )
    assert metadata_path.exists()
    metadata = _load_run_metadata(metadata_path)
    assert metadata["execution"]["completed_years"] == [2019, 2020]
    assert metadata["artifacts"]["outputs"][0] == prior_output
    assert run_state.write_progress_total == 2
    assert run_state.write_progress_current == 2
    assert run_state.write_progress_label == "progress"
    assert run_state.write_progress_prefix == "prefix"

    fresh_state = _state()
    fresh_state.l1_results_by_ssp_scenario = {None: {_l1_output_spec(): [_frame()]}}
    fresh_state.l2_results_by_ssp_scenario = {
        None: {_output_spec(bucket="l2_vs_global"): [_frame()]}
    }
    fresh_state.processed_years = [2020]
    run_write_mod._write_outputs(  # noqa: SLF001
        context=context,
        state=fresh_state,
        refresh=True,
        write_metadata=True,
        show_progress=False,
    )
    fresh_metadata = _load_run_metadata(metadata_path)
    assert fresh_metadata["execution"]["completed_years"] == [2020]
