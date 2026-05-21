from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd

from pyaesa.asocc.io.metadata import RunContext, RunState
from pyaesa.asocc.methods.registry.registry import REGISTRY
from pyaesa.asocc.orchestration.projection.config.types import ProjectionContext
from pyaesa.asocc.orchestration.yearly.shared.scenario_routing import (
    build_pr_hr_rp1_zero_fallback_recorder,
    emit_notice,
    flush_pr_hr_rp1_zero_fallback_notices,
    is_historical_reuse_l2_projection,
    is_regression_projection_year,
    is_scenario_dependent_l1,
    is_scenario_dependent_l2_projection,
    l2_projection_subfolder,
    record_pr_hr_rp1_zero_fallback_notice,
    regression_projection_subfolder_for_context,
    resolve_output_ssp_scenario,
)


class _RecorderLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str) -> None:
        self.messages.append(str(message))


def _run_context(
    *,
    wb_columns: list[object],
    projection_context: ProjectionContext | None = None,
) -> RunContext:
    logger = _RecorderLogger()
    wb_df = pd.DataFrame(columns=wb_columns)
    ssp_df = pd.DataFrame()
    return RunContext(
        project_name="scenario_routing_demo",
        source="oecd_v2025",
        fu_code="L2.a.a",
        group_version=None,
        group_version_reg=None,
        group_reg=False,
        group_sec=False,
        lcia_method=None,
        years_input=[2005, 2030],
        reference_years_input=None,
        ssp_scenario="SSP2",
        is_exio=False,
        l1_lcia_kind="CBA_FD",
        lcia_methods=["gwp100_lcia"],
        selected_l1=["PR(GDPcap)", "AR(E^{CBA_FD})"],
        combined=[("UT(FD)", "PR(GDPcap)")],
        selected_l2_one_step=["UT(FD)", "UT(GVAa)"],
        required_indices=set(),
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": None, "r_u": None},
        studied_indices_tag="demo",
        proj_base=Path("scenario_routing_demo"),
        logger=logger,
        requested_years=[2005, 2030],
        resolved_years=[2005, 2030],
        persisted_years=[2005, 2030],
        historical_years=[2005, 2006],
        reference_years=None,
        ssp_scenario_options=["SSP2"],
        run_signature={"selected_methods": {"L1": ["PR(GDPcap)"]}},
        needs_lcia=True,
        repo_root=Path("scenario_routing_demo"),
        wb_df=wb_df,
        ssp_df=ssp_df,
        wb_df_raw=wb_df.copy(),
        ssp_df_raw=ssp_df.copy(),
        selected_methods={"L1": ["PR(GDPcap)"], "L2": ["UT(FD)"]},
        l1_kinds_needed=set(),
        l1_only_no_mrio=False,
        l1_reg_aggreg="post",
        use_original_l1_post_domain=False,
        variant_tag=None,
        aggreg_indices=False,
        output_format="csv",
        intermediate_outputs=True,
        output_source_label="oecd_v2025",
        projection_context=projection_context,
        ssp_scenario_options_by_year={2030: ["SSP2"]},
    )


def _projection_context(
    *,
    mode: str,
    reg_window: tuple[int, int] | None,
    route_by_name: dict[str, str],
) -> SimpleNamespace:
    return SimpleNamespace(
        projection_context=ProjectionContext(
            enabled=True,
            mode=mode,
            max_historical_year=2006,
            future_years=(2030,),
            reg_window=reg_window,
            l2_reuse_years=(2005, 2006),
            ut_methods_in_scope=tuple(route_by_name),
            l2_method_route_by_name=route_by_name,
        )
    )


def test_resolve_output_ssp_scenario_and_l1_dependency_contracts() -> None:
    context = _run_context(wb_columns=[2005, 2006])

    assert (
        resolve_output_ssp_scenario(
            context=context,
            year=2005,
            ssp_scenario="SSP2",
            scenario_dependent=False,
        )
        is None
    )
    assert (
        resolve_output_ssp_scenario(
            context=context,
            year=2005,
            ssp_scenario="SSP2",
            scenario_dependent=True,
        )
        is None
    )
    assert (
        resolve_output_ssp_scenario(
            context=context,
            year=2030,
            ssp_scenario="SSP2",
            scenario_dependent=True,
        )
        == "SSP2"
    )
    assert is_scenario_dependent_l1(None) is False
    assert is_scenario_dependent_l1("AR(E^{CBA_FD})") is False
    assert is_scenario_dependent_l1("PR(GDPcap)") == any(
        spec.needs_pop or spec.needs_gdp for spec in REGISTRY.get_method("PR(GDPcap)", level="L1")
    )


def test_pr_hr_notice_cover_dedup_and_flush() -> None:
    context = _run_context(wb_columns=[2005, 2006])
    state = RunState()
    context_logger = cast(_RecorderLogger, context.logger)

    flush_pr_hr_rp1_zero_fallback_notices(context=context, state=state)

    emit_notice(context=context, state=state, key="demo", message="first warning")
    emit_notice(context=context, state=state, key="demo", message="second warning")
    assert state.notices_emitted == {"demo"}
    assert len(context_logger.messages) == 1

    pending_key = build_pr_hr_rp1_zero_fallback_recorder(
        state=state,
        l1_method="PR-HR(Ecap,cum)",
        lcia_kind="CBA_FD",
        use_original_domain=True,
        ssp_scenario="SSP2",
    )
    pending_key(["impact_b", "impact_a"], 2030, 2006)
    record_pr_hr_rp1_zero_fallback_notice(
        state=state,
        l1_method="PR-HR(Ecap,cum)",
        lcia_kind="PBA",
        impacts=["impact_a", "impact_b"],
        target_year=2030,
        fallback_year=2006,
        use_original_domain=True,
        ssp_scenario="SSP2",
    )

    flush_pr_hr_rp1_zero_fallback_notices(context=context, state=state)

    assert state.pr_hr_rp1_zero_fallback_pending == {}
    assert len(context_logger.messages) == 2


def test_projection_cover_all_reachable_routes_and_subfolders() -> None:
    disabled_context = SimpleNamespace(
        projection_context=ProjectionContext(
            enabled=False,
            mode=None,
            max_historical_year=2006,
            future_years=(),
            reg_window=None,
            l2_reuse_years=(),
            ut_methods_in_scope=(),
            l2_method_route_by_name={},
        )
    )
    regression_context = _projection_context(
        mode="regression",
        reg_window=(2005, 2006),
        route_by_name={"UT(FD)": "regression", "UT(GVAa)": "historical_reuse"},
    )
    regression_default_context = _projection_context(
        mode="regression",
        reg_window=None,
        route_by_name={"UT(FD)": "regression"},
    )
    historical_context = _projection_context(
        mode="historical_reuse",
        reg_window=(2005, 2006),
        route_by_name={"UT(FD)": "historical_reuse"},
    )

    assert (
        is_scenario_dependent_l2_projection(
            context=disabled_context,
            year=2030,
            l2_method="UT(FD)",
        )
        is False
    )
    assert (
        is_scenario_dependent_l2_projection(
            context=regression_context,
            year=2006,
            l2_method="UT(FD)",
        )
        is False
    )
    assert (
        is_scenario_dependent_l2_projection(
            context=regression_context,
            year=2030,
            l2_method="UT(FD)",
        )
        is True
    )
    assert (
        is_scenario_dependent_l2_projection(
            context=regression_context,
            year=2030,
            l2_method="UT(GVAa)",
        )
        is False
    )
    assert (
        is_scenario_dependent_l2_projection(
            context=historical_context,
            year=2030,
            l2_method="UT(FD)",
        )
        is False
    )

    assert (
        is_historical_reuse_l2_projection(
            context=disabled_context,
            year=2030,
            l2_method="UT(FD)",
        )
        is False
    )
    assert (
        is_historical_reuse_l2_projection(
            context=regression_context,
            year=2030,
            l2_method="UT(FD)",
        )
        is False
    )
    assert (
        is_historical_reuse_l2_projection(
            context=historical_context,
            year=2030,
            l2_method="UT(FD)",
        )
        is True
    )
    assert (
        is_historical_reuse_l2_projection(
            context=historical_context,
            year=2006,
            l2_method="UT(FD)",
        )
        is False
    )

    assert (
        l2_projection_subfolder(
            context=disabled_context,
            year=2030,
            l2_method="UT(FD)",
            bucket="l2_vs_global",
        )
        is None
    )
    assert (
        l2_projection_subfolder(
            context=regression_context,
            year=2006,
            l2_method="UT(FD)",
            bucket="l2_vs_global",
        )
        is None
    )
    assert (
        l2_projection_subfolder(
            context=regression_context,
            year=2030,
            l2_method="UT(FD)",
            bucket="l2_vs_global",
        )
        == "regression_proj"
    )
    assert (
        l2_projection_subfolder(
            context=regression_default_context,
            year=2030,
            l2_method="UT(FD)",
            bucket="l2_vs_global",
        )
        == "regression_proj"
    )
    assert (
        l2_projection_subfolder(
            context=regression_context,
            year=2030,
            l2_method="UT(FD)",
            bucket="utility_propagation_contrib",
        )
        == "regression_proj"
    )
    assert (
        l2_projection_subfolder(
            context=regression_context,
            year=2030,
            l2_method="UT(FD)",
            bucket="l2_in_l1",
        )
        == "regression_proj"
    )
    assert (
        l2_projection_subfolder(
            context=historical_context,
            year=2030,
            l2_method="UT(FD)",
            bucket="l2_vs_global",
        )
        == "historical_reuse"
    )
    assert (
        l2_projection_subfolder(
            context=historical_context,
            year=2030,
            l2_method="UT(FD)",
            bucket="l2_in_l1",
        )
        is None
    )
    assert (
        l2_projection_subfolder(
            context=regression_context,
            year=2030,
            l2_method="missing",
            bucket="l2_vs_global",
        )
        is None
    )
    assert (
        l2_projection_subfolder(
            context=regression_context,
            year=2030,
            l2_method="UT(FD)",
            bucket="unknown",
        )
        is None
    )

    assert is_regression_projection_year(context=disabled_context, year=2030) is False
    assert is_regression_projection_year(context=historical_context, year=2030) is False
    assert is_regression_projection_year(context=regression_context, year=2030) is True

    assert (
        regression_projection_subfolder_for_context(context=disabled_context) == "regression_proj"
    )
    assert (
        regression_projection_subfolder_for_context(context=regression_default_context)
        == "regression_proj"
    )
    assert (
        regression_projection_subfolder_for_context(context=regression_context) == "regression_proj"
    )
