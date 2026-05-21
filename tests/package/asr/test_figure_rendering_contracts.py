from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NEGATIVE,
    CC_FLOW_POSITIVE,
    GROSS_ALT_KYOTO_WO_AFOLU,
    SEQUESTRATION_SUBTOTAL,
)
from pyaesa.ar6_cc.uncertainty.evaluation.summary_identity import ar6_cc_summary_identity_groups
from pyaesa.asr.figures.axis import (
    ASR_LOG_SCALE,
    ASR_NORMAL_SCALE,
    apply_asr_log_axis,
    apply_frequency_axis,
    normal_asr_tick_text,
    normal_asr_ticks,
    positive_asr_values,
    resolve_asr_normal_limits,
    resolve_asr_scale_mode,
)
from pyaesa.asr.figures.frequency import (
    CUMULATIVE_FNT_FRACTION_COLUMN,
    FNT_FRACTION_COLUMN,
    fnt_legend_entry,
    format_fnt_math_label,
    format_fnt_percent,
    render_fnt_box,
    render_fnt_box_groups,
    render_fnt_boxes,
)
from pyaesa.asr.figures.dynamic_global_ar6 import (
    deterministic_global_ar6_source,
    uncertainty_global_ar6_source,
)
from pyaesa.asr.figures.common import (
    VALUE_ARRAY_COLUMN,
    apply_scaled_asr_axis_policy,
    asr_scope_stem,
    asr_scope_title,
    component_axis_limits,
    data_linear_limits,
    dynamic_linear_limits,
    format_acc_lca_component_axis,
)
from pyaesa.asr.figures.component_legend import ar6_cc_positive_flow_label, render_acc_lca_row_key
from pyaesa.asr.figures.polar import render_asr_polar
from pyaesa.asr.figures.polar_artists import (
    render_risk_background,
    render_threshold_arcs,
    render_uncertainty_glyph,
    risk_rgba,
    risk_scale_rgba,
    violin_density_peak,
)
from pyaesa.asr.figures.polar_layout import (
    render_bottom_legend,
    render_impact_labels,
    render_polar_title,
    render_polar_tick_marks,
)
from pyaesa.asr.figures.threshold_contract import has_max_asr_threshold
from pyaesa.asr.figures.transitions import (
    ASR_ASOCC_TRANSITION_LABEL,
    ASR_LCA_TRANSITION_LABEL,
    GENERIC_ASR_TRANSITION_LABEL,
    asr_transition_markers,
)
from pyaesa.shared.figures.multi_year_transitions import render_transition_markers
from pyaesa.asr.deterministic.figures.component_diagnostics import DeterministicComponentRows
from pyaesa.asr.deterministic.figures.render import (
    _min_variant_note,
    _plot_scope,
    _prepare_plot_rows,
    _render_dynamic_global_ar6_row,
    _scope_stem,
)
from pyaesa.asr.uncertainty.figures.component_data import ComponentDiagnosticRows
from pyaesa.asr.uncertainty.figures.product_renderers import (
    _render_component_mean_axis,
    plot_band_scope,
    plot_mean_line_scope,
)
from pyaesa.asr.uncertainty.figures.render import _plan_polar_checkpoint_jobs
from pyaesa.asr.uncertainty.figures.scope_planner import FigureContext
from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyRunPaths
from pyaesa.asr.uncertainty.figures.violin_renderers import plot_violin_scope
from pyaesa.shared.figures.dynamic_ar6 import (
    MODEL_SCENARIO_PAIR_COUNT_COLUMN,
    MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
)
from pyaesa.shared.runtime.metadata.json import write_json_dict
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
    EXT_LCA_SSP_SCENARIO_COLUMN,
    LCA_SSP_START_YEAR_COLUMN,
)
from pyaesa.shared.uncertainty_assessment.io.tables import (
    CompactRunMatrixWriter,
    SparseRunRows,
    SparseRunRowsWriter,
    write_uncertainty_table,
)
from pyaesa.shared.uncertainty_assessment.io.public_summary import (
    exact_summary_from_public_runs,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import build_manifest, write_manifest


def test_asr_axis_and_frequency_contracts_cover_public_labels(project_repo: Path) -> None:
    del project_repo
    assert resolve_asr_scale_mode(np.array([], dtype=float)) == ASR_NORMAL_SCALE
    assert resolve_asr_scale_mode(np.array([0.0, 0.0])) == ASR_NORMAL_SCALE
    assert resolve_asr_scale_mode(np.array([0.5, 10.0])) == ASR_NORMAL_SCALE
    assert resolve_asr_scale_mode(np.array([0.0, 5.0])) == ASR_NORMAL_SCALE
    assert resolve_asr_scale_mode(np.array([0.5, 10.0001])) == ASR_LOG_SCALE
    assert resolve_asr_scale_mode(np.array([1e-6, 20.0])) == ASR_LOG_SCALE
    with pytest.warns(UserWarning):
        assert resolve_asr_scale_mode(np.array([0.0, 20.0])) == ASR_NORMAL_SCALE
    assert resolve_asr_scale_mode(np.array([-0.1, 20.0])) == ASR_NORMAL_SCALE
    positive, zero_count = positive_asr_values(np.array([0.0, 0.2, 1.0]), context="demo")
    np.testing.assert_allclose(positive, [0.2, 1.0])
    assert zero_count == 1

    fig, axis = plt.subplots()
    try:
        omitted = apply_asr_log_axis(axis, values=np.array([0.01, 1.0, 12.0]), context="demo")
        assert omitted == 0
        tick_values = set(axis.get_yticks().round(12).tolist())
        assert 1.0 in tick_values
    finally:
        plt.close(fig)
    fig, axis = plt.subplots()
    try:
        apply_scaled_asr_axis_policy(
            axis,
            values=np.array([2.2, 8.4], dtype=float),
            frame=pd.DataFrame({"__asr_max_threshold": [np.nan]}),
            scale_mode=ASR_NORMAL_SCALE,
        )
        lower, upper = axis.get_ylim()
        assert lower < 1.0 < upper
        assert axis.yaxis.get_major_formatter()(1.0, None) == "1e+00"
    finally:
        plt.close(fig)
    fig, axis = plt.subplots()
    try:
        apply_frequency_axis(axis)
        render_fnt_box(axis, x=0.5, value=0.25)
        fitted_y = float(axis.texts[-1].get_position()[1])
        render_fnt_boxes(axis, entries=[(0.6, 0.5), (0.7, 0.75)], y=fitted_y)
        assert axis.get_ylabel() == r"$f^{\mathrm{NT}}$"
        assert axis.get_ylim() == (-5.0, 105.0)
    finally:
        plt.close(fig)

    fig_a, axis_a = plt.subplots()
    fig_b, axis_b = plt.subplots()
    try:
        render_fnt_box_groups([(axis_a, [])])
        with pytest.raises(ValueError, match="same figure"):
            render_fnt_box_groups([(axis_a, [(0.5, 0.25)]), (axis_b, [(0.5, 0.25)])])
    finally:
        plt.close(fig_a)
        plt.close(fig_b)
    run_root = Path("unused")
    context = FigureContext(
        manifest=build_manifest(
            family="asr",
            mode="fixed",
            output_format="csv_compact",
            active_sources=(),
        ),
        paths=ASRUncertaintyRunPaths(
            run_root=run_root,
            public_row_identity=run_root / "results" / "identity.csv",
            public_runs=run_root / "results" / "runs.csv",
            summary_stats_runs=run_root / "results" / "summary.csv",
            cumulative_row_identity=run_root / "results" / "cumulative_identity.csv",
            cumulative_runs=run_root / "results" / "cumulative_runs.csv",
            cumulative_summary_stats_runs=run_root / "results" / "cumulative_summary.csv",
            results_readme=run_root / "results" / "README.txt",
            source_methods=run_root / "logs" / "source_methods.csv",
            sobol_indices=run_root / "results" / "sobol" / "indices.csv",
            sobol_source_summary=run_root / "results" / "sobol" / "summary.csv",
            sobol_readme=run_root / "results" / "sobol" / "README.txt",
            scope_manifest=run_root / "logs" / "scope_manifest.json",
        ),
        figures_root=run_root / "figures",
        requested_years=(2020,),
        requested_asocc_ssps=("SSP2",),
        fu_code="L2.a.a",
        output_format="csv_compact",
        figure_output_format="svg",
        figure_dpi=10,
        per_method=True,
        multi_method=True,
        inter_method=True,
        active_sources=(),
        run_layout="compact_run_matrix",
        dynamic_category_uncertainty_active=False,
        polar_years=(2020,),
        polar_style="violin",
    )
    dynamic_jobs = _plan_polar_checkpoint_jobs(
        rows=pd.DataFrame(
            {
                "__method": ["UT(FD)"],
                "cc_type": ["dynamic_ar6"],
                "lcia_method": ["gwp100_lcia"],
                "impact": ["GWP_100"],
                "impact_unit": ["kg CO2-eq"],
                "year": [2020],
                "mean": [1.0],
            }
        ),
        context=context,
        role="mean",
        label=None,
        title_label=None,
        family_label="ASR",
        scale_modes={"UT(FD)": ASR_NORMAL_SCALE},
    )
    assert dynamic_jobs == []

    fig, (axis, budget_axis) = plt.subplots(ncols=2)
    try:
        negative_color = "#E68613"
        visible_negative = _render_dynamic_global_ar6_row(
            axis=axis,
            budget_axis=budget_axis,
            scoped_frame=pd.DataFrame(
                {
                    "cc_category": ["C1", "C1"],
                    "cc_flow": [CC_FLOW_POSITIVE, CC_FLOW_NEGATIVE],
                    "cc_model": ["model", "model"],
                    "cc_scenario": ["scenario", "scenario"],
                    "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
                    2020: [3.0, 0.0],
                    2021: [4.0, -1.0],
                }
            ),
            study_years=[2020, 2021],
            post_years=[],
            category_colors={"C1": "#ff7f0e"},
            flow_colors={CC_FLOW_POSITIVE: "#ff7f0e", CC_FLOW_NEGATIVE: negative_color},
            pathway_title="Pathways",
            budget_title="Budget",
            show_x_labels=True,
            show_study_label=False,
            title_pad=6,
        )
        assert visible_negative is True
        assert {line.get_linestyle() for line in axis.lines} == {"-"}
        negative_patches = [patch for patch in budget_axis.patches if patch.get_height() < 0.0]
        assert len(negative_patches) == 1
        assert negative_patches[0].get_hatch() in (None, "")
        assert negative_patches[0].get_facecolor()[3] > 0.0
        assert negative_patches[0].get_facecolor()[:3] == pytest.approx(
            mcolors.to_rgba(negative_color)[:3]
        )
        assert any(label.get_text() for label in axis.get_xticklabels())
    finally:
        plt.close(fig)

    fig, axis = plt.subplots()
    try:
        format_acc_lca_component_axis(
            axis=axis,
            frame=pd.DataFrame(
                {"impact_unit": ["kg CO2eq/yr kg CO2-eq / year m3 km3 Mm3 yr-1 m-2 H+ eq"]}
            ),
            years=[2020, 2021],
            show_x_labels=True,
            title="Component unit",
            limits=(0.1, 10.0),
        )
        assert (
            axis.get_ylabel() == r"kg CO$_2$eq yr$^{-1}$ kg CO$_2$-eq year$^{-1}$ "
            r"m$^3$ km$^3$ Mm$^3$ yr$^{-1}$ m$^{-2}$ H$^+$ eq"
        )
    finally:
        plt.close(fig)

    np.testing.assert_allclose(normal_asr_ticks(lower=1.0, upper=1.0), [1.0])
    np.testing.assert_allclose(
        normal_asr_ticks(lower=0.94, upper=1.06),
        [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06],
    )
    np.testing.assert_allclose(
        normal_asr_ticks(lower=0.8, upper=1.2),
        [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2],
    )
    np.testing.assert_allclose(
        normal_asr_ticks(lower=0.5, upper=1.5),
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    )
    assert normal_asr_tick_text(1.0) == "1"
    assert normal_asr_tick_text(1.5) == "1.5"
    assert data_linear_limits(np.array([], dtype=float)) == (0.0, 1.0)
    assert data_linear_limits(np.array([2.0, 2.0], dtype=float)) == (1.76, 2.24)
    assert dynamic_linear_limits(np.array([], dtype=float)) == (0.0, 1.0)
    assert dynamic_linear_limits(np.array([0.0, 0.0], dtype=float)) == (0.0, 1.0)
    assert dynamic_linear_limits(np.array([-2.0, -2.0], dtype=float)) == (-2.24, -1.76)
    assert dynamic_linear_limits(np.array([-2.0, -1.0], dtype=float)) == (-2.12, -0.88)
    assert resolve_asr_normal_limits(np.array([1.02, 1.04], dtype=float))[0] == 0.98
    assert resolve_asr_normal_limits(np.array([0.2, 2.0], dtype=float))[0] == 0.0
    assert resolve_asr_normal_limits(np.array([-0.5, 2.0], dtype=float))[0] < 0.0
    assert (
        component_axis_limits(
            values=np.array([0.2, 2.0], dtype=float),
            scale_mode=ASR_LOG_SCALE,
        )[0]
        > 0.0
    )

    assert format_fnt_percent(0.0) == "0%"
    assert format_fnt_percent(1.0) == "100%"
    assert format_fnt_percent(0.375) == "37.5%"
    assert format_fnt_math_label(0.25) == r"$f^{\mathrm{NT}}=25.0\%$"
    _handle, label = fnt_legend_entry(cc_source="pb_lcia")
    assert "frequency of no-transgression" in label
    assert has_max_asr_threshold(frame=None) is False


def test_asr_scope_file_stem_keeps_selector_and_impact_tokens(project_repo: Path) -> None:
    del project_repo
    frame = pd.DataFrame(
        {
            "lcia_method": ["pb_lcia"],
            "impact": ["AAL"],
            "asocc_ssp_scenario": ["SSP2"],
            "cc_category": ["C1"],
            "cc_bound": ["min_cc"],
        }
    )

    stem = asr_scope_stem(
        "multi_method",
        frame,
        product="frequency_of_no_transgression",
        selector_token="rp_FR__sp_D",
        include_impact=True,
        studied_year=2030,
    )

    expected = (
        "multi_method__frequency_of_no_transgression__rp_FR__sp_D__"
        "pb_lcia__AAL__SSP2__C1__min_cc__2030"
    )
    assert stem == expected

    dynamic_frame = pd.DataFrame(
        {
            "lcia_method": ["gwp100_lcia"],
            "impact": ["GWP_100"],
            "ar6_cc_ssp_scenario": ["SSP2"],
            "cc_category": ["C2"],
            "cc_model": ["REMIND 1"],
            "cc_scenario": ["Low Energy Demand"],
        }
    )
    assert (
        asr_scope_stem(
            "multi_method",
            dynamic_frame,
            include_impact=True,
        )
        == "multi_method__gwp100_lcia__GWP_100__SSP2__C2__REMIND_1_Low_Energy_Demand"
    )
    assert asr_scope_title("ASR", None, dynamic_frame, include_impact=False) == (
        "ASR | Climate change (GWP_100) | SSP2\n"
        "AR6 category: C2 | Model-scenario pair: REMIND 1 / Low Energy Demand"
    )
    assert "AR6 categories: C1-C2" in asr_scope_title(
        "ASR",
        None,
        pd.DataFrame(
            {
                "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
                "impact": ["GWP_100", "GWP_100"],
                "cc_category": ["C1", "C2"],
            }
        ),
        include_impact=False,
    )


def test_asr_transition_markers_cover_lca_and_asocc_boundaries() -> None:
    def frame(*, routes: list[str], lca_scenarios: list[str]) -> pd.DataFrame:
        years = [2020 + index for index in range(len(routes))]
        return pd.DataFrame(
            {
                "year": years,
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: routes,
                EXT_LCA_SSP_SCENARIO_COLUMN: lca_scenarios,
            }
        )

    same = asr_transition_markers(frame(routes=["", "regression_proj"], lca_scenarios=["", "SSP2"]))
    asocc_only = asr_transition_markers(
        frame(routes=["", "regression_proj"], lca_scenarios=["", ""])
    )
    lca_only = asr_transition_markers(frame(routes=["", ""], lca_scenarios=["", "SSP2"]))
    split = asr_transition_markers(
        frame(routes=["", "", "regression_proj"], lca_scenarios=["", "SSP2", "SSP2"])
    )
    all_lca_prospective = asr_transition_markers(
        frame(routes=["", ""], lca_scenarios=["SSP2", "SSP2"])
    )
    lca_start_inside = asr_transition_markers(
        pd.DataFrame(
            {
                "year": [2020, 2021, 2022],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["", "", ""],
                LCA_SSP_START_YEAR_COLUMN: [2021, 2021, 2021],
            }
        )
    )
    lca_start_missing = asr_transition_markers(
        pd.DataFrame(
            {
                "year": [2020, 2021],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["", ""],
                LCA_SSP_START_YEAR_COLUMN: [np.nan, np.nan],
            }
        )
    )
    lca_start_outside = asr_transition_markers(
        pd.DataFrame(
            {
                "year": [2020, 2021],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["", ""],
                LCA_SSP_START_YEAR_COLUMN: [2025, 2025],
            }
        )
    )
    duplicate_component_rows = asr_transition_markers(
        pd.DataFrame(
            {
                "year": [2020, 2020, 2021, 2021, 2022, 2022],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [
                    "",
                    "",
                    "",
                    "",
                    "regression_proj",
                    "",
                ],
                EXT_LCA_SSP_SCENARIO_COLUMN: ["", "", "SSP2", "", "SSP2", ""],
            }
        )
    )

    assert [(marker.year, marker.label) for marker in same] == [
        (2021, GENERIC_ASR_TRANSITION_LABEL)
    ]
    assert [(marker.year, marker.label) for marker in asocc_only] == [
        (2021, GENERIC_ASR_TRANSITION_LABEL)
    ]
    assert [(marker.year, marker.label) for marker in lca_only] == [
        (2021, GENERIC_ASR_TRANSITION_LABEL)
    ]
    assert [(marker.year, marker.label) for marker in split] == [
        (2022, ASR_ASOCC_TRANSITION_LABEL),
        (2021, ASR_LCA_TRANSITION_LABEL),
    ]
    assert all_lca_prospective == []
    assert [(marker.year, marker.label) for marker in lca_start_inside] == [
        (2021, GENERIC_ASR_TRANSITION_LABEL)
    ]
    assert lca_start_missing == []
    assert lca_start_outside == []
    assert [(marker.year, marker.label) for marker in duplicate_component_rows] == [
        (2022, ASR_ASOCC_TRANSITION_LABEL),
        (2021, ASR_LCA_TRANSITION_LABEL),
    ]

    fig, axis = plt.subplots()
    try:
        axis.plot([2020, 2021], [1.0, 2.0])
        render_transition_markers(axis, markers=split)
        labels = [text.get_text() for text in axis.texts]
        assert labels == ["Prospective start year", "LCA", "aSoCC"]
    finally:
        plt.close(fig)


def test_asr_component_row_key_sits_under_component_row() -> None:
    assert ar6_cc_positive_flow_label(emissions_mode="gross_alt") == "gross_alt emissions"
    assert ar6_cc_positive_flow_label(emissions_mode="gross") == "gross emissions"
    assert ar6_cc_positive_flow_label(emissions_mode="net") == "net emissions"
    fig, axes = plt.subplots(nrows=2, ncols=2)
    try:
        render_acc_lca_row_key(
            fig=fig,
            left_axis=axes[0, 0],
            right_axis=axes[0, 1],
            include_method_in_label=True,
        )
        labels = [text.get_text() for text in fig.texts]
        assert labels == ["LCA", "Other colors: aCC"]
        key_y = fig.texts[0].get_position()[1]
        component_bottom = axes[0, 0].get_position().y0
        below_top = axes[1, 0].get_position().y1
        assert below_top < key_y < component_bottom
        assert component_bottom - key_y < key_y - below_top
    finally:
        plt.close(fig)


def test_asr_polar_artist_primitives_cover_valid_styles() -> None:
    assert len(risk_scale_rgba(0.0, alpha=0.8, lighten=0.0)) == 4
    assert len(risk_scale_rgba(0.5, alpha=0.8, lighten=0.1)) == 4
    assert len(risk_rgba(20.0, 10.0, alpha=1.0, lighten=0.0)) == 4
    assert violin_density_peak(np.array([1.0])) == 0.0
    assert violin_density_peak(np.array([1.0, 1.0])) == 0.0
    assert violin_density_peak(np.array([0.0, 1.0])) > 0.0
    assert violin_density_peak(np.arange(1000, dtype=float)) > 0.0
    assert violin_density_peak(np.array([0.0, 0.3, 0.7, 1.2])) > 0.0

    fig, axis = plt.subplots(figsize=(4.0, 4.0), subplot_kw={"projection": "polar"})
    try:
        axis.set_ylim(-2.0, 2.0)
        theta_bounds = np.array([0.0, np.pi, 2.0 * np.pi])
        render_threshold_arcs(
            axis,
            theta_bounds=theta_bounds,
            max_radii=np.array([0.5, 1.0]),
            has_max_threshold=False,
            scale_mode=ASR_LOG_SCALE,
        )
        render_threshold_arcs(
            axis,
            theta_bounds=theta_bounds,
            max_radii=np.array([0.5, 1.0]),
            has_max_threshold=True,
            scale_mode=ASR_LOG_SCALE,
        )
        render_risk_background(
            axis,
            theta0=0.0,
            theta1=np.pi,
            r_min=-1.0,
            r_end=1.1,
            max_ratio=5.0,
            scale_mode=ASR_LOG_SCALE,
        )
        render_risk_background(
            axis,
            theta0=np.pi,
            theta1=2.0 * np.pi,
            r_min=-2.0,
            r_end=-0.4,
            max_ratio=5.0,
            scale_mode=ASR_LOG_SCALE,
        )
        render_risk_background(
            axis,
            theta0=np.pi,
            theta1=2.0 * np.pi,
            r_min=1.0,
            r_end=2.0,
            max_ratio=5.0,
            scale_mode=ASR_LOG_SCALE,
        )
        fig.canvas.draw()
        summary = {
            "mean": 1.6,
            "median": 1.5,
            "p5": 0.8,
            "p25": 1.0,
            "p75": 2.0,
            "p95": 3.0,
        }
        render_uncertainty_glyph(
            axis,
            theta_mid=0.4,
            sector_width=0.7,
            radial_payload=np.log10(np.array([0.8, 1.0, 1.5, 2.0, 3.0])),
            summary=summary,
            max_ratio=5.0,
            density_scale=1.0,
            style="violin",
            scale_mode=ASR_LOG_SCALE,
        )
        render_uncertainty_glyph(
            axis,
            theta_mid=1.4,
            sector_width=0.7,
            radial_payload=np.log10(np.array([0.8, 1.0, 1.5, 2.0, 3.0])),
            summary=summary,
            max_ratio=5.0,
            density_scale=1.0,
            style="whisker",
            scale_mode=ASR_LOG_SCALE,
        )
        constant_summary = {column: 1.0 for column in ("mean", "median", "p5", "p25", "p75", "p95")}
        render_uncertainty_glyph(
            axis,
            theta_mid=2.4,
            sector_width=0.7,
            radial_payload=np.array([0.0, 0.0]),
            summary=constant_summary,
            max_ratio=5.0,
            density_scale=1.0,
            style="violin",
            scale_mode=ASR_LOG_SCALE,
        )
        render_uncertainty_glyph(
            axis,
            theta_mid=3.4,
            sector_width=0.7,
            radial_payload=np.array([0.0, 0.0]),
            summary=constant_summary,
            max_ratio=5.0,
            density_scale=1.0,
            style="whisker",
            scale_mode=ASR_LOG_SCALE,
        )
    finally:
        plt.close(fig)


def test_asr_polar_layout_and_renderer_contracts(project_repo: Path, tmp_path: Path) -> None:
    del project_repo
    fig, axis = plt.subplots(figsize=(4.0, 4.0), subplot_kw={"projection": "polar"})
    try:
        axis.set_ylim(-1.0, 1.0)
        theta_bounds = np.array([0.0, np.pi, 2.0 * np.pi])
        render_impact_labels(
            fig,
            axis,
            theta_bounds=theta_bounds,
            labels=["left", "right"],
            frequency_labels=["0%", "100%"],
            r_max=1.0,
        )
        render_polar_tick_marks(
            axis,
            theta_bounds=theta_bounds,
            r_min=-1.0,
            r_max=1.0,
            scale_mode=ASR_LOG_SCALE,
        )
        axis.text(0.0, 1.1, "")
        hidden = axis.text(0.1, 1.1, "hidden")
        hidden.set_visible(False)
        render_polar_title(fig, axis, title="Measured polar title")
        fig.canvas.draw()
        renderer = cast(Any, fig.canvas).get_renderer()
        title_artist = fig.texts[-1]
        title_bottom = (
            title_artist.get_window_extent(renderer=renderer)
            .transformed(fig.transFigure.inverted())
            .y0
        )
        content_top = max(
            artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted()).y1
            for artist in [*axis.texts, *fig.texts[:-1]]
            if artist.get_visible() and str(artist.get_text()).strip()
        )
        assert title_bottom - content_top == pytest.approx(0.22 / 4.0, abs=0.02)
    finally:
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(1.0, 1.0), subplot_kw={"projection": "polar"})
    try:
        axis.set_ylim(0.0, 1.0)
        crowded_bounds = np.linspace(0.0, 2.0 * np.pi, 9)
        render_impact_labels(
            fig,
            axis,
            theta_bounds=crowded_bounds,
            labels=[f"crowded impact label {index} with extended text" for index in range(8)],
            frequency_labels=[None] * 8,
            r_max=1.0,
        )
    finally:
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(4.0, 4.0), subplot_kw={"projection": "polar"})
    try:
        render_polar_tick_marks(
            axis,
            theta_bounds=np.array([0.0, np.pi, 2.0 * np.pi]),
            r_min=0.0,
            r_max=6.2,
            scale_mode=ASR_NORMAL_SCALE,
        )
        labels = {text.get_text() for text in axis.texts}
        assert "1" in labels
        assert "0" not in labels
    finally:
        plt.close(fig)

    for style in ("deterministic", "violin", "whisker"):
        fig = plt.figure(figsize=(5.0, 2.0))
        try:
            render_bottom_legend(
                fig,
                label_bottom_y=0.20,
                style=style,
                min_label="Min SOS",
                max_label="Max SOS",
                lower_zone_label="Safe operating space",
                middle_zone_label="Zone of increasing risk",
                upper_zone_label="High risk zone",
                fnt_label="fNT",
                deterministic_note="Min variant compression: ref_year=2005.",
            )
            if style == "deterministic":
                render_bottom_legend(
                    fig,
                    label_bottom_y=0.20,
                    style=style,
                    min_label="Min SOS",
                    max_label=None,
                    lower_zone_label="Safe operating space",
                    middle_zone_label=None,
                    upper_zone_label="High risk zone",
                    fnt_label="fNT",
                )
        finally:
            plt.close(fig)

    pb_frame = pd.DataFrame(
        {
            "impact": ["AAL", "SOD", "N"],
            "cc_bound": ["both", "both", "both"],
        }
    )
    values = {
        "AAL": np.array([0.8, 1.1, 1.5], dtype=float),
        "SOD": np.array([0.6, 0.9, 1.2], dtype=float),
        "N": np.array([0.7, 1.0, 1.3], dtype=float),
    }
    summaries = {
        impact: {
            "mean": float(payload.mean()),
            "median": float(np.median(payload)),
            "p5": float(np.percentile(payload, 5)),
            "p25": float(np.percentile(payload, 25)),
            "p75": float(np.percentile(payload, 75)),
            "p95": float(np.percentile(payload, 95)),
        }
        for impact, payload in values.items()
    }
    paths = render_asr_polar(
        frame=pb_frame,
        values=values,
        summaries=summaries,
        frequencies={"AAL": 0.0, "SOD": 1.0, "N": 0.5},
        output_stem=tmp_path / "pb_polar",
        title="PB polar",
        lcia_method="pb_lcia",
        style="violin",
        dpi=1,
        output_format="svg",
    )
    assert paths == [tmp_path / "pb_polar.svg"]
    assert paths[0].exists()
    paths = render_asr_polar(
        frame=pb_frame,
        values=values,
        summaries=summaries,
        frequencies={"AAL": 0.0, "SOD": 1.0, "N": 0.5},
        output_stem=tmp_path / "pb_polar_normal",
        title="PB polar normal",
        lcia_method="pb_lcia",
        style="violin",
        dpi=1,
        output_format="svg",
        scale_mode=ASR_NORMAL_SCALE,
    )
    assert paths == [tmp_path / "pb_polar_normal.svg"]
    assert paths[0].exists()
    with pytest.raises(ValueError, match="cannot render negative ASR values"):
        render_asr_polar(
            frame=pb_frame,
            values={**values, "AAL": np.array([-0.1, 1.0], dtype=float)},
            summaries=summaries,
            frequencies={"AAL": 0.0, "SOD": 1.0, "N": 0.5},
            output_stem=tmp_path / "pb_polar_negative",
            title="PB polar negative",
            lcia_method="pb_lcia",
            style="violin",
            dpi=10,
            output_format="png",
            scale_mode=ASR_NORMAL_SCALE,
        )

    gwp_frame = pd.DataFrame({"impact": ["GWP_100"], "cc_bound": ["min_cc"]})
    paths = render_asr_polar(
        frame=gwp_frame,
        values={"GWP_100": np.array([0.5, 1.0, 2.0], dtype=float)},
        summaries=None,
        frequencies=None,
        output_stem=tmp_path / "gwp_polar",
        title="GWP polar",
        lcia_method="gwp100_lcia",
        style="deterministic",
        dpi=1,
        output_format="svg",
        deterministic_note="Min variant compression: ref_year=2005.",
    )
    assert paths == [tmp_path / "gwp_polar.svg"]
    assert paths[0].exists()

    fallback_frame = pd.DataFrame({"impact": ["fallback impact"], "cc_bound": ["min_cc"]})
    paths = render_asr_polar(
        frame=fallback_frame,
        values={"fallback impact": np.array([0.5, 1.0, 2.0], dtype=float)},
        summaries=None,
        frequencies=None,
        output_stem=tmp_path / "fallback_polar",
        title="Fallback polar",
        lcia_method="gwp100_lcia",
        style="deterministic",
        dpi=1,
        output_format="svg",
    )
    assert paths == [tmp_path / "fallback_polar.svg"]
    assert paths[0].exists()


def _asr_uncertainty_figure_frame(*, impacts: list[str], include_values: bool) -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(["UT(FD)", "AR(E^{CBA_FD})"]):
        for impact_index, impact in enumerate(impacts):
            for year_index, year in enumerate([2005, 2006]):
                value = 1.0 + method_index + impact_index * 0.5 + year_index * 0.1
                route = "" if year_index == 0 else "regression_proj"
                scenario = "" if year_index == 0 else "SSP2"
                row: dict[str, object] = {
                    "__method": method,
                    "l1_l2_method": method,
                    "l1_method": method.split("::")[0],
                    "l2_method": method,
                    "lcia_method": "pb_lcia",
                    "impact": impact,
                    "impact_unit": "kg",
                    "year": year,
                    "cc_type": "static",
                    "cc_bound": "min_cc",
                    ASOCC_SSP_SCENARIO_COLUMN: scenario,
                    EXT_LCA_SSP_SCENARIO_COLUMN: scenario,
                    ASOCC_TIME_ROUTE_PUBLIC_COLUMN: route,
                    FNT_FRACTION_COLUMN: 0.25 + 0.1 * method_index,
                    "__asr_max_threshold": 4.0,
                    "mean": value,
                    "median": value,
                    "p5": value * 0.8,
                    "p25": value * 0.9,
                    "p75": value * 1.1,
                    "p95": value * 1.2,
                }
                if include_values:
                    row[VALUE_ARRAY_COLUMN] = np.array([value * 0.8, value, value * 1.2])
                rows.append(row)
    return pd.DataFrame(rows)


def _asr_dynamic_uncertainty_figure_frame(*, include_values: bool = False) -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(["UT(FD)", "AR(E^{CBA_FD})"]):
        for year_index, (year, route, scenario) in enumerate(
            [(2020, "", ""), (2021, "regression_proj", "SSP2")]
        ):
            value = 1.0 + method_index * 0.25 + year_index * 0.1
            row: dict[str, object] = {
                "__method": method,
                "l1_l2_method": method,
                "l1_method": method,
                "l2_method": method,
                "lcia_method": "gwp100_lcia",
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "year": year,
                "cc_type": "dynamic_ar6",
                "cc_bound": "dynamic",
                "cc_category": "C1",
                "cc_model": "M1",
                "cc_scenario": "S1",
                "ar6_cc_ssp_scenario": "SSP2",
                ASOCC_SSP_SCENARIO_COLUMN: scenario,
                EXT_LCA_SSP_SCENARIO_COLUMN: scenario,
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: route,
                MODEL_SCENARIO_PAIR_COUNT_COLUMN: 2,
                MODEL_SCENARIO_SAMPLING_METHOD_COLUMN: "srs",
                FNT_FRACTION_COLUMN: 0.6 + year_index * 0.1,
                "__asr_max_threshold": np.nan,
                CUMULATIVE_FNT_FRACTION_COLUMN: 0.0 if method_index == 0 else 0.8,
                "__cumulative_values": np.array([1.2 + method_index, 1.4 + method_index]),
                "mean": value,
                "median": value,
                "p5": value * 0.8,
                "p25": value * 0.9,
                "p75": value * 1.1,
                "p95": value * 1.2,
            }
            if include_values:
                row[VALUE_ARRAY_COLUMN] = np.array([value * 0.8, value, value * 1.2])
            rows.append(row)
    return pd.DataFrame(rows)


def _asr_dynamic_component_rows(
    tmp_path: Path,
    *,
    uncertainty_ar6: bool = False,
) -> ComponentDiagnosticRows:
    acc_rows = []
    lca_rows = []
    for method_index, method in enumerate(["UT(FD)", "AR(E^{CBA_FD})"]):
        for year_index, (year, route, scenario) in enumerate(
            [(2020, "", ""), (2021, "regression_proj", "SSP2")]
        ):
            base = 8.0 + method_index * 2.0 + year_index
            acc_rows.append(
                {
                    "__method": method,
                    "l1_l2_method": method,
                    "lcia_method": "gwp100_lcia",
                    "impact": "GWP_100",
                    "impact_unit": "kg CO2-eq",
                    "year": year,
                    "cc_type": "dynamic_ar6",
                    "cc_category": "C1",
                    "cc_model": "M1",
                    "cc_scenario": "S1",
                    "ar6_cc_ssp_scenario": "SSP2",
                    ASOCC_SSP_SCENARIO_COLUMN: scenario,
                    EXT_LCA_SSP_SCENARIO_COLUMN: scenario,
                    ASOCC_TIME_ROUTE_PUBLIC_COLUMN: route,
                    "__component_cumulative_values": np.array([base * 1.8, base * 2.0]),
                    "mean": base,
                    "median": base,
                    "p5": base * 0.8,
                    "p25": base * 0.9,
                    "p75": base * 1.1,
                    "p95": base * 1.2,
                }
            )
    for year_index, (year, scenario) in enumerate([(2020, ""), (2021, "SSP2")]):
        base = 6.0 + year_index
        lca_rows.append(
            {
                "lcia_method": "gwp100_lcia",
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "year": year,
                "cc_type": "dynamic_ar6",
                "cc_category": "C1",
                "cc_model": "M1",
                "cc_scenario": "S1",
                "ar6_cc_ssp_scenario": "SSP2",
                EXT_LCA_SSP_SCENARIO_COLUMN: scenario,
                "__component_cumulative_values": np.array([base * 1.8, base * 2.0]),
                "mean": base,
                "median": base,
                "p5": base * 0.8,
                "p25": base * 0.9,
                "p75": base * 1.1,
                "p95": base * 1.2,
            }
        )
    acc = pd.DataFrame(acc_rows)
    ar6_manifest_path, _acc_output_file = _write_dynamic_ar6_prerequisite(tmp_path)
    uncertainty_ar6_manifest_path = (
        _write_uncertainty_dynamic_ar6_prerequisite(tmp_path) if uncertainty_ar6 else None
    )
    acc_manifest_path = tmp_path / "uncertainty_acc" / "logs" / "scope_manifest.json"
    acc_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(
        path=acc_manifest_path,
        manifest=build_manifest(
            family="acc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=(),
            completed_runs=2,
            artifacts={"scope_manifest": str(acc_manifest_path)},
            deterministic_prerequisites=(
                {
                    "base_function_source": "deterministic_ar6_cc",
                    "scope_manifest": str(ar6_manifest_path),
                    "process_ar6": _dynamic_process_ar6_payload(tmp_path),
                },
                *(
                    (
                        {
                            "base_function_source": "uncertainty_ar6_cc",
                            "scope_manifest": str(uncertainty_ar6_manifest_path),
                        },
                    )
                    if uncertainty_ar6_manifest_path is not None
                    else ()
                ),
            ),
        ),
    )
    asr_manifest_path = tmp_path / "uncertainty_asr" / "logs" / "scope_manifest.json"
    asr_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        family="asr",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        completed_runs=2,
        artifacts={"scope_manifest": str(asr_manifest_path)},
        deterministic_prerequisites=(
            {
                "base_function_source": "uncertainty_acc",
                "scope_manifest": str(acc_manifest_path),
            },
        ),
    )
    write_manifest(path=asr_manifest_path, manifest=manifest)
    return ComponentDiagnosticRows(
        acc_method=acc,
        acc_inter=acc.drop(columns=["__method"]),
        lca=pd.DataFrame(lca_rows),
        manifest=manifest,
        requested_years=(2020, 2021),
        emissions_mode="gross_alt",
    )


def _asr_deterministic_static_frame(*, impacts: list[str], years: list[int]) -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(["UT(FD)", "AR(E^{CBA_FD})"]):
        for impact_index, impact in enumerate(impacts):
            for year_index, year in enumerate(years):
                for reference_year, multiplier in [(2005, 1.0), (2006, 1.4)]:
                    route = "" if year_index == 0 else "regression_proj"
                    scenario = "" if year_index == 0 else "SSP2"
                    value = 1.0 + method_index + impact_index * 0.5 + year_index * 0.1
                    rows.append(
                        {
                            "__method": method,
                            "l1_l2_method": method,
                            "l1_method": method,
                            "l2_method": method,
                            "lcia_method": "pb_lcia",
                            "impact": impact,
                            "impact_unit": "ratio",
                            "year": year,
                            "cc_type": "static",
                            "cc_bound": "min_cc",
                            "reference_year": reference_year,
                            ASOCC_SSP_SCENARIO_COLUMN: scenario,
                            EXT_LCA_SSP_SCENARIO_COLUMN: scenario,
                            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: route,
                            "value": value * multiplier,
                            "cumulative_asr": value * multiplier * 2.0,
                        }
                    )
    return pd.DataFrame(rows)


def _asr_deterministic_dynamic_frame() -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(["UT(FD)", "AR(E^{CBA_FD})"]):
        for year_index, (year, route, scenario) in enumerate(
            [(2020, "", ""), (2021, "regression_proj", "SSP2")]
        ):
            for reference_year, multiplier in [(2005, 1.0), (2006, 1.5)]:
                value = (0.6 + method_index * 0.2 + year_index * 0.1) * multiplier
                rows.append(
                    {
                        "__method": method,
                        "l1_l2_method": method,
                        "l1_method": method,
                        "l2_method": method,
                        "lcia_method": "gwp100_lcia",
                        "impact": "GWP_100",
                        "impact_unit": "ratio",
                        "year": year,
                        "cc_type": "dynamic_ar6",
                        "cc_bound": "dynamic",
                        "cc_category": "C1",
                        "cc_model": "M1",
                        "cc_scenario": "S1",
                        "ar6_cc_ssp_scenario": "SSP2",
                        "reference_year": reference_year,
                        ASOCC_SSP_SCENARIO_COLUMN: scenario,
                        EXT_LCA_SSP_SCENARIO_COLUMN: scenario,
                        ASOCC_TIME_ROUTE_PUBLIC_COLUMN: route,
                        "value": value,
                        "cumulative_asr": value * 1.8,
                    }
                )
    return pd.DataFrame(rows)


def _asr_deterministic_component_rows(tmp_path: Path) -> DeterministicComponentRows:
    acc_rows = []
    lca_rows = []
    for method_index, method in enumerate(["UT(FD)", "AR(E^{CBA_FD})"]):
        for year_index, (year, route, scenario) in enumerate(
            [(2020, "", ""), (2021, "regression_proj", "SSP2")]
        ):
            for reference_year, multiplier in [(2005, 1.0), (2006, 1.6)]:
                value = (8.0 + method_index * 2.0 + year_index) * multiplier
                acc_rows.append(
                    {
                        "__method": method,
                        "l1_l2_method": method,
                        "lcia_method": "gwp100_lcia",
                        "impact": "GWP_100",
                        "impact_unit": "kg CO2-eq",
                        "year": year,
                        "cc_type": "dynamic_ar6",
                        "cc_category": "C1",
                        "cc_model": "M1",
                        "cc_scenario": "S1",
                        "ar6_cc_ssp_scenario": "SSP2",
                        "reference_year": reference_year,
                        ASOCC_SSP_SCENARIO_COLUMN: scenario,
                        EXT_LCA_SSP_SCENARIO_COLUMN: scenario,
                        ASOCC_TIME_ROUTE_PUBLIC_COLUMN: route,
                        "__component_value": value,
                    }
                )
    for year_index, (year, scenario) in enumerate([(2020, ""), (2021, "SSP2")]):
        value = 5.0 + year_index
        lca_rows.append(
            {
                "lcia_method": "gwp100_lcia",
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "year": year,
                "cc_type": "dynamic_ar6",
                "cc_category": "C1",
                "cc_model": "M1",
                "cc_scenario": "S1",
                "ar6_cc_ssp_scenario": "SSP2",
                EXT_LCA_SSP_SCENARIO_COLUMN: scenario,
                "__component_value": value,
            }
        )
    _ar6_manifest_path, acc_output_file = _write_dynamic_ar6_prerequisite(tmp_path)
    return DeterministicComponentRows(
        acc=pd.DataFrame(acc_rows),
        lca=pd.DataFrame(lca_rows),
        acc_output_files=(acc_output_file,),
    )


def _write_dynamic_ar6_prerequisite(tmp_path: Path) -> tuple[Path, Path]:
    ar6_results = tmp_path / "ar6_cc" / "results"
    ar6_logs = tmp_path / "ar6_cc" / "logs"
    acc_results = tmp_path / "acc" / "results"
    acc_logs = tmp_path / "acc" / "logs"
    for directory in (ar6_results, ar6_logs, acc_results, acc_logs):
        directory.mkdir(parents=True, exist_ok=True)
    output_file = ar6_results / "ar6_cc.csv"
    post_file = ar6_results / "ar6_cc_post_study_period.csv"
    pd.DataFrame(
        [
            {
                "cc_model": "M1",
                "cc_scenario": "S1",
                "cc_category": "C1",
                "ssp_scenario": "SSP2",
                "cc_flow": "positive_emissions",
                "cc_variable": "Emissions|Kyoto Gases",
                "impact_unit": "kg CO2-eq",
                "2020": 100.0,
                "2021": 95.0,
            },
            {
                "cc_model": "M1",
                "cc_scenario": "S1",
                "cc_category": "C1",
                "ssp_scenario": "SSP2",
                "cc_flow": CC_FLOW_NEGATIVE,
                "cc_variable": SEQUESTRATION_SUBTOTAL,
                "impact_unit": "kg CO2-eq",
                "2020": 0.0,
                "2021": 0.0,
            },
        ]
    ).to_csv(output_file, index=False)
    pd.DataFrame(
        [
            {
                "cc_model": "M1",
                "cc_scenario": "S1",
                "cc_category": "C1",
                "ssp_scenario": "SSP2",
                "cc_flow": "positive_emissions",
                "cc_variable": "Emissions|Kyoto Gases",
                "impact_unit": "kg CO2-eq",
                "2022": 90.0,
            },
            {
                "cc_model": "M1",
                "cc_scenario": "S1",
                "cc_category": "C1",
                "ssp_scenario": "SSP2",
                "cc_flow": CC_FLOW_NEGATIVE,
                "cc_variable": SEQUESTRATION_SUBTOTAL,
                "impact_unit": "kg CO2-eq",
                "2022": 0.0,
            },
        ]
    ).to_csv(post_file, index=False)
    ar6_manifest_path = ar6_logs / "scope_manifest.json"
    write_json_dict(
        ar6_manifest_path,
        {
            "function": "deterministic_ar6_cc",
            "arguments": {},
            "execution": {"status": "complete"},
            "reuse": {},
            "artifacts": {
                "output_file": str(output_file),
                "post_study_output_file": str(post_file),
            },
            "provenance": {"process_ar6": _dynamic_process_ar6_payload(tmp_path)},
        },
    )
    acc_output_file = acc_results / "UT_FD__gwp100_lcia__dynamic_ar6.csv"
    acc_output_file.write_text("placeholder\n", encoding="utf-8")
    write_json_dict(
        acc_logs / "scope_manifest.json",
        {
            "function": "deterministic_acc",
            "arguments": {},
            "execution": {"status": "complete"},
            "reuse": {},
            "artifacts": {"output_files": [str(acc_output_file)]},
            "provenance": {"cc_input_path": str(output_file)},
        },
    )
    return ar6_manifest_path, acc_output_file


def _dynamic_process_ar6_payload(tmp_path: Path) -> dict[str, object]:
    return {
        "reuse_status": "computed",
        "study_period": "2020-2021",
        "categories": ["C1"],
        "ssps": ["SSP2"],
        "harmonization": True,
        "harmonization_method": "offset",
        "output_root": str(tmp_path / "process_ar6"),
        "output_files_available": 1,
        "figures_available": 0,
        "variable_coverage": [],
    }


def _write_uncertainty_dynamic_ar6_prerequisite(tmp_path: Path) -> Path:
    run_root = tmp_path / "uncertainty_ar6_cc"
    results = run_root / "results"
    logs = run_root / "logs"
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    output_format = "csv_compact"
    identity_rows = []
    post_identity_rows = []
    public_row_id = 0
    for year in (2020, 2021):
        for flow, variable, base in (
            (CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, 100.0 + year - 2020),
            (CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, -20.0 - year + 2020),
        ):
            identity_rows.append(
                {
                    "public_row_id": public_row_id,
                    "cc_category": "C1",
                    "ssp_scenario": "SSP2",
                    "cc_flow": flow,
                    "cc_variable": variable,
                    "impact_unit": "kg CO2-eq",
                    "year": year,
                    "cc_model": "M1",
                    "cc_scenario": "S1",
                    "__base_value": base,
                }
            )
            public_row_id += 1
    for flow, variable, base in (
        (CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, 90.0),
        (CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, -18.0),
    ):
        post_identity_rows.append(
            {
                "public_row_id": public_row_id,
                "cc_category": "C1",
                "ssp_scenario": "SSP2",
                "cc_flow": flow,
                "cc_variable": variable,
                "impact_unit": "kg CO2-eq",
                "year": 2022,
                "cc_model": "M1",
                "cc_scenario": "S1",
                "__base_value": base,
            }
        )
        public_row_id += 1
    identity = pd.DataFrame(identity_rows)
    post_identity = pd.DataFrame(post_identity_rows)

    def sparse_runs(rows: pd.DataFrame) -> SparseRunRows:
        run_indices = []
        public_row_ids = []
        values = []
        for run_index, factor in [(0, 1.0), (1, 1.1), (2, 9.0)]:
            for row in rows.to_dict(orient="records"):
                run_indices.append(run_index)
                public_row_ids.append(int(row["public_row_id"]))
                values.append(float(row["__base_value"]) * factor)
        return SparseRunRows(
            run_index=np.asarray(run_indices, dtype=np.int64),
            public_row_id=np.asarray(public_row_ids, dtype=np.int64),
            values=np.asarray(values, dtype=np.float64),
            value_column="value",
        )

    public_identity_path = results / "public_row_identity.csv"
    public_runs_path = results / "cc_runs.csv"
    post_identity_path = results / "post_study_period_public_row_identity.csv"
    post_runs_path = results / "post_study_period_cc_runs.csv"
    summary_path = results / "summary_stats_runs.csv"
    post_summary_path = results / "post_study_period_summary_stats_runs.csv"
    public_identity = identity.drop(columns=["__base_value"])
    post_public_identity = post_identity.drop(columns=["__base_value"])
    write_uncertainty_table(
        path=public_identity_path,
        frame=public_identity,
        output_format=output_format,
    )
    with SparseRunRowsWriter(path=public_runs_path, output_format=output_format) as writer:
        writer.write_batch(rows=sparse_runs(identity), batch_index=0)
    write_uncertainty_table(
        path=post_identity_path,
        frame=post_public_identity,
        output_format=output_format,
    )
    with SparseRunRowsWriter(path=post_runs_path, output_format=output_format) as writer:
        writer.write_batch(rows=sparse_runs(post_identity), batch_index=0)
    summary_identity, summary_groups = ar6_cc_summary_identity_groups(
        identity=public_identity,
        category_uncertainty=False,
    )
    write_uncertainty_table(
        path=summary_path,
        frame=exact_summary_from_public_runs(
            identity_frame=summary_identity,
            runs_path=public_runs_path,
            output_format=output_format,
            run_count=2,
            public_row_groups=summary_groups,
            sparse=True,
        ),
        output_format=output_format,
    )
    post_summary_identity, post_summary_groups = ar6_cc_summary_identity_groups(
        identity=post_public_identity,
        category_uncertainty=False,
    )
    write_uncertainty_table(
        path=post_summary_path,
        frame=exact_summary_from_public_runs(
            identity_frame=post_summary_identity,
            runs_path=post_runs_path,
            output_format=output_format,
            run_count=2,
            public_row_groups=post_summary_groups,
            sparse=True,
        ),
        output_format=output_format,
    )

    budget_identity = pd.DataFrame(
        [
            {
                "budget_row_id": index,
                "cc_category": "C1",
                "ssp_scenario": "SSP2",
                "cc_flow": flow,
                "cc_variable": variable,
                "impact_unit": "kg CO2-eq",
                "period_segment": segment,
            }
            for index, (flow, variable, segment) in enumerate(
                [
                    (CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, "study_period"),
                    (CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, "study_period"),
                    (CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, "post_study_period"),
                    (CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, "post_study_period"),
                ]
            )
        ]
    )
    budget_runs = pd.DataFrame(
        {
            "run_index": [0, 1, 2],
            "0": [201.0, 221.1, 1809.0],
            "1": [-41.0, -45.1, -369.0],
            "2": [90.0, 99.0, 810.0],
            "3": [-18.0, -19.8, -162.0],
        }
    )
    write_uncertainty_table(
        path=results / "study_and_post_study_period_budget_row_identity.csv",
        frame=budget_identity,
        output_format=output_format,
    )
    with CompactRunMatrixWriter(
        path=results / "study_and_post_study_period_budget_runs.csv",
        output_format=output_format,
    ) as writer:
        writer.write_batch(
            run_indices=budget_runs["run_index"].to_numpy(dtype=np.int64),
            values=budget_runs.loc[:, ["0", "1", "2", "3"]].to_numpy(dtype=np.float64),
            batch_index=0,
        )
    pd.DataFrame(
        [
            {
                "cc_category": "C1",
                "ssp_scenario": "SSP2",
                "cc_flow": flow,
                "cc_variable": variable,
                "impact_unit": "kg CO2-eq",
                "cc_model": "M1",
                "cc_scenario": "S1",
            }
            for flow, variable in (
                (CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU),
                (CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL),
            )
        ]
    ).to_csv(logs / "source_methods.csv", index=False)
    manifest_path = logs / "scope_manifest.json"
    write_manifest(
        path=manifest_path,
        manifest=build_manifest(
            family="ar6_cc",
            mode="fixed",
            output_format=output_format,
            active_sources=("dynamic_ar6_cc_uncertainty",),
            status="complete",
            completed_runs=2,
            source_parameters={"dynamic_ar6_cc_uncertainty": {"category_uncertainty": False}},
            artifacts={
                "scope_manifest": str(manifest_path),
                "public_output": {"cc_runs": {"layout": "sparse_selected_rows"}},
            },
        ),
    )
    return manifest_path


def test_asr_uncertainty_mean_lines_split_multi_impact_products(
    project_repo: Path,
    tmp_path: Path,
) -> None:
    del project_repo
    frame = _asr_uncertainty_figure_frame(impacts=["AAL", "SOD"], include_values=False)

    band_paths = plot_band_scope(
        frame=frame,
        output_stem=tmp_path / "bands",
        title="ASR bands",
        dpi=1,
        output_format="svg",
        group_legend=False,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    paths = plot_mean_line_scope(
        frame=frame,
        requested_years=[2005, 2006],
        output_stem=tmp_path / "mean_lines",
        title="ASR mean lines",
        dpi=1,
        output_format="svg",
        group_legend=False,
        include_impact_in_label=False,
        include_method_in_label=True,
    )

    assert sorted(path.name for path in band_paths) == [
        "bands.svg",
        "bands__frequency_of_no_transgression.svg",
    ]
    assert sorted(path.name for path in paths) == ["mean_lines__AAL.svg", "mean_lines__SOD.svg"]
    assert all(path.exists() for path in [*band_paths, *paths])


def test_asr_uncertainty_violin_legends_cover_grouped_and_ungrouped_products(
    project_repo: Path,
    tmp_path: Path,
) -> None:
    del project_repo
    single_impact = _asr_uncertainty_figure_frame(impacts=["AAL"], include_values=True)
    multi_impact = _asr_uncertainty_figure_frame(impacts=["AAL", "SOD"], include_values=True)

    grouped_paths = plot_violin_scope(
        frame=single_impact.loc[single_impact["year"].eq(2005)].copy(),
        output_stem=tmp_path / "grouped_violin",
        title="Grouped violin",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    ungrouped_paths = plot_violin_scope(
        frame=multi_impact.loc[multi_impact["year"].eq(2005)].copy(),
        output_stem=tmp_path / "ungrouped_violin",
        title="Ungrouped violin",
        dpi=1,
        output_format="svg",
        group_legend=False,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    grouped_multi_paths = plot_violin_scope(
        frame=multi_impact.loc[multi_impact["year"].eq(2005)].copy(),
        output_stem=tmp_path / "grouped_multi_violin",
        title="Grouped multi violin",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )

    assert grouped_paths == [tmp_path / "grouped_violin.svg"]
    assert grouped_paths[0].exists()
    assert ungrouped_paths == [tmp_path / "ungrouped_violin.svg"]
    assert ungrouped_paths[0].exists()
    assert grouped_multi_paths == [tmp_path / "grouped_multi_violin.svg"]
    assert grouped_multi_paths[0].exists()


def test_asr_dynamic_uncertainty_renderers_cover_component_and_transition_products(
    project_repo: Path,
    tmp_path: Path,
) -> None:
    del project_repo
    frame = _asr_dynamic_uncertainty_figure_frame()
    components = _asr_dynamic_component_rows(tmp_path / "deterministic_ar6")
    uncertainty_components = _asr_dynamic_component_rows(
        tmp_path / "uncertainty_ar6",
        uncertainty_ar6=True,
    )
    uncertainty_global_ar6 = uncertainty_global_ar6_source(manifest=uncertainty_components.manifest)
    inter_frame = frame.drop(columns=["__method", "l1_l2_method", "l1_method", "l2_method"])

    band_paths = plot_band_scope(
        frame=inter_frame,
        output_stem=tmp_path / "dynamic_band",
        title="Dynamic ASR bands",
        dpi=1,
        output_format="svg",
        group_legend=False,
        include_impact_in_label=False,
        include_method_in_label=False,
        components=uncertainty_components,
        global_ar6_source=uncertainty_global_ar6,
    )
    no_sampling_frame = inter_frame.copy()
    no_sampling_frame[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN] = pd.NA
    no_sampling_paths = plot_band_scope(
        frame=no_sampling_frame,
        output_stem=tmp_path / "dynamic_band_no_sampling",
        title="Dynamic ASR bands no sampling",
        dpi=1,
        output_format="svg",
        group_legend=False,
        include_impact_in_label=False,
        include_method_in_label=False,
        components=uncertainty_components,
        global_ar6_source=uncertainty_global_ar6,
    )
    mean_paths = plot_mean_line_scope(
        frame=frame,
        requested_years=[2020, 2021],
        output_stem=tmp_path / "dynamic_mean",
        title="Dynamic ASR means",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
        components=components,
        global_ar6_source=uncertainty_global_ar6_source(manifest=components.manifest),
    )
    assert band_paths == [
        tmp_path / "dynamic_band__incl_post.svg",
        tmp_path / "dynamic_band__excl_post.svg",
    ]
    assert no_sampling_paths == [
        tmp_path / "dynamic_band_no_sampling__incl_post.svg",
        tmp_path / "dynamic_band_no_sampling__excl_post.svg",
    ]
    assert mean_paths == [
        tmp_path / "dynamic_mean__incl_post.svg",
        tmp_path / "dynamic_mean__excl_post.svg",
    ]
    assert all(path.exists() for path in [*band_paths, *no_sampling_paths, *mean_paths])
    fig, axis = plt.subplots()
    try:
        component_values, component_years = _render_component_mean_axis(
            axis=axis,
            acc_rows=components.acc_method,
            lca_rows=components.lca,
            group_legend=True,
            include_method_in_label=True,
            color_map={
                "UT(FD)": "#1f77b4",
                "AR(E^{CBA_FD})": "#ff7f0e",
            },
        )
    finally:
        plt.close(fig)
    assert component_values.size
    assert component_years == [2020, 2021, 2020, 2021, 2020, 2021]
    fig, axis = plt.subplots()
    try:
        ungrouped_values, ungrouped_years = _render_component_mean_axis(
            axis=axis,
            acc_rows=components.acc_method,
            lca_rows=components.lca,
            group_legend=False,
            include_method_in_label=False,
            color_map={"aCC": "#1f77b4"},
        )
    finally:
        plt.close(fig)
    assert ungrouped_values.size
    assert ungrouped_years == [2020, 2021, 2020, 2021, 2020, 2021]


def test_asr_deterministic_renderers_cover_variant_components_and_selector_stems(
    project_repo: Path,
    tmp_path: Path,
) -> None:
    del project_repo
    single = _prepare_plot_rows(
        _asr_deterministic_static_frame(impacts=["AAL"], years=[2005])
        .loc[lambda frame: frame["__method"].eq("UT(FD)")]
        .copy()
    )
    single_grouped = _prepare_plot_rows(
        _asr_deterministic_static_frame(impacts=["AAL"], years=[2005])
    )
    single_multi_year = _prepare_plot_rows(
        _asr_deterministic_static_frame(impacts=["AAL"], years=[2005, 2006])
    )
    polar = _prepare_plot_rows(
        _asr_deterministic_static_frame(impacts=["AAL", "SOD"], years=[2005])
        .loc[lambda frame: frame["__method"].eq("UT(FD)")]
        .copy()
    )
    multi = _prepare_plot_rows(
        _asr_deterministic_static_frame(impacts=["AAL", "SOD", "OA"], years=[2005, 2006])
    )
    dynamic = _prepare_plot_rows(_asr_deterministic_dynamic_frame())
    components = _asr_deterministic_component_rows(tmp_path)
    deterministic_global_ar6 = deterministic_global_ar6_source(
        acc_output_files=list(components.acc_output_files)
    )

    single_paths = _plot_scope(
        frame=single,
        requested_years=[2005],
        output_stem=tmp_path / "deterministic_single",
        title="Deterministic single",
        dpi=1,
        output_format="svg",
        group_legend=False,
        include_method_in_label=False,
        allow_polar=False,
        dynamic=False,
        emissions_mode=None,
        scale_mode=ASR_NORMAL_SCALE,
    )
    single_grouped_paths = _plot_scope(
        frame=single_grouped,
        requested_years=[2005],
        output_stem=tmp_path / "deterministic_single_grouped",
        title="Deterministic grouped single",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_method_in_label=True,
        allow_polar=False,
        dynamic=False,
        emissions_mode=None,
        scale_mode=ASR_LOG_SCALE,
    )
    single_multi_year_paths = _plot_scope(
        frame=single_multi_year,
        requested_years=[2005, 2006],
        output_stem=tmp_path / "deterministic_single_multi_year",
        title="Deterministic single multi year",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_method_in_label=True,
        allow_polar=False,
        dynamic=False,
        emissions_mode=None,
        scale_mode=ASR_LOG_SCALE,
    )
    polar_paths = _plot_scope(
        frame=polar,
        requested_years=[2005],
        output_stem=tmp_path / "deterministic_polar",
        title="Deterministic polar",
        dpi=1,
        output_format="svg",
        group_legend=False,
        include_method_in_label=False,
        allow_polar=True,
        dynamic=False,
        emissions_mode=None,
        scale_mode=ASR_NORMAL_SCALE,
    )
    multi_paths = _plot_scope(
        frame=multi,
        requested_years=[2005, 2006],
        output_stem=tmp_path / "deterministic_multi",
        title="Deterministic multi",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_method_in_label=True,
        allow_polar=False,
        dynamic=False,
        emissions_mode=None,
        scale_mode=ASR_LOG_SCALE,
    )
    dynamic_paths = _plot_scope(
        frame=dynamic,
        requested_years=[2020, 2021],
        output_stem=tmp_path / "deterministic_dynamic",
        title="Deterministic dynamic",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_method_in_label=True,
        allow_polar=False,
        dynamic=True,
        emissions_mode="gross_alt",
        scale_mode=ASR_NORMAL_SCALE,
        components=components,
        global_ar6_source=deterministic_global_ar6,
    )

    assert (
        _scope_stem(
            label="multi_method",
            frame=dynamic,
            requested_years=[2020, 2021],
            cc_source="gwp100_lcia",
            cc_type="dynamic_ar6",
            include_impact=True,
            selector_token="rp_FR__sp_D",
        )
        == "multi_method__rp_FR__sp_D__GWP_100__SSP2__C1__M1_S1"
    )
    dynamic_without_pair = dynamic.drop(columns=["cc_model", "cc_scenario"])
    assert (
        _scope_stem(
            label="multi_method",
            frame=dynamic_without_pair,
            requested_years=[2020, 2021],
            cc_source="gwp100_lcia",
            cc_type="dynamic_ar6",
        )
        == "multi_method__SSP2__C1"
    )
    assert (
        _scope_stem(
            label="multi_method",
            frame=multi,
            requested_years=[2005, 2006],
            cc_source="pb_lcia",
            cc_type="static",
            include_impact=True,
            selector_token="rp_FR__sp_D",
        )
        == "multi_method__pb_lcia__rp_FR__sp_D__AAL__SSP2"
    )
    assert (
        _scope_stem(
            label="multi_method",
            frame=multi,
            requested_years=[2005, 2006],
            cc_source="pb_lcia",
            cc_type="static",
        )
        == "multi_method__pb_lcia__SSP2"
    )
    assert single_paths == [tmp_path / "deterministic_single.svg"]
    assert single_grouped_paths == [tmp_path / "deterministic_single_grouped.svg"]
    assert single_multi_year_paths == [tmp_path / "deterministic_single_multi_year.svg"]
    assert polar_paths == [tmp_path / "polar_deterministic_polar.svg"]
    assert multi_paths == [tmp_path / "deterministic_multi.svg"]
    assert dynamic_paths == [
        tmp_path / "deterministic_dynamic__incl_post.svg",
        tmp_path / "deterministic_dynamic__excl_post.svg",
    ]
    min_note = _min_variant_note(
        pd.DataFrame(
            {
                "__method": ["A", "B"],
                "year": [2030, 2030],
                "value": [1.0, 2.0],
                "reference_year": [2020, 2020],
                "l2_reuse_year": [2028, 2028],
                "__variant_role": ["min", "min"],
            }
        )
    )
    assert min_note is not None
    assert "Min variant compression: ref_year=2020, l2_reuse_year=2028." in min_note
    assert "ref_year and l2_reuse_year: A; B" in min_note
    assert "l2_reuse_year affects only the L2 in L1 prospective allocation weighting." in min_note
    retained_note = _min_variant_note(
        pd.DataFrame(
            {
                "__method": ["A", "A"],
                "year": [2019, 2019],
                "value": [1.0, 2.0],
                "reference_year": [2011, 2011],
                "l2_reuse_year": [2030, 2030],
            }
        )
    )
    assert retained_note is not None
    assert "ref_year=2011, l2_reuse_year=2030." in retained_note
    assert (
        "l2_reuse_year affects only the L2 in L1 prospective allocation weighting." in retained_note
    )
    assert (
        _min_variant_note(
            pd.DataFrame(
                {
                    "__method": ["A", "B"],
                    "year": [2030, 2030],
                    "value": [1.0, 2.0],
                    "reference_year": [pd.NA, pd.NA],
                    "__variant_role": ["min", "max"],
                }
            )
        )
        is None
    )
    no_method_bucket_note = _min_variant_note(
        pd.DataFrame(
            [
                {
                    "__method": "A",
                    "year": 2030,
                    "value": 1.0,
                    "reference_year": pd.NA,
                    "__variant_role": "min",
                },
                {
                    "__method": "B",
                    "year": 2030,
                    "value": 2.0,
                    "reference_year": pd.NA,
                    "__variant_role": "min",
                },
                {
                    "__method": "",
                    "year": 2030,
                    "value": 3.0,
                    "reference_year": 2020,
                    "__variant_role": "min",
                },
            ]
        )
    )
    assert no_method_bucket_note is not None
    assert "only:" not in no_method_bucket_note
    assert all(
        path.exists()
        for path in [
            *single_paths,
            *single_grouped_paths,
            *polar_paths,
            *multi_paths,
            *dynamic_paths,
        ]
    )
