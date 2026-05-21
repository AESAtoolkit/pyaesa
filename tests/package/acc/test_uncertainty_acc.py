from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pyaesa import uncertainty_acc
from pyaesa.acc.figures.common import (
    AR6_CATEGORY_SCOPE_COLUMN,
    BUDGET_VALUES_COLUMN,
    PAIR_COUNT_COLUMN,
    VALUE_ARRAY_COLUMN,
    acc_scope_stem,
    format_year_axis,
    method_labels,
    ordered_impacts,
    scope_title,
)
from pyaesa.acc.uncertainty.figures.product_renderers import (
    _render_dynamic_budget_axis,
    plot_band_scope,
    plot_mean_line_scope,
)
from pyaesa.acc.uncertainty.figures.violin_renderers import plot_violin_scope
from pyaesa.acc.uncertainty.figures.render import _collapsed_inter_method_rows
from pyaesa.acc.uncertainty.evaluation import branches as branch_mod
from pyaesa.acc.uncertainty.evaluation import sparse_runs as sparse_render_mod
from pyaesa.acc.uncertainty.evaluation import sparse_rows as sparse_rows_mod
from pyaesa.acc.uncertainty.evaluation.summary import (
    ACC_SUMMARY_SCOPE_COLUMN,
    ACC_SUMMARY_SCOPE_INTER_METHOD,
    ACC_SUMMARY_SCOPE_PER_METHOD,
    acc_summary_identity_groups,
)
from pyaesa.acc.uncertainty.figures.row_reader import (
    attach_dynamic_budget_values,
    collapsed_value_rows,
    read_figure_tables,
)
from pyaesa.acc.uncertainty.figures.scope_planner import FigureContext
from pyaesa.acc.uncertainty.runtime.models import (
    ACCAsoccInput,
    ACCBranchPlan,
    ACCDynamicCCInput,
    ACCUncertaintyPlan,
    ACCUncertaintyRunPaths,
)
from pyaesa.acc.uncertainty.sources.source_keys import (
    AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE,
    ASOCC_PROJECTION_SOURCE,
    ASOCC_REFERENCE_YEAR_SOURCE,
)
from pyaesa.acc.uncertainty.io.paths import build_acc_uncertainty_run_paths
from pyaesa.acc.uncertainty.io.run_outputs import write_acc_run_outputs
from pyaesa.acc.uncertainty.io.manifest_payloads import build_acc_manifest_context
from pyaesa.acc.uncertainty.io.source_methods import build_acc_source_methods
from pyaesa.acc.uncertainty.request.normalization import normalize_acc_uncertainty_config
from pyaesa.asocc.runtime.paths.external import get_asocc_external_method_level_dir
from pyaesa.asocc.runtime.scope.branch_resolution import outputs_project_root
from pyaesa.shared.uncertainty_assessment.run_state.manifest import build_manifest, read_manifest
from pyaesa.shared.uncertainty_assessment.request.core import normalize_uncertainty_request
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    collapse_values_to_summary_groups,
)
from pyaesa.shared.uncertainty_assessment.io.tables import (
    CompactRunMatrixWriter,
    SparseRunRows,
    SparseRunRowsWriter,
    read_uncertainty_table,
)
from pyaesa.shared.lcia.paths import static_cc_csv_path
from pyaesa.shared.figures.colors import DEFAULT_SINGLE_SERIES_COLOR
from pyaesa.shared.figures.dynamic_ar6 import MODEL_SCENARIO_SAMPLING_METHOD_COLUMN
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from tests.package.helpers.acc_dummy_repo import (
    prepare_dynamic_acc_repo_with_years,
)


def _run_root(*, repo_root: Path, project_name: str, run_id: str, source: str) -> Path:
    return repo_root / f"{project_name}" / "B2_acc" / source / "monte_carlo" / run_id


def _static_kwargs(*, project_name: str) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "years": [2030],
        "lcia_method": "gwp100_lcia",
        "fu_code": "L2.a.b",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "ssp_scenario": ["SSP2"],
            "projection_mode": "historical_reuse",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005, 2006],
            "include_lcia_based_allocation_methods": False,
        },
        "base_cc_args": {"static": {"exclude_max_cc": False}},
        "uncertainty_config": {
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
            "projection_uncertainty": {},
        },
        "output_format": "csv_compact",
    }


def _fast_figure_format() -> dict[str, Any]:
    return {"format": "png", "dpi": 10}


def _inactive_default_sources() -> dict[str, dict[str, bool]]:
    return {
        "projection_uncertainty": {"active": False},
        "reference_year_uncertainty": {"active": False},
        "inter_method_uncertainty": {"active": False},
        "dynamic_ar6_cc_uncertainty": {"active": False},
    }


def _only_projection_source() -> dict[str, dict[str, bool]]:
    return {**_inactive_default_sources(), "projection_uncertainty": {}}


def _only_reference_year_source() -> dict[str, dict[str, bool]]:
    return {**_inactive_default_sources(), "reference_year_uncertainty": {}}


def test_uncertainty_acc_rejects_sobol_years_outside_studied_years() -> None:
    kwargs = _static_kwargs(project_name="acc_bad_sobol_year")
    with pytest.raises(ValueError, match="sobol_years.*studied years"):
        uncertainty_acc(
            **kwargs,
            sobol_parameters={"sobol_years": [2031]},
            refresh=True,
        )


def test_uncertainty_acc_figure_rows_collapse_dynamic_sampled_axes(tmp_path: Path) -> None:
    sources = (
        ASOCC_REFERENCE_YEAR_SOURCE,
        ASOCC_PROJECTION_SOURCE,
        AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE,
    )
    paths = ACCUncertaintyRunPaths(
        run_root=tmp_path / "acc_run",
        public_row_identity=tmp_path / "acc_run" / "results" / "identity.csv",
        public_runs=tmp_path / "acc_run" / "results" / "runs.csv",
        summary_stats_runs=tmp_path / "acc_run" / "results" / "summary.csv",
        results_readme=tmp_path / "acc_run" / "results" / "README.txt",
        source_methods=tmp_path / "acc_run" / "logs" / "source_methods.csv",
        sobol_indices=tmp_path / "acc_run" / "results" / "sobol" / "indices.csv",
        sobol_source_summary=tmp_path / "acc_run" / "results" / "sobol" / "summary.csv",
        sobol_readme=tmp_path / "acc_run" / "results" / "sobol" / "README.txt",
        scope_manifest=tmp_path / "acc_run" / "logs" / "scope_manifest.json",
    )
    context = FigureContext(
        manifest=build_manifest(
            family="acc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=sources,
        ),
        paths=paths,
        figures_root=tmp_path / "figures",
        requested_years=(2024, 2030),
        requested_asocc_ssps=("SSP2",),
        fu_code="L2.a.a",
        output_format="csv_compact",
        figure_output_format="png",
        figure_dpi=10,
        per_method=True,
        multi_method=True,
        inter_method=True,
        active_sources=sources,
        run_layout="compact_run_matrix",
        dynamic_category_uncertainty_active=True,
        dynamic_cc_sampling_method="srs",
    )
    paths.public_row_identity.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "year": [2024, 2030],
            VALUE_ARRAY_COLUMN: [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            "cc_type": ["dynamic_ar6", "dynamic_ar6"],
            "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
            "impact": ["GWP_100", "GWP_100"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
            "reference_year": [2019, 2019],
            "l2_reuse_year": [pd.NA, 2022],
            "cc_model": ["model_h", "model_a"],
            "cc_scenario": ["scenario_h", "scenario_a"],
            "cc_category": ["C1", "C2"],
            ASOCC_SSP_SCENARIO_COLUMN: ["", "SSP2"],
            AR6_CC_SSP_SCENARIO_COLUMN: ["SSP2", "SSP2"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical", "l2_reuse_year"],
            "__method": ["A", "A"],
        }
    )
    rows.drop(columns=[VALUE_ARRAY_COLUMN]).to_csv(paths.public_row_identity, index=False)
    tables = read_figure_tables(context=context, include_summary=False)
    assert MODEL_SCENARIO_SAMPLING_METHOD_COLUMN in tables.identity.columns
    assert MODEL_SCENARIO_SAMPLING_METHOD_COLUMN not in tables.summary.columns

    collapsed = collapsed_value_rows(rows=rows, context=context, include_method_axis=False)
    assert collapsed[ASOCC_SSP_SCENARIO_COLUMN].dropna().tolist() == ["SSP2"]
    assert collapsed[PAIR_COUNT_COLUMN].tolist() == [1, 1]
    assert collapsed[AR6_CATEGORY_SCOPE_COLUMN].tolist() == ["C1", "C2"]

    attached = attach_dynamic_budget_values(
        summary_rows=collapsed.drop(columns=[VALUE_ARRAY_COLUMN]),
        value_rows=rows,
        context=context,
        include_method_axis=False,
    )
    assert attached[AR6_CATEGORY_SCOPE_COLUMN].tolist() == ["C1", "C1"]
    np.testing.assert_allclose(attached[BUDGET_VALUES_COLUMN].iloc[0], [4.0, 6.0])
    np.testing.assert_allclose(attached[BUDGET_VALUES_COLUMN].iloc[1], [4.0, 6.0])

    static_collapsed = collapsed_value_rows(
        rows=rows.assign(cc_type="static"),
        context=context,
        include_method_axis=False,
    )
    assert PAIR_COUNT_COLUMN not in static_collapsed.columns
    mixed_collapsed = collapsed_value_rows(
        rows=pd.concat([rows, rows.assign(cc_type="static")], ignore_index=True),
        context=context,
        include_method_axis=False,
    )
    assert PAIR_COUNT_COLUMN in mixed_collapsed.columns
    static_single_ssp_rows = rows.iloc[[0, 1]].drop(
        columns=["cc_model", "cc_scenario", "cc_category", AR6_CC_SSP_SCENARIO_COLUMN]
    )
    static_single_ssp_rows = static_single_ssp_rows.assign(cc_type="static", year=2030)
    static_single_ssp = _collapsed_inter_method_rows(
        rows=static_single_ssp_rows,
        context=context,
    )
    assert len(static_single_ssp) == 1
    assert static_single_ssp[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2"]

    indexed_rows = rows.copy()
    indexed_rows["cc_model"] = ["model_a", "model_a"]
    indexed_rows["cc_scenario"] = ["scenario_a", "scenario_a"]
    indexed_rows["__run_indices"] = [
        np.array([0, 1], dtype=np.int64),
        np.array([0, 1], dtype=np.int64),
    ]
    indexed_rows[VALUE_ARRAY_COLUMN] = [
        np.array([1.0, 2.0], dtype=np.float64),
        np.array([3.0, 4.0], dtype=np.float64),
    ]
    indexed_attached = attach_dynamic_budget_values(
        summary_rows=collapsed_value_rows(
            rows=indexed_rows,
            context=context,
            include_method_axis=True,
        ).drop(columns=[VALUE_ARRAY_COLUMN, "__run_indices"]),
        value_rows=indexed_rows,
        context=context,
        include_method_axis=True,
    )
    np.testing.assert_allclose(indexed_attached[BUDGET_VALUES_COLUMN].iloc[0], [4.0, 6.0])

    dynamic_without_pair_columns = collapsed_value_rows(
        rows=rows.drop(columns=["cc_model", "cc_scenario"]),
        context=context,
        include_method_axis=True,
    )
    assert dynamic_without_pair_columns[PAIR_COUNT_COLUMN].tolist() == [1, 1]

    inter_method_dynamic = _collapsed_inter_method_rows(rows=rows, context=context)
    assert not inter_method_dynamic.empty

    plain_category_scope = collapsed_value_rows(
        rows=rows.drop(columns=["cc_model", "cc_scenario", "cc_category"]),
        context=FigureContext(
            manifest=context.manifest,
            paths=context.paths,
            figures_root=context.figures_root,
            requested_years=context.requested_years,
            requested_asocc_ssps=context.requested_asocc_ssps,
            fu_code=context.fu_code,
            output_format=context.output_format,
            figure_output_format=context.figure_output_format,
            figure_dpi=context.figure_dpi,
            per_method=context.per_method,
            multi_method=context.multi_method,
            inter_method=context.inter_method,
            active_sources=context.active_sources,
            run_layout=context.run_layout,
            dynamic_category_uncertainty_active=False,
        ),
        include_method_axis=False,
    )
    assert AR6_CATEGORY_SCOPE_COLUMN not in plain_category_scope.columns
    assert PAIR_COUNT_COLUMN not in plain_category_scope.columns


def test_uncertainty_acc_figure_renderers_cover_selector_and_budget_labels(
    project_repo: Path,
    tmp_path: Path,
) -> None:
    del project_repo
    frame = pd.DataFrame(
        {
            "year": [2020, 2021, 2020, 2021],
            "lcia_method": ["pb_lcia"] * 4,
            "impact": ["SOD", "SOD", "AAL", "AAL"],
            "impact_unit": ["km2 yr", "km2 yr", "PDF yr", "PDF yr"],
            "cc_type": ["static"] * 4,
            "cc_bound": ["min_cc"] * 4,
            "__method": ["A"] * 4,
            "mean": [1.0, 2.0, 1.5, 2.5],
            "std": [0.1] * 4,
            "min": [0.8, 1.8, 1.3, 2.3],
            "p5": [0.85, 1.85, 1.35, 2.35],
            "p25": [0.9, 1.9, 1.4, 2.4],
            "median": [1.0, 2.0, 1.5, 2.5],
            "p75": [1.1, 2.1, 1.6, 2.6],
            "p95": [1.15, 2.15, 1.65, 2.65],
            "max": [1.2, 2.2, 1.7, 2.7],
            ASOCC_SSP_SCENARIO_COLUMN: ["", "SSP2", "", "SSP2"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [
                "historical",
                "regression_proj",
                "historical",
                "regression_proj",
            ],
        }
    )

    assert acc_scope_stem(
        "multi_method",
        frame,
        selector_token="rp_FR__sp_D",
        studied_year=2030,
    ).startswith("multi_method__rp_FR__sp_D__pb_lcia")
    assert "France electricity" in scope_title(
        "aCC uncertainty",
        "demo",
        frame,
        include_impact=False,
        selector_title="France electricity",
    )
    assert "France electricity" not in scope_title(
        "aCC uncertainty",
        "demo",
        frame,
        include_impact=False,
        selector_title=" ",
    )
    assert acc_scope_stem("multi_method", frame, include_impact=False).startswith(
        "multi_method__pb_lcia"
    )
    gwp_frame = frame.loc[frame["impact"].eq("SOD")].copy()
    gwp_frame["lcia_method"] = "gwp100_lcia"
    gwp_frame["impact"] = "GWP_100"
    gwp_title = scope_title(
        "aCC uncertainty",
        "demo",
        gwp_frame,
        include_impact=False,
        selector_title="France electricity",
    )
    assert "Climate change (GWP_100)" in gwp_title
    assert "gwp100_lcia" not in gwp_title
    assert ordered_impacts(frame.loc[frame["impact"].eq("SOD")]) == ["SOD"]
    assert method_labels(
        pd.DataFrame({"l1_l2_method": ["EG(Pop)_UT(FD)", "PR(GDPcap)_UT(FD)"]})
    ).tolist() == ["EG(Pop)_UT(FD)", "PR(GDPcap)_UT(FD)"]
    fig, hidden_axis = plt.subplots()
    format_year_axis(hidden_axis, years=[2020, 2021], show_labels=False)
    assert not any(label.get_visible() for label in hidden_axis.get_xticklabels())
    plt.close(fig)

    paths = plot_band_scope(
        frame=frame,
        output_stem=tmp_path / "multi_impact_band",
        title="multi impact",
        dpi=10,
        output_format="png",
        group_legend=False,
        include_impact_in_label=False,
    )
    assert all(path.exists() for path in paths)

    single_panel_paths = plot_band_scope(
        frame=frame.loc[frame["impact"].eq("SOD")].copy(),
        output_stem=tmp_path / "single_impact_band",
        title="single impact",
        dpi=10,
        output_format="png",
        group_legend=False,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    two_method_frame = pd.concat(
        [
            frame.loc[frame["impact"].eq("SOD")].copy(),
            frame.loc[frame["impact"].eq("SOD")].assign(
                __method="B",
                mean=lambda data: data["mean"] + 0.25,
                median=lambda data: data["median"] + 0.25,
                p5=lambda data: data["p5"] + 0.25,
                p25=lambda data: data["p25"] + 0.25,
                p75=lambda data: data["p75"] + 0.25,
                p95=lambda data: data["p95"] + 0.25,
            ),
        ],
        ignore_index=True,
    )
    two_method_paths = plot_band_scope(
        frame=two_method_frame,
        output_stem=tmp_path / "two_method_band",
        title="two method",
        dpi=10,
        output_format="png",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    assert all(path.exists() for path in [*single_panel_paths, *two_method_paths])

    fig, axis = plt.subplots()
    dynamic_budget_frame = frame.loc[frame["impact"].eq("SOD")].copy()
    dynamic_budget_frame["cc_type"] = "dynamic_ar6"
    dynamic_budget_frame[PAIR_COUNT_COLUMN] = 2
    dynamic_budget_frame[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN] = "srs"
    dynamic_budget_frame[BUDGET_VALUES_COLUMN] = [
        np.array([1.0, 1.2]),
        np.array([2.0, 2.2]),
    ]
    _render_dynamic_budget_axis(
        axis=axis,
        frame=dynamic_budget_frame,
        include_method_in_label=True,
    )
    assert [label.get_text() for label in axis.get_xticklabels()] == ["A"]
    facecolors = [
        mcolors.to_hex(cast(Any, collection.get_facecolor()[0]))
        for collection in axis.collections
        if len(collection.get_facecolor()) > 0
    ]
    assert DEFAULT_SINGLE_SERIES_COLOR.lower() in facecolors
    plt.close(fig)

    dynamic_two_method = pd.concat(
        [
            dynamic_budget_frame,
            dynamic_budget_frame.assign(
                __method="B",
                mean=lambda data: data["mean"] + 0.25,
                median=lambda data: data["median"] + 0.25,
                p5=lambda data: data["p5"] + 0.25,
                p25=lambda data: data["p25"] + 0.25,
                p75=lambda data: data["p75"] + 0.25,
                p95=lambda data: data["p95"] + 0.25,
            ),
        ],
        ignore_index=True,
    )
    dynamic_paths = plot_band_scope(
        frame=dynamic_two_method,
        output_stem=tmp_path / "dynamic_two_method_band",
        title="dynamic two method",
        dpi=10,
        output_format="png",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    assert all(path.exists() for path in dynamic_paths)
    dynamic_mean_paths = plot_mean_line_scope(
        frame=dynamic_two_method,
        requested_years=[2020, 2021],
        output_stem=tmp_path / "dynamic_two_method_mean",
        title="dynamic two method mean",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    assert dynamic_mean_paths == [tmp_path / "dynamic_two_method_mean.svg"]
    assert dynamic_mean_paths[0].exists()
    dynamic_no_sampling = dynamic_two_method.copy()
    dynamic_no_sampling[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN] = pd.NA
    assert plot_mean_line_scope(
        frame=dynamic_no_sampling,
        requested_years=[2020, 2021],
        output_stem=tmp_path / "dynamic_two_method_mean_no_sampling",
        title="dynamic two method mean no sampling",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    ) == [tmp_path / "dynamic_two_method_mean_no_sampling.svg"]
    assert plot_band_scope(
        frame=dynamic_no_sampling,
        output_stem=tmp_path / "dynamic_two_method_band_no_sampling",
        title="dynamic two method band no sampling",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    ) == [tmp_path / "dynamic_two_method_band_no_sampling.svg"]

    min_max_band_frame = pd.concat(
        [
            frame.loc[frame["impact"].eq("SOD")].assign(cc_bound="min_cc"),
            frame.loc[frame["impact"].eq("SOD")].assign(cc_bound="max_cc"),
        ],
        ignore_index=True,
    )
    min_max_band_paths = plot_band_scope(
        frame=min_max_band_frame,
        output_stem=tmp_path / "single_impact_min_max_band",
        title="single impact min max",
        dpi=1,
        output_format="svg",
        group_legend=False,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    assert min_max_band_paths == [tmp_path / "single_impact_min_max_band.svg"]
    assert min_max_band_paths[0].exists()

    violin_frame = pd.concat(
        [
            frame.loc[frame["year"].eq(2020)].assign(
                __method="A",
                **{VALUE_ARRAY_COLUMN: [np.array([0.8, 1.0, 1.2])] * 2},
            ),
            frame.loc[frame["year"].eq(2020)].assign(
                __method="B",
                mean=lambda data: data["mean"] + 0.25,
                **{VALUE_ARRAY_COLUMN: [np.array([1.0, 1.2, 1.4])] * 2},
            ),
        ],
        ignore_index=True,
    )
    violin_paths = plot_violin_scope(
        frame=violin_frame,
        output_stem=tmp_path / "multi_impact_grouped_violin",
        title="multi impact grouped violin",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    assert violin_paths == [tmp_path / "multi_impact_grouped_violin.svg"]
    assert violin_paths[0].exists()


def test_uncertainty_acc_static_fixed_sobol_outputs(allocation_dummy_repo) -> None:
    manifest = uncertainty_acc(
        **_static_kwargs(project_name="acc_uncertainty_static_sobol"),
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 4},
            "convergence": {"active": False},
        },
        figures=False,
        subfigures=False,
        refresh=True,
    ).manifest

    assert manifest.family == "acc"
    assert manifest.completed_runs == 2
    assert manifest.artifacts is not None
    assert manifest.deterministic_prerequisites[0]["component_inventory"]["role"] == (
        "component_inventory"
    )
    assert manifest.deterministic_prerequisites[0]["completed_runs"] == 2
    asocc_manifest = read_manifest(
        path=Path(manifest.deterministic_prerequisites[0]["scope_manifest"])
    )
    assert asocc_manifest.run_id == manifest.run_id
    assert manifest.artifacts["public_output"] is not None
    assert manifest.sobol is not None and manifest.sobol["ran"] is True
    assert manifest.sobol["selected_output_years"] == [2030]
    root = _run_root(
        repo_root=allocation_dummy_repo.repo_root,
        project_name="acc_uncertainty_static_sobol",
        run_id=manifest.run_id,
        source="exiobase_396_ixi",
    )
    identity = pd.read_csv(manifest.artifacts["public_row_identity"])
    runs = pd.read_csv(manifest.artifacts["acc_runs"])
    summary = pd.read_csv(manifest.artifacts["summary_stats_runs"])
    sobol = pd.read_csv(manifest.artifacts["sobol_indices"])
    sobol_summary = pd.read_csv(manifest.artifacts["sobol_source_summary"])

    assert root.exists()
    assert {"cc_type", "cc_bound", "impact", "impact_unit", "year"}.issubset(identity.columns)
    assert set(identity["cc_type"]) == {"static"}
    assert set(identity["cc_bound"]) == {"min_cc", "max_cc"}
    assert manifest.artifacts["public_output"]["acc_runs"]["layout"] == "sparse_selected_rows"
    assert list(runs.columns) == ["run_index", "public_row_id", "acc"]
    assert "l2_reuse_year" not in summary.columns
    assert set(summary[ACC_SUMMARY_SCOPE_COLUMN]) == {
        ACC_SUMMARY_SCOPE_PER_METHOD,
        ACC_SUMMARY_SCOPE_INTER_METHOD,
    }
    inter_summary = summary.loc[
        summary[ACC_SUMMARY_SCOPE_COLUMN].eq(ACC_SUMMARY_SCOPE_INTER_METHOD)
    ]
    assert inter_summary[["l1_l2_method", "l1_method", "l2_method"]].isna().all().all()
    assert (
        summary.loc[
            summary[ACC_SUMMARY_SCOPE_COLUMN].eq(ACC_SUMMARY_SCOPE_PER_METHOD),
            "l1_l2_method",
        ]
        .notna()
        .any()
    )
    per_method_summary = summary.loc[
        summary[ACC_SUMMARY_SCOPE_COLUMN].eq(ACC_SUMMARY_SCOPE_PER_METHOD)
    ]
    assert len(per_method_summary) == len(identity)
    assert len(inter_summary) < len(identity)
    assert set(sobol["source_name"]) == {
        "asocc::inter_method_uncertainty",
        "asocc::projection_uncertainty",
    }
    assert "summary_level" in sobol_summary.columns
    assert "lcia_method" in sobol_summary.columns
    assert "sobol_source_summary" in manifest.artifacts["public_output"]
    assert not (root / "results" / "sobol" / "sobol_methods.csv").exists()
    assert not list(root.glob(".summary_values_*.dat"))


def test_uncertainty_acc_static_fixed_uses_deterministic_components(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    kwargs = _static_kwargs(project_name="acc_uncertainty_static_deterministic_components")
    kwargs["years"] = [2005]
    kwargs["base_asocc_args"] = {
        "method_plan": "one_step",
        "one_step_methods": ["UT(TD)"],
        "include_lcia_based_allocation_methods": False,
    }
    kwargs["uncertainty_config"] = {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 1}, "convergence": {"active": False}},
        **_inactive_default_sources(),
    }
    kwargs["base_cc_args"] = {"static": {"exclude_max_cc": True}}

    manifest = uncertainty_acc(
        **kwargs,
        sobol_parameters={"active": False},
        figures=False,
        subfigures=False,
        refresh=True,
    ).manifest

    assert manifest.active_sources == ()
    assert manifest.completed_runs == 1
    assert any(
        item["base_function_source"] == "deterministic_asocc"
        for item in manifest.deterministic_prerequisites
    )


def test_uncertainty_acc_external_asocc_monte_carlo_routes_to_component_inventory(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    project_name = "acc_uncertainty_external_asocc_mc"
    external_dir = get_asocc_external_method_level_dir(
        proj_base=outputs_project_root(project_name=project_name),
        storage_mode="monte_carlo",
        level="level_2",
    )
    external_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "run_index": [0],
            "year": [2005],
            ASOCC_SSP_SCENARIO_COLUMN: [None],
            "r_p": ["FR"],
            "s_p": ["D"],
            "value": [0.5],
        }
    ).to_csv(external_dir / "l1_CO(S)_l2_UT(FD).csv", index=False)

    kwargs = _static_kwargs(project_name=project_name)
    kwargs["years"] = [2005]
    kwargs["fu_code"] = "L2.a.a"
    kwargs["base_asocc_args"] = {
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
        "include_lcia_based_allocation_methods": False,
    }
    kwargs["external_method"] = {"l1_l2_pairs": ["CO(S)::UT(FD)"]}
    kwargs["uncertainty_config"] = {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 1}, "convergence": {"active": False}},
        **_inactive_default_sources(),
    }

    manifest = uncertainty_acc(
        **kwargs,
        sobol_parameters={"active": False},
        figures=False,
        subfigures=False,
        refresh=True,
    ).manifest

    assert any(
        item["base_function_source"] == "uncertainty_asocc"
        for item in manifest.deterministic_prerequisites
    )


def test_uncertainty_acc_static_rejects_missing_requested_max_cc(
    allocation_dummy_repo,
) -> None:
    static_path = static_cc_csv_path(lcia_method="gwp100_lcia")
    original = pd.read_csv(static_path)
    original.drop(columns=["max_cc"], errors="ignore").to_csv(static_path, index=False)
    kwargs = _static_kwargs(project_name="acc_uncertainty_missing_static_max")
    kwargs["years"] = [2005]
    kwargs["uncertainty_config"] = {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 1}, "convergence": {"active": False}}
    }

    with pytest.raises(ValueError, match="requires a 'max_cc' column"):
        uncertainty_acc(**kwargs, refresh=True)


def test_uncertainty_acc_dynamic_fixed_outputs(
    allocation_dummy_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_dynamic_acc_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2020, 2021],
        scenario_years=[2030],
    )

    manifest = uncertainty_acc(
        project_name="acc_uncertainty_dynamic",
        years=range(2020, 2022),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "ssp_scenario": ["SSP2"],
            "include_lcia_based_allocation_methods": False,
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C1"], "ssp_scenario": ["SSP1"]},
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            **_inactive_default_sources(),
        },
        output_format="csv_compact",
        figures=False,
        subfigures=False,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest
    runtime_output = capsys.readouterr().out

    assert manifest.family == "acc"
    assert runtime_output.strip()
    assert manifest.completed_runs == 1
    assert manifest.sobol is not None
    assert manifest.sobol["ran"] is False
    assert manifest.artifacts is not None
    assert (
        Path(manifest.artifacts["scope_manifest"]).parent / "composite_phase_index.json"
    ).exists()
    identity = read_uncertainty_table(
        path=Path(manifest.artifacts["public_row_identity"]),
        output_format="csv_compact",
    )
    runs = read_uncertainty_table(
        path=Path(manifest.artifacts["acc_runs"]),
        output_format="csv_compact",
    )
    source_methods = pd.read_csv(manifest.artifacts["source_methods"])
    readme_text = Path(manifest.artifacts["results_readme"]).read_text(encoding="utf-8")

    assert {
        "cc_type",
        "cc_category",
        "ar6_cc_ssp_scenario",
        "year",
    }.issubset(identity.columns)
    assert {"public_row_id_x", "public_row_id_y"}.isdisjoint(identity.columns)
    assert set(identity["cc_type"]) == {"dynamic_ar6"}
    assert set(identity["cc_category"]) == {"C1"}
    assert set(identity["ar6_cc_ssp_scenario"]) == {"SSP1"}
    assert set(identity["impact_unit"]) == {"kg CO2-eq"}
    value_columns = [column for column in runs.columns if column != "run_index"]
    assert set(value_columns) == set(identity["public_row_id"].astype(str))
    assert runs["run_index"].nunique() == 1
    assert not bool(runs[value_columns].isna().to_numpy().any())
    assert runs[value_columns].to_numpy().max() > 1e6
    assert "acc" in set(source_methods["source_component"])
    assert all(len(line) <= 100 for line in readme_text.splitlines())


def test_uncertainty_acc_dynamic_figures_cover_public_category_uncertainty(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_acc_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2020, 2021],
        scenario_years=[2030],
    )
    explorer_path = (
        allocation_dummy_repo.repo_root
        / "data_raw"
        / "carrying_capacities"
        / "dynamic_climate_change_ar6"
        / "ar6_public_explorer.csv"
    )
    explorer = pd.read_csv(explorer_path)
    m1_rows = explorer["model"].astype(str).eq("M1")
    explorer.loc[m1_rows, "Ssp_family"] = 2
    explorer.to_csv(explorer_path, index=False)

    manifest = uncertainty_acc(
        project_name="acc_uncertainty_dynamic_category_figures",
        years=range(2020, 2022),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.c",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["UT(GVA)", "AR(E^{PBA})"],
            "ssp_scenario": ["SSP2"],
            "include_lcia_based_allocation_methods": False,
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {
                "category": ["C1", "C2"],
                "ssp_scenario": ["SSP2"],
            },
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {
                "category_uncertainty": True,
                "sampling_method": "srs",
            },
        },
        output_format="csv_compact",
        figures=True,
        subfigures=False,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    paths = [Path(path) for path in manifest.artifacts["figure_paths"]]
    assert paths
    assert all(path.exists() for path in paths)


def test_uncertainty_acc_dynamic_asocc_only_keeps_deterministic_cc_rows(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_acc_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
        scenario_years=[2025],
    )

    manifest = uncertainty_acc(
        project_name="acc_uncertainty_dynamic_asocc_only",
        years=range(2024, 2026),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "reference_years": [2019],
            "projection_mode": "historical_reuse",
            "reg_window": [2018, 2019],
            "l2_reuse_years": [2018, 2019],
            "ssp_scenario": ["SSP2"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C2"], "ssp_scenario": ["SSP2"]},
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            **_only_projection_source(),
        },
        output_format="csv_compact",
        figures=False,
        subfigures=False,
        figure_format=_fast_figure_format(),
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 4},
            "convergence": {"active": False},
        },
        refresh=True,
    ).manifest

    assert manifest.family == "acc"
    assert manifest.artifacts is not None
    assert manifest.active_sources == ("asocc::projection_uncertainty",)
    assert manifest.sobol is not None
    assert manifest.sobol["ran"] is False
    assert manifest.sobol["active_source_count"] == 1
    assert manifest.artifacts is not None
    identity = read_uncertainty_table(
        path=Path(manifest.artifacts["public_row_identity"]),
        output_format="csv_compact",
    )
    runs = read_uncertainty_table(
        path=Path(manifest.artifacts["acc_runs"]),
        output_format="csv_compact",
    )
    source_methods = pd.read_csv(manifest.artifacts["source_methods"])

    assert {"cc_model", "cc_scenario", "cc_category", "ar6_cc_ssp_scenario"}.issubset(
        identity.columns
    )
    assert {"public_row_id_x", "public_row_id_y"}.isdisjoint(identity.columns)
    assert set(identity["cc_category"]) == {"C2"}
    assert set(identity["ar6_cc_ssp_scenario"]) == {"SSP2"}
    assert runs.shape == (2, 1 + len(identity))
    assert "ar6_cc" not in set(source_methods["source_component"])


def test_uncertainty_acc_convergence_reports_reuse_and_reached_modes(
    allocation_dummy_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del allocation_dummy_repo
    unreached_kwargs = _static_kwargs(project_name="acc_uncertainty_convergence_unreached")
    unreached_kwargs["years"] = [2005]
    unreached_kwargs["figures"] = False
    unreached_kwargs["uncertainty_config"] = {
        "mc_parameters": {
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 2, "rtol": 1e-12, "stable_runs": 2},
        },
        **_inactive_default_sources(),
    }
    unreached = uncertainty_acc(**unreached_kwargs, refresh=True).manifest
    first_output = capsys.readouterr().out
    assert unreached.convergence is not None
    assert unreached.convergence["reached"] is False
    assert unreached.convergence["completed_runs"] == 2
    assert first_output.strip()

    reused = uncertainty_acc(**unreached_kwargs, refresh=False).manifest
    reuse_output = capsys.readouterr().out
    assert reused.run_id == unreached.run_id
    assert reuse_output.strip()
    stale_run_file = Path(unreached.artifacts["scope_manifest"]).parents[1] / "stale.txt"
    stale_run_file.write_text("stale", encoding="utf-8")
    stale_upstream_file = (
        Path(unreached.deterministic_prerequisites[0]["scope_manifest"]).parents[1] / "stale.txt"
    )
    stale_upstream_file.write_text("stale", encoding="utf-8")
    refreshed = uncertainty_acc(**unreached_kwargs, refresh=True).manifest
    assert refreshed.status == "complete"
    assert not stale_run_file.exists()
    assert not stale_upstream_file.exists()

    compact_kwargs = _static_kwargs(project_name="acc_uncertainty_convergence_compact")
    compact_kwargs["years"] = [2005]
    compact_kwargs["figures"] = False
    compact_kwargs["uncertainty_config"] = {
        "mc_parameters": {
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 4, "stable_runs": 2},
        },
        **_inactive_default_sources(),
    }
    compact = uncertainty_acc(**compact_kwargs, refresh=True).manifest
    assert compact.convergence is not None
    assert compact.convergence["reached"] is True
    assert compact.artifacts["public_output"] is not None
    assert compact.artifacts["public_output"]["acc_runs"]["layout"] == "compact_run_matrix"


def test_acc_sparse_writer_preserves_empty_requested_runs(tmp_path: Path) -> None:
    asocc_root = tmp_path / "asocc_run" / "results"
    asocc_root.mkdir(parents=True)
    asocc_identity_path = asocc_root / "public_row_identity.csv"
    asocc_runs_path = asocc_root / "asocc_runs.csv"
    asocc_identity_path.write_text("public_row_id,year\n0,2030\n", encoding="utf-8")
    with SparseRunRowsWriter(path=asocc_runs_path, output_format="csv_compact") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.empty(0, dtype=np.int64),
                public_row_id=np.empty(0, dtype=np.int64),
                values=np.empty(0, dtype=np.float64),
                value_column="asocc",
            ),
            batch_index=0,
        )
    empty_path = asocc_root / "empty.csv"
    empty_path.write_text("", encoding="utf-8")
    asocc_manifest = build_manifest(
        family="asocc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("inter_method_uncertainty",),
        artifacts={
            "scope_manifest": str(tmp_path / "asocc_run" / "logs" / "scope_manifest.json"),
            "public_row_identity": str(asocc_identity_path),
            "asocc_runs": str(asocc_runs_path),
            "summary_stats_runs": str(empty_path),
            "results_readme": str(empty_path),
            "source_methods": str(empty_path),
            "public_output": {"asocc_runs": {"layout": "sparse_selected_rows"}},
        },
    )
    identity = pd.DataFrame(
        {
            "public_row_id": [0],
            "cc_type": ["static"],
            "lcia_method": ["gwp100_lcia"],
            "cc_bound": ["min_cc"],
            "year": [2030],
            "impact": ["GWP_100"],
            "impact_unit": ["kg CO2-eq"],
        }
    )
    plan = ACCUncertaintyPlan(
        identity=identity,
        summary_identity=identity,
        summary_public_row_groups=(("0",),),
        branch_plans=(
            ACCBranchPlan(
                identity=identity,
                asocc_positions=np.array([0], dtype=np.int64),
                cc_positions=None,
                static_cc_values=np.array([1.0], dtype=np.float64),
                dynamic_cc_factors=None,
                cc_type="static",
                cc_source="gwp100_lcia",
            ),
        ),
        asocc_input=ACCAsoccInput(
            identity=None,
            deterministic_values=None,
            manifest=asocc_manifest,
            deterministic_manifest_path=None,
            reuse_status="computed",
        ),
        dynamic_cc_input=None,
        acc_run_layout="sparse_selected_rows",
        deterministic_cc_values=None,
        source_method_rows=pd.DataFrame(),
        active_sources=("asocc::inter_method_uncertainty",),
    )
    paths = build_acc_uncertainty_run_paths(
        monte_carlo_root=tmp_path / "acc" / "monte_carlo",
        run_id="mc_empty_sparse",
        output_format="csv_compact",
    )
    runtime = normalize_uncertainty_request(
        family="acc",
        output_format="csv_compact",
        mc_parameters={"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}},
    )
    context = build_acc_manifest_context(
        base_args={},
        runtime=runtime,
        plan=plan,
        sobol_status={"ran": False},
    )
    assert "component_inventory" not in context["deterministic_prerequisites"][0]

    completed, convergence = write_acc_run_outputs(paths=paths, plan=plan, runtime=runtime)

    assert completed == 2
    assert convergence is None
    assert pd.read_csv(paths.public_runs).empty
    summary = pd.read_csv(paths.summary_stats_runs)
    assert len(summary) == 1
    assert pd.isna(summary.loc[0, "mean"])

    convergence_paths = build_acc_uncertainty_run_paths(
        monte_carlo_root=tmp_path / "acc" / "monte_carlo",
        run_id="mc_empty_sparse_convergence",
        output_format="csv_compact",
    )
    convergence_runtime = normalize_uncertainty_request(
        family="acc",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 4, "stable_runs": 2},
        },
    )
    completed, convergence = write_acc_run_outputs(
        paths=convergence_paths,
        plan=plan,
        runtime=convergence_runtime,
    )

    assert completed == 4
    assert convergence is not None
    assert convergence["reached"] is True


def test_uncertainty_acc_sobol_convergence_reports_unreached(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    manifest = uncertainty_acc(
        **_static_kwargs(project_name="acc_uncertainty_sobol_convergence_unreached"),
        sobol_parameters={
            "active": True,
            "fixed": {"active": False},
            "convergence": {"active": True, "max_base_samples": 128},
        },
        figures=False,
        subfigures=False,
        refresh=True,
    ).manifest

    assert manifest.sobol is not None
    assert manifest.sobol["ran"] is True
    assert manifest.sobol["reached"] is False


def test_uncertainty_acc_dynamic_sobol_covers_dynamic_and_deterministic_cc(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_acc_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2018, 2019],
        scenario_years=[2030, 2031],
    )

    dynamic_cc = uncertainty_acc(
        project_name="acc_uncertainty_dynamic_sobol_cc_source",
        years=range(2030, 2032),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "reference_years": [2019],
            "projection_mode": "historical_reuse",
            "reg_window": [2018, 2019],
            "l2_reuse_years": [2018, 2019],
            "ssp_scenario": ["SSP2"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C2"], "ssp_scenario": ["SSP2"]},
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            **_only_projection_source(),
            "dynamic_ar6_cc_uncertainty": {"sampling_method": "srs"},
        },
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 1},
            "convergence": {"active": False},
        },
        figures=False,
        subfigures=False,
        refresh=True,
    ).manifest
    assert dynamic_cc.sobol is not None
    assert dynamic_cc.sobol["ran"] is True
    assert "ar6_cc::dynamic_ar6_cc_uncertainty" in dynamic_cc.active_sources
    assert dynamic_cc.artifacts is not None
    dynamic_cc_sobol = pd.read_csv(dynamic_cc.artifacts["sobol_indices"])
    assert "ar6_cc::dynamic_ar6_cc_uncertainty" in set(dynamic_cc_sobol["source_name"])

    deterministic_cc = uncertainty_acc(
        project_name="acc_uncertainty_dynamic_sobol_deterministic_cc",
        years=range(2030, 2032),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "reference_years": [2019],
            "projection_mode": "historical_reuse",
            "reg_window": [2018, 2019],
            "l2_reuse_years": [2018, 2019],
            "ssp_scenario": ["SSP2"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C2"], "ssp_scenario": ["SSP2"]},
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            **_only_projection_source(),
            "inter_method_uncertainty": {},
            "dynamic_ar6_cc_uncertainty": {"active": False},
        },
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 1},
            "convergence": {"active": False},
        },
        figures=False,
        refresh=True,
    ).manifest
    assert deterministic_cc.sobol is not None
    assert deterministic_cc.sobol["ran"] is True
    assert "ar6_cc::dynamic_ar6_cc_uncertainty" not in deterministic_cc.active_sources


def test_uncertainty_acc_dynamic_convergence_refreshes_component_inventories(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_acc_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2018, 2019],
        scenario_years=[2030, 2031],
    )

    manifest = uncertainty_acc(
        project_name="acc_uncertainty_dynamic_convergence_subfigures",
        years=range(2030, 2032),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "reference_years": [2019],
            "projection_mode": "historical_reuse",
            "reg_window": [2018, 2019],
            "l2_reuse_years": [2018, 2019],
            "ssp_scenario": ["SSP2"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C2"], "ssp_scenario": ["SSP2"]},
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "rtol": 1e-12, "stable_runs": 1},
            },
            "projection_uncertainty": {},
            "dynamic_ar6_cc_uncertainty": {"sampling_method": "srs"},
        },
        output_format="csv_compact",
        sobol_parameters={"active": False},
        figures=True,
        figure_format=_fast_figure_format(),
        subfigures=True,
        refresh=True,
    ).manifest

    assert manifest.convergence is not None
    assert manifest.completed_runs == 2
    assert {
        "asocc::projection_uncertainty",
        "ar6_cc::dynamic_ar6_cc_uncertainty",
    }.issubset(manifest.active_sources)

    deterministic_asocc_manifest = uncertainty_acc(
        project_name="acc_uncertainty_convergence_deterministic_asocc_subfigures",
        years=range(2030, 2032),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "reference_years": [2019],
            "projection_mode": "historical_reuse",
            "reg_window": [2018, 2019],
            "l2_reuse_years": [2019],
            "ssp_scenario": ["SSP2"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C2"], "ssp_scenario": ["SSP2"]},
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "stable_runs": 1},
            },
            "inter_method_uncertainty": {"active": False},
            "projection_uncertainty": {"active": False},
            "reference_year_uncertainty": {"active": False},
        },
        output_format="csv_compact",
        sobol_parameters={"active": False},
        figures=True,
        figure_options={"per_method": False, "multi_method": True, "inter_method": False},
        figure_format=_fast_figure_format(),
        subfigures=True,
        refresh=True,
    ).manifest
    assert deterministic_asocc_manifest.completed_runs == 2


def test_uncertainty_acc_static_figures_are_public_for_single_and_multi_year(
    allocation_dummy_repo_factory,
) -> None:
    allocation_dummy_repo_factory(name="acc_uncertainty_public_figures")
    base_kwargs: dict[str, Any] = {
        "project_name": "acc_uncertainty_public_figures",
        "lcia_method": "gwp100_lcia",
        "fu_code": "L2.a.b",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "ssp_scenario": ["SSP2"],
            "include_lcia_based_allocation_methods": False,
        },
        "base_cc_args": {"static": {"exclude_max_cc": False}},
        "uncertainty_config": {
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 10},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
        },
        "output_format": "csv_compact",
        "figures": True,
        "subfigures": False,
        "figure_format": _fast_figure_format(),
    }
    no_figure_single = uncertainty_acc(
        years=[2005],
        refresh=True,
        **{**base_kwargs, "figures": False, "subfigures": True},
    ).manifest
    assert no_figure_single.artifacts is not None
    assert not no_figure_single.artifacts.get("figure_paths")

    single = uncertainty_acc(years=[2005], refresh=False, **base_kwargs).manifest
    assert single.artifacts is not None
    single_paths = [Path(path) for path in single.artifacts["figure_paths"]]

    assert single_paths
    assert all(path.exists() for path in single_paths)
    assert any("inter_method" in path.parts for path in single_paths)
    assert any("multi_method" in path.parts for path in single_paths)
    assert any("per_method" in path.parts for path in single_paths)
    assert any(path.name.endswith("__2005.png") for path in single_paths)
    reused_single = uncertainty_acc(years=[2005], refresh=False, **base_kwargs).manifest
    assert reused_single.artifacts is not None
    assert reused_single.artifacts["figure_paths"]
    no_product_single = uncertainty_acc(
        years=[2005],
        refresh=False,
        **{
            **base_kwargs,
            "figure_options": {
                "per_method": False,
                "multi_method": False,
                "inter_method": False,
            },
        },
    ).manifest
    assert no_product_single.artifacts is not None
    assert no_product_single.artifacts["figure_paths"] == []

    multi = uncertainty_acc(
        **{
            **base_kwargs,
            "project_name": "acc_uncertainty_public_figures_multi",
            "base_cc_args": {"static": {"exclude_max_cc": True}},
        },
        years=[2005, 2006],
        refresh=True,
    ).manifest
    assert multi.artifacts is not None
    multi_paths = [Path(path) for path in multi.artifacts["figure_paths"]]

    assert multi_paths
    assert all(path.exists() for path in multi_paths)
    assert any("inter_method" in path.parts for path in multi_paths)
    assert any("multi_method" in path.parts for path in multi_paths)
    assert any("per_method" in path.parts for path in multi_paths)


def test_uncertainty_acc_static_figures_cover_public_asocc_transition(
    allocation_dummy_repo_factory,
) -> None:
    repo = allocation_dummy_repo_factory(name="acc_uncertainty_public_transition_figures")
    repo.set_processed_pop_gdp_years(
        historical_years=[2020, 2021, 2022, 2023, 2024],
        scenario_years=[2025],
    )
    repo.write_mrio_metadata(
        source="exiobase_3102_ixi",
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=[2020, 2021, 2022, 2023, 2024],
    )
    repo.write_mrio_history(
        source="exiobase_3102_ixi",
        matrix_version=None,
        years=[2020, 2021, 2022, 2023, 2024],
    )
    repo.write_lcia_support(
        source="exiobase_3102_ixi",
        matrix_version=None,
        lcia_method="gwp100_lcia",
        available_years=[2020, 2021, 2022, 2023, 2024],
        impacts=["climate_child"],
        impact_parents={"climate_child": "GWP_100"},
    )
    manifest = uncertainty_acc(
        project_name="acc_uncertainty_public_transition_figures",
        years=[2024, 2025],
        lcia_method="gwp100_lcia",
        fu_code="L2.a.b",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_3102_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "include_lcia_based_allocation_methods": False,
            "projection_mode": "regression",
            "reg_window": [2020, 2021, 2022],
            "ssp_scenario": ["SSP2"],
        },
        base_cc_args={"static": {"exclude_max_cc": True}},
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            **_only_projection_source(),
        },
        output_format="csv_compact",
        figures=True,
        subfigures=False,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    paths = [Path(path) for path in manifest.artifacts["figure_paths"]]
    assert paths
    assert all(path.exists() for path in paths)


def test_uncertainty_acc_static_figures_cover_public_no_source_scalar_year(
    allocation_dummy_repo_factory,
) -> None:
    allocation_dummy_repo_factory(name="acc_uncertainty_public_no_source_figures")
    kwargs = _static_kwargs(project_name="acc_uncertainty_public_no_source_figures")
    kwargs["years"] = 2030
    kwargs["base_asocc_args"] = {
        **kwargs["base_asocc_args"],
        "l2_reuse_years": [2005],
    }
    kwargs["base_cc_args"] = {"static": {"exclude_max_cc": True}}
    kwargs["uncertainty_config"] = {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 1}, "convergence": {"active": False}}
    }

    manifest = uncertainty_acc(
        **kwargs,
        figures=True,
        subfigures=False,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    paths = [Path(path) for path in manifest.artifacts["figure_paths"]]
    assert paths
    assert all(path.exists() for path in paths)


def test_uncertainty_acc_static_figures_cover_public_multi_impact_min_max_violins(
    allocation_dummy_repo_factory,
) -> None:
    allocation_dummy_repo_factory(name="acc_uncertainty_public_multi_impact_violins")
    cc_path = static_cc_csv_path(lcia_method="custom_lcia")
    cc_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "impact_full_name": ["Impact X", "Impact Y", "Impact Z"],
            "impact": ["X1", "Y2", "Z3"],
            "impact_unit": ["kg", "kg", "kg"],
            "min_cc": [100.0, 200.0, 300.0],
            "max_cc": [150.0, 250.0, 350.0],
        }
    ).to_csv(cc_path, index=False)

    manifest = uncertainty_acc(
        project_name="acc_uncertainty_public_multi_impact_violins",
        years=[2005],
        lcia_method="custom_lcia",
        fu_code="L2.a.b",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "include_lcia_based_allocation_methods": False,
        },
        base_cc_args={"static": {"exclude_max_cc": False}},
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
        },
        output_format="csv_compact",
        figures=True,
        subfigures=False,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    paths = [Path(path) for path in manifest.artifacts["figure_paths"]]
    assert paths
    assert all(path.exists() for path in paths)
    assert any("inter_method" in path.parts for path in paths)

    multi_manifest = uncertainty_acc(
        project_name="acc_uncertainty_public_multi_impact_bands",
        years=[2005, 2006],
        lcia_method="custom_lcia",
        fu_code="L2.a.b",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "include_lcia_based_allocation_methods": False,
        },
        base_cc_args={"static": {"exclude_max_cc": False}},
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
        },
        output_format="csv_compact",
        figures=True,
        subfigures=False,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest

    assert multi_manifest.artifacts is not None
    multi_paths = [Path(path) for path in multi_manifest.artifacts["figure_paths"]]
    assert multi_paths
    assert all(path.exists() for path in multi_paths)


def test_uncertainty_acc_static_figures_cover_public_projection_source(
    allocation_dummy_repo_factory,
) -> None:
    repo = allocation_dummy_repo_factory(name="acc_uncertainty_public_figure_sources")
    repo.write_lcia_support(
        source="exiobase_396_ixi",
        matrix_version=None,
        lcia_method="gwp100_lcia",
        available_years=list(range(1995, 2007)),
        impacts=["climate_child"],
        impact_parents={"climate_child": "GWP_100"},
    )
    base_kwargs: dict[str, Any] = {
        "project_name": "acc_uncertainty_public_figure_sources",
        "lcia_method": "gwp100_lcia",
        "fu_code": "L2.a.b",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "projection_mode": "historical_reuse",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005, 2006],
            "ssp_scenario": "SSP2",
        },
        "base_cc_args": {"static": {"exclude_max_cc": False}},
        "output_format": "csv_compact",
        "figures": True,
        "subfigures": False,
        "figure_format": _fast_figure_format(),
    }
    projection_source = uncertainty_acc(
        **{**base_kwargs, "project_name": "acc_uncertainty_public_figure_projection_source"},
        years=[2030],
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            **_only_projection_source(),
        },
        refresh=True,
    ).manifest

    assert projection_source.artifacts is not None
    lcia_paths = [Path(path) for path in projection_source.artifacts["figure_paths"]]

    assert lcia_paths
    assert all(path.exists() for path in lcia_paths)
    assert any("multi_method" in path.parts for path in lcia_paths)
    assert any("per_method" in path.parts for path in lcia_paths)

    reference_kwargs = {
        **base_kwargs,
        "project_name": "acc_uncertainty_public_figure_reference_source",
        "fu_code": "L2.a.a",
        "years": [2030],
        "base_asocc_args": {
            "method_plan": "one_step",
            "one_step_methods": ["AR(E^{CBA_FD})"],
            "reference_years": [2005, 2006],
            "ssp_scenario": "SSP2",
        },
        "uncertainty_config": {
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            **_only_reference_year_source(),
        },
    }
    reference_source = uncertainty_acc(**reference_kwargs, refresh=True).manifest
    assert reference_source.artifacts is not None
    reference_paths = [Path(path) for path in reference_source.artifacts["figure_paths"]]
    assert reference_paths
    assert all(path.exists() for path in reference_paths)


def test_uncertainty_acc_static_figures_validate_inactive_reference_axis(
    allocation_dummy_repo_factory,
) -> None:
    repo = allocation_dummy_repo_factory(name="acc_uncertainty_public_figure_axis_validation")
    repo.write_lcia_support(
        source="exiobase_396_ixi",
        matrix_version=None,
        lcia_method="gwp100_lcia",
        available_years=list(range(1995, 2007)),
        impacts=["climate_child"],
        impact_parents={"climate_child": "GWP_100"},
    )
    base_kwargs = _static_kwargs(project_name="acc_uncertainty_public_figure_axis_validation")
    base_kwargs["fu_code"] = "L2.a.a"
    base_kwargs["base_asocc_args"] = {
        "method_plan": "one_step",
        "one_step_methods": ["AR(E^{CBA_FD})"],
        "l1_reg_aggreg": "pre",
        "reference_years": [2005, 2006],
        "ssp_scenario": ["SSP2"],
        "projection_mode": "historical_reuse",
        "reg_window": [2005, 2006],
        "l2_reuse_years": [2005],
        "include_lcia_based_allocation_methods": True,
    }
    base_kwargs["uncertainty_config"] = {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}},
        **_only_projection_source(),
    }

    with pytest.raises(ValueError, match="more than one reference year"):
        uncertainty_acc(
            **base_kwargs,
            figures=True,
            subfigures=False,
            figure_format=_fast_figure_format(),
            refresh=True,
        )


def test_uncertainty_acc_rejects_invalid_dynamic_cc_source_config(
    allocation_dummy_repo,
) -> None:
    with pytest.raises(ValueError, match="Unsupported uncertainty source names"):
        uncertainty_acc(
            **{
                **_static_kwargs(project_name="acc_uncertainty_unknown_config_key"),
                "uncertainty_config": {
                    "mc_parameters": {
                        "fixed": {"active": True, "n_runs": 1},
                        "convergence": {"active": False},
                    },
                    "not_a_source": True,
                },
            },
            refresh=True,
        )

    with pytest.raises(ValueError, match="dynamic_ar6_cc_uncertainty.*dictionary"):
        uncertainty_acc(
            project_name="acc_uncertainty_invalid_dynamic_cc_source",
            years=range(2020, 2022),
            lcia_method="gwp100_lcia",
            fu_code="L2.a.a",
            source="exiobase_396_ixi",
            base_asocc_args={
                "method_plan": "one_step",
                "one_step_methods": ["AR(E^{CBA_FD})"],
            },
            base_cc_args={
                "static": {"active": False},
                "dynamic_ar6": {"category": ["C1"], "ssp_scenario": ["SSP1"]},
            },
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 1},
                    "convergence": {"active": False},
                },
                "dynamic_ar6_cc_uncertainty": "bad",
            },
            output_format="csv_compact",
            refresh=True,
        )


def test_acc_uncertainty_normalization_keeps_absent_mc_parameters_absent() -> None:
    config = normalize_acc_uncertainty_config({"inter_method_uncertainty": {}})

    assert "mc_parameters" not in config
    assert "lcia_uncertainty" not in config
    assert "inter_mrio_uncertainty" not in config
    assert config["inter_method_uncertainty"] == {}
    assert config["projection_uncertainty"] == {}
    assert config["reference_year_uncertainty"] == {}
    assert config["dynamic_ar6_cc_uncertainty"] == {}


def test_acc_sampling_vectorized_contracts_cover_reachable_edge_paths(
    tmp_path: Path,
    project_repo: Path,
) -> None:
    del project_repo
    cc_source = "gwp100_lcia"
    cc_rows = branch_mod._static_cc_rows(cc_source=cc_source, bounds=["min_cc"])  # noqa: SLF001
    impact = str(cc_rows.loc[0, "impact"])
    static_branch = {
        "cc_type": "static",
        "cc_source": cc_source,
        "static_cc_bounds": ["min_cc"],
    }
    asocc_non_lcia = pd.DataFrame(
        {
            "public_row_id": [0],
            "year": [2030],
            "l1_l2_method": ["generic"],
            "r_p": ["FR"],
            "s_p": ["D"],
        }
    )
    non_lcia_plan = branch_mod._static_branch_plan(  # noqa: SLF001
        asocc_identity=asocc_non_lcia,
        branch=static_branch,
    )
    assert set(non_lcia_plan.identity["lcia_method"]) == {cc_source}
    assert len(non_lcia_plan.identity) == len(cc_rows)

    asocc_static = pd.DataFrame(
        {
            "public_row_id": [0],
            "lcia_method": [cc_source],
            "impact": [impact],
            "year": [2030],
            "l1_l2_method": ["tagged"],
            "r_p": ["FR"],
            "s_p": ["D"],
        }
    )

    static_plan = branch_mod._static_branch_plan(  # noqa: SLF001
        asocc_identity=asocc_static,
        branch=static_branch,
    )

    assert set(static_plan.identity["lcia_method"]) == {cc_source}
    assert len(static_plan.identity) == 1

    dynamic_branch = {"cc_type": "dynamic_ar6", "cc_source": cc_source}
    cc_identity = pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "year": [2030, 2030],
            "ssp_scenario": ["SSP1", "SSP2"],
            "cc_category": ["C1", "C1"],
            "cc_model": ["m1", "m2"],
            "cc_scenario": ["s1", "s2"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
        }
    )
    asocc_dynamic = pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "lcia_method": [cc_source, cc_source],
            "impact": [impact, impact],
            "year": [2030, 2030],
            ASOCC_SSP_SCENARIO_COLUMN: [None, "SSP2"],
            "l1_l2_method": ["generic", "specific"],
            "r_p": ["FR", "FR"],
            "s_p": ["D", "D"],
        }
    )

    dynamic_plan = branch_mod._dynamic_branch_plan(  # noqa: SLF001
        asocc_identity=asocc_dynamic,
        cc_identity=cc_identity,
        branch=dynamic_branch,
    )
    no_impact_plan = branch_mod._dynamic_branch_plan(  # noqa: SLF001
        asocc_identity=asocc_dynamic.drop(columns=["impact"]),
        cc_identity=cc_identity,
        branch=dynamic_branch,
    )

    assert len(dynamic_plan.identity) == 3
    assert len(no_impact_plan.identity) == 3
    invariant_only = branch_mod._dynamic_branch_plan(  # noqa: SLF001
        asocc_identity=asocc_dynamic.loc[[0]],
        cc_identity=cc_identity,
        branch=dynamic_branch,
    )
    specific_only = branch_mod._dynamic_branch_plan(  # noqa: SLF001
        asocc_identity=asocc_dynamic.loc[[1]],
        cc_identity=cc_identity,
        branch=dynamic_branch,
    )
    assert len(invariant_only.identity) == 2
    assert len(specific_only.identity) == 1

    with pytest.raises(
        ValueError,
        match="no overlapping aSoCC and AR6 carrying capacity year and SSP rows",
    ):
        branch_mod._dynamic_branch_plan(  # noqa: SLF001
            asocc_identity=asocc_dynamic.assign(year=2029).drop(
                columns=[ASOCC_SSP_SCENARIO_COLUMN]
            ),
            cc_identity=cc_identity,
            branch=dynamic_branch,
        )

    sparse = SparseRunRows(
        run_index=np.array([0], dtype=np.int64),
        public_row_id=np.array([0], dtype=np.int64),
        values=np.array([2.0], dtype=np.float64),
        value_column="asocc",
    )
    dynamic_sparse = sparse_render_mod.evaluate_acc_sparse_rows(
        asocc_rows=sparse,
        run_indices=np.array([0], dtype=np.int64),
        expansions=sparse_rows_mod.sparse_branch_expansions(branch_plans=(dynamic_plan,)),
        cc_values=np.array([[10.0, 20.0]], dtype=np.float64),
    )
    assert dynamic_sparse.values.tolist() == [20.0, 40.0]
    selected_dynamic_sparse = sparse_render_mod.evaluate_acc_sparse_rows(
        asocc_rows=sparse,
        run_indices=np.array([0], dtype=np.int64),
        expansions=sparse_rows_mod.sparse_branch_expansions(branch_plans=(dynamic_plan,)),
        cc_values=np.array([[np.nan, 20.0]], dtype=np.float64),
    )
    assert selected_dynamic_sparse.values.tolist() == [40.0]
    assert np.isfinite(selected_dynamic_sparse.values).all()
    empty_sparse = sparse_render_mod.evaluate_acc_sparse_rows(
        asocc_rows=SparseRunRows(
            run_index=np.array([0], dtype=np.int64),
            public_row_id=np.array([99], dtype=np.int64),
            values=np.array([2.0], dtype=np.float64),
            value_column="asocc",
        ),
        run_indices=np.array([0], dtype=np.int64),
        expansions=sparse_rows_mod.sparse_branch_expansions(branch_plans=(static_plan,)),
        cc_values=None,
    )
    assert empty_sparse.values.size == 0

    grouped = collapse_values_to_summary_groups(
        values=np.array([[1.0, 3.0], [np.nan, 4.0]], dtype=np.float64),
        public_row_groups=(("0", "1"),),
    )
    np.testing.assert_allclose(grouped[:, 0], [2.0, 4.0])
    summary_identity, summary_groups = acc_summary_identity_groups(
        identity=pd.DataFrame(
            {
                "public_row_id": [0, 1],
                "cc_category": ["C1", "C2"],
                "ar6_cc_ssp_scenario": ["SSP1", "SSP1"],
                "year": [2030, 2030],
            }
        ),
        active_sources=("ar6_cc::dynamic_ar6_cc_uncertainty",),
        dynamic_category_uncertainty_active=True,
    )
    assert "cc_category" not in summary_identity.columns
    assert summary_identity[ACC_SUMMARY_SCOPE_COLUMN].tolist() == [ACC_SUMMARY_SCOPE_PER_METHOD]
    assert summary_groups == (("0", "1"),)

    source_methods_path = tmp_path / "source_methods.csv"
    cc_source_methods_path = tmp_path / "cc_source_methods.csv"
    pd.DataFrame(
        {
            "source_component": ["asocc"],
            "source_name": ["reference_year_uncertainty"],
        }
    ).to_csv(
        source_methods_path,
        index=False,
    )
    pd.DataFrame(
        {
            "source_component": ["ar6_cc"],
            "source_name": ["category_uncertainty"],
        }
    ).to_csv(
        cc_source_methods_path,
        index=False,
    )
    asocc_manifest = build_manifest(
        family="asocc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("reference_year_uncertainty",),
        artifacts={"source_methods": str(source_methods_path)},
    )
    cc_manifest = build_manifest(
        family="ar6_cc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("category_uncertainty",),
        artifacts={
            "public_row_identity": str(tmp_path / "cc_identity.csv"),
            "source_methods": str(cc_source_methods_path),
        },
    )
    source_methods = build_acc_source_methods(
        asocc_input=ACCAsoccInput(
            identity=None,
            deterministic_values=None,
            manifest=asocc_manifest,
            deterministic_manifest_path=None,
            reuse_status="computed",
        ),
        dynamic_cc_input=ACCDynamicCCInput(
            identity=None,
            deterministic_values=None,
            manifest=cc_manifest,
            deterministic_manifest_path=None,
            reuse_status="computed",
        ),
    )
    assert source_methods.loc[0, "source_component"] == "asocc"
    assert "asocc::reference_year_uncertainty" in set(source_methods["source_name"])
    assert "ar6_cc::category_uncertainty" in set(source_methods["source_name"])
    assert "acc_formula" in set(source_methods["source_name"])


def test_acc_sparse_selected_layout_iterators_cover_source_combinations(tmp_path: Path) -> None:
    asocc_sparse_path = tmp_path / "asocc_sparse" / "results" / "asocc_runs.csv"
    asocc_sparse_path.parent.mkdir(parents=True)
    with SparseRunRowsWriter(path=asocc_sparse_path, output_format="csv_compact") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([0, 1], dtype=np.int64),
                public_row_id=np.array([0, 1], dtype=np.int64),
                values=np.array([2.0, 3.0], dtype=np.float64),
                value_column="asocc",
            ),
            batch_index=0,
        )
    asocc_compact_path = tmp_path / "asocc_compact" / "results" / "asocc_runs.csv"
    asocc_compact_path.parent.mkdir(parents=True)
    with CompactRunMatrixWriter(path=asocc_compact_path, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=np.array([0, 1], dtype=np.int64),
            values=np.array([[2.0, 4.0], [3.0, 5.0]], dtype=np.float64),
            batch_index=0,
        )
    cc_sparse_path = tmp_path / "cc_sparse" / "results" / "cc_runs.csv"
    cc_sparse_path.parent.mkdir(parents=True)
    with SparseRunRowsWriter(path=cc_sparse_path, output_format="csv_compact") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([0, 1], dtype=np.int64),
                public_row_id=np.array([0, 1], dtype=np.int64),
                values=np.array([5.0, 7.0], dtype=np.float64),
                value_column="cc",
            ),
            batch_index=0,
        )
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")

    def _asocc_manifest(*, root: str, runs_path: Path, layout: str):
        return build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=("inter_method_uncertainty",),
            artifacts={
                "scope_manifest": str(tmp_path / root / "logs" / "scope_manifest.json"),
                "public_row_identity": str(empty_path),
                "asocc_runs": str(runs_path),
                "summary_stats_runs": str(empty_path),
                "results_readme": str(empty_path),
                "source_methods": str(empty_path),
                "public_output": {"asocc_runs": {"layout": layout}},
            },
        )

    cc_manifest = build_manifest(
        family="ar6_cc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("dynamic_ar6_cc_uncertainty",),
        artifacts={
            "scope_manifest": str(tmp_path / "cc_sparse" / "logs" / "scope_manifest.json"),
            "public_row_identity": str(empty_path),
            "cc_runs": str(cc_sparse_path),
            "summary_stats_runs": str(empty_path),
            "results_readme": str(empty_path),
            "source_methods": str(empty_path),
            "public_output": {"cc_runs": {"layout": "sparse_selected_rows"}},
        },
    )
    static_branch = ACCBranchPlan(
        identity=pd.DataFrame({"public_row_id": [0]}),
        asocc_positions=np.array([0], dtype=np.int64),
        cc_positions=None,
        static_cc_values=np.array([10.0], dtype=np.float64),
        dynamic_cc_factors=None,
        cc_type="static",
        cc_source="gwp100_lcia",
    )
    dynamic_branch = ACCBranchPlan(
        identity=pd.DataFrame({"public_row_id": [1, 2]}),
        asocc_positions=np.array([0, 1], dtype=np.int64),
        cc_positions=np.array([0, 1], dtype=np.int64),
        static_cc_values=None,
        dynamic_cc_factors=np.array([1.0, 1.0], dtype=np.float64),
        cc_type="dynamic_ar6",
        cc_source="gwp100_lcia",
    )
    unmatched_dynamic_branch = ACCBranchPlan(
        identity=pd.DataFrame({"public_row_id": [3]}),
        asocc_positions=np.array([1], dtype=np.int64),
        cc_positions=np.array([5], dtype=np.int64),
        static_cc_values=None,
        dynamic_cc_factors=np.array([1.0], dtype=np.float64),
        cc_type="dynamic_ar6",
        cc_source="gwp100_lcia",
    )

    def _plan(*, asocc_manifest):
        return ACCUncertaintyPlan(
            identity=pd.DataFrame({"public_row_id": [0, 1, 2, 3]}),
            summary_identity=pd.DataFrame({"public_row_id": [0, 1, 2, 3]}),
            summary_public_row_groups=(("0",), ("1",), ("2",), ("3",)),
            branch_plans=(static_branch, dynamic_branch, unmatched_dynamic_branch),
            asocc_input=ACCAsoccInput(
                identity=None,
                deterministic_values=None,
                manifest=asocc_manifest,
                deterministic_manifest_path=None,
                reuse_status="computed",
            ),
            dynamic_cc_input=ACCDynamicCCInput(
                identity=None,
                deterministic_values=None,
                manifest=cc_manifest,
                deterministic_manifest_path=None,
                reuse_status="computed",
            ),
            acc_run_layout="sparse_selected_rows",
            deterministic_cc_values=None,
            source_method_rows=pd.DataFrame(),
            active_sources=(
                "asocc::inter_method_uncertainty",
                "ar6_cc::dynamic_ar6_cc_uncertainty",
            ),
        )

    sparse_sparse_batches = list(
        sparse_render_mod.iter_acc_sparse_run_batches(
            plan=_plan(
                asocc_manifest=_asocc_manifest(
                    root="asocc_sparse",
                    runs_path=asocc_sparse_path,
                    layout="sparse_selected_rows",
                )
            ),
            output_format="csv_compact",
        )
    )
    sparse_sparse_rows = sparse_rows_mod.concat_acc_sparse_rows(
        pieces=[batch[1] for batch in sparse_sparse_batches]
    )
    assert sparse_sparse_rows.run_index.tolist() == [0, 0, 1]
    assert sparse_sparse_rows.public_row_id.tolist() == [0, 1, 2]
    assert sparse_sparse_rows.values.tolist() == [20.0, 10.0, 21.0]

    compact_sparse_batches = list(
        sparse_render_mod.iter_acc_sparse_run_batches(
            plan=_plan(
                asocc_manifest=_asocc_manifest(
                    root="asocc_compact",
                    runs_path=asocc_compact_path,
                    layout="compact_run_matrix",
                )
            ),
            output_format="csv_compact",
        )
    )
    compact_sparse_rows = sparse_rows_mod.concat_acc_sparse_rows(
        pieces=[batch[1] for batch in compact_sparse_batches]
    )
    assert compact_sparse_rows.run_index.tolist() == [0, 0, 1, 1]
    assert compact_sparse_rows.public_row_id.tolist() == [0, 1, 0, 2]
    assert compact_sparse_rows.values.tolist() == [20.0, 10.0, 30.0, 35.0]

    pending, collected = sparse_rows_mod.collect_sparse_rows_for_range(
        pending=SparseRunRows(
            run_index=np.array([3], dtype=np.int64),
            public_row_id=np.array([0], dtype=np.int64),
            values=np.array([1.0], dtype=np.float64),
            value_column="cc",
        ),
        chunks=iter(()),
        start=0,
        stop=2,
    )
    assert pending.run_index.tolist() == [3]
    assert collected.run_index.size == 0
    assert sparse_rows_mod.concat_acc_sparse_rows(pieces=[]).run_index.size == 0
    assert sparse_rows_mod.concat_cc_sparse_rows(pieces=[]).run_index.size == 0

    no_matching_asocc = sparse_render_mod.evaluate_acc_sparse_source_rows(
        asocc_rows=SparseRunRows(
            run_index=np.array([1], dtype=np.int64),
            public_row_id=np.array([1], dtype=np.int64),
            values=np.array([3.0], dtype=np.float64),
            value_column="asocc",
        ),
        cc_rows=SparseRunRows(
            run_index=np.array([1], dtype=np.int64),
            public_row_id=np.array([0], dtype=np.int64),
            values=np.array([5.0], dtype=np.float64),
            value_column="cc",
        ),
        expansions=sparse_rows_mod.cc_sparse_branch_expansions(
            branch_plans=(
                ACCBranchPlan(
                    identity=pd.DataFrame({"public_row_id": [0]}),
                    asocc_positions=np.array([0], dtype=np.int64),
                    cc_positions=np.array([0], dtype=np.int64),
                    static_cc_values=None,
                    dynamic_cc_factors=np.array([1.0], dtype=np.float64),
                    cc_type="dynamic_ar6",
                    cc_source="gwp100_lcia",
                ),
            )
        ),
    )
    assert no_matching_asocc.run_index.size == 0
