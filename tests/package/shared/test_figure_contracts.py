from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyaesa.shared.figures import checkpoints as checkpoints_mod
from pyaesa.shared.figures import colors as colors_mod
from pyaesa.shared.figures import contracts as contracts_mod
from pyaesa.shared.figures import generation_policy as policy_mod
from pyaesa.shared.figures import jobs as jobs_mod
from pyaesa.shared.figures import layout as layout_mod
from pyaesa.shared.figures import nonnegative_axis as axis_mod
from pyaesa.shared.figures import paths as paths_mod
from pyaesa.shared.figures import request_validation as request_validation_mod
from pyaesa.shared.figures import save as save_mod
from pyaesa.shared.figures import scenario_scopes as scenario_scopes_mod
from pyaesa.shared.figures import scope_support as scope_support_mod
from pyaesa.shared.figures import value_order as value_order_mod
from pyaesa.shared.runtime.scenario.columns import AR6_CC_SSP_SCENARIO_COLUMN
from pyaesa.shared.figures.deterministic_variant_compressor import (
    MAX_ROLE,
    MIN_ROLE,
    ROLE_COLUMN,
    compress_variants,
)
from pyaesa.shared.figures.deterministic_variant_display import (
    base_variant_groups,
    format_year_scalar,
    has_complete_variant_roles,
    variant_note,
    variant_role_row,
    variant_styles,
)
from pyaesa.shared.figures.violin_summary import render_violin_summaries
from pyaesa.shared.figures.uncertainty_run_values import (
    collect_selected_compact_run_values,
    collect_selected_sparse_run_values,
)
from pyaesa.shared.runtime.reporting import figure_progress as figure_progress_mod
from pyaesa.shared.runtime.reporting.status import TransientStatusPrinter
from pyaesa.shared.uncertainty_assessment.io.run_writers import (
    CompactRunMatrixWriter,
    SparseRunRows,
    SparseRunRowsWriter,
)


def test_generation_policy_contracts_cover_default_and_override_paths() -> None:
    assert checkpoints_mod.unique_figure_years(None) == []
    assert policy_mod.resolve_polar_years([2030]) == [2030]
    assert policy_mod.resolve_polar_years([2030, 2031, 2035, 2040]) == [2030, 2040]
    assert policy_mod.resolve_polar_years([2040, 2030, 2030], user_override=[2040, 2030]) == [
        2030,
        2040,
    ]
    with pytest.raises(ValueError):
        policy_mod.resolve_polar_years([2030, 2040], user_override=[2035])


def test_shared_color_and_value_order_contracts_cover_public_figure_scale_paths() -> None:
    assert colors_mod.distinct_colors(0) == []
    assert colors_mod.distinct_colors(1) == [colors_mod.DEFAULT_SINGLE_SERIES_COLOR]
    assert colors_mod.DEFAULT_SINGLE_SERIES_COLOR == "#0072B2"
    assert colors_mod.distinct_colors(3) == list(colors_mod.HIGH_CONTRAST_COLORS[:3])
    assert colors_mod.single_or_distinct_colors(["only", "only"]) == {
        "only": colors_mod.DEFAULT_SINGLE_SERIES_COLOR
    }
    assert colors_mod.single_or_distinct_colors(["A", "B"]) == {
        "A": colors_mod.HIGH_CONTRAST_COLORS[0],
        "B": colors_mod.HIGH_CONTRAST_COLORS[1],
    }
    extended = colors_mod.distinct_colors(len(colors_mod.HIGH_CONTRAST_COLORS) + 1)
    assert len(extended) == len(colors_mod.HIGH_CONTRAST_COLORS) + 1
    assert len(set(extended)) == len(extended)

    assert value_order_mod.finite_average([1.0, float("nan"), 3.0]) == 2.0
    assert value_order_mod.finite_average([float("nan")]) is None
    assert value_order_mod.order_labels_by_average_score(
        {"low": [1.0], "high": [2.0], "missing": [float("nan")]}
    ) == ["high", "low"]
    assert value_order_mod.order_labels_by_average_within_group_rank(
        [
            ("A", "x", 3.0),
            ("A", "y", 1.0),
            ("B", "x", 1.0),
            ("B", "y", 4.0),
            ("B", "z", float("nan")),
        ]
    ) == ["x", "y"]
    assert (
        value_order_mod.row_average_score(
            pd.Series({"values": np.array([1.0, 3.0])}),
            value_array_column="values",
        )
        == 2.0
    )
    assert value_order_mod.row_average_score(pd.Series({"mean": pd.NA, "value": 4.0})) == 4.0
    assert value_order_mod.row_average_score(pd.Series({"label": "x"})) is None
    assert value_order_mod.frame_average_score(pd.DataFrame({"value": [1.0, 3.0]})) == 2.0
    assert value_order_mod.frame_average_score(pd.DataFrame({"label": ["x"]})) == float("-inf")


def test_figure_validation_and_request_validation_supported_inputs() -> None:
    assert contracts_mod.normalize_figure_output_format(" PDF ") == "pdf"
    assert contracts_mod.validate_figure_dpi(300) == 300
    assert contracts_mod.resolved_selector_columns(frame=None) == tuple()
    assert contracts_mod.resolved_selector_columns(frame=pd.DataFrame()) == tuple()
    assert contracts_mod.deterministic_prospective_series(
        pd.DataFrame({AR6_CC_SSP_SCENARIO_COLUMN: [None, " "]})
    ).tolist() == [None, None]

    normalized_format = request_validation_mod.normalize_figure_format(
        {"format": " png ", "dpi": 1050}
    )
    assert normalized_format == {"format": "png", "dpi": 1050}

    normalized_options = request_validation_mod.normalize_figure_options(
        {
            "single_year_style": " BOTH ",
            "polar_years": [2040, 2030, 2040],
            "polar_style": " WHISKER ",
            "per_method": False,
            "multi_method": True,
            "inter_method": False,
            "polar": {"active": False, "polar_years": [2035], "polar_style": "both"},
        },
        allow_single_year_style=True,
        allow_polar_years=True,
        allow_polar_style=True,
        allow_per_method=True,
        allow_multi_method=True,
        allow_inter_method=True,
        allow_polar=True,
    )
    assert normalized_options == {
        "per_method": False,
        "multi_method": True,
        "inter_method": False,
        "single_year_style": "both",
        "polar_years": [2030, 2040],
        "polar_style": "whisker",
        "polar": {"active": False, "polar_years": [2035], "polar_style": "both"},
    }
    assert request_validation_mod.normalize_subfigure_options({"single_year_style": "violin"}) == {
        "single_year_style": "violin"
    }

    with pytest.raises(ValueError):
        contracts_mod.normalize_figure_output_format("jpg")
    with pytest.raises(ValueError):
        contracts_mod.validate_figure_dpi(0)
    with pytest.raises(ValueError):
        request_validation_mod.normalize_figure_format({"bad": 1})
    with pytest.raises(ValueError):
        request_validation_mod.normalize_single_year_style("scatter", argument_name="style")
    assert request_validation_mod.normalize_polar_years([], argument_name="years") == []
    with pytest.raises(ValueError):
        request_validation_mod.normalize_polar_years((2030,), argument_name="years")
    with pytest.raises(ValueError):
        request_validation_mod.validate_consecutive_multi_year_figure_request(
            [2030, 2032],
            family_label="demo",
        )
    with pytest.raises(ValueError):
        request_validation_mod.normalize_figure_options(
            {"polar_years": [2030]},
            allow_single_year_style=True,
            allow_polar_years=False,
        )
    with pytest.raises(ValueError):
        request_validation_mod.normalize_figure_options(
            {"per_method": "yes"},
            allow_single_year_style=False,
            allow_polar_years=False,
            allow_per_method=True,
        )
    with pytest.raises(ValueError):
        request_validation_mod.normalize_polar_options("bad", argument_name="figure_options.polar")
    with pytest.raises(ValueError):
        request_validation_mod.normalize_polar_options(
            {"bad": True},
            argument_name="figure_options.polar",
        )


def test_figure_path_contracts_cover_routing_and_validation(tmp_path: Path) -> None:
    figures_root = paths_mod.figures_root_for_run(run_root=tmp_path / "run")
    assert figures_root == tmp_path / "run" / "figures"

    out_paths = paths_mod.output_paths(
        base_path=tmp_path / "outputs" / "demo_plot",
        output_format="png",
    )
    assert out_paths == [tmp_path / "outputs" / "demo_plot.png"]
    assert out_paths[0].parent.is_dir()

    assert paths_mod.year_token([]) == "years_none"
    assert (
        paths_mod.year_token([2030, 2031, 2032, 2035, 2037, 2038])
        == "years_2030-2032_2035_2037-2038"
    )
    assert paths_mod.top_level_figure_dir(figures_root=figures_root, folder="per_method") == (
        figures_root / "per_method"
    )
    assert (
        paths_mod.deterministic_figure_dir(
            figures_root=figures_root,
            timescale="single_year",
        )
        == figures_root / "single_year"
    )
    assert (
        paths_mod.deterministic_figure_dir(
            figures_root=figures_root,
            timescale="multi_year",
            role="multi_method",
        )
        == figures_root / "multi_method" / "multi_year"
    )
    assert (
        paths_mod.uncertainty_figure_dir(
            figures_root=figures_root,
            timescale="single_year",
            family="whisker",
            role="per_method",
        )
        == figures_root / "per_method" / "single_year" / "whisker"
    )
    assert (
        paths_mod.uncertainty_figure_dir(
            figures_root=figures_root,
            timescale="multi_year",
            family="trajectory_bands",
            role=None,
        )
        == figures_root / "multi_year" / "trajectory_bands"
    )
    assert (
        paths_mod.family_figure_dir(
            figures_root=figures_root,
            family="polar",
            role="multi_method",
            granularity="multi_year",
        )
        == figures_root / "multi_method" / "multi_year" / "polar"
    )
    assert (
        paths_mod.strip_lcia_method_suffix(
            stem="asr__gwp100",
            lcia_methods=["gwp100", "land"],
        )
        == "asr"
    )
    assert (
        paths_mod.strip_lcia_method_suffix(
            stem="asr_land",
            lcia_methods=["gwp100", "land"],
        )
        == "asr"
    )
    assert (
        paths_mod.strip_lcia_method_suffix(
            stem="  ",
            lcia_methods=["gwp100"],
        )
        == ""
    )
    assert (
        paths_mod.strip_lcia_method_suffix(
            stem="asr_demo",
            lcia_methods=["gwp100"],
        )
        == "asr_demo"
    )
    assert paths_mod.scope_filename_stem(base_stem="plot", lcia_method=None) == "plot"
    assert (
        paths_mod.scope_filename_stem(base_stem="plot", lcia_method="GWP 100 / total")
        == "plot__GWP_100_total"
    )
    assert (
        paths_mod.scope_filename_stem(base_stem="plot__gwp100_lcia", lcia_method="gwp100_lcia")
        == "plot__gwp100_lcia"
    )
    assert (
        paths_mod.scope_filename_stem(base_stem="plot_gwp100_lcia", lcia_method="gwp100_lcia")
        == "plot_gwp100_lcia"
    )
    assert paths_mod.scope_filename_stem(base_stem=" plot ", lcia_method="   ") == "plot__item"


def test_checkpoint_layout_and_nonnegative_axis_contracts_cover_real_branches() -> None:
    assert checkpoints_mod.default_checkpoint_years([]) == []
    assert checkpoints_mod.default_checkpoint_years([2030, 2032, 2035, 2040, 2041]) == [
        2030,
        2035,
        2040,
        2041,
    ]

    single_layout = layout_mod.resolve_layout(impacts_count=3)
    assert single_layout == {
        "layout": "single",
        "ncols": 1,
        "nrows": 3,
        "fig_width": 16.0,
        "fig_height": 15.5,
    }
    double_layout = layout_mod.resolve_layout(impacts_count=11)
    assert double_layout == {
        "layout": "double",
        "ncols": 2,
        "nrows": 6,
        "fig_width": 22.0,
        "fig_height": 18.6,
    }
    assert layout_mod.resolve_polar_availability(1) is False
    assert layout_mod.resolve_polar_availability(2) is True
    assert layout_mod.MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN == (
        layout_mod.SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN * 1.3
    )
    assert layout_mod.multi_impact_panel_figure_size(nrows=1) == (18.0, 7.2)
    assert layout_mod.multi_impact_panel_figure_size(nrows=1, compact=True) == (18.0, 3.6)
    assert layout_mod.multi_impact_panel_figure_size(nrows=5, compact=True) == (18.0, 11.25)
    assert not layout_mod.show_panel_x_labels(
        panel_index=0,
        bottom_indices={2},
    )
    assert not layout_mod.show_panel_x_labels(
        panel_index=0,
        bottom_indices={2},
    )
    assert layout_mod.show_panel_x_labels(
        panel_index=2,
        bottom_indices={2},
    )
    assert np.array_equal(
        layout_mod.build_year_columns(years=[2030, 2031, 2033, 2035], step=2),
        np.array([2030]),
    )
    assert np.array_equal(layout_mod.build_year_columns(years=[], step=2), np.array([], dtype=int))
    empty_year_fig, empty_year_axis = plt.subplots()
    layout_mod.format_integer_year_axis(empty_year_axis, years=[])
    assert empty_year_axis.has_data() is False
    empty_year_axis.set_xlabel("Year")
    layout_mod.hide_x_axis_tick_labels(empty_year_axis)
    assert empty_year_axis.get_xlabel() == ""
    plt.close(empty_year_fig)

    assert axis_mod.require_nonnegative_figure_ylim(values=np.array([]), context="demo") == (
        0.0,
        1.0,
    )
    assert axis_mod.require_nonnegative_figure_ylim(
        values=np.array([0.0, 0.0]),
        context="demo",
    ) == (0.0, 1.0)
    assert axis_mod.require_nonnegative_figure_ylim(
        values=np.array([1.0, np.nan, 4.0]),
        context="demo",
    ) == (
        0.0,
        4.48,
    )
    assert axis_mod.signed_figure_ylim(values=np.array([-2.0, 3.0])) == (-2.6, 3.6)
    assert axis_mod.signed_figure_ylim(values=np.array([])) == (-1.0, 1.0)
    assert axis_mod.signed_figure_ylim(values=np.array([2.0, 2.0])) == (1.76, 2.24)
    assert axis_mod.resolve_axis_ylim(
        values=np.array([1.0, 3.0]),
        context="demo",
        policy="nonnegative",
    ) == (0.0, 3.3600000000000003)
    assert axis_mod.resolve_axis_ylim(
        values=np.array([-2.0, 3.0]),
        context="demo",
        policy="signed",
    ) == (-2.6, 3.6)
    assert axis_mod.resolve_axis_ylim(
        values=np.array([]),
        context="demo",
        policy="signed",
    ) == (0.0, 1.0)
    with pytest.raises(ValueError):
        axis_mod.require_nonnegative_figure_ylim(values=np.array([-1.0, 2.0]), context="demo")


def test_save_figure_persists_and_closes_matplotlib_figure(tmp_path: Path) -> None:
    unmanaged_fig, axes = save_mod.create_subplots(nrows=1, ncols=2, figsize=(4.0, 2.0))
    assert len(np.ravel(axes)) == 2
    plt.close(unmanaged_fig)

    fig, ax = plt.subplots()
    ax.plot([2030, 2031], [1.0, 2.0])
    base_path = tmp_path / "figures" / "trajectory_plot"

    output_paths = save_mod.save_figure(
        fig,
        base_path,
        dpi=10,
        output_format="png",
    )

    assert output_paths == [base_path.with_suffix(".png")]
    assert output_paths[0].is_file()
    assert not plt.fignum_exists(fig.number)


def test_shared_scope_support_and_progress_helpers_cover_optional_paths(
    project_repo: Path,
    tmp_path: Path,
) -> None:
    del project_repo
    assert scope_support_mod.impact_column(pd.DataFrame({"variable": ["GWP_100"]})) == "variable"
    assert scope_support_mod.impact_column(pd.DataFrame({"value": [1.0]})) is None
    assert scope_support_mod.repeat_generic_impacts(
        pd.DataFrame({"impact": [None], "value": [1.0]}),
        impact_name=None,
    ).to_dict("records") == [{"impact": None, "value": 1.0}]
    assert scope_support_mod.repeat_generic_impacts(
        pd.DataFrame({"impact": [None], "value": [1.0]}),
        impact_name="impact",
    ).to_dict("records") == [{"impact": None, "value": 1.0}]
    assert scope_support_mod.repeat_generic_impacts(
        pd.DataFrame({"impact": ["A"], "value": [1.0]}),
        impact_name="impact",
    ).to_dict("records") == [{"impact": "A", "value": 1.0}]
    expanded = scope_support_mod.repeat_generic_impacts(
        pd.DataFrame({"impact": ["A", "B", None], "value": [1.0, 2.0, 3.0]}),
        impact_name="impact",
    )
    assert expanded.to_dict("records") == [
        {"impact": "A", "value": 1.0},
        {"impact": "B", "value": 2.0},
        {"impact": "A", "value": 3.0},
        {"impact": "B", "value": 3.0},
    ]

    paths = figure_progress_mod.render_with_progress(
        source="figures",
        items=["a"],
        describe=lambda item: item,
        render=lambda item: [tmp_path / f"{item}.png"],
        total=1,
    )
    assert paths == [tmp_path / "a.png"]

    class _Status:
        def __init__(self) -> None:
            self.messages: list[str] = []
            self.cleared = False
            self.finished = False

        def show(self, message: str) -> None:
            self.messages.append(message)

        def log_message(self, message: str, *, persistent: bool = True) -> None:
            self.messages.append(message)

        def clear_transient(self) -> None:
            self.cleared = True

        def finish(self) -> None:
            self.finished = True

    status = _Status()
    long_description = "x" * 90
    custom_status_paths = figure_progress_mod.render_with_progress(
        source="figures",
        items=[long_description],
        describe=lambda item: item,
        render=lambda item: [tmp_path / "custom.png"],
        total=1,
        status=cast(TransientStatusPrinter, status),
    )
    assert custom_status_paths == [tmp_path / "custom.png"]
    assert status.cleared is True
    assert status.finished is False
    assert status.messages[-1].startswith("[figures] Generated figure")

    assert (
        figure_progress_mod.render_with_progress(
            source="figures",
            items=[],
            describe=lambda item: str(item),
            render=lambda item: [tmp_path / f"{item}.png"],
            total=0,
        )
        == []
    )
    streamed_status = _Status()
    streamed_paths = figure_progress_mod.render_with_progress(
        source="figures",
        items=(item for item in ["streamed"]),
        describe=lambda item: str(item),
        render=lambda item: [tmp_path / f"{item}.png"],
        total=1,
        status=cast(TransientStatusPrinter, streamed_status),
    )
    assert streamed_paths == [tmp_path / "streamed.png"]
    assert len(streamed_status.messages) == 2
    assert streamed_status.messages[0].startswith("[figures] Generating figure")
    assert "Generating figures" not in streamed_status.messages[0]
    assert streamed_status.messages[1].startswith("[figures] Generated figure")
    split_status = _Status()
    split_paths = figure_progress_mod.render_with_progress(
        source="figures",
        items=["split"],
        describe=lambda item: str(item),
        render=lambda item: [tmp_path / f"{item}_a.png", tmp_path / f"{item}_b.png"],
        total=2,
        item_count=lambda _item: 2,
        status=cast(TransientStatusPrinter, split_status),
    )
    assert split_paths == [tmp_path / "split_a.png", tmp_path / "split_b.png"]
    assert "2/2" in split_status.messages[0]
    assert "2/2" in split_status.messages[-1]
    assert "figure jobs" not in split_status.messages[-1]
    planned_paths = jobs_mod.render_figure_jobs(
        source="figures",
        jobs=lambda: (
            jobs_mod.PlannedFigureJob(
                kind="demo",
                label=item,
                render=lambda item=item: [tmp_path / f"{item}.png"],
            )
            for item in ["planned"]
        ),
    )
    assert planned_paths == [tmp_path / "planned.png"]
    planned_split_status = _Status()
    planned_split_paths = jobs_mod.render_figure_jobs(
        source="figures",
        jobs=lambda: (
            jobs_mod.PlannedFigureJob(
                kind="demo",
                label="planned_split",
                planned_outputs=2,
                render=lambda: [tmp_path / "planned_a.png", tmp_path / "planned_b.png"],
            )
            for _item in [None]
        ),
        status=cast(TransientStatusPrinter, planned_split_status),
    )
    assert planned_split_paths == [tmp_path / "planned_a.png", tmp_path / "planned_b.png"]
    assert "2/2" in planned_split_status.messages[0]
    assert "2/2" in planned_split_status.messages[-1]


def test_scenario_scope_helpers_cover_requested_and_preplanned_scope_contracts() -> None:
    missing_column = list(
        scenario_scopes_mod.repeat_invariant_rows_into_scenarios(
            pd.DataFrame({"value": [1.0]}),
            scenario_column="scenario",
            scope_column="scope",
        )
    )
    assert missing_column[0]["value"].tolist() == [1.0]
    assert missing_column[0]["scope"].isna().tolist() == [True]
    invariant_only = list(
        scenario_scopes_mod.repeat_invariant_rows_into_scenarios(
            pd.DataFrame({"scenario": [pd.NA], "value": [1.0]}),
            scenario_column="scenario",
            scope_column="scope",
        )
    )
    assert invariant_only[0]["scope"].isna().tolist() == [True]
    frame = pd.DataFrame({"scenario": [pd.NA, "SSP1"], "value": [1.0, 2.0]})
    scopes = list(
        scenario_scopes_mod.repeat_invariant_rows_into_scenarios(
            frame,
            scenario_column="scenario",
            scope_column="scope",
            requested_scenarios=("SSP1", "SSP2"),
        )
    )
    assert len(scopes) == 1
    assert scopes[0]["scope"].tolist() == ["SSP1", "SSP1"]

    preplanned_frame = pd.DataFrame({"scenario": [pd.NA, "SSP1"], "value": [0.0, 1.0]})
    assert list(
        scenario_scopes_mod.preplanned_scenario_scope_slices(
            preplanned_frame,
            scenario_column="scenario",
            scope_column="scope",
        )
    )[0].equals(preplanned_frame)
    preplanned_invariant = list(
        scenario_scopes_mod.preplanned_scenario_scope_slices(
            preplanned_frame.iloc[[0]].assign(scope=[pd.NA]),
            scenario_column="scenario",
            scope_column="scope",
        )
    )
    assert preplanned_invariant[0]["scope"].isna().tolist() == [True]
    assert list(
        scenario_scopes_mod.preplanned_scenario_scope_slices(
            preplanned_frame.iloc[[1]],
            scenario_column="scenario",
            scope_column="scope",
        )
    )[0].to_dict("records") == [{"scenario": "SSP1", "value": 1.0}]
    preplanned = list(
        scenario_scopes_mod.preplanned_scenario_scope_slices(
            preplanned_frame.assign(scope=[pd.NA, "SSP1"]),
            scenario_column="scenario",
            scope_column="scope",
        )
    )
    assert preplanned[0].to_dict("records") == [{"scenario": "SSP1", "scope": "SSP1", "value": 1.0}]
    assert preplanned[1]["scenario"].map(pd.isna).tolist() == [True]
    assert (
        scenario_scopes_mod.visible_scenario_values(
            pd.DataFrame({"value": [1.0]}), scenario_column="scenario"
        )
        == []
    )
    assert scenario_scopes_mod.requested_visible_scenarios(
        visible_scenarios=["SSP1"], requested_scenarios=()
    ) == ["SSP1"]


def test_violin_summary_renderer_handles_empty_payload() -> None:
    fig, axis = plt.subplots()
    try:
        assert (
            render_violin_summaries(
                axis,
                values=[],
                positions=np.array([], dtype=float),
                colors=[],
            )
            == []
        )
    finally:
        plt.close(fig)


def test_uncertainty_run_value_readers_select_only_requested_ids(tmp_path: Path) -> None:
    compact_csv = tmp_path / "compact.csv"
    run_indices = np.array([0, 1], dtype=np.int64)
    compact_values = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float64)
    with CompactRunMatrixWriter(path=compact_csv, output_format="csv_compact") as writer:
        writer.write_batch(run_indices=run_indices, values=compact_values, batch_index=0)
    compact_parquet = tmp_path / "compact.parquet"
    with CompactRunMatrixWriter(path=compact_parquet, output_format="parquet") as writer:
        writer.write_batch(run_indices=run_indices, values=compact_values, batch_index=0)

    csv_values = collect_selected_compact_run_values(
        path=compact_csv,
        output_format="csv_compact",
        public_row_ids=[2, 0, 2],
    )
    parquet_values = collect_selected_compact_run_values(
        path=compact_parquet,
        output_format="parquet",
        public_row_ids=[2, 0],
    )
    compact_limited = collect_selected_compact_run_values(
        path=compact_csv,
        output_format="csv_compact",
        public_row_ids=[2, 0],
        stop_run_index=1,
    )
    empty_compact = collect_selected_compact_run_values(
        path=compact_csv,
        output_format="csv_compact",
        public_row_ids=[],
    )

    assert empty_compact == {}
    assert sorted(csv_values) == [0, 2]
    assert sorted(parquet_values) == [0, 2]
    np.testing.assert_allclose(csv_values[0], [1.0, 2.0])
    np.testing.assert_allclose(csv_values[2], [5.0, 6.0])
    np.testing.assert_allclose(parquet_values[0], [1.0, 2.0])
    np.testing.assert_allclose(parquet_values[2], [5.0, 6.0])
    np.testing.assert_allclose(compact_limited[0], [1.0])
    np.testing.assert_allclose(compact_limited[2], [5.0])

    sparse_csv = tmp_path / "sparse.csv"
    sparse_rows = SparseRunRows(
        run_index=np.array([0, 0, 1, 1], dtype=np.int64),
        public_row_id=np.array([0, 2, 0, 3], dtype=np.int64),
        values=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64),
        value_column="acc",
    )
    with SparseRunRowsWriter(path=sparse_csv, output_format="csv_compact") as writer:
        writer.write_batch(rows=sparse_rows, batch_index=0)
    sparse_parquet = tmp_path / "sparse.parquet"
    with SparseRunRowsWriter(path=sparse_parquet, output_format="parquet") as writer:
        writer.write_batch(rows=sparse_rows, batch_index=0)

    sparse_csv_values = collect_selected_sparse_run_values(
        path=sparse_csv,
        output_format="csv_compact",
        public_row_ids=[2, 99, 0],
    )
    sparse_parquet_values = collect_selected_sparse_run_values(
        path=sparse_parquet,
        output_format="parquet",
        public_row_ids=[2, 99, 0],
    )
    sparse_limited = collect_selected_sparse_run_values(
        path=sparse_csv,
        output_format="csv_compact",
        public_row_ids=[2, 0],
        stop_run_index=1,
    )
    empty_sparse = collect_selected_sparse_run_values(
        path=sparse_csv,
        output_format="csv_compact",
        public_row_ids=[],
    )
    unmatched_sparse = collect_selected_sparse_run_values(
        path=sparse_csv,
        output_format="csv_compact",
        public_row_ids=[99],
    )

    assert empty_sparse == {}
    assert unmatched_sparse[99].size == 0
    assert sorted(sparse_csv_values) == [0, 2, 99]
    assert sorted(sparse_parquet_values) == [0, 2, 99]
    np.testing.assert_allclose(sparse_csv_values[0], [10.0, 30.0])
    np.testing.assert_allclose(sparse_csv_values[2], [20.0])
    assert sparse_csv_values[99].size == 0
    np.testing.assert_allclose(sparse_parquet_values[0], [10.0, 30.0])
    np.testing.assert_allclose(sparse_parquet_values[2], [20.0])
    assert sparse_parquet_values[99].size == 0
    np.testing.assert_allclose(sparse_limited[0], [10.0])
    np.testing.assert_allclose(sparse_limited[2], [20.0])


def test_deterministic_variant_compression_and_display_contracts() -> None:
    no_variant = pd.DataFrame({"year": [2030], "value": [1.0], "__method": ["A"]})
    assert compress_variants(no_variant).to_dict("records") == no_variant.to_dict("records")
    assert variant_styles(no_variant) == ["solid"]
    assert variant_note(no_variant) is None
    assert format_year_scalar(2030.0) == "2030"
    assert format_year_scalar(pd.NA) == ""
    assert has_complete_variant_roles(no_variant) is False

    all_missing_variant = pd.DataFrame(
        {
            "year": [2030],
            "value": [1.0],
            "__method": ["A"],
            "reference_year": [pd.NA],
        }
    )
    assert compress_variants(all_missing_variant).to_dict("records") == (
        all_missing_variant.to_dict("records")
    )

    ambiguous_single_variant_note = variant_note(
        pd.DataFrame(
            {
                "year": [2030, 2030],
                "value": [1.0, 2.0],
                "__method": ["A", "A"],
                "reference_year": [2020, 2021],
            }
        )
    )
    assert ambiguous_single_variant_note is None

    invalid_role_frame = pd.DataFrame(
        {
            "year": [2030],
            "value": [1.0],
            "__method": ["A"],
            "reference_year": [2020],
            ROLE_COLUMN: [MIN_ROLE],
        }
    )
    assert variant_note(invalid_role_frame) is None

    frame = pd.DataFrame(
        [
            {
                "__method": "A",
                "impact": "I1",
                "year": 2029,
                "value": 8.0,
                "reference_year": 2020,
                "l2_reuse_year": pd.NA,
            },
            {
                "__method": "A",
                "impact": "I1",
                "year": 2030,
                "value": 10.0,
                "reference_year": 2020,
                "l2_reuse_year": 2028,
            },
            {
                "__method": "A",
                "impact": "I1",
                "year": 2030,
                "value": 20.0,
                "reference_year": 2020,
                "l2_reuse_year": 2029,
            },
            {
                "__method": "B",
                "impact": "I1",
                "year": 2029,
                "value": 11.0,
                "reference_year": 2020,
                "l2_reuse_year": pd.NA,
            },
            {
                "__method": "B",
                "impact": "I1",
                "year": 2030,
                "value": 12.0,
                "reference_year": 2020,
                "l2_reuse_year": pd.NA,
            },
            {
                "__method": "C",
                "impact": "I1",
                "year": 2030,
                "value": 14.0,
                "reference_year": pd.NA,
                "l2_reuse_year": pd.NA,
            },
            {
                "__method": "D",
                "impact": "I1",
                "year": 2030,
                "value": 16.0,
                "reference_year": pd.NA,
                "l2_reuse_year": 2028,
            },
            {
                "__method": "D",
                "impact": "I1",
                "year": 2030,
                "value": 18.0,
                "reference_year": pd.NA,
                "l2_reuse_year": 2029,
            },
        ]
    )
    compressed = compress_variants(frame)
    assert {MIN_ROLE, MAX_ROLE}.issubset(set(compressed[ROLE_COLUMN].dropna().astype(str)))
    assert "dotted" in variant_styles(compressed)
    grouped = base_variant_groups(compressed)
    complete_group = next(group for group in grouped if has_complete_variant_roles(group))
    assert variant_role_row(complete_group, role=MIN_ROLE)["l2_reuse_year"] == 2028
    note = variant_note(compressed, single_year=False)
    assert note is not None
    custom_geometry_note = variant_note(compressed, single_year=False, geometry_override="Custom.")
    assert custom_geometry_note is not None

    single_variant = compress_variants(
        pd.DataFrame(
            {
                "year": [2030],
                "value": [1.0],
                "__method": ["A"],
                "reference_year": [2020.0],
            }
        )
    )
    assert ROLE_COLUMN not in single_variant.columns
    single_variant_note = variant_note(single_variant, single_year=True)
    assert single_variant_note is not None

    fixed_multi_method_variant = compress_variants(
        pd.DataFrame(
            {
                "year": [2030, 2030, 2030, 2030],
                "value": [1.0, 2.0, 3.0, 4.0],
                "__method": ["A", "A", "B", "B"],
                "impact": ["I1", "I2", "I1", "I2"],
                "reference_year": [2020.0, 2020.0, 2020.0, 2020.0],
                "l2_reuse_year": [2028.0, 2028.0, 2028.0, 2028.0],
            }
        )
    )
    fixed_multi_method_note = variant_note(fixed_multi_method_variant, single_year=True)
    assert fixed_multi_method_note is not None

    partial = compress_variants(
        pd.DataFrame(
            [
                {
                    "__method": "A",
                    "impact": "I1",
                    "year": 2029,
                    "value": 1.0,
                    "reference_year": 2020,
                    "l2_reuse_year": pd.NA,
                },
                {
                    "__method": "A",
                    "impact": "I1",
                    "year": 2030,
                    "value": 2.0,
                    "reference_year": 2020,
                    "l2_reuse_year": 2028,
                },
                {
                    "__method": "A",
                    "impact": "I1",
                    "year": 2030,
                    "value": 3.0,
                    "reference_year": 2020,
                    "l2_reuse_year": pd.NA,
                },
            ]
        )
    )
    assert set(pd.Series(partial["l2_reuse_year"]).dropna().astype(int).tolist()) == {2028}

    reuse_only = compress_variants(
        pd.DataFrame(
            [
                {"year": 2030, "value": 2.0, "__method": "A", "l2_reuse_year": 2028},
                {"year": 2030, "value": 5.0, "__method": "A", "l2_reuse_year": 2029},
            ]
        )
    )
    assert set(reuse_only[ROLE_COLUMN].astype(str)) == {MIN_ROLE, MAX_ROLE}

    with pytest.raises(ValueError):
        compress_variants(
            pd.DataFrame(
                [
                    {
                        "year": 2030,
                        "value": 1.0,
                        "__method": "A",
                        "reference_year": 2020,
                    },
                    {
                        "year": 2031,
                        "value": 2.0,
                        "__method": "A",
                        "reference_year": 2021,
                    },
                ]
            )
        )
    with pytest.raises(ValueError):
        compress_variants(
            pd.DataFrame(
                [
                    {"year": 2030, "value": 1.0, "__method": "A", "reference_year": 2020},
                    {"year": 2031, "value": 1.2, "__method": "A", "reference_year": 2020},
                    {"year": 2030, "value": 2.0, "__method": "B", "reference_year": 2021},
                    {"year": 2031, "value": 2.2, "__method": "B", "reference_year": 2021},
                ]
            )
        )

    equal_variants = compress_variants(
        pd.DataFrame(
            {
                "year": [2030, 2030],
                "value": [1.0, 1.0],
                "__method": ["A", "A"],
                "reference_year": [2020, 2021],
            }
        )
    )
    assert set(equal_variants[ROLE_COLUMN].astype(str)) == {MIN_ROLE, MAX_ROLE}

    partial_role_frame = pd.DataFrame(
        {
            "year": [2030, 2030],
            "value": [1.0, 2.0],
            "__method": ["A", "A"],
            "impact": ["I1", "I2"],
            "reference_year": [2020, pd.NA],
            "l2_reuse_year": [pd.NA, 2029],
            ROLE_COLUMN: [MIN_ROLE, MAX_ROLE],
        }
    )
    partial_role_note = variant_note(partial_role_frame, single_year=True)
    assert partial_role_note is not None

    methodless_role_frame = pd.DataFrame(
        {
            "year": [2030, 2030],
            "value": [1.0, 2.0],
            "reference_year": [2020, 2021],
            ROLE_COLUMN: [MIN_ROLE, MAX_ROLE],
        }
    )
    methodless_note = variant_note(methodless_role_frame, single_year=True)
    assert methodless_note is not None

    blank_method_role_frame = pd.DataFrame(
        {
            "year": [2030, 2030],
            "value": [1.0, 2.0],
            "__method": ["", ""],
            "reference_year": [2020, 2021],
            ROLE_COLUMN: [MIN_ROLE, MAX_ROLE],
        }
    )
    blank_method_note = variant_note(blank_method_role_frame, single_year=True)
    assert blank_method_note is not None

    no_axis_note_frame = pd.DataFrame(
        [
            {
                "year": 2030,
                "value": 1.0,
                "__method": "A",
                "reference_year": pd.NA,
                ROLE_COLUMN: MIN_ROLE,
            },
            {
                "year": 2030,
                "value": 2.0,
                "__method": "B",
                "reference_year": pd.NA,
                ROLE_COLUMN: MAX_ROLE,
            },
            {
                "year": 2030,
                "value": 3.0,
                "__method": "",
                "reference_year": 2020,
                ROLE_COLUMN: MIN_ROLE,
            },
            {
                "year": 2030,
                "value": 4.0,
                "__method": "",
                "reference_year": 2021,
                ROLE_COLUMN: MAX_ROLE,
            },
        ]
    )
    no_axis_note = variant_note(no_axis_note_frame, single_year=True)
    assert no_axis_note is not None

    wrapped_methods = []
    for index in range(5):
        wrapped_methods.extend(
            [
                {
                    "year": 2030,
                    "value": 1.0 + index,
                    "__method": f"allocation_method_with_long_visible_name_{index}",
                    "reference_year": 2020,
                    ROLE_COLUMN: MIN_ROLE,
                },
                {
                    "year": 2030,
                    "value": 2.0 + index,
                    "__method": f"allocation_method_with_long_visible_name_{index}",
                    "reference_year": 2021,
                    ROLE_COLUMN: MAX_ROLE,
                },
            ]
        )
    wrapped_note = variant_note(pd.DataFrame(wrapped_methods), single_year=True)
    assert wrapped_note is not None
