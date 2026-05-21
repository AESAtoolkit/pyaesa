"""Rendering policy for ASR uncertainty interval and mean line products."""

from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pyaesa.ar6_cc.deterministic.figures.render import render_deterministic_ar6_cc_row
from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NEGATIVE,
    CC_FLOW_POSITIVE,
)
from pyaesa.ar6_cc.shared.runtime.figure_style import ar6_category_color
from pyaesa.ar6_cc.uncertainty.figures.period_panels import (
    render_study_transition as render_ar6_study_transition,
)
from pyaesa.ar6_cc.uncertainty.figures.product_renderers import (
    FLOW_COLORS as AR6_CC_FLOW_COLORS,
)
from pyaesa.ar6_cc.uncertainty.figures.product_renderers import (
    render_uncertainty_ar6_cc_row,
)
from pyaesa.asr.figures.axis import ASR_LOG_SCALE, ASRScaleMode, apply_frequency_axis
from pyaesa.asr.figures.common import (
    MEAN_LINE_NOTE,
    apply_scaled_asr_axis_policy,
    asr_axis_limits,
    component_axis_limits,
    format_acc_lca_component_axis,
    format_year_axis,
    impact_panel_title,
    ordered_impacts,
    save_figure,
    visible_values,
)
from pyaesa.asr.figures.component_legend import (
    LCA_COMPONENT_COLOR,
    acc_lca_cumulative_title,
    acc_lca_pathway_title,
    ar6_cc_flow_key_entries,
    frequency_color_map,
    lca_component_linewidth,
    lca_component_path_effects,
    render_acc_lca_row_key,
    render_ar6_cc_row_key,
)
from pyaesa.asr.figures.dynamic_global_ar6 import (
    UncertaintyGlobalAR6Rows,
    UncertaintyGlobalAR6Source,
    global_ar6_panel_title_pad,
    uncertainty_global_ar6_rows_from_source,
)
from pyaesa.asr.figures.frequency import (
    CUMULATIVE_FNT_FRACTION_COLUMN,
    FNT_FRACTION_COLUMN,
    fnt_legend_entry,
)
from pyaesa.asr.figures.risk_guides import (
    ASR_NON_POLAR_RISK_BACKGROUND_ALPHA_SCALE,
    ASR_RISK_BACKGROUND_VIOLIN_ALPHA,
    asr_risk_scale_footer_extra_height,
    render_asr_risk_scale_footer,
)
from pyaesa.asr.figures.transitions import (
    ASR_TRANSITION_SERIES_EXCLUDED_COLUMNS,
    asr_transition_markers,
    merged_asr_transition_markers,
)
from pyaesa.asr.uncertainty.figures.component_data import (
    ComponentDiagnosticRows,
    component_scope_rows,
    component_series_columns,
)
from pyaesa.shared.figures.colors import (
    DEFAULT_SINGLE_SERIES_COLOR,
    MULTI_METHOD_LINE_ALPHA,
    distinct_colors,
    single_or_distinct_colors,
)
from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.deterministic_legends_methods import legend_group_from_row
from pyaesa.shared.figures.dynamic_ar6 import (
    MODEL_SCENARIO_PAIR_COUNT_COLUMN,
    MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
    model_scenario_sampling_method,
)
from pyaesa.shared.figures.figure_footer import (
    _LEGEND_ROW_HEIGHT_IN,
    _LEGEND_TITLED_OVERHEAD_IN,
    _NOTE_LINE_HEIGHT_IN,
    align_lower_legend_stack_top_to_layout,
    center_legend_text,
    legend_display_rows,
    legend_note_lines,
    render_below_figure_legend,
    reserve_footer_space,
    set_footer_min_plot_height,
)
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN,
    SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN,
    TRANSITION_PANEL_TITLE_PAD,
    bottom_panel_indices,
    hide_unused_axes,
    multi_impact_panel_figure_size,
    show_panel_x_labels,
)
from pyaesa.shared.figures.multi_year_transitions import (
    TransitionMarker,
    render_transition_markers,
    transition_boundary_x,
    transition_title_pad,
)
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.figures.trajectory_bands import (
    SUMMARY_COLUMNS,
    render_trajectory_band,
    render_trajectory_band_legend_below,
    trajectory_band_legend_handles,
)
from pyaesa.shared.figures.value_order import (
    finite_average,
    order_labels_by_average_within_group_rank,
)
from pyaesa.shared.figures.violin_summary import (
    render_violin_summaries,
    violin_summary_legend_handler_map,
    violin_summary_legend_handles,
    violin_summary_legend_kwargs,
)

_DEFAULT_COLOR = DEFAULT_SINGLE_SERIES_COLOR
_STATIC_ASR_COLOR = DEFAULT_SINGLE_SERIES_COLOR
_LCA_COLOR = LCA_COMPONENT_COLOR
_UNCERTAINTY_ACC_COMPONENT_COLOR = "#54A24B"
_UNCERTAINTY_LEGEND_COLOR = "#666666"
_CUMULATIVE_VALUES_COLUMN = "__cumulative_values"
_LINE_ALPHA = 0.82
_PANEL_TITLE_PAD = 5
_TWO_COLUMN_PANEL_HSPACE = 0.32
_TWO_COLUMN_TRANSITION_HSPACE = 0.42
_TWO_COLUMN_COMPONENT_TRANSITION_HSPACE = 0.54
_TWO_COLUMN_PANEL_TOP = DOUBLE_COLUMN_TITLE_TOP
_TWO_COLUMN_TRANSITION_TOP = DOUBLE_COLUMN_TITLE_TOP
_TWO_COLUMN_COMPONENT_TRANSITION_TOP = DOUBLE_COLUMN_TITLE_TOP
_COMPACT_SINGLE_IMPACT_TWO_PANEL_SIZE = (15.5, 2.7)
_DYNAMIC_FIGURE_SIZE = (15.8, 15.0)
_DYNAMIC_ROW_COUNT = 4
_DYNAMIC_MIN_PLOT_HEIGHT_IN = _DYNAMIC_ROW_COUNT * SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN
_DYNAMIC_MULTI_METHOD_ROW_HEIGHT_SCALE = 1.3
_COMPACT_SINGLE_IMPACT_TOP = 0.88
_NO_TRANSITION_TWO_PANEL_TITLE_PAD = 6
_STACKED_LABEL_TITLE_PAD = 24
_TWO_PANEL_COMPONENT_TRANSITION_TITLE_PAD = _STACKED_LABEL_TITLE_PAD
_PANEL_COMPONENT_TRANSITION_TITLE_PAD = _STACKED_LABEL_TITLE_PAD
_FREQUENCY_BOX_TITLE_PAD = 36
_DYNAMIC_FOOTER_STACK_GAP_IN = 0.04
_NON_GLOBAL_POST_STUDY_TITLE_PAD = _STACKED_LABEL_TITLE_PAD


def _two_panel_top() -> float:
    return _COMPACT_SINGLE_IMPACT_TOP


def _two_panel_title_pad(*, markers: list[TransitionMarker]) -> int:
    return transition_title_pad(
        markers,
        no_transition=_NO_TRANSITION_TWO_PANEL_TITLE_PAD,
        single_transition=TRANSITION_PANEL_TITLE_PAD,
        component_transition=_TWO_PANEL_COMPONENT_TRANSITION_TITLE_PAD,
    )


def _dynamic_non_global_title_pad(
    *,
    markers: list[TransitionMarker],
    post_years: list[int],
) -> int:
    return max(
        _two_panel_title_pad(markers=markers),
        _NON_GLOBAL_POST_STUDY_TITLE_PAD if post_years else 0,
    )


def _dynamic_min_plot_height(*, include_method_in_label: bool) -> float:
    scale = _DYNAMIC_MULTI_METHOD_ROW_HEIGHT_SCALE if include_method_in_label else 1.0
    return _DYNAMIC_MIN_PLOT_HEIGHT_IN * scale


def _uses_component_transition_layout(markers: list[TransitionMarker]) -> bool:
    return (
        transition_title_pad(
            markers,
            no_transition=0,
            single_transition=1,
            component_transition=2,
        )
        == 2
    )


def _panel_transition_title_pad(markers: list[TransitionMarker]) -> int:
    return transition_title_pad(
        markers,
        no_transition=_PANEL_TITLE_PAD,
        single_transition=TRANSITION_PANEL_TITLE_PAD,
        component_transition=_PANEL_COMPONENT_TRANSITION_TITLE_PAD,
    )


def _two_column_hspace(*, has_transitions: bool, has_component_transitions: bool) -> float:
    return {
        (False, False): _TWO_COLUMN_PANEL_HSPACE,
        (True, False): _TWO_COLUMN_TRANSITION_HSPACE,
        (False, True): _TWO_COLUMN_COMPONENT_TRANSITION_HSPACE,
        (True, True): _TWO_COLUMN_COMPONENT_TRANSITION_HSPACE,
    }[(bool(has_transitions), bool(has_component_transitions))]


def _two_column_top(*, has_transitions: bool, has_component_transitions: bool) -> float:
    return {
        (False, False): _TWO_COLUMN_PANEL_TOP,
        (True, False): _TWO_COLUMN_TRANSITION_TOP,
        (False, True): _TWO_COLUMN_COMPONENT_TRANSITION_TOP,
        (True, True): _TWO_COLUMN_COMPONENT_TRANSITION_TOP,
    }[(bool(has_transitions), bool(has_component_transitions))]


def _band_panel_title_pad(
    markers: list[TransitionMarker],
    *,
    has_frequency_boxes: bool,
) -> int:
    return max(
        _panel_transition_title_pad(markers),
        _FREQUENCY_BOX_TITLE_PAD if has_frequency_boxes else _PANEL_TITLE_PAD,
    )


def plot_band_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_impact_in_label: bool,
    include_method_in_label: bool = True,
    components: ComponentDiagnosticRows | None = None,
    global_ar6_source: UncertaintyGlobalAR6Source | None = None,
    scale_mode: ASRScaleMode = ASR_LOG_SCALE,
) -> list[Path]:
    """Render one multi year ASR uncertainty interval scope."""
    del group_legend, include_impact_in_label
    impacts = ordered_impacts(frame)
    if len(impacts) > 1:
        paths = _plot_impact_panel_band_scope(
            frame=frame,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            include_method_in_label=include_method_in_label,
            scale_mode=scale_mode,
        )
        paths.extend(
            _plot_impact_panel_frequency_scope(
                frame=frame,
                output_stem=output_stem.parent
                / f"{output_stem.name}__frequency_of_no_transgression",
                title=f"{title} | frequency of no-transgression",
                dpi=dpi,
                output_format=output_format,
                include_method_in_label=include_method_in_label,
            )
        )
        return paths
    if _has_cumulative(frame):
        return _plot_dynamic_band_scope(
            frame=frame,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            include_method_in_label=include_method_in_label,
            components=cast(ComponentDiagnosticRows, components),
            global_ar6_source=cast(UncertaintyGlobalAR6Source, global_ar6_source),
            scale_mode=scale_mode,
        )
    fig, axes = plt.subplots(
        ncols=2,
        figsize=_COMPACT_SINGLE_IMPACT_TWO_PANEL_SIZE,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.24},
    )
    set_footer_min_plot_height(
        fig,
        height_in=_two_panel_min_plot_height(include_method_in_label=include_method_in_label),
    )
    axis = axes[0]
    frequency_axis = axes[1]
    render_figure_title(fig, title)
    values, years, markers = _render_band_axis(
        axis=axis,
        frame=frame,
        include_method_in_label=include_method_in_label,
        color_map=_static_asr_color_map(
            _series_labels(frame=frame, include_method_in_label=include_method_in_label),
            include_method_in_label=include_method_in_label,
        ),
    )
    _format_band_axis(
        axis=axis,
        frame=frame,
        values=values,
        years=years,
        show_x_labels=True,
        grouped_legend=False,
        scale_mode=scale_mode,
    )
    pathway_title_pad = _two_panel_title_pad(markers=markers)
    axis.set_title(
        "ASR pathways",
        fontweight="bold",
        pad=pathway_title_pad,
    )
    render_transition_markers(axis, markers=markers)
    frequency_markers = _render_frequency_axis(
        axis=frequency_axis,
        frame=frame,
        include_method_in_label=include_method_in_label,
        show_x_labels=True,
        group_legend=False,
    )
    frequency_title_pad = _two_panel_title_pad(markers=frequency_markers)
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            title,
            default_top=_two_panel_top(),
            panel_title_pad=max(pathway_title_pad, frequency_title_pad),
        )
    )
    frequency_axis.set_title(
        "Frequency of no-transgression",
        fontweight="bold",
        pad=frequency_title_pad,
    )
    render_trajectory_band_legend_below(
        fig,
        color=_trajectory_color(axis),
        extra_height_in=asr_risk_scale_footer_extra_height(),
        frameon=False,
        title="Uncertainty",
    )
    render_asr_risk_scale_footer(fig, frame=frame)
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def plot_mean_line_scope(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_impact_in_label: bool,
    include_method_in_label: bool = True,
    limits: tuple[float, float] | None = None,
    components: ComponentDiagnosticRows | None = None,
    global_ar6_source: UncertaintyGlobalAR6Source | None = None,
    scale_mode: ASRScaleMode = ASR_LOG_SCALE,
) -> list[Path]:
    """Render a method comparison scope from ASR mean values."""
    del include_impact_in_label
    impacts = ordered_impacts(frame)
    if len(impacts) > 1:
        common_limits = (
            limits if limits is not None else _common_band_limits(frame, scale_mode=scale_mode)
        )
        paths: list[Path] = []
        for impact in impacts:
            scoped = frame.loc[frame["impact"].astype(str).eq(str(impact))].copy()
            paths.extend(
                plot_mean_line_scope(
                    frame=scoped,
                    requested_years=requested_years,
                    output_stem=output_stem.parent / f"{output_stem.name}__{impact}",
                    title=f"{title} | {impact_panel_title(scoped, impact=str(impact))}",
                    dpi=dpi,
                    output_format=output_format,
                    group_legend=group_legend,
                    include_impact_in_label=False,
                    include_method_in_label=include_method_in_label,
                    limits=common_limits,
                    components=components,
                    global_ar6_source=global_ar6_source,
                    scale_mode=scale_mode,
                )
            )
        return paths
    if _has_cumulative(frame):
        return _plot_dynamic_mean_scope(
            frame=frame,
            requested_years=requested_years,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            components=cast(ComponentDiagnosticRows, components),
            global_ar6_source=cast(UncertaintyGlobalAR6Source, global_ar6_source),
            scale_mode=scale_mode,
        )
    fig, axes = plt.subplots(
        ncols=2,
        figsize=_COMPACT_SINGLE_IMPACT_TWO_PANEL_SIZE,
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.24},
    )
    set_footer_min_plot_height(
        fig,
        height_in=_two_panel_min_plot_height(include_method_in_label=include_method_in_label),
    )
    axis = axes[0]
    frequency_axis = axes[1]
    render_figure_title(fig, title)
    values, years, markers = _render_mean_axis(
        axis=axis,
        frame=frame,
        group_legend=group_legend,
        include_method_in_label=include_method_in_label,
        line_alpha=MULTI_METHOD_LINE_ALPHA if group_legend else _LINE_ALPHA,
        color_map=_static_asr_color_map(
            _series_labels(frame=frame, include_method_in_label=include_method_in_label),
            include_method_in_label=include_method_in_label,
        ),
    )
    _format_band_axis(
        axis=axis,
        frame=frame,
        values=values,
        years=years,
        show_x_labels=True,
        grouped_legend=group_legend,
        limits=limits,
        scale_mode=scale_mode,
    )
    pathway_title_pad = _two_panel_title_pad(markers=markers)
    axis.set_title(
        "ASR pathways",
        fontweight="bold",
        pad=pathway_title_pad,
    )
    render_transition_markers(axis, markers=markers)
    frequency_markers = _render_frequency_axis(
        axis=frequency_axis,
        frame=frame,
        include_method_in_label=include_method_in_label,
        show_x_labels=True,
        group_legend=group_legend,
    )
    frequency_title_pad = _two_panel_title_pad(markers=frequency_markers)
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            title,
            default_top=_two_panel_top(),
            panel_title_pad=max(pathway_title_pad, frequency_title_pad),
        )
    )
    frequency_axis.set_title(
        "Frequency of no-transgression",
        fontweight="bold",
        pad=frequency_title_pad,
    )
    footer_note = _frequency_note(MEAN_LINE_NOTE)
    risk_extra = asr_risk_scale_footer_extra_height()
    render_grouped_deterministic_legend_below(
        axis,
        legend_note=footer_note,
        extra_height_in=risk_extra,
    )
    render_asr_risk_scale_footer(fig, frame=frame)
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_dynamic_band_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    include_method_in_label: bool,
    components: ComponentDiagnosticRows,
    global_ar6_source: UncertaintyGlobalAR6Source,
    scale_mode: ASRScaleMode,
) -> list[Path]:
    component_acc, component_lca = _component_scope(
        components=components,
        frame=frame,
        include_method_axis=include_method_in_label,
    )
    target_unit = visible_values(component_lca, "impact_unit")[0]
    global_ar6 = uncertainty_global_ar6_rows_from_source(
        source=global_ar6_source,
        asr_frame=frame,
        requested_years=list(components.requested_years),
        target_unit=target_unit,
    )
    asr_color_map = _asr_series_color_map(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    acc_color_map = _component_colors_for_asr_scope(
        asr_color_map=asr_color_map,
        include_method_in_label=include_method_in_label,
    )
    cumulative_label_order = _dynamic_cumulative_label_order(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    paths: list[Path] = []
    for include_post, suffix in ((True, "__incl_post"), (False, "__excl_post")):
        post_years = global_ar6.post_years if include_post else []
        transition_shade_right = _post_study_transition_right(post_years)
        fig, axes = plt.subplots(
            nrows=4,
            ncols=2,
            figsize=_DYNAMIC_FIGURE_SIZE,
            gridspec_kw={
                "width_ratios": [3.0, 1.22],
                "height_ratios": _dynamic_height_ratios(components=components),
                "hspace": 0.44,
                "wspace": 0.24,
            },
        )
        set_footer_min_plot_height(
            fig,
            height_in=_dynamic_min_plot_height(
                include_method_in_label=include_method_in_label,
            ),
        )
        render_figure_title(fig, title)
        global_title_pad = global_ar6_panel_title_pad(post_years)
        global_key_entries = _render_global_ar6_row(
            axes=axes,
            global_ar6=global_ar6,
            post_years=post_years,
            title_pad=global_title_pad,
            emissions_mode=components.emissions_mode,
        )
        component_values, component_years = _render_component_band_axis(
            axis=axes[1, 0],
            acc_rows=component_acc,
            lca_rows=component_lca,
            include_method_in_label=include_method_in_label,
            color_map=acc_color_map,
        )
        component_markers = _visible_series_transition_markers(
            frame=frame,
            include_method_in_label=include_method_in_label,
        )
        component_cumulative_values = _render_component_cumulative_axis(
            axis=axes[1, 1],
            acc_rows=component_acc,
            lca_rows=component_lca,
            include_method_in_label=include_method_in_label,
            color_map=acc_color_map,
            label_order=cumulative_label_order,
        )
        component_pathway_limits = component_axis_limits(
            values=component_values,
            scale_mode=scale_mode,
        )
        component_cumulative_limits = component_axis_limits(
            values=component_cumulative_values,
            scale_mode=scale_mode,
        )
        component_title_pad = _dynamic_non_global_title_pad(
            markers=component_markers,
            post_years=post_years,
        )
        fig.subplots_adjust(
            top=title_layout_top(
                fig,
                title,
                default_top=0.93,
                panel_title_pad=max(global_title_pad, component_title_pad),
            )
        )
        format_acc_lca_component_axis(
            axis=axes[1, 0],
            frame=component_lca,
            years=component_years,
            show_x_labels=False,
            title=acc_lca_pathway_title(emissions_mode=components.emissions_mode),
            limits=component_pathway_limits,
            markers=component_markers,
            title_pad=component_title_pad,
            scale_mode=scale_mode,
            transition_shade_right=transition_shade_right,
        )
        format_acc_lca_component_axis(
            axis=axes[1, 1],
            frame=component_lca,
            years=[],
            show_x_labels=False,
            title=acc_lca_cumulative_title(emissions_mode=components.emissions_mode),
            limits=component_cumulative_limits,
            title_pad=component_title_pad,
            scale_mode=scale_mode,
        )
        if include_post:
            _extend_post_study_pathway_axis(
                axis=axes[1, 0],
                study_years=list(components.requested_years),
                post_years=post_years,
                show_x_labels=False,
            )
        row_index = 2
        pathway_axis = axes[row_index, 0]
        cumulative_axis = axes[row_index, 1]
        frequency_axis = axes[row_index + 1, 0]
        cumulative_frequency_axis = axes[row_index + 1, 1]
        values, years, markers = _render_band_axis(
            axis=pathway_axis,
            frame=frame,
            include_method_in_label=include_method_in_label,
            color_map=asr_color_map,
        )
        cumulative_values = _render_cumulative_asr_axis(
            axis=cumulative_axis,
            frame=frame,
            include_method_in_label=include_method_in_label,
            grouped_legend=False,
            show_x_labels=False,
            color_map=asr_color_map,
            label_order=cumulative_label_order,
        )
        row_limits = _asr_row_limits(
            frame,
            values,
            cumulative_values,
            scale_mode=scale_mode,
        )
        _format_band_axis(
            axis=pathway_axis,
            frame=frame,
            values=values,
            years=years,
            show_x_labels=False,
            grouped_legend=False,
            limits=row_limits,
            scale_mode=scale_mode,
        )
        if include_post:
            _extend_post_study_pathway_axis(
                axis=pathway_axis,
                study_years=list(components.requested_years),
                post_years=post_years,
                show_x_labels=False,
            )
        asr_title_pad = _dynamic_non_global_title_pad(
            markers=markers,
            post_years=post_years,
        )
        pathway_axis.set_title("ASR pathways", fontweight="bold", pad=asr_title_pad)
        render_transition_markers(
            pathway_axis,
            markers=markers,
            shade_right=transition_shade_right,
        )
        apply_scaled_asr_axis_policy(
            cumulative_axis,
            values=cumulative_values,
            frame=frame,
            scale_mode=scale_mode,
            grouped_legend=False,
            limits=row_limits,
        )
        cumulative_axis.grid(alpha=0.25, axis="y")
        cumulative_axis.set_title("Cumulative ASR", fontweight="bold", pad=asr_title_pad)
        frequency_markers = _render_frequency_axis(
            axis=frequency_axis,
            frame=frame,
            include_method_in_label=include_method_in_label,
            show_x_labels=True,
            group_legend=False,
            color_map=asr_color_map,
            transition_shade_right=transition_shade_right,
        )
        if include_post:
            _extend_post_study_pathway_axis(
                axis=frequency_axis,
                study_years=list(components.requested_years),
                post_years=post_years,
                show_x_labels=True,
            )
        frequency_title_pad = _dynamic_non_global_title_pad(
            markers=frequency_markers,
            post_years=post_years,
        )
        frequency_axis.set_title(
            "Frequency of no-transgression",
            fontweight="bold",
            pad=frequency_title_pad,
        )
        _render_cumulative_frequency_axis(
            axis=cumulative_frequency_axis,
            frame=frame,
            include_method_in_label=include_method_in_label,
            show_x_labels=True,
            color_map=asr_color_map,
            label_order=cumulative_label_order,
        )
        cumulative_frequency_axis.set_title(
            "Cumulative frequency of no-transgression",
            fontweight="bold",
            pad=frequency_title_pad,
        )
        distribution_handles = [
            *trajectory_band_legend_handles(color=_UNCERTAINTY_LEGEND_COLOR),
            *violin_summary_legend_handles(),
        ]
        _render_centered_dynamic_distribution_legend(
            fig=fig,
            handles=distribution_handles,
            ncol=5,
            note=_dynamic_uncertainty_note(frame=frame, include_distribution_note=False),
            extra_height_in=asr_risk_scale_footer_extra_height(),
        )
        render_asr_risk_scale_footer(fig, frame=frame)
        render_ar6_cc_row_key(
            fig=fig,
            left_axis=axes[0, 0],
            right_axis=axes[0, 1],
            entries=global_key_entries,
        )
        render_acc_lca_row_key(
            fig=fig,
            left_axis=axes[1, 0],
            right_axis=axes[1, 1],
            include_method_in_label=include_method_in_label,
            acc_color=_UNCERTAINTY_ACC_COMPONENT_COLOR,
            emissions_mode=components.emissions_mode,
        )
        paths.extend(
            save_figure(
                fig,
                output_stem=output_stem.with_name(f"{output_stem.name}{suffix}"),
                output_format=output_format,
                dpi=dpi,
            )
        )
        plt.close(fig)
    return paths


def _plot_dynamic_mean_scope(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
    components: ComponentDiagnosticRows,
    global_ar6_source: UncertaintyGlobalAR6Source,
    scale_mode: ASRScaleMode,
) -> list[Path]:
    component_acc, component_lca = _component_scope(
        components=components,
        frame=frame,
        include_method_axis=include_method_in_label,
    )
    target_unit = visible_values(component_lca, "impact_unit")[0]
    global_ar6 = uncertainty_global_ar6_rows_from_source(
        source=global_ar6_source,
        asr_frame=frame,
        requested_years=list(components.requested_years),
        target_unit=target_unit,
    )
    asr_color_map = _asr_series_color_map(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    acc_color_map = _component_colors_for_asr_scope(
        asr_color_map=asr_color_map,
        include_method_in_label=include_method_in_label,
    )
    cumulative_label_order = _dynamic_cumulative_label_order(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    paths: list[Path] = []
    for include_post, suffix in ((True, "__incl_post"), (False, "__excl_post")):
        post_years = global_ar6.post_years if include_post else []
        transition_shade_right = _post_study_transition_right(post_years)
        fig, axes = plt.subplots(
            nrows=4,
            ncols=2,
            figsize=_DYNAMIC_FIGURE_SIZE,
            gridspec_kw={
                "width_ratios": [1.0, 1.0] if group_legend else [3.0, 1.22],
                "height_ratios": _dynamic_height_ratios(components=components),
                "hspace": 0.44,
                "wspace": 0.24,
            },
        )
        set_footer_min_plot_height(
            fig,
            height_in=_dynamic_min_plot_height(
                include_method_in_label=include_method_in_label,
            ),
        )
        render_figure_title(fig, title)
        global_title_pad = global_ar6_panel_title_pad(post_years)
        global_key_entries = _render_global_ar6_row(
            axes=axes,
            global_ar6=global_ar6,
            post_years=post_years,
            title_pad=global_title_pad,
            emissions_mode=components.emissions_mode,
        )
        component_values, component_years = _render_component_mean_axis(
            axis=axes[1, 0],
            acc_rows=component_acc,
            lca_rows=component_lca,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            color_map=acc_color_map,
        )
        component_markers = _visible_series_transition_markers(
            frame=frame,
            include_method_in_label=include_method_in_label,
        )
        component_cumulative_values = _render_component_cumulative_axis(
            axis=axes[1, 1],
            acc_rows=component_acc,
            lca_rows=component_lca,
            include_method_in_label=include_method_in_label,
            color_map=acc_color_map,
            label_order=cumulative_label_order,
        )
        component_pathway_limits = component_axis_limits(
            values=component_values,
            scale_mode=scale_mode,
        )
        component_cumulative_limits = component_axis_limits(
            values=component_cumulative_values,
            scale_mode=scale_mode,
        )
        component_title_pad = _dynamic_non_global_title_pad(
            markers=component_markers,
            post_years=post_years,
        )
        fig.subplots_adjust(
            top=title_layout_top(
                fig,
                title,
                default_top=0.93,
                panel_title_pad=max(global_title_pad, component_title_pad),
            )
        )
        format_acc_lca_component_axis(
            axis=axes[1, 0],
            frame=component_lca,
            years=component_years,
            show_x_labels=False,
            title=acc_lca_pathway_title(emissions_mode=components.emissions_mode),
            limits=component_pathway_limits,
            markers=component_markers,
            title_pad=component_title_pad,
            scale_mode=scale_mode,
            transition_shade_right=transition_shade_right,
        )
        format_acc_lca_component_axis(
            axis=axes[1, 1],
            frame=component_lca,
            years=[],
            show_x_labels=False,
            title=acc_lca_cumulative_title(emissions_mode=components.emissions_mode),
            limits=component_cumulative_limits,
            title_pad=component_title_pad,
            scale_mode=scale_mode,
        )
        if include_post:
            _extend_post_study_pathway_axis(
                axis=axes[1, 0],
                study_years=list(components.requested_years),
                post_years=post_years,
                show_x_labels=False,
            )
        row_index = 2
        pathway_axis = axes[row_index, 0]
        cumulative_axis = axes[row_index, 1]
        frequency_axis = axes[row_index + 1, 0]
        cumulative_frequency_axis = axes[row_index + 1, 1]
        values, years, markers = _render_mean_axis(
            axis=pathway_axis,
            frame=frame,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            line_alpha=MULTI_METHOD_LINE_ALPHA if group_legend else _LINE_ALPHA,
            color_map=asr_color_map,
        )
        cumulative_values = _render_cumulative_asr_axis(
            axis=cumulative_axis,
            frame=frame,
            include_method_in_label=include_method_in_label,
            grouped_legend=group_legend,
            show_x_labels=False,
            color_map=asr_color_map,
            label_order=cumulative_label_order,
        )
        row_limits = _asr_row_limits(
            frame,
            values,
            cumulative_values,
            scale_mode=scale_mode,
        )
        _format_band_axis(
            axis=pathway_axis,
            frame=frame,
            values=values,
            years=years,
            show_x_labels=False,
            grouped_legend=group_legend,
            limits=row_limits,
            scale_mode=scale_mode,
        )
        if include_post:
            _extend_post_study_pathway_axis(
                axis=pathway_axis,
                study_years=list(components.requested_years),
                post_years=post_years,
                show_x_labels=False,
            )
        asr_title_pad = _dynamic_non_global_title_pad(
            markers=markers,
            post_years=post_years,
        )
        pathway_axis.set_title("ASR pathways", fontweight="bold", pad=asr_title_pad)
        render_transition_markers(
            pathway_axis,
            markers=markers,
            shade_right=transition_shade_right,
        )
        apply_scaled_asr_axis_policy(
            cumulative_axis,
            values=cumulative_values,
            frame=frame,
            scale_mode=scale_mode,
            grouped_legend=group_legend,
            limits=row_limits,
        )
        cumulative_axis.grid(alpha=0.25, axis="y")
        cumulative_axis.set_title("Cumulative ASR", fontweight="bold", pad=asr_title_pad)
        frequency_markers = _render_frequency_axis(
            axis=frequency_axis,
            frame=frame,
            include_method_in_label=include_method_in_label,
            show_x_labels=True,
            group_legend=group_legend,
            color_map=asr_color_map,
            transition_shade_right=transition_shade_right,
        )
        if include_post:
            _extend_post_study_pathway_axis(
                axis=frequency_axis,
                study_years=list(components.requested_years),
                post_years=post_years,
                show_x_labels=True,
            )
        frequency_title_pad = _dynamic_non_global_title_pad(
            markers=frequency_markers,
            post_years=post_years,
        )
        frequency_axis.set_title(
            "Frequency of no-transgression",
            fontweight="bold",
            pad=frequency_title_pad,
        )
        _render_cumulative_frequency_axis(
            axis=cumulative_frequency_axis,
            frame=frame,
            include_method_in_label=include_method_in_label,
            show_x_labels=True,
            color_map=asr_color_map,
            label_order=cumulative_label_order,
        )
        cumulative_frequency_axis.set_title(
            "Cumulative frequency of no-transgression",
            fontweight="bold",
            pad=frequency_title_pad,
        )
        distribution_handles = violin_summary_legend_handles()
        footer_note = _dynamic_uncertainty_note(frame=frame, include_distribution_note=True)
        render_grouped_deterministic_legend_below(
            pathway_axis,
            extra_height_in=(
                asr_risk_scale_footer_extra_height()
                + _dynamic_distribution_footer_extra_height(
                    fig=fig,
                    handles=distribution_handles,
                    ncol=5,
                    note=footer_note,
                )
            ),
        )
        _render_centered_dynamic_distribution_legend(
            fig=fig,
            handles=distribution_handles,
            ncol=5,
            note=footer_note,
            extra_height_in=asr_risk_scale_footer_extra_height(),
        )
        render_asr_risk_scale_footer(fig, frame=frame)
        render_ar6_cc_row_key(
            fig=fig,
            left_axis=axes[0, 0],
            right_axis=axes[0, 1],
            entries=global_key_entries,
        )
        render_acc_lca_row_key(
            fig=fig,
            left_axis=axes[1, 0],
            right_axis=axes[1, 1],
            include_method_in_label=include_method_in_label,
            acc_color=_UNCERTAINTY_ACC_COMPONENT_COLOR,
            emissions_mode=components.emissions_mode,
        )
        paths.extend(
            save_figure(
                fig,
                output_stem=output_stem.with_name(f"{output_stem.name}{suffix}"),
                output_format=output_format,
                dpi=dpi,
            )
        )
        plt.close(fig)
    return paths


def _two_panel_min_plot_height(
    *,
    include_method_in_label: bool,
) -> float:
    return (
        MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN
        if include_method_in_label
        else SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN
    )


def _dynamic_height_ratios(
    *,
    components: ComponentDiagnosticRows,
) -> list[float]:
    del components
    return [1.0, 1.0, 1.0, 1.0]


def _extend_post_study_pathway_axis(
    *,
    axis: Any,
    study_years: list[int],
    post_years: list[int],
    show_x_labels: bool,
) -> None:
    years = [*sorted(int(year) for year in study_years), *post_years]
    axis.set_xlim(float(min(years)) - 0.5, float(max(years)) + 0.5)
    format_year_axis(axis, years=years, show_labels=show_x_labels)
    render_ar6_study_transition(
        axis,
        study_years=sorted(int(year) for year in study_years),
        post_years=post_years,
        show_study_label=False,
    )


def _post_study_transition_right(post_years: list[int]) -> float | None:
    if not post_years:
        return None
    return transition_boundary_x(int(post_years[0]))


def _render_global_ar6_row(
    *,
    axes: Any,
    global_ar6: UncertaintyGlobalAR6Rows,
    post_years: list[int],
    title_pad: int,
    emissions_mode: str | None,
) -> list[tuple[str, str, str]]:
    deterministic = global_ar6.deterministic
    if deterministic is not None:
        category_colors = {
            category: ar6_category_color(category=category)
            for category in visible_values(deterministic.frame, "cc_category")
        }
        _pathway_handles, _budget_handles, visible_negative_flow = render_deterministic_ar6_cc_row(
            axis=axes[0, 0],
            budget_axis=axes[0, 1],
            scoped_frame=deterministic.frame,
            study_years=deterministic.study_years,
            post_years=post_years,
            category_colors=category_colors,
            flow_colors=AR6_CC_FLOW_COLORS,
            pathway_title="Global AR6 CC pathways",
            budget_title="Cumulative global AR6 CC budget",
            show_x_labels=False,
            show_study_label=True,
            title_pad=title_pad,
        )
        return _global_ar6_key_entries(
            visible_negative_flow=visible_negative_flow,
            emissions_mode=emissions_mode,
            negative_style=":",
        )
    summary = _global_ar6_summary_for_post(global_ar6=global_ar6, post_years=post_years)
    budget = _global_ar6_budget_for_post(global_ar6=global_ar6, post_years=post_years)
    render_uncertainty_ar6_cc_row(
        axis=axes[0, 0],
        budget_axis=axes[0, 1],
        scoped_frame=summary,
        budget_frame=budget,
        flow_colors=AR6_CC_FLOW_COLORS,
        study_years=global_ar6.study_years,
        pathway_title="Global AR6 CC pathways",
        budget_title="Cumulative global AR6 CC budget",
        show_x_labels=False,
        show_study_label=True,
        title_pad=title_pad,
    )
    return _global_ar6_key_entries(
        visible_negative_flow=CC_FLOW_NEGATIVE
        in {str(value) for value in summary["cc_flow"].astype(str)},
        emissions_mode=emissions_mode,
        negative_style="-",
    )


def _global_ar6_key_entries(
    *,
    visible_negative_flow: bool,
    emissions_mode: str | None,
    negative_style: str,
) -> list[tuple[str, str, str]]:
    return ar6_cc_flow_key_entries(
        emissions_mode=emissions_mode,
        positive_color=AR6_CC_FLOW_COLORS[CC_FLOW_POSITIVE],
        negative_color=AR6_CC_FLOW_COLORS[CC_FLOW_NEGATIVE],
        visible_negative_flow=visible_negative_flow,
        negative_style=negative_style,
    )


def _global_ar6_summary_for_post(
    *,
    global_ar6: UncertaintyGlobalAR6Rows,
    post_years: list[int],
) -> pd.DataFrame:
    if post_years:
        return global_ar6.summary
    study_end = max(int(year) for year in global_ar6.study_years)
    years = pd.Series(pd.to_numeric(global_ar6.summary["year"], errors="raise")).astype(int)
    return global_ar6.summary.loc[years.le(study_end)].copy()


def _global_ar6_budget_for_post(
    *,
    global_ar6: UncertaintyGlobalAR6Rows,
    post_years: list[int],
) -> pd.DataFrame:
    if post_years:
        return global_ar6.budget
    segment = pd.Series(global_ar6.budget["period_segment"], copy=False).astype(str)
    return global_ar6.budget.loc[segment.eq("study_period")].copy()


def _plot_impact_panel_band_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    include_method_in_label: bool,
    scale_mode: ASRScaleMode,
) -> list[Path]:
    impacts = ordered_impacts(frame)
    label_order = _series_label_order(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    color_map = _static_asr_color_map(
        label_order,
        include_method_in_label=include_method_in_label,
    )
    ncols = 2
    nrows = (len(impacts) + 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=multi_impact_panel_figure_size(nrows=nrows),
        squeeze=False,
    )
    bottom_indices = bottom_panel_indices(panel_count=len(impacts), ncols=ncols)
    first_axis = axes[0, 0]
    common_limits = _common_band_limits(frame, scale_mode=scale_mode)
    has_transitions = False
    has_component_transitions = False
    has_frequency_boxes = False
    for index, impact in enumerate(impacts):
        axis = axes[index // ncols, index % ncols]
        panel = frame.loc[frame["impact"].astype(str).eq(str(impact))].copy()
        values, years, markers = _render_band_axis(
            axis=axis,
            frame=panel,
            include_method_in_label=include_method_in_label,
            label_order=label_order,
            color_map=color_map,
        )
        _format_band_axis(
            axis=axis,
            frame=panel,
            values=values,
            years=years,
            show_x_labels=show_panel_x_labels(
                panel_index=index,
                bottom_indices=bottom_indices,
            ),
            grouped_legend=False,
            limits=common_limits,
            scale_mode=scale_mode,
        )
        if markers:
            has_transitions = True
            has_component_transitions = (
                has_component_transitions or _uses_component_transition_layout(markers)
            )
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            impact_panel_title(panel, impact=str(impact)),
            loc="left",
            pad=_band_panel_title_pad(markers, has_frequency_boxes=has_frequency_boxes),
        )
        render_transition_markers(axis, markers=markers)
    hide_unused_axes(axes=axes, used=len(impacts))
    render_figure_title(fig, title)
    top = title_layout_top(
        fig,
        title,
        default_top=_two_column_top(
            has_transitions=has_transitions,
            has_component_transitions=has_component_transitions,
        ),
        panel_title_pad=(
            _PANEL_COMPONENT_TRANSITION_TITLE_PAD
            if has_component_transitions
            else TRANSITION_PANEL_TITLE_PAD
            if has_transitions
            else _FREQUENCY_BOX_TITLE_PAD
            if has_frequency_boxes
            else _PANEL_TITLE_PAD
        ),
    )
    fig.subplots_adjust(
        hspace=_two_column_hspace(
            has_transitions=has_transitions,
            has_component_transitions=has_component_transitions,
        ),
        wspace=0.16,
        top=top,
    )
    render_trajectory_band_legend_below(
        fig,
        color=_trajectory_color(first_axis),
        prefix_handles=[],
        extra_height_in=asr_risk_scale_footer_extra_height(),
        frameon=False,
        title="Uncertainty",
    )
    render_asr_risk_scale_footer(fig, frame=frame)
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _plot_impact_panel_frequency_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    include_method_in_label: bool,
) -> list[Path]:
    impacts = ordered_impacts(frame)
    label_order = _series_label_order(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    color_map = {label: DEFAULT_SINGLE_SERIES_COLOR for label in label_order}
    ncols = 2
    nrows = (len(impacts) + 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=multi_impact_panel_figure_size(nrows=nrows),
        squeeze=False,
    )
    bottom_indices = bottom_panel_indices(panel_count=len(impacts), ncols=ncols)
    has_transitions = False
    has_component_transitions = False
    for index, impact in enumerate(impacts):
        axis = axes[index // ncols, index % ncols]
        panel = frame.loc[frame["impact"].astype(str).eq(str(impact))].copy()
        markers = _render_frequency_axis(
            axis=axis,
            frame=panel,
            include_method_in_label=include_method_in_label,
            show_x_labels=show_panel_x_labels(
                panel_index=index,
                bottom_indices=bottom_indices,
            ),
            group_legend=False,
            label_order=label_order,
            color_map=color_map,
        )
        if markers:
            has_transitions = True
            has_component_transitions = (
                has_component_transitions or _uses_component_transition_layout(markers)
            )
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            impact_panel_title(panel, impact=str(impact)),
            loc="left",
            pad=_panel_transition_title_pad(markers),
        )
    hide_unused_axes(axes=axes, used=len(impacts))
    render_figure_title(fig, title)
    top = title_layout_top(
        fig,
        title,
        default_top=_two_column_top(
            has_transitions=has_transitions,
            has_component_transitions=has_component_transitions,
        ),
        panel_title_pad=(
            _PANEL_COMPONENT_TRANSITION_TITLE_PAD
            if has_component_transitions
            else TRANSITION_PANEL_TITLE_PAD
            if has_transitions
            else _PANEL_TITLE_PAD
        ),
    )
    fig.subplots_adjust(
        hspace=_two_column_hspace(
            has_transitions=has_transitions,
            has_component_transitions=has_component_transitions,
        ),
        wspace=0.16,
        top=top,
    )
    render_below_figure_legend(
        fig,
        legend_note=None,
        max_columns=1,
        extra_entries=[_fnt_legend_entry(frame)],
    )
    paths = save_figure(fig, output_stem=output_stem, output_format=output_format, dpi=dpi)
    plt.close(fig)
    return paths


def _component_scope(
    *,
    components: ComponentDiagnosticRows,
    frame: pd.DataFrame,
    include_method_axis: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_axis = include_method_axis or bool(visible_values(frame, "__method"))
    acc_source = components.acc_method if method_axis else components.acc_inter
    acc_rows = component_scope_rows(
        acc_source,
        asr_frame=frame,
        include_method_axis=method_axis,
    )
    lca_rows = component_scope_rows(
        components.lca,
        asr_frame=frame,
        include_method_axis=False,
    )
    return acc_rows, lca_rows


def _render_component_band_axis(
    *,
    axis: Any,
    acc_rows: pd.DataFrame,
    lca_rows: pd.DataFrame,
    include_method_in_label: bool,
    color_map: dict[str, str],
) -> tuple[np.ndarray, list[int]]:
    colors = color_map
    years: list[int] = []
    values: list[np.ndarray] = []
    for _key, group in acc_rows.groupby(
        component_series_columns(acc_rows),
        dropna=False,
        sort=True,
    ):
        ordered = group.sort_values("year", kind="stable")
        row = pd.Series(ordered.iloc[0], copy=False)
        label = _component_label(row, include_method=True) if include_method_in_label else "aCC"
        year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise")).astype(int)
        rendered = render_trajectory_band(
            axis,
            years=year_values.to_numpy(dtype=int),
            summaries={column: ordered[column] for column in SUMMARY_COLUMNS},
            color=colors[label],
            line_alpha=MULTI_METHOD_LINE_ALPHA if include_method_in_label else _LINE_ALPHA,
            mean_linewidth=2.15,
            median_linewidth=1.55,
            outer_alpha=0.16,
            inner_alpha=0.28,
        )
        years.extend(int(year) for year in year_values.tolist())
        values.extend(rendered[column] for column in SUMMARY_COLUMNS)
    for _key, group in lca_rows.groupby(
        component_series_columns(lca_rows),
        dropna=False,
        sort=True,
    ):
        ordered = group.sort_values("year", kind="stable")
        year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise")).astype(int)
        rendered = render_trajectory_band(
            axis,
            years=year_values.to_numpy(dtype=int),
            summaries={column: ordered[column] for column in SUMMARY_COLUMNS},
            color=_LCA_COLOR,
            line_alpha=1.0,
            mean_linewidth=lca_component_linewidth(include_method_in_label=include_method_in_label),
            median_linewidth=1.6,
            outer_alpha=0.18,
            inner_alpha=0.32,
            line_path_effects=lca_component_path_effects(
                include_method_in_label=include_method_in_label
            ),
        )
        years.extend(int(year) for year in year_values.tolist())
        values.extend(rendered[column] for column in SUMMARY_COLUMNS)
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64), years


def _render_component_mean_axis(
    *,
    axis: Any,
    acc_rows: pd.DataFrame,
    lca_rows: pd.DataFrame,
    group_legend: bool,
    include_method_in_label: bool,
    color_map: dict[str, str],
) -> tuple[np.ndarray, list[int]]:
    colors = color_map
    years: list[int] = []
    values: list[float] = []
    seen: set[str] = set()
    for _key, group in acc_rows.groupby(
        component_series_columns(acc_rows),
        dropna=False,
        sort=True,
    ):
        ordered = group.sort_values("year", kind="stable")
        row = pd.Series(ordered.iloc[0], copy=False)
        label = _component_label(row, include_method=True) if include_method_in_label else "aCC"
        year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise")).astype(int)
        mean = pd.Series(pd.to_numeric(ordered["mean"], errors="raise")).astype(float)
        visible_label = (
            label
            if group_legend and include_method_in_label and label and label not in seen
            else "_nolegend_"
        )
        line = axis.plot(
            year_values.to_numpy(dtype=int),
            mean.to_numpy(dtype=float),
            linewidth=1.7,
            color=colors[label],
            alpha=MULTI_METHOD_LINE_ALPHA if group_legend else _LINE_ALPHA,
            label=visible_label,
        )[0]
        if group_legend and visible_label != "_nolegend_":
            bind_deterministic_legend_group(line, legend_group_from_row(row))
        seen.add(label)
        years.extend(int(year) for year in year_values.tolist())
        values.extend(float(value) for value in mean.tolist())
    for _key, group in lca_rows.groupby(
        component_series_columns(lca_rows),
        dropna=False,
        sort=True,
    ):
        ordered = group.sort_values("year", kind="stable")
        year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise")).astype(int)
        mean = pd.Series(pd.to_numeric(ordered["mean"], errors="raise")).astype(float)
        line = axis.plot(
            year_values.to_numpy(dtype=int),
            mean.to_numpy(dtype=float),
            linewidth=lca_component_linewidth(include_method_in_label=include_method_in_label),
            color=_LCA_COLOR,
            alpha=1.0,
            label="_nolegend_",
        )[0]
        effects = lca_component_path_effects(include_method_in_label=include_method_in_label)
        line.set_path_effects(effects)
        seen.add("LCA")
        years.extend(int(year) for year in year_values.tolist())
        values.extend(float(value) for value in mean.tolist())
    return np.asarray(values, dtype=np.float64), years


def _render_component_cumulative_axis(
    *,
    axis: Any,
    acc_rows: pd.DataFrame,
    lca_rows: pd.DataFrame,
    include_method_in_label: bool,
    color_map: dict[str, str],
    label_order: list[str] | None = None,
) -> np.ndarray:
    acc_entries = _component_cumulative_entries(
        acc_rows,
        include_method_in_label=include_method_in_label,
        label_order=label_order,
    )
    lca_entries = _component_cumulative_entries(lca_rows, include_method_in_label=False)
    labels = [label for label, _values in acc_entries]
    positions = np.arange(len(labels), dtype=float)
    colors_by_label = color_map
    colors = [colors_by_label[label] for label in labels]
    acc_values = [values for _label, values in acc_entries]
    render_violin_summaries(
        axis,
        values=acc_values,
        positions=positions,
        colors=colors,
        width=0.42,
        alpha=ASR_RISK_BACKGROUND_VIOLIN_ALPHA,
    )
    lca_values = [values for _label, values in lca_entries]
    repeated_lca = [lca_values[0] for _label in labels] if lca_values and labels else lca_values
    lca_positions = positions if labels else np.arange(len(repeated_lca), dtype=float)
    render_violin_summaries(
        axis,
        values=repeated_lca,
        positions=lca_positions,
        colors=[_LCA_COLOR] * len(repeated_lca),
        width=0.24,
        alpha=ASR_RISK_BACKGROUND_VIOLIN_ALPHA,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([])
    axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.5, max(0.5, float(len(positions)) - 0.5))
    numeric = [*acc_values, *repeated_lca]
    return np.concatenate(numeric) if numeric else np.empty(0, dtype=np.float64)


def _component_cumulative_entries(
    frame: pd.DataFrame,
    *,
    include_method_in_label: bool,
    label_order: list[str] | None = None,
) -> list[tuple[str, np.ndarray]]:
    entries: list[tuple[str, np.ndarray]] = []
    for _key, group in frame.groupby(component_series_columns(frame), dropna=False, sort=True):
        row = pd.Series(group.iloc[0], copy=False)
        label = _component_label(row, include_method=include_method_in_label)
        values = np.asarray(row["__component_cumulative_values"], dtype=np.float64)
        entries.append((label, values))
    return _ordered_cumulative_entries(entries, label_order=label_order)


def _component_label(row: pd.Series, *, include_method: bool) -> str:
    return str(row.get("__method", "")).strip() if include_method else "aCC"


def _render_band_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    include_method_in_label: bool,
    color_map: dict[str, str],
    label_order: list[str] | None = None,
) -> tuple[np.ndarray, list[int], list[TransitionMarker]]:
    colors = color_map
    years: list[int] = []
    values: list[np.ndarray] = []
    for label, group in _ordered_series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
        label_order=label_order,
    ):
        ordered = group.sort_values("year", kind="stable")
        year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise")).astype(int)
        run_values = render_trajectory_band(
            axis,
            years=year_values.to_numpy(dtype=int),
            summaries={column: ordered[column] for column in SUMMARY_COLUMNS},
            color=colors[label],
            line_alpha=MULTI_METHOD_LINE_ALPHA if include_method_in_label else _LINE_ALPHA,
        )
        years.extend(int(year) for year in year_values.tolist())
        values.extend(run_values[column] for column in SUMMARY_COLUMNS)
    return (
        np.concatenate(values) if values else np.empty(0),
        years,
        _visible_series_transition_markers(
            frame=frame,
            include_method_in_label=include_method_in_label,
        ),
    )


def _render_mean_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    group_legend: bool,
    include_method_in_label: bool,
    line_alpha: float = _LINE_ALPHA,
    color_map: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[int], list[TransitionMarker]]:
    years: list[int] = []
    values: list[float] = []
    seen: set[str] = set()
    labels = _series_labels(frame=frame, include_method_in_label=include_method_in_label)
    colors = color_map or single_or_distinct_colors(labels)
    for label, group in _series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
    ):
        ordered = group.sort_values("year", kind="stable")
        year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise")).astype(int)
        mean = pd.Series(pd.to_numeric(ordered["mean"], errors="raise")).astype(float)
        visible_label = label if label and label not in seen else "_nolegend_"
        line = axis.plot(
            year_values.to_numpy(dtype=int),
            mean.to_numpy(dtype=float),
            linewidth=1.7,
            color=colors[label],
            alpha=float(line_alpha),
            label=visible_label,
        )[0]
        if group_legend and visible_label != "_nolegend_":
            bind_deterministic_legend_group(line, legend_group_from_row(pd.Series(group.iloc[0])))
        seen.add(label)
        years.extend(int(year) for year in year_values.tolist())
        values.extend(float(value) for value in mean.tolist())
    return (
        np.asarray(values, dtype=float),
        years,
        _visible_series_transition_markers(
            frame=frame,
            include_method_in_label=include_method_in_label,
        ),
    )


def _visible_series_transition_markers(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> list[TransitionMarker]:
    return merged_asr_transition_markers(
        group.sort_values("year", kind="stable")
        for _label, group in _series_groups(
            frame=frame,
            include_method_in_label=include_method_in_label,
        )
    )


def _format_band_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    values: np.ndarray,
    years: list[int],
    show_x_labels: bool,
    grouped_legend: bool = False,
    limits: tuple[float, float] | None = None,
    threshold_background_alpha_scale: float = ASR_NON_POLAR_RISK_BACKGROUND_ALPHA_SCALE,
    scale_mode: ASRScaleMode = ASR_LOG_SCALE,
) -> None:
    year_values = sorted({int(year) for year in years})
    axis.set_xlim(min(year_values) - 0.5, max(year_values) + 0.5)
    format_year_axis(axis, years=year_values, show_labels=show_x_labels)
    apply_scaled_asr_axis_policy(
        axis,
        values=values,
        frame=frame,
        scale_mode=scale_mode,
        grouped_legend=grouped_legend,
        limits=limits,
        threshold_background_alpha_scale=threshold_background_alpha_scale,
    )


def _common_band_limits(frame: pd.DataFrame, *, scale_mode: ASRScaleMode) -> tuple[float, float]:
    values = [
        pd.Series(pd.to_numeric(frame[column], errors="raise")).to_numpy(dtype=np.float64)
        for column in SUMMARY_COLUMNS
        if column in frame.columns
    ]
    numeric = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    return asr_axis_limits(values=numeric, frame=frame, scale_mode=scale_mode)


def _asr_row_limits(
    frame: pd.DataFrame,
    *values: np.ndarray,
    scale_mode: ASRScaleMode,
) -> tuple[float, float]:
    numeric = np.concatenate([np.asarray(value, dtype=np.float64) for value in values])
    return asr_axis_limits(values=numeric, frame=frame, scale_mode=scale_mode)


def _has_cumulative(frame: pd.DataFrame) -> bool:
    return _CUMULATIVE_VALUES_COLUMN in frame.columns


def _render_frequency_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    include_method_in_label: bool,
    show_x_labels: bool,
    group_legend: bool,
    label_order: list[str] | None = None,
    color_map: dict[str, str] | None = None,
    transition_shade_right: float | None = None,
) -> list[TransitionMarker]:
    labels = label_order or _series_labels(
        frame=frame,
        include_method_in_label=include_method_in_label,
    )
    colors = color_map or frequency_color_map(
        labels,
        include_method_in_label=include_method_in_label,
    )
    years: list[int] = []
    markers: dict[int, TransitionMarker] = {}
    visible: set[str] = set()
    for label, group in _series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
    ):
        ordered = group.sort_values("year", kind="stable")
        year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise")).astype(int)
        frequency = pd.Series(pd.to_numeric(ordered[FNT_FRACTION_COLUMN], errors="raise")).astype(
            float
        )
        visible_label = (
            label
            if include_method_in_label and not group_legend and label and label not in visible
            else "_nolegend_"
        )
        axis.plot(
            year_values.to_numpy(dtype=int),
            frequency.to_numpy(dtype=float) * 100.0,
            linewidth=1.7,
            color=colors[label],
            alpha=MULTI_METHOD_LINE_ALPHA if include_method_in_label else 0.66,
            label=visible_label,
        )
        visible.add(label)
        years.extend(int(year) for year in year_values.tolist())
        for marker in asr_transition_markers(ordered):
            markers[int(marker.year)] = marker
    axis.set_xlim(min(years) - 0.5, max(years) + 0.5)
    format_year_axis(axis, years=sorted(set(years)), show_labels=show_x_labels)
    apply_frequency_axis(axis)
    render_transition_markers(
        axis,
        markers=list(markers.values()),
        shade_right=transition_shade_right,
    )
    return list(markers.values())


def _fnt_legend_entry(frame: pd.DataFrame) -> tuple[Any, str]:
    return fnt_legend_entry(cc_source=visible_values(frame, "lcia_method")[0])


def _render_cumulative_asr_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    include_method_in_label: bool,
    grouped_legend: bool,
    show_x_labels: bool = True,
    color_map: dict[str, str] | None = None,
    label_order: list[str] | None = None,
) -> np.ndarray:
    entries = _cumulative_entries(
        frame=frame,
        include_method_in_label=include_method_in_label,
        color_map=color_map,
        label_order=label_order,
    )
    labels = list(dict.fromkeys(label for label, _values, _color, _frequency in entries))
    positions = np.asarray(
        [labels.index(label) for label, _values, _color, _frequency in entries],
        dtype=float,
    )
    values = [np.asarray(payload, dtype=np.float64) for _label, payload, _color, _freq in entries]
    colors_by_label = color_map or frequency_color_map(
        labels,
        include_method_in_label=include_method_in_label,
    )
    colors = [colors_by_label[label] for label in labels]
    render_violin_summaries(
        axis,
        values=values,
        positions=positions,
        colors=colors,
        width=0.34,
        alpha=ASR_RISK_BACKGROUND_VIOLIN_ALPHA,
    )
    _format_cumulative_category_axis(
        axis=axis,
        labels=labels,
        include_method_in_label=include_method_in_label,
        show_x_labels=show_x_labels,
    )
    numeric = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    del grouped_legend
    return numeric


def _render_cumulative_frequency_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    include_method_in_label: bool,
    show_x_labels: bool = True,
    color_map: dict[str, str] | None = None,
    label_order: list[str] | None = None,
) -> None:
    entries = _cumulative_entries(
        frame=frame,
        include_method_in_label=include_method_in_label,
        color_map=color_map,
        label_order=label_order,
    )
    labels = list(dict.fromkeys(label for label, _values, _color, _frequency in entries))
    positions = np.asarray(
        [labels.index(label) for label, _values, _color, _frequency in entries],
        dtype=float,
    )
    frequencies = np.asarray(
        [float(frequency) * 100.0 for _label, _values, _color, frequency in entries],
        dtype=float,
    )
    colors = [color for _label, _values, color, _freq in entries]
    axis.bar(positions, frequencies, width=0.58, color=colors, alpha=_LINE_ALPHA, zorder=3)
    _render_zero_frequency_bars(
        axis=axis, positions=positions, frequencies=frequencies, colors=colors
    )
    _format_cumulative_category_axis(
        axis=axis,
        labels=labels,
        include_method_in_label=include_method_in_label,
        show_x_labels=show_x_labels,
    )
    apply_frequency_axis(axis)
    axis.grid(alpha=0.25, axis="y")


def _render_zero_frequency_bars(
    *,
    axis: Any,
    positions: np.ndarray,
    frequencies: np.ndarray,
    colors: list[str],
) -> None:
    positions_array = np.asarray(positions, dtype=float)
    frequencies_array = np.asarray(frequencies, dtype=float)
    zero_mask = np.isfinite(frequencies_array) & np.isclose(frequencies_array, 0.0)
    visible_positions = positions_array[zero_mask]
    visible_colors = [color for color, keep in zip(colors, zero_mask.tolist(), strict=True) if keep]
    if visible_positions.size == 0:
        return
    axis.hlines(
        np.zeros_like(visible_positions, dtype=float),
        visible_positions - 0.29,
        visible_positions + 0.29,
        colors=visible_colors,
        linewidth=2.0,
        zorder=5,
        clip_on=False,
    )


def _format_cumulative_category_axis(
    *,
    axis: Any,
    labels: list[str],
    include_method_in_label: bool,
    show_x_labels: bool,
) -> None:
    ticks = np.arange(len(labels), dtype=float)
    axis.set_xlim(-0.5, max(0.5, float(len(labels)) - 0.5))
    if show_x_labels and include_method_in_label and any(str(label).strip() for label in labels):
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels, rotation=45, ha="right")
    else:
        axis.set_xticks(ticks)
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)


def _cumulative_entries(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
    color_map: dict[str, str] | None = None,
    label_order: list[str] | None = None,
) -> list[tuple[str, np.ndarray, str, float]]:
    labels = _series_labels(frame=frame, include_method_in_label=include_method_in_label)
    colors = color_map or single_or_distinct_colors(labels)
    entries: list[tuple[str, np.ndarray, str, float]] = []
    for label, group in _ordered_series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
        label_order=label_order,
    ):
        row = pd.Series(group.iloc[0], copy=False)
        values = np.asarray(row[_CUMULATIVE_VALUES_COLUMN], dtype=np.float64)
        color = colors[label] if include_method_in_label else _DEFAULT_COLOR
        frequency_value = pd.Series(
            pd.to_numeric(pd.Series([row[CUMULATIVE_FNT_FRACTION_COLUMN]]), errors="raise")
        ).iloc[0]
        frequency = float(frequency_value)
        entries.append(
            (label if include_method_in_label else "Study period", values, color, frequency)
        )
    return entries


def _dynamic_cumulative_label_order(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> list[str] | None:
    if not include_method_in_label:
        return None
    scores: list[tuple[str, float]] = []
    for label, group in _series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
    ):
        row = pd.Series(group.iloc[0], copy=False)
        values = np.asarray(row[_CUMULATIVE_VALUES_COLUMN], dtype=np.float64)
        scores.append((label, finite_average(values.tolist()) or float("-inf")))
    return [
        label
        for label, _score in sorted(
            scores,
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _ordered_cumulative_entries(
    entries: list[tuple[str, np.ndarray]],
    *,
    label_order: list[str] | None,
) -> list[tuple[str, np.ndarray]]:
    if label_order is None:
        return entries
    ranks = {label: index for index, label in enumerate(label_order)}
    return sorted(entries, key=lambda item: (ranks.get(item[0], len(ranks)), item[0]))


def _series_groups(*, frame: pd.DataFrame, include_method_in_label: bool):
    excluded = _series_group_excluded_columns(frame)
    group_columns = [
        column
        for column in frame.columns
        if column not in {*excluded, "year"} and not str(column).startswith("__figure")
    ]
    for _key, group in frame.groupby(group_columns, dropna=False, sort=True):
        row = pd.Series(group.iloc[0], copy=False)
        yield _row_label(row, include_method_in_label=include_method_in_label), group


def _series_group_excluded_columns(frame: pd.DataFrame) -> set[str]:
    excluded = {
        *SUMMARY_COLUMNS,
        *ASR_TRANSITION_SERIES_EXCLUDED_COLUMNS,
        "std",
        "min",
        "max",
        "public_row_id",
        "year",
        "asr_metric",
        FNT_FRACTION_COLUMN,
        _CUMULATIVE_VALUES_COLUMN,
        CUMULATIVE_FNT_FRACTION_COLUMN,
        MODEL_SCENARIO_PAIR_COUNT_COLUMN,
        MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
    }
    return excluded


def _series_labels(*, frame: pd.DataFrame, include_method_in_label: bool) -> list[str]:
    return list(
        dict.fromkeys(
            label
            for label, _group in _series_groups(
                frame=frame,
                include_method_in_label=include_method_in_label,
            )
        )
    )


def _asr_series_color_map(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> dict[str, str]:
    labels = _series_labels(frame=frame, include_method_in_label=include_method_in_label)
    if not include_method_in_label:
        return {label: DEFAULT_SINGLE_SERIES_COLOR for label in [*labels, "Study period"]}
    return single_or_distinct_colors(labels)


def _static_asr_color_map(
    labels: list[str],
    *,
    include_method_in_label: bool,
) -> dict[str, str]:
    """Return static ASR colors: blue for one series, method colors for comparisons."""
    if not include_method_in_label:
        return {label: _STATIC_ASR_COLOR for label in labels}
    return {label: color for label, color in zip(labels, distinct_colors(len(labels)), strict=True)}


def _component_colors_for_asr_scope(
    *,
    asr_color_map: dict[str, str],
    include_method_in_label: bool,
) -> dict[str, str]:
    return asr_color_map if include_method_in_label else {"aCC": _UNCERTAINTY_ACC_COMPONENT_COLOR}


def _ordered_series_groups(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
    label_order: list[str] | None,
) -> list[tuple[str, pd.DataFrame]]:
    groups = list(_series_groups(frame=frame, include_method_in_label=include_method_in_label))
    if label_order is None:
        return groups
    ranks = {label: index for index, label in enumerate(label_order)}
    return sorted(groups, key=lambda item: (ranks.get(item[0], len(ranks)), item[0]))


def _series_label_order(*, frame: pd.DataFrame, include_method_in_label: bool) -> list[str]:
    values: list[tuple[str, str, float]] = []
    for label, group in _series_groups(
        frame=frame,
        include_method_in_label=include_method_in_label,
    ):
        score = _series_average_score(group)
        impact = str(pd.Series(group.iloc[0], copy=False).get("impact", "")).strip()
        values.append((impact, label, score))
    return order_labels_by_average_within_group_rank(values)


def _series_average_score(group: pd.DataFrame) -> float:
    values = pd.Series(pd.to_numeric(group["mean"], errors="raise")).to_numpy(dtype=float)
    return finite_average(values.tolist()) or float("-inf")


def _row_label(row: pd.Series, *, include_method_in_label: bool) -> str:
    return str(row.get("__method", "")).strip() if include_method_in_label else "ASR"


def _trajectory_color(axis: Any) -> Any:
    return axis.lines[0].get_color() if axis.lines else _DEFAULT_COLOR


def _dynamic_pair_note(frame: pd.DataFrame) -> str:
    values = pd.Series(pd.to_numeric(frame[MODEL_SCENARIO_PAIR_COUNT_COLUMN], errors="raise"))
    label = f"{int(values.max())} AR6 CC model-scenario pairs"
    sampling_method = model_scenario_sampling_method(frame)
    if sampling_method is not None:
        label = f"{label}; sampling method: {sampling_method}"
    return f"{label}."


def _dynamic_uncertainty_note(*, frame: pd.DataFrame, include_distribution_note: bool) -> str:
    lines = [_dynamic_pair_note(frame)]
    if include_distribution_note:
        lines.extend(
            [
                "ASR and aCC vs. LCA pathway lines represent Monte Carlo runs mean values.",
                (
                    "Frequency of no-transgression lines are computed from the "
                    "full uncertainty distribution."
                ),
            ]
        )
    return "\n".join(lines)


def _dynamic_distribution_footer_extra_height(
    *,
    fig: Any,
    handles: list[Any],
    ncol: int,
    note: str,
) -> float:
    labels = [format_scientific_figure_text(str(handle.get_label()).strip()) for handle in handles]
    rows = legend_display_rows(labels, ncol=int(ncol))
    note_lines = len(legend_note_lines(fig, note))
    return (
        _LEGEND_TITLED_OVERHEAD_IN
        + _LEGEND_ROW_HEIGHT_IN * rows
        + _NOTE_LINE_HEIGHT_IN * note_lines
        + _DYNAMIC_FOOTER_STACK_GAP_IN
    )


def _frequency_note(prefix: str) -> str:
    text = (
        "Frequency of no-transgression lines are computed from the full uncertainty distribution."
    )
    return f"{str(prefix).strip()}\n{text}"


def _render_centered_dynamic_distribution_legend(
    *,
    fig: Any,
    handles: list[Any],
    ncol: int,
    note: str,
    extra_height_in: float,
) -> None:
    labels = [format_scientific_figure_text(str(handle.get_label()).strip()) for handle in handles]
    wrapped_note = legend_note_lines(fig, note)
    note_handles = [
        Line2D([0], [0], color="none", linewidth=0.0, label=line) for line in wrapped_note
    ]
    layout = reserve_footer_space(
        fig,
        rows=legend_display_rows(labels, ncol=ncol) + len(wrapped_note),
        note_lines=0,
        title_rows=1,
        extra_height_in=extra_height_in,
    )
    note_legend = fig.legend(
        handles=note_handles,
        labels=wrapped_note,
        loc="lower center",
        bbox_to_anchor=(0.5, layout.anchor_y),
        ncol=1,
        frameon=False,
        fontsize=8,
        handlelength=0.0,
        handletextpad=0.0,
    )
    legend = fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        bbox_to_anchor=(0.5, layout.anchor_y),
        ncol=int(ncol),
        frameon=False,
        fontsize="small",
        title="Uncertainty",
        title_fontsize="small",
        handler_map=violin_summary_legend_handler_map(),
        **violin_summary_legend_kwargs(),
    )
    legend.get_title().set_fontweight("bold")
    center_legend_text(note_legend)
    align_lower_legend_stack_top_to_layout(
        fig,
        [note_legend, legend],
        layout=layout,
    )
