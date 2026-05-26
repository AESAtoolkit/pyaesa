"""Deterministic ASR figure orchestration and rendering."""

from dataclasses import replace
from functools import partial
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from pyaesa.ar6_cc.deterministic.figures.period_panels import (
    plot_budget_panel as plot_ar6_budget_panel,
)
from pyaesa.ar6_cc.deterministic.figures.period_panels import (
    render_study_transition as render_ar6_study_transition,
)
from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NEGATIVE,
    CC_FLOW_NET,
    CC_FLOW_POSITIVE,
)
from pyaesa.ar6_cc.shared.runtime.figure_style import ar6_category_color, ar6_cc_flow_color
from pyaesa.asr.figures.axis import ASR_LOG_SCALE, ASR_NORMAL_SCALE, ASRScaleMode
from pyaesa.asr.figures.common import (
    DYNAMIC_SCOPE_COLUMNS,
    apply_scaled_asr_axis_policy,
    asr_axis_limits,
    asr_scale_mode_for_values,
    asr_scope_title,
    component_axis_limits,
    format_acc_lca_component_axis,
    format_year_axis,
    ordered_impacts,
    scope_slices,
    static_asocc_ssp_slices,
    visible_values,
)
from pyaesa.asr.figures.component_legend import (
    ACC_COMPONENT_COLOR,
    LCA_COMPONENT_COLOR,
    acc_lca_cumulative_title,
    acc_lca_pathway_title,
    ar6_cc_flow_key_entries,
    lca_component_linewidth,
    lca_component_path_effects,
    render_acc_lca_row_key,
    render_ar6_cc_row_key,
)
from pyaesa.asr.figures.dynamic_global_ar6 import (
    DeterministicGlobalAR6Source,
    deterministic_global_ar6_rows_from_source,
    deterministic_global_ar6_source,
    global_ar6_panel_title_pad,
)
from pyaesa.asr.figures.risk_guides import (
    ASR_RISK_LEGEND_GROUP_TITLE,
    asr_risk_scale_footer_extra_height,
    render_asr_risk_scale_footer,
)
from pyaesa.asr.figures.transitions import (
    ASR_TRANSITION_SERIES_EXCLUDED_COLUMNS,
    merged_asr_transition_markers,
)
from pyaesa.asr.shared.runtime.paths import (
    ASRDeterministicPathContext,
    get_asr_figures_dir,
    get_asr_results_dir,
)
from pyaesa.shared.figures.colors import MULTI_METHOD_LINE_ALPHA
from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.deterministic_legends_methods import legend_group_from_row
from pyaesa.shared.figures.deterministic_variant_compressor import (
    MIN_ROLE,
    ROLE_COLUMN,
    VALUE_COLUMN,
    YEAR_COLUMN,
    compress_variants,
)
from pyaesa.shared.figures.deterministic_variant_display import (
    VARIANT_COLUMNS,
    base_variant_groups,
    variant_note as retained_variant_note,
    variant_combo_text,
    variant_display_name,
)
from pyaesa.shared.figures.deterministic_variant_method_note import (
    write_variant_compression_method_note,
)
from pyaesa.shared.figures.dynamic_ar6 import model_scenario_pair_token
from pyaesa.shared.figures.figure_footer import (
    render_below_figure_legend,
    set_footer_min_plot_height,
)
from pyaesa.shared.figures.jobs import PlannedFigureJob, render_figure_jobs
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN,
    SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN,
    TRANSITION_PANEL_TITLE_PAD,
    bottom_panel_indices,
    format_integer_year_axis,
    format_single_year_category_axis,
    multi_impact_panel_figure_size,
    show_panel_x_labels,
    single_impact_figure_size,
)
from pyaesa.shared.figures.lcia_metadata import resolve_frame_impact_title
from pyaesa.shared.figures.multi_year_transitions import (
    TransitionMarker,
    render_transition_markers,
    transition_boundary_x,
    transition_title_pad,
)
from pyaesa.shared.figures.nonnegative_axis import apply_zero_floor_if_nonnegative
from pyaesa.shared.figures.output_stems import dynamic_output_base_stem
from pyaesa.shared.figures.paths import (
    output_file_path,
    strip_lcia_method_suffix,
    top_level_figure_dir,
)
from pyaesa.shared.figures.request_validation import (
    resolve_nested_polar_years,
    validate_consecutive_multi_year_figure_request,
)
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter
from pyaesa.shared.figures.selector_slices import selector_slices
from pyaesa.shared.figures.title_contract import SelectorScopeRequest
from pyaesa.shared.figures.titles import render_dynamic_ar6_title, title_layout_top
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.scenario.columns import AR6_CC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.scalars import sanitize_token

from .component_diagnostics import (
    DeterministicComponentRows,
    _component_group_columns,
    _float_series,
    _integer_series,
    _scope_component_rows,
    load_component_rows_artifact,
)
from .groups import resolve_grouped_figure_inputs
from .row_reader import (
    PreparedAsrFigureGroup,
    all_prepared_asr_rows,
    combined_prepared_asr_rows,
    has_multiple_prepared_asr_groups,
    prepare_asr_figure_groups,
)
from .state import clear_persisted_asr_figures

_LABEL_COLUMN = "__series_label"
_COLOR_COLUMN = "__series_color"
_MAX_THRESHOLD_COLUMN = "__asr_max_threshold"
_LINE_ALPHA = 0.82
_LCA_COLOR = LCA_COMPONENT_COLOR
_PANEL_TITLE_PAD = 5
_GLOBAL_AR6_DENSE_LINE_WIDTH = 1.1
_GLOBAL_AR6_SPARSE_LINE_WIDTH = 1.6
_GLOBAL_AR6_SPARSE_PAIR_COUNT_LIMIT = 10
_TWO_COLUMN_PANEL_HSPACE = 0.32
_TWO_COLUMN_TRANSITION_HSPACE = 0.42
_TWO_COLUMN_COMPONENT_TRANSITION_HSPACE = 0.54
_TWO_COLUMN_PANEL_TOP = DOUBLE_COLUMN_TITLE_TOP
_TWO_COLUMN_TRANSITION_TOP = DOUBLE_COLUMN_TITLE_TOP
_TWO_COLUMN_COMPONENT_TRANSITION_TOP = DOUBLE_COLUMN_TITLE_TOP
_DYNAMIC_ROW_COUNT = 3
_DYNAMIC_MIN_PLOT_HEIGHT_IN = _DYNAMIC_ROW_COUNT * SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN
_DYNAMIC_COMPONENT_SIZE = (15.8, 11.2)
_DYNAMIC_MULTI_METHOD_ROW_HEIGHT_SCALE = 1.3
_STACKED_LABEL_TITLE_PAD = 24
_SINGLE_PANEL_TRANSITION_TITLE_PAD = TRANSITION_PANEL_TITLE_PAD
_SINGLE_PANEL_COMPONENT_TRANSITION_TITLE_PAD = _STACKED_LABEL_TITLE_PAD
_PANEL_COMPONENT_TRANSITION_TITLE_PAD = _STACKED_LABEL_TITLE_PAD
_NON_GLOBAL_POST_STUDY_TITLE_PAD = _STACKED_LABEL_TITLE_PAD


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


def _single_panel_transition_title_pad(markers: list[TransitionMarker]) -> int:
    return transition_title_pad(
        markers,
        no_transition=6,
        single_transition=_SINGLE_PANEL_TRANSITION_TITLE_PAD,
        component_transition=_SINGLE_PANEL_COMPONENT_TRANSITION_TITLE_PAD,
    )


def _dynamic_non_global_title_pad(
    markers: list[TransitionMarker],
    *,
    post_years: list[int],
) -> int:
    return max(
        _single_panel_transition_title_pad(markers),
        _NON_GLOBAL_POST_STUDY_TITLE_PAD if post_years else 0,
    )


def _dynamic_min_plot_height(*, group_legend: bool) -> float:
    scale = _DYNAMIC_MULTI_METHOD_ROW_HEIGHT_SCALE if group_legend else 1.0
    return _DYNAMIC_MIN_PLOT_HEIGHT_IN * scale


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


def render_asr_figures(
    *,
    path_context: ASRDeterministicPathContext,
    fu_code: str,
    cc_source: str,
    cc_type: str,
    requested_years: list[int],
    share_transition_meta: dict[str, dict[str, object]],
    emissions_mode: str | None,
    dpi: int,
    output_format: str,
    selector_scope_request: SelectorScopeRequest | None = None,
    figure_options: dict,
    status: StatusSink | None = None,
    output_paths: list[Path],
    acc_output_files: list[Path],
    component_rows_path: Path | None = None,
    coverage: dict[str, list[Any]] | None = None,
) -> list[Path]:
    """Render deterministic ASR figures from persisted public ASR tables."""
    validate_consecutive_multi_year_figure_request(
        requested_years=requested_years,
        family_label="deterministic ASR",
    )
    polar_years = resolve_nested_polar_years(
        studied_years=requested_years,
        polar=dict(figure_options["polar"]),
        argument_name="figure_options.polar",
    )
    if not (bool(figure_options["per_method"]) or bool(figure_options["multi_method"])):
        clear_persisted_asr_figures(path_context=path_context)
        return []
    figures_root = get_asr_figures_dir(context=path_context)
    clear_persisted_asr_figures(path_context=path_context)
    raw_groups = resolve_grouped_figure_inputs(
        root=get_asr_results_dir(context=path_context),
        output_paths=output_paths,
        field_name="output_paths",
        share_transition_meta=share_transition_meta,
        family_label="ASR",
    )
    groups = prepare_asr_figure_groups(
        groups=raw_groups,
        requested_years=requested_years,
        fu_code=fu_code,
    )
    for column, values in (coverage or {}).items():
        if column == "years":
            continue
        allowed = [str(value) for value in values]
        groups = [
            replace(
                group,
                rows=group.rows.loc[group.rows[column].astype(str).isin(allowed)].copy(),
            )
            if column in group.rows.columns
            else group
            for group in groups
        ]
    scale_modes = _scale_modes_by_lcia(
        rows=combined_prepared_asr_rows(groups),
    )
    component_rows = None
    if cc_type != "static":
        component_rows = _dynamic_component_rows(
            acc_output_files=acc_output_files,
            component_rows_path=cast(Path, component_rows_path),
        )
    global_ar6_source = (
        deterministic_global_ar6_source(acc_output_files=acc_output_files)
        if cc_type != "static" and _single_year(requested_years) is None
        else None
    )

    def jobs():
        """Yield deterministic ASR figure jobs one selector scope at a time."""
        if bool(figure_options["per_method"]):
            yield from _per_method_jobs(
                groups=groups,
                components=component_rows,
                global_ar6_source=global_ar6_source,
                figures_root=figures_root,
                requested_years=requested_years,
                cc_type=cc_type,
                cc_source=cc_source,
                emissions_mode=emissions_mode,
                dpi=dpi,
                output_format=output_format,
                selector_scope_request=selector_scope_request,
                scale_modes=scale_modes,
                polar_years=polar_years,
            )
        if bool(figure_options["multi_method"]) and has_multiple_prepared_asr_groups(groups):
            yield from _multi_method_jobs(
                groups=groups,
                components=component_rows,
                global_ar6_source=global_ar6_source,
                figures_root=figures_root,
                requested_years=requested_years,
                cc_type=cc_type,
                cc_source=cc_source,
                emissions_mode=emissions_mode,
                dpi=dpi,
                output_format=output_format,
                selector_scope_request=selector_scope_request,
                scale_modes=scale_modes,
            )

    paths = render_figure_jobs(source="deterministic_asr", jobs=jobs, status=status)
    rows = all_prepared_asr_rows(groups)
    write_variant_compression_method_note(figures_root=figures_root, rows=rows)
    return paths


def _dynamic_component_rows(
    *,
    acc_output_files: list[Path],
    component_rows_path: Path,
) -> DeterministicComponentRows:
    return load_component_rows_artifact(
        path=component_rows_path,
        acc_output_files=acc_output_files,
    )


def _per_method_jobs(
    *,
    groups: list[PreparedAsrFigureGroup],
    components: DeterministicComponentRows | None,
    global_ar6_source: DeterministicGlobalAR6Source | None,
    figures_root: Path,
    requested_years: list[int],
    cc_type: str,
    cc_source: str,
    emissions_mode: str | None,
    dpi: int,
    output_format: str,
    selector_scope_request: SelectorScopeRequest | None,
    scale_modes: dict[str, ASRScaleMode],
    polar_years: list[int],
) -> Iterator[PlannedFigureJob]:
    single_year = _single_year(requested_years)
    for group in groups:
        for branch_rows in _dynamic_branch_slices(rows=group.rows, cc_type=cc_type):
            for selector_token, selector_title, selector_rows in selector_slices(
                branch_rows,
                selector_scope_request=selector_scope_request,
            ):
                prepared = _prepare_plot_rows(selector_rows)
                impacts = ordered_impacts(prepared)
                polar_allowed = cc_type == "static" and bool(polar_years)
                scale_mode = _scale_mode_for_frame(prepared, scale_modes=scale_modes)
                output_base = top_level_figure_dir(
                    figures_root=figures_root, folder="per_method"
                ) / _scope_stem(
                    label=group.base_stem,
                    frame=prepared,
                    requested_years=requested_years,
                    cc_source=cc_source,
                    cc_type=cc_type,
                    selector_token=selector_token,
                )
                title = asr_scope_title(
                    "ASR",
                    group.title_label,
                    prepared,
                    include_impact=len(impacts) == 1,
                    studied_year=single_year,
                    selector_title=selector_title,
                )
                yield PlannedFigureJob(
                    kind="per_method",
                    label=output_base.name,
                    planned_outputs=_planned_scope_output_count(
                        frame=prepared,
                        requested_years=requested_years,
                        dynamic=cc_type != "static",
                    ),
                    render=partial(
                        _plot_scope,
                        frame=prepared,
                        requested_years=requested_years,
                        output_stem=output_base,
                        title=title,
                        dpi=dpi,
                        output_format=output_format,
                        group_legend=False,
                        include_method_in_label=False,
                        allow_polar=polar_allowed,
                        dynamic=cc_type != "static",
                        emissions_mode=emissions_mode,
                        scale_mode=scale_mode,
                        components=components,
                        global_ar6_source=global_ar6_source,
                    ),
                )
                if polar_allowed and single_year is None and len(impacts) > 1:
                    years = pd.Series(pd.to_numeric(prepared[YEAR_COLUMN], errors="raise")).astype(
                        int
                    )
                    for year in polar_years:
                        year_prepared = prepared.loc[years.eq(int(year))].copy()
                        year_output_base = top_level_figure_dir(
                            figures_root=figures_root, folder="per_method"
                        ) / _scope_stem(
                            label=group.base_stem,
                            frame=year_prepared,
                            requested_years=[int(year)],
                            cc_source=cc_source,
                            cc_type=cc_type,
                            selector_token=selector_token,
                        )
                        year_title = asr_scope_title(
                            "ASR",
                            group.title_label,
                            year_prepared,
                            include_impact=False,
                            studied_year=int(year),
                            selector_title=selector_title,
                        )
                        yield PlannedFigureJob(
                            kind="polar_deterministic",
                            label=year_output_base.name,
                            render=partial(
                                _plot_scope,
                                frame=year_prepared,
                                requested_years=[int(year)],
                                output_stem=year_output_base,
                                title=year_title,
                                dpi=dpi,
                                output_format=output_format,
                                group_legend=False,
                                include_method_in_label=False,
                                allow_polar=True,
                                dynamic=False,
                                emissions_mode=emissions_mode,
                                scale_mode=scale_mode,
                                components=components,
                                global_ar6_source=global_ar6_source,
                            ),
                        )


def _multi_method_jobs(
    *,
    groups: list[PreparedAsrFigureGroup],
    components: DeterministicComponentRows | None,
    global_ar6_source: DeterministicGlobalAR6Source | None,
    figures_root: Path,
    requested_years: list[int],
    cc_type: str,
    cc_source: str,
    emissions_mode: str | None,
    dpi: int,
    output_format: str,
    selector_scope_request: SelectorScopeRequest | None,
    scale_modes: dict[str, ASRScaleMode],
) -> Iterator[PlannedFigureJob]:
    single_year = _single_year(requested_years)
    for branch_rows in _dynamic_branch_slices(
        rows=combined_prepared_asr_rows(groups),
        cc_type=cc_type,
    ):
        for selector_token, selector_title, selector_rows in selector_slices(
            branch_rows,
            selector_scope_request=selector_scope_request,
        ):
            prepared = _prepare_plot_rows(selector_rows)
            scale_mode = _scale_mode_for_frame(prepared, scale_modes=scale_modes)
            scopes = (
                [prepared]
                if single_year is not None
                else [
                    prepared.loc[prepared["impact"].astype(str).eq(impact)].copy()
                    for impact in ordered_impacts(prepared)
                ]
            )
            common_limits = (
                None
                if single_year is not None
                else _common_asr_limits(prepared, scale_mode=scale_mode)
            )
            for scope in scopes:
                include_impact = single_year is None and len(ordered_impacts(prepared)) > 1
                output_base = top_level_figure_dir(
                    figures_root=figures_root, folder="multi_method"
                ) / _scope_stem(
                    label="multi_method",
                    frame=scope,
                    requested_years=requested_years,
                    cc_source=cc_source,
                    cc_type=cc_type,
                    include_impact=include_impact,
                    selector_token=selector_token,
                )
                title = asr_scope_title(
                    "ASR",
                    None,
                    scope,
                    include_impact=include_impact or len(ordered_impacts(scope)) == 1,
                    studied_year=single_year,
                    selector_title=selector_title,
                )
                yield PlannedFigureJob(
                    kind="multi_method",
                    label=output_base.name,
                    planned_outputs=_planned_scope_output_count(
                        frame=scope,
                        requested_years=requested_years,
                        dynamic=cc_type != "static",
                    ),
                    render=partial(
                        _plot_scope,
                        frame=scope,
                        requested_years=requested_years,
                        output_stem=output_base,
                        title=title,
                        dpi=dpi,
                        output_format=output_format,
                        group_legend=True,
                        include_method_in_label=True,
                        allow_polar=False,
                        dynamic=cc_type != "static",
                        emissions_mode=emissions_mode,
                        scale_mode=scale_mode,
                        limits=common_limits,
                        components=components,
                        global_ar6_source=global_ar6_source,
                    ),
                )


def _dynamic_branch_slices(*, rows: pd.DataFrame, cc_type: str) -> Iterator[pd.DataFrame]:
    if str(cc_type) == "static":
        for branch_scope in scope_slices(rows, ("lcia_method",)):
            yield from static_asocc_ssp_slices(branch_scope)
        return
    yield from scope_slices(rows, ("lcia_method", *DYNAMIC_SCOPE_COLUMNS))


def _planned_scope_output_count(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    dynamic: bool,
) -> int:
    """Return the number of figure files rendered by one deterministic ASR job."""
    single_year = _single_year(requested_years)
    return 2 if dynamic and single_year is None and len(ordered_impacts(frame)) == 1 else 1


def _prepare_plot_rows(rows: pd.DataFrame) -> pd.DataFrame:
    thresholded = _with_static_asr_threshold(rows)
    compressed = _min_variant_rows(compress_variants(thresholded))
    out = compressed.copy()
    out[_LABEL_COLUMN] = (
        pd.Series(out["__method"], copy=False).astype("string").str.strip()
        if "__method" in out.columns
        else ""
    )
    labels = list(dict.fromkeys(str(label) for label in out[_LABEL_COLUMN].tolist()))
    colors = _colors(labels)
    out[_COLOR_COLUMN] = [colors[str(label)] for label in out[_LABEL_COLUMN].tolist()]
    return out


def _min_variant_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if ROLE_COLUMN not in frame.columns:
        return frame.copy()
    roles = pd.Series(frame[ROLE_COLUMN], copy=False)
    mask = roles.isna() | roles.astype("string").str.strip().eq(MIN_ROLE)
    return frame.loc[mask].copy()


def _with_static_asr_threshold(rows: pd.DataFrame) -> pd.DataFrame:
    if "cc_bound" not in rows.columns:
        out = rows.copy()
        out[_MAX_THRESHOLD_COLUMN] = np.nan
        return out
    bounds = {str(value).strip() for value in rows["cc_bound"].dropna().astype(str)}
    if {"min_cc", "max_cc"}.issubset(bounds):
        min_rows = rows.loc[rows["cc_bound"].astype(str).eq("min_cc")].copy()
        max_rows = rows.loc[rows["cc_bound"].astype(str).eq("max_cc")].copy()
        key_columns = [
            column
            for column in min_rows.columns
            if column not in {"cc_bound", VALUE_COLUMN} and column in max_rows.columns
        ]
        paired = min_rows.merge(
            max_rows[[*key_columns, VALUE_COLUMN]].rename(
                columns={VALUE_COLUMN: "__max_bound_asr_value"}
            ),
            on=key_columns,
            how="left",
        )
        paired[_MAX_THRESHOLD_COLUMN] = np.divide(
            paired[VALUE_COLUMN].to_numpy(dtype=np.float64),
            paired["__max_bound_asr_value"].to_numpy(dtype=np.float64),
            out=np.full(len(paired), np.nan, dtype=np.float64),
            where=paired["__max_bound_asr_value"].to_numpy(dtype=np.float64) > 0.0,
        )
        paired["cc_bound"] = "both"
        return paired.drop(columns=["__max_bound_asr_value"])
    out = rows.loc[~rows["cc_bound"].astype(str).eq("max_cc")].copy()
    out[_MAX_THRESHOLD_COLUMN] = np.nan
    return out


def _plot_scope(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
    allow_polar: bool,
    dynamic: bool,
    emissions_mode: str | None,
    scale_mode: ASRScaleMode,
    limits: tuple[float, float] | None = None,
    components: DeterministicComponentRows | None = None,
    global_ar6_source: DeterministicGlobalAR6Source | None = None,
) -> list[Path]:
    single_year = _single_year(requested_years)
    impacts = ordered_impacts(frame)
    if dynamic and single_year is None and len(impacts) == 1:
        return _plot_dynamic_scope(
            frame=frame,
            requested_years=requested_years,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            emissions_mode=emissions_mode,
            scale_mode=scale_mode,
            limits=limits,
            components=cast(DeterministicComponentRows, components),
            global_ar6_source=cast(DeterministicGlobalAR6Source, global_ar6_source),
        )
    if allow_polar and single_year is not None and len(impacts) > 1:
        from pyaesa.asr.figures.polar import render_asr_polar

        values = {impact: _deterministic_polar_values(frame, impact=impact) for impact in impacts}
        return render_asr_polar(
            frame=frame,
            values=values,
            frequencies=None,
            output_stem=output_stem.parent / f"polar_{output_stem.name}",
            title=title,
            lcia_method=visible_values(frame, "lcia_method")[0],
            style="deterministic",
            scale_mode=scale_mode,
            dpi=dpi,
            output_format=output_format,
            deterministic_note=_min_variant_note(frame),
        )
    if len(impacts) > 1:
        return _plot_impact_panels(
            frame=frame,
            requested_years=requested_years,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            dynamic=dynamic,
            scale_mode=scale_mode,
            limits=limits,
        )
    fig, axis = plt.subplots(figsize=single_impact_figure_size(single_year=single_year is not None))
    if single_year is None:
        set_footer_min_plot_height(
            fig,
            height_in=_single_panel_min_plot_height(
                group_legend=group_legend,
                include_method_in_label=include_method_in_label,
            ),
        )
    else:
        set_footer_min_plot_height(fig, height_in=SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN)
    values, years, markers = _render_axis(
        axis=axis,
        frame=frame,
        requested_years=requested_years,
        group_legend=group_legend,
        include_method_in_label=include_method_in_label,
        show_x_labels=True,
    )
    _format_asr_axis(
        axis,
        values=values,
        frame=frame,
        years=years,
        grouped_legend=group_legend,
        limits=limits,
        scale_mode=scale_mode,
    )
    axis.set_title(
        title,
        fontweight="bold",
        pad=_single_panel_transition_title_pad(markers),
    )
    _render_footer(fig=fig, axis=axis, frame=frame, group_legend=group_legend)
    output_path = output_file_path(base_path=output_stem, output_format=output_format)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight", format=output_format)
    plt.close(fig)
    return [output_path]


def _plot_dynamic_scope(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
    emissions_mode: str | None,
    scale_mode: ASRScaleMode,
    limits: tuple[float, float] | None = None,
    components: DeterministicComponentRows,
    global_ar6_source: DeterministicGlobalAR6Source,
) -> list[Path]:
    nrows = 3
    component_acc, component_lca = _component_scope(
        components=components,
        frame=frame,
        include_method_axis=include_method_in_label,
    )
    target_unit = visible_values(component_lca, "impact_unit")[0]
    global_ar6 = deterministic_global_ar6_rows_from_source(
        source=global_ar6_source,
        asr_frame=frame,
        requested_years=requested_years,
        target_unit=target_unit,
    )
    acc_color_map = _component_colors_for_asr_scope(
        frame=frame,
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
            nrows=nrows,
            ncols=2,
            figsize=_DYNAMIC_COMPONENT_SIZE,
            squeeze=False,
            gridspec_kw={
                "width_ratios": [1.0, 1.0] if group_legend else [3.0, 1.22],
                "height_ratios": [1.0, 1.0, 1.0],
                "hspace": 0.44,
                "wspace": 0.28,
            },
        )
        set_footer_min_plot_height(
            fig,
            height_in=_dynamic_min_plot_height(group_legend=group_legend),
        )
        render_dynamic_ar6_title(fig, title)
        global_title_pad = global_ar6_panel_title_pad(post_years)
        flow_colors = _ar6_flow_colors()
        category_colors = {
            category: ar6_category_color(category=category)
            for category in visible_values(global_ar6.frame, "cc_category")
        }
        visible_negative_flow = _render_dynamic_global_ar6_row(
            axis=axes[0, 0],
            budget_axis=axes[0, 1],
            scoped_frame=global_ar6.frame,
            study_years=global_ar6.study_years,
            post_years=post_years,
            category_colors=category_colors,
            flow_colors=flow_colors,
            pathway_title="Global AR6 CC pathways",
            budget_title="Cumulative global AR6 CC budget",
            show_x_labels=False,
            show_study_label=True,
            title_pad=global_title_pad,
        )
        component_values, component_years = _render_component_comparison_axis(
            axis=axes[1, 0],
            acc_rows=component_acc,
            lca_rows=component_lca,
            include_method_in_label=include_method_in_label,
            group_legend=group_legend,
            color_map=acc_color_map,
        )
        component_markers = _visible_series_transition_markers(
            frame,
            include_method_in_label=include_method_in_label,
        )
        component_cumulative_values = _render_component_cumulative_axis(
            axis=axes[1, 1],
            acc_rows=component_acc,
            lca_rows=component_lca,
            include_method_in_label=include_method_in_label,
            group_legend=group_legend,
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
            component_markers,
            post_years=post_years,
        )
        fig.subplots_adjust(
            top=title_layout_top(
                fig,
                title,
                default_top=0.91,
                panel_title_pad=max(global_title_pad, component_title_pad),
            )
        )
        format_acc_lca_component_axis(
            axis=axes[1, 0],
            frame=component_lca,
            years=component_years,
            show_x_labels=False,
            title=acc_lca_pathway_title(emissions_mode=emissions_mode),
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
            title=acc_lca_cumulative_title(emissions_mode=emissions_mode),
            limits=component_cumulative_limits,
            title_pad=component_title_pad,
            scale_mode=scale_mode,
        )
        if include_post:
            _extend_post_study_pathway_axis(
                axis=axes[1, 0],
                study_years=requested_years,
                post_years=post_years,
                show_x_labels=False,
            )
        row_index = 2
        axis = axes[row_index, 0]
        cumulative_axis = axes[row_index, 1]
        values, years, markers = _render_axis(
            axis=axis,
            frame=frame,
            requested_years=requested_years,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            show_x_labels=True,
            transition_shade_right=transition_shade_right,
        )
        cumulative_values = _render_cumulative_asr_axis(
            axis=cumulative_axis,
            frame=frame,
            include_method_in_label=include_method_in_label,
            group_legend=group_legend,
            show_x_labels=True,
            label_order=cumulative_label_order,
        )
        row_limits = limits or _asr_row_limits(
            frame,
            values,
            cumulative_values,
            scale_mode=scale_mode,
        )
        _format_asr_axis(
            axis,
            values=values,
            frame=frame,
            years=years,
            grouped_legend=group_legend,
            limits=row_limits,
            scale_mode=scale_mode,
        )
        if include_post:
            _extend_post_study_pathway_axis(
                axis=axis,
                study_years=requested_years,
                post_years=post_years,
                show_x_labels=True,
            )
        asr_title_pad = _dynamic_non_global_title_pad(markers, post_years=post_years)
        axis.set_title("ASR pathways", fontweight="bold", pad=asr_title_pad)
        _format_asr_axis(
            cumulative_axis,
            values=cumulative_values,
            frame=frame,
            years=[int(requested_years[-1])],
            grouped_legend=group_legend,
            limits=row_limits,
            scale_mode=scale_mode,
        )
        cumulative_axis.set_title("Cumulative ASR", fontweight="bold", pad=asr_title_pad)
        _render_footer(fig=fig, axis=axis, frame=frame, group_legend=group_legend)
        render_ar6_cc_row_key(
            fig=fig,
            left_axis=axes[0, 0],
            right_axis=axes[0, 1],
            entries=ar6_cc_flow_key_entries(
                emissions_mode=emissions_mode,
                positive_color=flow_colors[CC_FLOW_POSITIVE],
                negative_color=flow_colors[CC_FLOW_NEGATIVE],
                visible_negative_flow=visible_negative_flow,
                negative_style="-",
            ),
        )
        render_acc_lca_row_key(
            fig=fig,
            left_axis=axes[1, 0],
            right_axis=axes[1, 1],
            include_method_in_label=include_method_in_label,
            emissions_mode=emissions_mode,
        )
        output_path = output_file_path(
            base_path=output_stem.with_name(f"{output_stem.name}{suffix}"),
            output_format=output_format,
        )
        fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight", format=output_format)
        plt.close(fig)
        paths.append(output_path)
    return paths


def _single_panel_min_plot_height(
    *,
    group_legend: bool,
    include_method_in_label: bool,
) -> float:
    return (
        MIDDLE_MULTI_METHOD_PLOT_HEIGHT_IN
        if group_legend and include_method_in_label
        else SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN
    )


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


def _render_dynamic_global_ar6_row(
    *,
    axis: Any,
    budget_axis: Any,
    scoped_frame: pd.DataFrame,
    study_years: list[int],
    post_years: list[int],
    category_colors: dict[str, str],
    flow_colors: dict[str, str],
    pathway_title: str,
    budget_title: str,
    show_x_labels: bool,
    show_study_label: bool,
    title_pad: int,
) -> bool:
    """Render the ASR dynamic Global AR6 row with flow color semantics."""
    years = [*study_years, *post_years]
    minimum_value: float | None = None
    visible_negative_flow = False
    categories = sorted({str(value) for value in scoped_frame["cc_category"].astype(str)})
    for category in categories:
        category_frame = scoped_frame.loc[
            scoped_frame["cc_category"].astype(str) == category
        ].copy()
        grouped = category_frame.groupby(
            ["cc_flow", "cc_model", "cc_scenario"],
            dropna=False,
            sort=True,
        )
        pair_count = _global_ar6_pair_count(category_frame)
        line_width = (
            _GLOBAL_AR6_DENSE_LINE_WIDTH
            if pair_count > _GLOBAL_AR6_SPARSE_PAIR_COUNT_LIMIT
            else _GLOBAL_AR6_SPARSE_LINE_WIDTH
        )
        line_alpha = 0.35 if pair_count > _GLOBAL_AR6_SPARSE_PAIR_COUNT_LIMIT else 1.0
        for group_key, group in grouped:
            cc_flow = str(group_key[0])
            row = group.iloc[0]
            plot_years = years
            values = row.loc[years].to_numpy(dtype=float).tolist()
            if cc_flow == CC_FLOW_NEGATIVE and not any(value < 0.0 for value in values):
                continue
            if cc_flow == CC_FLOW_NEGATIVE:
                visible_negative_flow = True
                first_negative_index = next(
                    index for index, value in enumerate(values) if value < 0.0
                )
                plot_years = years[first_negative_index:]
                values = values[first_negative_index:]
            axis.plot(
                plot_years,
                values,
                alpha=line_alpha,
                linewidth=line_width,
                color=flow_colors.get(cc_flow, category_colors[str(category)]),
                linestyle="-",
            )
            local_min = min(values)
            minimum_value = local_min if minimum_value is None else min(minimum_value, local_min)
    axis.set_title(pathway_title, fontweight="bold", pad=int(title_pad))
    axis.set_xlabel("")
    axis.set_ylabel(format_scientific_figure_text(str(scoped_frame["impact_unit"].iloc[0])))
    axis.yaxis.set_major_formatter(FuncFormatter(scientific_tick_formatter))
    axis.set_xlim(float(min(years)) - 0.5, float(max(years)) + 0.5)
    format_integer_year_axis(axis, years=years)
    if not show_x_labels:
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    axis.grid(alpha=0.25)
    apply_zero_floor_if_nonnegative(axis=axis, minimum_value=minimum_value)
    if post_years:
        render_ar6_study_transition(
            axis,
            study_years=study_years,
            post_years=post_years,
            show_study_label=show_study_label,
        )
    plot_ar6_budget_panel(
        axis=budget_axis,
        frame=scoped_frame,
        study_years=study_years,
        post_years=post_years,
        category_colors=category_colors,
        flow_colors=flow_colors,
        title=budget_title,
        title_pad=title_pad,
        negative_sequestration_style="plain",
    )
    return visible_negative_flow


def _global_ar6_pair_count(frame: pd.DataFrame) -> int:
    return int(
        len(
            frame.loc[:, ["cc_model", "cc_scenario"]].astype(str).drop_duplicates(ignore_index=True)
        )
    )


def _ar6_flow_colors() -> dict[str, str]:
    return {
        CC_FLOW_NET: ar6_cc_flow_color(CC_FLOW_NET),
        CC_FLOW_POSITIVE: ar6_cc_flow_color(CC_FLOW_POSITIVE),
        CC_FLOW_NEGATIVE: ar6_cc_flow_color(CC_FLOW_NEGATIVE),
    }


def _component_scope(
    *,
    components: DeterministicComponentRows,
    frame: pd.DataFrame,
    include_method_axis: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del include_method_axis
    acc_rows = _scope_component_rows(
        components.acc,
        asr_frame=frame,
        include_method_axis=True,
    )
    lca_rows = _scope_component_rows(
        components.lca,
        asr_frame=frame,
        include_method_axis=False,
    )
    return acc_rows, lca_rows


def _render_component_comparison_axis(
    *,
    axis: Any,
    acc_rows: pd.DataFrame,
    lca_rows: pd.DataFrame,
    include_method_in_label: bool,
    group_legend: bool,
    color_map: dict[str, str],
) -> tuple[np.ndarray, list[int]]:
    values: list[float] = []
    years: list[int] = []
    seen: set[str] = set()
    acc_colors = color_map
    for _key, group in acc_rows.groupby(
        _component_group_columns(acc_rows),
        dropna=False,
        sort=True,
    ):
        ordered = group.sort_values("year", kind="stable")
        row = pd.Series(ordered.iloc[0], copy=False)
        label = _component_acc_label(row, include_method=include_method_in_label)
        visible_label = (
            label
            if group_legend and include_method_in_label and label and label not in seen
            else "_nolegend_"
        )
        year_values = _integer_series(ordered, "year")
        numeric = _float_series(ordered, "__component_value")
        line = axis.plot(
            year_values.to_numpy(dtype=int),
            numeric.to_numpy(dtype=float),
            color=acc_colors[label],
            linestyle="-",
            linewidth=1.7,
            alpha=MULTI_METHOD_LINE_ALPHA if group_legend else _LINE_ALPHA,
            label=visible_label,
            zorder=4,
        )[0]
        if group_legend and visible_label != "_nolegend_":
            bind_deterministic_legend_group(line, legend_group_from_row(row))
        seen.add(label)
        years.extend(int(year) for year in year_values.tolist())
        values.extend(float(value) for value in numeric.tolist())
    for _key, group in lca_rows.groupby(
        _component_group_columns(lca_rows),
        dropna=False,
        sort=True,
    ):
        ordered = group.sort_values("year", kind="stable")
        year_values = _integer_series(ordered, "year")
        numeric = _float_series(ordered, "__component_value")
        line = axis.plot(
            year_values.to_numpy(dtype=int),
            numeric.to_numpy(dtype=float),
            color=_LCA_COLOR,
            linestyle="-",
            linewidth=lca_component_linewidth(include_method_in_label=include_method_in_label),
            alpha=1.0,
            label="_nolegend_",
            zorder=5,
        )[0]
        effects = lca_component_path_effects(include_method_in_label=include_method_in_label)
        line.set_path_effects(effects)
        seen.add("LCA")
        years.extend(int(year) for year in year_values.tolist())
        values.extend(float(value) for value in numeric.tolist())
    return np.asarray(values, dtype=np.float64), years


def _render_component_cumulative_axis(
    *,
    axis: Any,
    acc_rows: pd.DataFrame,
    lca_rows: pd.DataFrame,
    include_method_in_label: bool,
    group_legend: bool,
    color_map: dict[str, str],
    label_order: list[str] | None = None,
) -> np.ndarray:
    acc_entries = _component_cumulative_entries(
        acc_rows,
        include_method_in_label=include_method_in_label,
        label_order=label_order,
    )
    labels = [label for label, _row, _low, _high in acc_entries]
    positions = np.arange(len(labels), dtype=float)
    acc_colors = color_map
    values: list[float] = []
    seen: set[str] = set()
    range_positions: list[float] = []
    range_lows: list[float] = []
    range_highs: list[float] = []
    range_colors: list[str] = []
    for index, (label, row, low, high) in enumerate(acc_entries):
        visible_label = (
            label
            if group_legend and include_method_in_label and label and label not in seen
            else "_nolegend_"
        )
        handle = axis.bar(
            [positions[index]],
            [low],
            width=0.72,
            color=acc_colors[label],
            alpha=0.54,
            label=visible_label,
            zorder=3,
        )
        if group_legend and visible_label != "_nolegend_":
            bind_deterministic_legend_group(handle, legend_group_from_row(row))
            bind_deterministic_legend_group(handle[0], legend_group_from_row(row))
        range_positions.append(float(positions[index]))
        range_lows.append(float(low))
        range_highs.append(float(high))
        range_colors.append(acc_colors[label])
        seen.add(label)
        values.extend([low, high])
    range_mask = np.greater(
        np.asarray(range_highs, dtype=float),
        np.asarray(range_lows, dtype=float),
    )
    range_positions_array = np.asarray(range_positions, dtype=float)[range_mask]
    range_lows_array = np.asarray(range_lows, dtype=float)[range_mask]
    range_highs_array = np.asarray(range_highs, dtype=float)[range_mask]
    range_colors_visible = [
        color for color, keep in zip(range_colors, range_mask.tolist(), strict=True) if keep
    ]
    axis.vlines(
        range_positions_array,
        range_lows_array,
        range_highs_array,
        colors=range_colors_visible,
        linestyles=":",
        linewidth=1.8,
        zorder=5,
    )
    axis.hlines(
        range_highs_array,
        range_positions_array - 0.36,
        range_positions_array + 0.36,
        colors=range_colors_visible,
        linestyles=":",
        linewidth=1.8,
        zorder=5,
    )
    lca_value = _component_total(lca_rows)
    lca_positions = positions if len(positions) else np.asarray([0.0], dtype=float)
    axis.bar(
        lca_positions,
        np.full(len(lca_positions), lca_value, dtype=np.float64),
        width=0.34,
        color=_LCA_COLOR,
        alpha=0.62,
        label="_nolegend_",
        zorder=4,
    )
    values.append(lca_value)
    axis.set_xlim(-0.5, max(0.5, float(len(labels)) - 0.5))
    axis.set_xticks(positions)
    axis.set_xticklabels([])
    axis.tick_params(axis="x", length=0)
    return np.asarray(values, dtype=np.float64)


def _component_cumulative_entries(
    frame: pd.DataFrame,
    *,
    include_method_in_label: bool,
    label_order: list[str] | None = None,
) -> list[tuple[str, pd.Series, float, float]]:
    entries = []
    group_columns = [
        column
        for column in _component_group_columns(frame)
        if column not in {ROLE_COLUMN, *VARIANT_COLUMNS}
    ]
    for _key, group in frame.groupby(group_columns, dropna=False, sort=True):
        row = pd.Series(group.iloc[0], copy=False)
        label = _component_acc_label(row, include_method=include_method_in_label)
        value = _component_total(group)
        entries.append(
            (
                label,
                row,
                value,
                value,
            )
        )
    return _ordered_labeled_entries(entries, label_order=label_order)


def _component_total(frame: pd.DataFrame) -> float:
    return float(_float_series(frame, "__component_value").sum())


def _component_acc_label(row: pd.Series, *, include_method: bool) -> str:
    return str(row.get("__method", "")).strip() if include_method else "aCC"


def _component_colors_for_asr_scope(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> dict[str, str]:
    if not include_method_in_label:
        return {"aCC": ACC_COMPONENT_COLOR}
    pairs = frame.loc[:, [_LABEL_COLUMN, _COLOR_COLUMN]].drop_duplicates()
    return {
        str(label): str(color)
        for label, color in pairs.itertuples(index=False, name=None)
        if str(label).strip()
    }


def _render_cumulative_asr_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    include_method_in_label: bool,
    group_legend: bool,
    show_x_labels: bool = True,
    label_order: list[str] | None = None,
) -> np.ndarray:
    entries = []
    for group in base_variant_groups(frame):
        row = pd.Series(group.iloc[0], copy=False)
        label = _row_label(row, include_method=include_method_in_label)
        value = _cumulative_asr_value(group)
        entries.append(
            (
                label,
                row,
                value,
                value,
            )
        )
    entries = _ordered_labeled_entries(entries, label_order=label_order)
    labels = list(dict.fromkeys(label for label, _row, _low, _high in entries))
    positions = {label: index for index, label in enumerate(labels)}
    values: list[float] = []
    seen: set[str] = set()
    for label, row, low, high in entries:
        position = float(positions[label])
        color = str(row[_COLOR_COLUMN])
        visible_label = (
            label
            if group_legend and include_method_in_label and label not in seen
            else "_nolegend_"
        )
        handle = axis.bar(
            [position],
            [low],
            width=0.72,
            color=color,
            alpha=_LINE_ALPHA,
            label=visible_label,
            zorder=3,
        )
        if group_legend and visible_label != "_nolegend_":
            bind_deterministic_legend_group(handle, legend_group_from_row(row))
            bind_deterministic_legend_group(handle[0], legend_group_from_row(row))
        seen.add(label)
        values.extend([low, high])
    ticks = np.arange(len(labels), dtype=float)
    axis.set_xlim(-0.5, max(0.5, float(len(labels)) - 0.5))
    if show_x_labels and include_method_in_label:
        axis.set_xticks(ticks)
        axis.set_xticklabels(labels, rotation=45, ha="right")
    else:
        axis.set_xticks(ticks)
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    return np.asarray(values, dtype=np.float64)


def _dynamic_cumulative_label_order(
    *,
    frame: pd.DataFrame,
    include_method_in_label: bool,
) -> list[str] | None:
    if not include_method_in_label:
        return None
    scores: list[tuple[str, float]] = []
    for group in base_variant_groups(frame):
        row = pd.Series(group.iloc[0], copy=False)
        label = _row_label(row, include_method=True)
        scores.append((label, _cumulative_asr_value(group)))
    return [
        label
        for label, _score in sorted(
            scores,
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _ordered_labeled_entries(
    entries: list[tuple[str, pd.Series, float, float]],
    *,
    label_order: list[str] | None,
) -> list[tuple[str, pd.Series, float, float]]:
    if label_order is None:
        return entries
    ranks = {label: index for index, label in enumerate(label_order)}
    return sorted(entries, key=lambda item: (ranks.get(item[0], len(ranks)), item[0]))


def _plot_impact_panels(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_method_in_label: bool,
    dynamic: bool,
    scale_mode: ASRScaleMode,
    limits: tuple[float, float] | None = None,
) -> list[Path]:
    impacts = ordered_impacts(frame)
    single_year = _single_year(requested_years) is not None
    ncols = 2
    nrows = (len(impacts) + 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=multi_impact_panel_figure_size(
            nrows=nrows,
            compact=single_year and not group_legend,
        ),
        squeeze=False,
    )
    bottom_indices = bottom_panel_indices(panel_count=len(impacts), ncols=ncols)
    first_axis = axes[0, 0]
    label_order = _single_year_label_order(frame) if single_year else None
    common_limits = (
        limits if limits is not None else _common_asr_limits(frame, scale_mode=scale_mode)
    )
    has_transitions = False
    has_component_transitions = False
    for index, impact in enumerate(impacts):
        axis = axes[index // ncols, index % ncols]
        panel = frame.loc[frame["impact"].astype(str).eq(impact)].copy()
        values, years, markers = _render_axis(
            axis=axis,
            frame=panel,
            requested_years=requested_years,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            show_x_labels=show_panel_x_labels(
                panel_index=index,
                bottom_indices=bottom_indices,
            ),
            label_order=label_order,
        )
        _format_asr_axis(
            axis,
            values=values,
            frame=panel,
            years=years,
            limits=common_limits,
            grouped_legend=group_legend,
            scale_mode=scale_mode,
        )
        if markers:
            has_transitions = True
            has_component_transitions = (
                has_component_transitions or _uses_component_transition_layout(markers)
            )
        axis.tick_params(axis="y", labelleft=True)
        axis.set_title(
            _impact_title(panel, impact),
            loc="left",
            pad=_panel_transition_title_pad(markers),
        )
    for index in range(len(impacts), nrows * ncols):
        axes[index // ncols, index % ncols].axis("off")
    del dynamic
    render_dynamic_ar6_title(fig, title)
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
    _render_footer(fig=fig, axis=first_axis, frame=frame, group_legend=group_legend)
    output_path = output_file_path(base_path=output_stem, output_format=output_format)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight", format=output_format)
    plt.close(fig)
    return [output_path]


def _render_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    requested_years: list[int],
    group_legend: bool,
    include_method_in_label: bool,
    show_x_labels: bool,
    label_order: list[str] | None = None,
    transition_shade_right: float | None = None,
) -> tuple[np.ndarray, list[int], list[TransitionMarker]]:
    if _single_year(requested_years) is not None:
        values = _render_single_year_axis(
            axis=axis,
            frame=frame,
            group_legend=group_legend,
            include_method_in_label=include_method_in_label,
            show_x_labels=show_x_labels,
            label_order=label_order,
        )
        return values, [int(requested_years[0])], []
    values, years, markers = _render_multi_year_axis(
        axis=axis,
        frame=frame,
        group_legend=group_legend,
        include_method_in_label=include_method_in_label,
    )
    axis.set_xlim(min(years) - 0.5, max(years) + 0.5)
    format_year_axis(axis, years=sorted(set(years)), show_labels=show_x_labels)
    render_transition_markers(axis, markers=markers, shade_right=transition_shade_right)
    return values, years, markers


def _render_single_year_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    group_legend: bool,
    include_method_in_label: bool,
    show_x_labels: bool,
    label_order: list[str] | None,
) -> np.ndarray:
    entries = _single_year_entries(frame, include_method_in_label=include_method_in_label)
    if label_order is not None:
        ranks = {label: index for index, label in enumerate(label_order)}
        entries = sorted(entries, key=lambda item: (ranks.get(item[0], len(ranks)), item[0]))
    positions = np.arange(len(entries), dtype=float)
    values: list[float] = []
    seen: set[str] = set()
    range_positions: list[float] = []
    range_lows: list[float] = []
    range_highs: list[float] = []
    range_colors: list[str] = []
    for index, (label, row, low, high) in enumerate(entries):
        color = str(row[_COLOR_COLUMN])
        visible_label = label if label and label not in seen else "_nolegend_"
        handle = axis.bar(
            [positions[index]],
            [low],
            width=0.72,
            color=color,
            alpha=_LINE_ALPHA,
            label=visible_label,
            zorder=3,
        )
        if group_legend and label:
            group = legend_group_from_row(row)
            bind_deterministic_legend_group(handle, group)
            bind_deterministic_legend_group(handle[0], group)
        if label:
            seen.add(label)
        range_positions.append(float(positions[index]))
        range_lows.append(float(low))
        range_highs.append(float(high))
        range_colors.append(color)
        values.extend([low, high])
    range_mask = np.greater(
        np.asarray(range_highs, dtype=float),
        np.asarray(range_lows, dtype=float),
    )
    range_positions_array = np.asarray(range_positions, dtype=float)[range_mask]
    range_lows_array = np.asarray(range_lows, dtype=float)[range_mask]
    range_highs_array = np.asarray(range_highs, dtype=float)[range_mask]
    range_colors_visible = [
        color for color, keep in zip(range_colors, range_mask.tolist(), strict=True) if keep
    ]
    axis.vlines(
        range_positions_array,
        range_lows_array,
        range_highs_array,
        colors=range_colors_visible,
        linestyles=":",
        linewidth=1.9,
        zorder=4,
    )
    axis.hlines(
        range_highs_array,
        range_positions_array - 0.36,
        range_positions_array + 0.36,
        colors=range_colors_visible,
        linestyles=":",
        linewidth=1.9,
        zorder=4,
    )
    labels = [label for label, _row, _low, _high in entries]
    if show_x_labels and any(str(label).strip() for label in labels):
        format_single_year_category_axis(axis, positions=positions.tolist(), labels=labels)
    else:
        axis.set_xticks(positions)
        axis.set_xticklabels([])
        axis.tick_params(axis="x", length=0)
    axis.set_xlim(float(positions.min()) - 0.65, float(positions.max()) + 0.65)
    return np.asarray(values, dtype=np.float64)


def _render_multi_year_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    group_legend: bool,
    include_method_in_label: bool,
) -> tuple[np.ndarray, list[int], list[TransitionMarker]]:
    years: list[int] = []
    values: list[float] = []
    seen: set[str] = set()
    for label, group in _line_groups(frame, include_method_in_label=include_method_in_label):
        ordered = group.sort_values(YEAR_COLUMN, kind="stable")
        year_values = pd.Series(pd.to_numeric(ordered[YEAR_COLUMN], errors="raise")).astype(int)
        numeric = pd.Series(pd.to_numeric(ordered[VALUE_COLUMN], errors="raise")).astype(float)
        first_row = pd.Series(ordered.iloc[0], copy=False)
        visible_label = label if label and label not in seen else "_nolegend_"
        line = axis.plot(
            year_values.to_numpy(dtype=int),
            numeric.to_numpy(dtype=float),
            color=str(first_row[_COLOR_COLUMN]),
            linestyle="-",
            linewidth=1.8,
            alpha=MULTI_METHOD_LINE_ALPHA if group_legend else _LINE_ALPHA,
            label=visible_label,
            zorder=4,
        )[0]
        if group_legend and visible_label != "_nolegend_":
            bind_deterministic_legend_group(line, legend_group_from_row(first_row))
        seen.add(label)
        years.extend(int(year) for year in year_values.tolist())
        values.extend(float(value) for value in numeric.tolist())
    return (
        np.asarray(values, dtype=np.float64),
        years,
        _visible_series_transition_markers(
            frame,
            include_method_in_label=include_method_in_label,
        ),
    )


def _visible_series_transition_markers(
    frame: pd.DataFrame,
    *,
    include_method_in_label: bool,
) -> list[TransitionMarker]:
    return merged_asr_transition_markers(
        group.sort_values(YEAR_COLUMN, kind="stable")
        for _label, group in _line_groups(
            frame,
            include_method_in_label=include_method_in_label,
        )
    )


def _format_asr_axis(
    axis: Any,
    *,
    values: np.ndarray,
    frame: pd.DataFrame,
    years: list[int],
    limits: tuple[float, float] | None = None,
    grouped_legend: bool = False,
    scale_mode: ASRScaleMode = ASR_LOG_SCALE,
) -> None:
    thresholds = _max_threshold_values(frame)
    axis_values = values
    if thresholds:
        axis_values = np.concatenate([values, np.asarray(thresholds, dtype=np.float64)])
    apply_scaled_asr_axis_policy(
        axis,
        values=axis_values,
        frame=frame,
        scale_mode=scale_mode,
        grouped_legend=grouped_legend,
        limits=limits,
    )
    axis.grid(alpha=0.25, axis="y" if len(set(years)) == 1 else "both")


def _render_footer(*, fig: Any, axis: Any, frame: pd.DataFrame, group_legend: bool) -> None:
    note = _min_variant_note(frame)
    risk_extra = asr_risk_scale_footer_extra_height()
    if group_legend:
        render_grouped_deterministic_legend_below(
            axis,
            legend_note=note,
            extra_height_in=risk_extra,
            excluded_group_titles={ASR_RISK_LEGEND_GROUP_TITLE},
        )
    else:
        render_below_figure_legend(
            fig,
            legend_note=note,
            max_columns=3,
            extra_height_in=risk_extra,
        )
    render_asr_risk_scale_footer(fig, frame=frame)


def _single_year_entries(
    frame: pd.DataFrame,
    *,
    include_method_in_label: bool,
) -> list[tuple[str, pd.Series, float, float]]:
    entries = []
    for group in base_variant_groups(frame):
        legend_row = pd.Series(group.iloc[0], copy=False)
        label = _row_label(legend_row, include_method=include_method_in_label)
        value = _row_float(legend_row, VALUE_COLUMN)
        entries.append((label, legend_row, value, value))
    return sorted(entries, key=lambda item: (-item[2], item[0]))


def _single_year_label_order(frame: pd.DataFrame) -> list[str]:
    scores: dict[str, list[float]] = {}
    for label, _row, low, _high in _single_year_entries(frame, include_method_in_label=True):
        scores.setdefault(label, []).append(low)
    return [
        label
        for label, _score in sorted(
            ((label, float(np.nanmean(values))) for label, values in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def _line_groups(frame: pd.DataFrame, *, include_method_in_label: bool):
    excluded = _line_group_excluded_columns()
    group_columns = [
        column
        for column in frame.columns
        if column not in {*excluded, YEAR_COLUMN} and not str(column).startswith("__figure")
    ]
    for _key, group in frame.groupby(group_columns, dropna=False, sort=True):
        row = pd.Series(group.iloc[0], copy=False)
        yield _row_label(row, include_method=include_method_in_label), group


def _line_group_excluded_columns() -> set[str]:
    return {
        VALUE_COLUMN,
        ROLE_COLUMN,
        _LABEL_COLUMN,
        _COLOR_COLUMN,
        _MAX_THRESHOLD_COLUMN,
        "cumulative_asr",
        *ASR_TRANSITION_SERIES_EXCLUDED_COLUMNS,
    }


def _cumulative_asr_value(group: pd.DataFrame) -> float:
    values = pd.Series(pd.to_numeric(group["cumulative_asr"], errors="coerce")).dropna()
    return float(values.iloc[0])


def _row_label(row: pd.Series, *, include_method: bool) -> str:
    if include_method:
        return str(row.get("__method", "")).strip()
    return ""


def _colors(labels: list[str]) -> dict[str, str]:
    from pyaesa.shared.figures.colors import distinct_colors

    return {
        label: color
        for label, color in zip(labels, distinct_colors(max(1, len(labels))), strict=True)
    }


def _max_threshold_values(frame: pd.DataFrame) -> list[float]:
    values = pd.Series(pd.to_numeric(frame[_MAX_THRESHOLD_COLUMN], errors="coerce")).dropna()
    unique = sorted({round(float(value), 12) for value in values.tolist() if float(value) > 1.0})
    return [float(value) for value in unique[:4]]


def _common_asr_limits(frame: pd.DataFrame, *, scale_mode: ASRScaleMode) -> tuple[float, float]:
    values = pd.Series(pd.to_numeric(frame[VALUE_COLUMN], errors="raise")).to_numpy(
        dtype=np.float64
    )
    if "cumulative_asr" in frame.columns:
        cumulative = pd.Series(pd.to_numeric(frame["cumulative_asr"], errors="coerce")).dropna()
        values = np.concatenate([values, cumulative.to_numpy(dtype=np.float64)])
    return asr_axis_limits(values=values, frame=frame, scale_mode=scale_mode)


def _asr_row_limits(
    frame: pd.DataFrame,
    *values: np.ndarray,
    scale_mode: ASRScaleMode,
) -> tuple[float, float]:
    numeric = np.concatenate([np.asarray(value, dtype=np.float64) for value in values])
    return asr_axis_limits(values=numeric, frame=frame, scale_mode=scale_mode)


def _scale_modes_by_lcia(*, rows: pd.DataFrame) -> dict[str, ASRScaleMode]:
    modes: dict[str, ASRScaleMode] = {}
    for lcia_method, group in rows.groupby("lcia_method", dropna=False, sort=True):
        prepared = _prepare_plot_rows(group)
        values = pd.Series(pd.to_numeric(prepared[VALUE_COLUMN], errors="raise")).to_numpy(
            dtype=np.float64
        )
        if "cumulative_asr" in prepared.columns:
            cumulative = pd.Series(
                pd.to_numeric(prepared["cumulative_asr"], errors="coerce")
            ).dropna()
            values = np.concatenate([values, cumulative.to_numpy(dtype=np.float64)])
        modes[str(lcia_method)] = asr_scale_mode_for_values(values)
    return modes


def _scale_mode_for_frame(
    frame: pd.DataFrame,
    *,
    scale_modes: dict[str, ASRScaleMode],
) -> ASRScaleMode:
    lcia_methods = visible_values(frame, "lcia_method")
    return scale_modes.get(lcia_methods[0], ASR_NORMAL_SCALE) if lcia_methods else ASR_NORMAL_SCALE


def _deterministic_polar_values(frame: pd.DataFrame, *, impact: str) -> np.ndarray:
    scoped = frame.loc[frame["impact"].astype(str).eq(str(impact))]
    if ROLE_COLUMN in scoped.columns:
        scoped = scoped.loc[scoped[ROLE_COLUMN].astype(str).eq(MIN_ROLE)]
    return pd.Series(pd.to_numeric(scoped[VALUE_COLUMN], errors="raise")).to_numpy(dtype=np.float64)


def _min_variant_note(frame: pd.DataFrame) -> str | None:
    if ROLE_COLUMN not in frame.columns:
        return retained_variant_note(frame, single_year=_frame_has_single_year(frame))
    active_columns = [
        column
        for column in VARIANT_COLUMNS
        if column in frame.columns and bool(frame[column].notna().any())
    ]
    if not active_columns:
        return None
    scoped = frame.loc[frame[ROLE_COLUMN].astype(str).eq(MIN_ROLE)]
    complete = scoped.loc[~scoped[active_columns].isna().any(axis=1)]
    row = pd.Series(complete.iloc[0], copy=False)
    values = tuple(row[column] for column in active_columns)
    lines = [f"Min variant compression: {variant_combo_text(active_columns, values)}."]
    method_note = _min_variant_method_note(scoped, active_columns=active_columns)
    if method_note is not None:
        lines.append(method_note)
    if "l2_reuse_year" in active_columns:
        lines.append(
            f"{variant_display_name('l2_reuse_year')} affects only the L2 in L1 "
            "prospective allocation weighting."
        )
    return "\n".join(lines)


def _frame_has_single_year(frame: pd.DataFrame) -> bool:
    return YEAR_COLUMN in frame.columns and len(visible_values(frame, YEAR_COLUMN)) == 1


def _min_variant_method_note(
    frame: pd.DataFrame,
    *,
    active_columns: list[str],
) -> str | None:
    if "__method" not in frame.columns or len(visible_values(frame, "__method")) <= 1:
        return None
    buckets: dict[tuple[str, ...], list[str]] = {}
    for group in base_variant_groups(frame):
        columns = tuple(
            column
            for column in active_columns
            if column in group.columns and bool(group[column].notna().any())
        )
        method = str(pd.Series(group.iloc[0], copy=False)["__method"]).strip()
        if method:
            buckets.setdefault(columns, []).append(method)
    entries = [
        f"{_variant_bucket_label(columns)}: {'; '.join(list(dict.fromkeys(buckets[columns])))}"
        for columns in _ordered_variant_buckets(buckets)
        if columns
    ]
    return "\n".join(entries) if entries else None


def _ordered_variant_buckets(
    buckets: dict[tuple[str, ...], list[str]],
) -> list[tuple[str, ...]]:
    order = {
        ("reference_year",): 0,
        ("l2_reuse_year",): 1,
        ("reference_year", "l2_reuse_year"): 2,
    }
    return sorted(buckets, key=lambda columns: (order.get(columns, 99), columns))


def _variant_bucket_label(columns: tuple[str, ...]) -> str:
    if len(columns) == 1:
        return f"{variant_display_name(columns[0])} only"
    return " and ".join(variant_display_name(column) for column in columns)


def _impact_title(frame: pd.DataFrame, impact: str) -> str:
    return resolve_frame_impact_title(frame) or impact


def _single_year(requested_years: list[int]) -> int | None:
    years = sorted({int(year) for year in requested_years})
    return years[0] if len(years) == 1 else None


def _scope_stem(
    *,
    label: str,
    frame: pd.DataFrame,
    requested_years: list[int],
    cc_source: str,
    cc_type: str,
    include_impact: bool = False,
    selector_token: str = "all",
) -> str:
    if cc_type != "static":
        base = dynamic_output_base_stem(
            base_stem=label,
            lcia_method=visible_values(frame, "lcia_method")[0],
        )
        parts = [base]
        if str(selector_token).strip() and selector_token != "all":
            parts.append(str(selector_token).strip())
        if include_impact:
            parts.extend(visible_values(frame, "impact")[:1])
        parts.extend(_scope_scenario_values(frame)[:1])
        parts.extend(visible_values(frame, "cc_category")[:1])
        model_pair = model_scenario_pair_token(
            models=visible_values(frame, "cc_model"),
            scenarios=visible_values(frame, "cc_scenario"),
        )
        if model_pair is not None:
            parts.append(model_pair)
        return "__".join(sanitize_token(part) for part in parts if str(part).strip())
    parts = [strip_lcia_method_suffix(stem=label, lcia_methods=[cc_source]), cc_source]
    if str(selector_token).strip() and selector_token != "all":
        parts.append(str(selector_token).strip())
    if include_impact:
        parts.extend(visible_values(frame, "impact")[:1])
    parts.extend(_scope_scenario_values(frame)[:1])
    year = _single_year(requested_years)
    if year is not None:
        parts.append(str(year))
    return "__".join(sanitize_token(part) for part in parts if str(part).strip())


def _scope_scenario_values(frame: pd.DataFrame) -> list[str]:
    values = visible_values(frame, AR6_CC_SSP_SCENARIO_COLUMN)
    if values:
        return values
    return visible_values(frame, "asocc_ssp_scenario")


def _row_float(row: pd.Series, column: str) -> float:
    return float(pd.Series(pd.to_numeric(pd.Series([row[column]]), errors="raise")).iloc[0])
