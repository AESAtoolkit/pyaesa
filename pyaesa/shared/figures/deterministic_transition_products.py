"""Shared deterministic transition figure products for long form trajectories."""

from collections.abc import Callable, Iterator
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_hex
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from pyaesa.shared.figures.deterministic_legends import (
    bind_deterministic_legend_group,
    render_grouped_deterministic_legend_below,
)
from pyaesa.shared.figures.checkpoints import (
    default_checkpoint_years,
    has_exact_single_year_scope,
    unique_figure_years,
)
from pyaesa.shared.figures.deterministic_single_year import render_single_year_panels
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    format_integer_year_axis,
    resolve_layout,
)
from pyaesa.shared.figures.lcia_metadata import resolve_frame_impact_title
from pyaesa.shared.figures.lcia_scope import (
    combined_lcia_impact_slices,
    lcia_method_slices,
    suffix_path,
)
from pyaesa.shared.figures.paths import scope_filename_stem
from pyaesa.shared.figures.multi_year_transitions import (
    TransitionMarker,
    render_transition_markers,
)
from pyaesa.shared.figures.nonnegative_axis import require_nonnegative_figure_ylim
from pyaesa.shared.figures.save import save_figure
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.series_labels import with_series_label_column
from pyaesa.shared.figures.selector_slices import selector_slices
from pyaesa.shared.figures.title_contract import (
    SelectorScopeRequest,
    build_resolved_figure_title,
    prospective_scope_slices,
    resolve_panel_title,
)
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.figures.variant_selection import (
    VariantCompression,
    split_variant_frames,
    variant_footer_note,
)
from pyaesa.shared.tabular.scalars import sanitize_token
from pyaesa.shared.figures.method_identity import resolve_figure_display_label

from pyaesa.shared.figures.transition_panel_payloads import (
    panel_groups,
    panel_markers,
    prepare_transition_frame,
    series_line_specs,
    series_payloads,
    series_transition_years,
    VariantLineSpec,
)

_DISTINCT_SERIES_PALETTE = [
    "#006ba4",
    "#ff800e",
    "#ababab",
    "#595959",
    "#5f9ed1",
    "#c85200",
    "#898989",
    "#a2c8ec",
    "#ffbc79",
    "#cfcfcf",
    "#cfe8f3",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#bc80bd",
]

ScopedOutputBuilder = Callable[
    [Path, str | None, str, str, str | None],
    Path,
]
TransitionProductScope = tuple[str, dict[str, str | None], pd.DataFrame, Path]


def _panel_variant_entries(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
    marker_label: str,
    marker_color: str,
    transition_grouping_skip_columns: set[str] | None = None,
    variant_display_aliases: dict[str, str] | None = None,
) -> tuple[
    list[tuple[str, pd.DataFrame, tuple[VariantCompression, ...], str | None]],
    pd.DataFrame,
    str | None,
    str | None,
]:
    """Return per-panel compressed transition frames plus combined frame and footer notes."""
    panel_column = "impact" if "impact" in frame.columns else None
    raw_groups = panel_groups(frame, panel_column=panel_column)
    entries: list[tuple[str, pd.DataFrame, tuple[VariantCompression, ...], str | None]] = []
    main_frames: list[pd.DataFrame] = []
    notes: list[tuple[str, str]] = []
    for panel_label, panel_frame in raw_groups:
        main_frame, compressions = split_variant_frames(
            frame=panel_frame,
            requested_years=requested_years,
        )
        note = variant_footer_note(
            compressions,
            average_over_years=True,
            display_aliases=variant_display_aliases,
        )
        prepared = prepare_transition_frame(
            frame=main_frame,
            requested_years=requested_years,
            marker_label=marker_label,
            marker_color=marker_color,
            transition_grouping_skip_columns=transition_grouping_skip_columns,
        )
        entries.append((panel_label, prepared, compressions, note))
        main_frames.append(main_frame)
        if note:
            notes.append((panel_label, note))
    combined_main = (
        pd.concat(main_frames, ignore_index=True) if main_frames else frame.iloc[0:0].copy()
    )
    combined_note: str | None = None
    if notes:
        unique_notes = {note for _panel_label, note in notes}
        if len(unique_notes) == 1:
            combined_note = notes[0][1]
        else:
            combined_note = "\n".join(f"{panel_label}: {note}" for panel_label, note in notes)
    single_year_note = None
    if notes:
        panel_single_year_notes = [
            (panel_label, note)
            for panel_label, _prepared, compressions, _note in entries
            for note in [
                variant_footer_note(
                    compressions,
                    average_over_years=False,
                    display_aliases=variant_display_aliases,
                )
            ]
            if note is not None
        ]
        single_year_notes = {note for _panel_label, note in panel_single_year_notes}
        if len(single_year_notes) == 1:
            single_year_note = next(iter(single_year_notes))
        else:
            single_year_note = "\n".join(
                f"{panel_label}: {note}" for panel_label, note in panel_single_year_notes
            )
    return entries, combined_main, combined_note, single_year_note


def _build_variant_line_specs(
    groups: list[tuple[str, pd.DataFrame, tuple[VariantCompression, ...], str | None]],
    *,
    default_colors: list[str],
    include_panel_column_in_series: bool = False,
) -> dict[tuple[str, ...], tuple[str, VariantLineSpec]]:
    """Build {series_key: (color, variant_line_spec)} from all panels.

    Colors are assigned per unique base_color_key in sorted-key order so that the
    same method gets the same color across all panels within one figure.
    """
    combined: dict[tuple[str, ...], VariantLineSpec] = {}
    for _panel_label, panel_frame, compressions, _note in groups:
        panel_column = (
            None
            if include_panel_column_in_series
            else "impact"
            if "impact" in panel_frame.columns
            else None
        )
        specs = series_line_specs(
            panel_frame,
            panel_column=panel_column,
            compressions=compressions,
            skip_columns={"series_label"},
        )
        if include_panel_column_in_series and "impact" in panel_frame.columns:
            specs = {
                key: VariantLineSpec(
                    base_color_key=(str(_panel_label), *spec.base_color_key),
                    line_style=spec.line_style,
                    show_in_legend=spec.show_in_legend,
                    prospective_only=spec.prospective_only,
                )
                for key, spec in specs.items()
            }
        combined.update(specs)
    if not combined:
        return {}
    palette = [*_DISTINCT_SERIES_PALETTE]
    base_key_order: list[tuple[str, ...]] = list(
        dict.fromkeys(v.base_color_key for _k, v in sorted(combined.items()))
    )
    if len(base_key_order) > len(palette):
        palette.extend(
            _fallback_series_colors(
                n_colors=len(base_key_order) - len(palette),
                excluded=set(palette),
            )
        )
    palette.extend(color for color in default_colors if str(color) not in palette)
    color_by_group = {bk: palette[i % len(palette)] for i, bk in enumerate(base_key_order)}
    return {k: (color_by_group[v.base_color_key], v) for k, v in combined.items()}


def _fallback_series_colors(*, n_colors: int, excluded: set[str]) -> list[str]:
    """Return extra deterministic colors when more than 20 method families are present."""
    if n_colors <= 0:
        return []
    fallback: list[str] = []
    step_count = max(24, int(n_colors) * 2)
    hsv_map = colormaps["hsv"]
    for index in range(step_count):
        color = hsv_map(index / float(step_count))
        color_hex = str(to_hex(color, keep_alpha=False))
        if color_hex in excluded or color_hex in fallback:
            continue
        fallback.append(color_hex)
        if len(fallback) >= n_colors:
            break
    return fallback


def _plot_series(
    axis,
    *,
    years: list[int],
    values: list[float],
    label: str,
    legend_group: str,
    color: str | None,
    line_style: str,
    show_in_legend: bool,
    prospective_only: bool,
    transition_year: int | None,
) -> None:
    """Run one deterministic series, optionally switching style at the transition year."""
    common: dict[str, object] = {
        "linewidth": 1.7,
        "alpha": 0.92,
    }
    if color is not None:
        common["color"] = color
    if not prospective_only or transition_year is None or line_style == "solid":
        line = axis.plot(
            years,
            values,
            label=label if show_in_legend else "_nolegend_",
            linestyle=line_style,
            **common,
        )[0]
        bind_deterministic_legend_group(line, legend_group)
        return
    historical_years: list[int] = []
    historical_values: list[float] = []
    prospective_years: list[int] = []
    prospective_values: list[float] = []
    last_historical_point: tuple[int, float] | None = None
    for year, value in zip(years, values, strict=True):
        if int(year) < int(transition_year):
            historical_years.append(year)
            historical_values.append(value)
            last_historical_point = (int(year), float(value))
        else:
            prospective_years.append(year)
            prospective_values.append(value)
    if historical_years:
        line = axis.plot(
            historical_years,
            historical_values,
            label="_nolegend_",
            linestyle="solid",
            **common,
        )[0]
        bind_deterministic_legend_group(line, legend_group)
    if prospective_years:
        if last_historical_point is not None:
            prospective_years.insert(0, last_historical_point[0])
            prospective_values.insert(0, last_historical_point[1])
        line = axis.plot(
            prospective_years,
            prospective_values,
            label=label if show_in_legend else "_nolegend_",
            linestyle=line_style,
            **common,
        )[0]
        bind_deterministic_legend_group(line, legend_group)


def _format_year_axis(axis, *, requested_years: list[int]) -> None:
    """Keep multi-year deterministic figures labelled by integer study years."""
    years = sorted({int(year) for year in requested_years})
    format_integer_year_axis(axis, years=years)


def _render_multi_year_panels(
    *,
    groups: list[tuple[str, pd.DataFrame, tuple[VariantCompression, ...], str | None]],
    requested_years: list[int],
    output_base: Path,
    title_parts: dict[str, str | None],
    ylabel: str,
    dpi: int,
    output_format: str,
    value_scale: float,
    percent_ticks: bool,
    axis_styler: Callable[[object, pd.DataFrame], None] | None,
    marker_transform: Callable[[pd.DataFrame, list], list] | None,
    ylabel_resolver: Callable[[pd.DataFrame], str] | None,
    split_panels: bool,
    overlay_panels: bool,
    footer_note: str | None,
    force_zero_ymin: bool,
    include_overlay_panel_column_in_series: bool = False,
) -> list[Path]:
    _default_colors: list[str] = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _global_specs = _build_variant_line_specs(
        groups,
        default_colors=_default_colors,
        include_panel_column_in_series=bool(
            overlay_panels and include_overlay_panel_column_in_series
        ),
    )
    panel_count = len(groups)
    if overlay_panels:
        fig, axis = plt.subplots(figsize=(16.0, 6.8))
        merged_markers: dict[tuple[int, str, str], TransitionMarker] = {}
        for panel_label, panel_frame, _panel_compressions, _panel_note in groups:
            if axis_styler is not None:
                axis_styler(axis, panel_frame)
            panel_column = "impact" if "impact" in panel_frame.columns else None
            payload_panel_column = None if include_overlay_panel_column_in_series else panel_column
            payloads = series_payloads(
                panel_frame,
                requested_years=requested_years,
                panel_column=payload_panel_column,
                value_scale=value_scale,
            )
            transition_years = series_transition_years(
                panel_frame,
                panel_column=payload_panel_column,
            )
            for key in sorted(payloads):
                label, years, values, legend_group = payloads[key]
                _spec = _global_specs[key]
                _col, _line_spec = _spec
                _plot_series(
                    axis,
                    years=years,
                    values=values,
                    label=label,
                    legend_group=legend_group,
                    color=_col,
                    line_style=_line_spec.line_style,
                    show_in_legend=_line_spec.show_in_legend,
                    prospective_only=_line_spec.prospective_only,
                    transition_year=transition_years.get(key),
                )
            markers = panel_markers(panel_frame)
            if marker_transform is not None:
                markers = marker_transform(panel_frame, markers)
            for marker in markers:
                merged_markers[(int(marker.year), str(marker.label), str(marker.color))] = marker
        render_transition_markers(axis, markers=list(merged_markers.values()))
        axis.set_xlabel("")
        axis.set_ylabel(format_scientific_figure_text(ylabel))
        if percent_ticks:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        if force_zero_ymin:
            series_values = np.asarray(
                [
                    value
                    for _panel_label, panel_frame, _panel_compressions, _panel_note in groups
                    for _key, (_label, _years, values, _group) in series_payloads(
                        panel_frame,
                        requested_years=requested_years,
                        panel_column=(
                            None
                            if include_overlay_panel_column_in_series
                            else "impact"
                            if "impact" in panel_frame.columns
                            else None
                        ),
                        value_scale=value_scale,
                    ).items()
                    for value in values
                ],
                dtype=float,
            )
            axis.set_ylim(
                *require_nonnegative_figure_ylim(
                    values=series_values,
                    context=build_resolved_figure_title(
                        title_parts=title_parts,
                        panel_count=panel_count,
                    ),
                )
            )
        axis.grid(alpha=0.25)
        render_grouped_deterministic_legend_below(axis, legend_note=footer_note)
        overlay_panel_title = None
        if panel_count == 1:
            only_label, only_frame, _only_comp, _only_note = groups[0]
            overlay_panel_title = resolve_frame_impact_title(only_frame) or str(only_label).strip()
        figure_title = build_resolved_figure_title(
            title_parts=title_parts,
            panel_title=overlay_panel_title,
            panel_count=panel_count,
        )
        render_figure_title(fig, figure_title)
        fig.subplots_adjust(top=title_layout_top(fig, figure_title, default_top=0.94))
        return save_figure(fig, output_base, dpi=dpi, output_format=output_format)
    if split_panels:
        paths: list[Path] = []
        for panel_label, panel_frame, panel_compressions, panel_note in groups:
            fig, axis = plt.subplots(figsize=(16.0, 6.2))
            if axis_styler is not None:
                axis_styler(axis, panel_frame)
            panel_column = "impact" if "impact" in panel_frame.columns else None
            payloads = series_payloads(
                panel_frame,
                requested_years=requested_years,
                panel_column=panel_column,
                value_scale=value_scale,
            )
            transition_years = series_transition_years(panel_frame, panel_column=panel_column)
            _split_specs = _build_variant_line_specs(
                [(panel_label, panel_frame, panel_compressions, panel_note)],
                default_colors=_default_colors,
            )
            for key in sorted(payloads):
                label, years, values, legend_group = payloads[key]
                _spec = _split_specs[key]
                _col, _line_spec = _spec
                _plot_series(
                    axis,
                    years=years,
                    values=values,
                    label=label,
                    legend_group=legend_group,
                    color=_col,
                    line_style=_line_spec.line_style,
                    show_in_legend=_line_spec.show_in_legend,
                    prospective_only=_line_spec.prospective_only,
                    transition_year=transition_years.get(key),
                )
            markers = panel_markers(panel_frame)
            if marker_transform is not None:
                markers = marker_transform(panel_frame, markers)
            render_transition_markers(axis, markers=markers)
            panel_text = resolve_frame_impact_title(panel_frame) or str(panel_label).strip()
            axis.set_xlabel("")
            axis.set_ylabel(
                format_scientific_figure_text(
                    ylabel if ylabel_resolver is None else ylabel_resolver(panel_frame)
                )
            )
            if percent_ticks:
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            _format_year_axis(axis, requested_years=requested_years)
            if force_zero_ymin:
                series_values = np.asarray(
                    [
                        value
                        for _label, _years, values, _group in payloads.values()
                        for value in values
                    ],
                    dtype=float,
                )
                axis.set_ylim(
                    *require_nonnegative_figure_ylim(
                        values=series_values,
                        context=build_resolved_figure_title(
                            title_parts=title_parts,
                            panel_title=panel_text,
                            panel_count=1,
                        ),
                    )
                )
            axis.grid(alpha=0.25)
            if payloads:
                render_grouped_deterministic_legend_below(
                    axis,
                    legend_note=panel_note or footer_note,
                )
            figure_title = build_resolved_figure_title(
                title_parts=title_parts,
                panel_title=panel_text,
                panel_count=1,
            )
            render_figure_title(fig, figure_title)
            fig.subplots_adjust(top=title_layout_top(fig, figure_title, default_top=0.94))
            paths.extend(
                save_figure(
                    fig,
                    output_base.parent / f"{output_base.name}__{sanitize_token(panel_label)}",
                    dpi=dpi,
                    output_format=output_format,
                )
            )
        return paths
    layout = resolve_layout(impacts_count=max(1, len(groups)))
    fig, axes = plt.subplots(
        int(layout["nrows"]),
        int(layout["ncols"]),
        figsize=(float(layout["fig_width"]), float(layout["fig_height"])),
        squeeze=False,
    )
    flat_axes = list(axes.flatten())
    for axis in flat_axes[len(groups) :]:
        axis.set_visible(False)
    for axis, (panel_label, panel_frame, _panel_compressions, _panel_note) in zip(
        flat_axes, groups, strict=False
    ):
        if axis_styler is not None:
            axis_styler(axis, panel_frame)
        panel_column = "impact" if "impact" in panel_frame.columns else None
        payloads = series_payloads(
            panel_frame,
            requested_years=requested_years,
            panel_column=panel_column,
            value_scale=value_scale,
        )
        transition_years = series_transition_years(panel_frame, panel_column=panel_column)
        for key in sorted(payloads):
            label, years, values, legend_group = payloads[key]
            _spec = _global_specs[key]
            _col, _line_spec = _spec
            _plot_series(
                axis,
                years=years,
                values=values,
                label=label,
                legend_group=legend_group,
                color=_col,
                line_style=_line_spec.line_style,
                show_in_legend=_line_spec.show_in_legend,
                prospective_only=_line_spec.prospective_only,
                transition_year=transition_years.get(key),
            )
        markers = panel_markers(panel_frame)
        if marker_transform is not None:
            markers = marker_transform(panel_frame, markers)
        render_transition_markers(axis, markers=markers)
        axis.set_xlabel("")
        axis.set_ylabel(
            format_scientific_figure_text(
                ylabel if ylabel_resolver is None else ylabel_resolver(panel_frame)
            )
        )
        if percent_ticks:
            axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        _format_year_axis(axis, requested_years=requested_years)
        if force_zero_ymin:
            series_values = np.asarray(
                [value for _label, _years, values, _group in payloads.values() for value in values],
                dtype=float,
            )
            axis.set_ylim(
                *require_nonnegative_figure_ylim(
                    values=series_values,
                    context=build_resolved_figure_title(
                        title_parts=title_parts,
                        panel_count=panel_count,
                    ),
                )
            )
        axis.grid(alpha=0.25)
        panel_text = resolve_frame_impact_title(panel_frame) or str(panel_label).strip()
        resolved_panel_title = resolve_panel_title(
            panel_title=panel_text,
            panel_count=panel_count,
        )
        if resolved_panel_title is not None:
            axis.set_title(format_scientific_figure_text(resolved_panel_title), loc="left")
    if any(axis.get_legend_handles_labels()[1] for axis in flat_axes[: len(groups)]):
        render_grouped_deterministic_legend_below(flat_axes[0], legend_note=footer_note)
    figure_panel_title = None
    if panel_count == 1:
        only_label, only_frame, _only_comp, _only_note = groups[0]
        figure_panel_title = resolve_frame_impact_title(only_frame) or str(only_label).strip()
    figure_title = build_resolved_figure_title(
        title_parts=title_parts,
        panel_title=figure_panel_title,
        panel_count=panel_count,
    )
    render_figure_title(fig, figure_title)
    fig.subplots_adjust(
        top=title_layout_top(
            fig,
            figure_title,
            default_top=DOUBLE_COLUMN_TITLE_TOP,
            panel_title_pad=5 if panel_count > 1 else 0,
        )
    )
    return save_figure(fig, output_base, dpi=dpi, output_format=output_format)


def render_transition_products(
    *,
    long_frame: pd.DataFrame,
    requested_years: list[int],
    output_base: Path,
    family: str,
    user_facing_override_label: str | None = None,
    ylabel: str,
    dpi: int,
    output_format: str,
    marker_label: str,
    marker_color: str,
    value_scale: float = 1.0,
    percent_ticks: bool = False,
    axis_styler: Callable[[object, pd.DataFrame], None] | None = None,
    marker_transform: Callable[[pd.DataFrame, list], list] | None = None,
    ylabel_resolver: Callable[[pd.DataFrame], str] | None = None,
    split_panels: bool = False,
    overlay_panels: bool = False,
    single_year_renderer: Callable[..., list[Path]] | None = None,
    include_single_year_products: bool = True,
    group_combined_by_impact: bool = False,
    force_zero_ymin: bool = False,
    transition_grouping_skip_columns: set[str] | None = None,
    selector_scope_request: SelectorScopeRequest | None = None,
    scoped_output_builder: ScopedOutputBuilder | None = None,
    include_overlay_panel_column_in_series: bool = False,
    variant_display_aliases: dict[str, str] | None = None,
) -> list[Path]:
    """Render main, supplemental, and checkpoint figures from one long transition frame."""
    paths: list[Path] = []
    exact_single_year_scope = has_exact_single_year_scope(requested_years)
    single_years = (
        unique_figure_years(requested_years)
        if exact_single_year_scope
        else default_checkpoint_years(requested_years)
    )

    def outer_scopes() -> Iterator[TransitionProductScope]:
        """Yield final transition figure scopes without retaining all selector frames."""
        if group_combined_by_impact:
            for (
                impact_token,
                _impact_title,
                impact_frame,
                lcia_method,
            ) in combined_lcia_impact_slices(long_frame):
                for selector_token, selector_title, selector_frame in selector_slices(
                    impact_frame,
                    selector_scope_request=selector_scope_request,
                ):
                    for scenario_token, scenario_title, scenario_frame in prospective_scope_slices(
                        selector_frame
                    ):
                        scoped_output = (
                            _default_scoped_output_base(
                                output_base=output_base,
                                lcia_method=lcia_method,
                                selector_token=selector_token,
                                scenario_token=scenario_token,
                                impact_token=impact_token,
                            )
                            if scoped_output_builder is None
                            else scoped_output_builder(
                                output_base,
                                lcia_method,
                                selector_token,
                                scenario_token,
                                impact_token,
                            )
                        )
                        yield (
                            impact_token,
                            {
                                "family": family,
                                "selector_scope": selector_title or None,
                                "lcia_method": lcia_method,
                                "user_facing_override_label": resolve_figure_display_label(
                                    frame=scenario_frame,
                                    user_facing_override_label=user_facing_override_label,
                                ),
                                "prospective_scope": scenario_title,
                            },
                            scenario_frame,
                            scoped_output,
                        )
            return
        for _lcia_token, _lcia_title, lcia_frame, lcia_method in lcia_method_slices(long_frame):
            for selector_token, selector_title, selector_frame in selector_slices(
                lcia_frame,
                selector_scope_request=selector_scope_request,
            ):
                for scenario_token, scenario_title, scenario_frame in prospective_scope_slices(
                    selector_frame
                ):
                    scoped_output = (
                        _default_scoped_output_base(
                            output_base=output_base,
                            lcia_method=lcia_method,
                            selector_token=selector_token,
                            scenario_token=scenario_token,
                            impact_token=None,
                        )
                        if scoped_output_builder is None
                        else scoped_output_builder(
                            output_base,
                            lcia_method,
                            selector_token,
                            scenario_token,
                            None,
                        )
                    )
                    title_parts = {
                        "family": family,
                        "selector_scope": selector_title or None,
                        "lcia_method": lcia_method,
                        "user_facing_override_label": resolve_figure_display_label(
                            frame=scenario_frame,
                            user_facing_override_label=user_facing_override_label,
                        ),
                        "prospective_scope": scenario_title,
                    }
                    yield (
                        selector_token,
                        title_parts,
                        scenario_frame,
                        scoped_output,
                    )

    for _scope_token, scoped_title_parts, scoped_frame, scoped_base in outer_scopes():
        panel_entries, main_frame, footer_note, single_year_footer_note = _panel_variant_entries(
            frame=scoped_frame,
            requested_years=requested_years,
            marker_label=marker_label,
            marker_color=marker_color,
            transition_grouping_skip_columns=transition_grouping_skip_columns,
            variant_display_aliases=variant_display_aliases,
        )
        if not exact_single_year_scope:
            paths.extend(
                _render_multi_year_panels(
                    groups=panel_entries,
                    requested_years=requested_years,
                    output_base=scoped_base,
                    title_parts=scoped_title_parts,
                    ylabel=ylabel,
                    dpi=dpi,
                    output_format=output_format,
                    value_scale=value_scale,
                    percent_ticks=percent_ticks,
                    axis_styler=axis_styler,
                    marker_transform=marker_transform,
                    ylabel_resolver=ylabel_resolver,
                    split_panels=split_panels,
                    overlay_panels=overlay_panels,
                    footer_note=footer_note,
                    force_zero_ymin=force_zero_ymin,
                    include_overlay_panel_column_in_series=(include_overlay_panel_column_in_series),
                )
            )
        if include_single_year_products or exact_single_year_scope:
            renderer = (
                render_single_year_panels if single_year_renderer is None else single_year_renderer
            )
            paths.extend(
                renderer(
                    frame=main_frame,
                    years=single_years,
                    output_base=scoped_base,
                    title_parts=scoped_title_parts,
                    ylabel=ylabel,
                    dpi=dpi,
                    output_format=output_format,
                    value_scale=value_scale,
                    percent_ticks=percent_ticks,
                    split_panels=split_panels,
                    overlay_panels=overlay_panels,
                    ylabel_resolver=ylabel_resolver,
                    footer_note=single_year_footer_note,
                    force_zero_ymin=force_zero_ymin,
                )
            )
    return paths


def render_transition_products_with_series_labels(
    *,
    long_frame: pd.DataFrame,
    label_columns: tuple[str, ...],
    skip_columns: set[str] | None = None,
    label_display_aliases: dict[str, str] | None = None,
    context: str,
    **kwargs,
) -> list[Path]:
    """Render transition products after constructing one explicit series label column."""
    prepared = with_series_label_column(
        long_frame,
        label_columns=label_columns,
        skip_columns=skip_columns,
        display_aliases=label_display_aliases,
        context=context,
    )
    return render_transition_products(
        long_frame=prepared,
        **kwargs,
    )


def _default_scoped_output_base(
    *,
    output_base: Path,
    lcia_method: str | None,
    selector_token: str,
    scenario_token: str,
    impact_token: str | None,
) -> Path:
    """Return the default deterministic scoped output path."""
    scoped = output_base.parent / scope_filename_stem(
        base_stem=output_base.name,
        lcia_method=lcia_method,
    )
    if selector_token != "all":
        scoped = suffix_path(scoped, selector_token)
    if scenario_token != "all":
        scoped = suffix_path(scoped, scenario_token)
    if impact_token is not None:
        scoped = suffix_path(scoped, impact_token)
    return scoped
