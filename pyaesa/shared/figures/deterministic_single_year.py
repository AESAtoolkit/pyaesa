"""Single year deterministic figure helpers for long form trajectory tables."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from collections.abc import Callable

from pyaesa.shared.tabular.scalars import display_scalar, sanitize_token

from .colors import DEFAULT_SINGLE_SERIES_COLOR
from .figure_footer import render_below_figure_legend
from .layout import DOUBLE_COLUMN_TITLE_TOP
from .lcia_metadata import resolve_frame_impact_title
from .lcia_scope import impact_slices
from .nonnegative_axis import require_nonnegative_figure_ylim
from .save import save_figure
from .scientific_text import format_scientific_figure_text
from .series_labels import require_series_label
from .title_contract import build_resolved_figure_title, clean_panel_title
from .titles import render_figure_title, title_layout_top

_MARKER_COLUMNS = {
    "__transition_marker_year",
    "__transition_marker_label",
    "__transition_marker_color",
}


def exact_single_year_output_base(
    *,
    output_base: Path,
    year: int,
    impact_token: object | None = None,
) -> Path:
    """Return the canonical exact-year deterministic figure output base path."""
    stem = f"{output_base.name}__{int(year)}"
    if impact_token is not None:
        stem = f"{output_base.name}__{sanitize_token(impact_token)}__{int(year)}"
    return output_base.parent / stem


def _grouped_panel_subsets(
    panel_rows: pd.DataFrame,
    *,
    panel_column: str | None,
) -> list[tuple[object | None, pd.DataFrame]]:
    """Return deterministic single-year grouped subsets for one panel."""
    group_columns = [
        column
        for column in panel_rows.columns
        if column not in {"year", "value", panel_column, *_MARKER_COLUMNS}
    ]
    return (
        [(None, panel_rows)]
        if not group_columns
        else list(panel_rows.groupby(group_columns, dropna=False, sort=True))
    )


def _single_year_series_items(
    panel_rows: pd.DataFrame,
    *,
    panel_column: str | None,
    value_scale: float,
) -> list[tuple[str, float]]:
    """Return validated single-year labels and values for one panel."""
    items: list[tuple[str, float]] = []
    for _key, subset in _grouped_panel_subsets(panel_rows, panel_column=panel_column):
        first_row = pd.Series(subset.iloc[0], copy=False)
        items.append(
            (
                require_series_label(
                    first_row,
                    context="Single-year deterministic figure rendering",
                ),
                float(value_scale) * float(subset["value"].iloc[0]),
            )
        )
    return items


def render_single_year_panels(
    *,
    frame: pd.DataFrame,
    years: list[int],
    output_base: Path,
    title_parts: dict[str, str | None],
    ylabel: str,
    dpi: int,
    output_format: str,
    value_scale: float = 1.0,
    percent_ticks: bool = False,
    split_panels: bool = False,
    overlay_panels: bool = False,
    ylabel_resolver: Callable[[pd.DataFrame], str] | None = None,
    footer_note: str | None = None,
    force_zero_ymin: bool = False,
    axis_styler: Callable[[object, pd.DataFrame], None] | None = None,
) -> list[Path]:
    """Render one bar figure per selected year from a long deterministic frame."""
    if frame.empty or not years:
        return []
    work = frame.copy()
    work["year"] = pd.Series(
        pd.to_numeric(pd.Series(work["year"], copy=False), errors="raise"),
        copy=False,
    ).astype(int)
    work["value"] = pd.to_numeric(pd.Series(work["value"], copy=False), errors="raise")
    panel_column = (
        "impact" if "impact" in work.columns else "variable" if "variable" in work.columns else None
    )
    paths: list[Path] = []
    for year in [int(value) for value in years]:
        year_frame = work.loc[pd.Series(work["year"], copy=False).eq(int(year))].copy()
        if year_frame.empty:
            continue
        panels = (
            [("value", year_frame)]
            if panel_column is None
            else list(
                impact_slices(
                    year_frame,
                    impact_column=panel_column,
                    repeat_generic=True,
                )
            )
        )
        panels_count = len(panels)
        if overlay_panels:
            overlay_panel_title = None
            labels: list[str] = []
            overlay_values: list[float] = []
            for panel_value, panel_rows in panels:
                for base_label, value in _single_year_series_items(
                    panel_rows,
                    panel_column=panel_column,
                    value_scale=value_scale,
                ):
                    panel_text = resolve_frame_impact_title(panel_rows) or (
                        display_scalar(panel_value) or "value"
                    )
                    if panel_text == "value":
                        panel_text = ""
                    label = f"{panel_text} | {base_label}" if panel_text else base_label
                    labels.append(label)
                    overlay_values.append(value)
            fig, axis = plt.subplots(figsize=(max(6.8, 1.3 * len(panels)), 5.2))
            for x_pos, (label, value) in enumerate(zip(labels, overlay_values, strict=True)):
                axis.bar(
                    float(x_pos),
                    value,
                    color=DEFAULT_SINGLE_SERIES_COLOR,
                    alpha=0.8,
                    label=label,
                )
            axis.set_xticks(np.arange(len(labels), dtype=float))
            axis.set_xticklabels(labels, rotation=45, ha="right")
            axis.set_ylabel(
                format_scientific_figure_text(
                    ylabel if ylabel_resolver is None else ylabel_resolver(year_frame)
                )
            )
            if axis_styler is not None:
                axis_styler(axis, year_frame)
            if percent_ticks:
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            if force_zero_ymin:
                axis.set_ylim(
                    *require_nonnegative_figure_ylim(
                        values=np.asarray(overlay_values, dtype=float),
                        context=build_resolved_figure_title(
                            title_parts=title_parts,
                            year=year,
                            panel_title=overlay_panel_title,
                            panel_count=panels_count,
                        ),
                    )
                )
            axis.grid(axis="y", alpha=0.25)
            if panels_count == 1:
                overlay_panel_title = resolve_frame_impact_title(panels[0][1]) or (
                    display_scalar(panels[0][0]) or "value"
                )
            figure_title = build_resolved_figure_title(
                title_parts=title_parts,
                year=year,
                panel_title=overlay_panel_title,
                panel_count=panels_count,
            )
            render_figure_title(fig, figure_title)
            fig.subplots_adjust(top=title_layout_top(fig, figure_title, default_top=0.94))
            render_below_figure_legend(fig, legend_note=footer_note)
            paths.extend(
                save_figure(
                    fig,
                    exact_single_year_output_base(
                        output_base=output_base,
                        year=year,
                    ),
                    dpi=dpi,
                    output_format=output_format,
                )
            )
            continue
        if split_panels:
            for panel_value, panel_rows in panels:
                series_items = _single_year_series_items(
                    panel_rows,
                    panel_column=panel_column,
                    value_scale=value_scale,
                )
                labels = [label for label, _value in series_items]
                values = [value for _label, value in series_items]
                fig, axis = plt.subplots(figsize=(max(6.4, 1.1 * len(panel_rows)), 4.8))
                x = np.arange(len(labels), dtype=float)
                axis.bar(x, values, color=DEFAULT_SINGLE_SERIES_COLOR, alpha=0.8)
                axis.set_xticks(x)
                axis.set_xticklabels(labels, rotation=45, ha="right")
                panel_text = resolve_frame_impact_title(panel_rows) or str(panel_value).strip()
                axis.set_ylabel(
                    format_scientific_figure_text(
                        ylabel if ylabel_resolver is None else ylabel_resolver(panel_rows)
                    )
                )
                if axis_styler is not None:
                    axis_styler(axis, panel_rows)
                if percent_ticks:
                    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
                if force_zero_ymin:
                    axis.set_ylim(
                        *require_nonnegative_figure_ylim(
                            values=np.asarray(values, dtype=float),
                            context=build_resolved_figure_title(
                                title_parts=title_parts,
                                year=year,
                                panel_title=panel_text,
                                panel_count=1,
                            ),
                        )
                    )
                axis.grid(axis="y", alpha=0.25)
                figure_title = build_resolved_figure_title(
                    title_parts=title_parts,
                    year=year,
                    panel_title=panel_text,
                    panel_count=1,
                )
                render_figure_title(fig, figure_title)
                fig.subplots_adjust(top=title_layout_top(fig, figure_title, default_top=0.94))
                render_below_figure_legend(fig, legend_note=footer_note)
                normalized_panel_title = clean_panel_title(panel_title=panel_text)
                paths.extend(
                    save_figure(
                        fig,
                        exact_single_year_output_base(
                            output_base=output_base,
                            year=year,
                            impact_token=None if normalized_panel_title is None else panel_value,
                        ),
                        dpi=dpi,
                        output_format=output_format,
                    )
                )
            continue
        fig_width = max(6.4, 1.1 * max(len(panel) for _label, panel in panels))
        fig, axes = plt.subplots(
            max(1, len(panels)),
            1,
            figsize=(fig_width, 4.8 * len(panels)),
            squeeze=False,
        )
        flat_axes = list(axes.flatten())
        for axis, (panel_value, panel_rows) in zip(flat_axes, panels, strict=True):
            series_items = _single_year_series_items(
                panel_rows,
                panel_column=panel_column,
                value_scale=value_scale,
            )
            labels = [label for label, _value in series_items]
            values = [value for _label, value in series_items]
            x = np.arange(len(labels), dtype=float)
            axis.bar(x, values, color=DEFAULT_SINGLE_SERIES_COLOR, alpha=0.8)
            axis.set_xticks(x)
            axis.set_xticklabels(labels, rotation=45, ha="right")
            axis.set_ylabel(
                format_scientific_figure_text(
                    ylabel if ylabel_resolver is None else ylabel_resolver(panel_rows)
                )
            )
            if axis_styler is not None:
                axis_styler(axis, panel_rows)
            if percent_ticks:
                axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            if force_zero_ymin:
                axis.set_ylim(
                    *require_nonnegative_figure_ylim(
                        values=np.asarray(values, dtype=float),
                        context=build_resolved_figure_title(
                            title_parts=title_parts,
                            year=year,
                            panel_count=panels_count,
                        ),
                    )
                )
            axis.grid(axis="y", alpha=0.25)
            panel_text = resolve_frame_impact_title(panel_rows) or str(panel_value).strip()
        grid_panel_title = None
        if panels_count == 1:
            panel_value, panel_rows = panels[0]
            grid_panel_title = resolve_frame_impact_title(panel_rows) or str(panel_value).strip()
        figure_title = build_resolved_figure_title(
            title_parts=title_parts,
            year=year,
            panel_title=grid_panel_title,
            panel_count=panels_count,
        )
        render_figure_title(fig, figure_title)
        fig.subplots_adjust(
            top=title_layout_top(
                fig,
                figure_title,
                default_top=DOUBLE_COLUMN_TITLE_TOP,
                panel_title_pad=5 if panels_count > 1 else 0,
            )
        )
        render_below_figure_legend(fig, legend_note=footer_note)
        paths.extend(
            save_figure(
                fig,
                exact_single_year_output_base(
                    output_base=output_base,
                    year=year,
                ),
                dpi=dpi,
                output_format=output_format,
            )
        )
    return paths
