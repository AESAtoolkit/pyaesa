"""Panel scope renderers for IO-LCA uncertainty figures."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from pyaesa.io_lca.figures.common import (
    figure_stem,
    impact_panel_layout,
    lca_transition_markers,
    lcia_method_tag,
    ordered_impacts,
    panel_impact_unit,
    selector_scope_token,
)
from pyaesa.io_lca.uncertainty.figures.scope_planner import VALUE_ARRAY_COLUMN
from pyaesa.shared.figures.colors import (
    DEFAULT_SINGLE_SERIES_COLOR,
)
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN,
    TRANSITION_PANEL_TITLE_PAD,
    bottom_panel_indices,
    format_integer_year_axis,
    show_panel_x_labels,
)
from pyaesa.shared.figures.figure_footer import set_footer_min_plot_height
from pyaesa.shared.figures.multi_year_transitions import (
    render_transition_markers,
    transition_title_pad,
)
from pyaesa.shared.figures.nonnegative_axis import require_nonnegative_figure_ylim
from pyaesa.shared.figures.paths import output_file_path
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.trajectory_bands import (
    SUMMARY_COLUMNS,
    render_trajectory_band,
    render_trajectory_band_legend_below,
)
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.figures.title_contract import (
    build_resolved_figure_title,
    resolve_panel_title,
    resolve_selector_scope,
)
from pyaesa.shared.figures.violin_summary import (
    ViolinSummaryMode,
    render_violin_summaries,
    render_violin_summary_legend_below,
)

_LINE_ALPHA = 0.82
_TWO_COLUMN_PANEL_HSPACE = 0.32
_TWO_COLUMN_TRANSITION_HSPACE = 0.42
_PANEL_TITLE_PAD = 5
_DEFAULT_COLOR = DEFAULT_SINGLE_SERIES_COLOR
_BAND_COLOR = DEFAULT_SINGLE_SERIES_COLOR


def write_band_scope(
    *,
    group_df: pd.DataFrame,
    reference_frame: pd.DataFrame,
    figures_dir: Path,
    lcia_method: str,
    family_label: str,
    selector_cols: list[str],
    impact_labels: dict[str, str],
    impacts: list[str],
    years: list[int],
    scenario_token: str | None,
    scenario_title: str | None,
    dpi: int,
    output_format: str,
    file_stem_prefix: str | None = None,
) -> list[Path]:
    """Write one multi-year uncertainty band figure scope."""
    panel_count = len(impacts)
    layout = impact_panel_layout(impacts_count=panel_count)
    fig, axes = plt.subplots(
        int(layout["nrows"]),
        int(layout["ncols"]),
        figsize=(float(layout["fig_width"]), float(layout["fig_height"])),
        squeeze=False,
    )
    if panel_count == 1:
        set_footer_min_plot_height(fig, height_in=SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN)
    flat_axes = list(axes.flatten())
    for axis in flat_axes[len(impacts) :]:
        axis.set_visible(False)
    ncols = int(layout["ncols"])
    bottom_indices = bottom_panel_indices(panel_count=panel_count, ncols=ncols)
    has_transition = False
    for panel_index, (axis, impact) in enumerate(
        zip(flat_axes[: len(impacts)], impacts, strict=True)
    ):
        panel = group_df.loc[group_df["impact"].astype(str).eq(str(impact))].copy()
        markers = lca_transition_markers(panel)
        has_transition = has_transition or bool(markers)
        _render_band_panel(
            axis=axis,
            frame=panel,
            lcia_method=lcia_method,
            impact=impact,
            years=years,
            show_x_labels=show_panel_x_labels(
                panel_index=panel_index,
                bottom_indices=bottom_indices,
            ),
        )
        render_transition_markers(axis, markers=markers)
        panel_title = resolve_panel_title(
            panel_title=impact_labels[impact],
            panel_count=panel_count,
        )
        axis.set_title(
            "" if panel_title is None else format_scientific_figure_text(panel_title),
            loc="left",
            pad=transition_title_pad(
                markers,
                no_transition=_PANEL_TITLE_PAD,
                single_transition=TRANSITION_PANEL_TITLE_PAD,
                component_transition=TRANSITION_PANEL_TITLE_PAD,
            ),
        )
        axis.set_ylabel(panel_impact_unit(frame=panel))
    figure_title = _title(
        family_label=family_label,
        lcia_method=lcia_method,
        group_df=group_df,
        selector_cols=selector_cols,
        prospective_scope=scenario_title,
    )
    render_figure_title(fig, figure_title)
    fig.subplots_adjust(
        hspace=_TWO_COLUMN_TRANSITION_HSPACE if has_transition else _TWO_COLUMN_PANEL_HSPACE,
        wspace=0.16,
        top=title_layout_top(
            fig,
            figure_title,
            default_top=DOUBLE_COLUMN_TITLE_TOP,
            panel_title_pad=TRANSITION_PANEL_TITLE_PAD
            if has_transition
            else (_PANEL_TITLE_PAD if len(impacts) > 1 else 0),
        ),
    )
    render_trajectory_band_legend_below(fig, color=_BAND_COLOR, ncol=4)
    selector_token = selector_scope_token(
        group_frame=group_df,
        selector_cols=selector_cols,
        reference_frame=reference_frame,
    )
    out_base = figures_dir / figure_stem(
        lcia_method=lcia_method_tag(lcia_method),
        selector_scope_token=selector_token,
        scenario_token=scenario_token,
        stem_prefix=file_stem_prefix,
    )
    return _save(fig=fig, output_stem=out_base, output_format=output_format, dpi=dpi)


def write_violin_scope(
    *,
    year_df: pd.DataFrame,
    reference_frame: pd.DataFrame,
    figures_dir: Path,
    lcia_method: str,
    family_label: str,
    selector_cols: list[str],
    impact_labels: dict[str, str],
    impacts: list[str],
    scenario_token: str | None,
    scenario_title: str | None,
    checkpoint_year: int,
    dpi: int,
    output_format: str,
    file_stem_prefix: str | None = None,
) -> list[Path]:
    """Write one single year uncertainty violin figure scope."""
    panel_count = len(impacts)
    layout = impact_panel_layout(impacts_count=panel_count, single_year=True)
    fig, axes = plt.subplots(
        int(layout["nrows"]),
        int(layout["ncols"]),
        figsize=(float(layout["fig_width"]), float(layout["fig_height"])),
        squeeze=False,
    )
    if panel_count == 1:
        set_footer_min_plot_height(fig, height_in=SINGLE_IMPACT_MIN_PLOT_HEIGHT_IN)
    flat_axes = list(axes.flatten())
    for axis in flat_axes[len(impacts) :]:
        axis.set_visible(False)
    summary: ViolinSummaryMode = "full"
    for axis, impact in zip(flat_axes[: len(impacts)], impacts, strict=True):
        panel = year_df.loc[year_df["impact"].astype(str).eq(str(impact))].copy()
        _render_violin_panel(
            axis=axis,
            frame=panel,
            lcia_method=lcia_method,
            impact=impact,
            checkpoint_year=checkpoint_year,
            summary=summary,
        )
        panel_title = resolve_panel_title(
            panel_title=impact_labels[impact],
            panel_count=panel_count,
        )
        axis.set_title(
            "" if panel_title is None else format_scientific_figure_text(panel_title),
            loc="left",
            pad=_PANEL_TITLE_PAD,
        )
        axis.set_ylabel(panel_impact_unit(frame=panel))
    figure_title = _title(
        family_label=family_label,
        lcia_method=lcia_method,
        group_df=year_df,
        selector_cols=selector_cols,
        year=checkpoint_year,
        prospective_scope=scenario_title,
    )
    render_figure_title(fig, figure_title)
    fig.subplots_adjust(
        hspace=_TWO_COLUMN_PANEL_HSPACE,
        wspace=0.16,
        top=title_layout_top(
            fig,
            figure_title,
            default_top=DOUBLE_COLUMN_TITLE_TOP,
            panel_title_pad=_PANEL_TITLE_PAD if len(impacts) > 1 else 0,
        ),
    )
    render_violin_summary_legend_below(fig, summary=summary)
    selector_token = selector_scope_token(
        group_frame=year_df,
        selector_cols=selector_cols,
        reference_frame=reference_frame,
    )
    out_base = figures_dir / figure_stem(
        lcia_method=lcia_method_tag(lcia_method),
        selector_scope_token=selector_token,
        scenario_token=scenario_token,
        year=checkpoint_year,
        stem_prefix=file_stem_prefix,
    )
    return _save(fig=fig, output_stem=out_base, output_format=output_format, dpi=dpi)


def _render_band_panel(
    *,
    axis,
    frame: pd.DataFrame,
    lcia_method: str,
    impact: str,
    years: list[int],
    show_x_labels: bool,
) -> None:
    ordered = frame.sort_values("year", kind="stable")
    year_values = pd.Series(pd.to_numeric(ordered["year"], errors="raise"), copy=False).astype(int)
    x = year_values.to_numpy(dtype=int)
    values = render_trajectory_band(
        axis,
        years=x,
        summaries={column: ordered[column] for column in SUMMARY_COLUMNS},
        color=_BAND_COLOR,
        line_alpha=_LINE_ALPHA,
    )
    axis.set_xlim(min(years) - 0.5, max(years) + 0.5)
    format_integer_year_axis(axis, years=years, rotation=45, ha="right")
    axis.set_xlabel("")
    if not show_x_labels:
        axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    all_values = np.concatenate([values[column] for column in values])
    axis.set_ylim(
        *require_nonnegative_figure_ylim(
            values=all_values,
            context=(
                f"IO-LCA uncertainty multi-year figure for LCIA method '{lcia_method}' "
                f"and impact '{impact}'"
            ),
        )
    )
    _format_numeric_axis(axis)


def _render_violin_panel(
    *,
    axis,
    frame: pd.DataFrame,
    lcia_method: str,
    impact: str,
    checkpoint_year: int,
    summary: ViolinSummaryMode,
) -> None:
    rows = [
        np.asarray(values, dtype=np.float64)
        for values in frame[VALUE_ARRAY_COLUMN].tolist()
        if len(values) > 0
    ]
    values = np.concatenate(rows)
    values = values[np.isfinite(values)]
    render_violin_summaries(
        axis,
        values=[values],
        positions=np.asarray([0.0], dtype=float),
        colors=[_DEFAULT_COLOR],
        summary=summary,
    )
    axis.set_xticks([])
    axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.7, 0.7)
    axis.set_ylim(
        *require_nonnegative_figure_ylim(
            values=values,
            context=(
                f"IO-LCA uncertainty single year figure for LCIA method '{lcia_method}', "
                f"impact '{impact}', year {checkpoint_year}"
            ),
        )
    )
    _format_numeric_axis(axis)


def _format_numeric_axis(axis) -> None:
    axis.set_axisbelow(True)
    axis.yaxis.set_major_formatter(FuncFormatter(scientific_tick_formatter))
    axis.grid(True, axis="y", alpha=0.25)
    axis.grid(True, axis="x", alpha=0.18)


def _title(
    *,
    family_label: str,
    lcia_method: str,
    group_df: pd.DataFrame,
    selector_cols: list[str],
    year: int | None = None,
    prospective_scope: str | None = None,
) -> str:
    impact_order, impact_labels = ordered_impacts(frame=group_df, lcia_method=lcia_method)
    panel_count = len([impact for impact in impact_order if impact in set(group_df["impact"])])
    panel_title = impact_labels[impact_order[0]] if panel_count == 1 and impact_order else None
    return build_resolved_figure_title(
        title_parts={
            "family": family_label,
            "selector_scope": resolve_selector_scope(
                frame=group_df,
                reference_frame=group_df,
                selector_columns=tuple(selector_cols),
            ),
            "lcia_method": lcia_method,
            "user_facing_override_label": None,
            "prospective_scope": prospective_scope,
        },
        year=year,
        panel_title=panel_title,
        panel_count=panel_count,
    )


def _save(*, fig, output_stem: Path, output_format: str, dpi: int) -> list[Path]:
    path = output_file_path(base_path=output_stem, output_format=output_format)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", format=output_format)
    plt.close(fig)
    return [path]
