"""Figure rendering ownership for IO-LCA plotting."""

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
    lca_prospective_scope_slices,
    lca_transition_markers,
    lcia_method_tag,
    normalize_plot_years,
    ordered_impacts,
    panel_impact_unit,
    selector_groups,
    selector_scope_token,
)
from pyaesa.shared.figures.colors import DEFAULT_SINGLE_SERIES_COLOR
from pyaesa.shared.figures.layout import (
    DOUBLE_COLUMN_TITLE_TOP,
    TRANSITION_PANEL_TITLE_PAD,
    bottom_panel_indices,
    format_integer_year_axis,
    show_panel_x_labels,
)
from pyaesa.shared.figures.multi_year_transitions import (
    render_transition_markers,
    transition_title_pad,
)
from pyaesa.shared.figures.nonnegative_axis import require_nonnegative_figure_ylim
from pyaesa.shared.figures.paths import output_paths
from pyaesa.shared.figures.request_validation import validate_consecutive_multi_year_figure_request
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.titles import render_figure_title, title_layout_top
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.figures.title_contract import (
    SelectorScopeRequest,
    build_resolved_figure_title,
    resolve_panel_title,
    resolve_selector_scope,
)

_PANEL_TITLE_PAD = 5
_TWO_COLUMN_PANEL_HSPACE = 0.32
_TWO_COLUMN_TRANSITION_HSPACE = 0.42
_TWO_COLUMN_PANEL_WSPACE = 0.16


def write_lcia_method_figures(
    *,
    lcia_method_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    figures_dir: Path,
    lcia_method: str,
    dpi: int,
    output_format: str,
    family_label: str = "IO-LCA",
    selector_columns: tuple[str, ...] | None = None,
    selector_scope_request: SelectorScopeRequest | None = None,
    file_stem_prefix: str | None = None,
) -> list[Path]:
    """Write deterministic figures for one LCIA method table."""
    selector_cols, groups = selector_groups(
        frame=lcia_method_frame,
        selector_columns=selector_columns,
    )

    impact_order, impact_labels = ordered_impacts(
        frame=lcia_method_frame,
        lcia_method=lcia_method,
    )
    all_paths: list[Path] = []
    for _group_key, base_group_df in groups:
        base_group_df = normalize_plot_years(frame=base_group_df.copy())
        for scenario_token, scenario_title, group_df in lca_prospective_scope_slices(base_group_df):
            years = sorted({int(year) for year in group_df["year"].tolist()})
            validate_consecutive_multi_year_figure_request(
                requested_years=years,
                family_label=family_label,
            )
            available_impacts = [
                impact for impact in impact_order if impact in set(group_df["impact"])
            ]
            panel_count = len(available_impacts)
            layout = impact_panel_layout(impacts_count=len(available_impacts))
            fig, axes = plt.subplots(
                int(layout["nrows"]),
                int(layout["ncols"]),
                figsize=(float(layout["fig_width"]), float(layout["fig_height"])),
                squeeze=False,
            )
            flat_axes = list(axes.flatten())
            for ax in flat_axes[len(available_impacts) :]:
                ax.set_visible(False)
            ncols = int(layout["ncols"])
            bottom_indices = bottom_panel_indices(panel_count=panel_count, ncols=ncols)
            has_transition = False
            for panel_index, (ax, impact) in enumerate(
                zip(flat_axes[: len(available_impacts)], available_impacts, strict=True)
            ):
                sub = group_df.loc[group_df["impact"].astype(str).eq(str(impact))].copy()
                markers = lca_transition_markers(sub)
                has_transition = has_transition or bool(markers)
                _render_lca_line_panel(
                    axis=ax,
                    frame=sub,
                    years=years,
                    show_x_labels=show_panel_x_labels(
                        panel_index=panel_index,
                        bottom_indices=bottom_indices,
                    ),
                    context=(
                        f"IO-LCA multi-year figure for LCIA method '{lcia_method}' and impact "
                        f"'{impact}'"
                    ),
                )
                render_transition_markers(ax, markers=markers)
                ax.set_ylabel(panel_impact_unit(frame=sub))
                panel_title = resolve_panel_title(
                    panel_title=impact_labels[impact],
                    panel_count=panel_count,
                )
                ax.set_title(
                    "" if panel_title is None else format_scientific_figure_text(panel_title),
                    loc="left",
                    pad=transition_title_pad(
                        markers,
                        no_transition=_PANEL_TITLE_PAD,
                        single_transition=TRANSITION_PANEL_TITLE_PAD,
                        component_transition=TRANSITION_PANEL_TITLE_PAD,
                    ),
                )
                ax.grid(True, axis="y", alpha=0.25)
                ax.grid(True, axis="x", alpha=0.18)
            figure_title = build_resolved_figure_title(
                title_parts={
                    "family": family_label,
                    "selector_scope": resolve_selector_scope(
                        frame=group_df,
                        reference_frame=reference_frame,
                        selector_columns=tuple(selector_cols),
                        selector_scope_request=selector_scope_request,
                    ),
                    "lcia_method": lcia_method,
                    "user_facing_override_label": None,
                    "prospective_scope": scenario_title,
                },
                panel_title=impact_labels[available_impacts[0]] if panel_count == 1 else None,
                panel_count=panel_count,
            )
            render_figure_title(fig, figure_title)
            fig.subplots_adjust(
                hspace=_TWO_COLUMN_TRANSITION_HSPACE
                if has_transition
                else _TWO_COLUMN_PANEL_HSPACE,
                wspace=_TWO_COLUMN_PANEL_WSPACE,
                top=title_layout_top(
                    fig,
                    figure_title,
                    default_top=DOUBLE_COLUMN_TITLE_TOP,
                    panel_title_pad=TRANSITION_PANEL_TITLE_PAD
                    if has_transition
                    else (_PANEL_TITLE_PAD if panel_count > 1 else 0),
                ),
            )
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
            out_paths = output_paths(base_path=out_base, output_format=output_format)
            for out_path in out_paths:
                out_path = ensure_file_parent(out_path)
                fmt = out_path.suffix.lower().lstrip(".")
                fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", format=fmt)
            plt.close(fig)
            all_paths.extend(out_paths)
    return all_paths


def write_lcia_method_checkpoint_figures(
    *,
    lcia_method_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    figures_dir: Path,
    lcia_method: str,
    checkpoint_years: list[int],
    dpi: int,
    output_format: str,
    family_label: str = "IO-LCA",
    selector_columns: tuple[str, ...] | None = None,
    selector_scope_request: SelectorScopeRequest | None = None,
    file_stem_prefix: str | None = None,
) -> list[Path]:
    """Write one single year checkpoint figure set per requested year and LCIA method."""
    selector_cols, groups = selector_groups(
        frame=lcia_method_frame,
        selector_columns=selector_columns,
    )
    impact_order, impact_labels = ordered_impacts(
        frame=lcia_method_frame,
        lcia_method=lcia_method,
    )
    all_paths: list[Path] = []
    for _group_key, base_group_df in groups:
        base_group_df = normalize_plot_years(frame=base_group_df.copy())
        for scenario_token, scenario_title, group_df in lca_prospective_scope_slices(base_group_df):
            for checkpoint_year in [int(year) for year in checkpoint_years]:
                year_df = group_df.loc[group_df["year"].astype(int).eq(int(checkpoint_year))].copy()
                impacts = [impact for impact in impact_order if impact in set(year_df["impact"])]
                panel_count = len(impacts)
                layout = impact_panel_layout(impacts_count=len(impacts), single_year=True)
                fig, axes = plt.subplots(
                    int(layout["nrows"]),
                    int(layout["ncols"]),
                    figsize=(float(layout["fig_width"]), float(layout["fig_height"])),
                    squeeze=False,
                )
                flat_axes = list(axes.flatten())
                for ax in flat_axes[len(impacts) :]:
                    ax.set_visible(False)
                for ax, impact in zip(flat_axes[: len(impacts)], impacts, strict=True):
                    impact_df = year_df.loc[year_df["impact"].astype(str).eq(str(impact))].copy()
                    values = np.asarray(
                        pd.to_numeric(impact_df["lca_value"], errors="raise"),
                        dtype=float,
                    )
                    ax.bar(
                        [0.0],
                        values[:1],
                        color=DEFAULT_SINGLE_SERIES_COLOR,
                        alpha=0.85,
                        zorder=2,
                    )
                    ax.set_xticks([])
                    ax.tick_params(axis="x", length=0)
                    ax.set_ylabel(panel_impact_unit(frame=impact_df))
                    ax.set_ylim(
                        *require_nonnegative_figure_ylim(
                            values=values,
                            context=(
                                f"IO-LCA single-year figure for LCIA method '{lcia_method}', "
                                f"impact '{impact}', year {checkpoint_year}"
                            ),
                        )
                    )
                    _format_lca_axis(ax)
                    panel_title = resolve_panel_title(
                        panel_title=impact_labels[impact],
                        panel_count=panel_count,
                    )
                    ax.set_title(
                        "" if panel_title is None else format_scientific_figure_text(panel_title),
                        loc="left",
                        pad=_PANEL_TITLE_PAD,
                    )
                    ax.grid(True, axis="y", alpha=0.25)
                figure_title = build_resolved_figure_title(
                    title_parts={
                        "family": family_label,
                        "selector_scope": resolve_selector_scope(
                            frame=group_df,
                            reference_frame=reference_frame,
                            selector_columns=tuple(selector_cols),
                            selector_scope_request=selector_scope_request,
                        ),
                        "lcia_method": lcia_method,
                        "user_facing_override_label": None,
                        "prospective_scope": scenario_title,
                    },
                    year=int(checkpoint_year),
                    panel_title=impact_labels[impacts[0]] if panel_count == 1 else None,
                    panel_count=panel_count,
                )
                render_figure_title(fig, figure_title)
                fig.subplots_adjust(
                    hspace=_TWO_COLUMN_PANEL_HSPACE,
                    wspace=_TWO_COLUMN_PANEL_WSPACE,
                    top=title_layout_top(
                        fig,
                        figure_title,
                        default_top=DOUBLE_COLUMN_TITLE_TOP,
                        panel_title_pad=_PANEL_TITLE_PAD if panel_count > 1 else 0,
                    ),
                )
                selector_token = selector_scope_token(
                    group_frame=group_df,
                    selector_cols=selector_cols,
                    reference_frame=reference_frame,
                )
                out_base = figures_dir / figure_stem(
                    lcia_method=lcia_method_tag(lcia_method),
                    selector_scope_token=selector_token,
                    scenario_token=scenario_token,
                    year=int(checkpoint_year),
                    stem_prefix=file_stem_prefix,
                )
                out_paths = output_paths(base_path=out_base, output_format=output_format)
                for out_path in out_paths:
                    out_path = ensure_file_parent(out_path)
                    fmt = out_path.suffix.lower().lstrip(".")
                    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", format=fmt)
                plt.close(fig)
                all_paths.extend(out_paths)
    return all_paths


def _render_lca_line_panel(
    *,
    axis,
    frame: pd.DataFrame,
    years: list[int],
    show_x_labels: bool,
    context: str,
) -> None:
    year_values = np.asarray(frame["year"], dtype=int)
    order = np.argsort(year_values)
    x = year_values[order]
    y = np.asarray(pd.to_numeric(frame.iloc[order]["lca_value"], errors="raise"), dtype=float)
    axis.plot(x, y, color=DEFAULT_SINGLE_SERIES_COLOR, linewidth=1.9, alpha=0.95)
    axis.set_xlim(min(years) - 0.5, max(years) + 0.5)
    format_integer_year_axis(axis, years=years, rotation=45, ha="right")
    axis.set_xlabel("")
    if not show_x_labels:
        axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    axis.set_ylim(*require_nonnegative_figure_ylim(values=y, context=context))
    _format_lca_axis(axis)


def _format_lca_axis(axis) -> None:
    axis.set_axisbelow(True)
    axis.yaxis.set_major_formatter(FuncFormatter(scientific_tick_formatter))
