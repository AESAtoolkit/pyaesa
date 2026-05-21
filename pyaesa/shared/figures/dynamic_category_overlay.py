"""Shared dynamic category overlay rendering for aCC/ASR figures."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

from pyaesa.shared.figures.contracts import DETERMINISTIC_PROSPECTIVE_COLUMNS
from .selector_slices import selector_slices
from .figure_footer import render_below_figure_legend
from .layout import DOUBLE_COLUMN_TITLE_TOP
from .lcia_metadata import resolve_frame_impact_title
from .lcia_scope import lcia_method_slices
from .multi_year_transitions import render_transition_markers
from .nonnegative_axis import require_nonnegative_figure_ylim
from pyaesa.shared.figures.paths import scope_filename_stem
from .save import save_figure
from .scientific_text import format_scientific_figure_text
from .title_contract import (
    SelectorScopeRequest,
    build_resolved_figure_title,
    prospective_scope_slices,
    resolve_panel_title,
)
from .titles import render_figure_title, title_layout_top
from .variant_selection import split_variant_frames, variant_footer_note
from pyaesa.shared.tabular.scalars import sanitize_token

_MARKER_COLUMNS = {
    "__transition_marker_year",
    "__transition_marker_label",
    "__transition_marker_color",
}
_TRANSITION_METADATA_COLUMNS = {"asocc_ssp_start_year", "lca_ssp_start_year"}


def render_dynamic_category_overlay(
    *,
    prepared: pd.DataFrame,
    requested_years: list[int],
    output_base: Path,
    family: str,
    user_facing_override_label: str | None,
    ylabel: str,
    dpi: int,
    output_format: str,
    meta_columns: set[str],
    marker_resolver: Callable[[pd.DataFrame], list],
    axis_styler: Callable[[object, pd.DataFrame], None] | None = None,
    ylabel_resolver: Callable[[pd.DataFrame], str] | None = None,
    selector_scope_request: SelectorScopeRequest | None = None,
) -> list[Path]:
    """Render category bands/means over multi-year dynamic deterministic trajectories."""
    paths: list[Path] = []
    for _lcia_token, _lcia_title, lcia_frame, lcia_method in lcia_method_slices(prepared):
        lcia_output = output_base.parent / scope_filename_stem(
            base_stem=output_base.name,
            lcia_method=lcia_method,
        )
        for selector_token, selector_title, selector_frame in selector_slices(
            lcia_frame,
            selector_scope_request=selector_scope_request,
        ):
            selector_output = (
                lcia_output
                if selector_token == "all"
                else lcia_output.parent / f"{lcia_output.name}__{selector_token}"
            )
            for scenario_token, scenario_title, scenario_frame in prospective_scope_slices(
                selector_frame
            ):
                scoped_output = (
                    selector_output
                    if scenario_token == "all"
                    else selector_output.parent / f"{selector_output.name}__{scenario_token}"
                )
                paths.extend(
                    _render_selector_overlay(
                        prepared=scenario_frame,
                        requested_years=requested_years,
                        output_base=scoped_output,
                        title_parts={
                            "family": family,
                            "selector_scope": selector_title or None,
                            "lcia_method": lcia_method,
                            "user_facing_override_label": user_facing_override_label,
                            "prospective_scope": scenario_title,
                        },
                        ylabel=ylabel,
                        dpi=dpi,
                        output_format=output_format,
                        meta_columns=meta_columns,
                        marker_resolver=marker_resolver,
                        axis_styler=axis_styler,
                        ylabel_resolver=ylabel_resolver,
                    )
                )
    return paths


def _render_selector_overlay(
    *,
    prepared: pd.DataFrame,
    requested_years: list[int],
    output_base: Path,
    title_parts: dict[str, str | None],
    ylabel: str,
    dpi: int,
    output_format: str,
    meta_columns: set[str],
    marker_resolver: Callable[[pd.DataFrame], list],
    axis_styler: Callable[[object, pd.DataFrame], None] | None = None,
    ylabel_resolver: Callable[[pd.DataFrame], str] | None = None,
) -> list[Path]:
    """Render one selector scoped category overlay family."""
    prepared, compressions = split_variant_frames(
        frame=prepared,
        requested_years=requested_years,
    )
    footer_note = variant_footer_note(compressions, average_over_years=True)
    panel_columns = [
        column
        for column in prepared.columns
        if column
        not in {
            "year",
            "value",
            "cc_category",
            "lcia_method",
            "impact",
            "impact_unit",
            "fu_code",
            "r_f",
            "r_c",
            "r_p",
            "s_p",
            "reference_year",
            "l2_reuse_year",
            "series_label",
            "l1_l2_method",
            "l1_method",
            "l2_method",
            *DETERMINISTIC_PROSPECTIVE_COLUMNS,
            *meta_columns,
            *_TRANSITION_METADATA_COLUMNS,
            *_MARKER_COLUMNS,
        }
    ]
    grouped_panels = (
        [(None, prepared)]
        if not panel_columns
        else list(prepared.groupby(panel_columns, dropna=False, sort=True))
    )
    panel_count = len(grouped_panels)
    expected_years = sorted({int(year) for year in requested_years})
    x_values = np.asarray(expected_years, dtype=float)
    paths: list[Path] = []
    for _panel_key, panel_frame in grouped_panels:
        fig, axis = plt.subplots(figsize=(10.0, 5.8))
        if axis_styler is not None:
            axis_styler(axis, panel_frame)
        render_transition_markers(axis, markers=marker_resolver(panel_frame))
        impact_title = resolve_frame_impact_title(panel_frame)
        panel_title = impact_title
        categories = sorted(
            {str(value) for value in panel_frame["cc_category"].dropna().astype(str).tolist()}
        )
        colors = colormaps["tab10"](np.linspace(0.0, 1.0, num=max(1, len(categories))))
        color_map = {category: colors[index] for index, category in enumerate(categories)}
        for category in categories:
            category_frame = panel_frame.loc[
                panel_frame["cc_category"].astype(str) == str(category)
            ].copy()
            trajectory_columns = [
                column
                for column in category_frame.columns
                if column
                not in {
                    "year",
                    "value",
                    "cc_category",
                    "fu_code",
                    "series_label",
                    *DETERMINISTIC_PROSPECTIVE_COLUMNS,
                    *_TRANSITION_METADATA_COLUMNS,
                    *_MARKER_COLUMNS,
                }
            ]
            groups = (
                [(None, category_frame)]
                if not trajectory_columns
                else list(category_frame.groupby(trajectory_columns, dropna=False, sort=True))
            )
            trajectories: list[np.ndarray] = []
            for _trajectory_key, trajectory_frame in groups:
                ordered = trajectory_frame.sort_values("year", kind="stable")
                trajectories.append(np.asarray(ordered["value"].tolist(), dtype=float))
            stacked = np.vstack(trajectories)
            axis.fill_between(
                x_values,
                np.nanmin(stacked, axis=0),
                np.nanmax(stacked, axis=0),
                color=color_map[category],
                alpha=0.16,
            )
            axis.plot(
                x_values,
                np.nanmean(stacked, axis=0),
                color=color_map[category],
                linewidth=1.8,
                label=f"{category} (n={stacked.shape[0]})",
            )
        axis.set_ylim(
            *require_nonnegative_figure_ylim(
                values=np.asarray(panel_frame["value"].tolist(), dtype=float),
                context=build_resolved_figure_title(
                    title_parts=title_parts,
                    panel_title=panel_title,
                    panel_count=panel_count,
                ),
            )
        )
        axis.set_xlabel("")
        axis.set_ylabel(
            format_scientific_figure_text(
                ylabel if ylabel_resolver is None else ylabel_resolver(panel_frame)
            )
        )
        axis.grid(alpha=0.25)
        render_below_figure_legend(fig, legend_note=footer_note)
        resolved_panel_title = resolve_panel_title(
            panel_title=panel_title,
            panel_count=panel_count,
        )
        if resolved_panel_title is not None:
            axis.set_title(format_scientific_figure_text(resolved_panel_title), loc="left")
        figure_title = build_resolved_figure_title(
            title_parts=title_parts,
            panel_title=panel_title,
            panel_count=panel_count,
        )
        render_figure_title(fig, figure_title)
        fig.subplots_adjust(
            top=title_layout_top(
                fig,
                figure_title,
                default_top=DOUBLE_COLUMN_TITLE_TOP,
                panel_title_pad=5 if resolved_panel_title is not None else 0,
            )
        )
        paths.extend(
            save_figure(
                fig,
                output_base
                if not panel_title
                else output_base.parent / f"{output_base.name}__{sanitize_token(panel_title)}",
                dpi=dpi,
                output_format=output_format,
            )
        )
    return paths
