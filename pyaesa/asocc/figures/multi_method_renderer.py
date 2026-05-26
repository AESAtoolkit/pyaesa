"""Lean multi-method deterministic aSoCC figure renderer."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

from pyaesa.shared.figures.jobs import PlannedFigureJob
from pyaesa.shared.figures.lcia_scopes import lcia_impact_slices
from pyaesa.shared.figures.selector_slices import selector_slices

from .per_method_renderer import (
    figure_ssp_slices,
    lcia_slices,
    scoped_stem,
    scope_title,
    single_requested_year,
    visible_values,
)
from .plot_jobs import planned_plot


def plan_multi_method_jobs(
    *,
    rows: pd.DataFrame,
    figures_root: Path,
    requested_years: list[int],
    dpi: int,
    output_format: str,
    plotter: Callable[..., list[Path]],
    row_preparer: Callable[[pd.DataFrame], pd.DataFrame],
) -> Iterator[PlannedFigureJob]:
    """Plan method comparison products before per method products."""
    if not has_multiple_methods(rows):
        return
    studied_year = single_requested_year(requested_years)
    single_year = studied_year is not None
    for scoped_rows in figure_ssp_slices(expand_generic_lcia_rows(rows)):
        for lcia_scope in lcia_slices(scoped_rows):
            for selector_token, selector_title, selector_rows in selector_slices(lcia_scope):
                prepared = row_preparer(selector_rows)
                product_scopes = [prepared] if single_year else lcia_impact_slices(prepared)
                include_impact = not single_year and len(visible_values(prepared, "impact")) > 1
                for impact_rows in product_scopes:
                    yield _multi_method_job(
                        impact_rows=impact_rows,
                        figures_root=figures_root,
                        requested_years=requested_years,
                        studied_year=studied_year,
                        dpi=dpi,
                        output_format=output_format,
                        plotter=plotter,
                        include_impact=include_impact,
                        selector_token=selector_token,
                        selector_title=selector_title,
                    )


def _multi_method_job(
    *,
    impact_rows: pd.DataFrame,
    figures_root: Path,
    requested_years: list[int],
    studied_year: int | None,
    dpi: int,
    output_format: str,
    plotter: Callable[..., list[Path]],
    include_impact: bool,
    selector_token: str,
    selector_title: str,
) -> PlannedFigureJob:
    output_base = (
        Path(figures_root)
        / "multi_method"
        / scoped_stem(
            "multi_method",
            impact_rows,
            include_impact=include_impact,
            selector_token=selector_token,
            studied_year=studied_year,
        )
    )
    title = scope_title(
        "aSoCC",
        None,
        impact_rows,
        include_impact=include_impact,
        selector_title=selector_title,
        studied_year=studied_year,
    )
    return PlannedFigureJob(
        kind="multi_method",
        label=output_base.name,
        render=planned_plot(
            plotter=plotter,
            frame=impact_rows,
            requested_years=requested_years,
            output_stem=output_base,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=True,
            include_impact_in_label=False,
        ),
    )


def has_multiple_methods(rows: pd.DataFrame) -> bool:
    """Return whether one figure scope contains more than one method."""
    return rows["__method"].dropna().astype(str).drop_duplicates().size > 1


def expand_generic_lcia_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Repeat LCIA generic rows into visible LCIA comparison scopes."""
    if frame.empty or "lcia_method" not in frame.columns or "impact" not in frame.columns:
        return frame
    method_series = frame["lcia_method"]
    impact_series = frame["impact"]
    generic_mask = method_series.map(pd.isna) & impact_series.map(pd.isna)
    generic = frame.loc[generic_mask].copy()
    scoped = frame.loc[~generic_mask].copy()
    if generic.empty or scoped.empty:
        return frame
    repeated = []
    scopes = scoped.loc[:, ["lcia_method", "impact"]].drop_duplicates()
    for lcia_method, impact in scopes.itertuples(index=False, name=None):
        copy = generic.copy()
        copy["lcia_method"] = lcia_method
        copy["impact"] = impact
        repeated.append(copy)
    return pd.concat([scoped, *repeated], ignore_index=True)
