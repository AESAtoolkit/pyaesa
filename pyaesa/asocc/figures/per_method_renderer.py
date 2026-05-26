"""Lean per method deterministic aSoCC figure renderer."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pandas as pd

from pyaesa.shared.figures.jobs import PlannedFigureJob
from pyaesa.shared.figures.lcia_metadata import lcia_title_parts
from pyaesa.shared.figures.scenario_scopes import preplanned_scenario_scope_slices
from pyaesa.shared.figures.selector_slices import selector_slices
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.scalars import is_display_missing

from .file_stems import asocc_scope_stem, visible_scope_values
from .plot_jobs import planned_plot
from .transition_planner import transition_year


def plan_per_method_jobs(
    *,
    rows: pd.DataFrame,
    figures_root: Path,
    requested_years: list[int],
    dpi: int,
    output_format: str,
    plotter: Callable[..., list[Path]],
    row_preparer: Callable[[pd.DataFrame], pd.DataFrame],
) -> Iterator[PlannedFigureJob]:
    """Plan one multi-year or single year product per method and SSP scope."""
    methods = sorted(
        {
            str(value).strip()
            for value in rows["__method"].tolist()
            if not is_display_missing(value) and str(value).strip()
        }
    )
    for method in methods:
        method_rows = rows.loc[rows["__method"].astype(str).eq(str(method))].copy()
        for scoped_rows in figure_ssp_slices(method_rows):
            for lcia_rows in lcia_slices(scoped_rows):
                for selector_token, selector_title, selector_rows in selector_slices(lcia_rows):
                    prepared_rows = row_preparer(selector_rows)
                    output_base = (
                        Path(figures_root)
                        / "per_method"
                        / scoped_stem(
                            method,
                            prepared_rows,
                            selector_token=selector_token,
                            studied_year=single_requested_year(requested_years),
                        )
                    )
                    title = scope_title(
                        "aSoCC",
                        method,
                        prepared_rows,
                        selector_title=selector_title,
                        include_impact=True,
                        studied_year=single_requested_year(requested_years),
                    )
                    yield PlannedFigureJob(
                        kind="per_method",
                        label=output_base.name,
                        render=planned_plot(
                            plotter=plotter,
                            frame=prepared_rows,
                            requested_years=requested_years,
                            output_stem=output_base,
                            title=title,
                            dpi=dpi,
                            output_format=output_format,
                            group_legend=False,
                            include_impact_in_label=True,
                            include_method_in_label=False,
                        ),
                    )


def visible_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Return sorted nonmissing display values for one column."""
    return visible_scope_values(frame, column)


def figure_ssp_slices(frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield final SSP figure scopes while keeping row owned SSP metadata."""
    return preplanned_scenario_scope_slices(
        frame,
        scenario_column=ASOCC_SSP_SCENARIO_COLUMN,
        scope_column="__figure_ssp_scope",
        identity_excluded_columns={"asocc"},
    )


def lcia_slices(frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield per method LCIA method figure scopes."""
    values = visible_values(frame, "lcia_method")
    if not values:
        yield frame.copy()
        return
    for value in values:
        yield frame.loc[frame["lcia_method"].astype(str).eq(value)].copy()


def scoped_stem(
    label: str,
    frame: pd.DataFrame,
    *,
    include_impact: bool = False,
    selector_token: str = "all",
    studied_year: int | None = None,
) -> str:
    """Return one deterministic file stem with the final SSP scope."""
    return asocc_scope_stem(
        label,
        frame,
        include_impact=include_impact,
        selector_token=selector_token,
        studied_year=studied_year,
    )


def scope_title(
    family: str,
    label: str | None,
    frame: pd.DataFrame,
    *,
    include_impact: bool = False,
    selector_title: str | None = None,
    studied_year: int | None = None,
) -> str:
    """Return a compact deterministic figure title."""
    parts = [family]
    if label is not None:
        parts.append(label)
    if selector_title is not None and str(selector_title).strip():
        parts.append(str(selector_title).strip())
    parts.extend(lcia_title_parts(frame, include_impact=include_impact))
    if studied_year is not None:
        parts.append(str(int(studied_year)))
    ssp_values = visible_values(frame, ASOCC_SSP_SCENARIO_COLUMN)
    if ssp_values:
        parts.append(f"Prospective: {ssp_values[0]}")
    return " | ".join(parts)


def single_requested_year(requested_years: list[int]) -> int | None:
    """Return the studied year when the figure request is a single year scope."""
    years = sorted({int(year) for year in requested_years})
    if len(years) != 1:
        return None
    return years[0]


def series_transition(group: pd.DataFrame) -> int | None:
    """Return the visible transition year for one plotted series."""
    return transition_year(group)
