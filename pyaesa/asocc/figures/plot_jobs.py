"""Local deterministic aSoCC figure plot job helpers."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd


def planned_plot(
    *,
    plotter: Callable[..., list[Path]],
    frame: pd.DataFrame,
    requested_years: list[int],
    output_stem: Path,
    title: str,
    dpi: int,
    output_format: str,
    group_legend: bool,
    include_impact_in_label: bool,
    include_method_in_label: bool = True,
) -> Callable[[], list[Path]]:
    """Return a deferred plot call for one planned deterministic aSoCC figure."""

    def _render() -> list[Path]:
        return plotter(
            frame=frame,
            requested_years=requested_years,
            output_stem=output_stem,
            title=title,
            dpi=dpi,
            output_format=output_format,
            group_legend=group_legend,
            include_impact_in_label=include_impact_in_label,
            include_method_in_label=include_method_in_label,
        )

    return _render
