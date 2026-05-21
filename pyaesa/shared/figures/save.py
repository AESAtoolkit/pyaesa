"""Persist figures with one shared save contract."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pyaesa.shared.figures.paths import output_paths


def create_figure(*, figsize: tuple[float, float]) -> Figure:
    """Return one unmanaged Agg-backed figure."""
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)
    return fig


def create_subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    figsize: tuple[float, float],
    squeeze: bool = True,
    subplot_kw: dict[str, object] | None = None,
) -> tuple[Figure, Any]:
    """Return one unmanaged figure together with its subplots."""
    fig = create_figure(figsize=figsize)
    axes = fig.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=squeeze,
        subplot_kw=subplot_kw,
    )
    return fig, axes


def save_figure(
    fig: Figure,
    base_path: Path,
    *,
    dpi: int,
    output_format: str,
) -> list[Path]:
    """Save a matplotlib figure to disk and close it."""
    paths = output_paths(base_path=base_path, output_format=output_format)
    for out_path in paths:
        fmt = out_path.suffix.lower().lstrip(".")
        fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight", format=fmt)
    plt.close(fig)
    return paths
