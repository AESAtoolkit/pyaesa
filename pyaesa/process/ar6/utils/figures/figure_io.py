"""Shared figure save ownership for processed AR6 outputs."""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt


def save_figure(
    fig,
    output_path: Path,
    *,
    dpi: int,
    out_paths: list[str],
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Save one figure file and update the in memory output list."""
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    out_paths.append(str(output_path))
    if metadata_callback is not None:
        metadata_callback(out_paths)
