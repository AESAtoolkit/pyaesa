"""ASR uncertainty polar figure rendering."""

from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

from pyaesa.asr.figures.axis import ASR_LOG_SCALE, ASRScaleMode
from pyaesa.asr.figures.common import VALUE_ARRAY_COLUMN, ordered_impacts, visible_values
from pyaesa.asr.figures.frequency import FNT_FRACTION_COLUMN
from pyaesa.asr.figures.polar import PolarStyle, render_asr_polar
from pyaesa.shared.figures.lcia_scope import resolve_unique_lcia_method


def plot_polar_scope(
    *,
    frame: pd.DataFrame,
    output_stem: Path,
    title: str,
    polar_style: str,
    dpi: int,
    output_format: str,
    scale_mode: ASRScaleMode = ASR_LOG_SCALE,
) -> list[Path]:
    """Render one ASR uncertainty polar checkpoint from selected run arrays."""
    impacts = ordered_impacts(frame)
    values: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float]] = {}
    frequencies: dict[str, float] = {}
    for impact in impacts:
        panel = frame.loc[frame["impact"].astype(str).eq(str(impact))]
        arrays = [np.asarray(value, dtype=np.float64) for value in panel[VALUE_ARRAY_COLUMN]]
        payload = np.concatenate(arrays)
        values[str(impact)] = payload
        summaries[str(impact)] = _summary_for_panel(panel)
        frequencies[str(impact)] = summaries[str(impact)].get(FNT_FRACTION_COLUMN, np.nan)
    lcia_method = str(resolve_unique_lcia_method(frame) or visible_values(frame, "lcia_method")[0])
    return render_asr_polar(
        frame=frame,
        values=values,
        summaries=summaries,
        frequencies=frequencies,
        output_stem=output_stem,
        title=title,
        lcia_method=lcia_method,
        style=cast(PolarStyle, str(polar_style)),
        scale_mode=scale_mode,
        dpi=dpi,
        output_format=output_format,
    )


def _summary_for_panel(panel: pd.DataFrame) -> dict[str, float]:
    row = panel.iloc[0]
    columns = (
        "mean",
        "std",
        "min",
        "p5",
        "p25",
        "median",
        "p75",
        "p95",
        "max",
        FNT_FRACTION_COLUMN,
    )
    return {column: float(row[column]) for column in columns if column in panel.columns}
