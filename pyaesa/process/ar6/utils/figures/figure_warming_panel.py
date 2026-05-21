"""Warming distribution figure writer for processed AR6 climate outputs."""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure_io import save_figure
from .plot_helpers import (
    CATEGORY_COLORS,
    FIGURE_MODEL_LABEL,
    WARMING_METADATA_COLUMN,
    plot_violin,
)

SSP_POSITION_DIVISOR = 8


def _numeric_series(values: pd.Series) -> pd.Series:
    """Return one numeric pandas Series for warming figure inputs."""
    return pd.Series(pd.to_numeric(values, errors="raise"), copy=False)


def _median_warming_filename(
    database: str,
    categories_repr: str,
    study_period: list[int],
    ext: str,
) -> str:
    return (
        f"fig-median-warming-{database}-MOD={FIGURE_MODEL_LABEL}-CAT={categories_repr}"
        f"-studyperiod={int(study_period[0])}to{int(study_period[1])}.{ext}"
    )


def write_median_warming_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    scenario_rows_df: pd.DataFrame,
    source_metadata: pd.DataFrame,
    categories: list[str],
    study_period: list[int],
    database: str,
    categories_repr: str,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Write the scenario warming distribution figure."""
    if source_metadata.empty:
        raise RuntimeError(
            "AR6 source metadata is empty, so the warming distribution figure cannot be built."
        )
    if WARMING_METADATA_COLUMN not in source_metadata.columns:
        raise RuntimeError(
            "AR6 source metadata is missing the warming column required for the "
            "warming distribution figure."
        )
    fig, ax = plt.subplots(1, 1, figsize=(5, 4), layout="constrained")
    fig.suptitle(
        "Overview\n"
        f"model={FIGURE_MODEL_LABEL}\n"
        f"period={int(study_period[0])}-{int(study_period[1])}\n",
        fontweight="bold",
    )
    all_ssps_l = (
        list(sorted(set(scenario_rows_df["Ssp_family"]))) if not scenario_rows_df.empty else []
    )
    for idx_cat, curr_cat in enumerate(categories):
        index_sel = scenario_rows_df.loc[
            (scenario_rows_df["Category"] == curr_cat),
            :,
        ].index
        index_sel = index_sel[index_sel.isin(source_metadata.index)]
        if len(index_sel) > 0:
            warming_values = _numeric_series(
                pd.Series(source_metadata.loc[index_sel, WARMING_METADATA_COLUMN], copy=False)
            ).dropna()
            if len(warming_values) > 0:
                parts = ax.violinplot(
                    warming_values.to_numpy(dtype=float),
                    positions=[idx_cat],
                    widths=1 / (len(all_ssps_l) + 3),
                    showextrema=False,
                    showmedians=False,
                )
                plot_violin(parts, CATEGORY_COLORS.get(curr_cat, "black"), 0.8)
                ax.scatter(
                    x=[idx_cat],
                    y=float(np.median(warming_values.to_numpy(dtype=float))),
                    c=CATEGORY_COLORS.get(curr_cat, "black"),
                    marker=".",
                    edgecolor="black",
                )
        for idx_ssp, curr_ssp in enumerate(all_ssps_l):
            x_offset_ssp = 0.2
            index_sel = scenario_rows_df.loc[
                (scenario_rows_df["Ssp_family"] == curr_ssp)
                & (scenario_rows_df["Category"] == curr_cat),
                :,
            ].index
            index_sel = index_sel[index_sel.isin(source_metadata.index)]
            if len(index_sel) == 0:
                continue
            warming_values = _numeric_series(
                pd.Series(source_metadata.loc[index_sel, WARMING_METADATA_COLUMN], copy=False)
            ).dropna()
            if len(warming_values) == 0:
                continue
            # This fixed offset keeps the SSP specific violins visually
            # separated inside the single warming panel.
            position = x_offset_ssp + idx_cat + idx_ssp / SSP_POSITION_DIVISOR
            parts = ax.violinplot(
                warming_values.to_numpy(dtype=float),
                positions=[position],
                widths=1 / (len(all_ssps_l) + 3),
                showextrema=False,
                showmedians=False,
            )
            plot_violin(parts, CATEGORY_COLORS.get(curr_cat, "black"), 0.5)
            ax.scatter(
                x=[x_offset_ssp + idx_cat + idx_ssp / (len(all_ssps_l) + 3)],
                y=float(np.median(warming_values.to_numpy(dtype=float))),
                c=CATEGORY_COLORS.get(curr_cat, "black"),
                marker=".",
                edgecolor="black",
            )
            ax.text(
                x_offset_ssp + idx_cat + idx_ssp / (len(all_ssps_l) + 3),
                ax.get_ylim()[0],
                f"SSP{curr_ssp} (n={len(index_sel.drop_duplicates())})",
                rotation=90,
                color="black",
                fontsize=6,
            )
    ax.set_title(WARMING_METADATA_COLUMN)
    ax.set_ylabel(f"{WARMING_METADATA_COLUMN}\n[degree C]")
    ax.set_xticks([idx_cat for idx_cat, _cat in enumerate(categories)], list(categories))
    ax.grid(axis="y", color="grey", alpha=0.3, zorder=-1)
    save_figure(
        fig,
        figures_dir / _median_warming_filename(database, categories_repr, study_period, ext),
        dpi=dpi,
        out_paths=out_paths,
        metadata_callback=metadata_callback,
    )
