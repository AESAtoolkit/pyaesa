"""Sequestration pathway figure writers for processed AR6 outputs."""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure_io import save_figure
from .plot_helpers import CATEGORY_COLORS, FIGURE_MODEL_LABEL, MT_TO_GT, numeric_year_columns


def _sequestration_contributions_filename(
    database: str,
    categories_repr: str,
    study_period: list[int],
    ext: str,
) -> str:
    return (
        f"fig-sequestration-contributions-{database}-MOD={FIGURE_MODEL_LABEL}"
        f"-CAT={categories_repr}-studyperiod={int(study_period[0])}"
        f"to{int(study_period[1])}.{ext}"
    )


def write_sequestration_contributions_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    original_data: pd.DataFrame,
    all_variables_l: list[str],
    categories: list[str],
    study_period: list[int],
    database: str,
    categories_repr: str,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Write the reference style sequestration contribution pathway figure."""
    years = numeric_year_columns(original_data)
    fig, ax = plt.subplots(
        len(categories),
        len(all_variables_l),
        figsize=(4 * len(all_variables_l), 3.5 * len(categories)),
        squeeze=False,
        sharey=True,
    )
    fig.suptitle(
        "Overview\n"
        f"model={FIGURE_MODEL_LABEL}\n"
        f"period={int(study_period[0])}-{int(study_period[1])}\n"
        "Carbon sequestration contributions\n",
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.88, bottom=0.07, left=0.06, right=0.99, hspace=0.6, wspace=0.35)
    x_values = np.asarray(years, dtype=float)
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    for idx_cat, curr_cat in enumerate(categories):
        for idx_var, curr_var in enumerate(all_variables_l):
            curr_ax = ax[idx_cat, idx_var]
            selected_var_df = original_data.loc[(slice(None), slice(None), curr_var), :]
            selected_cat_df = selected_var_df.loc[selected_var_df["Category"] == curr_cat, years]
            y_values = (
                selected_cat_df.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float).T
                * MT_TO_GT
            )
            curr_ax.plot(
                x_values,
                y_values,
                color=CATEGORY_COLORS.get(curr_cat, "black"),
                alpha=0.1,
            )
            curr_ax.set_xlim(x_min, x_max)
            curr_ax.text(
                x_max,
                curr_ax.get_ylim()[1],
                s=f"n={len(selected_cat_df)}",
                ha="right",
                va="top",
            )
            curr_ax.set_ylabel(f"{curr_cat}\nGtCO$_2$ yr$^{{-1}}$")
            curr_ax.set_title(curr_var)
            curr_ax.grid(axis="y", color="grey", alpha=0.3, zorder=-1)
    save_figure(
        fig,
        figures_dir
        / _sequestration_contributions_filename(database, categories_repr, study_period, ext),
        dpi=dpi,
        out_paths=out_paths,
        metadata_callback=metadata_callback,
    )
