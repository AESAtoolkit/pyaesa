"""Sampling comparison figure writers for processed AR6 climate outputs."""

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .figure_io import save_figure
from .plot_budgets import plot_budgets_summary
from .plot_helpers import FIGURE_MODEL_LABEL, write_drop_csv
from .plot_sampling import (
    build_sampling_runs_until_convergence,
    build_sampling_probability_df,
)
from .figure_sampling_config import RUN_BATCH_SIZE, STABLE_CHECKS_REQUIRED
from .variable_groups import emission_mode_variable_groups, emission_variable_groups

ORDER_OF_MAGNITUDE_RATIO = 10.0
_VIOLIN_MAX_SAMPLES = 5000


def _numeric_series(values: pd.Series) -> pd.Series:
    """Return one numeric pandas Series for sampling figure calculations."""
    return pd.Series(pd.to_numeric(values, errors="raise"), copy=False)


def _ratio_series(lhs_values: pd.Series, rhs_values: pd.Series) -> pd.Series:
    """Return one sorted, non null ratio series."""
    return (
        (_numeric_series(lhs_values) / _numeric_series(rhs_values))
        .sort_values(ascending=False)
        .dropna()
    )


def _subsample_sampled_index(sampled_index: list[tuple]) -> list[tuple]:
    """Subsample rendered indices for violin plot rendering
    (to accelerate figure generation)

    The convergence loop guarantees that sampling statistics have stabilised,
    so KDE based violin plots are visually identical on a smaller subset.
    This avoids O(n) KDE evaluation and O(n) MultiIndex lookups on the full
    run list (which can reach millions of entries).
    """
    n = len(sampled_index)
    if n <= _VIOLIN_MAX_SAMPLES:
        return sampled_index
    rng = np.random.default_rng(seed=0)
    chosen = sorted(rng.choice(n, _VIOLIN_MAX_SAMPLES, replace=False))
    return [sampled_index[i] for i in chosen]


def _study_period_suffix(study_period: list[int], ext: str) -> str:
    return f"studyperiod={int(study_period[0])}to{int(study_period[1])}.{ext}"


def _apply_dense_sampling_layout(fig: Figure) -> None:
    """Apply stable manual spacing for dense sampling comparison panels."""
    fig.subplots_adjust(top=0.89, bottom=0.07, left=0.06, right=0.98, hspace=0.4, wspace=0.38)


def write_sampling_figures(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    harmonized_data: pd.DataFrame,
    all_variables_l: list[str],
    study_period: list[int],
    remaining_budget_end_year_value: int | None,
    database: str,
    categories: list[str],
    categories_repr: str,
    relative_tolerance: float,
    max_runs_per_bucket: int,
    status_callback: Callable[[str], None] | None = None,
    metadata_callback: Callable[[list[str]], None] | None = None,
    run_batch_size: int = RUN_BATCH_SIZE,
    stable_checks_required: int = STABLE_CHECKS_REQUIRED,
    build_sampling_runs_until_convergence_func: Callable = build_sampling_runs_until_convergence,
) -> pd.DataFrame:
    """Write the LHS versus SRS sampling comparison figures and return the log table."""
    tmp_proba_df = build_sampling_probability_df(harmonized_data, all_variables_l, categories)
    montecarlo_srs_index_d, montecarlo_lhs_index_d, ratio_lhs_vs_srs, convergence_log_df = (
        build_sampling_runs_until_convergence_func(
            harmonized_data=harmonized_data,
            tmp_proba_df=tmp_proba_df,
            all_variables_l=all_variables_l,
            categories=categories,
            study_period=study_period,
            remaining_budget_end_year_value=remaining_budget_end_year_value,
            relative_tolerance=relative_tolerance,
            max_runs_per_bucket=max_runs_per_bucket,
            status_callback=status_callback,
            run_batch_size=run_batch_size,
            stable_checks_required=stable_checks_required,
        )
    )

    variable_groups = emission_variable_groups(all_variables_l)
    for variable_group, group_variables in variable_groups:
        if status_callback is not None:
            status_callback(f"sampling probability ratio ({variable_group})")
        _write_sampling_probability_ratio_figure(
            figures_dir=figures_dir,
            ext=ext,
            dpi=dpi,
            out_paths=out_paths,
            tmp_proba_df=tmp_proba_df,
            all_variables_l=group_variables,
            study_period=study_period,
            database=database,
            categories_repr=categories_repr,
            variable_group=variable_group,
            metadata_callback=metadata_callback,
        )
    for variable_group, group_variables in variable_groups:
        if status_callback is not None:
            status_callback(f"sampling median ratio ({variable_group})")
        _write_sampling_median_ratio_figure(
            figures_dir=figures_dir,
            ext=ext,
            dpi=dpi,
            out_paths=out_paths,
            ratio_lhs_vs_srs=ratio_lhs_vs_srs,
            all_variables_l=group_variables,
            study_period=study_period,
            database=database,
            categories_repr=categories_repr,
            variable_group=variable_group,
            metadata_callback=metadata_callback,
        )
    for variable_group, group_variables in variable_groups:
        for emissions_mode, mode_variables in emission_mode_variable_groups(group_variables):
            if status_callback is not None:
                status_callback(f"sampling budget ({variable_group}, {emissions_mode})")
            _generate_sampling_budget_figure(
                figures_dir=figures_dir,
                ext=ext,
                dpi=dpi,
                out_paths=out_paths,
                harmonized_data=harmonized_data,
                all_variables_l=mode_variables,
                tmp_proba_df=tmp_proba_df,
                ratio_lhs_vs_srs=ratio_lhs_vs_srs,
                montecarlo_srs_index_d=montecarlo_srs_index_d,
                montecarlo_lhs_index_d=montecarlo_lhs_index_d,
                study_period=study_period,
                remaining_budget_end_year_value=remaining_budget_end_year_value,
                database=database,
                categories_repr=categories_repr,
                figure_stub=f"fig-LHSSRS-budgets-{variable_group}-{emissions_mode}",
                metadata_callback=metadata_callback,
            )
    return convergence_log_df


def _write_sampling_probability_ratio_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    tmp_proba_df: pd.DataFrame,
    all_variables_l: list[str],
    study_period: list[int],
    database: str,
    categories_repr: str,
    variable_group: str,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    fig, ax = plt.subplots(
        1,
        len(all_variables_l),
        figsize=(5 * len(all_variables_l), 4),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    ax = ax.ravel()
    for idx_var, var_sel in enumerate(all_variables_l):
        ratio_proba = _ratio_series(
            pd.Series(
                tmp_proba_df.loc[(slice(None), slice(None), var_sel), "proba_LHS"],
                copy=False,
            ),
            pd.Series(
                tmp_proba_df.loc[(slice(None), slice(None), var_sel), "proba_SRS"],
                copy=False,
            ),
        )
        if len(ratio_proba) == 0:
            continue
        ratio_proba.droplevel(2).plot.bar(ax=ax[idx_var])
        ax[idx_var].set_xticks([], minor=False)
        ax[idx_var].axhline(y=1, xmin=0, xmax=100, color="red")
        # The dashed guides mark one order of magnitude above and below parity
        # between the LHS and SRS selection probabilities.
        ax[idx_var].axhline(
            y=1 / ORDER_OF_MAGNITUDE_RATIO,
            xmin=0,
            xmax=100,
            color="red",
            linestyle="dashed",
            alpha=0.5,
        )
        ax[idx_var].axhline(
            y=ORDER_OF_MAGNITUDE_RATIO,
            xmin=0,
            xmax=100,
            color="red",
            linestyle="dashed",
            alpha=0.5,
        )
        ind_g1 = ratio_proba.loc[ratio_proba > 1].index
        ax[idx_var].text(
            x=len(ratio_proba.loc[ind_g1]),
            y=1,
            s=f"{100 * len(ratio_proba.loc[ind_g1]) / len(ratio_proba):0.1f}%",
        )
        ax[idx_var].set_ylabel("LHS/SRS probability ratio [/]")
        ax[idx_var].set_title(f"{var_sel}")
    output_path = (
        figures_dir / f"fig-LHSSRS-ratioproba-{variable_group}-{database}-MOD={FIGURE_MODEL_LABEL}"
        f"-CAT={categories_repr}-{_study_period_suffix(study_period, ext)}"
    )
    save_figure(
        fig,
        output_path,
        dpi=dpi,
        out_paths=out_paths,
        metadata_callback=metadata_callback,
    )


def _write_sampling_median_ratio_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    ratio_lhs_vs_srs: pd.DataFrame,
    all_variables_l: list[str],
    study_period: list[int],
    database: str,
    categories_repr: str,
    variable_group: str,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    fig, ax = plt.subplots(
        1,
        len(all_variables_l),
        figsize=(5 * len(all_variables_l), 4),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    ax = ax.ravel()
    fig.suptitle(
        "Overview\n"
        f"model={FIGURE_MODEL_LABEL}\n"
        f"period={int(study_period[0])}-{int(study_period[1])}\n"
        "(SRS vs LHS))",
        fontweight="bold",
    )
    for idx_var, var_sel in enumerate(all_variables_l):
        tmp_ratio = _numeric_series(
            pd.Series(
                ratio_lhs_vs_srs.loc[(var_sel, slice(None), slice(None)), "median"].droplevel(0),
                copy=False,
            )
        ).dropna()
        if len(tmp_ratio) == 0:
            continue
        tmp_ratio.plot.bar(ax=ax[idx_var], alpha=0.5)
        ax[idx_var].set_title(var_sel)
        ax[idx_var].axhline(
            y=1,
            xmin=0,
            xmax=len(tmp_ratio),
            color="red",
            linestyle="solid",
            alpha=0.5,
        )
        ax[idx_var].axhline(
            y=float(tmp_ratio.min()),
            xmin=0,
            xmax=len(tmp_ratio),
            color="orange",
            linestyle="dotted",
            alpha=0.5,
            label="min",
        )
        ax[idx_var].axhline(
            y=float(tmp_ratio.max()),
            xmin=0,
            xmax=len(tmp_ratio),
            color="orange",
            linestyle="dashed",
            alpha=0.5,
            label="max",
        )
        ax[idx_var].grid(axis="y", color="grey", alpha=0.3, zorder=-1)
        ax[idx_var].set_ylabel("LHS/SRS median ratio [/]")
    output_path = (
        figures_dir / f"fig-LHSSRS-ratiomedian-{variable_group}-{database}-MOD={FIGURE_MODEL_LABEL}"
        f"-CAT={categories_repr}-{_study_period_suffix(study_period, ext)}"
    )
    save_figure(
        fig,
        output_path,
        dpi=dpi,
        out_paths=out_paths,
        metadata_callback=metadata_callback,
    )


def _generate_sampling_budget_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    harmonized_data: pd.DataFrame,
    all_variables_l: list[str],
    tmp_proba_df: pd.DataFrame,
    ratio_lhs_vs_srs: pd.DataFrame,
    montecarlo_srs_index_d: dict[str, list[tuple]],
    montecarlo_lhs_index_d: dict[str, list[tuple]],
    study_period: list[int],
    remaining_budget_end_year_value: int | None,
    database: str,
    categories_repr: str,
    figure_stub: str,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Write one combined SRS/LHS budget figure family."""
    if not all_variables_l:
        return
    fig, ax = plt.subplots(
        3,
        2 * len(all_variables_l),
        figsize=(5 * 2 * len(all_variables_l), 11),
        squeeze=False,
    )
    fig.suptitle(
        "Overview\n"
        f"model={FIGURE_MODEL_LABEL}\n"
        f"period={int(study_period[0])}-{int(study_period[1])}\n"
        "(SRS vs LHS)",
        fontweight="bold",
    )
    _apply_dense_sampling_layout(fig)
    remaining_drop_records: list[dict] = []
    for idx_var, var_sel in enumerate(all_variables_l):
        plot_budgets_summary(
            data_df=harmonized_data.loc[_subsample_sampled_index(montecarlo_srs_index_d[var_sel])],
            var_selected=var_sel,
            timewindow_l=[int(study_period[0]), int(study_period[1])],
            ax=ax[1:, 2 * idx_var],
            remaining_budget_end_year_value=remaining_budget_end_year_value,
            remaining_budget_drop_records=remaining_drop_records,
            remaining_budget_figure_name=figure_stub,
            remaining_budget_subset_name="SRS",
        )
        ax[1, 2 * idx_var].set_title("(SRS)")
        plot_budgets_summary(
            data_df=harmonized_data.loc[_subsample_sampled_index(montecarlo_lhs_index_d[var_sel])],
            var_selected=var_sel,
            timewindow_l=[int(study_period[0]), int(study_period[1])],
            ax=ax[1:, 2 * idx_var + 1],
            remaining_budget_end_year_value=remaining_budget_end_year_value,
            remaining_budget_drop_records=remaining_drop_records,
            remaining_budget_figure_name=figure_stub,
            remaining_budget_subset_name="LHS",
        )
        ax[1, 2 * idx_var + 1].set_title("(LHS)")
        tmp_ratio = _numeric_series(
            pd.Series(
                ratio_lhs_vs_srs.loc[(var_sel, slice(None), slice(None)), "median"].droplevel(0),
                copy=False,
            )
        ).dropna()
        if len(tmp_ratio) > 0:
            tmp_ratio.plot.bar(ax=ax[0, 2 * idx_var + 1], alpha=0.5)
            ax[0, 2 * idx_var + 1].set_title(var_sel)
            ax[0, 2 * idx_var + 1].axhline(
                y=1,
                xmin=0,
                xmax=len(tmp_ratio),
                color="red",
                linestyle="solid",
                alpha=0.5,
            )
            ax[0, 2 * idx_var + 1].axhline(
                y=float(tmp_ratio.min()),
                xmin=0,
                xmax=len(tmp_ratio),
                color="orange",
                linestyle="dotted",
                alpha=0.5,
                label="min",
            )
            ax[0, 2 * idx_var + 1].axhline(
                y=float(tmp_ratio.max()),
                xmin=0,
                xmax=len(tmp_ratio),
                color="orange",
                linestyle="dashed",
                alpha=0.5,
                label="max",
            )
        ax[0, 2 * idx_var + 1].tick_params(axis="x", labelbottom=False)
        ax[0, 2 * idx_var + 1].grid(axis="y", color="grey", alpha=0.3, zorder=-1)
        ax[0, 2 * idx_var + 1].set_ylabel("LHS/SRS median ratio [/]")
        ratio_proba = _ratio_series(
            pd.Series(
                tmp_proba_df.loc[(slice(None), slice(None), var_sel), "proba_LHS"],
                copy=False,
            ),
            pd.Series(
                tmp_proba_df.loc[(slice(None), slice(None), var_sel), "proba_SRS"],
                copy=False,
            ),
        )
        if len(ratio_proba) > 0:
            ratio_proba.droplevel(2).plot.bar(ax=ax[0, 2 * idx_var])
            ax[0, 2 * idx_var].set_xticks([], minor=False)
            ax[0, 2 * idx_var].axhline(y=1, xmin=0, xmax=100, color="red")
            ax[0, 2 * idx_var].axhline(
                y=1 / ORDER_OF_MAGNITUDE_RATIO,
                xmin=0,
                xmax=100,
                color="red",
                linestyle="dashed",
                alpha=0.5,
            )
            ax[0, 2 * idx_var].axhline(
                y=ORDER_OF_MAGNITUDE_RATIO,
                xmin=0,
                xmax=100,
                color="red",
                linestyle="dashed",
                alpha=0.5,
            )
            ind_g1 = ratio_proba.loc[ratio_proba > 1].index
            ax[0, 2 * idx_var].text(
                x=len(ratio_proba.loc[ind_g1]),
                y=1,
                s=f"{100 * len(ratio_proba.loc[ind_g1]) / len(ratio_proba):0.1f}%",
            )
        ax[0, 2 * idx_var].set_ylabel("LHS/SRS probability ratio [/]")
        ax[0, 2 * idx_var].set_title(var_sel)
    output_path = (
        figures_dir / f"{figure_stub}-{database}-MOD={FIGURE_MODEL_LABEL}"
        f"-CAT={categories_repr}-{_study_period_suffix(study_period, ext)}"
    )
    save_figure(
        fig,
        output_path,
        dpi=dpi,
        out_paths=out_paths,
        metadata_callback=metadata_callback,
    )
    write_drop_csv(
        figures_dir,
        f"{output_path.stem}-remaining-budget-panel",
        remaining_drop_records,
    )
