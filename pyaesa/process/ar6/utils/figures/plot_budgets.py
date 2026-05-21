"""Budget and pathway plots for AR6 figures."""

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .plot_helpers import (
    CATEGORY_COLORS,
    MT_TO_GT,
    append_remaining_budget_drop_records,
    max_year,
    plot_violin,
    remaining_budget_end_year,
    var_df,
    year_slice,
)


def _is_co2_unit_variable(var_selected: str) -> bool:
    """Return whether a processed AR6 variable should be shown in CO2 units."""
    return "CO2" in var_selected or var_selected.startswith("Carbon Sequestration|")


def _plot_pathway_matrix(
    *,
    ax: Axes,
    pathway_df: pd.DataFrame,
    years: list[int],
    color: str,
    alpha: float,
    zorder: int,
) -> None:
    """Plot one or more pathway rows against explicit calendar year x values."""
    if pathway_df.empty or not years:
        return
    x_values = np.asarray(years, dtype=float)
    y_values = pathway_df.loc[:, years].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float).T
    ax.plot(
        x_values,
        y_values * MT_TO_GT,
        color=color,
        alpha=alpha,
        zorder=zorder,
    )


def plot_carrying_capacities_summary(
    *,
    data_df: pd.DataFrame,
    data_historic_df: pd.Series,
    var_selected: str,
    timewindow_l: list[int],
    ax: np.ndarray,
    remaining_budget_end_year_value: int | None = None,
    remaining_budget_drop_records: list[dict] | None = None,
    remaining_budget_figure_name: str | None = None,
    remaining_budget_subset_name: str | None = None,
) -> np.ndarray:
    """Plot pathways plus study and remaining budgets for one variable."""
    flat_ax = np.ravel(ax)
    plot_pathways(
        data_df=data_df,
        data_historic_df=data_historic_df,
        var_selected=var_selected,
        timewindow_l=timewindow_l,
        ax=flat_ax[0],
    )
    flat_ax[0].fill_between(
        [timewindow_l[0], timewindow_l[1]],
        [flat_ax[0].get_ylim()[0]],
        [flat_ax[0].get_ylim()[1]],
        color="gray",
        alpha=0.2,
        zorder=-1,
    )
    plot_budgets_summary(
        data_df=data_df,
        var_selected=var_selected,
        timewindow_l=timewindow_l,
        ax=flat_ax[1:],
        remaining_budget_end_year_value=remaining_budget_end_year_value,
        remaining_budget_drop_records=remaining_budget_drop_records,
        remaining_budget_figure_name=remaining_budget_figure_name,
        remaining_budget_subset_name=remaining_budget_subset_name,
    )
    return flat_ax


def plot_pathways(
    *,
    data_df: pd.DataFrame,
    data_historic_df: pd.Series,
    var_selected: str,
    timewindow_l: list[int],
    ax: Axes,
) -> Axes:
    """Plot historical and model pathways for one output variable."""
    selected_var_df = var_df(data_df, var_selected)
    all_categories_l = list(sorted(set(data_df["Category"])))
    selected_max_year = max_year(selected_var_df)
    if selected_max_year is None:
        raise RuntimeError(
            f"AR6 figure inputs for '{var_selected}' did not contain any numeric year columns."
        )
    study_years = year_slice(selected_var_df, timewindow_l[0], timewindow_l[1])
    remaining_years = year_slice(selected_var_df, timewindow_l[1] + 1, selected_max_year)
    for curr_cat in all_categories_l:
        tmp_filtered_df = pd.DataFrame(
            selected_var_df.loc[(data_df["Category"] == curr_cat), study_years]
        )
        tmp_filtered_remaining_df = pd.DataFrame(
            selected_var_df.loc[(data_df["Category"] == curr_cat), remaining_years]
        )
        if len(tmp_filtered_df) == 0:
            continue
        _plot_pathway_matrix(
            ax=ax,
            pathway_df=tmp_filtered_df,
            years=study_years,
            color=CATEGORY_COLORS.get(curr_cat, "black"),
            alpha=0.1,
            zorder=-1,
        )
        if len(tmp_filtered_remaining_df.columns) > 0:
            _plot_pathway_matrix(
                ax=ax,
                pathway_df=tmp_filtered_remaining_df,
                years=remaining_years,
                color=CATEGORY_COLORS.get(curr_cat, "black"),
                alpha=0.01,
                zorder=-1,
            )
    if not data_historic_df.empty:
        hist_values = pd.Series(
            pd.to_numeric(data_historic_df, errors="raise"),
            dtype=float,
        ).to_numpy()
        ax.plot(
            np.asarray(data_historic_df.index, dtype=float),
            hist_values,
            color="black",
            linewidth=4,
        )
    ax.set_title(f"{var_selected}")
    ax.set_xlim(1945, 2105)
    unit_label = (
        "GtCO$_2$ yr$^{-1}$" if _is_co2_unit_variable(var_selected) else "GtCO$_2$eq yr$^{-1}$"
    )
    ax.set_ylabel(f"{var_selected}\n[{unit_label}]")
    ax.grid(axis="y", color="grey", alpha=0.3, zorder=-1)
    return ax


def plot_budgets_summary(
    *,
    data_df: pd.DataFrame,
    var_selected: str,
    timewindow_l: list[int],
    ax: np.ndarray,
    remaining_budget_end_year_value: int | None = None,
    remaining_budget_drop_records: list[dict] | None = None,
    remaining_budget_figure_name: str | None = None,
    remaining_budget_subset_name: str | None = None,
) -> None:
    """Plot study and remaining budget violins for one variable."""
    flat_ax = np.ravel(ax)
    selected_var_df = var_df(data_df, var_selected)
    all_categories_l = list(sorted(set(data_df["Category"])))
    all_ssps_l = list(sorted(set(data_df["Ssp_family"])))
    selected_max_year = max_year(selected_var_df)
    if selected_max_year is None:
        raise RuntimeError(
            f"AR6 figure inputs for '{var_selected}' did not contain any numeric year columns."
        )
    study_years = year_slice(selected_var_df, timewindow_l[0], timewindow_l[1])
    budget_end_year = (
        remaining_budget_end_year(selected_var_df)
        if remaining_budget_end_year_value is None
        else int(remaining_budget_end_year_value)
    )
    remaining_years = (
        year_slice(selected_var_df, timewindow_l[1] + 1, budget_end_year)
        if budget_end_year is not None
        else []
    )
    remaining_ok = append_remaining_budget_drop_records(
        drop_records=remaining_budget_drop_records,
        data_all_cats_df=data_df,
        var_data_df=selected_var_df,
        remaining_budget_end_year_value=budget_end_year,
        figure_name=remaining_budget_figure_name,
        subset_name=remaining_budget_subset_name,
        study_period=timewindow_l,
    )
    remaining_var_df = (
        pd.DataFrame(selected_var_df.loc[remaining_ok, remaining_years])
        if remaining_years
        else selected_var_df.iloc[0:0, :]
    )
    remaining_meta_df = (
        pd.DataFrame(selected_var_df.loc[remaining_ok, :])
        if bool(remaining_ok.any())
        else selected_var_df.iloc[0:0, :]
    )
    for idx_cat, curr_cat in enumerate(all_categories_l):
        tmp_filtered_df = pd.DataFrame(
            selected_var_df.loc[(data_df["Category"] == curr_cat), study_years]
        )
        tmp_filtered_remaining_df = pd.DataFrame(
            remaining_var_df.loc[(remaining_meta_df["Category"] == curr_cat), remaining_years]
        )
        if len(tmp_filtered_df) > 0:
            tmp_cumul_df = tmp_filtered_df.sum(axis=1)
            parts = flat_ax[0].violinplot(
                tmp_cumul_df.to_numpy(dtype=float) * MT_TO_GT,
                positions=[idx_cat],
                showmedians=False,
                widths=1 / (len(all_ssps_l) + 3),
                showextrema=False,
            )
            plot_violin(parts, CATEGORY_COLORS.get(curr_cat, "black"), 0.8)
            flat_ax[0].scatter(
                x=[idx_cat],
                y=float(np.median(tmp_cumul_df.to_numpy(dtype=float) * MT_TO_GT)),
                c=CATEGORY_COLORS.get(curr_cat, "black"),
                marker=".",
                edgecolor="black",
            )
            if len(tmp_filtered_remaining_df) > 0:
                tmp_cumul_remaining_df = tmp_filtered_remaining_df.sum(axis=1)
                parts = flat_ax[1].violinplot(
                    tmp_cumul_remaining_df.to_numpy(dtype=float) * MT_TO_GT,
                    positions=[idx_cat],
                    showmedians=False,
                    widths=1 / (len(all_ssps_l) + 3),
                    showextrema=False,
                )
                plot_violin(parts, "black", 0.8)
                flat_ax[1].scatter(
                    x=[idx_cat],
                    y=float(np.median(tmp_cumul_remaining_df.to_numpy(dtype=float) * MT_TO_GT)),
                    c="black",
                    marker=".",
                    edgecolor="black",
                )
        for idx_ssp, curr_ssp in enumerate(all_ssps_l):
            x_offset_ssp = 0.2
            tmp_filtered_df = pd.DataFrame(
                selected_var_df.loc[
                    (data_df["Ssp_family"] == curr_ssp) & (data_df["Category"] == curr_cat),
                    study_years,
                ]
            )
            tmp_filtered_remaining_df = pd.DataFrame(
                remaining_var_df.loc[
                    (remaining_meta_df["Ssp_family"] == curr_ssp)
                    & (remaining_meta_df["Category"] == curr_cat),
                    remaining_years,
                ]
            )
            if len(tmp_filtered_df) == 0:
                continue
            tmp_cumul_df = tmp_filtered_df.sum(axis=1)
            position = x_offset_ssp + idx_cat + idx_ssp / (len(all_ssps_l) + 3)
            parts = flat_ax[0].violinplot(
                tmp_cumul_df.to_numpy(dtype=float) * MT_TO_GT,
                positions=[position],
                showmedians=False,
                widths=1 / (len(all_ssps_l) + 3),
                showextrema=False,
            )
            plot_violin(parts, CATEGORY_COLORS.get(curr_cat, "black"), 0.5)
            flat_ax[0].scatter(
                x=[position],
                y=float(np.median(tmp_cumul_df.to_numpy(dtype=float) * MT_TO_GT)),
                c=CATEGORY_COLORS.get(curr_cat, "black"),
                marker=".",
                edgecolor="black",
            )
            flat_ax[0].text(
                position,
                100,
                f"SSP{curr_ssp} (n={len(tmp_filtered_df)})",
                rotation=90,
                color="black",
                fontsize=6,
            )
            if len(tmp_filtered_remaining_df) > 0:
                tmp_cumul_remaining_df = tmp_filtered_remaining_df.sum(axis=1)
                parts = flat_ax[1].violinplot(
                    tmp_cumul_remaining_df.to_numpy(dtype=float) * MT_TO_GT,
                    positions=[position],
                    showmedians=False,
                    widths=1 / (len(all_ssps_l) + 3),
                    showextrema=False,
                )
                plot_violin(parts, "black", 0.4)
                flat_ax[1].scatter(
                    x=[position],
                    y=float(np.median(tmp_cumul_remaining_df.to_numpy(dtype=float) * MT_TO_GT)),
                    c="black",
                    marker=".",
                    edgecolor="black",
                )
    all_cumul = pd.DataFrame(selected_var_df.loc[:, study_years]).sum(axis=1) * MT_TO_GT
    flat_ax[0].set_title(f"Budgets\n{timewindow_l[0]}-{timewindow_l[1]}")
    flat_ax[0].set_ylim([min([0, float(all_cumul.min())]) * 1.03, float(all_cumul.max()) * 1.03])
    flat_ax[0].set_ylabel(
        f"{var_selected}\n[{'GtCO$_2$' if _is_co2_unit_variable(var_selected) else 'GtCO$_2$eq'}]"
    )
    flat_ax[0].set_xticks(
        [idx_cat for idx_cat, _curr_cat in enumerate(all_categories_l)], all_categories_l
    )
    flat_ax[0].grid(axis="y", color="grey", alpha=0.3, zorder=-1)
    remaining_title_end_year = budget_end_year if budget_end_year is not None else selected_max_year
    flat_ax[1].set_title(f"Remaining budgets\n{timewindow_l[1] + 1}-{remaining_title_end_year}")
    flat_ax[1].set_ylabel(
        f"{var_selected}\n[{'GtCO$_2$' if _is_co2_unit_variable(var_selected) else 'GtCO$_2$eq'}]"
    )
    flat_ax[1].set_xticks(
        [idx_cat for idx_cat, _curr_cat in enumerate(all_categories_l)], all_categories_l
    )
    flat_ax[1].grid(axis="y", color="grey", alpha=0.3, zorder=-1)
