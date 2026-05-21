"""Overview style AR6 figure writers."""

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from pyaesa.download.ar6.utils.config import (
    NET_KYOTO_WO_AFOLU,
    SEQUESTRATION_SUBTOTAL,
    SEQUESTRATION_TOTAL,
)

from .figure_io import save_figure
from .plot_budgets import plot_budgets_summary, plot_carrying_capacities_summary, plot_pathways
from .plot_helpers import (
    FIGURE_MODEL_LABEL,
    historical_series,
    write_drop_csv,
)
from .plot_historical import plot_historical_emissions
from .variable_groups import emission_variable_groups


def _numeric_series(values: pd.Series) -> pd.Series:
    """Return one numeric pandas Series for plotting calculations."""
    return pd.Series(pd.to_numeric(values, errors="raise"), copy=False)


def _pathway_filename(
    database: str,
    categories_repr: str,
    study_period: list[int],
    ext: str,
) -> str:
    return (
        f"fig-harmonization-pathways-{database}"
        f"-MOD={FIGURE_MODEL_LABEL}-CAT={categories_repr}"
        f"-studyperiod={int(study_period[0])}to{int(study_period[1])}.{ext}"
    )


def _delta_tconv_filename(
    database: str,
    categories_repr: str,
    study_period: list[int],
    ext: str,
) -> str:
    return (
        f"fig-harmonization-stats-delta-tconv-{database}"
        f"-MOD={FIGURE_MODEL_LABEL}-CAT={categories_repr}"
        f"-studyperiod={int(study_period[0])}to{int(study_period[1])}.{ext}"
    )


def _harm_stats_filename(
    database: str,
    categories_repr: str,
    study_period: list[int],
    ext: str,
) -> str:
    return (
        f"fig-harmonization-stats-{database}-MOD={FIGURE_MODEL_LABEL}"
        f"-CAT={categories_repr}-studyperiod={int(study_period[0])}"
        f"to{int(study_period[1])}.{ext}"
    )


def _budgets_filename(
    variable_group: str,
    database: str,
    categories_repr: str,
    study_period: list[int],
    ext: str,
) -> str:
    return (
        f"fig-budgets-{variable_group}-{database}-MOD={FIGURE_MODEL_LABEL}"
        f"-CAT={categories_repr}-studyperiod={int(study_period[0])}"
        f"to{int(study_period[1])}.{ext}"
    )


def _sequestration_budgets_filename(
    variable_group: str,
    database: str,
    categories_repr: str,
    study_period: list[int],
    ext: str,
) -> str:
    return (
        f"fig-sequestration-budgets-for{variable_group}-{database}-MOD={FIGURE_MODEL_LABEL}"
        f"-CAT={categories_repr}-studyperiod={int(study_period[0])}"
        f"to{int(study_period[1])}.{ext}"
    )


def _sequestration_variable_for_emissions_variable(emissions_variable: str) -> str:
    """Return the companion sequestration row plotted for an emissions output variable."""
    if emissions_variable.startswith("Emissions(gross_alt)"):
        return SEQUESTRATION_SUBTOTAL
    return SEQUESTRATION_TOTAL


def _rows_for_variable_scope(
    harmonized_data: pd.DataFrame,
    emissions_variable: str,
    sequestration_variable: str,
) -> pd.DataFrame:
    """Return companion sequestration rows for the retained emissions row scope."""
    emissions_rows = harmonized_data.loc[
        harmonized_data.index.isin([emissions_variable], level="variable")
    ]
    pairs = emissions_rows.index.droplevel("variable").drop_duplicates()
    companion_rows = harmonized_data.loc[
        harmonized_data.index.isin([sequestration_variable], level="variable")
    ]
    return companion_rows.loc[companion_rows.index.droplevel("variable").isin(pairs)].copy()


def _apply_dense_budget_layout(fig: Figure, *, row_count: int, top: float = 0.89) -> None:
    """Apply a stable manual layout for dense AR6 budget panel figures.

    These overview products carry long titles, multi-row axes, and annotation
    overlays. Manual spacing keeps the panel family readable and avoids
    depending on Matplotlib auto-layout heuristics for this dense layout.
    """
    hspace = 0.4 if row_count >= 3 else 0.45
    fig.subplots_adjust(top=top, bottom=0.07, left=0.08, right=0.98, hspace=hspace, wspace=0.32)


def write_historical_emissions_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    historical_data: pd.DataFrame,
    raw_data_dir: Path,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Write the historical emissions overview figure."""
    fig, _axes = plot_historical_emissions(
        historical_data_df=historical_data,
        raw_data_dir=raw_data_dir,
    )
    save_figure(
        fig,
        figures_dir / f"fig-processed-historical-emissions.{ext}",
        dpi=dpi,
        out_paths=out_paths,
        metadata_callback=metadata_callback,
    )


def write_harmonization_pathways_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    original_comparison_data: pd.DataFrame,
    harmonized_data: pd.DataFrame,
    historical_data: pd.DataFrame,
    all_variables_l: list[str],
    study_period: list[int],
    process_end_year: int,
    database: str,
    categories_repr: str,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Write the harmonized versus original pathway figure."""
    year_columns = [
        int(col) for col in harmonized_data.columns if isinstance(col, int) or str(col).isdigit()
    ]
    figure_period = (
        f"{min(year_columns)}-{max(year_columns)}"
        if year_columns
        else f"{int(study_period[0])}-{int(study_period[1])}"
    )
    fig, ax = plt.subplots(
        2,
        len(all_variables_l),
        figsize=(5 * len(all_variables_l), 7),
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle(
        "Overview\n"
        f"model={FIGURE_MODEL_LABEL}\n"
        f"period={figure_period}\n"
        "Pathways comparison (unharmonized vs. harmonized)\n",
        fontweight="bold",
    )
    for idx_var, var_sel in enumerate(all_variables_l):
        hist_full = historical_series(historical_data, var_sel, start_year=1950)
        plot_pathways(
            data_df=original_comparison_data,
            data_historic_df=hist_full,
            var_selected=var_sel,
            timewindow_l=[2000, process_end_year],
            ax=ax[0, idx_var],
        )
        hist_to_study_start = hist_full.loc[hist_full.index <= int(study_period[0])]
        plot_pathways(
            data_df=harmonized_data,
            data_historic_df=hist_to_study_start,
            var_selected=var_sel,
            timewindow_l=[int(study_period[0]), process_end_year],
            ax=ax[1, idx_var],
        )
        ax[1, idx_var].vlines(
            x=int(study_period[0]),
            ymin=[ax[1, idx_var].get_ylim()[0]],
            ymax=[ax[1, idx_var].get_ylim()[1]],
            color="gray",
            alpha=0.5,
            linestyle="dashed",
            zorder=-1,
        )
    save_figure(
        fig,
        figures_dir / _pathway_filename(database, categories_repr, study_period, ext),
        dpi=dpi,
        out_paths=out_paths,
        metadata_callback=metadata_callback,
    )


def write_delta_tconv_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    harmonization_log: pd.DataFrame,
    database: str,
    categories_repr: str,
    study_period: list[int],
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Write the delta to convergence summary figure."""
    fig, ax = plt.subplots(1, 1, figsize=(2, 3), layout="constrained")
    horizon = _numeric_series(pd.Series(harmonization_log["horizon-for-harmonization"], copy=False))
    netzero = _numeric_series(pd.Series(harmonization_log["model-netzero-year"], copy=False))
    ind = (horizon - netzero).abs().sort_values(ascending=False).dropna()
    ind = ind[ind > 0]
    if len(ind) > 0:
        ax.violinplot(ind.to_numpy(dtype=float))
    ax.set_title(f"n={len(ind)}")
    ax.set_ylabel(
        "Difference between model first year of\nnegative emissions and $t_{conv}$ [year]"
    )
    ax.set_xticks([])
    save_figure(
        fig,
        figures_dir / _delta_tconv_filename(database, categories_repr, study_period, ext),
        dpi=dpi,
        out_paths=out_paths,
        metadata_callback=metadata_callback,
    )


def write_harmonization_stats_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    harmonization_log: pd.DataFrame,
    all_variables_l: list[str],
    study_period: list[int],
    database: str,
    categories_repr: str,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Write the harmonization statistics figure family."""
    fig, ax = plt.subplots(
        2,
        len(all_variables_l),
        figsize=(12, 5),
        layout="constrained",
        squeeze=False,
    )
    fig.suptitle(
        "Ratio between model and historical cumulative emissions\n"
        "(between model base year and harmonization year)\n",
        fontweight="bold",
    )
    for idx_var, curr_var in enumerate(all_variables_l):
        curr_log_df = harmonization_log.loc[(slice(None), slice(None), curr_var), :]
        ratio_model_to_historical_df = _numeric_series(
            _numeric_series(pd.Series(curr_log_df["pathway-cumulative"], copy=False))
            / _numeric_series(pd.Series(curr_log_df["historic-cumulative"], copy=False))
        )
        ratio_model_to_historical_df = ratio_model_to_historical_df.astype(float).dropna()
        if len(ratio_model_to_historical_df) > 0:
            ax[0, idx_var].violinplot(
                ratio_model_to_historical_df.to_numpy(dtype=float),
                orientation="vertical",
            )
        ax[0, idx_var].set_ylabel("Cumulative model vs.\ncumulative historic [/]")
        ax[0, idx_var].set_title(f"{curr_var}\nn={len(ratio_model_to_historical_df)}")
        ax[0, idx_var].hlines(y=1, xmin=0.7, xmax=1.3, color="red", linestyle="dashed")
        ax[0, idx_var].set_xticklabels([])
        yearly_correction = _numeric_series(
            pd.Series(curr_log_df["yearly-correction"], copy=False)
        ).dropna()
        if len(yearly_correction) > 0:
            ax[1, idx_var].violinplot(yearly_correction.to_numpy(dtype=float))
        ax[1, idx_var].hlines(y=0, xmin=0.7, xmax=1.3, color="black", linestyle="dashed")
        unit_values = (
            pd.Series(curr_log_df["unit"], copy=False).dropna().astype(str)
            if "unit" in curr_log_df.columns
            else pd.Series([], dtype=str)
        )
        unit_label = unit_values.iloc[0] if len(unit_values) > 0 else ""
        ax[1, idx_var].set_ylabel(f"yearly correction\n{unit_label}")
        ax[1, idx_var].set_title(f"{curr_var}\nn={len(yearly_correction)}")
        ax[1, idx_var].set_xticklabels([])
    save_figure(
        fig,
        figures_dir / _harm_stats_filename(database, categories_repr, study_period, ext),
        dpi=dpi,
        out_paths=out_paths,
        metadata_callback=metadata_callback,
    )


def write_processed_budgets_figure(
    *,
    figures_dir: Path,
    ext: str,
    dpi: int,
    out_paths: list[str],
    harmonized_data: pd.DataFrame,
    historical_data: pd.DataFrame,
    all_variables_l: list[str],
    study_period: list[int],
    remaining_budget_end_year_value: int | None,
    database: str,
    categories_repr: str,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Write harmonized budget figures for CO2 and Kyoto Gases outputs."""
    for variable_group, group_variables in emission_variable_groups(all_variables_l):
        fig, ax = plt.subplots(
            3,
            len(group_variables),
            figsize=(5 * len(group_variables), 11),
            squeeze=False,
        )
        fig.suptitle(
            "Overview\n"
            f"model={FIGURE_MODEL_LABEL}\n"
            f"period={int(study_period[0])}-{int(study_period[1])}\n"
            f"Processed data (harmonized {variable_group} emissions)\n",
            fontweight="bold",
        )
        _apply_dense_budget_layout(fig, row_count=3)
        budgets_remaining_drop_records: list[dict] = []
        for idx_var, var_sel in enumerate(group_variables):
            hist_full = historical_series(historical_data, var_sel, start_year=1950)
            hist_to_study_start = hist_full.loc[hist_full.index <= int(study_period[0])]
            plot_carrying_capacities_summary(
                data_df=harmonized_data,
                data_historic_df=hist_to_study_start,
                var_selected=var_sel,
                timewindow_l=[int(study_period[0]), int(study_period[1])],
                ax=ax[:, idx_var],
                remaining_budget_end_year_value=remaining_budget_end_year_value,
                remaining_budget_drop_records=budgets_remaining_drop_records,
                remaining_budget_figure_name=f"fig-budgets-{variable_group}",
                remaining_budget_subset_name="all",
            )
            if list(map(int, study_period)) == [2010, 2100] and var_sel == NET_KYOTO_WO_AFOLU:
                ax[1, idx_var].fill_between(
                    [-0.1, 1.8],
                    [481],
                    [1791],
                    color="darkgreen",
                    alpha=0.2,
                    zorder=-1,
                )
                ax[1, idx_var].hlines(
                    y=1156,
                    xmin=-0.1,
                    xmax=1.8,
                    zorder=-1,
                    color="darkgreen",
                    linestyle="dashed",
                )
                ax[1, idx_var].fill_between(
                    [1.9, 2.8],
                    [1260],
                    [2598],
                    color="darkgreen",
                    alpha=0.2,
                    zorder=-1,
                )
                ax[1, idx_var].hlines(
                    y=1749,
                    xmin=1.9,
                    xmax=2.8,
                    zorder=-1,
                    color="darkgreen",
                    linestyle="dashed",
                )
                ax[1, idx_var].set_ylim([0, ax[1, idx_var].get_ylim()[1] * 1.05])
        budgets_path = figures_dir / _budgets_filename(
            variable_group, database, categories_repr, study_period, ext
        )
        save_figure(
            fig,
            budgets_path,
            dpi=dpi,
            out_paths=out_paths,
            metadata_callback=metadata_callback,
        )
        write_drop_csv(
            figures_dir,
            f"{budgets_path.stem}-remaining-budget-panel",
            budgets_remaining_drop_records,
        )


def write_sequestration_budgets_figure(
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
    categories_repr: str,
    metadata_callback: Callable[[list[str]], None] | None = None,
) -> None:
    """Write sequestration budget figures aligned to final emissions outputs."""
    for variable_group, group_variables in emission_variable_groups(all_variables_l):
        fig, ax = plt.subplots(
            3,
            len(group_variables),
            figsize=(5 * len(group_variables), 11),
            squeeze=False,
        )
        fig.suptitle(
            "Overview\n"
            f"model={FIGURE_MODEL_LABEL}\n"
            f"period={int(study_period[0])}-{int(study_period[1])}\n"
            f"Processed data (sequestration companions for {variable_group} emissions)\n",
            fontweight="bold",
        )
        _apply_dense_budget_layout(fig, row_count=3, top=0.84)
        remaining_drop_records: list[dict] = []
        for idx_var, emissions_variable in enumerate(group_variables):
            sequestration_variable = _sequestration_variable_for_emissions_variable(
                emissions_variable
            )
            scoped_sequestration = _rows_for_variable_scope(
                harmonized_data,
                emissions_variable,
                sequestration_variable,
            )
            plot_pathways(
                data_df=scoped_sequestration,
                data_historic_df=pd.Series(dtype=float),
                var_selected=sequestration_variable,
                timewindow_l=[int(study_period[0]), int(study_period[1])],
                ax=ax[0, idx_var],
            )
            ax[0, idx_var].set_title(
                f"{sequestration_variable}\nfor {emissions_variable}",
                fontsize=9,
            )
            plot_budgets_summary(
                data_df=scoped_sequestration,
                var_selected=sequestration_variable,
                timewindow_l=[int(study_period[0]), int(study_period[1])],
                ax=ax[1:, idx_var],
                remaining_budget_end_year_value=remaining_budget_end_year_value,
                remaining_budget_drop_records=remaining_drop_records,
                remaining_budget_figure_name=f"fig-sequestration-budgets-for{variable_group}",
                remaining_budget_subset_name=emissions_variable,
            )
        budgets_path = figures_dir / _sequestration_budgets_filename(
            variable_group, database, categories_repr, study_period, ext
        )
        save_figure(
            fig,
            budgets_path,
            dpi=dpi,
            out_paths=out_paths,
            metadata_callback=metadata_callback,
        )
        write_drop_csv(
            figures_dir,
            f"{budgets_path.stem}-remaining-budget-panel",
            remaining_drop_records,
        )
