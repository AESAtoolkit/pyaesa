"""Figure orchestration for processed AR6 climate outputs."""

from pathlib import Path
from typing import Callable

import pandas as pd
from pyaesa.shared.figures.contracts import normalize_figure_output_format
from pyaesa.shared.runtime.io.filesystem import ensure_dir

from .figure_sampling_config import RUN_BATCH_SIZE, STABLE_CHECKS_REQUIRED
from .figure_overview_panels import (
    write_delta_tconv_figure,
    write_harmonization_pathways_figure,
    write_harmonization_stats_figure,
    write_historical_emissions_figure,
    write_processed_budgets_figure,
    write_sequestration_budgets_figure,
)
from .figure_sampling_panels import write_sampling_figures
from .figure_sequestration_panels import write_sequestration_contributions_figure
from .figure_warming_panel import write_median_warming_figure
from .plot_helpers import max_year, remaining_budget_end_year
from .variable_groups import (
    SEQUESTRATION_CONTRIBUTION_VARIABLES,
    SEQUESTRATION_FIGURE_VARIABLES,
    missing_requested_variables,
    ordered_available_variables,
)


_TOTAL_FIGURES = 18


def _figure_status(
    status_callback: Callable[[str], None] | None,
    counter: list[int],
    label: str,
) -> None:
    """Emit a numbered figure progress message via *status_callback*."""
    if status_callback is None:
        return
    counter[0] += 1
    status_callback(f"Generating figure {counter[0]}/{_TOTAL_FIGURES}: {label}")


def generate_ar6_figures(
    *,
    output_dir: str,
    harmonized_data: pd.DataFrame,
    original_data: pd.DataFrame,
    harmonization_log: pd.DataFrame,
    historical_data: pd.DataFrame,
    source_metadata: pd.DataFrame,
    raw_data_dir: str,
    study_period: list[int],
    database: str,
    categories: list[str],
    variables_output: list[str],
    figure_output_format: str = "png",
    dpi: int = 500,
    figure_convergence_tol: float,
    figure_convergence_max_runs: int,
    status_callback: Callable[[str], None] | None = None,
    metadata_callback: Callable[[list[str]], None] | None = None,
    sampling_run_batch_size: int = RUN_BATCH_SIZE,
    sampling_stable_checks_required: int = STABLE_CHECKS_REQUIRED,
    write_sampling_figures_func=write_sampling_figures,
) -> tuple[list[str], pd.DataFrame]:
    """Generate AR6 diagnostic figures and return their file list plus sampling log."""
    ext = normalize_figure_output_format(figure_output_format).lstrip(".")

    figures_dir = Path(output_dir)
    figures_dir = ensure_dir(figures_dir)
    categories_repr = str(list(categories))
    process_end_year = max_year(harmonized_data) or max_year(original_data)
    if process_end_year is None:
        raise RuntimeError("AR6 figure inputs did not contain any numeric year columns.")
    harmonization_variables = list(sorted(set(harmonization_log.index.get_level_values(level=2))))
    available_harmonized_variables = harmonized_data.index.get_level_values("variable")
    missing_processed_variables = missing_requested_variables(
        requested_variables=variables_output,
        available_variables=available_harmonized_variables,
    )
    if missing_processed_variables:
        missing_display = ", ".join(missing_processed_variables)
        raise RuntimeError(
            "AR6 figure generation requires processed output variables in HARMONIZED_AR6. "
            f"Missing variables: {missing_display}."
        )
    processed_variables = ordered_available_variables(
        requested_variables=variables_output,
        available_variables=available_harmonized_variables,
    )
    gross_output_requested = any(
        variable.startswith(("Emissions(gross)", "Emissions(gross_alt)"))
        for variable in variables_output
    )
    if gross_output_requested:
        sequestration_variables = ordered_available_variables(
            requested_variables=SEQUESTRATION_FIGURE_VARIABLES,
            available_variables=available_harmonized_variables,
        )
        if len(sequestration_variables) != len(SEQUESTRATION_FIGURE_VARIABLES):
            missing_sequestration = missing_requested_variables(
                requested_variables=SEQUESTRATION_FIGURE_VARIABLES,
                available_variables=available_harmonized_variables,
            )
            missing_display = ", ".join(missing_sequestration)
            raise RuntimeError(
                "AR6 gross figure generation requires sequestration companion rows in "
                f"HARMONIZED_AR6. Missing variables: {missing_display}."
            )
        available_original_variables = original_data.index.get_level_values("variable")
        sequestration_contribution_variables = ordered_available_variables(
            requested_variables=SEQUESTRATION_CONTRIBUTION_VARIABLES,
            available_variables=available_original_variables,
        )
        missing_contribution_variables = missing_requested_variables(
            requested_variables=SEQUESTRATION_FIGURE_VARIABLES,
            available_variables=available_original_variables,
        )
        if missing_contribution_variables:
            missing_display = ", ".join(missing_contribution_variables)
            raise RuntimeError(
                "AR6 sequestration contribution figure requires sequestration total "
                f"and subtotal rows in ORIGINAL_AR6. Missing variables: {missing_display}."
            )
    else:
        sequestration_variables = []
        sequestration_contribution_variables = []
    original_comparison_data = original_data.loc[
        original_data.index.intersection(harmonized_data.index)
    ].sort_index()
    original_index_df = original_data.index.to_frame(index=False)
    warming_scenario_rows_df = pd.DataFrame(
        {
            "model": original_index_df["model"].to_numpy(copy=False),
            "scenario": original_index_df["scenario"].to_numpy(copy=False),
            "Category": original_data["Category"].to_numpy(copy=False),
            "Ssp_family": original_data["Ssp_family"].to_numpy(copy=False),
        }
    ).set_index(["model", "scenario"])
    warming_scenario_rows_df = warming_scenario_rows_df.sort_index()
    remaining_budget_end_year_value = remaining_budget_end_year(harmonized_data)
    out_paths: list[str] = []
    fig_counter: list[int] = [0]

    _figure_status(status_callback, fig_counter, "historical emissions")
    write_historical_emissions_figure(
        figures_dir=figures_dir,
        ext=ext,
        dpi=dpi,
        out_paths=out_paths,
        historical_data=historical_data,
        raw_data_dir=Path(raw_data_dir),
        metadata_callback=metadata_callback,
    )
    if not harmonization_variables:
        raise RuntimeError(
            "AR6 figure generation requires at least one variable in the harmonization log."
        )

    _figure_status(status_callback, fig_counter, "harmonization pathways")
    write_harmonization_pathways_figure(
        figures_dir=figures_dir,
        ext=ext,
        dpi=dpi,
        out_paths=out_paths,
        original_comparison_data=original_comparison_data,
        harmonized_data=harmonized_data,
        historical_data=historical_data,
        all_variables_l=harmonization_variables,
        study_period=study_period,
        process_end_year=process_end_year,
        database=database,
        categories_repr=categories_repr,
        metadata_callback=metadata_callback,
    )
    _figure_status(status_callback, fig_counter, "delta/convergence")
    write_delta_tconv_figure(
        figures_dir=figures_dir,
        ext=ext,
        dpi=dpi,
        out_paths=out_paths,
        harmonization_log=harmonization_log,
        database=database,
        categories_repr=categories_repr,
        study_period=study_period,
        metadata_callback=metadata_callback,
    )
    _figure_status(status_callback, fig_counter, "harmonization stats")
    write_harmonization_stats_figure(
        figures_dir=figures_dir,
        ext=ext,
        dpi=dpi,
        out_paths=out_paths,
        harmonization_log=harmonization_log,
        all_variables_l=harmonization_variables,
        study_period=study_period,
        database=database,
        categories_repr=categories_repr,
        metadata_callback=metadata_callback,
    )
    _figure_status(status_callback, fig_counter, "processed budgets")
    write_processed_budgets_figure(
        figures_dir=figures_dir,
        ext=ext,
        dpi=dpi,
        out_paths=out_paths,
        harmonized_data=harmonized_data,
        historical_data=historical_data,
        all_variables_l=processed_variables,
        study_period=study_period,
        remaining_budget_end_year_value=remaining_budget_end_year_value,
        database=database,
        categories_repr=categories_repr,
        metadata_callback=metadata_callback,
    )
    if sequestration_contribution_variables:
        _figure_status(status_callback, fig_counter, "sequestration contributions")
        write_sequestration_contributions_figure(
            figures_dir=figures_dir,
            ext=ext,
            dpi=dpi,
            out_paths=out_paths,
            original_data=original_data,
            all_variables_l=sequestration_contribution_variables,
            categories=list(categories),
            study_period=study_period,
            database=database,
            categories_repr=categories_repr,
            metadata_callback=metadata_callback,
        )
    if sequestration_variables:
        _figure_status(status_callback, fig_counter, "sequestration budgets")
        write_sequestration_budgets_figure(
            figures_dir=figures_dir,
            ext=ext,
            dpi=dpi,
            out_paths=out_paths,
            harmonized_data=harmonized_data,
            all_variables_l=processed_variables,
            study_period=study_period,
            remaining_budget_end_year_value=remaining_budget_end_year_value,
            database=database,
            categories_repr=categories_repr,
            metadata_callback=metadata_callback,
        )
    _figure_status(status_callback, fig_counter, "median warming")
    write_median_warming_figure(
        figures_dir=figures_dir,
        ext=ext,
        dpi=dpi,
        out_paths=out_paths,
        scenario_rows_df=warming_scenario_rows_df,
        source_metadata=source_metadata,
        categories=categories,
        study_period=study_period,
        database=database,
        categories_repr=categories_repr,
        metadata_callback=metadata_callback,
    )

    def _sampling_status(message: str) -> None:
        if message.startswith("Generating"):
            if status_callback is not None:
                status_callback(message)
        else:
            _figure_status(status_callback, fig_counter, message)

    convergence_log_df = write_sampling_figures_func(
        figures_dir=figures_dir,
        ext=ext,
        dpi=dpi,
        out_paths=out_paths,
        harmonized_data=harmonized_data,
        all_variables_l=processed_variables,
        study_period=study_period,
        remaining_budget_end_year_value=remaining_budget_end_year_value,
        database=database,
        categories=list(categories),
        categories_repr=categories_repr,
        relative_tolerance=figure_convergence_tol,
        max_runs_per_bucket=figure_convergence_max_runs,
        status_callback=_sampling_status,
        metadata_callback=metadata_callback,
        run_batch_size=sampling_run_batch_size,
        stable_checks_required=sampling_stable_checks_required,
    )
    return out_paths, convergence_log_df
