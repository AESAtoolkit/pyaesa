"""Mode specific pathway processing ownership for AR6 climate outputs."""

from pathlib import Path
from typing import TypedDict

import pandas as pd

from pyaesa.download.ar6.utils.config import PROCESSED_OUTPUT_VARIABLES

from .derived_variables import (
    NET_HARMONIZATION_VARIABLES,
    SEQUESTRATION_COMPANION_VARIABLES,
    build_final_harmonized_emission_variables,
    build_pre_harmonization_variables,
)
from .harmonization import harmonize_emissions, stats_from_retained_pathways
from .historical_processing import process_historical_emissions
from .preprocessing import (
    filter_and_format_rawdata,
    interpolate_and_check,
)


class PathwayOutputs(TypedDict):
    """Structured return payload for one AR6 processing run."""

    final_all: pd.DataFrame
    original_all: pd.DataFrame
    harmonization_log_all: pd.DataFrame | None
    stats_var: pd.DataFrame
    historical_emissions: pd.DataFrame
    drop_logs: list[pd.DataFrame]
    variable_coverage_summary_counts: dict[str, dict[str, object]]
    latest_historical_year: int | None
    requested_harmonization_year: int | None
    harmonization_year: int | None
    harmonization_message: str | None


def build_pathway_outputs(
    *,
    explorer,
    categories: list[str],
    ssps: list[int],
    variables_output: list[str],
    study_period: list[int],
    database_raw_dir: Path,
    models_relevant_all: list[str],
    harmonization: bool,
    harmonization_method: str,
) -> PathwayOutputs:
    """Build processed AR6 pathway tables for one study period and mode."""
    summary_variables = tuple(
        variable for variable in PROCESSED_OUTPUT_VARIABLES if variable in set(variables_output)
    )
    final_variables = tuple(dict.fromkeys([*variables_output, *SEQUESTRATION_COMPANION_VARIABLES]))
    historical_emissions = pd.DataFrame()
    latest_historical_year: int | None = None
    requested_harmonization_year: int | None = None
    harmonization_year: int | None = None
    harmonization_message: str | None = None
    harmonization_period: list[int] = []
    requested_harmonization_year_for_run = 0

    if harmonization:
        historical_emissions = process_historical_emissions(f"{database_raw_dir.as_posix()}/")
        historical_years = [int(col) for col in historical_emissions.columns if str(col).isdigit()]
        latest_historical_year = max(historical_years)
        requested_harmonization_year_for_run = int(study_period[0])
        requested_harmonization_year = requested_harmonization_year_for_run
        harmonization_year = min(
            requested_harmonization_year_for_run,
            int(latest_historical_year),
        )
        harmonization_period = [harmonization_year, int(study_period[1])]
        if harmonization_year != requested_harmonization_year_for_run:
            harmonization_message = (
                "Warning: requested study period starts in "
                f"{requested_harmonization_year_for_run}, but historical data are available only "
                f"through {int(latest_historical_year)}. Harmonization was therefore anchored "
                f"in {harmonization_year} and applied over "
                f"{harmonization_year}-{int(study_period[1])}; downstream outputs still select "
                f"the requested study years {int(study_period[0])}-{int(study_period[1])}."
            )

    mem_cats_original = []
    mem_cats_final = []
    mem_cats_stats = []
    mem_cats_harmonization_log = []
    drop_logs = []
    original_pairs_by_variable = {variable: set() for variable in summary_variables}
    retained_pairs_by_variable = {variable: set() for variable in summary_variables}

    for category in categories:
        concat_all_ssps = pd.concat(
            [
                filter_and_format_rawdata(
                    explorer,
                    {
                        "category": category,
                        "ssp_family": ssp,
                        "model": ["ALL", models_relevant_all],
                    },
                )
                for ssp in ssps
            ]
        )
        data_ic = interpolate_and_check(concat_all_ssps)
        pre_harmonization_rows, drop_derived_df = build_pre_harmonization_variables(
            data_ic,
            study_start_year=int(study_period[0]),
        )
        _append_category_log(drop_logs, drop_derived_df, category=category)
        mem_cats_original.append(pre_harmonization_rows)

        retained_net = _select_variables(pre_harmonization_rows, NET_HARMONIZATION_VARIABLES)

        if harmonization and not retained_net.empty:
            harmonized_net, harmonization_log = harmonize_emissions(
                data_df=retained_net,
                historic_data_df=historical_emissions,
                study_timeperiod=harmonization_period,
                requested_harmonization_year=requested_harmonization_year_for_run,
                harmonization_method=harmonization_method,
            )
            mem_cats_harmonization_log.append(harmonization_log)
        else:
            harmonized_net = retained_net

        companion_rows = _select_variables(
            pre_harmonization_rows,
            SEQUESTRATION_COMPANION_VARIABLES,
        )
        # Sequestration companions are evidence rows for gross construction. They are
        # trimmed to the retained net scope, but are not offset harmonized as emissions rows.
        companion_rows = _align_companions_to_net_scope(
            companion_rows=companion_rows,
            net_rows=harmonized_net,
        )
        final_input = _concat_non_empty([harmonized_net, companion_rows], columns=harmonized_net)
        final_rows, gross_log, before_gross_sign_filter = build_final_harmonized_emission_variables(
            final_input
        )
        _append_category_log(drop_logs, gross_log, category=category)
        final_visible = _select_variables(final_rows, final_variables)
        before_visible = _select_variables(before_gross_sign_filter, summary_variables)

        for variable in summary_variables:
            original_pairs_by_variable[variable].update(
                _pairs_for_variable(before_visible, variable=variable)
            )
            retained_pairs_by_variable[variable].update(
                _pairs_for_variable(final_visible, variable=variable)
            )

        mem_cats_stats.append(harmonized_net)
        mem_cats_final.append(final_visible)

    final_all = pd.concat(mem_cats_final).sort_index()
    original_all = pd.concat(mem_cats_original).sort_index()
    stats_source_all = pd.concat(mem_cats_stats).sort_index()
    harmonization_log_all = None
    if harmonization and mem_cats_harmonization_log:
        harmonization_log_all = pd.concat(mem_cats_harmonization_log).sort_index()

    stats_var = _build_budget_stats(stats_source_all=stats_source_all, study_period=study_period)
    issue_pairs_by_variable = _issue_pairs_by_variable(drop_logs, summary_variables)
    variable_coverage_summary_counts = _coverage_summary_counts(
        summary_variables=summary_variables,
        original_pairs_by_variable=original_pairs_by_variable,
        retained_pairs_by_variable=retained_pairs_by_variable,
        issue_pairs_by_variable=issue_pairs_by_variable,
    )
    return {
        "final_all": final_all,
        "original_all": original_all,
        "harmonization_log_all": harmonization_log_all,
        "stats_var": stats_var,
        "historical_emissions": historical_emissions,
        "drop_logs": drop_logs,
        "variable_coverage_summary_counts": variable_coverage_summary_counts,
        "latest_historical_year": latest_historical_year,
        "requested_harmonization_year": requested_harmonization_year,
        "harmonization_year": harmonization_year,
        "harmonization_message": harmonization_message,
    }


def _select_variables(data_df: pd.DataFrame, variables: tuple[str, ...]) -> pd.DataFrame:
    """Return rows for the selected AR6 variable names."""
    if data_df.empty:
        return data_df.copy()
    return data_df.loc[data_df.index.isin(variables, level="variable"), :].copy()


def _pairs_for_variable(data_df: pd.DataFrame, *, variable: str) -> set[tuple[str, str]]:
    """Return model-scenario pairs present for one retained variable."""
    return {
        (str(model), str(scenario))
        for model, scenario, frame_variable in data_df.index
        if str(frame_variable) == variable
    }


def _append_category_log(
    drop_logs: list[pd.DataFrame],
    frame: pd.DataFrame,
    *,
    category: str,
) -> None:
    """Append one row issue log frame with its AR6 category column."""
    if frame.empty:
        return
    category_frame = frame.copy()
    category_frame["category"] = category
    drop_logs.append(category_frame)


def _align_companions_to_net_scope(
    *,
    companion_rows: pd.DataFrame,
    net_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Return sequestration companion rows for retained harmonized net scopes."""
    if companion_rows.empty or net_rows.empty:
        return companion_rows.iloc[0:0].reindex(columns=net_rows.columns).copy()
    net_pairs = net_rows.index.droplevel("variable").drop_duplicates()
    matched = companion_rows.loc[
        companion_rows.index.droplevel("variable").isin(net_pairs),
        :,
    ].copy()
    return matched.reindex(columns=net_rows.columns)


def _concat_non_empty(frames: list[pd.DataFrame], *, columns: pd.DataFrame) -> pd.DataFrame:
    """Concatenate frames while preserving the expected column order for empty scopes."""
    non_empty = [frame for frame in frames if not frame.empty]
    if non_empty:
        return pd.concat(non_empty, axis=0).sort_index()
    return columns.iloc[0:0].copy()


def _build_budget_stats(
    *,
    stats_source_all: pd.DataFrame,
    study_period: list[int],
) -> pd.DataFrame:
    """Return budget statistics for retained net pathways."""
    if stats_source_all.empty:
        return pd.DataFrame()
    stats_frames = []
    for curr_var in sorted(set(stats_source_all.index.get_level_values("variable"))):
        stats_harmonized = stats_from_retained_pathways(
            data_df=stats_source_all,
            var_selected=curr_var,
            timewindow_l=study_period,
        )
        old_idx = stats_harmonized.index.to_frame()
        old_idx.insert(0, "variable", [curr_var] * len(stats_harmonized))
        stats_harmonized.index = pd.MultiIndex.from_frame(old_idx)
        stats_frames.append(stats_harmonized)
    stats_var = pd.concat(stats_frames)
    return stats_var.reindex(
        columns=["median", "mean", "min", "max", "nmodel", "nscenario"]
    ).sort_index()


def _issue_pairs_by_variable(
    drop_logs: list[pd.DataFrame],
    summary_variables: tuple[str, ...],
) -> dict[str, dict[str, set[tuple[str, str]]]]:
    """Return dropped model-scenario pairs grouped by output variable and reason."""
    out: dict[str, dict[str, set[tuple[str, str]]]] = {}
    for frame in drop_logs:
        for row in frame.loc[:, ["model", "scenario", "variable", "drop_reason"]].itertuples(
            index=False,
            name=None,
        ):
            model, scenario, variable, reason = row
            variable_str = str(variable)
            if variable_str not in summary_variables:
                continue
            reason_str = str(reason)
            out.setdefault(variable_str, {}).setdefault(reason_str, set()).add(
                (str(model), str(scenario))
            )
    return out


def _coverage_summary_counts(
    *,
    summary_variables: tuple[str, ...],
    original_pairs_by_variable: dict[str, set[tuple[str, str]]],
    retained_pairs_by_variable: dict[str, set[tuple[str, str]]],
    issue_pairs_by_variable: dict[str, dict[str, set[tuple[str, str]]]],
) -> dict[str, dict[str, object]]:
    """Return variable coverage counts for process metadata and reports."""
    out: dict[str, dict[str, object]] = {}
    for variable in summary_variables:
        reason_pairs = issue_pairs_by_variable.get(variable, {})
        issue_pairs = set().union(*reason_pairs.values()) if reason_pairs else set()
        available_pairs = len(original_pairs_by_variable[variable].union(issue_pairs))
        if available_pairs <= 0:
            continue
        out[variable] = {
            "available_model_scenario_pairs": available_pairs,
            "retained_model_scenario_pairs": len(retained_pairs_by_variable[variable]),
            "missing_reason_counts": {
                reason: len(pairs) for reason, pairs in sorted(reason_pairs.items())
            },
        }
    return out
