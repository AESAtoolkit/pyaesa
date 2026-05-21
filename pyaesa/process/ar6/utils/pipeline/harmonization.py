"""Harmonization ownership for AR6 climate pathways."""

from typing import cast

import numpy as np
import pandas as pd

from pyaesa.download.ar6.utils.config import (
    NET_CO2_WITH_AFOLU,
    NET_CO2_WO_AFOLU,
    NET_KYOTO_WITH_AFOLU,
    NET_KYOTO_WO_AFOLU,
)

from .preprocessing import YEAR_COLUMNS, YEAR_MIN


def _require_row_series(data_df: pd.DataFrame, row_label: str, *, context: str) -> pd.Series:
    """Return the row series for one historical variable."""
    del context
    return data_df.loc[row_label]


def _column_position(columns: pd.Index, column_name: str) -> int:
    """Return one unique integer position for ``column_name``."""
    return cast(int, columns.get_loc(column_name))


def _resolve_offset_variant(row_values: np.ndarray, year_cols: list[int]) -> tuple[str, float]:
    """Return the internal offset variant and the model net zero proxy year."""
    tmp_idx_negative = np.where(row_values < 0)[0]
    if len(tmp_idx_negative) == 0:
        return "constant_offset", np.nan
    return "reduced_offset", float(year_cols[tmp_idx_negative[0] - 1])


def _initial_offset_horizon_year(
    *,
    row_end_year: int,
    model_netzero_year: float,
    offset_variant: str,
) -> int:
    """Return the initial harmonization horizon for one internal offset variant."""
    if offset_variant == "constant_offset":
        return int(row_end_year)
    return int(model_netzero_year)


def get_stats_from_series(data_s: pd.Series) -> dict[str, float]:
    """Return descriptive statistics for one numeric series."""
    arr = data_s.to_numpy(dtype=float)
    return {
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def stats_from_retained_pathways(data_df: pd.DataFrame, var_selected: str, timewindow_l: list):
    """Compute category and SSP summary statistics from retained final pathways."""
    data_all_cats_df = data_df.copy()
    all_categories_l = list(sorted(set(data_all_cats_df["Category"])))
    all_ssps_l = list(sorted(set(data_all_cats_df["Ssp_family"])))
    data_var_df = data_all_cats_df.loc[(slice(None), slice(None), var_selected), :]
    my_mi = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["Category", "Ssp_family"])
    stats_df = pd.DataFrame(
        index=my_mi,
        columns=["nmodel", "nscenario", "median", "mean", "min", "max"],
    )
    for curr_cat in all_categories_l:
        tmp_filtered_df = pd.DataFrame(
            data_var_df.loc[
                (data_all_cats_df["Category"] == curr_cat),
                range(timewindow_l[0], timewindow_l[1] + 1),
            ]
        )
        if len(tmp_filtered_df) > 0:
            stats_df.loc[(curr_cat, "all"), ["median", "mean", "min", "max"]] = (
                get_stats_from_series(tmp_filtered_df.sum(axis=1))
            )
            stats_df.loc[(curr_cat, "all"), "nmodel"] = len(
                set(tmp_filtered_df.index.get_level_values(0))
            )
            stats_df.loc[(curr_cat, "all"), "nscenario"] = len(tmp_filtered_df)
        for curr_ssp in all_ssps_l:
            tmp_filtered_df = pd.DataFrame(
                data_var_df.loc[
                    (data_all_cats_df["Ssp_family"] == curr_ssp)
                    & (data_all_cats_df["Category"] == curr_cat),
                    range(timewindow_l[0], timewindow_l[1] + 1),
                ]
            )
            if len(tmp_filtered_df) > 0:
                stats_df.loc[(curr_cat, int(curr_ssp)), ["median", "mean", "min", "max"]] = (
                    get_stats_from_series(tmp_filtered_df.sum(axis=1))
                )
                stats_df.loc[(curr_cat, int(curr_ssp)), "nmodel"] = len(
                    set(tmp_filtered_df.index.get_level_values(0))
                )
                stats_df.loc[(curr_cat, int(curr_ssp)), "nscenario"] = len(tmp_filtered_df)
    return stats_df


def harmonize_emissions(
    data_df: pd.DataFrame,
    historic_data_df: pd.DataFrame,
    study_timeperiod: list,
    requested_harmonization_year: int | None = None,
    harmonization_method: str = "offset",
):
    """Harmonize retained pathways to the historical baseline."""
    keep_original_index = data_df.index
    year_cols = YEAR_COLUMNS
    cols_to_keep = [col for col in data_df.columns if col not in year_cols]
    pathways_harmonized_df = data_df.copy()
    data_year_values = data_df.loc[:, year_cols].astype(float).to_numpy(copy=True)
    harmonized_year_values = data_year_values.copy()
    scenario_start_year_df = data_df.loc[:, cols_to_keep].copy()
    extra_cols = [
        "model-base-year",
        "pathway-last-year",
        "harmonization-year-requested",
        "harmonization-year",
        "offset-variant-used",
        "model-netzero-year",
        "harmonization-netzero-year",
        "horizon-for-harmonization",
        "harmonization-method-note",
        "pathway-cumulative",
        "historic-cumulative",
        "delta-cumulative",
        "yearly-correction",
    ]
    for col in extra_cols:
        scenario_start_year_df[col] = np.nan
    scenario_start_year_df["harmonization-method"] = pd.Series(
        [None] * len(scenario_start_year_df),
        index=scenario_start_year_df.index,
        dtype="object",
    )
    scenario_start_year_df["offset-variant-used"] = pd.Series(
        [None] * len(scenario_start_year_df),
        index=scenario_start_year_df.index,
        dtype="object",
    )
    scenario_start_year_df["harmonization-method-note"] = pd.Series(
        [None] * len(scenario_start_year_df),
        index=scenario_start_year_df.index,
        dtype="object",
    )
    year_to_pos = {year: idx for idx, year in enumerate(year_cols)}
    harmonization_year = int(study_timeperiod[0])
    harmonization_year_requested = int(requested_harmonization_year or harmonization_year)
    harmonization_pos = year_to_pos[harmonization_year]
    first_corr_pos = year_to_pos.get(harmonization_year + 1, harmonization_pos + 1)
    tracked_vars = {
        NET_CO2_WITH_AFOLU,
        NET_CO2_WO_AFOLU,
        NET_KYOTO_WITH_AFOLU,
        NET_KYOTO_WO_AFOLU,
    }
    historic_lookup = {}
    for var_name in tracked_vars:
        if var_name not in historic_data_df.index:
            continue
        hist_series = _require_row_series(
            historic_data_df,
            var_name,
            context="Historical harmonization lookup",
        )
        hist_arr = np.full(len(year_cols), np.nan, dtype=float)
        for year in year_cols:
            if year in hist_series.index:
                hist_arr[year_to_pos[year]] = float(hist_series.loc[year])
        historic_lookup[var_name] = hist_arr

    _col_base_year = _column_position(scenario_start_year_df.columns, "model-base-year")
    _col_pathway_last = _column_position(scenario_start_year_df.columns, "pathway-last-year")
    _col_harm_year_requested = _column_position(
        scenario_start_year_df.columns,
        "harmonization-year-requested",
    )
    _col_harm_year = _column_position(scenario_start_year_df.columns, "harmonization-year")
    _col_pathway_cumul = _column_position(
        scenario_start_year_df.columns,
        "pathway-cumulative",
    )
    _col_netzero = _column_position(scenario_start_year_df.columns, "model-netzero-year")
    _col_historic_cumul = _column_position(
        scenario_start_year_df.columns,
        "historic-cumulative",
    )
    _col_delta_cumul = _column_position(scenario_start_year_df.columns, "delta-cumulative")
    _col_harm_method = _column_position(
        scenario_start_year_df.columns,
        "harmonization-method",
    )
    _col_offset_variant = _column_position(
        scenario_start_year_df.columns,
        "offset-variant-used",
    )
    _col_harm_method_note = _column_position(
        scenario_start_year_df.columns,
        "harmonization-method-note",
    )
    _col_horizon = _column_position(
        scenario_start_year_df.columns,
        "horizon-for-harmonization",
    )
    _col_yearly_corr = _column_position(
        scenario_start_year_df.columns,
        "yearly-correction",
    )
    _col_harm_netzero = _column_position(
        scenario_start_year_df.columns,
        "harmonization-netzero-year",
    )

    for i, mi in enumerate(data_df.index):
        row_values = data_year_values[i]
        finite_mask = np.isfinite(row_values)
        if not finite_mask.any():
            continue
        finite_positions = np.flatnonzero(finite_mask)
        start_pos = int(finite_positions[0])
        end_pos = int(finite_positions[-1])
        tmp_start_year = year_cols[start_pos]
        row_end_year = year_cols[end_pos]
        scenario_start_year_df.iat[i, _col_base_year] = tmp_start_year
        scenario_start_year_df.iat[i, _col_pathway_last] = row_end_year
        scenario_start_year_df.iat[i, _col_harm_year_requested] = harmonization_year_requested
        scenario_start_year_df.iat[i, _col_harm_year] = harmonization_year
        pathway_cumulative = np.nansum(row_values[start_pos : harmonization_pos + 1])
        scenario_start_year_df.iat[i, _col_pathway_cumul] = pathway_cumulative
        offset_variant, model_netzero_year = _resolve_offset_variant(row_values, year_cols)
        if not np.isnan(model_netzero_year):
            scenario_start_year_df.iat[i, _col_netzero] = model_netzero_year

        var_name = mi[-1]
        if var_name not in tracked_vars or var_name not in historic_lookup:
            continue

        historic_row = historic_lookup[var_name]
        historic_cumulative = np.nansum(historic_row[start_pos : harmonization_pos + 1])
        historic_harmonization_value = historic_row[harmonization_pos]
        scenario_start_year_df.iat[i, _col_historic_cumul] = historic_cumulative
        delta_cumulative = pathway_cumulative - historic_cumulative
        scenario_start_year_df.iat[i, _col_delta_cumul] = delta_cumulative
        row_harmonized = harmonized_year_values[i].copy()
        method_note_parts: list[str] = []
        horizon_year = _initial_offset_horizon_year(
            row_end_year=int(row_end_year),
            model_netzero_year=model_netzero_year,
            offset_variant=offset_variant,
        )
        span_years = horizon_year - harmonization_year
        yearly_correction = 0.0 if span_years <= 0 else delta_cumulative / span_years
        initial_horizon_year = horizon_year
        # UNCASExt-specific safeguard: reduce the effective harmonization horizon until
        # the first negative emissions year is preserved after correction.
        while horizon_year > harmonization_year + 1:
            horizon_pos = year_to_pos[horizon_year]
            tmp_slice = row_harmonized[first_corr_pos : horizon_pos + 1]
            tmp_min = np.nanmin(tmp_slice) if len(tmp_slice) else np.inf
            if tmp_min + yearly_correction >= 0:
                break
            horizon_year -= 1
            span_years = horizon_year - harmonization_year
            yearly_correction = 0.0 if span_years <= 0 else delta_cumulative / span_years
        if horizon_year != initial_horizon_year:
            method_note_parts.append(
                "pyaesa reduced the effective harmonization horizon to preserve the first "
                "negative-emissions year."
            )
        method_note = " ".join(method_note_parts) or None
        scenario_start_year_df.iat[i, _col_harm_method] = harmonization_method
        scenario_start_year_df.iat[i, _col_offset_variant] = offset_variant
        scenario_start_year_df.iat[i, _col_harm_method_note] = method_note
        scenario_start_year_df.iat[i, _col_horizon] = horizon_year
        scenario_start_year_df.iat[i, _col_yearly_corr] = yearly_correction
        row_harmonized[harmonization_pos] = historic_harmonization_value
        horizon_pos = year_to_pos[int(horizon_year)]
        row_harmonized[first_corr_pos : horizon_pos + 1] += yearly_correction
        harmonized_year_values[i] = row_harmonized
        tmp_idx_negative_postharmo = np.where(row_harmonized < 0)[0]
        harmonization_netzero_year = np.nan
        if len(tmp_idx_negative_postharmo) > 0:
            harmonization_netzero_year = year_cols[tmp_idx_negative_postharmo[0] - 1]
            scenario_start_year_df.iat[i, _col_harm_netzero] = harmonization_netzero_year

    pathways_harmonized_df.loc[:, year_cols] = harmonized_year_values
    pathways_harmonized_df.drop(range(YEAR_MIN, study_timeperiod[0]), axis=1, inplace=True)
    pathways_harmonized_df = pathways_harmonized_df.loc[keep_original_index, :]
    scenario_start_year_df = scenario_start_year_df.loc[keep_original_index, :]
    return pathways_harmonized_df, scenario_start_year_df
