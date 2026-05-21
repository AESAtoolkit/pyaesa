"""Preprocessing ownership for AR6 scenario pathways."""

from collections.abc import Sequence
from typing import cast

import pandas as pd

YEAR_MIN = 2000
YEAR_MAX = 2100
YEAR_COLUMNS = list(range(YEAR_MIN, YEAR_MAX + 1))
CH4_AR6_GWP100 = 27
N2O_AR6_GWP100 = 273
KT_TO_MT = 1e-3


def _empty_filtered_dataframe() -> pd.DataFrame:
    """Return an empty wide scenario dataframe with the expected columns."""
    my_index_all = pd.MultiIndex(
        levels=[[], [], []],
        codes=[[], [], []],
        names=["model", "scenario", "variable"],
    )
    return pd.DataFrame(
        index=my_index_all,
        columns=["Category", "Category_name", "Ssp_family", "unit", "region"] + YEAR_COLUMNS,
    )


def _require_unique_scenario_variable_rows(
    *,
    filtered_data: pd.DataFrame,
    idx_cols: list[str],
    keep_cols: Sequence[str | int],
) -> pd.DataFrame:
    """Return unique AR6 pathway rows, failing on any duplicate identity."""
    selected = filtered_data.loc[:, keep_cols].copy()
    duplicated = selected.duplicated(subset=idx_cols, keep=False)
    if not bool(duplicated.any()):
        return selected.set_index(idx_cols)
    duplicate_rows = selected.loc[duplicated, idx_cols].drop_duplicates().reset_index(drop=True)
    raise ValueError(
        "AR6 processing requires exactly one row per model-scenario variable. "
        f"Duplicate identities: {duplicate_rows.to_dict(orient='records')}"
    )


def _scenario_category_name_map(filtered_data: pd.DataFrame) -> dict[tuple[object, object], object]:
    """Return one Category_name per model-scenario, leaving absent metadata empty."""
    if "Category_name" not in filtered_data.columns:
        return {}
    category_map: dict[tuple[object, object], object] = {}
    grouped = filtered_data.loc[:, ["model", "scenario", "Category_name"]].groupby(
        ["model", "scenario"],
        sort=False,
        dropna=False,
    )
    for group_key, group in grouped:
        present_values = pd.Series(group["Category_name"], copy=False).dropna()
        values = pd.Series(
            [
                value
                for value in present_values.tolist()
                if not (isinstance(value, str) and not value.strip())
            ]
        ).drop_duplicates()
        if values.empty:
            category_map[cast(tuple[object, object], group_key)] = pd.NA
            continue
        if len(values.index) > 1:
            model, scenario = group_key
            observed = sorted(str(value) for value in values.tolist())
            raise ValueError(
                "AR6 processing found conflicting Category_name values for a model-scenario. "
                f"model='{model}', scenario='{scenario}', observed={observed}."
            )
        category_map[cast(tuple[object, object], group_key)] = values.iloc[0]
    return category_map


def filter_and_format_rawdata(explorer_df, filters_d: dict) -> pd.DataFrame:
    """Filter the wide explorer table for one category/SSP selection."""
    selected_models = set(filters_d["model"][1])
    wide_df = explorer_df.data
    required_vetting = ["Vetting_historical", "Vetting_future"]
    missing_vetting = [column for column in required_vetting if column not in wide_df.columns]
    if missing_vetting:
        raise ValueError(
            "Downloaded AR6 explorer data is missing pathway vetting metadata columns: "
            f"{missing_vetting}. Run download_ar6(refresh=True) before process_ar6()."
        )
    filter_mask = (
        wide_df["Category"].eq(filters_d["category"])
        & wide_df["Ssp_family"].eq(filters_d["ssp_family"])
        & wide_df["Vetting_historical"].eq("Pass")
        & wide_df["Vetting_future"].eq("Pass")
        & wide_df["model"].isin(selected_models)
    )
    filtered_data = wide_df.loc[filter_mask, :].copy()
    if filtered_data.empty:
        return _empty_filtered_dataframe()

    idx_cols = ["model", "scenario", "variable"]
    wide_year_cols = [col for col in filtered_data.columns if str(col).isdigit()]
    available_years = sorted([int(col) for col in wide_year_cols if int(col) in YEAR_COLUMNS])
    if not available_years:
        return _empty_filtered_dataframe()

    year_cols_map = {int(col): col for col in wide_year_cols if int(col) in YEAR_COLUMNS}
    keep_cols = ["model", "scenario", "variable", "unit", "region"] + [
        year_cols_map[year] for year in available_years
    ]
    filtered_dense = _require_unique_scenario_variable_rows(
        filtered_data=filtered_data,
        idx_cols=idx_cols,
        keep_cols=keep_cols,
    )
    filtered_dense = filtered_dense.rename(
        columns={year_cols_map[year]: year for year in available_years}
    )
    raw_data_dense_df = filtered_dense.reindex(columns=["unit", "region"] + YEAR_COLUMNS)
    category_name_map = _scenario_category_name_map(filtered_data)
    category_name_values = []
    for model, scenario, _variable in raw_data_dense_df.index:
        category_name_values.append(category_name_map.get((model, scenario), pd.NA))
    raw_data_dense_df.insert(loc=0, column="Ssp_family", value=filters_d["ssp_family"])
    raw_data_dense_df.insert(loc=0, column="Category_name", value=category_name_values)
    raw_data_dense_df.insert(loc=0, column="Category", value=filters_d["category"])
    return raw_data_dense_df


def interpolate_and_check(data_df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate only internal yearly gaps and preserve truncated tails."""
    interpolated_data_df = data_df.copy()
    years = YEAR_COLUMNS
    interpolated_years_df = (
        interpolated_data_df.loc[:, years].astype("float").interpolate(axis=1, limit_area="inside")
    )
    interpolated_data_df.loc[:, years] = interpolated_years_df
    return interpolated_data_df
