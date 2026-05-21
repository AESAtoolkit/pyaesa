"""Dataframe loaders used by AR6 processing."""

from typing import cast

import pandas as pd


def _unique_metadata_value(
    *,
    group: pd.DataFrame,
    column: str,
    model: object,
    scenario: object,
) -> object:
    """Return one metadata value, failing on conflicting non missing values."""
    values = pd.Series(group[column], copy=False).dropna().drop_duplicates()
    if len(values.index) > 1:
        observed = sorted(str(value) for value in values.tolist())
        raise ValueError(
            "AR6 source metadata is not unique for a model-scenario. "
            f"model='{model}', scenario='{scenario}', column='{column}', observed={observed}."
        )
    if values.empty:
        return pd.NA
    return values.iloc[0]


def scenario_metadata_from_wide(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Return unique scenario metadata rows indexed by model-scenario."""
    year_cols = [col for col in wide_df.columns if str(col).isdigit()]
    key_cols = ["model", "scenario", "variable", "unit", "region"]
    meta_cols = [col for col in wide_df.columns if col not in key_cols and col not in year_cols]
    metadata_df = pd.DataFrame(wide_df.loc[:, ["model", "scenario"] + meta_cols])
    if not meta_cols:
        return (
            metadata_df.loc[:, ["model", "scenario"]]
            .drop_duplicates()
            .set_index(["model", "scenario"])
            .sort_index()
        )
    records: list[dict[str, object]] = []
    for group_key, group in metadata_df.groupby(["model", "scenario"], sort=True, dropna=False):
        model, scenario = cast(tuple[object, object], group_key)
        record: dict[str, object] = {"model": model, "scenario": scenario}
        for column in meta_cols:
            record[column] = _unique_metadata_value(
                group=group,
                column=column,
                model=model,
                scenario=scenario,
            )
        records.append(record)
    return pd.DataFrame.from_records(records).set_index(["model", "scenario"]).sort_index()
