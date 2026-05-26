"""Parent-region aggregation ownership for processed pop/gdp tables."""

from typing import Sequence, cast

import pandas as pd


def apply_parent_aggregation(
    df: pd.DataFrame,
    year_cols: Sequence[str],
    mapping_df: pd.DataFrame,
    *,
    name_column: str,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate child regions into parent records when requested by MRIO metadata."""
    df_work = df.copy()
    if "agg_parent" not in mapping_df.columns:
        return df_work
    agg_rows = mapping_df[mapping_df["agg_parent"].str.upper() == "YES"]
    if agg_rows.empty:
        return df_work
    year_list = list(year_cols)
    df_work[year_list] = df_work[year_list].apply(pd.to_numeric, errors="raise")
    children_by_parent = agg_rows.groupby("parent_iso3_code")["iso3_code"].apply(list)
    aggregated_children: set[str] = set()

    for parent_iso3, child_codes in children_by_parent.items():
        valid_children = [code for code in child_codes if isinstance(code, str) and code]
        if not parent_iso3 or not valid_children:
            continue
        aggregated_children.update(valid_children)
        subset = df_work[df_work["iso3_code"].isin([parent_iso3] + valid_children)]
        if subset.empty:
            continue

        aggregated_sum = cast(
            pd.DataFrame,
            subset.groupby(list(group_columns))[year_list].sum(min_count=1),
        )
        sums = cast(pd.DataFrame, aggregated_sum.reset_index())
        parent_name_series = subset.loc[subset["iso3_code"] == parent_iso3, name_column].dropna()
        parent_name = (
            str(parent_name_series.iloc[0]) if not parent_name_series.empty else str(parent_iso3)
        )

        for _, row in sums.iterrows():
            row_series = cast(pd.Series, row)
            values = cast(pd.Series, row_series.loc[year_list])
            mask = df_work["iso3_code"] == parent_iso3
            new_row: dict[str, object] = {
                name_column: parent_name,
                "iso3_code": parent_iso3,
            }
            for col in group_columns:
                group_value = row_series[col]
                mask = cast(pd.Series, mask & (df_work[col] == group_value))
                new_row[col] = group_value

            if bool(mask.any()):
                numeric_values = cast(
                    pd.Series,
                    pd.to_numeric(values.loc[year_list], errors="raise"),
                ).to_list()
                df_work.loc[mask, year_list] = numeric_values
                df_work.loc[mask, name_column] = parent_name
                continue

            for col in year_list:
                new_row[col] = values[col]
            df_work = pd.concat([df_work, pd.DataFrame([new_row])], ignore_index=True)

    if aggregated_children:
        df_work = cast(
            pd.DataFrame,
            df_work[~df_work["iso3_code"].isin(sorted(aggregated_children))],
        )
    return df_work
