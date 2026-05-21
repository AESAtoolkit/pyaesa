"""Helpers for loading and grouping processed pop/gdp tables."""

from pathlib import Path
from typing import Optional
from collections.abc import Sequence

import pandas as pd

from pyaesa.process.mrios.utils.io.paths import _get_group_map_path

from .source_schema import region_code_column_for_source
from .region_group_mapping import load_region_group_mapping


def _load_processed_table(path: Path) -> pd.DataFrame:
    """Load a processed pop/gdp CSV.

    Args:
        path: CSV path.

    Returns:
        DataFrame with processed pop/gdp data.
    """
    if not path.exists():
        raise FileNotFoundError(f"Processed pop/gdp file not found at {path}")
    df = pd.read_csv(path)
    df.attrs["source_csv"] = str(path)
    return df


def _source_csv_hint(df: pd.DataFrame) -> str:
    """Return a source CSV suffix for error messages when available."""
    source_csv = df.attrs.get("source_csv")
    return f" CSV: {source_csv}" if source_csv else ""


def _numeric_indexed_series(
    *,
    frame: pd.DataFrame,
    index_cols: str | list[str],
    value_col: str,
) -> pd.Series:
    """Return one numeric Series from a DataFrame column after indexing."""
    indexed = frame.set_index(index_cols)
    column_frame = pd.DataFrame(indexed.loc[:, [value_col]], copy=False)
    series = pd.Series(column_frame.iloc[:, 0], copy=False)
    numeric = pd.to_numeric(series, errors="raise")
    return pd.Series(numeric, index=series.index, name=series.name, copy=False)


def _indexed_series(
    *,
    frame: pd.DataFrame,
    index_cols: str | list[str],
    value_col: str,
) -> pd.Series:
    """Return one Series from a DataFrame column after indexing."""
    indexed = frame.set_index(index_cols)
    column_frame = pd.DataFrame(indexed.loc[:, [value_col]], copy=False)
    return pd.Series(column_frame.iloc[:, 0], copy=False)


def _duplicated_label_sample(series: pd.Series) -> list[str]:
    """Return a short stringified sample of duplicated index labels."""
    duplicates = pd.Index(series.index[series.index.duplicated()]).unique().tolist()
    return [str(value) for value in duplicates[:10]]


def _select_variable(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Filter a processed table to one variable.

    Args:
        df: Processed data.
        variable: Variable name to select.

    Returns:
        Filtered DataFrame.
    """
    if "variable" not in df.columns:
        raise ValueError(
            f"Processed pop/gdp table missing 'variable' column.{_source_csv_hint(df)}"
        )
    return pd.DataFrame(df.loc[df["variable"] == variable].copy(), copy=False)


def _apply_grouping_to_series(
    series: pd.Series,
    *,
    source_key: str,
    group_version: Optional[str],
) -> pd.Series:
    """Apply MRIO region grouping to a series.

    Args:
        series: Series indexed by MRIO region codes.
        source_key: MRIO source key.
        group_version: Grouping version tag.

    Returns:
        Grouped Series.
    """
    if not group_version:
        return series
    # Grouping is label based: values are summed only when multiple original
    # regions map to the same grouped region.
    mapping = load_region_group_mapping(
        source_key=source_key,
        group_version=group_version,
    )
    grouped = series.copy()
    missing = sorted({str(idx) for idx in grouped.index if idx not in mapping})
    if missing:
        sample = missing[:10]
        map_path = _get_group_map_path(
            source_key,
            kind="reg",
            group_version=group_version,
        )
        raise ValueError(
            "Grouping map is missing MRIO labels required by pop/gdp data. "
            f"Missing labels (sample): {sample}. CSV: {map_path}"
        )
    grouped.index = grouped.index.map(mapping.__getitem__)
    if not grouped.index.is_unique:
        grouped = pd.Series(grouped.groupby(level=0).sum(min_count=1), copy=False)
    return pd.Series(grouped, copy=False)


def _get_series_for_year(
    *,
    df: pd.DataFrame,
    variable: str,
    year: int,
    source_key: str,
    group_version: Optional[str],
    ssp_scenario: Optional[str] = None,
    region_col_override: Optional[str] = None,
) -> pd.Series:
    """Return a variable series for a given year.

    Args:
        df: Processed data.
        variable: Variable name.
        year: Year of interest.
        source_key: MRIO source key.
        group_version: Grouping version tag.
        ssp_scenario: Optional SSP scenario.

    Returns:
        Series indexed by MRIO region codes.
    """
    df_var = _select_variable(df, variable)
    year_col = str(int(year))
    if year_col not in df_var.columns:
        raise ValueError(
            f"Year {year} missing in processed pop/gdp data.{_source_csv_hint(df_var)}"
        )
    region_col = region_col_override or region_code_column_for_source(source_key)
    if region_col not in df_var.columns:
        raise ValueError(
            f"Processed pop/gdp missing column {region_col}.{_source_csv_hint(df_var)}"
        )
    series = _numeric_indexed_series(
        frame=df_var,
        index_cols=region_col,
        value_col=year_col,
    )
    if "ssp_scenario" in df_var.columns:
        if ssp_scenario is not None:
            ssp_scenario_frame = pd.DataFrame(
                df_var.loc[df_var["ssp_scenario"] == ssp_scenario].copy(),
                copy=False,
            )
            series = _numeric_indexed_series(
                frame=ssp_scenario_frame,
                index_cols=region_col,
                value_col=year_col,
            )
            series = series.dropna()
            if not series.index.is_unique:
                sample = _duplicated_label_sample(series)
                raise ValueError(
                    "Processed pop/gdp SSP scenario rows are not unique for region labels. "
                    f"Duplicate labels: {sample}.{_source_csv_hint(df_var)}"
                )
            return _apply_grouping_to_series(
                series,
                source_key=source_key,
                group_version=group_version,
            )
        # Preserve SSP scenario + region index so caller can route WB/SSP data by year.
        series = _numeric_indexed_series(
            frame=df_var,
            index_cols=["ssp_scenario", region_col],
            value_col=year_col,
        )
    series = series.dropna()
    if not series.index.is_unique:
        sample = _duplicated_label_sample(series)
        raise ValueError(
            "Processed pop/gdp rows are not unique for the selected index labels. "
            f"Duplicate labels: {sample}.{_source_csv_hint(df_var)}"
        )
    grouped = _apply_grouping_to_series(
        series,
        source_key=source_key,
        group_version=group_version,
    )
    return grouped


def _get_pr_iso3_inputs(
    *,
    df: pd.DataFrame,
    year: int,
    source_key: str,
    gdp_variable: str,
    pop_variable: str,
    ssp_scenario: Optional[str] = None,
    region_col_override: Optional[str] = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return ISO3-level GDP/Pop and ISO3->MRIO mapping.

    Args:
        df: Processed data.
        year: Year of interest.
        source_key: MRIO source key.
        gdp_variable: GDP variable name.
        pop_variable: Population variable name.
        ssp_scenario: Optional SSP scenario.

    Returns:
        Tuple of (pop_iso, gdp_iso, iso_to_mrio).
    """
    year_col = str(int(year))
    region_col = region_col_override or region_code_column_for_source(source_key)
    if "iso3_code" not in df.columns:
        raise ValueError(
            f"Processed pop/gdp missing iso3_code for PR entities.{_source_csv_hint(df)}"
        )
    if region_col not in df.columns:
        raise ValueError(f"Processed pop/gdp missing column {region_col}.{_source_csv_hint(df)}")

    df_gdp = _select_variable(df, gdp_variable)
    df_pop = _select_variable(df, pop_variable)
    if "ssp_scenario" in df_gdp.columns and ssp_scenario is not None:
        df_gdp = pd.DataFrame(
            df_gdp.loc[df_gdp["ssp_scenario"] == ssp_scenario].copy(),
            copy=False,
        )
        df_pop = pd.DataFrame(
            df_pop.loc[df_pop["ssp_scenario"] == ssp_scenario].copy(),
            copy=False,
        )
    if year_col not in df_gdp.columns or year_col not in df_pop.columns:
        raise ValueError(
            f"Year {year} missing in processed pop/gdp data.{_source_csv_hint(df_gdp)}"
        )

    gdp_iso = _numeric_indexed_series(
        frame=df_gdp,
        index_cols="iso3_code",
        value_col=year_col,
    )
    pop_iso = _numeric_indexed_series(
        frame=df_pop,
        index_cols="iso3_code",
        value_col=year_col,
    )
    gdp_iso = gdp_iso.dropna()
    pop_iso = pop_iso.dropna()
    if not gdp_iso.index.is_unique:
        sample = _duplicated_label_sample(gdp_iso)
        raise ValueError(
            "Processed GDP rows are not unique at iso3 level for PR entities. "
            f"Duplicate iso3 labels: {sample}.{_source_csv_hint(df_gdp)}"
        )
    if not pop_iso.index.is_unique:
        sample = _duplicated_label_sample(pop_iso)
        raise ValueError(
            "Processed population rows are not unique at iso3 level for PR entities. "
            f"Duplicate iso3 labels: {sample}.{_source_csv_hint(df_pop)}"
        )
    # PR(GDPcap) needs both GDP and population: keep only common ISO3 rows.
    common_iso = gdp_iso.index.intersection(pop_iso.index)
    if len(common_iso) == 0:
        raise ValueError(
            "No overlapping ISO3 coverage between GDP and population for PR entities."
            f"{_source_csv_hint(df_gdp)}"
        )
    gdp_iso = gdp_iso.reindex(common_iso)
    pop_iso = pop_iso.reindex(common_iso)

    if region_col == "iso3_code":
        # When PR inputs are already in ISO3 space, downstream routines still
        # receive the same mapping contract through the identity relation.
        iso_codes = pd.Index(df_gdp["iso3_code"].astype("string"), name="iso3_code")
        iso_to_mrio = pd.Series(iso_codes.to_numpy(copy=False), index=iso_codes, copy=False)
    else:
        iso_to_mrio = _indexed_series(
            frame=pd.DataFrame(
                {
                    "iso3_code": df_gdp["iso3_code"],
                    region_col: df_gdp[region_col].astype("string"),
                }
            ),
            index_cols="iso3_code",
            value_col=region_col,
        ).astype("string")
    if not iso_to_mrio.index.is_unique:
        sample = _duplicated_label_sample(iso_to_mrio)
        raise ValueError(
            "iso3->MRIO source mapping is not unique in processed data. "
            f"Duplicate iso3 labels: {sample}.{_source_csv_hint(df_gdp)}"
        )
    iso_to_mrio = iso_to_mrio.reindex(common_iso)
    if bool(iso_to_mrio.isna().any()):
        missing_iso_index = pd.Index(iso_to_mrio.index[iso_to_mrio.isna()])
        sample = [str(value) for value in missing_iso_index.tolist()[:10]]
        raise ValueError(
            "Missing iso3->MRIO mapping for PR entities. "
            f"Missing iso3 labels (sample): {sample}.{_source_csv_hint(df_gdp)}"
        )

    return pop_iso, gdp_iso, iso_to_mrio


def _resolve_ssp_scenarios(
    *,
    resolved_years: list[int],
    wb_df: pd.DataFrame,
    ssp_df: pd.DataFrame,
    ssp_scenario: str | Sequence[str] | None,
) -> list[str | None]:
    """Resolve SSP scenarios to use for a run."""
    ssp_scenario_list: list[str] = []
    if isinstance(ssp_scenario, str):
        ssp_scenario_list = [str(ssp_scenario)]
    elif isinstance(ssp_scenario, Sequence):
        ssp_scenario_list = [str(s).strip() for s in ssp_scenario if str(s).strip()]
        if len(ssp_scenario_list) != len(set(ssp_scenario_list)):
            seen: set[str] = set()
            duplicates: list[str] = []
            for ssp_scenario_value in ssp_scenario_list:
                if ssp_scenario_value in seen and ssp_scenario_value not in duplicates:
                    duplicates.append(ssp_scenario_value)
                seen.add(ssp_scenario_value)
            raise ValueError(
                f"Duplicate ssp_scenario values are not allowed. Duplicates: {duplicates}."
            )

    ssp_years = [y for y in resolved_years if str(int(y)) not in wb_df.columns]
    if ssp_years:
        if "ssp_scenario" not in ssp_df.columns:
            raise ValueError(
                "SSP years are required but processed SSP data has no 'ssp_scenario' "
                f"column.{_source_csv_hint(ssp_df)}"
            )
        available_ssp_scenarios = sorted(
            {str(s).strip() for s in ssp_df["ssp_scenario"].dropna() if str(s).strip()}
        )
        if ssp_scenario_list:
            missing = sorted(set(ssp_scenario_list) - set(available_ssp_scenarios))
            if missing:
                raise ValueError(
                    "Requested ssp_scenario values are unavailable in processed SSP data. "
                    f"Missing scenarios: {missing}.{_source_csv_hint(ssp_df)}"
                )
        else:
            # If SSP backed years are requested and user did not specify
            # SSP scenarios, run all SSP scenarios found in processed inputs.
            ssp_scenario_list = available_ssp_scenarios
    result: list[str | None] = list(ssp_scenario_list) if ssp_scenario_list else [None]
    return result
