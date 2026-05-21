"""Derived AR6 variable construction for process_ar6."""

from typing import cast

import pandas as pd

from pyaesa.download.ar6.utils.config import (
    GROSS_ALT_CO2_WITH_AFOLU,
    GROSS_ALT_CO2_WO_AFOLU,
    GROSS_ALT_KYOTO_WITH_AFOLU,
    GROSS_ALT_KYOTO_WO_AFOLU,
    GROSS_CO2_WITH_AFOLU,
    GROSS_CO2_WO_AFOLU,
    GROSS_KYOTO_WITH_AFOLU,
    GROSS_KYOTO_WO_AFOLU,
    NET_CO2_WITH_AFOLU,
    NET_CO2_WO_AFOLU,
    NET_KYOTO_WITH_AFOLU,
    NET_KYOTO_WO_AFOLU,
    RAW_CH4_AFOLU,
    RAW_CO2_AFOLU,
    RAW_CO2_WITH_AFOLU,
    RAW_KYOTO_WITH_AFOLU,
    RAW_N2O_AFOLU,
    RAW_SEQUESTRATION_COMPONENTS,
    RAW_SEQUESTRATION_SUBTOTAL_COMPONENTS,
    SEQUESTRATION_SUBTOTAL,
    SEQUESTRATION_TOTAL,
)

from .co2_reconstruction import (
    CO2_RECONSTRUCTION_DROP_REASON as _CO2_RECONSTRUCTION_DROP_REASON,
    drop_co2_reconstruction_failed_pairs,
)
from .preprocessing import CH4_AR6_GWP100, KT_TO_MT, N2O_AR6_GWP100, YEAR_COLUMNS

CO2_RECONSTRUCTION_DROP_REASON = _CO2_RECONSTRUCTION_DROP_REASON
CO2_WO_AFOLU_NOT_PRODUCED_REASON = "wo_afolu_not_produced_missing_co2_afolu_component_keep_co2"
KYOTO_WO_AFOLU_NOT_PRODUCED_REASON = (
    "wo_afolu_not_produced_missing_kyoto_afolu_components_keep_kyoto"
)
MISSING_REQUIRED_CO2_COVERAGE_ROW_REASON = "missing_required_co2_coverage_row"
MISSING_REQUIRED_END_YEAR_REASON = "missing_value_at_required_end_year_2100"
NEGATIVE_SEQUESTRATION_DROP_REASON = "negative_carbon_sequestration_value"
NEGATIVE_GROSS_EMISSIONS_DROP_REASON = "negative_gross_emissions_value"
PRE_RECONSTRUCTION_COVERAGE_STAGE = "pre_reconstruction_coverage_check"
REQUIRED_COVERAGE_END_YEAR = 2100

NET_HARMONIZATION_VARIABLES = (
    NET_CO2_WITH_AFOLU,
    NET_CO2_WO_AFOLU,
    NET_KYOTO_WITH_AFOLU,
    NET_KYOTO_WO_AFOLU,
)
SEQUESTRATION_COMPANION_VARIABLES = (SEQUESTRATION_TOTAL, SEQUESTRATION_SUBTOTAL)
GROSS_FILTER_VARIABLES = (
    GROSS_CO2_WITH_AFOLU,
    GROSS_CO2_WO_AFOLU,
    GROSS_KYOTO_WITH_AFOLU,
    GROSS_KYOTO_WO_AFOLU,
    GROSS_ALT_CO2_WITH_AFOLU,
    GROSS_ALT_CO2_WO_AFOLU,
    GROSS_ALT_KYOTO_WITH_AFOLU,
    GROSS_ALT_KYOTO_WO_AFOLU,
)
GROSS_CONSTRUCTION_RULES = (
    (NET_CO2_WITH_AFOLU, SEQUESTRATION_TOTAL, GROSS_CO2_WITH_AFOLU),
    (NET_CO2_WO_AFOLU, SEQUESTRATION_TOTAL, GROSS_CO2_WO_AFOLU),
    (NET_KYOTO_WITH_AFOLU, SEQUESTRATION_TOTAL, GROSS_KYOTO_WITH_AFOLU),
    (NET_KYOTO_WO_AFOLU, SEQUESTRATION_TOTAL, GROSS_KYOTO_WO_AFOLU),
    (NET_CO2_WITH_AFOLU, SEQUESTRATION_SUBTOTAL, GROSS_ALT_CO2_WITH_AFOLU),
    (NET_CO2_WO_AFOLU, SEQUESTRATION_SUBTOTAL, GROSS_ALT_CO2_WO_AFOLU),
    (NET_KYOTO_WITH_AFOLU, SEQUESTRATION_SUBTOTAL, GROSS_ALT_KYOTO_WITH_AFOLU),
    (NET_KYOTO_WO_AFOLU, SEQUESTRATION_SUBTOTAL, GROSS_ALT_KYOTO_WO_AFOLU),
)


def build_pre_harmonization_variables(
    data_df: pd.DataFrame,
    *,
    study_start_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw plus pre harmonization AR6 derived rows and row issue logs."""
    if data_df.empty:
        return data_df, pd.DataFrame()
    coverage_checked, coverage_log = _drop_missing_required_co2_coverage_pairs(
        data_df,
        study_start_year=int(study_start_year),
    )
    if coverage_checked.empty:
        return coverage_checked, coverage_log
    reconstruction_checked, reconstruction_log = drop_co2_reconstruction_failed_pairs(
        coverage_checked
    )
    if reconstruction_checked.empty:
        logs = [frame for frame in [coverage_log, reconstruction_log] if not frame.empty]
        return reconstruction_checked, pd.concat(logs, ignore_index=True)
    sequestration_frames = [
        _sequestration_rows(
            reconstruction_checked,
            variable=SEQUESTRATION_TOTAL,
            components=RAW_SEQUESTRATION_COMPONENTS,
        ),
        _sequestration_rows(
            reconstruction_checked,
            variable=SEQUESTRATION_SUBTOTAL,
            components=RAW_SEQUESTRATION_SUBTOTAL_COMPONENTS,
        ),
    ]
    with_sequestration, sequestration_log = _drop_negative_variable_pairs(
        pd.concat(
            [
                reconstruction_checked,
                *[frame for frame in sequestration_frames if not frame.empty],
            ],
            axis=0,
        ).sort_index(),
        variables=(*RAW_SEQUESTRATION_COMPONENTS, SEQUESTRATION_TOTAL, SEQUESTRATION_SUBTOTAL),
        reason=NEGATIVE_SEQUESTRATION_DROP_REASON,
        stage="sequestration_sign_check",
    )
    if with_sequestration.empty:
        return with_sequestration, sequestration_log

    co2_wo_rows, co2_wo_log = _co2_wo_afolu_rows(with_sequestration)
    kyoto_wo_rows, kyoto_wo_log = _kyoto_wo_afolu_rows(with_sequestration)
    derived_frames = [
        with_sequestration,
        _alias_rows(
            with_sequestration,
            raw_variable=RAW_CO2_WITH_AFOLU,
            alias_variable=NET_CO2_WITH_AFOLU,
        ),
        _alias_rows(
            with_sequestration,
            raw_variable=RAW_KYOTO_WITH_AFOLU,
            alias_variable=NET_KYOTO_WITH_AFOLU,
        ),
        co2_wo_rows,
        kyoto_wo_rows,
    ]
    out = pd.concat([frame for frame in derived_frames if not frame.empty], axis=0).sort_index()
    logs = [
        frame
        for frame in [coverage_log, reconstruction_log, sequestration_log, co2_wo_log, kyoto_wo_log]
        if not frame.empty
    ]
    return out, pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()


def build_final_harmonized_emission_variables(
    data_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return final rows, gross sign logs, and rows before gross sign filtering."""
    if data_df.empty:
        return data_df, pd.DataFrame(), data_df
    gross_frames = [
        _gross_rows(
            data_df,
            net_variable=net_variable,
            sequestration_variable=sequestration_variable,
            gross_variable=gross_variable,
        )
        for net_variable, sequestration_variable, gross_variable in GROSS_CONSTRUCTION_RULES
    ]
    before_sign_filter = pd.concat(
        [data_df, *[frame for frame in gross_frames if not frame.empty]],
        axis=0,
    ).sort_index()
    out, gross_log = _drop_negative_gross_variable_pairs(before_sign_filter)
    return out, gross_log, before_sign_filter


def _variable_rows(data_df: pd.DataFrame, variable: str) -> pd.DataFrame:
    if variable not in data_df.index.get_level_values("variable"):
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["model", "scenario"]),
            columns=data_df.columns,
        )
    return cast(pd.DataFrame, data_df.xs(variable, level="variable", drop_level=True))


def _drop_missing_required_co2_coverage_pairs(
    data_df: pd.DataFrame,
    *,
    study_start_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove model-scenario pairs without required raw CO2 coverage."""
    all_pairs = data_df.index.droplevel("variable").drop_duplicates()
    co2 = _variable_rows(data_df, RAW_CO2_WITH_AFOLU)
    bad_pairs = all_pairs.difference(co2.index)
    bad_reason = pd.Series(MISSING_REQUIRED_CO2_COVERAGE_ROW_REASON, index=bad_pairs)
    if not co2.empty:
        co2_required_years = co2.loc[:, [int(study_start_year), REQUIRED_COVERAGE_END_YEAR]].apply(
            pd.to_numeric,
            errors="raise",
        )
        missing_start = co2.index[co2_required_years.loc[:, int(study_start_year)].isna()]
        missing_2100 = co2.index[
            co2_required_years.loc[:, int(study_start_year)].notna()
            & co2_required_years.loc[:, REQUIRED_COVERAGE_END_YEAR].isna()
        ]
        bad_reason = pd.concat(
            [
                bad_reason,
                pd.Series(
                    f"missing_value_at_study_start_year_{int(study_start_year)}",
                    index=missing_start,
                ),
                pd.Series(MISSING_REQUIRED_END_YEAR_REASON, index=missing_2100),
            ]
        )
    if bad_reason.empty:
        return data_df, pd.DataFrame()
    retained = data_df.loc[~data_df.index.droplevel("variable").isin(bad_reason.index)].copy()
    ssp_by_pair = data_df["Ssp_family"].groupby(level=["model", "scenario"], sort=False).first()
    logs = pd.DataFrame(
        {
            "model": bad_reason.index.get_level_values("model").astype(str),
            "scenario": bad_reason.index.get_level_values("scenario").astype(str),
            "variable": RAW_CO2_WITH_AFOLU,
            "retained_variable": pd.NA,
            "ssp_family": ssp_by_pair.reindex(bad_reason.index).to_numpy(copy=False),
            "drop_reason": bad_reason.to_numpy(copy=False),
            "drop_stage": PRE_RECONSTRUCTION_COVERAGE_STAGE,
        }
    )
    return retained, logs


def _year_values(frame: pd.DataFrame) -> pd.DataFrame:
    year_columns = [year for year in YEAR_COLUMNS if year in frame.columns]
    return cast(pd.DataFrame, frame.loc[:, year_columns].astype(float))


def _rows_from_template(
    *,
    template: pd.DataFrame,
    variable: str,
    values: pd.DataFrame,
) -> pd.DataFrame:
    out = template.copy()
    year_columns = [year for year in values.columns if year in out.columns]
    out.loc[:, year_columns] = values.loc[:, year_columns].to_numpy(dtype=float)
    index_frame = out.index.to_frame(index=False)
    index_frame["variable"] = variable
    out.index = pd.MultiIndex.from_frame(index_frame)
    return out


def _alias_rows(data_df: pd.DataFrame, *, raw_variable: str, alias_variable: str) -> pd.DataFrame:
    source = _variable_rows(data_df, raw_variable)
    if source.empty:
        return pd.DataFrame()
    return _rows_from_template(
        template=source,
        variable=alias_variable,
        values=_year_values(source),
    )


def _co2_wo_afolu_rows(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    co2 = _variable_rows(data_df, RAW_CO2_WITH_AFOLU)
    afolu = _variable_rows(data_df, RAW_CO2_AFOLU)
    pairs = co2.index.intersection(afolu.index)
    if pairs.empty:
        return pd.DataFrame(), _missing_derived_log(
            source=co2,
            variable=NET_CO2_WO_AFOLU,
            retained_variable=RAW_CO2_WITH_AFOLU,
            reason=CO2_WO_AFOLU_NOT_PRODUCED_REASON,
        )
    values = _year_values(co2.loc[pairs]) - _year_values(afolu.loc[pairs])
    rows = _rows_from_template(
        template=co2.loc[pairs],
        variable=NET_CO2_WO_AFOLU,
        values=values,
    )
    missing = co2.index.difference(pairs)
    log = _missing_derived_log(
        source=co2.loc[missing],
        variable=NET_CO2_WO_AFOLU,
        retained_variable=RAW_CO2_WITH_AFOLU,
        reason=CO2_WO_AFOLU_NOT_PRODUCED_REASON,
    )
    return rows, log


def _kyoto_wo_afolu_rows(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kyoto = _variable_rows(data_df, RAW_KYOTO_WITH_AFOLU)
    components = (
        (_variable_rows(data_df, RAW_CO2_AFOLU), 1.0),
        (_variable_rows(data_df, RAW_CH4_AFOLU), CH4_AR6_GWP100),
        (_variable_rows(data_df, RAW_N2O_AFOLU), N2O_AR6_GWP100 * KT_TO_MT),
    )
    if kyoto.empty:
        return pd.DataFrame(), _missing_derived_log(
            source=kyoto,
            variable=NET_KYOTO_WO_AFOLU,
            retained_variable=RAW_KYOTO_WITH_AFOLU,
            reason=KYOTO_WO_AFOLU_NOT_PRODUCED_REASON,
        )
    present_pairs: pd.Index = kyoto.index
    component_pairs: pd.Index = pd.Index([])
    for frame, _factor in components:
        component_pairs = component_pairs.union(frame.index)
    present_pairs = present_pairs.intersection(component_pairs)
    if present_pairs.empty:
        return pd.DataFrame(), _missing_derived_log(
            source=kyoto,
            variable=NET_KYOTO_WO_AFOLU,
            retained_variable=RAW_KYOTO_WITH_AFOLU,
            reason=KYOTO_WO_AFOLU_NOT_PRODUCED_REASON,
        )
    component_values = pd.DataFrame(0.0, index=present_pairs, columns=YEAR_COLUMNS)
    for frame, factor in components:
        aligned = _year_values(frame).reindex(present_pairs)
        component_values = component_values + aligned.fillna(0.0) * factor
    afolu_values = component_values.loc[present_pairs]
    afolu_rows = _rows_from_template(
        template=kyoto.loc[present_pairs],
        variable="Emissions|Kyoto Gases|AFOLU",
        values=afolu_values,
    )
    wo_values = _year_values(kyoto.loc[present_pairs]) - afolu_values
    wo_rows = _rows_from_template(
        template=kyoto.loc[present_pairs],
        variable=NET_KYOTO_WO_AFOLU,
        values=wo_values,
    )
    missing = kyoto.index.difference(present_pairs)
    log = _missing_derived_log(
        source=kyoto.loc[missing],
        variable=NET_KYOTO_WO_AFOLU,
        retained_variable=RAW_KYOTO_WITH_AFOLU,
        reason=KYOTO_WO_AFOLU_NOT_PRODUCED_REASON,
    )
    return pd.concat([afolu_rows, wo_rows], axis=0), log


def _missing_derived_log(
    *,
    source: pd.DataFrame,
    variable: str,
    retained_variable: str,
    reason: str,
) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "model": source.index.get_level_values("model").astype(str),
            "scenario": source.index.get_level_values("scenario").astype(str),
            "variable": variable,
            "retained_variable": retained_variable,
            "ssp_family": source["Ssp_family"].to_numpy(copy=False),
            "drop_reason": reason,
            "drop_stage": "derived_variable_construction",
        }
    )


def _drop_negative_variable_pairs(
    data_df: pd.DataFrame, *, variables: tuple[str, ...], reason: str, stage: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = data_df.loc[data_df.index.isin(variables, level="variable")]
    if selected.empty:
        return data_df, pd.DataFrame()
    year_columns = [year for year in YEAR_COLUMNS if year in selected.columns]
    values = selected.loc[:, year_columns].apply(pd.to_numeric, errors="raise")
    bad_rows = selected.index[values.min(axis=1).lt(0.0)]
    if bad_rows.empty:
        return data_df, pd.DataFrame()
    bad_pairs = bad_rows.droplevel("variable").drop_duplicates()
    retained = data_df.loc[~data_df.index.droplevel("variable").isin(bad_pairs)].copy()
    logs = pd.DataFrame(
        {
            "model": bad_rows.get_level_values("model").astype(str),
            "scenario": bad_rows.get_level_values("scenario").astype(str),
            "variable": bad_rows.get_level_values("variable").astype(str),
            "retained_variable": pd.NA,
            "ssp_family": selected.loc[bad_rows, "Ssp_family"].to_numpy(copy=False),
            "drop_reason": reason,
            "drop_stage": stage,
        }
    )
    return retained, logs


def _drop_negative_gross_variable_pairs(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = data_df.loc[data_df.index.isin(GROSS_FILTER_VARIABLES, level="variable")]
    year_columns = [year for year in YEAR_COLUMNS if year in selected.columns]
    values = selected.loc[:, year_columns].apply(pd.to_numeric, errors="raise")
    bad_rows = selected.index[values.min(axis=1).lt(0.0)]
    if bad_rows.empty:
        return data_df, pd.DataFrame()
    retained = data_df.loc[~data_df.index.isin(bad_rows)].copy()
    logs = pd.DataFrame(
        {
            "model": bad_rows.get_level_values("model").astype(str),
            "scenario": bad_rows.get_level_values("scenario").astype(str),
            "variable": bad_rows.get_level_values("variable").astype(str),
            "retained_variable": pd.NA,
            "ssp_family": selected.loc[bad_rows, "Ssp_family"].to_numpy(copy=False),
            "drop_reason": NEGATIVE_GROSS_EMISSIONS_DROP_REASON,
            "drop_stage": "gross_emissions_sign_check",
        }
    )
    return retained, logs


def _sequestration_rows(
    data_df: pd.DataFrame, *, variable: str, components: tuple[str, ...]
) -> pd.DataFrame:
    frames = [_variable_rows(data_df, component) for component in components]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    pairs = frames[0].index
    for frame in frames[1:]:
        pairs = pairs.union(frame.index)
    year_columns = [year for year in YEAR_COLUMNS if year in data_df.columns]
    metadata_columns = [column for column in data_df.columns if column not in year_columns]
    template = pd.DataFrame(index=pairs, columns=data_df.columns)
    template.loc[:, metadata_columns] = (
        pd.concat([frame.loc[:, metadata_columns] for frame in frames], axis=0)
        .groupby(level=["model", "scenario"], sort=False)
        .first()
        .reindex(pairs)
    )
    values = pd.DataFrame(0.0, index=pairs, columns=year_columns)
    present = pd.Series(False, index=pairs, dtype=bool)
    for frame in frames:
        aligned = _year_values(frame).reindex(pairs)
        values = values + aligned.fillna(0.0)
        present = present | aligned.notna().any(axis=1)
    return _rows_from_template(
        template=template.loc[present],
        variable=variable,
        values=values.loc[present],
    )


def _gross_rows(
    data_df: pd.DataFrame,
    *,
    net_variable: str,
    sequestration_variable: str,
    gross_variable: str,
) -> pd.DataFrame:
    net = _variable_rows(data_df, net_variable)
    sequestration = _variable_rows(data_df, sequestration_variable)
    pairs = net.index.intersection(sequestration.index)
    if pairs.empty:
        return pd.DataFrame()
    values = _year_values(net.loc[pairs]) + _year_values(sequestration.loc[pairs])
    return _rows_from_template(
        template=net.loc[pairs],
        variable=gross_variable,
        values=values,
    )
