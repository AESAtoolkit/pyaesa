"""Dynamic deterministic ASR aCC and LCA figure row preparation."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.asr.figures.common import visible_values
from pyaesa.asr.figures.transitions import ASR_TRANSITION_SERIES_EXCLUDED_COLUMNS
from pyaesa.shared.figures.deterministic_variant_compressor import (
    MIN_ROLE,
    ROLE_COLUMN,
    VARIANT_COLUMNS,
)
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    EXT_LCA_SSP_SCENARIO_COLUMN,
)
from pyaesa.shared.tabular.scalars import is_display_missing
from pyaesa.shared.tabular.table_io import read_table, write_table


@dataclass(frozen=True)
class DeterministicComponentRows:
    """Prepared dynamic deterministic component rows rendered in ASR figures."""

    acc: pd.DataFrame
    lca: pd.DataFrame
    acc_output_files: tuple[Path, ...]


_COMPONENT_COLUMN = "__component"


def component_rows_from_runtime_frame(
    *,
    component_frame: pd.DataFrame,
    lca_rows: pd.DataFrame,
    acc_output_files: list[Path],
) -> DeterministicComponentRows:
    """Return figure component rows from dynamic ASR runtime component values."""
    components = component_frame.rename(
        columns={
            "__acc_component": "acc_value",
            "__lca_component": "lca_converted_value",
        }
    )
    converted = _convert_to_lca_unit(components, lca_rows=lca_rows)
    acc = _component_value_rows(converted, value_column="acc_value")
    lca = _deduplicated_lca_rows(
        _component_value_rows(converted, value_column="lca_converted_value")
    )
    return DeterministicComponentRows(
        acc=acc.reset_index(drop=True),
        lca=lca.reset_index(drop=True),
        acc_output_files=tuple(acc_output_files),
    )


def write_component_rows_artifact(
    *,
    path: Path,
    rows: DeterministicComponentRows,
) -> None:
    """Write dynamic deterministic ASR component figure inputs."""
    acc = rows.acc.copy()
    acc[_COMPONENT_COLUMN] = "acc"
    lca = rows.lca.copy()
    lca[_COMPONENT_COLUMN] = "lca"
    write_table(path=path, frame=pd.concat([acc, lca], ignore_index=True, sort=False))


def load_component_rows_artifact(
    *,
    path: Path,
    acc_output_files: list[Path],
) -> DeterministicComponentRows:
    """Read dynamic deterministic ASR component figure inputs."""
    rows = read_table(path=path)
    acc = _component_rows_from_artifact(rows=rows, component="acc")
    lca = _component_rows_from_artifact(rows=rows, component="lca")
    return DeterministicComponentRows(
        acc=acc,
        lca=lca,
        acc_output_files=tuple(acc_output_files),
    )


def _component_rows_from_artifact(*, rows: pd.DataFrame, component: str) -> pd.DataFrame:
    scoped = rows.loc[rows[_COMPONENT_COLUMN].astype(str).eq(component)].copy()
    return (
        scoped.drop(columns=[_COMPONENT_COLUMN])
        .dropna(axis="columns", how="all")
        .reset_index(drop=True)
    )


def _convert_to_lca_unit(frame: pd.DataFrame, *, lca_rows: pd.DataFrame) -> pd.DataFrame:
    keys = [
        column for column in ("lcia_method", "impact") if column in frame and column in lca_rows
    ]
    lca_units = lca_rows.loc[:, [*keys, "impact_unit"]].drop_duplicates()
    out = frame.merge(
        lca_units.rename(columns={"impact_unit": "__lca_impact_unit"}),
        on=keys,
        how="left",
    )
    out["acc_value"] = _numeric_array(out, "acc_value")
    out["lca_converted_value"] = _numeric_array(out, "lca_converted_value")
    out["impact_unit"] = out["__lca_impact_unit"]
    return out.drop(columns=["__lca_impact_unit"])


def _component_value_rows(frame: pd.DataFrame, *, value_column: str) -> pd.DataFrame:
    metadata = frame.drop(columns=["acc_value", "lca_converted_value"], errors="ignore")
    out = metadata.copy()
    out["__component_value"] = _numeric_array(frame, value_column)
    return out.loc[out["__component_value"].notna()].reset_index(drop=True)


def _numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.Series(pd.to_numeric(pd.Series(frame.loc[:, column], copy=False), errors="raise"))
    return values.to_numpy(dtype="float64")


def _deduplicated_lca_rows(frame: pd.DataFrame) -> pd.DataFrame:
    denominator_columns = {
        "__method",
        "l1_method",
        "l2_method",
        "l1_l2_method",
        "reference_year",
        "l2_reuse_year",
        "asocc_time_route",
        "asocc_ssp_start_year",
        AR6_CC_SSP_SCENARIO_COLUMN,
        ASOCC_SSP_SCENARIO_COLUMN,
        "cc_category",
        "cc_model",
        "cc_scenario",
    }
    columns = [column for column in frame.columns if column not in denominator_columns]
    return frame.loc[:, columns].drop_duplicates().reset_index(drop=True)


def _scope_component_rows(
    rows: pd.DataFrame,
    *,
    asr_frame: pd.DataFrame,
    include_method_axis: bool,
) -> pd.DataFrame:
    out = rows.copy()
    columns = [
        "cc_type",
        "lcia_method",
        "impact",
        "cc_category",
        "cc_model",
        "cc_scenario",
        "ar6_cc_ssp_scenario",
        "asocc_ssp_scenario",
        EXT_LCA_SSP_SCENARIO_COLUMN,
        "s_p",
        "r_c",
    ]
    if include_method_axis:
        columns.append("__method")
    for column in columns:
        values = visible_values(asr_frame, column)
        if column in out.columns and values:
            column_values = cast(pd.Series, out.loc[:, column])
            out = out.loc[_scope_filter(column_values, values, column=column)].copy()
    return _scope_component_variant_roles(rows=out, asr_frame=asr_frame)


def _scope_component_variant_roles(*, rows: pd.DataFrame, asr_frame: pd.DataFrame) -> pd.DataFrame:
    role_columns = [
        column
        for column in VARIANT_COLUMNS
        if column in rows.columns and column in asr_frame.columns
    ]
    if not role_columns or ROLE_COLUMN not in asr_frame.columns:
        return rows
    roles = pd.Series(asr_frame[ROLE_COLUMN], copy=False).astype("string").str.strip()
    identity_columns = [
        column
        for column in (
            "__method",
            "l1_l2_method",
            "l1_method",
            "l2_method",
            "r_c",
            "s_p",
            "r_p",
            "r_f",
            "lcia_method",
            "impact",
            "cc_category",
            "cc_model",
            "cc_scenario",
            "ar6_cc_ssp_scenario",
        )
        if column in rows.columns and column in asr_frame.columns
    ]
    merge_columns = [*identity_columns, *role_columns]
    retained = asr_frame.loc[
        roles.isna() | roles.eq("") | roles.eq(MIN_ROLE),
        [*merge_columns, ROLE_COLUMN],
    ].drop_duplicates()
    return rows.merge(retained, on=merge_columns, how="inner")


def _component_group_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"year", "__component_value", ROLE_COLUMN, *ASR_TRANSITION_SERIES_EXCLUDED_COLUMNS}
    return [column for column in frame.columns if column not in excluded]


def _scope_filter(values: pd.Series, accepted: list[str], *, column: str) -> pd.Series:
    text = values.astype("string").str.strip()
    accepted_values = {str(value).strip() for value in accepted if str(value).strip()}
    mask = text.isin(accepted_values)
    if column == EXT_LCA_SSP_SCENARIO_COLUMN or column == "asocc_ssp_scenario":
        mask |= values.map(is_display_missing) | text.isna() | text.eq("")
    return mask


def _integer_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.Series(pd.to_numeric(pd.Series(frame.loc[:, column], copy=False), errors="raise"))
    return values.astype(int)


def _float_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(
        pd.to_numeric(pd.Series(frame.loc[:, column], copy=False), errors="raise"),
        dtype="float64",
    )
