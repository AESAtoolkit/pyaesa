"""Deterministic ASR persisted table I/O."""

from pathlib import Path

import pandas as pd
from pyaesa.shared.acc_asr_common.deterministic.downstream import tabular_io
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.tabular.wide_tables import persisted_method_block_columns

detect_year_columns = tabular_io.detect_year_columns
requested_year_columns = tabular_io.requested_year_columns
detect_id_columns = tabular_io.detect_id_columns
write_output_table = tabular_io.write_output_table

_DYNAMIC_CC_COLUMNS = (
    "cc_model",
    "cc_scenario",
    "cc_category",
    AR6_CC_SSP_SCENARIO_COLUMN,
    "cc_flow",
    "cc_variable",
)
_SELECTOR_COLUMNS = ("r_p", "s_p", "r_c", "r_f")


def write_asr_output(
    df: pd.DataFrame,
    output_path: Path,
    output_format: str,
) -> None:
    """Write one ASR output file."""
    ordered = ordered_asr_output_columns(df)
    write_output_table(
        df=df.loc[:, ordered].copy(),
        output_path=output_path,
        output_format=output_format,
    )


def ordered_asr_output_columns(frame: pd.DataFrame) -> list[str]:
    """Return canonical deterministic ASR output column order."""
    year_columns = detect_year_columns(frame)
    ordered: list[str] = []
    ordered.extend(column for column in _DYNAMIC_CC_COLUMNS if column in frame.columns)
    ordered.extend(persisted_method_block_columns(frame))
    ordered.extend(column for column in ("impact", "impact_unit") if column in frame.columns)
    ordered.extend(column for column in _SELECTOR_COLUMNS if column in frame.columns)
    if ASOCC_TIME_ROUTE_PUBLIC_COLUMN in frame.columns:
        ordered.append(ASOCC_TIME_ROUTE_PUBLIC_COLUMN)
    if "reference_year" in frame.columns:
        ordered.append("reference_year")
    ordered.extend(
        column
        for column in ("asocc_ssp_start_year", "lca_ssp_start_year")
        if column in frame.columns
    )
    ordered.extend(
        column for column in frame.columns if column not in ordered and column not in year_columns
    )
    return [*ordered, *year_columns]
