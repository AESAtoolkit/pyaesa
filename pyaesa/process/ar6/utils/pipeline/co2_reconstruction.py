"""CO2 decomposition reconstruction filter for AR6 processing."""

import numpy as np
import pandas as pd

from pyaesa.download.ar6.utils.config import (
    RAW_CO2_AFOLU,
    RAW_CO2_ENERGY,
    RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES,
    RAW_CO2_INDUSTRIAL_PROCESSES,
    RAW_CO2_OTHER,
    RAW_CO2_WASTE,
    RAW_CO2_WITH_AFOLU,
)

from .preprocessing import YEAR_COLUMNS

CO2_RECONSTRUCTION_ERROR_THRESHOLD = 1e-5
CO2_RECONSTRUCTION_DROP_REASON = "co2_reconstruction_error_not_below_threshold"

_DIRECT_CO2_COMPONENTS = (
    RAW_CO2_AFOLU,
    RAW_CO2_OTHER,
    RAW_CO2_WASTE,
)


def drop_co2_reconstruction_failed_pairs(
    data_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove pathways failing the v2 reference CO2 decomposition check.

    The v2 reference notebook retains a model-scenario only when cumulative
    reported CO2 emissions can be reconstructed from the available CO2
    decomposition rows within the reference relative error threshold.
    """
    co2 = _variable_rows(data_df, RAW_CO2_WITH_AFOLU)
    co2_values = _year_values(co2).fillna(0.0)
    reconstructed = pd.DataFrame(0.0, index=co2.index, columns=co2_values.columns)
    for component in _DIRECT_CO2_COMPONENTS:
        reconstructed = reconstructed + _aligned_values(
            _variable_rows(data_df, component),
            co2.index,
            co2_values.columns,
        )

    eip = _variable_rows(data_df, RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES)
    eip_pairs = eip.index.intersection(co2.index)
    if not eip_pairs.empty:
        reconstructed.loc[eip_pairs] = (
            reconstructed.loc[eip_pairs]
            + _aligned_values(eip, eip_pairs, co2_values.columns).loc[eip_pairs]
        )

    energy = _variable_rows(data_df, RAW_CO2_ENERGY)
    indirect_pairs = co2.index.difference(eip_pairs).intersection(energy.index)
    if not indirect_pairs.empty:
        reconstructed.loc[indirect_pairs] = (
            reconstructed.loc[indirect_pairs]
            + _aligned_values(energy, indirect_pairs, co2_values.columns).loc[indirect_pairs]
            + _aligned_values(
                _variable_rows(data_df, RAW_CO2_INDUSTRIAL_PROCESSES),
                indirect_pairs,
                co2_values.columns,
            ).loc[indirect_pairs]
        )

    denominator = co2_values.sum(axis=1)
    reconstruction_error = (co2_values - reconstructed).sum(axis=1) / denominator
    passed = reconstruction_error.abs().lt(CO2_RECONSTRUCTION_ERROR_THRESHOLD) & pd.Series(
        np.isfinite(reconstruction_error.to_numpy(dtype=float)),
        index=reconstruction_error.index,
    )
    failed_pairs = reconstruction_error.index[~passed]
    if failed_pairs.empty:
        return data_df, pd.DataFrame()

    retained = data_df.loc[~data_df.index.droplevel("variable").isin(failed_pairs)].copy()
    log = pd.DataFrame(
        {
            "model": failed_pairs.get_level_values("model").astype(str),
            "scenario": failed_pairs.get_level_values("scenario").astype(str),
            "variable": RAW_CO2_WITH_AFOLU,
            "retained_variable": pd.NA,
            "ssp_family": co2.loc[failed_pairs, "Ssp_family"].to_numpy(copy=False),
            "drop_reason": CO2_RECONSTRUCTION_DROP_REASON,
            "drop_stage": "co2_reconstruction_check",
        }
    )
    return retained, log


def _variable_rows(data_df: pd.DataFrame, variable: str) -> pd.DataFrame:
    if data_df.empty or variable not in data_df.index.get_level_values("variable"):
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["model", "scenario"]),
            columns=data_df.columns,
        )
    return data_df.xs(variable, level="variable", drop_level=True)


def _year_values(frame: pd.DataFrame) -> pd.DataFrame:
    year_columns = [year for year in YEAR_COLUMNS if year in frame.columns]
    return frame.loc[:, year_columns].astype(float)


def _aligned_values(
    frame: pd.DataFrame,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(0.0, index=index, columns=columns)
    return _year_values(frame).reindex(index=index, columns=columns).fillna(0.0)
