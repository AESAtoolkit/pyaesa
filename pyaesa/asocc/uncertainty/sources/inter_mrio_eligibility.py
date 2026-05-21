"""Inter-MRIO eligible row classification."""

import numpy as np
import pandas as pd

from pyaesa.asocc.methods.registry.registry import REGISTRY


def non_lcia_final_mask(*, frame: pd.DataFrame) -> pd.Series:
    """Return rows eligible for pyaesa owned inter-MRIO interpolation."""
    l2_method = optional_column(frame=frame, column="l2_method")
    l1_method = optional_column(frame=frame, column="l1_method")
    l1_l2_method = optional_column(frame=frame, column="l1_l2_method")
    mask = np.zeros(len(frame), dtype=bool)
    combined = l2_method.notna().to_numpy(dtype=bool)
    l2_flags = _non_lcia_l2_flags(values=l2_method.loc[combined])
    l1_values = l1_method.loc[combined]
    mask[combined] = l2_flags & _optional_non_lcia_l1_flags(values=l1_values)
    direct = (~combined) & l1_method.isna().to_numpy(dtype=bool)
    direct_l2 = direct & l1_l2_method.notna().to_numpy(dtype=bool)
    if direct_l2.any():
        mask[direct_l2] = _non_lcia_l2_flags(values=l1_l2_method.loc[direct_l2])
    return pd.Series(mask, index=frame.index, dtype=bool)


def optional_column(*, frame: pd.DataFrame, column: str) -> pd.Series:
    """Return an optional identity column aligned to the frame index."""
    if column in frame.columns:
        return pd.Series(frame.loc[:, column], copy=False)
    return pd.Series(pd.NA, index=frame.index)


def _optional_non_lcia_l1_flags(*, values: pd.Series) -> np.ndarray:
    flags = _non_lcia_l1_flags(values=values.loc[values.notna()])
    output = np.ones(len(values), dtype=bool)
    output[values.notna().to_numpy(dtype=bool)] = flags
    return output


def _non_lcia_l1_flags(*, values: pd.Series) -> np.ndarray:
    method_values = values.astype(str)
    unique_methods = pd.unique(method_values)
    flags = {method: _known_non_lcia_l1(method=method) for method in unique_methods}
    return method_values.map(flags).to_numpy(dtype=bool)


def _non_lcia_l2_flags(*, values: pd.Series) -> np.ndarray:
    method_values = values.astype(str)
    unique_methods = pd.unique(method_values)
    flags = {method: _known_non_lcia_l2(method=method) for method in unique_methods}
    return method_values.map(flags).to_numpy(dtype=bool)


def _known_non_lcia_l2(*, method: str) -> bool:
    specs = REGISTRY.get_method(method, level="L2")
    return bool(specs) and not any(item.needs_lcia for item in specs)


def _known_non_lcia_l1(*, method: str) -> bool:
    specs = REGISTRY.get_method(method, level="L1")
    return bool(specs) and not any(item.needs_lcia for item in specs)
