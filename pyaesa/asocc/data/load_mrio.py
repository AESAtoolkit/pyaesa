"""Load precomputed MRIO matrices for allocation methods."""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.process.mrios.utils.io.metadata import _read_metadata

from ..io.pickle_io import read_pickle

_METRIC_INDEX_NAMES = {
    "fd_rf": ["r_f"],
    "gva_rp": ["r_p"],
    "fd_rp_sp": ["r_p", "s_p"],
    "gva_rp_sp": ["r_p", "s_p"],
    "fd_rf_sp": ["r_f", "s_p"],
    "fd_rp_sp_rf": ["r_p", "s_p", "r_f"],
    "e_cba_fd_reg": ["impact", "r_f"],
    "e_pba_reg": ["impact", "r_p"],
    "e_cba_fd_reg_cap": ["impact", "r_f"],
    "e_pba_reg_cap": ["impact", "r_p"],
    "e_cba_fd_reg_cap_cum": ["impact", "r_f"],
    "e_pba_reg_cap_cum": ["impact", "r_p"],
    "e_pba_rp_sp": ["impact", "r_p", "s_p"],
    "e_cba_fd_rp_sp": ["impact", "r_p", "s_p"],
    "e_cba_td_rp_sp": ["impact", "r_p", "s_p"],
    "e_cba_fd_rp_sp_rf": ["impact", "r_p", "s_p", "r_f"],
    "e_cba_td_rp_sp_rc": ["impact", "r_p", "s_p", "r_c"],
    "e_cba_fd_rf_sp": ["impact", "r_f", "s_p"],
    "e_cba_td_rc_sp": ["impact", "r_c", "s_p"],
    "x_rp_sp": ["r_p", "s_p"],
    "x_rp_sp_rc": ["r_p", "s_p", "r_c"],
    "x_rc_sp": ["r_c", "s_p"],
}


def _axis_names(axis: pd.Index | pd.MultiIndex) -> list[str]:
    """Return canonical axis names as a list."""
    if isinstance(axis, pd.MultiIndex):
        return [str(n) for n in axis.names]
    return [str(axis.name)]


def _require_names(
    obj: Any,
    *,
    name: str,
    index_names: list[str] | None = None,
    column_names: list[str] | None = None,
) -> Any:
    """Validate index/column names."""
    if index_names is not None:
        index_axis = getattr(obj, "index", None)
        if not isinstance(index_axis, (pd.Index, pd.MultiIndex)):
            raise ValueError(f"{name} must have an index.")
        got_index = _axis_names(index_axis)
        if got_index != index_names:
            raise ValueError(f"{name} index names must be {index_names}, got {got_index}.")
    if column_names is not None:
        column_axis = getattr(obj, "columns", None)
        if not isinstance(column_axis, (pd.Index, pd.MultiIndex)):
            raise ValueError(f"{name} must have columns.")
        got_columns = _axis_names(column_axis)
        if got_columns != column_names:
            raise ValueError(f"{name} column names must be {column_names}, got {got_columns}.")
    return obj


def _load_pickle_required(path: Path, name: str) -> Any:
    """Load a pickle and raise if missing.

    Args:
        path: Pickle path.
        name: Logical name for error messages.

    Returns:
        Loaded object.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing MRIO pickle: {name} at {path}")
    return read_pickle(path)


_L1_METRIC_SCHEMA: dict[str, tuple[list[str], list[str] | None]] = {
    "fd_rf": (["r_f"], None),
    "gva_rp": (["r_p"], None),
}
_L2_METRIC_SCHEMA: dict[str, tuple[list[str], list[str] | None]] = {
    "fd_rp_sp_rf": (["r_p", "s_p"], ["r_f"]),
    "fd_rp_sp": (["r_p", "s_p"], None),
    "fd_rf_sp": (["r_f", "s_p"], None),
    "gva_rp_sp": (["r_p", "s_p"], None),
}
_UTILITY_SCHEMA: dict[str, tuple[list[str], list[str] | None]] = {
    "x_to_rc": (["r_p", "s_p"], ["r_c"]),
    "kappa": (["r_c", "r_p", "s_p"], ["r_f"]),
    "omega_reg": (["r_u"], ["r_p", "s_p"]),
}


def _load_enacting_metric_l1_metric(saved_dir: Path, metric: str) -> pd.Series:
    """Load one level-1 enacting metric by name."""
    idx_names, col_names = _L1_METRIC_SCHEMA[metric]
    base = saved_dir / "enacting_metrics" / "level_1"
    payload = _load_pickle_required(base / f"{metric}.pickle", metric)
    result = _require_names(
        payload,
        name=metric,
        index_names=idx_names,
        column_names=col_names,
    )
    if not isinstance(result, pd.Series):
        raise ValueError(f"{metric} must be stored as a Series, got {type(result).__name__}.")
    return result


def _load_enacting_metric_l2_metric(saved_dir: Path, metric: str) -> pd.DataFrame | pd.Series:
    """Load one level-2 enacting metric by name."""
    idx_names, col_names = _L2_METRIC_SCHEMA[metric]
    base = saved_dir / "enacting_metrics" / "level_2"
    payload = _load_pickle_required(base / f"{metric}.pickle", metric)
    result = _require_names(
        payload,
        name=metric,
        index_names=idx_names,
        column_names=col_names,
    )
    if col_names is None:
        if isinstance(result, pd.Series):
            return result
        raise ValueError(f"{metric} must be stored as a Series, got {type(result).__name__}.")
    return cast(pd.DataFrame, result)


def _load_utility_metric(saved_dir: Path, metric: str) -> pd.DataFrame:
    """Load one utility propagation metric by name."""
    idx_names, col_names = _UTILITY_SCHEMA[metric]
    base = saved_dir / "utility_propag_uncasext"
    payload = _load_pickle_required(base / f"{metric}.pickle", metric)
    result = _require_names(
        payload,
        name=metric,
        index_names=idx_names,
        column_names=col_names,
    )
    return cast(pd.DataFrame, result)


_LCIA_L1_SCHEMA: dict[str, tuple[list[str], list[str] | None]] = {
    "e_cba_fd_reg": (["impact"], ["r_f"]),
    "e_pba_reg": (["impact"], ["r_p"]),
}
_LCIA_L2_SCHEMA: dict[str, tuple[list[str], list[str] | None]] = {
    "e_pba_rp_sp": (["impact"], ["r_p", "s_p"]),
    "e_cba_fd_rp_sp": (["impact"], ["r_p", "s_p"]),
    "e_cba_td_rp_sp": (["impact"], ["r_p", "s_p"]),
    "e_cba_fd_rp_sp_rf": (["impact", "r_p", "s_p"], ["r_f"]),
    "e_cba_td_rp_sp_rc": (["impact", "r_p", "s_p"], ["r_c"]),
    "e_cba_fd_rf_sp": (["impact"], ["r_f", "s_p"]),
    "e_cba_td_rc_sp": (["impact"], ["r_c", "s_p"]),
}


def _load_lcia_l1_metric(saved_dir: Path, lcia_method: str, metric: str) -> pd.DataFrame:
    """Load one LCIA level-1 MRIO enacting metric by key."""
    idx_names, col_names = _LCIA_L1_SCHEMA[metric]
    base = saved_dir / "enacting_metrics" / "level_1" / lcia_method
    payload = _load_pickle_required(base / f"{metric}.pickle", metric)
    return _require_names(
        payload,
        name=metric,
        index_names=idx_names,
        column_names=col_names,
    )


def _load_lcia_l2_metric(saved_dir: Path, lcia_method: str, metric: str) -> pd.DataFrame:
    """Load one LCIA level-2 MRIO enacting metric by key."""
    idx_names, col_names = _LCIA_L2_SCHEMA[metric]
    base = saved_dir / "enacting_metrics" / "level_2" / lcia_method
    payload = _load_pickle_required(base / f"{metric}.pickle", metric)
    return _require_names(
        payload,
        name=metric,
        index_names=idx_names,
        column_names=col_names,
    )


def _years_from_metadata(
    source: str,
    group_version: str | None,
) -> list[int]:
    """Load available years from MRIO metadata."""
    # Setup uses this list as the single source of truth for available MRIO years.
    metadata = _read_metadata(source, matrix_version=group_version)
    years = metadata.get("years", {})
    return sorted(int(y) for y in years.keys())


def _metric_to_series(metric_name: str, payload: pd.Series | pd.DataFrame) -> pd.Series:
    """Normalize a metric payload into a Series with named index levels."""
    if isinstance(payload, pd.Series):
        series = payload
    else:
        series = _frame_to_metric_series(payload)

    names = _METRIC_INDEX_NAMES.get(metric_name)
    if names:
        got = _axis_names(series.index)
        if got != names:
            raise ValueError(f"Metric '{metric_name}' index names must be {names}, got {got}.")
    return cast(pd.Series, series)


def _frame_to_metric_series(frame: pd.DataFrame) -> pd.Series:
    """Flatten one labelled metric frame without pandas stack overhead."""
    row_count, column_count = frame.shape
    row_levels, row_codes, row_names = _stack_axis_parts(
        frame.index,
        repeat_each=column_count,
        tile_count=None,
    )
    column_levels, column_codes, column_names = _stack_axis_parts(
        frame.columns,
        repeat_each=None,
        tile_count=row_count,
    )
    index = pd.MultiIndex(
        levels=[*row_levels, *column_levels],
        codes=[*row_codes, *column_codes],
        names=[*row_names, *column_names],
        verify_integrity=False,
    )
    values = frame.to_numpy(copy=False).reshape(row_count * column_count)
    return pd.Series(values, index=index)


def _stack_axis_parts(
    axis: pd.Index | pd.MultiIndex,
    *,
    repeat_each: int | None,
    tile_count: int | None,
) -> tuple[list[pd.Index], list[np.ndarray], list[str | None]]:
    """Return MultiIndex constructor parts for one axis in row major frame order."""
    if isinstance(axis, pd.MultiIndex):
        levels = [pd.Index(level, copy=False) for level in axis.levels]
        base_codes = [np.asarray(code, dtype=np.intp) for code in axis.codes]
        names = [None if name is None else str(name) for name in axis.names]
    else:
        codes, labels = pd.factorize(axis, sort=False)
        levels = [pd.Index(labels, copy=False)]
        base_codes = [np.asarray(codes, dtype=np.intp)]
        names = [None if axis.name is None else str(axis.name)]
    if repeat_each is not None:
        out_codes = [np.repeat(code, repeat_each) for code in base_codes]
    else:
        repeat_count = cast(int, tile_count)
        out_codes = [np.tile(code, repeat_count) for code in base_codes]
    return levels, out_codes, names
