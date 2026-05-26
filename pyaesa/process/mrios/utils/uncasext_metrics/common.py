"""Shared ownership for UNCASExt metrics preprocessing."""

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, overload

import numpy as np
import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

from .enacting_metric_clip_log import write_clipping_log

_PREPARED_ATTR = "_uncasext_prepared_inputs"


@dataclass
class _PreparedUncasextInputs:
    """Cached clipped/reaggregated inputs reused across UNCASExt metric builders."""

    x_vec: pd.Series
    y_fd_raw: pd.DataFrame
    y_fd: pd.DataFrame
    z_reg: pd.DataFrame
    gva_by_prod_raw: pd.Series
    gva_by_prod: pd.Series
    clipping_unit: str | None = None
    clipping_logged: bool = False


def _require_dataframe(value: Any, *, label: str) -> pd.DataFrame:
    """Require a pandas DataFrame value."""
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    return value


@overload
def _set_index_names(obj: pd.Series, names: list[str]) -> pd.Series: ...


@overload
def _set_index_names(obj: pd.DataFrame, names: list[str]) -> pd.DataFrame: ...


def _set_index_names(
    obj: pd.Series | pd.DataFrame,
    names: list[str],
) -> pd.Series | pd.DataFrame:
    """Return a copy with canonical index names."""
    out = obj.copy()
    idx = out.index
    if isinstance(idx, pd.MultiIndex):
        if len(names) != idx.nlevels:
            raise ValueError(
                f"Index level mismatch: expected {idx.nlevels} names, got {len(names)}."
            )
        out.index = idx.set_names(names)
    else:
        if len(names) != 1:
            raise ValueError(f"Index level mismatch: expected 1 name, got {len(names)}.")
        out.index = idx.rename(names[0])
    return out


def _set_column_names(obj: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Return a copy with canonical column names."""
    out = obj.copy()
    cols = out.columns
    if isinstance(cols, pd.MultiIndex):
        if len(names) != cols.nlevels:
            raise ValueError(
                f"Column level mismatch: expected {cols.nlevels} names, got {len(names)}."
            )
        out.columns = cols.set_names(names)
    else:
        if len(names) != 1:
            raise ValueError(f"Column level mismatch: expected 1 name, got {len(names)}.")
        out.columns = cols.rename(names[0])
    return out


def _normalize_x_series(x: Any) -> pd.Series:
    """Normalize ``iosys.x`` into a Series."""
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("iosys.x must have exactly one column to be normalized.")
        return x.iloc[:, 0]
    raise TypeError("iosys.x must be a pandas Series or single-column DataFrame.")


def _resolve_single_mrio_unit(iosys: Any) -> str | None:
    """Return one canonical MRIO unit from ``iosys.unit`` when available."""
    unit_obj = getattr(iosys, "unit", None)
    if unit_obj is None:
        return None
    if isinstance(unit_obj, pd.Series):
        values = unit_obj.astype(str).str.strip()
    elif isinstance(unit_obj, pd.DataFrame):
        values = unit_obj.stack().astype(str).str.strip()
    else:
        values = pd.Series([str(unit_obj).strip()])
    cleaned = [value for value in values.tolist() if value]
    unique_units = sorted(set(cleaned))
    if len(unique_units) == 1:
        return unique_units[0]
    return None


def _write_pickle(path: Path, payload: Any) -> None:
    """Serialize ``payload`` to ``path`` using pickle."""
    path = ensure_file_parent(path)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _all_exist(paths: list[Path]) -> bool:
    """Return True when all paths exist."""
    return all(path.exists() for path in paths)


@overload
def _clip_nonnegative(obj: pd.Series) -> pd.Series: ...


@overload
def _clip_nonnegative(obj: pd.DataFrame) -> pd.DataFrame: ...


def _clip_nonnegative(
    obj: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Return a copy clipped at zero for all numeric cells."""
    return obj.clip(lower=0.0)


def _sum_columns_by_region(frame: pd.DataFrame) -> pd.DataFrame:
    """Return columns summed by their ``region`` level."""
    region_values = frame.columns.get_level_values("region")
    region_array = np.asarray([str(value) for value in region_values.to_numpy()], dtype=object)
    regions = list(dict.fromkeys(region_array.tolist()))
    values = np.nan_to_num(frame.to_numpy(dtype=float), nan=0.0)
    out = np.empty((values.shape[0], len(regions)), dtype=float)
    for idx, region in enumerate(regions):
        out[:, idx] = values[:, region_array == region].sum(axis=1)
    return pd.DataFrame(
        out,
        index=frame.index,
        columns=pd.Index(regions, name="region"),
    )


def _build_prepared_uncasext_inputs(iosys: Any) -> _PreparedUncasextInputs:
    """Build clipped and region aggregated inputs for UNCASExt metrics."""
    z = _require_dataframe(getattr(iosys, "Z", None), label="iosys.Z")
    y = _require_dataframe(getattr(iosys, "Y", None), label="iosys.Y")

    x_vec = _set_index_names(_normalize_x_series(iosys.x), ["r_p", "s_p"])

    y_fd_raw = _sum_columns_by_region(y)
    y_fd_raw = _set_index_names(y_fd_raw, ["r_p", "s_p"])
    y_fd_raw = _set_column_names(y_fd_raw, ["r_f"])
    y_fd = _clip_nonnegative(y_fd_raw)

    regions = list(y_fd.columns)
    z_reg = _sum_columns_by_region(z)
    z_reg = _set_index_names(z_reg, ["r_p", "s_p"])
    z_reg = _set_column_names(z_reg, ["r_c"])
    z_reg = z_reg.loc[:, regions]

    factor_inputs_obj = getattr(iosys, "factor_inputs", None)
    factor_inputs_f = getattr(factor_inputs_obj, "F", None)
    if factor_inputs_f is None:
        raise ValueError(
            "Parsed MRIO is missing factor_inputs.F, which is required to build "
            "UNCASExt gross value added enacting metrics."
        )
    if isinstance(factor_inputs_f, pd.Series):
        gva_raw = factor_inputs_f
    elif isinstance(factor_inputs_f, pd.DataFrame):
        gva_raw = factor_inputs_f.sum(axis=0)
    else:
        raise TypeError("iosys.factor_inputs.F must be a pandas Series or DataFrame.")
    gva_raw = _set_index_names(gva_raw, ["r_p", "s_p"])
    gva_clipped = _clip_nonnegative(gva_raw)

    return _PreparedUncasextInputs(
        x_vec=x_vec,
        y_fd_raw=y_fd_raw,
        y_fd=y_fd,
        z_reg=z_reg,
        gva_by_prod_raw=gva_raw,
        gva_by_prod=gva_clipped,
        clipping_unit=_resolve_single_mrio_unit(iosys),
    )


def _get_prepared_uncasext_inputs(
    iosys: Any,
    *,
    source_key: str | None = None,
    matrix_version: str | None = None,
    saved_dir: Path | None = None,
) -> _PreparedUncasextInputs:
    """Return cached clipped/reaggregated inputs, computing them once per IOSystem."""
    prepared = getattr(iosys, _PREPARED_ATTR, None)
    if not isinstance(prepared, _PreparedUncasextInputs):
        prepared = _build_prepared_uncasext_inputs(iosys)
        setattr(iosys, _PREPARED_ATTR, prepared)

    if source_key is not None and saved_dir is not None and not prepared.clipping_logged:
        write_clipping_log(
            before=prepared.y_fd_raw,
            matrix_name="y_fd",
            unit=prepared.clipping_unit,
            source_key=source_key,
            matrix_version=matrix_version,
            saved_dir=saved_dir,
        )
        write_clipping_log(
            before=prepared.gva_by_prod_raw,
            matrix_name="f_factor_inputs_column",
            unit=prepared.clipping_unit,
            source_key=source_key,
            matrix_version=matrix_version,
            saved_dir=saved_dir,
        )
        prepared.clipping_logged = True

    return prepared
