"""Input validation and compute time slicing for L2 orchestration."""

import numpy as np
import pandas as pd
from typing import cast

from ....methods.registry.registry import REGISTRY
from pyaesa.asocc.runtime.scope.filtering import (
    normalize_filter_values,
    slice_frame_any_axis,
    slice_series_any_axis,
)
from .l2_types import _L2ComputeInputs, _L2RunContext


def _mask_for_allowed(labels: pd.Index, allowed: set[str]):
    """Build membership mask for normalized textual labels."""
    return labels.isin(allowed)


def _slice_series_index(
    series: pd.Series,
    *,
    level: str,
    allowed: set[str] | None,
) -> pd.Series:
    """Slice a Series by one index level using string matched labels."""
    if not allowed:
        return series
    if isinstance(series.index, pd.MultiIndex):
        names = [str(name) for name in series.index.names]
        if level not in names:
            return series
        mask = _mask_for_allowed(series.index.get_level_values(level), allowed)
        return series.loc[mask]
    mask = _mask_for_allowed(series.index, allowed)
    return series.loc[mask]


def _slice_frame_index(
    frame: pd.DataFrame,
    *,
    level: str,
    allowed: set[str] | None,
) -> pd.DataFrame:
    """Slice a DataFrame by one index level using string matched labels."""
    if not allowed:
        return frame
    if isinstance(frame.index, pd.MultiIndex):
        names = [str(name) for name in frame.index.names]
        if level not in names:
            return frame
        mask = _mask_for_allowed(frame.index.get_level_values(level), allowed)
        return frame.loc[mask]
    mask = _mask_for_allowed(frame.index, allowed)
    return frame.loc[mask]


def _slice_frame_columns(
    frame: pd.DataFrame,
    *,
    allowed: set[str] | None,
) -> pd.DataFrame:
    """Slice DataFrame columns by string matched labels."""
    if not allowed:
        return frame
    mask = _mask_for_allowed(frame.columns, allowed)
    return frame.loc[:, mask]


def _slice_frame_column_level(
    frame: pd.DataFrame,
    *,
    level: str,
    allowed: set[str] | None,
) -> pd.DataFrame:
    """Slice DataFrame columns by one named level (supports MultiIndex)."""
    if not allowed:
        return frame
    if isinstance(frame.columns, pd.MultiIndex):
        names = [str(name) for name in frame.columns.names]
        if level not in names:
            return frame
        mask = _mask_for_allowed(
            frame.columns.get_level_values(level),
            allowed,
        )
        return frame.loc[:, mask]
    if frame.columns.name is not None and str(frame.columns.name) == level:
        mask = _mask_for_allowed(frame.columns, allowed)
        return frame.loc[:, mask]
    return frame


def _slice_l2_inputs_for_compute(
    *,
    context,
    inputs: _L2ComputeInputs,
) -> _L2ComputeInputs:
    """Apply compute time slicing using selected run filters."""
    rp_values = normalize_filter_values(context.filters.get("r_p"))
    sp_values = normalize_filter_values(context.filters.get("s_p"))
    rc_values = normalize_filter_values(context.filters.get("r_c"))
    rf_values = normalize_filter_values(context.filters.get("r_f"))
    ru_values = normalize_filter_values(context.filters.get("r_u"))

    keep_full_weight_axes = context.fu_code in {"L2.a.b", "L2.b.b", "L2.c.b"}
    if keep_full_weight_axes:
        rf_values = None
        ru_values = None

    fd_rf = _slice_series_index(inputs.fd_rf, level="r_f", allowed=rf_values)
    gva_rp = _slice_series_index(inputs.gva_rp, level="r_p", allowed=rp_values)

    fd_rp_sp_rf = _slice_frame_index(inputs.fd_rp_sp_rf, level="r_p", allowed=rp_values)
    fd_rp_sp_rf = _slice_frame_index(fd_rp_sp_rf, level="s_p", allowed=sp_values)
    fd_rp_sp_rf = _slice_frame_columns(fd_rp_sp_rf, allowed=rf_values)

    fd_rp_sp = _slice_series_index(inputs.fd_rp_sp, level="r_p", allowed=rp_values)
    fd_rp_sp = _slice_series_index(fd_rp_sp, level="s_p", allowed=sp_values)

    fd_rf_sp = _slice_series_index(inputs.fd_rf_sp, level="r_f", allowed=rf_values)
    fd_rf_sp = _slice_series_index(fd_rf_sp, level="s_p", allowed=sp_values)

    gva_rp_sp = _slice_series_index(inputs.gva_rp_sp, level="r_p", allowed=rp_values)
    gva_rp_sp = _slice_series_index(gva_rp_sp, level="s_p", allowed=sp_values)

    x_to_rc = _slice_frame_index(inputs.x_to_rc, level="r_p", allowed=rp_values)
    x_to_rc = _slice_frame_index(x_to_rc, level="s_p", allowed=sp_values)
    x_to_rc = _slice_frame_columns(x_to_rc, allowed=rc_values)

    kappa = _slice_frame_index(inputs.kappa, level="r_p", allowed=rp_values)
    kappa = _slice_frame_index(kappa, level="s_p", allowed=sp_values)
    kappa = _slice_frame_index(kappa, level="r_c", allowed=rc_values)
    kappa = _slice_frame_columns(kappa, allowed=rf_values)

    omega_reg = _slice_frame_column_level(inputs.omega_reg, level="r_p", allowed=rp_values)
    omega_reg = _slice_frame_column_level(omega_reg, level="s_p", allowed=sp_values)
    omega_reg = _slice_frame_index(omega_reg, level="r_u", allowed=ru_values)

    return _L2ComputeInputs(
        fd_rf=fd_rf,
        gva_rp=gva_rp,
        fd_rp_sp_rf=fd_rp_sp_rf,
        fd_rp_sp=fd_rp_sp,
        fd_rf_sp=fd_rf_sp,
        gva_rp_sp=gva_rp_sp,
        x_to_rc=x_to_rc,
        kappa=kappa,
        omega_reg=omega_reg,
    )


def _slice_lcia_payload_for_compute(
    *,
    context,
    payload: dict,
) -> dict:
    """Slice LCIA MRIO enacting metrics at compute time using run filters."""
    keep_full_weight_axes = context.fu_code in {"L2.a.b", "L2.b.b", "L2.c.b"}
    sp_values = normalize_filter_values(context.filters.get("s_p"))
    allowed_by_axis = {
        "r_p": normalize_filter_values(context.filters.get("r_p")),
        "s_p": sp_values,
        "r_c": normalize_filter_values(context.filters.get("r_c")),
        "r_f": normalize_filter_values(context.filters.get("r_f")),
        "r_u": normalize_filter_values(context.filters.get("r_u")),
    }
    if keep_full_weight_axes:
        allowed_by_axis["r_f"] = None
        allowed_by_axis["r_u"] = None

    out: dict = {}
    for key, value in payload.items():
        sliced = value
        if isinstance(value, pd.DataFrame):
            sliced_frame = value
            for axis_name, allowed in allowed_by_axis.items():
                sliced_frame = slice_frame_any_axis(
                    sliced_frame, axis_name=axis_name, allowed=allowed
                )
            sliced = sliced_frame
        elif isinstance(value, pd.Series):
            sliced_series = value
            for axis_name, allowed in allowed_by_axis.items():
                sliced_series = slice_series_any_axis(
                    sliced_series, axis_name=axis_name, allowed=allowed
                )
            sliced = sliced_series
        out[key] = sliced
    return out


def _slice_l1_weights_for_compute(
    *,
    run: _L2RunContext,
    l2_method: str,
    weights: pd.Series | None,
) -> pd.Series | None:
    """Slice L1 weights by the expected L2 weighting axis when applicable."""
    if weights is None:
        return None
    axis = REGISTRY.l2_weight_axis_for_method(l2_method, run.context.fu_code)
    allowed = normalize_filter_values(run.context.filters.get(axis))
    if run.context.fu_code in {"L2.a.b", "L2.b.b", "L2.c.b"} and axis in {"r_f", "r_u"}:
        return weights
    return slice_series_any_axis(weights, axis_name=axis, allowed=allowed)


def _slice_l1_weight_frame_for_compute(
    *,
    run: _L2RunContext,
    l2_method: str,
    weights: pd.DataFrame | None,
) -> pd.DataFrame | None:
    """Slice one L1 weight frame by the expected L2 weighting axis once."""
    if weights is None:
        return None
    axis = REGISTRY.l2_weight_axis_for_method(l2_method, run.context.fu_code)
    allowed = normalize_filter_values(run.context.filters.get(axis))
    if run.context.fu_code in {"L2.a.b", "L2.b.b", "L2.c.b"} and axis in {"r_f", "r_u"}:
        return weights
    return slice_frame_any_axis(weights, axis_name=axis, allowed=allowed)


def _impact_weight_items(
    l1_weights: pd.DataFrame | pd.Series | None,
) -> list[tuple[str | None, pd.Series | None]]:
    """Split optional L1 weights by impact level when present."""
    impact_weights: list[tuple[str | None, pd.Series | None]] = []
    if isinstance(l1_weights, pd.DataFrame):
        if isinstance(l1_weights.index, pd.MultiIndex):
            l1_series = l1_weights.iloc[:, 0]
            multi_index = l1_series.index
            impact_codes = np.asarray(multi_index.codes[0], dtype=np.intp)
            boundaries = np.empty(impact_codes.size + 1, dtype=bool)
            boundaries[0] = True
            boundaries[-1] = True
            boundaries[1:-1] = impact_codes[1:] != impact_codes[:-1]
            edges = np.flatnonzero(boundaries)
            for start, end in zip(edges[:-1], edges[1:], strict=True):
                code_value = int(impact_codes[start])
                impact_name = multi_index.levels[0][code_value]
                sliced = l1_series.iloc[int(start) : int(end)]
                impact_weights.append((str(impact_name), sliced.droplevel(0)))
        else:
            impact_weights.append((None, l1_weights.iloc[:, 0]))
    elif l1_weights is not None:
        impact_weights.append((None, l1_weights))
    else:
        impact_weights.append((None, None))
    return impact_weights


def _trailing_index_slice(
    *,
    index: pd.MultiIndex,
    start: int,
    end: int,
) -> pd.Index:
    """Return one index slice without the leading impact level."""
    if index.nlevels == 2:
        codes = np.asarray(index.codes[1][start:end], dtype=np.intp)
        out = pd.Index(index.levels[1], copy=False).take(codes)
        out.name = index.names[1]
        return out
    return pd.MultiIndex(
        levels=list(index.levels[1:]),
        codes=[np.asarray(code[start:end], dtype=np.intp) for code in index.codes[1:]],
        names=list(index.names[1:]),
        verify_integrity=False,
    )


def _impact_weight_matrix(
    l1_weights: pd.DataFrame | None,
) -> tuple[tuple[str, ...], pd.Index, np.ndarray] | None:
    """Return impact names, shared weight index, and numeric impact weight matrix."""
    if l1_weights is None or not isinstance(l1_weights.index, pd.MultiIndex):
        return None
    multi_index = cast(pd.MultiIndex, l1_weights.index)
    impact_codes = np.asarray(multi_index.codes[0], dtype=np.intp)
    boundaries = np.empty(impact_codes.size + 1, dtype=bool)
    boundaries[0] = True
    boundaries[-1] = True
    boundaries[1:-1] = impact_codes[1:] != impact_codes[:-1]
    edges = np.flatnonzero(boundaries)
    first_start = int(edges[0])
    first_end = int(edges[1])
    weight_index = _trailing_index_slice(
        index=multi_index,
        start=first_start,
        end=first_end,
    )
    values = np.asarray(l1_weights.to_numpy(dtype=np.float64, copy=False)[:, 0], dtype=np.float64)
    impact_names = tuple(
        str(multi_index.levels[0][int(impact_codes[int(start)])]) for start in edges[:-1]
    )
    return (
        impact_names,
        weight_index,
        values.reshape(len(impact_names), first_end - first_start),
    )


def _normalize_l1_weights(
    weights: pd.Series,
) -> pd.Series:
    """Validate L1 weights shape before L2 weighting."""
    if not weights.index.is_unique:
        dup = pd.Index(weights.index[weights.index.duplicated()]).unique().tolist()
        sample = [str(v) for v in dup[:10]]
        raise ValueError(
            f"L1 weights must have a unique index before L2 weighting. Duplicate labels: {sample}"
        )
    return weights
