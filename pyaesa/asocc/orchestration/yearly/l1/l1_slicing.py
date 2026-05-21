"""Compute-time slicing for L1 orchestration."""

import pandas as pd
import numpy as np

from ....methods.registry.registry import REGISTRY
from pyaesa.asocc.runtime.scope.filtering import normalize_filter_values
from .l1_types import _L1RunContext

_TD_L2_FUS = {"L2.a.b", "L2.b.b", "L2.c.b"}


def _l2_axes_for_l1_method(*, run: _L1RunContext, l1_method: str) -> dict[str, str]:
    """Return L2-method -> expected L1 axis for one L1 method in combined runs."""
    axes: dict[str, str] = {}
    if not run.context.fu_code.startswith("L2."):
        return axes
    for l2_method, paired_l1_method in run.context.combined:
        if paired_l1_method != l1_method:
            continue
        axes[l2_method] = REGISTRY.l2_weight_axis_for_method(l2_method, run.context.fu_code)
    return axes


def _single_axis_or_none(axis_by_l2: dict[str, str]) -> str | None:
    """Return single required axis when unambiguous."""
    if not axis_by_l2:
        return None
    unique = sorted(set(axis_by_l2.values()))
    if len(unique) == 1:
        return unique[0]
    return None


def _active_l1_slice_filters(*, run: _L1RunContext) -> list[tuple[str, set[str]]]:
    """Return active L1 compute filters for the run."""
    preserve_full = run.context.fu_code in _TD_L2_FUS
    filters: list[tuple[str, set[str]]] = []
    for axis in ("r_p", "r_f", "r_u", "r_c", "s_p"):
        allowed = normalize_filter_values(run.context.filters.get(axis))
        if not allowed:
            continue
        if preserve_full and axis in {"r_f", "r_u"}:
            continue
        filters.append((axis, allowed))
    return filters


def _multi_index_filter_mask(
    *,
    index: pd.MultiIndex,
    filters: list[tuple[str, set[str]]],
) -> np.ndarray:
    """Return one combined mask for all active MultiIndex filters."""
    names = tuple(str(name) for name in index.names)
    mask = np.ones(len(index), dtype=bool)
    for axis, allowed in filters:
        if axis not in names:
            continue
        level_position = names.index(axis)
        level = pd.Index(index.levels[level_position], copy=False)
        allowed_codes = level.get_indexer(list(allowed))
        allowed_codes = allowed_codes[allowed_codes >= 0]
        axis_codes = np.asarray(index.codes[level_position], dtype=np.intp)
        mask &= np.isin(axis_codes, allowed_codes)
    return mask


def _slice_l1_frame_for_compute(
    *,
    run: _L1RunContext,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply compute time slicing to one L1 frame."""
    if frame.empty:
        return frame
    filters = _active_l1_slice_filters(run=run)
    if not filters:
        return frame
    if isinstance(frame.index, pd.MultiIndex):
        return frame.loc[_multi_index_filter_mask(index=frame.index, filters=filters)]
    idx_name = str(frame.index.name) if frame.index.name is not None else None
    for axis, allowed in filters:
        if idx_name == axis:
            return frame.loc[frame.index.isin(allowed)]
    return frame
