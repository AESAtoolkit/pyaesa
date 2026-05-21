"""Shared helpers for L2 two step weighting logic."""

from typing import NamedTuple

import pandas as pd

from pyaesa.asocc.methods.registry.registry import REGISTRY
from pyaesa.asocc.runtime.scope.branch_resolution import (
    asocc_l1_dir,
    build_asocc_deterministic_path_scope,
    outputs_project_root,
)

from .io_helpers import list_output_files, read_output
from .workflow_helpers import validation_project_name

_L1_WEIGHTS_CACHE: dict[tuple[object, ...], pd.Series | None] = {}


class L1RegionWeightRequest(NamedTuple):
    """Input contract for loading L1 region weights."""

    validation_project_name_root: str
    source: str
    group_version: str | None
    group_reg: bool | None
    aggreg_indices: bool
    l1_mode: str
    output_format: str
    l1_method: str
    year: int
    impact: str | None
    reference_year: int | None


def _cache_key(request: L1RegionWeightRequest) -> tuple[object, ...]:
    """Return deterministic cache key for one L1-weight request."""
    return (
        request.validation_project_name_root,
        request.source,
        request.group_version or "",
        bool(request.group_reg),
        bool(request.aggreg_indices),
        request.l1_mode,
        request.output_format,
        request.l1_method,
        int(request.year),
        request.impact or "",
        request.reference_year,
    )


def _resolve_weight_axis(columns: pd.Index) -> str | None:
    """Return first supported regional axis column from one output table."""
    available = set(columns.map(str))
    for name in ("r_f", "r_p", "r_u", "region"):
        if name in available:
            return name
    return None


def _match_rows_for_request(
    *,
    frame: pd.DataFrame,
    request: L1RegionWeightRequest,
    year_col: str,
) -> pd.DataFrame:
    """Filter output table to rows matching one request context."""
    if year_col not in frame.columns or "l1_method" not in frame.columns:
        return frame.iloc[0:0].copy()

    part = frame.loc[frame["l1_method"].astype(str) == str(request.l1_method)].copy()
    if part.empty:
        return part

    if request.impact and "impact" in part.columns:
        part = part.loc[part["impact"].astype(str) == str(request.impact)]
    if part.empty:
        return part

    if request.reference_year is not None and "reference_year" in part.columns:
        ref_vals = pd.to_numeric(part["reference_year"], errors="coerce")
        part = part.loc[ref_vals == float(request.reference_year)]
    return part


def _weights_from_table(
    *,
    frame: pd.DataFrame,
    request: L1RegionWeightRequest,
    year_col: str,
) -> pd.Series | None:
    """Return regional weights from one matching output frame, if possible."""
    part = _match_rows_for_request(frame=frame, request=request, year_col=year_col)
    if part.empty:
        return None
    axis = _resolve_weight_axis(part.columns)
    if axis is None:
        return None
    numeric_values = pd.to_numeric(part[year_col], errors="coerce")
    grouped = (
        part.assign(_weight_value=numeric_values)
        .groupby(axis, dropna=False)["_weight_value"]
        .sum(min_count=1)
    )
    weights = pd.Series(grouped, copy=False).astype(float)
    weights.index = weights.index.map(str)
    return weights


def _l1_share_dir(*, request: L1RegionWeightRequest, l1_fu: str):
    """Return current published deterministic L1 share directory for one FU."""
    project_name = validation_project_name(
        base_project_name=request.validation_project_name_root,
        source=request.source,
        fu_code=l1_fu,
        l1_reg_aggreg=request.l1_mode,
    )
    project_root = outputs_project_root(project_name=project_name)
    path_scope = build_asocc_deterministic_path_scope(
        proj_base=project_root,
        source_label=request.source,
        group_version=request.group_version,
    )
    return asocc_l1_dir(scope=path_scope, lcia_sub=None)


def load_l1_region_weights(request: L1RegionWeightRequest) -> pd.Series | None:
    """Return cached L1 region weights for one method/year context."""
    key = _cache_key(request)
    if key in _L1_WEIGHTS_CACHE:
        return _L1_WEIGHTS_CACHE[key]

    year_col = str(request.year)
    for l1_fu in ("L1.a", "L1.b"):
        l1_dir = _l1_share_dir(request=request, l1_fu=l1_fu)
        if not l1_dir.exists():
            continue
        for path in list_output_files(
            l1_dir,
            preferred_format=request.output_format,
        ):
            df = read_output(path)
            weights = _weights_from_table(
                frame=df,
                request=request,
                year_col=year_col,
            )
            if weights is None:
                continue
            _L1_WEIGHTS_CACHE[key] = weights
            return weights

    _L1_WEIGHTS_CACHE[key] = None
    return None


def split_l1_l2_method(method_label: str, fu_code: str) -> tuple[str | None, str]:
    """Split a method label into optional L1 prefix and L2 method name."""
    l2_candidates = sorted(
        set(REGISTRY.list_l2_methods(fu_code=fu_code, l1_weighting=None)),
        key=len,
        reverse=True,
    )
    for l2_name in l2_candidates:
        if method_label == l2_name:
            return None, l2_name
        suffix = f"_{l2_name}"
        if method_label.endswith(suffix):
            return method_label[: -len(suffix)], l2_name
    return None, method_label
