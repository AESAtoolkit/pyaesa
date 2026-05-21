"""Expected overlap helpers for L2*b UT method checks."""

from typing import Literal, NamedTuple, Sequence, TypedDict

import numpy as np
import pandas as pd

from .io_helpers import is_missing_scalar, parse_optional_int
from .l2_star_b_load_inputs import L2StarBTotals
from .l2_two_step_helpers import L1RegionWeightRequest, load_l1_region_weights

_UT_TD = "UT(TD)"
_UT_FDA = "UT(FDa)"
_UT_GVAA = "UT(GVAa)"


class _AdjustedUTSpec(TypedDict):
    """Metadata needed to compute deterministic overlap targets per UT method."""

    axis: Literal["r_f", "r_u"]
    overlap_key: Literal["overlap_fd_by_rf", "overlap_gvaa_by_ru"]
    rule_in_l1: str
    rule_global: str


_ADJUSTED_UT_SPECS: dict[str, _AdjustedUTSpec] = {
    _UT_FDA: {
        "axis": "r_f",
        "overlap_key": "overlap_fd_by_rf",
        "rule_in_l1": "L2*b UT(FDa): sum(r_f) = sum(x_to_rc*kappa)/FD_r_f",
        "rule_global": "L2*b UT(FDa): sum = sum_r[L1(r) * overlap_FDa(r)]",
    },
    _UT_GVAA: {
        "axis": "r_u",
        "overlap_key": "overlap_gvaa_by_ru",
        "rule_in_l1": "L2*b UT(GVAa): sum(r_u) = sum(x*omega)/GVA_r_u",
        "rule_global": "L2*b UT(GVAa): sum = sum_r[L1(r) * overlap_GVAa(r)]",
    },
}


class L2StarBOverlapRequest(NamedTuple):
    """Input contract for deterministic L2*b overlap expectation checks."""

    checked: float
    validation_project_name_root: str
    source: str
    matrix_version: str | None
    group_reg: bool | None
    aggreg_indices: bool
    l1_mode: str
    output_format: str
    year: int
    bucket: str
    l1_method: str | None
    l2_method: str
    item: pd.Series
    totals: L2StarBTotals
    atol: float


def method_l2_star_b_in_l1_axis(
    *,
    l2_method: str,
    columns: Sequence[str],
) -> str | None:
    """Return country axis for L2*b in-l1 checks based on method columns."""
    cols = set(columns)
    if l2_method == _UT_FDA and "r_f" in cols:
        return "r_f"
    if l2_method == _UT_GVAA and "r_u" in cols:
        return "r_u"
    return None


def weighted_overlap_expected(
    *,
    weight_request: L1RegionWeightRequest,
    overlap_by_axis: pd.Series,
) -> float:
    """Return global weighted overlap using L1 region weights."""
    weights = load_l1_region_weights(weight_request)
    if weights is None or weights.empty:
        return np.nan
    w = pd.Series(pd.to_numeric(weights, errors="coerce"), copy=False).astype(float)
    w.index = w.index.map(str)
    o = pd.Series(pd.to_numeric(overlap_by_axis, errors="coerce"), copy=False).astype(float)
    o.index = o.index.map(str)
    common = w.index.intersection(o.index)
    if len(common) == 0:
        return np.nan
    return float((w.loc[common] * o.loc[common]).sum(min_count=1))


def _compare_expected(
    *,
    checked: float,
    expected: float,
    atol: float,
    ok_rule: str,
    missing_rule: str,
) -> tuple[float, bool, str]:
    """Compare observed ratio vs expected and return (expected, passed, rule)."""
    if np.isfinite(expected):
        return expected, bool(np.isclose(checked, expected, atol=atol, rtol=0)), ok_rule
    return expected, True, missing_rule


def _coerce_optional_float(value: object) -> float:
    """Convert scalar like value to float; return NaN for missing/non scalar."""
    if is_missing_scalar(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return np.nan
        return float(text)
    return np.nan


def _weight_request_for_row(
    *,
    request: L2StarBOverlapRequest,
) -> L1RegionWeightRequest | None:
    """Build L1-weight request for one grouped L2*b output row."""
    if request.l1_method is None:
        return None
    impact = str(request.item["impact"]) if "impact" in request.item else None
    ref_year = parse_optional_int(request.item.get("reference_year", ""))
    return L1RegionWeightRequest(
        request.validation_project_name_root,
        request.source,
        request.matrix_version,
        request.group_reg,
        request.aggreg_indices,
        request.l1_mode,
        request.output_format,
        request.l1_method,
        request.year,
        impact,
        ref_year,
    )


def _expected_ut_td(request: L2StarBOverlapRequest) -> tuple[float, bool, str]:
    """Return deterministic expected overlap for ``UT(TD)``."""
    if not (request.bucket == "l2_vs_global" and request.l1_method is None):
        return np.nan, True, "L2*b overlap ratio reported (no deterministic expected)"
    expected = request.totals["x_w"] / request.totals["fd_w"]
    return _compare_expected(
        checked=request.checked,
        expected=expected,
        atol=request.atol,
        ok_rule="L2*b UT(TD): sum = X_W / FD_W",
        missing_rule="L2*b UT(TD): expected overlap unavailable",
    )


def _expected_adjusted_ut(
    *,
    request: L2StarBOverlapRequest,
    spec: _AdjustedUTSpec,
) -> tuple[float, bool, str]:
    """Return deterministic expected overlap for adjusted UT variants."""
    overlap = request.totals[spec["overlap_key"]]
    axis = spec["axis"]
    if request.bucket == "l2_in_l1" and axis in request.item:
        expected = _coerce_optional_float(overlap.get(str(request.item[axis]), np.nan))
        return _compare_expected(
            checked=request.checked,
            expected=expected,
            atol=request.atol,
            ok_rule=spec["rule_in_l1"],
            missing_rule=f"L2*b {request.l2_method}: expected overlap unavailable",
        )
    if request.bucket == "l2_vs_global":
        weight_request = _weight_request_for_row(request=request)
        if weight_request is not None:
            expected = weighted_overlap_expected(
                weight_request=weight_request,
                overlap_by_axis=overlap,
            )
            return _compare_expected(
                checked=request.checked,
                expected=expected,
                atol=request.atol,
                ok_rule=spec["rule_global"],
                missing_rule=f"L2*b {request.l2_method}: expected overlap unavailable",
            )
    return np.nan, True, "L2*b overlap ratio reported (no deterministic expected)"


def expected_l2_star_b_overlap(
    request: L2StarBOverlapRequest,
) -> tuple[float, bool, str]:
    """Return expected value, pass flag, and rule label for L2*b UT checks."""
    if request.l2_method == _UT_TD:
        return _expected_ut_td(request)
    spec = _ADJUSTED_UT_SPECS.get(request.l2_method)
    if spec is None:
        return np.nan, True, "L2*b overlap ratio reported (no deterministic expected)"
    return _expected_adjusted_ut(request=request, spec=spec)
