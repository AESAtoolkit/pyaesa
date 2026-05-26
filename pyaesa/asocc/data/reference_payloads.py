"""Reference year deterministic payload loading for aSoCC families."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pandas as pd

from ..runtime.scope.filtering import (
    normalize_filter_values,
    slice_frame_any_axis,
)
from .load_mrio import _load_lcia_l1_metric
from .paths import _get_mrio_year_dir
from .run_lcia import _load_lcia_for_year


def _reference_saved_dir(
    *,
    source: str,
    ref_year: int,
    agg_version: str | None,
) -> Path:
    """Return the deterministic MRIO directory for one reference year."""
    return _get_mrio_year_dir(
        source=source,
        year=int(ref_year),
        agg_version=agg_version,
    )


def _slice_payload_dict(
    payload: Mapping[str, pd.DataFrame],
    *,
    allowed_by_axis: dict[str, set[str] | None],
) -> dict[str, pd.DataFrame]:
    """Slice all LCIA reference payload frames with one axis contract."""
    out: dict[str, pd.DataFrame] = {}
    for key, value in payload.items():
        sliced = value
        for axis_name, allowed in allowed_by_axis.items():
            sliced = slice_frame_any_axis(sliced, axis_name=axis_name, allowed=allowed)
        out[key] = sliced
    return out


def _reference_lcia_reason_suffix(
    *,
    state,
    ref_year: int,
    lcia_method: str,
) -> str:
    """Return optional skipped-year reason suffix for one reference-year LCIA error."""
    year_reason = state.skipped_years.get(int(ref_year), {})
    if not isinstance(year_reason, dict):
        return ""
    reason = year_reason.get(str(lcia_method))
    if reason is None:
        return ""
    return f" Reason: {reason}"


def _reference_allowed_axes(*, context) -> dict[str, set[str] | None]:
    """Return branch scoped reference payload slicing axes."""
    keep_full_weight_axes = context.fu_code in {"L2.a.b", "L2.b.b", "L2.c.b"}
    allowed_by_axis: dict[str, set[str] | None] = {
        "r_p": normalize_filter_values(context.filters.get("r_p")),
        "s_p": normalize_filter_values(context.filters.get("s_p")),
        "r_c": normalize_filter_values(context.filters.get("r_c")),
        "r_f": normalize_filter_values(context.filters.get("r_f")),
        "r_u": normalize_filter_values(context.filters.get("r_u")),
    }
    if keep_full_weight_axes:
        allowed_by_axis["r_f"] = None
        allowed_by_axis["r_u"] = None
    return allowed_by_axis


def load_reference_lcia_method_payload(
    *,
    context,
    state,
    ref_year: int,
    lcia_method: str,
    agg_version: str | None,
    allow_method_year_fallback: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load one deterministic reference-year LCIA payload for one method."""
    saved_dir = _reference_saved_dir(
        source=context.source,
        ref_year=ref_year,
        agg_version=agg_version,
    )
    lcia_by_method_raw = _load_lcia_for_year(
        context=context,
        state=state,
        year=int(ref_year),
        saved_dir=saved_dir,
        agg_version_override=agg_version,
        allow_method_year_fallback=allow_method_year_fallback,
        selected_lcia_methods=[str(lcia_method)],
    )
    if lcia_by_method_raw is None:
        raise ValueError(f"Missing LCIA payload '{lcia_method}' at reference year {ref_year}.")
    lcia_by_method = dict(lcia_by_method_raw)
    return {str(key): value for key, value in lcia_by_method[lcia_method].items()}


def load_reference_lcia_reg(
    *,
    context,
    state,
    ref_year: int,
    lcia_method: str,
    lcia_kind: str,
    agg_version: str | None,
) -> pd.DataFrame:
    """Load reference-year regional LCIA data for one method/boundary."""
    try:
        payload = load_reference_lcia_method_payload(
            context=context,
            state=state,
            ref_year=ref_year,
            lcia_method=lcia_method,
            agg_version=agg_version,
            allow_method_year_fallback=False,
        )
    except ValueError as exc:
        suffix = _reference_lcia_reason_suffix(
            state=state,
            ref_year=ref_year,
            lcia_method=lcia_method,
        )
        raise ValueError(
            f"Missing LCIA payload '{lcia_method}' at reference year {ref_year}.{suffix}"
        ) from exc
    payload_key = "e_cba_fd_reg" if lcia_kind == "CBA_FD" else "e_pba_reg"
    lcia_reg = payload.get(payload_key)
    if not isinstance(lcia_reg, pd.DataFrame):
        suffix = _reference_lcia_reason_suffix(
            state=state,
            ref_year=ref_year,
            lcia_method=lcia_method,
        )
        raise ValueError(
            f"Missing LCIA payload '{lcia_method}' at reference year {ref_year}.{suffix}"
        )
    return lcia_reg


def load_reference_lcia_reg_for_domain(
    *,
    context,
    state,
    ref_year: int,
    lcia_method: str,
    lcia_kind: str,
    use_original_domain: bool,
) -> pd.DataFrame:
    """Load one reference-year regional LCIA payload for one domain choice."""
    return load_reference_lcia_reg(
        context=context,
        state=state,
        ref_year=ref_year,
        lcia_method=lcia_method,
        lcia_kind=lcia_kind,
        agg_version=None if use_original_domain else context.agg_version,
    )


def ensure_pr_hr_child_impact_timeseries_loaded(
    *,
    context,
    state,
    through_year: int,
    lcia_method: str,
    lcia_kind: str,
    use_original_domain: bool,
) -> dict[int, pd.DataFrame]:
    """Load missing PR-HR child-impact LCIA yearly metrics up to one year."""
    lcia_store = state.lcia_timeseries_original if use_original_domain else state.lcia_timeseries
    method_store = lcia_store.setdefault(lcia_method, {"CBA_FD": {}, "PBA": {}})
    kind_store = method_store.setdefault(lcia_kind, {})
    agg_version = None if use_original_domain else context.agg_version
    metric_key = "e_cba_fd_reg" if lcia_kind == "CBA_FD" else "e_pba_reg"
    for hist_year in sorted(y for y in context.historical_years if y <= int(through_year)):
        if hist_year in kind_store:
            continue
        saved_dir = _get_mrio_year_dir(
            source=context.source,
            year=int(hist_year),
            agg_version=agg_version,
        )
        try:
            # PR-HR cumulative windows keep child impacts until after integration.
            kind_store[hist_year] = _load_lcia_l1_metric(
                saved_dir=saved_dir,
                lcia_method=lcia_method,
                metric=metric_key,
            )
        except (FileNotFoundError, ValueError):
            continue
    return kind_store


def load_ar_l2_reference_lcia_payload(
    *,
    context,
    state,
    ref_year: int,
    lcia_key: str,
) -> dict[str, pd.DataFrame]:
    """Load and filter deterministic LCIA reference payloads for AR L2 formulas."""
    lcia_ref = load_reference_lcia_method_payload(
        context=context,
        state=state,
        ref_year=ref_year,
        lcia_method=lcia_key,
        agg_version=context.agg_version,
        allow_method_year_fallback=True,
    )
    allowed_by_axis = _reference_allowed_axes(context=context)
    return cast(
        dict[str, pd.DataFrame],
        _slice_payload_dict(lcia_ref, allowed_by_axis=allowed_by_axis),
    )
