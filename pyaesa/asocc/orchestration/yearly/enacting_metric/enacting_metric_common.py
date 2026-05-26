"""Shared enacting metric functions."""

import pandas as pd

from ....data.region_agg_mapping import load_region_agg_mapping
from ....io.metadata import EnactingMetricKey, RunContext, RunState
from ....methods.registry.registry import REGISTRY
from pyaesa.asocc.runtime.scope.filtering import normalize_filter_values


def _record_enacting_metric_input(
    *,
    context: RunContext,
    state: RunState,
    key: EnactingMetricKey,
    year: int,
    series: pd.Series,
    level: str,
) -> None:
    """Record one enacting metric with explicit level contract."""
    _store_enacting_metric_input(
        state=state,
        key=key,
        year=year,
        series=_slice_enacting_metric_series_for_run(
            context=context,
            series=series,
        ),
        level=level,
    )


def _store_enacting_metric_input(
    *,
    state: RunState,
    key: EnactingMetricKey,
    year: int,
    series: pd.Series,
    level: str,
) -> None:
    """Store one enacting metric series after the owner has applied run scope."""
    prev = state.enacting_metric_levels.get(key)
    if prev is not None and prev != level:
        raise ValueError(f"Enacting metric '{key}' has conflicting levels: {prev} vs {level}.")
    state.enacting_metric_levels[key] = level
    state.enacting_metric_inputs.setdefault(key, {})[year] = series


def _slice_enacting_metric_payload_for_run(
    *,
    context: RunContext,
    payload: pd.DataFrame,
) -> pd.DataFrame:
    """Apply run scope slicing before wide payloads are reshaped."""
    out = payload
    allowed_by_axis = _enacting_metric_allowed_filters(context=context)
    out = _slice_frame_axis_for_run(frame=out, axis_name="index", allowed_by_axis=allowed_by_axis)
    out = _slice_frame_axis_for_run(frame=out, axis_name="columns", allowed_by_axis=allowed_by_axis)
    return out


def _enacting_metric_allowed_filters(*, context: RunContext) -> dict[str, set[str] | None]:
    """Return index filters used by enacting metric output ownership."""
    keep_full_weight_axes = context.fu_code in {"L2.a.b", "L2.b.b", "L2.c.b"}
    allowed_by_axis = {
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


def _slice_frame_axis_for_run(
    *,
    frame: pd.DataFrame,
    axis_name: str,
    allowed_by_axis: dict[str, set[str] | None],
) -> pd.DataFrame:
    """Slice one DataFrame axis by named levels before stack operations."""
    axis = frame.index if axis_name == "index" else frame.columns
    if isinstance(axis, pd.MultiIndex):
        names = [str(name) for name in axis.names]
        mask = None
        for level_name, allowed in allowed_by_axis.items():
            if not allowed or level_name not in names:
                continue
            level_mask = axis.get_level_values(level_name).isin(allowed)
            mask = level_mask if mask is None else mask & level_mask
        if mask is None:
            return frame
        return frame.loc[mask, :] if axis_name == "index" else frame.loc[:, mask]
    name = str(axis.name)
    allowed = allowed_by_axis.get(name)
    if not allowed:
        return frame
    mask = axis.isin(allowed)
    return frame.loc[mask, :] if axis_name == "index" else frame.loc[:, mask]


def _slice_enacting_metric_series_for_run(
    *,
    context: RunContext,
    series: pd.Series,
) -> pd.Series:
    """Apply run scope slicing to enacting metric series by available index axes."""
    out = series
    allowed_by_axis = _enacting_metric_allowed_filters(context=context)
    if isinstance(out.index, pd.MultiIndex):
        names = [str(name) for name in out.index.names]
        region_allowed = normalize_filter_values(
            context.filters.get("r_p")
        ) or normalize_filter_values(context.filters.get("r_f"))
        if region_allowed:
            if "aggregated_mrio_code" in names:
                mask = out.index.get_level_values("aggregated_mrio_code").isin(region_allowed)
                out = out.loc[mask]
            elif "mrio_code" in names:
                mask = out.index.get_level_values("mrio_code").isin(region_allowed)
                out = out.loc[mask]
        for axis, allowed in allowed_by_axis.items():
            if not allowed or axis not in names:
                continue
            mask = out.index.get_level_values(axis).isin(allowed)
            out = out.loc[mask]
        return out
    if out.index.name is None:
        raise ValueError(
            "Enacting series with a single index must define that index name "
            "so run-scope filtering can be applied deterministically."
        )
    allowed = allowed_by_axis.get(str(out.index.name))
    if allowed:
        mask = out.index.isin(allowed)
        out = out.loc[mask]
    return out


def _append_aggregated_mrio_code_level(
    *,
    series: pd.Series,
    region_label: str,
    source_key: str,
    agg_version: str | None,
) -> pd.Series:
    """Append aggregated_mrio_code level next to MRIO region level."""
    if not agg_version:
        return series
    if not isinstance(series.index, pd.MultiIndex):
        return series
    names = [str(n) for n in series.index.names]
    if region_label not in names:
        return series
    mapping = load_region_agg_mapping(
        source_key=source_key,
        agg_version=agg_version,
    )
    region_values = series.index.get_level_values(region_label)
    missing = sorted({str(code) for code in region_values if code not in mapping})
    if missing:
        raise ValueError(
            "Regional MRIO aggregation and disaggregation map is missing MRIO labels "
            "referenced by LCIA enacting metric "
            f"outputs. Missing labels (sample): {missing[:10]}"
        )
    aggregated_values = region_values.map(mapping.__getitem__)

    arrays: list[pd.Index] = []
    out_names: list[str] = []
    for pos, name in enumerate(names):
        level_values = series.index.get_level_values(pos)
        arrays.append(pd.Index(level_values, name=name))
        out_names.append(name)
        if name == region_label:
            arrays.append(pd.Index(aggregated_values, name="aggregated_mrio_code"))
            out_names.append("aggregated_mrio_code")
    new_index = pd.MultiIndex.from_arrays(arrays, names=out_names)
    return pd.Series(series.to_numpy(), index=new_index, name=series.name)


def _l1_kinds_for_selected_method(
    *,
    context: RunContext,
    l1_method: str,
) -> set[str]:
    """Resolve LCIA boundary kinds for one selected L1 method."""
    del context
    return set(REGISTRY.l1_kinds_for_method(l1_method))
