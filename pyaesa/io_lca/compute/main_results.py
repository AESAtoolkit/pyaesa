"""Main IO-LCA results extraction from processed LCIA enacting metric matrices."""

from typing import cast

import pandas as pd

from pyaesa.asocc.runtime.scope.filtering import (
    normalize_filter_values,
    slice_frame_any_axis,
)
from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec
from pyaesa.io_lca.data.loaders import (
    YearMethodMainPayload,
    impact_unit_text_map,
)
from pyaesa.io_lca.orchestration.io.method_support import main_result_columns


def _stack_to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Stack any DataFrame payload to long form with ``lca_value`` column."""
    if isinstance(frame.columns, pd.MultiIndex):
        stacked = frame.stack(list(range(frame.columns.nlevels)), future_stack=True)
    else:
        stacked = frame.stack(future_stack=True)
    return cast(pd.Series, stacked).rename("lca_value").reset_index()


def _apply_selector_slices(
    *,
    frame: pd.DataFrame,
    spec: IOLCAFUSpec,
    filters: dict[str, list[str] | None],
) -> pd.DataFrame:
    """Apply user selectors on the corresponding internal matrix axes."""
    out = frame
    for user_axis in spec.selector_axes:
        values = filters.get(user_axis)
        if not values:
            continue
        allowed = normalize_filter_values(values)
        out = slice_frame_any_axis(out, axis_name=user_axis, allowed=allowed)
    return out


def _require_nonzero_requested_impacts(*, payload: YearMethodMainPayload) -> None:
    """Fail when a processed LCIA impact matrix is entirely zero for one year."""
    metric = payload.metric
    if metric.empty:
        return
    numeric = metric.apply(pd.to_numeric, errors="raise")
    impact_totals = cast(
        pd.Series,
        pd.Series(
            numeric.abs().sum(axis=1, min_count=1).to_numpy(dtype=float),
            index=pd.Index(metric.index.astype(str), name="impact"),
        )
        .groupby(level=0, sort=True)
        .sum(min_count=1),
    )
    impact_labels = [str(impact) for impact in impact_totals.index.tolist()]
    impact_values = impact_totals.to_numpy(dtype=float)
    zero_impacts = [
        impact
        for impact, value in zip(impact_labels, impact_values, strict=True)
        if pd.notna(value) and float(value) == 0.0
    ]
    if zero_impacts:
        raise ValueError(
            "Processed MRIO LCIA output contains impact categories with zero values "
            "across all matrix cells. deterministic_io_lca treats these rows as "
            "missing data. "
            f"lcia_method='{payload.lcia_method}', year={int(payload.year)}, "
            f"impacts={zero_impacts}."
        )


def _resolve_unit_map_for_impacts(
    *,
    impacts: pd.Series,
    unit_by_impact: pd.Series,
) -> pd.Series:
    """Resolve units for detailed impacts using exact or parent prefix matching.

    Some processed LCIA payloads expose detailed impact labels (for example
    ``"BI FD GHG"``) while metadata unit maps are keyed at parent level
    (for example ``"BI FD"``). We map exact keys first, then fall back to the
    longest matching parent prefix.
    """
    unit_map = impact_unit_text_map(unit_by_impact=unit_by_impact)
    unresolved: list[str] = []
    for impact in sorted(set(impacts.astype(str).tolist())):
        if impact in unit_map:
            continue
        candidates = [parent for parent in unit_map if impact.startswith(f"{parent} ")]
        if not candidates:
            unresolved.append(impact)
            continue
        parent = sorted(candidates, key=len, reverse=True)[0]
        unit_map[impact] = unit_map[parent]
    if unresolved:
        missing = sorted(unresolved)
        raise ValueError(
            f"Missing LCIA impact units in processed metadata. Unresolved impacts: {missing[:10]}."
        )
    return pd.Series(unit_map, dtype=str)


def build_main_results_rows(
    *,
    payload: YearMethodMainPayload,
    spec: IOLCAFUSpec,
    filters: dict[str, list[str] | None],
) -> pd.DataFrame:
    """Build long form main IO-LCA rows for one year/LCIA method.

    Args:
        payload: Loaded year/method metrics and unit metadata.
        spec: FU mapping payload.
        filters: User selectors.

    Returns:
        Long form rows with deterministic schema.
    """
    _require_nonzero_requested_impacts(payload=payload)
    sliced = _apply_selector_slices(frame=payload.metric, spec=spec, filters=filters)
    if sliced.empty:
        return pd.DataFrame(columns=main_result_columns(spec.selector_axes))
    long_frame = _stack_to_long(sliced)

    impact_series = cast(pd.Series, long_frame["impact"]).astype(str)
    long_frame["impact"] = impact_series
    long_frame["lca_value"] = pd.to_numeric(long_frame["lca_value"], errors="raise")
    resolved_unit_map = _resolve_unit_map_for_impacts(
        impacts=impact_series,
        unit_by_impact=payload.unit_by_impact,
    )
    long_frame["lcia_method"] = str(payload.lcia_method)
    long_frame["impact_unit"] = impact_series.map(resolved_unit_map.to_dict())
    long_frame["year"] = int(payload.year)
    output_cols = main_result_columns(spec.selector_axes)
    return long_frame.loc[:, output_cols].copy()
