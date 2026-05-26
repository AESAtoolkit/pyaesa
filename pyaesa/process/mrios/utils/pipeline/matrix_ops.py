"""Shared ownership for per year MRIO compute routines.

These functions keep heavy table transformations and IO matrix construction
separate from ``parse_and_calc_year`` orchestration.
"""

from collections.abc import Hashable, Iterable, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import pymrio

from pyaesa.download.mrios.utils.logging import suppress_pymrio_logging
from pyaesa.process.mrios.utils.aggregation.aggregation import AggregationSpec
from pyaesa.process.mrios.utils.pipeline.weighted_aggregation import aggregate_iosys_weighted

_PRODUCT_AXIS_NAMES = ("region", "sector")


def _map_values(values: pd.Index, mapping: dict[str, str]) -> list[str]:
    """Return mapped labels for ``values`` using ``mapping`` with identity fallback."""
    out: list[str] = []
    for value in values.tolist():
        key = str(value)
        out.append(mapping.get(key, key))
    return out


def _normal_product_axis(axis: pd.Index) -> pd.MultiIndex:
    """Return a product axis with canonical ``region`` and ``sector`` levels."""
    normal_axis = cast(pd.MultiIndex, axis)
    names = list(normal_axis.names)
    for position, expected in enumerate(_PRODUCT_AXIS_NAMES):
        names[position] = expected
    return normal_axis.set_names(names)


def _normal_final_demand_axis(axis: pd.Index) -> pd.Index:
    """Return a final demand axis with canonical leading ``region`` level."""
    if isinstance(axis, pd.MultiIndex):
        names = list(axis.names)
        names[0] = "region"
        if len(names) > 1 and names[1] is None:
            names[1] = "final_demand"
        return axis.set_names(names)
    return axis.rename("region")


def _normalize_mrio_axes(iosys: pymrio.IOSystem) -> None:
    """Normalize PyMRIO axes to UNCASExt canonical labels in place."""
    z = cast(pd.DataFrame, iosys.Z)
    y = cast(pd.DataFrame, iosys.Y)

    z.index = _normal_product_axis(z.index)
    z.columns = _normal_product_axis(z.columns)
    y.index = _normal_product_axis(y.index)
    y.columns = _normal_final_demand_axis(y.columns)

    unit_obj = getattr(iosys, "unit", None)
    if isinstance(unit_obj, pd.DataFrame) or isinstance(unit_obj, pd.Series):
        unit_obj.index = _normal_product_axis(unit_obj.index)

    get_extensions = getattr(iosys, "get_extensions", None)
    if not callable(get_extensions):
        return

    ext_iterable = cast(Iterable[Any], get_extensions(data=True))
    for ext in ext_iterable:
        f_obj = getattr(ext, "F", None)
        if isinstance(f_obj, pd.DataFrame):
            f_obj.columns = _normal_product_axis(f_obj.columns)
        fy_obj = getattr(ext, "F_Y", None)
        if isinstance(fy_obj, pd.DataFrame):
            fy_obj.columns = _normal_final_demand_axis(fy_obj.columns)


def _build_product_agg_key(
    axis: pd.Index,
    *,
    region_map: dict[str, str],
    sector_map: dict[str, str],
) -> pd.MultiIndex:
    """Build aggregated product key (region, sector) for a product axis."""
    product_axis = _normal_product_axis(axis)
    region_values = product_axis.get_level_values("region")
    sector_values = product_axis.get_level_values("sector")

    mapped_region = _map_values(region_values, region_map)
    mapped_sector = _map_values(sector_values, sector_map)
    return pd.MultiIndex.from_arrays(
        [mapped_region, mapped_sector],
        names=["region", "sector"],
    )


def _build_fd_agg_key(
    columns: pd.Index,
    *,
    region_map: dict[str, str],
) -> pd.Index | pd.MultiIndex:
    """Build aggregated final demand column key (region remapped, other levels kept)."""
    normal_columns = _normal_final_demand_axis(columns)
    if isinstance(normal_columns, pd.MultiIndex):
        mapped_region = _map_values(normal_columns.get_level_values("region"), region_map)
        arrays: list[pd.Index | list[str]] = []
        for level in range(normal_columns.nlevels):
            if level == 0:
                arrays.append(mapped_region)
            else:
                arrays.append(normal_columns.get_level_values(level))
        return pd.MultiIndex.from_arrays(arrays, names=list(normal_columns.names))
    return pd.Index(_map_values(normal_columns, region_map), name="region")


def _aggregate_rows(df: pd.DataFrame, key: pd.Index | pd.MultiIndex) -> pd.DataFrame:
    """Aggregate DataFrame rows using ``key``."""
    out = cast(pd.DataFrame, df.groupby(key, sort=False).sum(min_count=1))
    if isinstance(key, pd.MultiIndex) and not isinstance(out.index, pd.MultiIndex):
        out.index = _multiindex_from_labels(
            out.index,
            names=list(key.names),
            label="Aggregated row index",
        )
    return out


def _aggregate_columns(
    df: pd.DataFrame,
    key: pd.Index | pd.MultiIndex,
) -> pd.DataFrame:
    """Aggregate DataFrame columns using ``key``."""
    df_transposed = cast(pd.DataFrame, df.T)
    aggregated = cast(pd.DataFrame, df_transposed.groupby(key, sort=False).sum(min_count=1))
    out = cast(pd.DataFrame, aggregated.T)
    if isinstance(key, pd.MultiIndex) and not isinstance(out.columns, pd.MultiIndex):
        out.columns = _multiindex_from_labels(
            out.columns,
            names=list(key.names),
            label="Aggregated column index",
        )
    return out


def _multiindex_from_labels(
    index: pd.Index,
    *,
    names: Sequence[Hashable | None],
    label: str,
) -> pd.MultiIndex:
    """Return a MultiIndex built from aggregated tuple labels."""
    del label
    return pd.MultiIndex.from_tuples(
        cast(list[tuple[Hashable, ...]], index.tolist()),
        names=list(names),
    )


def _strict_spec_from_map(
    labels: list[str],
    mapping: dict[str, str],
) -> AggregationSpec:
    """Build a strict aggregation spec from the current mapping contract."""
    aggregated = _map_values(pd.Index(labels), mapping)
    aggregated_labels = tuple(dict.fromkeys(aggregated))
    aggregated_index = {label: idx for idx, label in enumerate(aggregated_labels)}
    return AggregationSpec(
        original_order=tuple(labels),
        aggregated_labels=aggregated_labels,
        weighted=False,
        rows=tuple(
            (original_idx, aggregated_index[aggregated_label], 1.0)
            for original_idx, aggregated_label in enumerate(aggregated)
        ),
    )


def _strict_map_from_spec(spec: AggregationSpec) -> dict[str, str]:
    """Return strict aggregation labels keyed by original label."""
    return {
        spec.original_order[original_idx]: spec.aggregated_labels[aggregated_idx]
        for original_idx, aggregated_idx, _ in spec.rows
    }


def _clear_extension_derived_accounts(ext: Any) -> None:
    """Clear extension attributes derived from IO system calculations."""
    derived_attrs = (
        "S",
        "S_Y",
        "M",
        "M_down",
        "D_cba",
        "D_pba",
        "D_imp",
        "D_exp",
        "D_cba_reg",
        "D_pba_reg",
        "D_imp_reg",
        "D_exp_reg",
        "D_cba_cap",
        "D_pba_cap",
        "D_imp_cap",
        "D_exp_cap",
    )
    for attr in derived_attrs:
        setattr(ext, attr, None)


def _aggregate_iosys_fast(
    *,
    iosys: pymrio.IOSystem,
    agg_reg: bool,
    region_map: dict[str, str] | None = None,
    sector_map: dict[str, str] | None = None,
    region_spec: AggregationSpec | None = None,
    sector_spec: AggregationSpec | None = None,
) -> None:
    """Aggregate only the matrices required by the minimal UNCASExt path.

    This function is narrower than ``pymrio.IOSystem.aggregate``. It updates the
    matrices used by the minimal processed MRIO route and clears derived accounts
    so UNCASExt can rebuild only the required matrices afterward removing unecessary
    processing.
    """
    _normalize_mrio_axes(iosys)
    z = cast(pd.DataFrame, iosys.Z)
    y = cast(pd.DataFrame, iosys.Y)

    regions, sectors = _labels_from_product_index(z.index)
    if region_spec is None:
        region_spec = _strict_spec_from_map(regions, region_map or {})
    if sector_spec is None:
        sector_spec = _strict_spec_from_map(sectors, sector_map or {})

    if region_spec.weighted or sector_spec.weighted:
        aggregate_iosys_weighted(
            iosys=iosys,
            agg_reg=agg_reg,
            region_spec=region_spec,
            sector_spec=sector_spec,
            clear_extension_derived_accounts=_clear_extension_derived_accounts,
        )
        iosys.x = None
        iosys.A = None
        iosys.L = None
        iosys.G = None
        return

    region_map = _strict_map_from_spec(region_spec)
    sector_map = _strict_map_from_spec(sector_spec)

    product_row_key = _build_product_agg_key(
        z.index,
        region_map=region_map,
        sector_map=sector_map,
    )
    product_col_key = _build_product_agg_key(
        z.columns,
        region_map=region_map,
        sector_map=sector_map,
    )

    z_agg = _aggregate_rows(z, product_row_key)
    z_agg = _aggregate_columns(z_agg, product_col_key)
    iosys.Z = z_agg

    y_agg = _aggregate_rows(y, product_row_key)
    if agg_reg:
        y_col_key = _build_fd_agg_key(y_agg.columns, region_map=region_map)
        y_agg = _aggregate_columns(y_agg, y_col_key)
    iosys.Y = y_agg

    unit_obj = getattr(iosys, "unit", None)
    if isinstance(unit_obj, pd.DataFrame) and unit_obj.index.equals(z.index):
        aggregated_unit = cast(pd.DataFrame, unit_obj.groupby(product_row_key, sort=False).first())
        aggregated_unit.index = _multiindex_from_labels(
            aggregated_unit.index,
            names=list(product_row_key.names),
            label="Aggregated unit DataFrame index",
        )
        iosys.unit = aggregated_unit
    elif isinstance(unit_obj, pd.Series) and unit_obj.index.equals(z.index):
        aggregated_unit = cast(pd.Series, unit_obj.groupby(product_row_key, sort=False).first())
        aggregated_unit.index = _multiindex_from_labels(
            aggregated_unit.index,
            names=list(product_row_key.names),
            label="Aggregated unit Series index",
        )
        iosys.unit = aggregated_unit

    get_extensions = getattr(iosys, "get_extensions", None)
    if callable(get_extensions):
        ext_iterable = cast(Iterable[Any], get_extensions(data=True))
        for ext in ext_iterable:
            f_obj = getattr(ext, "F", None)
            if isinstance(f_obj, pd.DataFrame):
                f_col_key = _build_product_agg_key(
                    f_obj.columns,
                    region_map=region_map,
                    sector_map=sector_map,
                )
                ext.F = _aggregate_columns(f_obj, f_col_key)

            if agg_reg:
                fy_obj = getattr(ext, "F_Y", None)
                if isinstance(fy_obj, pd.DataFrame):
                    fy_col_key = _build_fd_agg_key(fy_obj.columns, region_map=region_map)
                    ext.F_Y = _aggregate_columns(fy_obj, fy_col_key)

            _clear_extension_derived_accounts(ext)

    iosys.x = None
    iosys.A = None
    iosys.L = None
    iosys.G = None


def _calc_aggregated_full_system_after_fast_aggregation(*, iosys: pymrio.IOSystem) -> None:
    """Rebuild the aggregated core system, then let PyMRIO expand extensions.

    After ``_aggregate_iosys_fast(...)`` aggregates the matrices, full aggregated
    runs need the core matrices rebuilt on the same aggregated basis before
    PyMRIO computes extension accounts.
    """
    _calc_core_system_minimal(
        iosys=iosys,
        include_ghosh=True,
    )
    with suppress_pymrio_logging():
        iosys.calc_extensions(include_ghosh=True)


def _agg_final_demand_by_region(y: pd.DataFrame) -> pd.DataFrame:
    """Aggregate final demand columns to regions."""
    columns = _normal_final_demand_axis(y.columns)
    y = y.copy(deep=False)
    y.columns = columns
    aggregated = cast(pd.DataFrame, y.T.groupby(level="region", sort=False).sum(min_count=1))
    return cast(pd.DataFrame, aggregated.T)


def _labels_from_product_index(index: pd.Index) -> tuple[list[str], list[str]]:
    """Return region/sector labels from a product MultiIndex."""
    product_index = _normal_product_axis(index)
    reg_values = product_index.get_level_values("region")
    sec_values = product_index.get_level_values("sector")

    regions = list(dict.fromkeys(str(value) for value in reg_values.tolist()))
    sectors = list(dict.fromkeys(str(value) for value in sec_values.tolist()))
    return regions, sectors


def _calc_x_from_clipped_fd(*, z: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """Compute output vector from ``Z`` and region aggregated clipped final demand."""
    y_reg_clipped = _agg_final_demand_by_region(y).clip(lower=0.0)
    x_series = z.sum(axis=1, min_count=1).add(
        y_reg_clipped.sum(axis=1, min_count=1),
        fill_value=0.0,
    )
    out = pd.DataFrame(x_series.astype(float), columns=["indout"])
    out.index = z.index
    return out


def _calc_core_system_minimal(
    *,
    iosys: pymrio.IOSystem,
    include_ghosh: bool,
) -> None:
    """Compute only core IO matrices required by UNCASExt processing."""
    _normalize_mrio_axes(iosys)
    z = cast(pd.DataFrame, iosys.Z)
    y = cast(pd.DataFrame, iosys.Y)
    iosys.x = _calc_x_from_clipped_fd(z=z, y=y)
    iosys.A = pymrio.calc_A(iosys.Z, iosys.x)
    iosys.L = pymrio.calc_L(iosys.A)

    if not include_ghosh:
        iosys.A = None
        iosys.G = None
        return

    l_frame = cast(pd.DataFrame, iosys.L)
    x_frame = cast(pd.DataFrame, iosys.x)
    x_vec = x_frame.iloc[:, 0].to_numpy(dtype=float)
    l_np = l_frame.to_numpy(dtype=float)
    inv_x = np.divide(1.0, x_vec, out=np.zeros_like(x_vec), where=x_vec != 0.0)
    g_np = (inv_x[:, None] * l_np) * x_vec[None, :]
    iosys.G = pd.DataFrame(g_np, index=l_frame.index, columns=l_frame.columns)
