"""Sparse matrix routines for weighted MRIO aggregation and disaggregation."""

from collections.abc import Iterable, Sequence
from typing import Any, cast

import pandas as pd
import pymrio
from scipy import sparse

from pyaesa.process.mrios.utils.aggregation.aggregation import AggregationSpec

_PRODUCT_AXIS_NAMES = ("region", "sector")


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


def _rows_by_original(spec: AggregationSpec) -> dict[int, list[tuple[int, float]]]:
    """Return aggregated row positions keyed by original label position."""
    out: dict[int, list[tuple[int, float]]] = {}
    for original_idx, aggregated_idx, weight in spec.rows:
        out.setdefault(original_idx, []).append((aggregated_idx, float(weight)))
    return out


def _product_concordance(
    axis: pd.Index,
    *,
    region_spec: AggregationSpec,
    sector_spec: AggregationSpec,
) -> tuple[sparse.csr_matrix, pd.MultiIndex]:
    """Build product concordance and aggregated product index for one product axis."""
    product_axis = _normal_product_axis(axis)
    region_pos = {label: idx for idx, label in enumerate(region_spec.original_order)}
    sector_pos = {label: idx for idx, label in enumerate(sector_spec.original_order)}
    region_rows = _rows_by_original(region_spec)
    sector_rows = _rows_by_original(sector_spec)
    out_rows: list[int] = []
    out_cols: list[int] = []
    out_values: list[float] = []
    aggregated_sector_count = len(sector_spec.aggregated_labels)
    for col_idx, (region, sector) in enumerate(product_axis.tolist()):
        reg_idx = region_pos[str(region)]
        sec_idx = sector_pos[str(sector)]
        for aggregated_reg_idx, reg_weight in region_rows[reg_idx]:
            for aggregated_sec_idx, sec_weight in sector_rows[sec_idx]:
                out_rows.append(aggregated_reg_idx * aggregated_sector_count + aggregated_sec_idx)
                out_cols.append(col_idx)
                out_values.append(reg_weight * sec_weight)
    index = pd.MultiIndex.from_product(
        [region_spec.aggregated_labels, sector_spec.aggregated_labels],
        names=["region", "sector"],
    )
    return (
        sparse.csr_matrix(
            (out_values, (out_rows, out_cols)),
            shape=(len(index), len(product_axis)),
            dtype=float,
        ),
        index,
    )


def _final_demand_concordance(
    axis: pd.Index,
    *,
    region_spec: AggregationSpec,
) -> tuple[sparse.csr_matrix, pd.Index | pd.MultiIndex]:
    """Build final demand concordance and aggregated final demand index."""
    normal_axis = _normal_final_demand_axis(axis)
    if isinstance(normal_axis, pd.MultiIndex):
        categories = list(dict.fromkeys(normal_axis.droplevel("region").tolist()))
        category_pos = {category: idx for idx, category in enumerate(categories)}
        category_count = len(categories)
        region_pos = {label: idx for idx, label in enumerate(region_spec.original_order)}
        region_rows = _rows_by_original(region_spec)
        out_rows: list[int] = []
        out_cols: list[int] = []
        out_values: list[float] = []
        for col_idx, label in enumerate(normal_axis.tolist()):
            region = str(label[0])
            category = label[1:] if len(label) > 2 else label[1]
            for aggregated_reg_idx, weight in region_rows[region_pos[region]]:
                out_rows.append(aggregated_reg_idx * category_count + category_pos[category])
                out_cols.append(col_idx)
                out_values.append(weight)
        category_arrays = list(zip(*categories, strict=False)) if categories else []
        arrays: list[Sequence[object]] = [
            [region for region in region_spec.aggregated_labels for _ in range(category_count)]
        ]
        arrays.extend(
            list(values) * len(region_spec.aggregated_labels) for values in category_arrays
        )
        if len(normal_axis.names) == 2:
            arrays = [
                arrays[0],
                [category for _ in region_spec.aggregated_labels for category in categories],
            ]
        columns = pd.MultiIndex.from_arrays(arrays, names=list(normal_axis.names))
        return (
            sparse.csr_matrix(
                (out_values, (out_rows, out_cols)),
                shape=(len(columns), len(normal_axis)),
                dtype=float,
            ),
            columns,
        )

    region_pos = {label: idx for idx, label in enumerate(region_spec.original_order)}
    region_rows = _rows_by_original(region_spec)
    out_rows = []
    out_cols = []
    out_values = []
    for col_idx, region in enumerate(normal_axis.tolist()):
        for aggregated_reg_idx, weight in region_rows[region_pos[str(region)]]:
            out_rows.append(aggregated_reg_idx)
            out_cols.append(col_idx)
            out_values.append(weight)
    columns = pd.Index(region_spec.aggregated_labels, name="region")
    return (
        sparse.csr_matrix(
            (out_values, (out_rows, out_cols)),
            shape=(len(columns), len(normal_axis)),
            dtype=float,
        ),
        columns,
    )


def _sparse_frame_product(
    left: sparse.csr_matrix,
    frame: pd.DataFrame,
    right: sparse.csr_matrix,
    *,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    """Return ``left @ frame @ right.T`` as a DataFrame."""
    frame_values = frame.to_numpy(dtype=float, copy=False)
    left_values = left @ frame_values
    values = (right @ left_values.T).T
    return pd.DataFrame(values, index=index, columns=columns)


def aggregate_iosys_weighted(
    *,
    iosys: pymrio.IOSystem,
    agg_reg: bool,
    region_spec: AggregationSpec,
    sector_spec: AggregationSpec,
    clear_extension_derived_accounts,
) -> None:
    """Aggregate required MRIO matrices with sparse pymrio concordance semantics."""
    z = cast(pd.DataFrame, iosys.Z)
    y = cast(pd.DataFrame, iosys.Y)
    product_conc, product_index = _product_concordance(
        z.index,
        region_spec=region_spec,
        sector_spec=sector_spec,
    )
    fd_conc, fd_columns = _final_demand_concordance(y.columns, region_spec=region_spec)

    iosys.Z = _sparse_frame_product(
        product_conc,
        z,
        product_conc,
        index=product_index,
        columns=product_index,
    )
    iosys.Y = _sparse_frame_product(
        product_conc,
        y,
        fd_conc if agg_reg else sparse.csr_matrix(sparse.eye(y.shape[1], format="csr")),
        index=product_index,
        columns=fd_columns if agg_reg else y.columns,
    )

    unit_obj = getattr(iosys, "unit", None)
    if isinstance(unit_obj, pd.DataFrame):
        value = unit_obj.iloc[0].tolist()[0] if not unit_obj.empty else None
        iosys.unit = pd.DataFrame(value, index=product_index, columns=unit_obj.columns)
    elif isinstance(unit_obj, pd.Series):
        value = unit_obj.iloc[0] if len(unit_obj) else None
        iosys.unit = pd.Series(value, index=product_index, name=unit_obj.name)

    get_extensions = getattr(iosys, "get_extensions", None)
    if not callable(get_extensions):
        return

    ext_iterable = cast(Iterable[Any], get_extensions(data=True))
    for ext in ext_iterable:
        f_obj = getattr(ext, "F", None)
        if isinstance(f_obj, pd.DataFrame):
            f_values = f_obj.to_numpy(dtype=float, copy=False)
            ext.F = pd.DataFrame(
                (product_conc @ f_values.T).T,
                index=f_obj.index,
                columns=product_index,
            )

        if agg_reg:
            fy_obj = getattr(ext, "F_Y", None)
            if isinstance(fy_obj, pd.DataFrame):
                fy_values = fy_obj.to_numpy(dtype=float, copy=False)
                ext.F_Y = pd.DataFrame(
                    (fd_conc @ fy_values.T).T,
                    index=fy_obj.index,
                    columns=fd_columns,
                )

        clear_extension_derived_accounts(ext)
