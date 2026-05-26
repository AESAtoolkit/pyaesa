"""L2*b overlap validation helpers."""

from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from pyaesa.asocc.methods.registry.registry import REGISTRY

from .io_helpers import (
    VALIDATION_METHOD_COLUMN,
    aggregate_share_by_group_keys,
    clean_text,
    is_lcia_output,
    is_missing_scalar,
    list_output_files,
    method_label_from_row,
    parse_lcia_method,
    parse_optional_int,
    parse_optional_int_or_empty,
    read_output,
    scalar_float,
    with_validation_method_column,
)
from .l2_star_b_ar_expected import (
    load_l2_star_b_ar_cba_td_expected_global_by_impact,
    method_is_ar_cba_td,
)
from .l2_star_b_load_inputs import L2StarBTotals, load_l2_star_b_totals
from .l2_star_b_overlap_expected import (
    L2StarBOverlapRequest,
    expected_l2_star_b_overlap,
    method_l2_star_b_in_l1_axis,
)
from .l2_contexts import L2RunContext as L2StarBValidationContext
from .report_metrics import share_metric_fields
from .l2_two_step_helpers import split_l1_l2_method
from .lcia_helpers import FyShareRowRequest, fy_share_for_row

_L2_STAR_B_IN_L1_METHODS = {"UT(FDa)", "UT(GVAa)"}


class _L2StarBMethodContext(NamedTuple):
    """Context for one method level validation in one output file."""

    run: L2StarBValidationContext
    bucket: str
    path: Path
    is_lcia: bool
    lcia_method: str | None
    l1_method: str | None
    l2_method: str
    group_cols: list[str]
    axis: str | None
    totals: L2StarBTotals


def _fixed_l2_axis(axis: str | None) -> Callable[[str], str | None]:
    """Return an axis selector callback with a fixed axis for one grouped row."""

    def _selector(_fu_code: str, axis_value: str | None = axis) -> str | None:
        """Ignore FU input and always return the pre bound axis name."""
        return axis_value

    return _selector


def _metric_fields_for_l2b(
    base_sum: float,
    fy_add: float,
    checked: float,
    expected: float,
    abs_error: float,
) -> dict[str, float]:
    """Return common share metrics for L2*b report rows."""
    keys = ("base_sum", "fy_add", "checked", "expected", "abs_error")
    values = (base_sum, fy_add, checked, expected, abs_error)
    payload = dict(zip(keys, values))
    return share_metric_fields(**payload)


def _reference_year_for_calc(item: pd.Series) -> int | None:
    """Parse optional reference year from a grouped output row."""
    value = item.get("reference_year", "")
    if value == "" or is_missing_scalar(value):
        return None
    return parse_optional_int(value)


def _reference_year_for_output(item: pd.Series) -> int | str:
    """Format optional reference year for CSV output (int or empty string)."""
    return parse_optional_int_or_empty(item.get("reference_year", ""))


def _resolve_l2_star_b_grouping_keys(
    *,
    method_df: pd.DataFrame,
    is_lcia: bool,
    bucket: str,
    l2_method: str,
) -> tuple[list[str], str | None]:
    """Return group by keys and optional L2 axis used for one method table."""
    group_cols = [VALIDATION_METHOD_COLUMN]
    if is_lcia and "impact" in method_df.columns:
        group_cols.append("impact")
    if "reference_year" in method_df.columns:
        group_cols.append("reference_year")
    axis: str | None = None
    if bucket == "l2_in_l1":
        axis = method_l2_star_b_in_l1_axis(
            l2_method=l2_method,
            columns=list(method_df.columns),
        )
        if axis and axis in method_df.columns:
            group_cols.append(axis)
    return group_cols, axis


def _fy_add_for_row(
    *,
    method_context: _L2StarBMethodContext,
    item: pd.Series,
    l2_is_lcia: bool,
) -> float:
    """Return additive F_Y share for one grouped row."""
    if not (method_context.is_lcia and method_context.lcia_method and l2_is_lcia):
        return 0.0
    ref_year = _reference_year_for_calc(item)
    return float(
        fy_share_for_row(
            FyShareRowRequest(
                row=item,
                fu_code=method_context.run.fu_code,
                l2_bucket=method_context.bucket,
                source=method_context.run.source,
                year=method_context.run.year,
                reference_year=ref_year,
                lcia_method=method_context.lcia_method,
                l2_country_axis_fn=_fixed_l2_axis(method_context.axis),
                matrix_version=method_context.run.matrix_version,
            )
        )
    )


def _ar_cba_td_override(
    *,
    method_context: _L2StarBMethodContext,
    item: pd.Series,
    checked: float,
    l2_is_lcia: bool,
) -> tuple[float, bool, str] | None:
    """Return AR(CBA_TD) deterministic override when applicable."""
    if not (
        l2_is_lcia
        and method_context.l1_method is None
        and method_is_ar_cba_td(method_context.l2_method)
        and method_context.lcia_method
        and method_context.bucket == "l2_vs_global"
    ):
        return None
    impact = clean_text(item.get("impact"))
    ref_year = _reference_year_for_calc(item)
    exp_by_impact = load_l2_star_b_ar_cba_td_expected_global_by_impact(
        source=method_context.run.source,
        year=method_context.run.year,
        reference_year=ref_year,
        lcia_method=method_context.lcia_method,
        matrix_version=method_context.run.matrix_version,
    )
    expected = scalar_float(exp_by_impact.get(impact, np.nan))
    ok = bool(np.isclose(checked, expected, atol=method_context.run.atol, rtol=0))
    rule = "L2*b AR(E^{CBA_TD}): sum = E_CBA_TD/E_CBA_FD + F_Y/E_CBA_FD"
    return expected, ok, rule


def _expected_for_row(
    *,
    method_context: _L2StarBMethodContext,
    item: pd.Series,
    checked: float,
    l2_is_lcia: bool,
) -> tuple[float, bool, str]:
    """Return deterministic expected overlap and pass state for one row."""
    expected, ok, rule = expected_l2_star_b_overlap(
        L2StarBOverlapRequest(
            checked=checked,
            validation_project_name_root=method_context.run.validation_project_name_root,
            source=method_context.run.source,
            matrix_version=method_context.run.matrix_version,
            agg_reg=method_context.run.agg_reg,
            group_indices=method_context.run.group_indices,
            l1_mode=method_context.run.l1_mode,
            output_format=method_context.run.output_format,
            year=method_context.run.year,
            bucket=method_context.bucket,
            l1_method=method_context.l1_method,
            l2_method=method_context.l2_method,
            item=item,
            totals=method_context.totals,
            atol=method_context.run.atol,
        )
    )
    override = _ar_cba_td_override(
        method_context=method_context,
        item=item,
        checked=checked,
        l2_is_lcia=l2_is_lcia,
    )
    if override is not None:
        expected, ok, rule = override
    if not np.isfinite(expected):
        return expected, False, "Missing deterministic expected overlap for L2*b method"
    return expected, ok, rule


def _build_l2_star_b_row(
    *,
    method_context: _L2StarBMethodContext,
    item: pd.Series,
    base_sum: float,
    fy_add: float,
    expected_outcome: tuple[float, bool, str],
) -> dict[str, object]:
    """Build one normalized report row for the L2*b validation CSV."""
    expected, ok, rule = expected_outcome
    checked = base_sum + fy_add
    is_deterministic = bool(np.isfinite(expected))
    abs_error = abs(checked - float(expected)) if is_deterministic else np.nan
    metric_fields = _metric_fields_for_l2b(
        base_sum,
        fy_add,
        checked,
        expected,
        abs_error,
    )
    group_key = "|".join(
        f"{col}={item.get(col)}" for col in method_context.group_cols if col in item.index
    )
    return {
        "group_indices": "ungrouped",
        "source": method_context.run.source,
        "fu_code": method_context.run.fu_code,
        "year": int(method_context.run.year),
        "l1_reg_aggreg": method_context.run.l1_mode,
        "bucket": method_context.bucket,
        "l2_country_axis": method_context.axis or "",
        "l2_country_code": clean_text(item.get(method_context.axis)) if method_context.axis else "",
        "file": str(method_context.path),
        "method": method_label_from_row(item),
        "impact": clean_text(item.get("impact")),
        "reference_year": _reference_year_for_output(item),
        "group_key": group_key,
        **metric_fields,
        "atol_used": float(method_context.run.atol) if is_deterministic else np.nan,
        "passed": bool(ok),
        "rule": rule,
    }


def _rows_for_method(
    *,
    method_context: _L2StarBMethodContext,
    method_frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Validate one method table and return report rows."""
    year_col = str(method_context.run.year)
    sums = aggregate_share_by_group_keys(
        method_frame,
        year_col=year_col,
        group_cols=method_context.group_cols,
    )
    l2_is_lcia = REGISTRY.method_requires_lcia(
        method_context.l2_method,
        method_context.run.fu_code,
    )
    rows: list[dict[str, object]] = []
    for _, item in sums.iterrows():
        base_sum = scalar_float(item.get("sum_share"))
        fy_add = _fy_add_for_row(
            method_context=method_context,
            item=item,
            l2_is_lcia=l2_is_lcia,
        )
        checked = base_sum + fy_add
        expected, ok, rule = _expected_for_row(
            method_context=method_context,
            item=item,
            checked=checked,
            l2_is_lcia=l2_is_lcia,
        )
        rows.append(
            _build_l2_star_b_row(
                method_context=method_context,
                item=item,
                base_sum=base_sum,
                fy_add=fy_add,
                expected_outcome=(expected, ok, rule),
            )
        )
    return rows


def _iter_method_contexts(
    *,
    context: L2StarBValidationContext,
    totals: L2StarBTotals,
) -> list[tuple[_L2StarBMethodContext, pd.DataFrame]]:
    """Return method level validation contexts and their filtered frames."""
    items: list[tuple[_L2StarBMethodContext, pd.DataFrame]] = []
    year_col = str(context.year)
    for bucket in context.buckets:
        bucket_dir = context.l2_root / bucket
        for path in list_output_files(bucket_dir, preferred_format=context.output_format):
            frame = with_validation_method_column(read_output(path))
            if year_col not in frame.columns or VALIDATION_METHOD_COLUMN not in frame.columns:
                continue
            items.extend(
                _method_contexts_for_frame(
                    context=context,
                    totals=totals,
                    bucket=bucket,
                    path=path,
                    frame=frame,
                )
            )
    return items


def _method_contexts_for_frame(
    *,
    context: L2StarBValidationContext,
    totals: L2StarBTotals,
    bucket: str,
    path: Path,
    frame: pd.DataFrame,
) -> list[tuple[_L2StarBMethodContext, pd.DataFrame]]:
    """Return method contexts for one loaded level-2 output frame."""
    items: list[tuple[_L2StarBMethodContext, pd.DataFrame]] = []
    is_lcia = is_lcia_output(path)
    lcia_method = parse_lcia_method(path) if is_lcia else None
    method_values = frame[VALIDATION_METHOD_COLUMN].dropna().astype("string").str.strip()
    for method_label in sorted(method_values[method_values != ""].astype(str).unique()):
        method_frame = frame.loc[
            frame[VALIDATION_METHOD_COLUMN].astype("string").str.strip() == method_label
        ].copy()
        l1_method, l2_method = split_l1_l2_method(method_label, context.fu_code)
        if bucket == "l2_in_l1" and l2_method not in _L2_STAR_B_IN_L1_METHODS:
            continue
        group_cols, axis = _resolve_l2_star_b_grouping_keys(
            method_df=method_frame,
            is_lcia=is_lcia,
            bucket=bucket,
            l2_method=l2_method,
        )
        items.append(
            (
                _L2StarBMethodContext(
                    run=context,
                    bucket=bucket,
                    path=path,
                    is_lcia=is_lcia,
                    lcia_method=lcia_method,
                    l1_method=l1_method,
                    l2_method=l2_method,
                    group_cols=group_cols,
                    axis=axis,
                    totals=totals,
                ),
                method_frame,
            )
        )
    return items


def validate_l2_star_b_outputs(
    context: L2StarBValidationContext,
) -> list[dict[str, object]]:
    """Validate L2*b output files and return row wise report items."""
    rows: list[dict[str, object]] = []
    totals = load_l2_star_b_totals(
        context.source,
        context.year,
        matrix_version=context.matrix_version,
    )
    for method_context, method_frame in _iter_method_contexts(
        context=context,
        totals=totals,
    ):
        rows.extend(
            _rows_for_method(
                method_context=method_context,
                method_frame=method_frame,
            )
        )
    return rows
