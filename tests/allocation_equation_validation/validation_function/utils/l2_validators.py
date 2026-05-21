"""L2 validation helpers for allocation checks."""

from pathlib import Path
from typing import NamedTuple

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
    read_output,
    scalar_float,
    with_validation_method_column,
)
from .l2_two_step_helpers import split_l1_l2_method
from .l2_contexts import L2RunContext as L2ValidationContext
from .report_metrics import share_metric_fields
from .lcia_helpers import (
    FyShareRowRequest,
    LciaWeightingRequest,
    fy_share_for_row,
    l1_weight_coverage_for_lcia_global,
    weighted_l1_fy_global_share,
)

# L2 logic (non-L2*b): sum to one checks on L2 outputs.
_L2_STATIC_FIELDS: dict[str, object] = {
    "aggreg_indices": "ungrouped",
}


class _L2FileContext(NamedTuple):
    """Context for one L2 output file validation pass."""

    run: L2ValidationContext
    bucket: str
    path: Path
    is_lcia: bool
    lcia_method: str | None
    group_cols: list[str]
    axis: str | None


def l2_country_axis(fu_code: str) -> str | None:
    """Return country axis used for non-L2*b L2-in-L1 checks."""
    if fu_code == "L2.a.c":
        return "r_p"
    if fu_code in {"L2.a.a", "L2.b.a", "L2.c.a"}:
        return "r_f"
    return None


def _reference_year(item: pd.Series) -> int | None:
    """Parse optional reference year from a grouped output row."""
    value = item.get("reference_year", "")
    if value == "" or is_missing_scalar(value):
        return None
    return parse_optional_int(value)


def _resolve_l2_grouping_keys(
    *,
    df: pd.DataFrame,
    is_lcia: bool,
    bucket: str,
    fu_code: str,
) -> tuple[list[str], str | None]:
    """Return group by keys and optional L2 axis for one output table."""
    group_cols = [VALIDATION_METHOD_COLUMN]
    if is_lcia and "impact" in df.columns:
        group_cols.append("impact")
    if "reference_year" in df.columns:
        group_cols.append("reference_year")
    axis: str | None = None
    if bucket == "l2_in_l1":
        axis = l2_country_axis(fu_code)
        if axis and axis in df.columns:
            group_cols.append(axis)
    return group_cols, axis


def _lcia_weight_request(
    *,
    file_context: _L2FileContext,
    item: pd.Series,
    reference_year: int | None,
    l1_method: str,
) -> LciaWeightingRequest:
    """Build one LCIA weighting request for global L2 validation."""
    l2_method = method_label_from_row(item)
    boundary = "PBA" if "PBA" in l2_method else "CBA_FD"
    return LciaWeightingRequest(
        validation_project_name_root=file_context.run.validation_project_name_root,
        source=file_context.run.source,
        group_reg=file_context.run.group_reg,
        aggreg_indices=file_context.run.aggreg_indices,
        l1_mode=file_context.run.l1_mode,
        output_format=file_context.run.output_format,
        l1_method=l1_method,
        year=file_context.run.year,
        impact=clean_text(item.get("impact")),
        reference_year=reference_year,
        lcia_method=str(file_context.lcia_method),
        boundary=boundary,
        matrix_version=file_context.run.matrix_version,
    )


def _resolve_l2_fy_add(
    *,
    item: pd.Series,
    file_context: _L2FileContext,
) -> tuple[float, str, str]:
    """Return F_Y add back share and human readable rule label for one row."""
    validation_note = ""
    if not (file_context.is_lcia and file_context.lcia_method):
        return 0.0, "L2 non-LCIA sum to 1", validation_note

    method_label = method_label_from_row(item)
    l1_method, l2_method = split_l1_l2_method(method_label, file_context.run.fu_code)
    if not REGISTRY.method_requires_lcia(l2_method, file_context.run.fu_code):
        return 0.0, "L2 non-LCIA sum to 1", validation_note

    reference_year = _reference_year(item)
    if file_context.bucket == "l2_vs_global" and l1_method:
        weight_request = _lcia_weight_request(
            file_context=file_context,
            item=item,
            reference_year=reference_year,
            l1_method=l1_method,
        )
        fy_add = weighted_l1_fy_global_share(weight_request)
        effective_target, lost_weight, invalid_regions = l1_weight_coverage_for_lcia_global(
            weight_request
        )
        if invalid_regions and lost_weight > 0.0 and effective_target is not None:
            validation_note = (
                "LCIA denominator has zero/non-finite region entries for this "
                f"impact/reference-year (regions={invalid_regions}); excluded L1 "
                f"weight mass={lost_weight:.9g}, effective_target={effective_target:.9g}."
            )
        return float(fy_add), "L2 LCIA + F_Y sum to 1", validation_note

    fy_add = fy_share_for_row(
        FyShareRowRequest(
            row=item,
            fu_code=file_context.run.fu_code,
            l2_bucket=file_context.bucket,
            source=file_context.run.source,
            year=file_context.run.year,
            reference_year=reference_year,
            lcia_method=str(file_context.lcia_method),
            l2_country_axis_fn=l2_country_axis,
            matrix_version=file_context.run.matrix_version,
        )
    )
    return float(fy_add), "L2 LCIA + F_Y sum to 1", validation_note


def _build_l2_row(
    *,
    file_context: _L2FileContext,
    item: pd.Series,
    base_sum: float,
    fy_resolution: tuple[float, str, str],
) -> dict[str, object]:
    """Build one normalized report row for the non-L2*b validation CSV."""
    fy_add, rule, validation_note = fy_resolution
    ref_year = _reference_year(item)
    checked = base_sum + fy_add
    expected = 1.0
    abs_error = abs(checked - expected)
    group_key = "|".join(
        f"{col}={item.get(col)}" for col in file_context.group_cols if col in item.index
    )
    return {
        **_L2_STATIC_FIELDS,
        "source": file_context.run.source,
        "fu_code": file_context.run.fu_code,
        "year": int(file_context.run.year),
        "l1_reg_aggreg": file_context.run.l1_mode,
        "bucket": file_context.bucket,
        "l2_country_axis": file_context.axis or "",
        "l2_country_code": clean_text(item.get(file_context.axis)) if file_context.axis else "",
        "file": str(file_context.path),
        "method": method_label_from_row(item),
        "impact": clean_text(item.get("impact")),
        "reference_year": ("" if ref_year is None else ref_year),
        "group_key": group_key,
        **share_metric_fields(
            base_sum=base_sum,
            fy_add=fy_add,
            checked=checked,
            expected=expected,
            abs_error=abs_error,
        ),
        "atol_used": float(file_context.run.atol),
        "passed": bool(np.isclose(checked, expected, atol=file_context.run.atol, rtol=0)),
        "rule": rule,
        "validation_note": validation_note,
    }


def _rows_for_output_file(
    *,
    file_context: _L2FileContext,
    frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Validate one output table and return report rows."""
    year_col = str(file_context.run.year)
    sums = aggregate_share_by_group_keys(
        frame,
        year_col=year_col,
        group_cols=file_context.group_cols,
    )
    rows: list[dict[str, object]] = []
    for _, item in sums.iterrows():
        base_sum = scalar_float(item.get("sum_share"))
        fy_resolution = _resolve_l2_fy_add(
            item=item,
            file_context=file_context,
        )
        rows.append(
            _build_l2_row(
                file_context=file_context,
                item=item,
                base_sum=base_sum,
                fy_resolution=fy_resolution,
            )
        )
    return rows


def validate_l2_outputs(context: L2ValidationContext) -> list[dict[str, object]]:
    """Validate L2 output files and return row wise report items."""
    rows: list[dict[str, object]] = []
    year_col = str(context.year)
    for bucket in context.buckets:
        bucket_dir = context.l2_root / bucket
        for path in list_output_files(bucket_dir, preferred_format=context.output_format):
            frame = with_validation_method_column(read_output(path))
            if year_col not in frame.columns:
                continue
            is_lcia = is_lcia_output(path)
            group_cols, axis = _resolve_l2_grouping_keys(
                df=frame,
                is_lcia=is_lcia,
                bucket=bucket,
                fu_code=context.fu_code,
            )
            file_context = _L2FileContext(
                run=context,
                bucket=bucket,
                path=path,
                is_lcia=is_lcia,
                lcia_method=parse_lcia_method(path) if is_lcia else None,
                group_cols=group_cols,
                axis=axis,
            )
            rows.extend(_rows_for_output_file(file_context=file_context, frame=frame))
    return rows
