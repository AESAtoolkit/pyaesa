"""L1 validation helpers for allocation checks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io_helpers import (
    VALIDATION_METHOD_COLUMN,
    aggregate_share_by_group_keys,
    clean_text,
    list_output_files,
    parse_optional_int_or_empty,
    read_output,
    scalar_float,
    with_validation_method_column,
)

_L1_STATIC_FIELDS: dict[str, object] = {
    "group_indices": "ungrouped",
    "bucket": "l1",
    "l2_country_axis": "",
    "l2_country_code": "",
    "fy_add_share_observed": 0.0,
    "rule": "L1 sum to 1",
}


@dataclass(frozen=True)
class L1ValidationContext:
    """Execution context for L1 validation checks."""

    share_dir: Path
    source: str
    fu_code: str
    year: int
    l1_mode: str
    output_format: str
    atol: float


def validate_l1_outputs(context: L1ValidationContext) -> list[dict[str, object]]:
    """Validate L1 output files and return row wise report items."""
    rows: list[dict[str, object]] = []
    year_col = str(context.year)
    target = 1.0
    for path in list_output_files(context.share_dir, preferred_format=context.output_format):
        frame = with_validation_method_column(read_output(path))
        if year_col not in frame.columns:
            continue
        group_cols = [
            col
            for col in (VALIDATION_METHOD_COLUMN, "impact", "reference_year")
            if col in frame.columns
        ]
        sums = aggregate_share_by_group_keys(
            frame,
            year_col=year_col,
            group_cols=group_cols,
        )
        for _, item in sums.iterrows():
            checked = scalar_float(item["sum_share"])
            abs_error = abs(checked - target)
            rows.append(
                {
                    **_L1_STATIC_FIELDS,
                    "source": context.source,
                    "fu_code": context.fu_code,
                    "year": int(context.year),
                    "l1_reg_aggreg": context.l1_mode,
                    "file": str(path),
                    "method": clean_text(item.get(VALIDATION_METHOD_COLUMN)),
                    "impact": clean_text(item.get("impact")),
                    "reference_year": parse_optional_int_or_empty(item.get("reference_year", "")),
                    "group_key": "|".join(f"{col}={item[col]}" for col in group_cols),
                    "sum_share_observed": checked,
                    "all_incl_fy_share_observed": checked,
                    "ratio_expected": target,
                    "abs_error": abs_error,
                    "atol_used": float(context.atol),
                    "passed": bool(np.isclose(checked, target, atol=context.atol, rtol=0)),
                }
            )
    return rows
