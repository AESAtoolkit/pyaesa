"""Shared report row metric field helpers for validation CSV builders."""


def share_metric_fields(
    *,
    base_sum: float,
    fy_add: float,
    checked: float,
    expected: float,
    abs_error: float,
) -> dict[str, float]:
    """Return common numeric metric fields used in validation report rows."""
    return {
        "sum_share_observed": base_sum,
        "fy_add_share_observed": fy_add,
        "all_incl_fy_share_observed": checked,
        "ratio_expected": expected,
        "abs_error": abs_error,
    }
