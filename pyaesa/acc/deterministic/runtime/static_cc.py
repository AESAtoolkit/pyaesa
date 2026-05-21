"""Deterministic static-CC matching for aCC branches."""

import numpy as np
import pandas as pd


def _optional_float(value: object) -> float | None:
    """Return one optional float scalar from a static CC cell."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)) and bool(np.isnan(value)):
        return None
    parsed = pd.to_numeric(pd.Series([value], copy=False), errors="raise")
    array = np.asarray(pd.Series(parsed, copy=False), dtype=np.float64)
    return float(array[0])


def match_cc_for_share(
    share_path,
    cc_df: pd.DataFrame,
    *,
    forced_impacts: list[str] | tuple[str, ...] | None = None,
) -> list[tuple[str, float, float | None, str]]:
    """Match static CC rows to one deterministic share file."""
    if forced_impacts:
        impact_map = {
            str(impact): (
                float(min_cc),
                _optional_float(max_cc),
                str(unit),
            )
            for impact, unit, min_cc, max_cc in zip(
                cc_df["impact"],
                cc_df["impact_unit"],
                cc_df["min_cc"],
                cc_df["max_cc"] if "max_cc" in cc_df.columns else [None] * len(cc_df),
            )
        }
        requested = [str(impact) for impact in forced_impacts]
        return [(impact, *impact_map[impact]) for impact in requested]
    stem = (share_path.stem if hasattr(share_path, "stem") else str(share_path)).lower()
    matched: list[tuple[str, float, float | None, str]] = []
    for impact, unit, min_cc, max_cc in zip(
        cc_df["impact"],
        cc_df["impact_unit"],
        cc_df["min_cc"],
        cc_df["max_cc"] if "max_cc" in cc_df.columns else [None] * len(cc_df),
    ):
        imp_str = str(impact)
        imp_lower = imp_str.lower()
        if imp_lower in stem or imp_lower.replace("_", " ") in stem:
            matched.append((imp_str, float(min_cc), _optional_float(max_cc), str(unit)))
    if matched:
        return matched
    return [
        (str(imp), float(mn), _optional_float(mx), str(unit))
        for imp, unit, mn, mx in zip(
            cc_df["impact"],
            cc_df["impact_unit"],
            cc_df["min_cc"],
            cc_df["max_cc"] if "max_cc" in cc_df.columns else [None] * len(cc_df),
        )
    ]
