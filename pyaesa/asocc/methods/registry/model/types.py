"""Core model types for the allocation method registry."""

import re
from typing import NamedTuple, Optional


def normalize_fu_code(raw: object) -> str:
    """Normalize FU code to canonical dot format."""
    if raw is None:
        raise ValueError("fu_code is required")
    value = str(raw).strip().replace("_", ".")
    while ".." in value:
        value = value.replace("..", ".")
    if re.fullmatch(r"L1\.[a-z]", value):
        return value
    if re.fullmatch(r"L2\.[a-z]\.[a-z](\.[a-z])?", value):
        return value
    raise ValueError(f"Invalid fu_code '{raw}'.")


class MethodSpec(NamedTuple):
    """Specification of one supported allocation method variant."""

    name: str
    level: str
    fu_code: Optional[str]
    l1_weighting: bool
    needs_lcia: bool
    needs_pop: bool
    needs_gdp: bool
    needs_utility: bool
    needs_rp: bool
    indices: tuple[str, ...]
    l1_kind: Optional[str]
    l2_weight_axis: Optional[str]
    expand_ar_years: bool
    family: str
