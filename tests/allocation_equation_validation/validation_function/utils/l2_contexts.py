"""Shared execution context types for L2 and L2*b validation utilities."""

from pathlib import Path
from typing import NamedTuple, Sequence


class L2RunContext(NamedTuple):
    """Common runtime context for one L2-family validation pass."""

    l2_root: Path
    validation_project_name_root: str
    source: str
    fu_code: str
    year: int
    l1_mode: str
    buckets: Sequence[str]
    output_format: str
    atol: float
    matrix_version: str | None = None
    agg_reg: bool | None = None
    group_indices: bool = False
