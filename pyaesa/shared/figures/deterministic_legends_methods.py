"""Deterministic legend classification plus shared grouped-legend compatibility."""

import re

from pyaesa.asocc.runtime.methods.labels import (
    parse_raw_asocc_method_label,
)
from pyaesa.shared.tabular.scalars import display_scalar

_METHOD_TOKEN_RE = re.compile(r"(UT\([^)]*\)|AR\([^)]*\)|EG\([^)]*\)|PR(?:-HR)?\([^)]*\))")
_NO_CLASSIFICATION = ""

_PRODUCTION_TWO_STEP = "Production-anchored two steps"
_CONSUMPTION_TWO_STEP = "Consumption-anchored two steps"
_GENERIC_TWO_STEP = "Two-step"
_ONE_STEP = "One-step"

_ORDER = [
    _CONSUMPTION_TWO_STEP,
    _PRODUCTION_TWO_STEP,
    _GENERIC_TWO_STEP,
    _ONE_STEP,
]


def legend_group_from_row(row) -> str:
    """Return one grouped deterministic legend title from row metadata."""
    if _is_l1_level(row):
        return _NO_CLASSIFICATION
    if _is_two_step_row(row):
        if not _is_l2_b_fu(row):
            return _GENERIC_TWO_STEP
        return _l2b_group_from_row(row=row)
    return _ONE_STEP


def _l2b_group_from_row(*, row) -> str:
    """Classify one L2*b two-step row by its production or consumption path."""
    l1_path = _path_from_label(_value(row, "l1_method"))
    l2_path = _path_from_label(_value(row, "l2_method"))
    combined_path = _path_from_tokens(_row_method_tokens(row))
    resolved = _merge_paths(paths=(l1_path, l2_path, combined_path))
    if resolved == "consumption":
        return _CONSUMPTION_TWO_STEP
    if resolved == "production":
        return _PRODUCTION_TWO_STEP
    return _GENERIC_TWO_STEP


def _is_two_step_row(row) -> bool:
    """Return whether one deterministic row represents a two-step method."""
    paired = _value(row, "l1_method")
    if paired is not None:
        return True
    l2_method = _value(row, "l2_method")
    l1_l2_method = _value(row, "l1_l2_method")
    if l2_method is not None and l1_l2_method is not None and l2_method != l1_l2_method:
        return True
    return len(_row_method_tokens(row)) >= 2


def _row_method_tokens(row) -> tuple[str, ...]:
    """Return visible scientific method tokens from one deterministic row."""
    tokens: list[str] = []
    for column in ("l1_l2_method", "l2_method", "l1_method"):
        value = _value(row, column)
        if value is None:
            continue
        tokens.extend(_METHOD_TOKEN_RE.findall(value))
    return tuple(token for token in dict.fromkeys(tokens) if token)


def _path_from_tokens(method_tokens: tuple[str, ...]) -> str | None:
    """Return one resolved path from a set of scientific method tokens."""
    return _merge_paths(paths=tuple(_path_from_label(token) for token in method_tokens))


def _path_from_label(value: str | None) -> str | None:
    """Return one resolved path from one scientific method label when possible."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        _sharing_principle, _subprinciple, enacting_metric = parse_raw_asocc_method_label(text)
    except ValueError:
        return None
    metric = str(enacting_metric).strip()
    if metric in {
        "CBA_FD",
        "CBA_TD",
        "FDa",
        "TDa",
        "E_CBA_FD",
        "E_CBA_TD",
        "Ecap_CBA_FD",
        "Ecap_CBA_TD",
        "cum_CBA_FD",
        "cum_CBA_TD",
    }:
        return "consumption"
    if metric in {
        "PBA",
        "GVAa",
        "E_PBA",
        "Ecap_PBA",
        "cum_PBA",
    }:
        return "production"
    return None


def _merge_paths(*, paths: tuple[str | None, ...]) -> str | None:
    """Return one consistent path from recognized production or consumption hints."""
    resolved = sorted({path for path in paths if path is not None})
    if not resolved:
        return None
    return resolved[0]


def _value(row, column: str) -> str | None:
    if column not in row.index:
        return None
    return display_scalar(row[column])


def _is_l1_level(row) -> bool:
    """Return whether one deterministic row belongs to the L1 figure scope."""
    level = _value(row, "level")
    if level is not None:
        return level == "level_1"
    fu_code = _value(row, "fu_code")
    return fu_code is not None and fu_code.startswith("L1.")


def _is_l2_b_fu(row) -> bool:
    fu_code = _value(row, "fu_code")
    return fu_code is not None and fu_code.startswith("L2.") and fu_code.endswith(".b")


def _is_known_method_token(value: str) -> bool:
    return value.startswith(("EG(", "PR(", "PR-HR(", "UT(", "AR("))
