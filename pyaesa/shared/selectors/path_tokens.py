"""Canonical selector tokens for output paths and figure file names."""

from collections.abc import Mapping, Sequence
import re
from typing import Any

import pandas as pd

from pyaesa.shared.tabular.scalars import canonical_scalar_text, is_display_missing

SELECTOR_AXIS_ALIASES: dict[str, str] = {
    "r_p": "rp",
    "s_p": "sp",
    "r_c": "rc",
    "r_f": "rf",
}

DEFAULT_SELECTOR_VALUE_TOKEN_LEN = 16
DEFAULT_SELECTOR_SEGMENT_LEN = 120

_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}
_INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_NON_TOKEN_CHARS_RE = re.compile(r"[^A-Za-z0-9._()-]+")
_RUN_UNDERSCORES_RE = re.compile(r"_+")


def safe_path_token(value: Any, *, max_len: int | None = None) -> str:
    """Return one deterministic path-safe token for a public selector value."""
    token = canonical_scalar_text(value)
    token = _INVALID_PATH_CHARS_RE.sub("_", token)
    token = re.sub(r"\s+", "_", token)
    token = _NON_TOKEN_CHARS_RE.sub("_", token)
    token = _RUN_UNDERSCORES_RE.sub("_", token)
    token = token.strip(" ._-") or "item"
    if max_len is not None:
        token = token[: int(max_len)].rstrip(" ._-") or "item"
    if token.upper() in _WINDOWS_RESERVED_NAMES:
        token = f"_{token}"
    return token


def selector_axis_token(axis: str) -> str:
    """Return the short token used for one functional-unit selector axis."""
    text = canonical_scalar_text(axis)
    return SELECTOR_AXIS_ALIASES.get(text, safe_path_token(text, max_len=16))


def selector_value_text(value: Any) -> str:
    """Return one canonical selector value text, using ``all`` for missing selectors."""
    return "all" if is_display_missing(value) else canonical_scalar_text(value)


def selector_value_token(
    value: Any,
    *,
    max_len: int = DEFAULT_SELECTOR_VALUE_TOKEN_LEN,
) -> str:
    """Return one bounded selector value token."""
    return safe_path_token(selector_value_text(value), max_len=max_len)


def deduplicated_selector_value_tokens(
    values: Sequence[Any],
    *,
    max_len: int = DEFAULT_SELECTOR_VALUE_TOKEN_LEN,
) -> dict[str, str]:
    """Return canonical selector value texts mapped to collision-safe short tokens."""
    value_texts = sorted({selector_value_text(value) for value in values})
    base_tokens = [selector_value_token(value_text, max_len=max_len) for value_text in value_texts]
    counts: dict[str, int] = {}
    tokens: dict[str, str] = {}
    for value_text, base_token in zip(value_texts, base_tokens, strict=True):
        count = counts.get(base_token, 0) + 1
        counts[base_token] = count
        tokens[value_text] = base_token if count == 1 else f"{base_token}_{count}"
    return tokens


def selector_axis_values_token(
    values: Sequence[Any],
    *,
    max_value_len: int = DEFAULT_SELECTOR_VALUE_TOKEN_LEN,
    max_segment_len: int = DEFAULT_SELECTOR_SEGMENT_LEN,
) -> str:
    """Return one bounded token for all requested values of one selector axis."""
    value_tokens = list(deduplicated_selector_value_tokens(values, max_len=max_value_len).values())
    if not value_tokens:
        return "all"
    joined = "+".join(value_tokens)
    if len(joined) <= max_segment_len:
        return joined
    fallback = f"n{len(value_tokens)}_{value_tokens[0]}"
    return safe_path_token(fallback, max_len=max_segment_len)


def selector_scope_token_from_values(
    selector_values: Mapping[str, Any],
    *,
    selector_columns: Sequence[str],
) -> str:
    """Return one figure filename token for a concrete selector value combination."""
    parts: list[str] = []
    for column in selector_columns:
        if column not in selector_values:
            continue
        axis = selector_axis_token(str(column))
        value = selector_value_token(selector_values[column])
        parts.append(f"{axis}_{value}")
    return "__".join(parts)


def selector_scope_token_from_frame(
    *,
    group_frame: pd.DataFrame,
    selector_columns: Sequence[str],
    reference_frame: pd.DataFrame | None = None,
) -> str:
    """Return one collision-safe selector token for one grouped figure frame."""
    present = [column for column in selector_columns if column in group_frame.columns]
    if not present:
        return "all_selectors"
    reference = group_frame if reference_frame is None else reference_frame
    values_by_column: dict[str, Any] = {}
    token_maps: dict[str, dict[str, str]] = {}
    for column in present:
        reference_values = reference[column].tolist() if column in reference.columns else []
        token_maps[column] = deduplicated_selector_value_tokens(reference_values)
        visible_values = sorted(
            {selector_value_text(value) for value in group_frame[column].tolist()}
        )
        values_by_column[column] = visible_values[0] if len(visible_values) == 1 else visible_values
    parts: list[str] = []
    for column in present:
        axis = selector_axis_token(str(column))
        value = values_by_column[column]
        if isinstance(value, list):
            value_token = selector_axis_values_token(value)
        else:
            value_token = token_maps[column][selector_value_text(value)]
        parts.append(f"{axis}_{value_token}")
    return "__".join(parts) if parts else "all_selectors"


def selector_scope_request_axes_token(
    axes: Sequence[tuple[str, Sequence[str] | None]],
    *,
    empty_token: str = "all_selectors",
) -> str:
    """Return one filename token for an explicit selector scope request."""
    if not axes:
        return empty_token
    parts: list[str] = []
    for column, values in axes:
        axis = selector_axis_token(str(column))
        if not values:
            parts.append(f"{axis}_all")
            continue
        values_token = selector_axis_values_token(tuple(values))
        parts.append(f"{axis}_{values_token}")
    return "__".join(parts) if parts else empty_token


def build_selector_filter_segment(
    *,
    key: str,
    values: Sequence[Any],
    max_value_len: int = DEFAULT_SELECTOR_VALUE_TOKEN_LEN,
    max_segment_len: int = DEFAULT_SELECTOR_SEGMENT_LEN,
) -> str:
    """Build one bounded output path segment for one index filter key."""
    value_texts = sorted(
        {
            canonical_scalar_text(value)
            for value in values
            if canonical_scalar_text(value) != "missing"
        }
    )
    if not value_texts:
        return ""
    value_token = selector_axis_values_token(
        value_texts,
        max_value_len=max_value_len,
        max_segment_len=max_segment_len - len(str(key)) - 1,
    )
    segment = f"{key}-{value_token}"
    if len(segment) <= max_segment_len:
        return segment
    return safe_path_token(segment, max_len=max_segment_len)
