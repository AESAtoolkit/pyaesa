"""Shared figure label and method identity helpers."""

from collections.abc import Sequence
from typing import Any

import pandas as pd

from pyaesa.shared.tabular.scalars import display_scalar, is_display_missing

_VISIBLE_COLUMN_ALIASES = {
    "reference_year": "ref_year",
}
_BARE_VALUE_COLUMNS = {"combined_group_label", "l1_l2_method", "l2_method"}


def display_pair(
    column: str,
    value: Any,
    *,
    display_aliases: dict[str, str] | None = None,
) -> str | None:
    """Return one ``column=value`` string for visible figure labels."""
    text = display_scalar(value)
    if text is None:
        return None
    if str(column) in _BARE_VALUE_COLUMNS:
        return text
    aliases = display_aliases or _VISIBLE_COLUMN_ALIASES
    visible = aliases.get(str(column), str(column))
    return f"{visible}={text}"


def simplified_method_identity_columns(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> list[str]:
    """Return grouping columns with redundant method metadata removed."""
    out = [str(column) for column in columns]
    if _has_non_missing_column(frame, "l1_l2_method"):
        ignored = {"l2_method", "l1_method"}
        return [column for column in out if column not in ignored]
    return out


def visible_method_identity(frame: pd.DataFrame) -> str | None:
    """Return one visible method identity token for a figure slice."""
    for column in ("l1_l2_method", "l2_method"):
        if column not in frame.columns:
            continue
        values = sorted(
            {
                text
                for value in frame[column].dropna().tolist()
                for text in [display_scalar(value)]
                if text is not None
            }
        )
        if len(values) == 1 and values[0]:
            return values[0]
    return None


def method_scope_slices(frame: pd.DataFrame) -> list[tuple[str | None, pd.DataFrame]]:
    """Return frame slices grouped by allocation method identity only."""
    method_columns = [
        column for column in ("l1_l2_method", "l2_method", "l1_method") if column in frame
    ]
    scope_columns = simplified_method_identity_columns(
        frame,
        columns=method_columns,
    )
    if not scope_columns:
        return [(None, frame.copy())]
    grouped = frame.groupby(scope_columns, dropna=False, sort=True)
    return [(visible_method_identity(subset), subset.copy()) for _key, subset in grouped]


def resolve_figure_display_label(
    *,
    frame: pd.DataFrame,
    user_facing_override_label: str | None = None,
) -> str | None:
    """Return the final visible method label for one figure scope.

    Args:
        frame: Figure scoped frame carrying canonical method identity columns.
        user_facing_override_label: Optional final user-facing display label
            injected by the rendering layer when the figure intentionally wants
            a noncanonical visible label.

    Returns:
        The explicit display label when provided, otherwise the visible method
        identity derived from canonical method columns.
    """
    if user_facing_override_label is not None:
        text = str(user_facing_override_label).strip()
        if text:
            return text
    return visible_method_identity(frame)


def _has_non_missing_column(frame: pd.DataFrame, column: str) -> bool:
    """Return whether one frame column exists and carries any visible value."""
    if column not in frame.columns:
        return False
    return any(not is_display_missing(value) for value in frame[column].tolist())
