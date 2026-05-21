"""Canonical AR6 CC figure title ownership."""

from collections.abc import Iterable

from pyaesa.shared.figures.dynamic_ar6 import category_scope_label


def ar6_cc_title(*, variable_name: str, ssp_scenario: str, categories: Iterable[str]) -> str:
    """Return the canonical AR6 CC figure title."""
    title = f"AR6 pathways | {variable_name} | {ssp_scenario}"
    category_scope = category_scope_label(categories)
    if not category_scope:
        return title
    return f"{title} | {category_scope}"
