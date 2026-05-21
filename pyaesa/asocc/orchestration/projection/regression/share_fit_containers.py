"""Helper functions and types for share regression kernels."""

from typing import TypedDict, cast

import pandas as pd

_ShareCoef = tuple[float, float, float, float, int, float]


class ShareFitSpec(TypedDict):
    """Cached fit payload for one container domain."""

    emit: list[object]
    baseline: object | None
    coefs: dict[object, _ShareCoef]
    structural_zero_categories: list[object]
    last_vector: pd.Series
    all_fitted: bool


ShareFitMap = dict[tuple[object, ...], ShareFitSpec]


def _numeric_series(series: pd.Series) -> pd.Series:
    """Return one numeric Series with stable pandas metadata."""
    numeric = pd.to_numeric(pd.Series(series, copy=False), errors="raise")
    return pd.Series(numeric, index=series.index, name=series.name, copy=False)


def as_level_list(levels: str | list[str]) -> list[str]:
    """Normalize one or many level names to list form."""
    if isinstance(levels, str):
        return [levels]
    return [str(level) for level in levels]


def slice_container(
    *,
    series: pd.Series,
    container_levels: list[str],
    category_level: str,
    container_key: tuple[object, ...],
) -> pd.Series:
    """Return one container slice indexed only by category labels."""
    if not isinstance(series.index, pd.MultiIndex):
        return _numeric_series(series)
    mask = pd.Series(True, index=series.index)
    for level, value in zip(container_levels, container_key):
        mask &= series.index.get_level_values(level) == value
    sliced = _numeric_series(pd.Series(series.loc[mask], copy=False))
    categories = sliced.index.get_level_values(category_level)
    out = pd.Series(sliced.to_numpy(dtype=float), index=categories)
    return pd.Series(out.groupby(level=0).sum(min_count=1), copy=False)


def container_category_map(
    *,
    template: pd.Series,
    container_levels: list[str],
    category_level: str,
) -> dict[tuple[object, ...], list[object]]:
    """Return deterministic category list per container from template index."""
    if not isinstance(template.index, pd.MultiIndex):
        categories: list[object] = []
        seen: set[str] = set()
        for value in template.index:
            marker = str(value)
            if marker in seen:
                continue
            seen.add(marker)
            categories.append(value)
        return {tuple(): categories}
    container_pos = [template.index.names.index(level) for level in container_levels]
    category_pos = template.index.names.index(category_level)
    out: dict[tuple[object, ...], list[object]] = {}
    seen_by_container: dict[tuple[object, ...], set[str]] = {}
    for idx in template.index:
        container = tuple(idx[pos] for pos in container_pos)
        category = idx[category_pos]
        seen = seen_by_container.setdefault(container, set())
        marker = str(category)
        if marker in seen:
            continue
        seen.add(marker)
        out.setdefault(container, []).append(category)
    return out


def container_label(container_key: tuple[object, ...]) -> str:
    """Render container tuple into stable diagnostics label."""
    if not container_key:
        return "global"
    return "|".join(str(value) for value in container_key)


def as_selected_set(values: list[str] | None) -> set[str] | None:
    """Normalize one optional selected values list into a string set."""
    if not values:
        return None
    return {str(value) for value in values}


def _container_matches_filters(
    *,
    container_key: tuple[object, ...],
    container_levels: list[str],
    selected_by_level: dict[str, set[str] | None],
) -> bool:
    """Return whether one container key is inside selected filter domains."""
    for level, value in zip(container_levels, container_key):
        selected = selected_by_level.get(level)
        if selected is None:
            continue
        if str(value) not in selected:
            return False
    return True


def filter_container_map(
    *,
    container_map: dict[tuple[object, ...], list[object]],
    container_levels: list[str],
    selected_containers: dict[str, list[str] | None] | None,
) -> dict[tuple[object, ...], list[object]]:
    """Restrict container domains to selected index filters."""
    if not selected_containers:
        return container_map
    selected_by_level = {
        level: as_selected_set(selected_containers.get(level)) for level in container_levels
    }
    if all(selected is None for selected in selected_by_level.values()):
        return container_map
    return {
        key: categories
        for key, categories in container_map.items()
        if _container_matches_filters(
            container_key=key,
            container_levels=container_levels,
            selected_by_level=selected_by_level,
        )
    }


def selected_signature(values: list[str] | None) -> tuple[str, ...]:
    """Return deterministic cache key signature for one selected values list."""
    if not values:
        return ("__ALL__",)
    return tuple(sorted(str(value) for value in values))


def container_signature(
    *,
    container_levels: list[str],
    selected_containers: dict[str, list[str] | None] | None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return deterministic cache key signature for container filters."""
    if not selected_containers:
        return tuple()
    return tuple(
        (
            str(level),
            selected_signature(selected_containers.get(level)),
        )
        for level in container_levels
    )


def share_fit_map_or_none(value: object) -> ShareFitMap | None:
    """Return cached share fit map when runtime structure is valid."""
    if not isinstance(value, dict):
        return None
    for container_key, spec in value.items():
        if not isinstance(container_key, tuple):
            return None
        if not isinstance(spec, dict):
            return None
        if not {
            "emit",
            "baseline",
            "coefs",
            "structural_zero_categories",
            "last_vector",
            "all_fitted",
        }.issubset(spec):
            return None
        if not isinstance(spec["emit"], list):
            return None
        if not isinstance(spec["coefs"], dict):
            return None
        if not isinstance(spec["structural_zero_categories"], list):
            return None
        if not isinstance(spec["last_vector"], pd.Series):
            return None
        if not isinstance(spec["all_fitted"], bool):
            return None
    return cast(ShareFitMap, value)
