"""Types and shared utilities for strict share logit time fit mapping."""

from typing import NamedTuple

import numpy as np
import pandas as pd

from .regression_core_utils import MIN_OLS_UNCERTAINTY_OBS
from .share_fit_containers import container_label as _container_label

_ShareCoef = tuple[float, float, float, float, int, float]
_FitPoint = tuple[int, float, float, float, float, float]


class ShareFitBuildConfig(NamedTuple):
    """Inputs required to build strict share regression fit payloads."""

    source: str
    fu_code: str
    l2_method: str
    target_object: str
    historical_years: list[int]
    share_by_year: dict[int, pd.Series]
    future_years: list[int]
    containers: list[str]
    category_level: str
    selected_categories: list[str] | None
    selected_containers: dict[str, list[str] | None] | None


class _ContainerSelection(NamedTuple):
    """Selection payload for one container domain."""

    container_key: tuple[object, ...]
    categories_all: list[object]
    selected: list[object]
    full_selection: bool
    modeled: list[object]

    @property
    def container_name(self) -> str:
        """Render one stable diagnostics label for this container key."""
        return _container_label(self.container_key)


def _positive_coverage_value(value: object) -> int:
    """Return one positivity flag for one fitted share value."""
    scalar = value
    if scalar is None:
        return 0
    if scalar is pd.NA:
        return 0
    if isinstance(scalar, float) and pd.isna(scalar):
        return 0
    if isinstance(scalar, bool):
        return int(float(scalar) > 0.0)
    if isinstance(scalar, int):
        return int(float(scalar) > 0.0)
    if isinstance(scalar, float):
        return int(scalar > 0.0)
    if isinstance(scalar, str):
        return int(float(scalar.strip()) > 0.0)
    numeric_array = pd.Series([scalar], dtype="object").astype("float64").to_numpy(dtype=np.float64)
    numeric_scalar = float(numeric_array[0])
    if bool(pd.isna(numeric_scalar)):
        return 0
    return int(numeric_scalar > 0.0)


def _select_baseline(
    *,
    modeled: list[object],
    modeled_by_year: dict[int, pd.Series],
    historical_years: list[int],
    container_key: tuple[object, ...],
) -> object:
    """Select baseline category with maximal positivity coverage."""
    pos_count: dict[object, int] = {}
    for candidate in modeled:
        pos_count[candidate] = sum(
            _positive_coverage_value(modeled_by_year[int(year)].get(candidate, 0.0))
            for year in historical_years
        )
    best = max(pos_count.values(), default=0)
    if best < MIN_OLS_UNCERTAINTY_OBS:
        counts = ", ".join(
            f"{str(category)}={count}" for category, count in sorted(pos_count.items(), key=str)
        )
        raise ValueError(
            "Share regression baseline selection failed: "
            f"container='{_container_label(container_key)}', "
            f"historical_years={historical_years}, positivity_counts={{{counts}}}. "
            "At least one baseline candidate must have positivity in "
            f">= {MIN_OLS_UNCERTAINTY_OBS} years."
        )
    winners = [category for category, count in pos_count.items() if count == best]
    return sorted(winners, key=str)[0]
