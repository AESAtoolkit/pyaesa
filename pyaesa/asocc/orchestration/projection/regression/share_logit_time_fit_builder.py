"""Internal builder for strict share logit time regression fit maps."""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from .regression_core_utils import (
    MIN_OLS_UNCERTAINTY_OBS,
    coerce_numeric_scalar,
    fit_simple_ols,
)
from .share_fit_window_log import write_share_fit_window_log_row
from .share_fit_containers import ShareFitMap as _ShareFitMap
from .share_fit_containers import ShareFitSpec as _ShareFitSpec
from .share_fit_containers import as_selected_set as _as_selected_set
from .share_fit_containers import container_category_map as _container_category_map
from .share_fit_containers import filter_container_map as _filter_container_map
from .share_fit_containers import slice_container as _slice_container
from .share_logit_time_fit_diagnostics import (
    ShareFitDiagnosticsContext,
    ShareFitDiagnosticsPayload,
    last_modeled_vector,
    persist_share_fit_diagnostics,
)
from .share_logit_time_fit_types import (
    ShareFitBuildConfig,
    _ContainerSelection,
    _FitPoint,
    _ShareCoef,
    _select_baseline,
)

logger = logging.getLogger(__name__)
_EMPTY_FIT_SPEC: _ShareFitSpec = {
    "emit": [],
    "baseline": None,
    "coefs": {},
    "structural_zero_categories": [],
    "last_vector": pd.Series(dtype=float),
    "all_fitted": False,
}


class _FitWindow(NamedTuple):
    valid_years: list[int]
    dropped_numerator_zero_years: list[int]
    dropped_baseline_zero_years: list[int]


class _FitInputs(NamedTuple):
    year_center: float
    x_centered: np.ndarray
    y_values: np.ndarray
    fit_points: list[_FitPoint]


class _ShareFitBuilder:  # pylint: disable=too few public methods
    def __init__(self, *, config: ShareFitBuildConfig, state) -> None:
        self._config = config
        self._state = state
        self._selected_set = _as_selected_set(config.selected_categories)
        self._fit_start = int(min(config.historical_years))
        self._fit_end = int(max(config.historical_years))
        self._diagnostics_context = ShareFitDiagnosticsContext(
            config=config,
            fit_start=self._fit_start,
            fit_end=self._fit_end,
            state=state,
        )

    def build(self) -> _ShareFitMap:
        """Build strict share regression fit payload for all containers."""
        template = self._config.share_by_year[int(self._config.historical_years[-1])]
        container_map = _container_category_map(
            template=template,
            container_levels=self._config.containers,
            category_level=self._config.category_level,
        )
        filtered_map = _filter_container_map(
            container_map=container_map,
            container_levels=self._config.containers,
            selected_containers=self._config.selected_containers,
        )
        fitted: _ShareFitMap = {}
        for container_key, categories_all in filtered_map.items():
            selection = self._build_selection(
                container_key=container_key, categories_all=categories_all
            )
            if selection is None:
                fitted[container_key] = _EMPTY_FIT_SPEC
                continue
            fitted[container_key] = self._fit_container(selection=selection)
        return fitted

    def _build_selection(
        self,
        *,
        container_key: tuple[object, ...],
        categories_all: list[object],
    ) -> _ContainerSelection | None:
        if self._selected_set is None:
            selected = list(categories_all)
        else:
            selected = [
                category for category in categories_all if str(category) in self._selected_set
            ]
        if not selected:
            return None
        full_selection = len(selected) == len(categories_all)
        modeled = list(selected) if full_selection else [*selected, "__REST__"]
        return _ContainerSelection(
            container_key=container_key,
            categories_all=list(categories_all),
            selected=selected,
            full_selection=full_selection,
            modeled=modeled,
        )

    def _fit_container(self, *, selection: _ContainerSelection) -> _ShareFitSpec:
        modeled_by_year = self._build_modeled_by_year(selection=selection)
        baseline = _select_baseline(
            modeled=selection.modeled,
            modeled_by_year=modeled_by_year,
            historical_years=self._config.historical_years,
            container_key=selection.container_key,
        )
        coefs: dict[object, _ShareCoef] = {}
        structural_zero_categories: list[object] = []
        for category in selection.modeled:
            if category == baseline:
                continue
            fit = self._fit_category(
                selection=selection,
                modeled_by_year=modeled_by_year,
                baseline=baseline,
                category=category,
            )
            if fit is None:
                structural_zero_categories.append(category)
                continue
            coefs[category] = fit
        return {
            "emit": selection.selected
            if not selection.full_selection
            else list(selection.categories_all),
            "baseline": baseline,
            "coefs": coefs,
            "structural_zero_categories": structural_zero_categories,
            "last_vector": last_modeled_vector(
                historical_years=self._config.historical_years,
                modeled_by_year=modeled_by_year,
            ),
            "all_fitted": True,
        }

    def _build_modeled_by_year(
        self,
        *,
        selection: _ContainerSelection,
    ) -> dict[int, pd.Series]:
        modeled_by_year: dict[int, pd.Series] = {}
        for year in self._config.historical_years:
            observed = (
                _slice_container(
                    series=self._config.share_by_year[int(year)],
                    container_levels=self._config.containers,
                    category_level=self._config.category_level,
                    container_key=selection.container_key,
                )
                .reindex(selection.categories_all)
                .fillna(0.0)
                .clip(lower=0.0)
            )
            total = float(observed.sum(min_count=1))
            if total > 0.0 and np.isfinite(total):
                observed = observed / total
            if selection.full_selection:
                modeled_by_year[int(year)] = observed.reindex(selection.modeled).fillna(0.0)
                continue
            selected_total = float(observed.reindex(selection.selected).sum(min_count=1))
            data = {
                **{
                    category: coerce_numeric_scalar(observed.get(category, 0.0))
                    for category in selection.selected
                },
                "__REST__": max(0.0, 1.0 - selected_total),
            }
            modeled_by_year[int(year)] = pd.Series(data, dtype=float)
        return modeled_by_year

    def _fit_category(
        self,
        *,
        selection: _ContainerSelection,
        modeled_by_year: dict[int, pd.Series],
        baseline: object,
        category: object,
    ) -> _ShareCoef | None:
        nonzero_years = [
            int(year)
            for year in self._config.historical_years
            if coerce_numeric_scalar(modeled_by_year[int(year)].get(category, 0.0)) > 0.0
        ]
        if len(nonzero_years) == 0:
            self._write_all_zero_category_log(
                selection=selection,
                baseline=baseline,
                category=category,
            )
            return None
        if len(nonzero_years) < MIN_OLS_UNCERTAINTY_OBS:
            raise ValueError(
                "Share regression requires at least three nonzero years for "
                f"category='{category}' in container='{selection.container_name}'. "
                f"Found nonzero_years={nonzero_years}."
            )
        fit_window = self._valid_fit_window(
            modeled_by_year=modeled_by_year,
            baseline=baseline,
            category=category,
        )
        if len(fit_window.valid_years) < MIN_OLS_UNCERTAINTY_OBS:
            raise ValueError(
                "Share regression has fewer than three valid years after zero "
                "filtering: "
                f"container='{selection.container_name}', category='{category}', "
                f"baseline='{baseline}', valid_years={fit_window.valid_years}, "
                "dropped_numerator_zero_years="
                f"{fit_window.dropped_numerator_zero_years}, "
                "dropped_baseline_zero_years="
                f"{fit_window.dropped_baseline_zero_years}."
            )
        self._write_subset_fit_window_log_if_needed(
            selection=selection,
            baseline=baseline,
            category=category,
            fit_window=fit_window,
        )
        fit_inputs = self._build_fit_inputs(
            modeled_by_year=modeled_by_year,
            category=category,
            baseline=baseline,
            valid_years=fit_window.valid_years,
        )
        intercept, slope, r_squared, p_value = fit_simple_ols(
            x=fit_inputs.x_centered,
            y=fit_inputs.y_values,
        )
        n_obs = len(fit_window.valid_years)
        domain_key = f"{selection.container_name}|{category}/{baseline}"
        persist_share_fit_diagnostics(
            context=self._diagnostics_context,
            payload=ShareFitDiagnosticsPayload(
                domain_key=domain_key,
                container_name=selection.container_name,
                category=category,
                baseline=baseline,
                n_obs=int(n_obs),
                intercept=float(intercept),
                slope=float(slope),
                r_squared=float(r_squared),
                p_value=float(p_value),
                year_center=float(fit_inputs.year_center),
                x_centered=fit_inputs.x_centered,
                y_values=fit_inputs.y_values,
                fit_points=fit_inputs.fit_points,
                valid_years=fit_window.valid_years,
            ),
        )
        return (
            float(intercept),
            float(slope),
            float(r_squared),
            float(p_value),
            int(n_obs),
            float(fit_inputs.year_center),
        )

    def _valid_fit_window(
        self,
        *,
        modeled_by_year: dict[int, pd.Series],
        baseline: object,
        category: object,
    ) -> _FitWindow:
        valid_years: list[int] = []
        dropped_numerator_zero_years: list[int] = []
        dropped_baseline_zero_years: list[int] = []
        for year in self._config.historical_years:
            values = modeled_by_year[int(year)]
            numer = coerce_numeric_scalar(values.get(category, np.nan))
            denom = coerce_numeric_scalar(values.get(baseline, np.nan))
            if not np.isfinite(numer) or numer <= 0.0:
                dropped_numerator_zero_years.append(int(year))
                continue
            if not np.isfinite(denom) or denom <= 0.0:
                dropped_baseline_zero_years.append(int(year))
                continue
            valid_years.append(int(year))
        return _FitWindow(
            valid_years=valid_years,
            dropped_numerator_zero_years=dropped_numerator_zero_years,
            dropped_baseline_zero_years=dropped_baseline_zero_years,
        )

    def _write_all_zero_category_log(
        self,
        *,
        selection: _ContainerSelection,
        baseline: object,
        category: object,
    ) -> None:
        write_share_fit_window_log_row(
            source=self._config.source,
            fu_code=self._config.fu_code,
            l2_method=self._config.l2_method,
            target_object=self._config.target_object,
            container_label=selection.container_name,
            category=category,
            baseline=baseline,
            fit_start_year=int(self._fit_start),
            fit_end_year=int(self._fit_end),
            years_used=[],
            dropped_numerator_zero_years=sorted(
                int(year) for year in self._config.historical_years
            ),
            dropped_baseline_zero_years=[],
            case="all_zero_category",
            state=self._state,
        )

    def _write_subset_fit_window_log_if_needed(
        self,
        *,
        selection: _ContainerSelection,
        baseline: object,
        category: object,
        fit_window: _FitWindow,
    ) -> None:
        expected_years = [int(year) for year in self._config.historical_years]
        if fit_window.valid_years == expected_years:
            return
        write_share_fit_window_log_row(
            source=self._config.source,
            fu_code=self._config.fu_code,
            l2_method=self._config.l2_method,
            target_object=self._config.target_object,
            container_label=selection.container_name,
            category=category,
            baseline=baseline,
            fit_start_year=int(self._fit_start),
            fit_end_year=int(self._fit_end),
            years_used=fit_window.valid_years,
            dropped_numerator_zero_years=fit_window.dropped_numerator_zero_years,
            dropped_baseline_zero_years=fit_window.dropped_baseline_zero_years,
            case="subset_fit_window",
            state=self._state,
        )
        logger.warning(
            "Share regression uses subset fit window: target=%s, "
            "container=%s, category=%s, baseline=%s, historical_years=%s, "
            "valid_years=%s",
            self._config.target_object,
            selection.container_name,
            category,
            baseline,
            self._config.historical_years,
            fit_window.valid_years,
        )

    def _build_fit_inputs(
        self,
        *,
        modeled_by_year: dict[int, pd.Series],
        category: object,
        baseline: object,
        valid_years: list[int],
    ) -> _FitInputs:
        year_center = float(np.mean(np.asarray(valid_years, dtype=float)))
        x_centered = np.asarray(
            [float(year) - year_center for year in valid_years],
            dtype=float,
        )
        y_values: list[float] = []
        fit_points: list[_FitPoint] = []
        for year in valid_years:
            values = modeled_by_year[int(year)]
            numer = coerce_numeric_scalar(values.get(category, np.nan))
            denom = coerce_numeric_scalar(values.get(baseline, np.nan))
            ratio = float(numer / denom)
            log_ratio = float(np.log(ratio))
            y_values.append(log_ratio)
            fit_points.append(
                (int(year), float(year) - year_center, log_ratio, ratio, numer, denom)
            )
        return _FitInputs(
            year_center=year_center,
            x_centered=x_centered,
            y_values=np.asarray(y_values, dtype=float),
            fit_points=fit_points,
        )


def build_share_fit_map_impl(*, config: ShareFitBuildConfig, state) -> _ShareFitMap:
    """Build strict share regression fit payload for all selected containers."""
    return _ShareFitBuilder(config=config, state=state).build()
