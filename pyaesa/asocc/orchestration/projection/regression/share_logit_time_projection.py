"""Share regression kernels for post historical UT projection."""

import numpy as np
import pandas as pd

from .regression_core_utils import emit_regression_start_notice, fit_cache_key
from .share_fit_containers import (
    as_level_list as _as_level_list,
)
from .share_fit_containers import (
    container_signature as _container_signature,
)
from .share_fit_containers import (
    selected_signature as _selected_signature,
)
from .share_fit_containers import (
    share_fit_map_or_none as _share_fit_map_or_none,
)
from .share_logit_time_fit_builder import build_share_fit_map_impl
from .share_logit_time_fit_types import ShareFitBuildConfig

_TOL = 1.0e-12


def project_share_from_time_logit(
    *,
    source: str,
    fu_code: str,
    l2_method: str,
    target_object: str,
    historical_years: list[int],
    share_by_year: dict[int, pd.Series],
    target_year: int,
    future_years: list[int],
    container_levels: str | list[str],
    category_level: str,
    selected_categories: list[str] | None,
    selected_containers: dict[str, list[str] | None] | None,
    state,
) -> pd.Series:
    """Project shares with deterministic binary/multinomial log ratio models."""
    containers = _as_level_list(container_levels)
    selected_signature = _selected_signature(selected_categories)
    container_signature = _container_signature(
        container_levels=containers,
        selected_containers=selected_containers,
    )
    cache_key = fit_cache_key(
        source=source,
        fu_code=fu_code,
        l2_method=l2_method,
        model_type="log_ratio_time",
        target_object=(
            f"{target_object}|{tuple(containers)}|{category_level}|"
            f"{selected_signature}|{container_signature}"
        ),
        historical_years=historical_years,
    )
    # Fits are cached by method/target/domain signature and reused across years.
    cached = _share_fit_map_or_none(state.regression_fit_cache.get(cache_key))
    if cached is None:
        emit_regression_start_notice(
            source=source,
            fu_code=fu_code,
            l2_method=l2_method,
            model_type="log_ratio_time",
            target_object=target_object,
            state=state,
        )
        fitted = build_share_fit_map_impl(
            config=ShareFitBuildConfig(
                source=source,
                fu_code=fu_code,
                l2_method=l2_method,
                target_object=target_object,
                historical_years=historical_years,
                share_by_year=share_by_year,
                future_years=future_years,
                containers=containers,
                category_level=category_level,
                selected_categories=selected_categories,
                selected_containers=selected_containers,
            ),
            state=state,
        )
        state.regression_fit_cache[cache_key] = fitted
        cached = fitted

    predicted: dict[object, float] = {}
    for container_key, spec in cached.items():
        emit = spec["emit"]
        if not emit:
            continue
        baseline = spec["baseline"]
        if baseline is None:
            continue
        coefs = spec["coefs"]
        structural_zeros = set(spec["structural_zero_categories"])
        logits: dict[object, float] = {}
        for category, (intercept, slope, _r2, _p, _n_obs, year_center) in coefs.items():
            x0 = float(target_year) - float(year_center)
            logits[category] = float(intercept + slope * x0)

        values_dict: dict[object, float] = {}
        if logits:
            # Stable softmax shift: exponentiate logits - max(logits) to avoid
            # overflow while preserving exact normalized shares.
            m = max(logits.values())
            exp_terms = {
                category: float(np.exp(value - m))
                for category, value in sorted(logits.items(), key=lambda item: str(item[0]))
            }
            baseline_term = float(np.exp(-m))
            denom = baseline_term + float(sum(exp_terms.values()))
            values_dict[baseline] = baseline_term / denom
            for category, exp_value in exp_terms.items():
                values_dict[category] = exp_value / denom
        else:
            values_dict[baseline] = 1.0
        for category in structural_zeros:
            values_dict[category] = 0.0

        for category in emit:
            index_key = (*container_key, category) if container_key else category
            predicted[index_key] = float(values_dict.get(category, 0.0))
    out = pd.Series(predicted, dtype=float)
    if isinstance(out.index, pd.MultiIndex):
        out.index = out.index.set_names([*containers, category_level])
    else:
        out.index.name = category_level
    return out
