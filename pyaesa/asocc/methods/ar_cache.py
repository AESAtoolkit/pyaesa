"""Shared cache and reference payload ownership for AR methods."""

import pandas as pd

from ..data.reference_payloads import load_ar_l2_reference_lcia_payload
from .equations.ar_e import compute_ar_e_l2


def _project_cached_baseline_for_year(
    *,
    cache: dict[tuple, pd.DataFrame],
    cache_key: tuple,
    year: int,
    compute_baseline,
    force_recompute_at_ref: bool = False,
) -> pd.DataFrame:
    """Shared AR cache flow for `year == ref_year` / `year > ref_year`."""
    # AR projection policy: compute once at reference year, reuse baseline shape,
    # and only rewrite the studied year column label.
    cached = cache.get(cache_key)
    if force_recompute_at_ref or cached is None:
        cached = compute_baseline()
        cache[cache_key] = cached
    return pd.DataFrame(
        cached.to_numpy(copy=False)[:, :1],
        index=cached.index,
        columns=pd.Index([int(year)]),
    )


def _ensure_ar_l2_cached(
    *,
    context,
    state,
    ssp_scenario: str | None,
    cache_key: tuple,
    l2_method: str,
    ref_year: int,
    lcia_key: str,
    l1_weights,
    pre_weighting: bool,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Ensure AR baseline for `ref_year` is cached and return it."""
    cache = state.ar_l2_cache_by_ssp_scenario[ssp_scenario]
    cached = cache.get(cache_key)
    if isinstance(cached, pd.DataFrame) and not force_recompute:
        return cached

    lcia_ref = load_ar_l2_reference_lcia_payload(
        context=context,
        state=state,
        ref_year=ref_year,
        lcia_key=lcia_key,
    )
    baseline = compute_ar_e_l2(
        l2_method=l2_method,
        fu_code=context.fu_code,
        l1_weights=l1_weights,
        lcia=lcia_ref,
        reference_year=ref_year,
        pre_weighting=pre_weighting,
    )
    cache[cache_key] = baseline
    return baseline
