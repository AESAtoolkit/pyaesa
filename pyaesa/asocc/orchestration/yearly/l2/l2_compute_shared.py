"""Shared functions for L2 compute orchestration."""

from ....data.source_schema import default_historical_cutoff_for_source
from ....methods.registry.registry import REGISTRY
from .l2_slicing import _slice_lcia_payload_for_compute
from .l2_types import _L2RunContext, _is_ar_l1, _is_ar_l2


def _filters_cache_key(*, run: _L2RunContext) -> tuple:
    """Return deterministic filter signature for sliced payload caches."""
    out: list[tuple[str, tuple[str, ...] | None]] = []
    for axis in ("r_p", "s_p", "r_c", "r_f", "r_u"):
        values = run.context.filters.get(axis)
        if not values:
            out.append((axis, None))
            continue
        out.append((axis, tuple(sorted(str(v) for v in values))))
    return tuple(out)


def _lcia_items_for_method(
    *,
    run: _L2RunContext,
    l2_method: str,
    l1_name: str | None = None,
) -> dict[str | None, dict | None]:
    """Resolve LCIA items required for one method (or method pair)."""
    needs_lcia_l2 = REGISTRY.method_requires_lcia(l2_method, run.context.fu_code)
    needs_lcia_l1 = bool(l1_name) and REGISTRY.method_requires_lcia(l1_name, None)
    if not (needs_lcia_l1 or needs_lcia_l2):
        return {None: None}
    if not run.lcia_by_method:
        run.state.skipped_years.setdefault(run.year, "LCIA unavailable")
        return {}
    items: dict[str | None, dict | None] = {}
    filter_key = _filters_cache_key(run=run)
    for key, value in run.lcia_by_method.items():
        cache_key = (
            str(run.context.fu_code),
            str(key),
            filter_key,
            id(value),
        )
        cached = run.state.lcia_sliced_payload_cache.get(cache_key)
        if cached is None:
            cached = _slice_lcia_payload_for_compute(
                context=run.context,
                payload=value,
            )
            run.state.lcia_sliced_payload_cache[cache_key] = cached
        items[key] = cached
    return items


def _reference_years_for(
    *,
    run: _L2RunContext,
    l2_method: str,
    l1_name: str | None,
) -> list[int | None]:
    """Return applicable reference years for a method/method pair."""
    is_ar = _is_ar_l2(l2_method=l2_method, fu_code=run.context.fu_code) or bool(
        l1_name and _is_ar_l1(l1_name)
    )
    if not is_ar:
        return [None]
    refs: list[int | None] = []
    if run.context.reference_years:
        for ref in run.context.reference_years:
            refs.append(int(ref))
    elif run.context.historical_years:
        default_cutoff = default_historical_cutoff_for_source(run.context.source)
        for ref in run.context.historical_years:
            if default_cutoff is None or int(ref) <= default_cutoff:
                refs.append(int(ref))
    else:
        refs = [None]
    return refs


def _l1_weights_key_for_pair(
    *,
    base_key: str,
    l1_name: str,
    l2_method: str,
) -> str:
    """Return L1 weight key for a specific L1/L2 pair."""
    family = REGISTRY.method_family(l1_name, level="L1")
    if family in {"EG_POP", "PR_GDPCAP"}:
        return f"{base_key}__for__{l2_method}"
    return base_key
