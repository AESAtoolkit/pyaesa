"""AR result indexing for allocation runs."""

import numpy as np
import pandas as pd

_IndexCacheKey = tuple[object, ...]
_CachedIndexEntry = tuple[pd.Index, pd.Index]


def _cached_index_for(
    *,
    index_cache: dict[_IndexCacheKey, _CachedIndexEntry] | None,
    cache_key: _IndexCacheKey | None,
    source_index: pd.Index,
) -> pd.Index | None:
    """Return one cached attached index only when it belongs to this exact source index."""
    if index_cache is None or cache_key is None:
        return None
    cached_entry = index_cache.get(cache_key)
    if cached_entry is None:
        return None
    cached_source_index, cached_index = cached_entry
    if cached_source_index is not source_index:
        return None
    return cached_index


def _store_cached_index(
    *,
    index_cache: dict[_IndexCacheKey, _CachedIndexEntry] | None,
    cache_key: _IndexCacheKey | None,
    source_index: pd.Index,
    attached_index: pd.Index,
) -> None:
    """Store one attached index together with its exact source index owner."""
    if index_cache is None or cache_key is None:
        return
    index_cache[cache_key] = (source_index, attached_index)


def _apply_impact_level(
    result: pd.DataFrame,
    impact: str,
) -> pd.DataFrame:
    """Add impact level to a result index."""
    idx = result.index
    if isinstance(idx, pd.MultiIndex) and "impact" in idx.names:
        mask = idx.get_level_values("impact") == impact
        return result.loc[mask].copy(deep=False)
    if idx.name == "impact":
        return result.loc[idx == impact].copy(deep=False)
    if isinstance(idx, pd.MultiIndex):
        new_index = pd.MultiIndex(
            levels=[pd.Index([impact], name="impact"), *idx.levels],
            codes=[
                np.zeros(len(result), dtype=np.intp),
                *[np.asarray(code, dtype=np.intp) for code in idx.codes],
            ],
            names=["impact", *idx.names],
            verify_integrity=False,
        )
    else:
        base_codes, base_levels = pd.factorize(idx, sort=False)
        new_index = pd.MultiIndex(
            levels=[
                pd.Index([impact], name="impact"),
                pd.Index(base_levels, name=idx.name, copy=False),
            ],
            codes=[
                np.zeros(len(result), dtype=np.intp),
                np.asarray(base_codes, dtype=np.intp),
            ],
            names=["impact", idx.name],
            verify_integrity=False,
        )
    return result.set_axis(new_index, axis=0)


def _add_reference_level(
    result: pd.DataFrame,
    ref_year: int,
    index_cache: dict[_IndexCacheKey, _CachedIndexEntry] | None = None,
) -> pd.DataFrame:
    """Attach ``reference_year`` as an index level on the result."""
    return _attach_trailing_constant_levels(
        result=result,
        levels=(("reference_year", int(ref_year)),),
        index_cache=index_cache,
    )


def _attach_trailing_constant_levels(
    *,
    result: pd.DataFrame,
    levels: tuple[tuple[str, int], ...],
    index_cache: dict[_IndexCacheKey, _CachedIndexEntry] | None = None,
) -> pd.DataFrame:
    """Attach or replace trailing constant index levels in one rebuild."""
    if not levels:
        return result
    idx = result.index
    cache_key: _IndexCacheKey | None = None
    if index_cache is not None:
        cache_key = ("trailing_levels", id(idx), levels)
        cached = _cached_index_for(
            index_cache=index_cache,
            cache_key=cache_key,
            source_index=idx,
        )
        if cached is not None:
            return result.set_axis(cached, axis=0)
    if isinstance(idx, pd.MultiIndex):
        names = list(idx.names)
        levels_out = list(idx.levels)
        codes_out = [np.asarray(code, dtype=np.intp) for code in idx.codes]
        positions = {str(name): pos for pos, name in enumerate(names)}
        for level_name, level_value in levels:
            if level_name in positions:
                level_pos = positions[level_name]
                levels_out[level_pos] = pd.Index([level_value], name=level_name)
                codes_out[level_pos] = np.zeros(len(result), dtype=np.intp)
                continue
            levels_out.append(pd.Index([level_value], name=level_name))
            codes_out.append(np.zeros(len(result), dtype=np.intp))
            names.append(level_name)
            positions[level_name] = len(names) - 1
        out_index = pd.MultiIndex(
            levels=levels_out,
            codes=codes_out,
            names=names,
            verify_integrity=False,
        )
        _store_cached_index(
            index_cache=index_cache,
            cache_key=cache_key,
            source_index=idx,
            attached_index=out_index,
        )
        return result.set_axis(out_index, axis=0)

    existing_name = str(idx.name) if idx.name is not None else None
    if len(levels) == 1 and existing_name == levels[0][0]:
        level_name, level_value = levels[0]
        out_index = pd.Index([level_value] * len(result), name=level_name)
        _store_cached_index(
            index_cache=index_cache,
            cache_key=cache_key,
            source_index=idx,
            attached_index=out_index,
        )
        return result.set_axis(out_index, axis=0)

    base_codes, base_levels = pd.factorize(idx, sort=False)
    index_levels = [pd.Index(base_levels, name=idx.name, copy=False)]
    index_codes = [np.asarray(base_codes, dtype=np.intp)]
    index_names = [idx.name]
    for level_name, level_value in levels:
        index_levels.append(pd.Index([level_value], name=level_name))
        index_codes.append(np.zeros(len(result), dtype=np.intp))
        index_names.append(level_name)
    out_index = pd.MultiIndex(
        levels=index_levels,
        codes=index_codes,
        names=index_names,
        verify_integrity=False,
    )
    _store_cached_index(
        index_cache=index_cache,
        cache_key=cache_key,
        source_index=idx,
        attached_index=out_index,
    )
    return result.set_axis(out_index, axis=0)


def _attach_impact_reference_levels(
    *,
    result: pd.DataFrame,
    impact: str | None,
    reference_year: int | None,
    trailing_levels: tuple[tuple[str, int], ...] = (),
    index_cache: dict[_IndexCacheKey, _CachedIndexEntry] | None = None,
) -> pd.DataFrame:
    """Attach optional impact/reference levels in one index rebuild on hot path."""
    if impact is None and reference_year is None and not trailing_levels:
        return result
    idx = result.index
    cache_key: _IndexCacheKey | None = None
    if index_cache is not None:
        cache_key = (
            "impact_reference_levels",
            id(idx),
            impact,
            reference_year,
            trailing_levels,
        )
        cached = _cached_index_for(
            index_cache=index_cache,
            cache_key=cache_key,
            source_index=idx,
        )
        if cached is not None:
            return result.set_axis(cached, axis=0)
    if isinstance(idx, pd.MultiIndex):
        names_raw = list(idx.names)
        names_text = [str(name) for name in names_raw]
    else:
        names_raw = [idx.name]
        names_text = [str(idx.name)]
    has_impact = "impact" in names_text
    trailing_level_names = {level_name for level_name, _ in trailing_levels}
    has_reference = "reference_year" in names_text
    has_trailing = any(level_name in names_text for level_name in trailing_level_names)
    if (
        (impact is not None and has_impact)
        or (reference_year is not None and has_reference)
        or has_trailing
    ):
        out = result
        if impact is not None:
            out = _apply_impact_level(out, impact)
        levels_to_attach = trailing_levels
        if reference_year is not None:
            levels_to_attach = (("reference_year", int(reference_year)), *trailing_levels)
        if levels_to_attach:
            out = _attach_trailing_constant_levels(
                result=out,
                levels=levels_to_attach,
                index_cache=index_cache,
            )
        _store_cached_index(
            index_cache=index_cache,
            cache_key=cache_key,
            source_index=idx,
            attached_index=out.index,
        )
        return out

    if isinstance(idx, pd.MultiIndex):
        levels = list(idx.levels)
        codes = [np.asarray(code, dtype=np.intp) for code in idx.codes]
        names = list(idx.names)
    else:
        base_codes, base_levels = pd.factorize(idx, sort=False)
        levels = [pd.Index(base_levels, name=idx.name, copy=False)]
        codes = [np.asarray(base_codes, dtype=np.intp)]
        names = [idx.name]
    if impact is not None:
        levels = [pd.Index([impact], name="impact"), *levels]
        codes = [np.zeros(len(result), dtype=np.intp), *codes]
        names = ["impact", *names]
    levels_to_attach = trailing_levels
    if reference_year is not None:
        levels_to_attach = (("reference_year", int(reference_year)), *trailing_levels)
    for level_name, level_value in levels_to_attach:
        levels.append(pd.Index([level_value], name=level_name))
        codes.append(np.zeros(len(result), dtype=np.intp))
        names.append(level_name)
    out_index = pd.MultiIndex(
        levels=levels,
        codes=codes,
        names=names,
        verify_integrity=False,
    )
    _store_cached_index(
        index_cache=index_cache,
        cache_key=cache_key,
        source_index=idx,
        attached_index=out_index,
    )
    return result.set_axis(out_index, axis=0)
