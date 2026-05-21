"""Shared typed accessors for output spec cache state."""

from typing import TypeAlias, cast

from ....runtime.output.contracts import OutputSpec

OutputSpecCacheKey: TypeAlias = tuple[object, ...]
OutputSpecCache: TypeAlias = dict[OutputSpecCacheKey, OutputSpec]


def output_spec_cache_for_state(state: object | None) -> OutputSpecCache | None:
    """Return typed output spec cache when available on state."""
    if state is None:
        return None
    cache = getattr(state, "output_spec_cache", None)
    if not isinstance(cache, dict):
        return None
    return cast(OutputSpecCache, cache)


def get_cached_output_spec(
    *,
    state: object | None,
    key: OutputSpecCacheKey,
) -> OutputSpec | None:
    """Return cached OutputSpec for key when cache exists."""
    cache = output_spec_cache_for_state(state)
    if cache is None:
        return None
    return cache.get(key)


def set_cached_output_spec(
    *,
    state: object | None,
    key: OutputSpecCacheKey,
    spec: OutputSpec,
) -> None:
    """Store OutputSpec in cache when cache exists on state."""
    cache = output_spec_cache_for_state(state)
    if cache is None:
        return
    cache[key] = spec
