"""LCIA data loading for per-year allocation runs."""

from typing import Any, cast

import pandas as pd

from pyaesa.process.mrios.utils.io.metadata import (
    _get_year_entry,
    _read_metadata,
)
from pyaesa.process.mrios.utils.io.paths import _get_metadata_path

from .enacting_metric_units import lcia_unit_series_for_method
from .load_mrio import _load_lcia_l1_metric, _load_lcia_l2_metric
from .lcia_status import resolve_lcia_status
from .paths import _get_mrio_year_dir
from ..runtime.methods.fallback_policy import resolve_latest_available_historical_year
from ..runtime.reporting.family import emit_deduplicated_family_warning
from ..methods.lcia_inputs import (
    aggregate_lcia_to_parent,
    load_impact_parent_mapping,
)
from ..methods.lcia_key_selection import required_lcia_metric_keys_for_context
from ..methods.registry.registry import REGISTRY

_USE_CONTEXT_GROUP_VERSION = object()


def _emit_lcia_notice(*, context, state, key: str, message: str) -> None:
    """Emit one deduplicated LCIA availability notice."""
    emit_deduplicated_family_warning(
        context=context,
        state=state,
        key=key,
        message=message,
    )


def _metadata_for_matrix_version(
    *,
    context,
    state,
    matrix_version: str | None,
) -> tuple[dict, str]:
    """Return cached metadata payload and metadata path for one matrix version."""
    key = (str(context.source), matrix_version)
    cached = state.lcia_metadata_cache.get(key)
    if cached is not None:
        return cached
    meta = _read_metadata(context.source, matrix_version=matrix_version)
    meta_path = str(_get_metadata_path(context.source, matrix_version=matrix_version))
    state.lcia_metadata_cache[key] = (meta, meta_path)
    return meta, meta_path


def _method_payload_cache_key(
    *,
    matrix_version: str | None,
    saved_dir,
    lcia_method: str,
) -> tuple[str | None, str, str]:
    """Build stable cache key for one method payload loaded from one year directory."""
    return (matrix_version, str(saved_dir), str(lcia_method))


def _available_lcia_years_for_method(
    *,
    context,
    state,
    matrix_version: str | None,
    lcia_method: str,
) -> list[int]:
    """Return sorted historical years with available LCIA payloads for one method."""
    cache_key = (str(context.source), matrix_version, str(lcia_method))
    cached = state.lcia_available_years_cache.get(cache_key)
    if cached is not None:
        return cached

    meta, meta_path = _metadata_for_matrix_version(
        context=context,
        state=state,
        matrix_version=matrix_version,
    )
    years_meta = meta.get("years", {})
    available_years: list[int] = []
    for y_raw in years_meta.keys():
        try:
            y = int(y_raw)
        except (TypeError, ValueError):
            continue
        y_entry = cast(
            dict,
            _get_year_entry({"years": {str(y_raw): years_meta[y_raw]}}, y),
        )
        y_available, _ = resolve_lcia_status(
            y_entry,
            lcia_method,
            metadata_path=meta_path,
            year=y,
        )
        if not y_available:
            continue
        y_saved_dir = _get_mrio_year_dir(
            source=context.source,
            year=y,
            group_version=matrix_version,
        )
        if y_saved_dir.exists():
            available_years.append(int(y))

    available_years = sorted(set(available_years))
    state.lcia_available_years_cache[cache_key] = available_years
    return available_years


def _load_lcia_for_year(
    *,
    context,
    state,
    year: int,
    saved_dir,
    group_version_override: str | None | object = _USE_CONTEXT_GROUP_VERSION,
    allow_method_year_fallback: bool = False,
    selected_lcia_methods: list[str] | None = None,
    method_year_out: dict[str, int] | None = None,
) -> dict[str, dict[str, pd.DataFrame]] | None:
    """Load and normalize LCIA payloads for one year.

    Args:
        context: Run context.
        state: Mutable run state.
        year: Year being processed.
        saved_dir: MRIO saved directory for this year.

    Returns:
        Mapping ``method -> lcia payload`` or ``None`` when unavailable.
    """
    if not context.needs_lcia:
        return None

    if group_version_override is _USE_CONTEXT_GROUP_VERSION:
        matrix_version: str | None = context.group_version
    else:
        matrix_version = cast(str | None, group_version_override)
    meta, meta_path = _metadata_for_matrix_version(
        context=context,
        state=state,
        matrix_version=matrix_version,
    )
    year_entry = _get_year_entry(meta, year)
    if year_entry is None:
        raise ValueError(
            f"Processed MRIO metadata at {meta_path} is missing year {year} "
            f"for source '{context.source}' and matrix version '{matrix_version}'."
        )

    lcia_by_method: dict[str, dict[str, pd.DataFrame]] = {}
    required_l1_keys, required_l2_keys = required_lcia_metric_keys_for_context(
        context=context,
        registry=REGISTRY,
    )

    lcia_methods = (
        selected_lcia_methods if selected_lcia_methods is not None else context.lcia_methods
    )
    for lcia_method in lcia_methods or []:
        # Metadata gated loading: skip methods not marked available for this year.
        selected_year_entry = year_entry
        available, reason = resolve_lcia_status(
            year_entry,
            lcia_method,
            metadata_path=meta_path,
            year=year,
        )
        load_saved_dir = saved_dir
        loaded_year = int(year)
        if not available and allow_method_year_fallback:
            available_years = _available_lcia_years_for_method(
                context=context,
                state=state,
                matrix_version=matrix_version,
                lcia_method=lcia_method,
            )
            fallback_year = resolve_latest_available_historical_year(
                requested_year=int(year),
                available_years=available_years,
            )
            if fallback_year is not None:
                loaded_year = int(fallback_year.resolved_year)
                load_saved_dir = _get_mrio_year_dir(
                    source=context.source,
                    year=loaded_year,
                    group_version=matrix_version,
                )
                selected_year_entry = _get_year_entry(meta, loaded_year)
                available = True
                reason = None
        if not available:
            state.skipped_years.setdefault(year, {})[lcia_method] = reason
            if allow_method_year_fallback:
                reason_msg = reason or "LCIA unavailable for this method/year."
                _emit_lcia_notice(
                    context=context,
                    state=state,
                    key=f"lcia-missing:{matrix_version}:{lcia_method}",
                    message=(
                        "No LCIA data available for method "
                        f"'{lcia_method}' up to studied year {year}; "
                        "LCIA-dependent methods are skipped for this year. "
                        f"Reason: {reason_msg}. Metadata: {meta_path}"
                    ),
                )
            continue

        payload_key = _method_payload_cache_key(
            matrix_version=matrix_version,
            saved_dir=load_saved_dir,
            lcia_method=lcia_method,
        )
        lcia = state.lcia_method_payload_cache.get(payload_key)
        if lcia is None:
            lcia_raw: dict[str, pd.DataFrame] = {}
            for key in sorted(required_l1_keys):
                lcia_raw[key] = _load_lcia_l1_metric(load_saved_dir, lcia_method, key)
            for key in sorted(required_l2_keys):
                lcia_raw[key] = _load_lcia_l2_metric(load_saved_dir, lcia_method, key)

            impact_parent_map = state.cf_by_method.get(lcia_method)
            if impact_parent_map is None:
                impact_parent_map = load_impact_parent_mapping(
                    source=context.source,
                    lcia_method=lcia_method,
                )
                state.cf_by_method[lcia_method] = impact_parent_map
            # Runtime LCIA payloads are normalized to parent impacts before method
            # equations and enacting metric recording consume them.
            lcia = aggregate_lcia_to_parent(lcia_raw, impact_parent_map)
            state.lcia_method_payload_cache[payload_key] = lcia
        lcia_by_method[lcia_method] = lcia
        if method_year_out is not None:
            method_year_out[str(lcia_method)] = int(loaded_year)

        if lcia_method not in state.lcia_units:
            state.lcia_units[lcia_method] = lcia_unit_series_for_method(
                year_entry=cast(dict[str, Any], selected_year_entry),
                year=loaded_year,
                lcia_method=lcia_method,
            )

    return lcia_by_method or None
