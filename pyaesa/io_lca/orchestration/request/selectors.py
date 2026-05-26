"""Selector normalization and validation ownership for IO-LCA entrypoints."""

from pyaesa.asocc.orchestration.setup.request.selection import (
    apply_filter_messages,
    build_indices_tag,
)
from pyaesa.asocc.orchestration.setup.loading.loading import (
    _load_source_tables,
    _validate_region_filter_labels,
    _validate_sector_filter_labels,
)

from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec


def _normalize_selector_values(
    value: str | list[str] | None,
) -> list[str] | None:
    """Normalize API selectors to cleaned list form.

    Args:
        value: Optional string or list of selector labels.

    Returns:
        ``None`` when omitted/empty, else cleaned list preserving order.
    """
    if value is None:
        return None
    raw = [value] if isinstance(value, str) else list(value)
    cleaned = [str(item).strip() for item in raw if str(item).strip()]
    return cleaned or None


def resolve_selectors(
    *,
    spec: IOLCAFUSpec,
    r_f: str | list[str] | None,
    r_c: str | list[str] | None,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
) -> tuple[dict[str, list[str] | None], str]:
    """Normalize selectors and build deterministic studied indices identity tag.

    Args:
        spec: Resolved FU routing specification.
        r_f: Final demand region selector(s).
        r_c: Consuming region selector(s).
        r_p: Producing region selector(s).
        s_p: Producing sector selector(s).

    Returns:
        Pair ``(filters, studied_indices_tag)`` where filters includes
        ``r_f``, ``r_c``, ``r_p``, ``s_p`` and the tag is used for run
        identity and metadata, not as a folder token.

    Raises:
        ValueError: If selectors are provided for axes not allowed by FU.
    """
    filters = {
        "r_f": _normalize_selector_values(r_f),
        "r_c": _normalize_selector_values(r_c),
        "r_p": _normalize_selector_values(r_p),
        "s_p": _normalize_selector_values(s_p),
    }
    validated = apply_filter_messages(required_indices=set(spec.selector_axes), filters=filters)
    tag = build_indices_tag(validated)
    return validated, tag


def has_multi_selected_indices(filters: dict[str, list[str] | None]) -> bool:
    """Return whether at least one selector axis has multiple explicit values."""
    for key in ("r_p", "s_p", "r_c", "r_f"):
        values = filters.get(key)
        if values and len(set(values)) > 1:
            return True
    return False


def validate_selector_labels(
    *,
    source: str,
    agg_version: str | None,
    agg_reg: bool,
    agg_sec: bool,
    filters: dict[str, list[str] | None],
) -> None:
    """Fail fast when selector labels are unknown in the source and MRIO classification domain.

    Args:
        source: Source key.
        agg_version: Aggregation version or ``None``.
        agg_reg: Region aggregation flag.
        agg_sec: Sector MRIO aggregation and disaggregation flag.
        filters: Selector payload produced by :func:`resolve_selectors`.
    """
    wb_df, ssp_df, _wb_raw, _ssp_raw = _load_source_tables(source=source)
    _validate_region_filter_labels(
        source=source,
        agg_version=agg_version,
        agg_reg=agg_reg,
        filters=filters,
        wb_df=wb_df,
        ssp_df=ssp_df,
    )
    if agg_sec or filters.get("s_p"):
        _validate_sector_filter_labels(
            source=source,
            agg_version=agg_version,
            filters=filters,
        )
