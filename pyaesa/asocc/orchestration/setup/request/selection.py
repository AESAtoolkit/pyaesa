"""Selection, filter, and index-tag normalization for setup orchestration."""

from pyaesa.process.mrios.utils.io.paths import _resolve_version_tag
from pyaesa.shared.selectors.path_tokens import build_selector_filter_segment
from ....data.source_schema import ISO3_SOURCE_KEY
from ....methods.registry.registry import REGISTRY, normalize_fu_code, resolve_required_indices
from ....runtime.methods.labels import l1_l2_method_label
from pyaesa.asocc.orchestration.setup.request.types import _GroupingBundle, _SelectionBundle

_ISO3_ALLOWED_L1_METHODS = {"EG(Pop)", "PR(GDPcap)"}
_FILTER_KEYS = {
    "r_p": "r_p",
    "s_p": "s_p",
    "r_c": "r_c",
    "r_f": "r_f",
}


def normalize_filter(values: list[str] | None) -> list[str] | None:
    """Normalize a filter list by stripping whitespace."""
    if values is None:
        return None
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    return cleaned or None


def build_indices_tag(filters: dict[str, list[str] | None]) -> str:
    """Build the deterministic studied-indices tag."""
    parts: list[str] = []
    for key, label in _FILTER_KEYS.items():
        values = filters.get(label)
        if not values:
            continue
        segment = build_selector_filter_segment(key=key, values=values)
        if segment:
            parts.append(segment)
    return "__".join(parts) if parts else "all_indices"


def apply_filter_messages(
    *,
    required_indices: set[str],
    filters: dict[str, list[str] | None],
) -> dict[str, list[str] | None]:
    """Validate filters and keep only those required by selected methods."""
    out: dict[str, list[str] | None] = {}
    for index_key, filter_key in _FILTER_KEYS.items():
        values = filters.get(filter_key)
        if index_key in required_indices:
            out[filter_key] = values
            continue
        if values:
            raise ValueError(
                f"{filter_key} provided but index '{index_key}' is not "
                "required by the selected methods."
            )
        out[filter_key] = None
    return out


def resolve_l1_kinds(
    *,
    fu_code: str,
    l1_lcia_kind: str,
    combined: list[tuple[str, str]],
) -> set[str]:
    """Resolve which L1 LCIA boundary kinds are required for this run."""
    fu_norm = normalize_fu_code(fu_code)
    l1_kinds_needed: set[str] = set()
    if fu_norm in {"L1.a", "L1.b"}:
        l1_kinds_needed.add(l1_lcia_kind)
    else:
        for l2_name, _ in combined:
            l1_kinds_needed.add(REGISTRY.l1_kind_for_l2_method(l2_name))
    return l1_kinds_needed


def needs_lcia(
    *,
    fu_code: str,
    selected_l1: list[str],
    combined: list[tuple[str, str]],
    selected_l2_one_step: list[str],
) -> bool:
    """Return whether at least one selected method needs LCIA inputs."""
    l2_needs = any(REGISTRY.method_requires_lcia(name, fu_code) for name, _ in combined) or any(
        REGISTRY.method_requires_lcia(name, fu_code) for name in selected_l2_one_step
    )
    return l2_needs or any(REGISTRY.method_requires_lcia(name, None) for name in selected_l1)


def _validate_td_grouped_output(*, fu_code: str, aggreg_indices: bool) -> None:
    """Reject unsupported grouped outputs for TD functional units."""
    if aggreg_indices and fu_code in {"L2.a.b", "L2.b.b", "L2.c.b"}:
        raise ValueError(
            "aggreg_indices=True is not allowed for L2.a.b/L2.b.b/L2.c.b because "
            "CBA_TD perimeters can introduce double counting when aggregating "
            "allocation outputs. Run MRIO processing with grouping enabled "
            "(group_reg/group_sec with group_version) so grouped matrices are "
            "saved under the grouped MRIO version (custom_classification_<group_version>), then "
            "run deterministic_asocc on that grouped version."
        )


def _resolve_grouping(
    *,
    group_reg: bool | None,
    group_sec: bool | None,
    group_version: str | None,
) -> _GroupingBundle:
    """Resolve grouping flags to deterministic booleans and region version."""
    apply_group_reg = bool(group_reg) if group_reg is not None else False
    apply_group_sec = bool(group_sec) if group_sec is not None else False
    return _GroupingBundle(
        apply_group_reg=apply_group_reg,
        apply_group_sec=apply_group_sec,
        group_version_reg=group_version if apply_group_reg else None,
    )


def _build_selection_bundle(
    *,
    fu_code: str,
    selected_l1: list[str],
    combined: list[tuple[str, str]],
    selected_l2_one_step: list[str],
    l1_lcia_kind: str,
) -> _SelectionBundle:
    """Build a normalized selection bundle from already pruned method lists."""
    l1_kinds_needed = resolve_l1_kinds(
        fu_code=fu_code,
        l1_lcia_kind=l1_lcia_kind,
        combined=combined,
    )
    required_indices = resolve_required_indices(
        fu_code=fu_code,
        selected_l1=selected_l1,
        combined=combined,
        selected_l2_one_step=selected_l2_one_step,
        l1_kinds_needed=l1_kinds_needed,
    )
    needs_lcia_flag = needs_lcia(
        fu_code=fu_code,
        selected_l1=selected_l1,
        combined=combined,
        selected_l2_one_step=selected_l2_one_step,
    )
    needs_mrio = bool(combined or selected_l2_one_step)
    l1_only_no_mrio = (not needs_mrio) and (not needs_lcia_flag)
    selected_methods = {
        "l1": selected_l1,
        "l2_in_l1": [l1_l2_method_label(l1_method=l1, l2_method=l2) for l2, l1 in combined],
        "l2_vs_global": selected_l2_one_step,
    }
    return _SelectionBundle(
        selected_l1=selected_l1,
        combined=combined,
        selected_l2_one_step=selected_l2_one_step,
        l1_kinds_needed=l1_kinds_needed,
        required_indices=required_indices,
        needs_lcia_flag=needs_lcia_flag,
        l1_only_no_mrio=l1_only_no_mrio,
        selected_methods=selected_methods,
    )


def _resolve_selection_bundle(
    *,
    fu_code: str,
    l_1: list[str] | None,
    l_2_combined_with_l_1: list[tuple[str, str]] | None,
    l_2_one_step: list[str] | None,
    l1_lcia_kind: str,
) -> _SelectionBundle:
    """Validate and normalize L1/L2 method selections."""
    selected_l1 = sorted(set(l_1 or []))
    combined = list(dict.fromkeys(l_2_combined_with_l_1 or []))
    selected_l2_one_step = sorted(set(l_2_one_step or []))

    if selected_l2_one_step:
        REGISTRY.validate_selection(fu_code, selected_l2_one_step, l1_weighting=False)
    if combined:
        REGISTRY.validate_selection(fu_code, [m for m, _ in combined], l1_weighting=True)

    return _build_selection_bundle(
        fu_code=fu_code,
        selected_l1=selected_l1,
        combined=combined,
        selected_l2_one_step=selected_l2_one_step,
        l1_lcia_kind=l1_lcia_kind,
    )


def _prune_lcia_methods_without_lcia_input(
    *,
    fu_code: str,
    lcia_methods: list[str] | None,
    selection: _SelectionBundle,
) -> tuple[_SelectionBundle, list[str]]:
    """Remove LCIA dependent methods when no LCIA method is requested."""
    if lcia_methods:
        return selection, []

    selected_l1 = [
        name for name in selection.selected_l1 if not REGISTRY.method_requires_lcia(name, None)
    ]
    selected_l2_one_step = [
        name
        for name in selection.selected_l2_one_step
        if not REGISTRY.method_requires_lcia(name, fu_code)
    ]
    combined = [
        (l2_name, l1_name)
        for l2_name, l1_name in selection.combined
        if (not REGISTRY.method_requires_lcia(l2_name, fu_code))
        and (not REGISTRY.method_requires_lcia(l1_name, None))
    ]
    dropped = sorted(
        set(selection.selected_l1) - set(selected_l1)
        | set(selection.selected_l2_one_step) - set(selected_l2_one_step)
        | {f"{l2}::{l1}" for l2, l1 in selection.combined if (l2, l1) not in combined}
    )
    if not dropped:
        return selection, []

    return (
        _build_selection_bundle(
            fu_code=fu_code,
            selected_l1=selected_l1,
            combined=combined,
            selected_l2_one_step=selected_l2_one_step,
            l1_lcia_kind=("PBA" if fu_code == "L1.b" else "CBA_FD"),
        ),
        dropped,
    )


def _restrict_selection_for_iso3_mode(
    *,
    fu_code: str,
    selection: _SelectionBundle,
) -> _SelectionBundle:
    """Restrict no source ISO3 mode to L1 EG/PR(GDPcap) methods only."""
    if fu_code not in {"L1.a", "L1.b"}:
        raise ValueError("source='iso3' only supports L1 functional units (L1.a or L1.b).")
    selected_l1 = [name for name in selection.selected_l1 if name in _ISO3_ALLOWED_L1_METHODS]
    if not selected_l1:
        raise ValueError(
            f"source='iso3' supports only L1 methods {sorted(_ISO3_ALLOWED_L1_METHODS)}."
        )
    return _build_selection_bundle(
        fu_code=fu_code,
        selected_l1=selected_l1,
        combined=[],
        selected_l2_one_step=[],
        l1_lcia_kind=("PBA" if fu_code == "L1.b" else "CBA_FD"),
    )


def _resolve_filters(
    *,
    required_indices: set[str],
    r_p: list[str] | None,
    s_p: list[str] | None,
    r_c: list[str] | None,
    r_f: list[str] | None,
) -> tuple[dict[str, list[str] | None], str]:
    """Normalize and validate index filters."""
    filters = {
        "r_p": normalize_filter(r_p),
        "s_p": normalize_filter(s_p),
        "r_c": normalize_filter(r_c),
        "r_f": normalize_filter(r_f),
    }
    validated = apply_filter_messages(
        required_indices=required_indices,
        filters=filters,
    )
    return validated, build_indices_tag(validated)


def _resolve_output_domain_tag(
    *,
    source: str,
    group_version: str | None,
) -> str | None:
    """Resolve output domain folder tag for MRIO runs.

    Mirrors processed MRIO matrix version tagging:
    - MRIO source + no group_version -> ``original_classification``
    - MRIO source + group_version -> ``custom_classification_<group_version>``
    - ISO3 source -> no domain tag
    """
    if source == ISO3_SOURCE_KEY:
        return None
    return _resolve_version_tag(group_version)


def _l1_methods_in_scope(selection: _SelectionBundle) -> set[str]:
    """Collect all L1 methods that can affect this run."""
    methods = set(selection.selected_l1)
    methods.update(pair[1] for pair in selection.combined)
    return methods


def _uses_l1_post_original_domain(
    *,
    selection: _SelectionBundle,
    grouping: _GroupingBundle,
    l1_reg_aggreg: str,
) -> bool:
    """Return whether this run requires original domain L1 post computation."""
    if l1_reg_aggreg != "post" or grouping.group_version_reg is None:
        return False
    families = {
        REGISTRY.method_family(name, level="L1") for name in _l1_methods_in_scope(selection)
    }
    return bool(families.intersection({"PR_HR", "AR_ECAP"}))
