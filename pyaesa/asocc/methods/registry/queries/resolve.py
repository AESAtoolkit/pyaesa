"""Resolution logic used by the registry facade."""

from pyaesa.asocc.methods.registry.model.types import normalize_fu_code
from pyaesa.asocc.methods.registry.specs.all_specs import build_raw_method_specs


def _registry_name_sets() -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Return canonical method-name sets derived from registry specs."""
    l1_names: set[str] = set()
    l2_names_by_fu: dict[str, set[str]] = {}
    l1_kinds_by_name: dict[str, set[str]] = {}
    for raw in build_raw_method_specs():
        name = str(raw["name"])
        level = str(raw["level"])
        if level == "L1":
            l1_names.add(name)
            l1_kind = raw.get("l1_kind")
            if l1_kind is not None:
                l1_kinds_by_name.setdefault(name, set()).add(str(l1_kind))
            continue
        fu_code = raw["fu_code"]
        l2_names_by_fu.setdefault(normalize_fu_code(str(fu_code)), set()).add(name)
    return l1_names, l2_names_by_fu, l1_kinds_by_name


_CANONICAL_L1_NAMES, _CANONICAL_L2_NAMES_BY_FU, _L1_KINDS_BY_NAME = _registry_name_sets()


def resolve_user_l1_method_name(
    name: str,
    *,
    l1_kind: str | None = None,
) -> str:
    """Validate and return one canonical L1 registry method label."""
    cleaned = str(name).strip()
    if cleaned not in _CANONICAL_L1_NAMES:
        supported = sorted(_CANONICAL_L1_NAMES)
        raise ValueError(
            "L1 method labels must use one canonical scientific registry label. "
            f"Received {cleaned!r}. Supported labels: {supported}."
        )
    if l1_kind is None:
        return cleaned
    supported_kinds = _L1_KINDS_BY_NAME.get(cleaned, set())
    if not supported_kinds or l1_kind in supported_kinds:
        return cleaned
    raise ValueError(
        "L1 method label is incompatible with the requested boundary kind. "
        f"Received method={cleaned!r}, l1_kind={l1_kind!r}, "
        f"supported_l1_kinds={sorted(supported_kinds)}."
    )


def resolve_user_l2_method_name(
    *,
    name: str,
    fu_code: str,
) -> str:
    """Validate and return one canonical L2 registry method label for an FU."""
    cleaned = str(name).strip()
    fu_norm = normalize_fu_code(fu_code)
    supported = _CANONICAL_L2_NAMES_BY_FU.get(fu_norm)
    if supported is None:
        raise ValueError(f"Unsupported functional unit {fu_code!r}.")
    if cleaned not in supported:
        raise ValueError(
            "L2 method labels must use a canonical scientific registry label "
            f"compatible with functional unit {fu_norm!r}. Received {cleaned!r}. "
            f"Supported labels: {sorted(supported)}."
        )
    return cleaned


def resolve_required_indices(
    *,
    fu_code: str,
    selected_l1: list[str],
    combined: list[tuple[str, str]],
    selected_l2_one_step: list[str],
    l1_kinds_needed: set[str],
    registry,
    normalize_fu_code,
) -> set[str]:
    """Resolve required indices over all selected methods."""
    fu_norm = normalize_fu_code(fu_code)
    required_indices: set[str] = set()
    for name in selected_l1:
        required_indices.update(registry.required_indices(name, None))

    if any(
        registry.method_family(name, level="L1") in {"EG_POP", "PR_GDPCAP"} for name in selected_l1
    ):
        # Neutral L1 methods infer region side from FU / required L1 boundary kind.
        if fu_norm == "L1.b":
            required_indices.add("r_p")
        elif fu_norm == "L1.a":
            required_indices.add("r_f")
        else:
            if "CBA_FD" in l1_kinds_needed:
                required_indices.add("r_f")
            if "PBA" in l1_kinds_needed:
                required_indices.add("r_p")

    for name, _ in combined:
        required_indices.update(registry.required_indices(name, fu_code, l1_weighting=True))
    for name in selected_l2_one_step:
        required_indices.update(registry.required_indices(name, fu_code, l1_weighting=False))
    return required_indices
