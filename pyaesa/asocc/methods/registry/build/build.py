"""Registry construction for allocation method metadata."""

from collections.abc import Callable
from typing import TypedDict

from pyaesa.asocc.methods.registry.specs.all_specs import build_raw_method_specs


class MethodSpecPayload(TypedDict):
    """Normalized payload used to instantiate registry MethodSpec entries."""

    name: str
    level: str
    fu_code: str | None
    l1_weighting: bool
    needs_lcia: bool
    needs_pop: bool
    needs_gdp: bool
    needs_utility: bool
    needs_rp: bool
    indices: tuple[str, ...]
    l1_kind: str | None
    l2_weight_axis: str | None
    expand_ar_years: bool
    family: str


_L2_FU_L1_KIND_MAP = {
    "L2.a.a": "CBA_FD",
    "L2.b.a": "CBA_FD",
    "L2.c.a": "CBA_FD",
    "L2.a.b": "CBA_TD",
    "L2.b.b": "CBA_TD",
    "L2.c.b": "CBA_TD",
    "L2.a.c": "PBA",
    "L2.b.c": "PBA",
    "L2.c.c": "PBA",
}

_METHOD_FAMILY_BY_LEVEL_NAME = {
    # level, method
    ("L1", "EG(Pop)"): "EG_POP",
    ("L1", "PR(GDPcap)"): "PR_GDPCAP",
    ("L1", "PR-HR(Ecap,cum^{CBA_FD})"): "PR_HR",
    ("L1", "PR-HR(Ecap,cum^{PBA})"): "PR_HR",
    ("L1", "AR(E^{CBA_FD})"): "AR_E",
    ("L1", "AR(E^{PBA})"): "AR_E",
    ("L1", "AR(Ecap^{CBA_FD})"): "AR_ECAP",
    ("L1", "AR(Ecap^{PBA})"): "AR_ECAP",
    ("L2", "UT(FD)"): "UT_FD",
    ("L2", "UT(FDa)"): "UT_FDA",
    ("L2", "UT(GVAa)"): "UT_GVAA",
    ("L2", "UT(TD)"): "UT_TD",
    ("L2", "UT(GVA)"): "UT_GVA",
    ("L2", "AR(E^{CBA_FD})"): "AR_E",
    ("L2", "AR(E^{CBA_TD})"): "AR_E",
    ("L2", "AR(E^{PBA})"): "AR_E",
}


def _required_str(raw: dict[str, object], key: str) -> str:
    """Return required string from raw spec payload."""
    if key not in raw:
        raise ValueError(f"Missing required method spec key: {key}")
    return str(raw[key])


def _optional_str(raw: dict[str, object], key: str) -> str | None:
    """Return optional string from raw spec payload."""
    value = raw.get(key)
    return None if value is None else str(value)


def _required_indices(
    raw: dict[str, object],
    *,
    method_name: str,
) -> tuple[str, ...]:
    """Return required indices tuple from raw spec payload."""
    raw_indices = raw.get("indices")
    if not isinstance(raw_indices, tuple):
        raise ValueError(
            f"indices must be provided as a tuple for method '{method_name}', got "
            f"{type(raw_indices).__name__}."
        )
    return tuple(str(v) for v in raw_indices)


def _family_for_method(*, name: str, level: str) -> str:
    """Return canonical family for one exact level/name pair."""
    key = (level, name)
    if key not in _METHOD_FAMILY_BY_LEVEL_NAME:
        raise ValueError(f"Unsupported method family mapping for level={level!r}, method={name!r}.")
    return _METHOD_FAMILY_BY_LEVEL_NAME[key]


def _spec_from_raw(
    *,
    raw: dict[str, object],
    normalize_fu_code: Callable[[str], str],
) -> MethodSpecPayload:
    """Build one method spec entry from a raw payload."""
    name = _required_str(raw, "name")
    level = _required_str(raw, "level")
    fu_code = _optional_str(raw, "fu_code")
    if fu_code is not None:
        fu_code = normalize_fu_code(fu_code)
    indices = _required_indices(raw, method_name=name)
    l1_weighting = bool(raw.get("l1_weighting", False))
    l1_kind = _optional_str(raw, "l1_kind")
    l2_weight_axis = _optional_str(raw, "l2_weight_axis")
    expand_ar_years = bool(raw.get("expand_ar_years", True))

    resolved_l1_kind = l1_kind
    if level == "L2":
        if fu_code is None:
            raise ValueError("L2 registry entries must define fu_code.")
        # L2 methods must resolve a single L1 boundary kind for compatibility
        # checks and two step pairing policy.
        if resolved_l1_kind is None:
            resolved_l1_kind = _L2_FU_L1_KIND_MAP.get(fu_code)
        if resolved_l1_kind is None:
            raise ValueError(f"No L1 boundary metadata for L2 registry entry fu_code={fu_code}.")
    resolved_l2_weight_axis = l2_weight_axis
    if level == "L2" and l1_weighting and resolved_l2_weight_axis is None:
        raise ValueError(
            "Two-step L2 registry entries must declare l2_weight_axis explicitly "
            f"for method '{name}' ({fu_code})."
        )

    return MethodSpecPayload(
        name=name,
        level=level,
        fu_code=fu_code,
        l1_weighting=l1_weighting,
        needs_lcia=bool(raw.get("needs_lcia", False)),
        needs_pop=bool(raw.get("needs_pop", False)),
        needs_gdp=bool(raw.get("needs_gdp", False)),
        needs_utility=bool(raw.get("needs_utility", False)),
        needs_rp=bool(raw.get("needs_rp", False)),
        indices=indices,
        l1_kind=resolved_l1_kind,
        l2_weight_axis=resolved_l2_weight_axis,
        expand_ar_years=expand_ar_years,
        family=_family_for_method(name=name, level=level),
    )


def build_method_specs(
    *,
    normalize_fu_code: Callable[[str], str],
) -> list[MethodSpecPayload]:
    """Build canonical method specs from declarative registry specs."""
    raw_method_specs = build_raw_method_specs()
    return [
        _spec_from_raw(
            raw=raw,
            normalize_fu_code=normalize_fu_code,
        )
        for raw in raw_method_specs
    ]
