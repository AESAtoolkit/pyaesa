"""LCIA unit normalization and conversion helpers."""

_MASS_SCALES_IN_KG: dict[str, float] = {
    "mg": 1e-6,
    "g": 1e-3,
    "kg": 1.0,
    "t": 1e3,
    "kt": 1e6,
    "mt": 1e9,
    "tg": 1e9,
    "gt": 1e12,
}

_MASS_PREFIXES = tuple(sorted(_MASS_SCALES_IN_KG, key=len, reverse=True))


def _strip_annual_suffix(normalized: str) -> tuple[str, str]:
    """Split one normalized unit into base and optional annual suffix."""
    suffix_map = (
        ("/year", "_per_yr"),
        ("/yr", "_per_yr"),
        ("year1", "_per_yr"),
        ("yr1", "_per_yr"),
        ("year", "_per_yr"),
        ("yr", "_per_yr"),
    )
    for marker, suffix in suffix_map:
        if normalized.endswith(marker):
            return normalized.removesuffix(marker), suffix
    return normalized, ""


def _normalize_unit_label(unit: str) -> str:
    """Return one canonical unit key for LCIA unit matching."""
    normalized = str(unit).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if normalized in set(_MASS_SCALES_IN_KG):
        return normalized
    normalized, suffix = _strip_annual_suffix(normalized)
    if "co2eq" not in normalized:
        normalized = normalized.replace("co2e", "co2eq")
    prefixes = ("kg", "kt", "mt", "gt", "t")
    prefix = next((value for value in prefixes if normalized.startswith(value)), "")
    if prefix:
        body = normalized[len(prefix) :]
        for marker in ("equivalent", "equiv", "eq"):
            if body.endswith(marker):
                body = body[: -len(marker)]
                break
        if body:
            return f"{prefix}_{body}{suffix}"
        return f"{prefix}{suffix}"
    return f"{normalized}{suffix}"


def _split_mass_unit(unit: str) -> tuple[str, str, str] | None:
    """Return ``(prefix, body, suffix)`` for mass based units, or ``None``."""
    normalized = _normalize_unit_label(unit)
    suffix = ""
    if normalized.endswith("_per_yr"):
        normalized = normalized.removesuffix("_per_yr")
        suffix = "_per_yr"
    for prefix in _MASS_PREFIXES:
        if not normalized.startswith(prefix):
            continue
        body = normalized[len(prefix) :]
        if not body:
            return prefix, "", suffix
        return prefix, body, suffix
    return None


def try_unit_conversion(
    source_unit: str,
    target_unit: str,
) -> float | None:
    """Return a multiplicative conversion factor for known unit pairs."""
    key = (_normalize_unit_label(source_unit), _normalize_unit_label(target_unit))
    if key[0] == key[1]:
        return 1.0
    source_mass = _split_mass_unit(source_unit)
    target_mass = _split_mass_unit(str(target_unit))
    if source_mass is None or target_mass is None:
        return None
    source_prefix, source_body, _source_suffix = source_mass
    target_prefix, target_body, _target_suffix = target_mass
    if source_body != target_body:
        return None
    return _MASS_SCALES_IN_KG[source_prefix] / _MASS_SCALES_IN_KG[target_prefix]
