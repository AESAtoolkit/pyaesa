"""Shared CC family configuration normalization for composite aCC and ASR."""

import re
from collections.abc import Mapping
from typing import Any

from pyaesa.ar6_cc.deterministic.request.contracts import (
    normalize_emission_type,
    normalize_emissions_mode,
)
from pyaesa.ar6_cc.shared.runtime.signatures import DEFAULT_AR6_CATEGORIES
from pyaesa.shared.lcia.paths import static_cc_csv_path
from pyaesa.shared.lcia.static_cc import read_static_cc, require_static_cc_bounds_available
from pyaesa.shared.selectors.scenarios import DEFAULT_SSP_SCENARIOS, normalize_optional_ssp_selector

_ALLOWED_BASE_CC_KEYS = {"static", "dynamic_ar6"}
_ALLOWED_STATIC_CC_KEYS = {"active", "exclude_max_cc"}
_ALLOWED_DYNAMIC_CC_KEYS = {
    "active",
    "harmonization",
    "harmonization_method",
    "category",
    "ssp_scenario",
    "emission_type",
    "include_afolu",
    "emissions_mode",
    "subset_version",
}
_STATIC_CC_BOUNDS = ("min_cc", "max_cc")
_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def cc_branch_token(*, cc_source: str, cc_type: str) -> str:
    """Return the filesystem safe carrying capacity branch token."""
    source_token = _SAFE_TOKEN_RE.sub("_", str(cc_source).strip()).strip("._-") or "item"
    if str(cc_type).strip().lower() == "static":
        return f"static__{source_token}"
    return f"dynamic_ar6__{source_token}"


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Return one public configuration block as a plain dictionary."""
    if not isinstance(value, Mapping):
        raise ValueError(f"'{name}' must be a dictionary when provided.")
    return dict(value)


def _require_bool(value: Any, *, name: str) -> bool:
    """Return one public boolean without string or numeric coercion."""
    if not isinstance(value, bool):
        raise ValueError(f"'{name}' must be a boolean.")
    return value


def _normalize_text(value: Any, *, name: str, default: str) -> str:
    """Return one public string selector without implicit type conversion."""
    raw = default if value is None else value
    if not isinstance(raw, str):
        raise ValueError(f"'{name}' must be a string.")
    text = raw.strip()
    if not text:
        raise ValueError(f"'{name}' must be a non empty string.")
    return text


def _normalize_optional_text(value: Any, *, name: str) -> str | None:
    """Return one optional public string selector."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string when provided.")
    text = value.strip()
    if not text:
        raise ValueError(f"'{name}' must be a non empty string when provided.")
    return text


def _normalize_static_cc_config(
    static: dict[str, Any],
    *,
    allowed_keys: set[str],
) -> dict[str, Any] | None:
    """Normalize one public static CC block to the canonical internal bounds payload."""
    unknown_static = sorted(set(static) - allowed_keys)
    if unknown_static:
        raise ValueError(f"'base_cc_args.static' contains unknown key(s): {unknown_static}.")
    active = static.get("active", True)
    if not isinstance(active, bool):
        raise ValueError("'base_cc_args.static.active' must be a boolean.")
    if not active:
        return None
    exclude_max_cc = static.get("exclude_max_cc", False)
    if not isinstance(exclude_max_cc, bool):
        raise ValueError("'base_cc_args.static.exclude_max_cc' must be a boolean.")
    return {
        "exclude_max_cc": exclude_max_cc,
        "bounds": ["min_cc"] if exclude_max_cc else list(_STATIC_CC_BOUNDS),
    }


def _normalize_dynamic_cc_config(dynamic: dict[str, Any]) -> dict[str, Any]:
    unknown_dynamic = sorted(set(dynamic) - _ALLOWED_DYNAMIC_CC_KEYS)
    if unknown_dynamic:
        raise ValueError(f"'base_cc_args.dynamic_ar6' contains unknown key(s): {unknown_dynamic}.")
    active = dynamic.get("active", True)
    if not isinstance(active, bool):
        raise ValueError("'base_cc_args.dynamic_ar6.active' must be a boolean.")
    if not active:
        return {}
    raw_categories = dynamic.get("category")
    if raw_categories is None:
        categories = list(DEFAULT_AR6_CATEGORIES)
    elif isinstance(raw_categories, str):
        category = raw_categories.strip()
        if not category:
            raise ValueError("'base_cc_args.dynamic_ar6.category' must be non empty.")
        categories = [category]
    elif isinstance(raw_categories, list):
        if not raw_categories:
            raise ValueError("'base_cc_args.dynamic_ar6.category' must be a non empty list.")
        if not all(isinstance(value, str) for value in raw_categories):
            raise ValueError("'base_cc_args.dynamic_ar6.category' must contain only strings.")
        if any(not value.strip() for value in raw_categories):
            raise ValueError(
                "'base_cc_args.dynamic_ar6.category' must contain only non empty strings."
            )
        categories = sorted({value.strip() for value in raw_categories})
    else:
        raise ValueError("'base_cc_args.dynamic_ar6.category' must be a string or list.")
    raw_ssp = dynamic.get("ssp_scenario", DEFAULT_SSP_SCENARIOS)
    if raw_ssp is None:
        raw_ssp = DEFAULT_SSP_SCENARIOS
    ssp_values = normalize_optional_ssp_selector(
        raw_ssp,
        argument_name="base_cc_args.dynamic_ar6.ssp_scenario",
    )
    return {
        "harmonization": _require_bool(
            dynamic.get("harmonization", True),
            name="base_cc_args.dynamic_ar6.harmonization",
        ),
        "harmonization_method": _normalize_text(
            dynamic.get("harmonization_method"),
            name="base_cc_args.dynamic_ar6.harmonization_method",
            default="offset",
        ),
        "category": categories,
        "ssp_scenario": ssp_values,
        "emission_type": normalize_emission_type(
            _normalize_text(
                dynamic.get("emission_type"),
                name="base_cc_args.dynamic_ar6.emission_type",
                default="kyoto_gases",
            )
        ),
        "include_afolu": _require_bool(
            dynamic.get("include_afolu", False),
            name="base_cc_args.dynamic_ar6.include_afolu",
        ),
        "emissions_mode": normalize_emissions_mode(
            _normalize_text(
                dynamic.get("emissions_mode"),
                name="base_cc_args.dynamic_ar6.emissions_mode",
                default="gross_alt",
            )
        ),
        "subset_version": _normalize_optional_text(
            dynamic.get("subset_version"),
            name="base_cc_args.dynamic_ar6.subset_version",
        ),
    }


def normalize_base_cc_args(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize shared aCC static and dynamic CC configuration."""
    if not isinstance(raw, dict):
        raise ValueError("'base_cc_args' must be a dictionary.")
    unknown = sorted(set(raw) - _ALLOWED_BASE_CC_KEYS)
    if unknown:
        raise ValueError(f"'base_cc_args' contains unknown key(s): {unknown}.")
    out: dict[str, Any] = {}
    static_config = (
        _require_mapping(raw["static"], name="base_cc_args.static")
        if "static" in raw and raw["static"] is not None
        else {}
    )
    static = _normalize_static_cc_config(
        static_config,
        allowed_keys=_ALLOWED_STATIC_CC_KEYS,
    )
    if static is not None:
        out["static"] = static
    if "dynamic_ar6" in raw and raw["dynamic_ar6"] is not None:
        dynamic = _normalize_dynamic_cc_config(
            _require_mapping(raw["dynamic_ar6"], name="base_cc_args.dynamic_ar6")
        )
        if dynamic:
            out["dynamic_ar6"] = dynamic
    if not out:
        raise ValueError("'base_cc_args' must include at least one active family.")
    return out


def require_asr_static_cc_source_compatibility(
    *,
    cc_source: str,
    static_cc_bounds: list[str],
) -> None:
    """Fail when an ASR static request requires max_cc that is not available."""
    if static_cc_bounds != ["min_cc", "max_cc"]:
        return
    cc_df = read_static_cc(static_cc_csv_path(lcia_method=cc_source))
    try:
        require_static_cc_bounds_available(
            cc_df=cc_df,
            requested_bounds=static_cc_bounds,
            context="ASR static carrying capacity request",
        )
    except ValueError as exc:
        raise ValueError(
            "ASR static carrying capacity uses both min_cc and max_cc unless "
            "'base_cc_args.static.exclude_max_cc=True'. Switch 'exclude_max_cc' to True "
            "or add numeric max_cc values for every impact in the static carrying "
            "capacity CSV."
        ) from exc
