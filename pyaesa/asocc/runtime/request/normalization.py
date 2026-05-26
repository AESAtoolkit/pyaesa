"""Normalization for explicit deterministic aSoCC request scopes."""

from typing import Any

from pyaesa.asocc.runtime.paths.family_roots import (
    effective_agg_flags_for_source,
    effective_agg_version_for_source,
    is_native_asocc_source,
)
from pyaesa.asocc.methods.lcia_inputs import normalize_lcia_methods
from pyaesa.asocc.methods.registry.registry import normalize_fu_code
from pyaesa.asocc.orchestration.projection.config.config import (
    _normalize_projection_mode,
    _normalize_year_selector,
)
from pyaesa.asocc.runtime.request.defaults import (
    UNCERTAINTY_BASE_ALLOCATE_DEFAULTS,
)
from pyaesa.shared.selectors.time_selectors import normalize_reg_window_for_storage

_FORBIDDEN_BASE_ALLOCATE_KEYS = {
    "output_format",
    "intermediate_outputs",
    "refresh",
}
_BASE_ALLOCATE_DEFAULTS = dict(UNCERTAINTY_BASE_ALLOCATE_DEFAULTS)
_ALLOWED_BASE_ALLOCATE_KEYS = set(_BASE_ALLOCATE_DEFAULTS) | {"project_name", "source", "fu_code"}


def _normalize_optional_string_list(value: Any) -> list[str] | None:
    """Normalize one optional string or string-list payload."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    if isinstance(value, list):
        cleaned = sorted({str(item).strip() for item in value if str(item).strip()})
        return cleaned or None
    text = str(value).strip()
    return [text] if text else None


def _normalize_optional_string(value: Any) -> str | None:
    """Normalize an optional string argument."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_non_empty_string(value: Any, *, name: str) -> str:
    """Normalize one required non-empty string argument."""
    if value is None:
        raise ValueError(f"'{name}' must be a non-empty string.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"'{name}' must be a non-empty string.")
    return text


def _require_bool(value: Any, *, name: str) -> bool:
    """Require one strict boolean value."""
    if isinstance(value, bool):
        return value
    raise TypeError(f"'{name}' must be a boolean. Received {type(value).__name__}.")


def _normalize_optional_years(value: Any, *, name: str) -> list[int] | None:
    """Normalize an optional year selector."""
    if value is None:
        return None
    years = _normalize_year_selector(value=value, name=name)
    return years or None


def _normalize_group_indices(
    value: Any,
    *,
    name: str = "base_allocate_args.group_indices",
) -> bool:
    """Normalize the grouped output selector."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() == "both":
        raise ValueError(
            f"'{name}' must resolve one deterministic output scope. Use True or False, not 'both'."
        )
    raise ValueError(f"'{name}' must be a boolean.")


def _normalize_l1_reg_aggreg(
    value: Any,
    *,
    name: str = "base_allocate_args.l1_reg_aggreg",
) -> str:
    """Normalize the L1 aggregation selector."""
    if isinstance(value, list):
        raise ValueError(
            f"'{name}' must resolve one deterministic output scope. Use 'pre' or 'post', "
            "not a list."
        )
    normalized = _require_non_empty_string(value, name=name).lower()
    if normalized == "both":
        raise ValueError(
            f"'{name}' must resolve one deterministic output scope. Use 'pre' or 'post', "
            "not 'both'."
        )
    if normalized not in {"pre", "post"}:
        raise ValueError(f"'{name}' must be either 'pre' or 'post'.")
    return normalized


def normalize_source_aggregation_scope(
    *,
    source: Any,
    agg_reg: Any,
    agg_sec: Any,
    agg_version: Any,
    source_name: str = "base_allocate_args.source",
    agg_reg_name: str = "base_allocate_args.agg_reg",
    agg_sec_name: str = "base_allocate_args.agg_sec",
    agg_version_name: str = "base_allocate_args.agg_version",
) -> tuple[str, bool, bool, str | None]:
    """Normalize one published aSoCC source and its aggregation controls."""
    source_clean = _require_non_empty_string(source, name=source_name)
    normalized_agg_reg = False if agg_reg is None else _require_bool(agg_reg, name=agg_reg_name)
    normalized_agg_sec = False if agg_sec is None else _require_bool(agg_sec, name=agg_sec_name)
    normalized_agg_version = _normalize_optional_string(agg_version)
    if not is_native_asocc_source(source=source_clean) and (
        normalized_agg_version is not None or normalized_agg_reg or normalized_agg_sec
    ):
        raise ValueError(
            "Disaggregated published aSoCC sources must be called directly without "
            "aggregation controls (agg_reg/agg_sec/agg_version)."
        )
    published_agg_version = effective_agg_version_for_source(
        source=source_clean,
        agg_version=normalized_agg_version,
    )
    published_agg_reg, published_agg_sec = effective_agg_flags_for_source(
        source=source_clean,
        agg_reg=normalized_agg_reg,
        agg_sec=normalized_agg_sec,
    )
    return source_clean, published_agg_reg, published_agg_sec, published_agg_version


def normalize_deterministic_scope_args(
    raw: dict[str, Any],
    *,
    payload_name: str = "base_allocate_args",
) -> dict[str, Any]:
    """Normalize deterministic aSoCC scope selectors shared across public entrypoints."""
    if not isinstance(raw, dict):
        raise ValueError(f"'{payload_name}' must be a dictionary.")
    normalized: dict[str, Any] = {
        "project_name": _require_non_empty_string(
            raw.get("project_name"),
            name=f"{payload_name}.project_name",
        ),
        "fu_code": normalize_fu_code(
            _require_non_empty_string(
                raw.get("fu_code"),
                name=f"{payload_name}.fu_code",
            )
        ),
        "group_indices": _normalize_group_indices(
            raw.get("group_indices", False),
            name=f"{payload_name}.group_indices",
        ),
        "l1_reg_aggreg": _normalize_l1_reg_aggreg(
            raw.get("l1_reg_aggreg", "post"),
            name=f"{payload_name}.l1_reg_aggreg",
        ),
    }
    (
        normalized["source"],
        normalized["agg_reg"],
        normalized["agg_sec"],
        normalized["agg_version"],
    ) = normalize_source_aggregation_scope(
        source=raw.get("source"),
        agg_reg=raw.get("agg_reg"),
        agg_sec=raw.get("agg_sec"),
        agg_version=raw.get("agg_version"),
        source_name=f"{payload_name}.source",
        agg_reg_name=f"{payload_name}.agg_reg",
        agg_sec_name=f"{payload_name}.agg_sec",
        agg_version_name=f"{payload_name}.agg_version",
    )
    return normalized


def normalize_base_allocate_args(
    raw: dict[str, Any],
    *,
    additional_forbidden_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Normalize base allocate arguments for one explicit aSoCC scope."""
    from pyaesa.asocc.runtime.selection.resolve import resolve_method_selection

    if not isinstance(raw, dict):
        raise ValueError(
            "'base_asocc_args' must be a dictionary identifying the deterministic "
            "aSoCC prerequisite scope."
        )
    forbidden_keys = _FORBIDDEN_BASE_ALLOCATE_KEYS | (
        set() if additional_forbidden_keys is None else set(additional_forbidden_keys)
    )
    unknown = sorted(set(raw) - _ALLOWED_BASE_ALLOCATE_KEYS - forbidden_keys)
    if unknown:
        raise ValueError(
            "'base_asocc_args' contains unknown key(s): "
            f"{unknown}. Use only aSoCC envelope keys in this block."
        )
    forbidden = sorted(set(raw) & forbidden_keys)
    if forbidden:
        raise ValueError(
            "'base_asocc_args' contains forbidden key(s): "
            f"{forbidden}. Use the top-level runtime arguments instead of "
            "placing these controls inside the base aSoCC envelope."
        )
    out = dict(_BASE_ALLOCATE_DEFAULTS)
    out.update(raw)
    out.update(normalize_deterministic_scope_args(out))
    out["years"] = _normalize_optional_years(out.get("years"), name="base_allocate_args.years")
    out["r_p"] = _normalize_optional_string_list(out.get("r_p"))
    out["s_p"] = _normalize_optional_string_list(out.get("s_p"))
    out["r_c"] = _normalize_optional_string_list(out.get("r_c"))
    out["r_f"] = _normalize_optional_string_list(out.get("r_f"))
    out["method_plan"] = _require_non_empty_string(
        out.get("method_plan"),
        name="base_allocate_args.method_plan",
    )
    out["l1_methods"] = _normalize_optional_string_list(out.get("l1_methods"))
    out["one_step_methods"] = _normalize_optional_string_list(out.get("one_step_methods"))
    out["two_step_methods"] = _normalize_optional_string_list(out.get("two_step_methods"))
    out["l1_l2_pairs"] = _normalize_optional_string_list(out.get("l1_l2_pairs"))
    out["lcia_method"] = normalize_lcia_methods(out.get("lcia_method"))
    out["reference_years"] = _normalize_optional_years(
        out.get("reference_years"),
        name="base_allocate_args.reference_years",
    )
    out["ssp_scenario"] = _normalize_optional_string_list(out.get("ssp_scenario"))
    out["projection_mode"] = _normalize_projection_mode(out.get("projection_mode"))
    out["reg_window"] = normalize_reg_window_for_storage(
        out.get("reg_window"),
        name="base_allocate_args.reg_window",
    )
    out["l2_reuse_years"] = _normalize_optional_years(
        out.get("l2_reuse_years"),
        name="base_allocate_args.l2_reuse_years",
    )
    resolve_method_selection(
        fu_code=out["fu_code"],
        method_plan=out["method_plan"],
        l1_methods=out["l1_methods"],
        one_step_methods=out["one_step_methods"],
        two_step_methods=out["two_step_methods"],
        l1_l2_pairs=out["l1_l2_pairs"],
    )
    return out
