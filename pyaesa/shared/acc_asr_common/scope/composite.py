"""Shared composite-scope normalization for aCC and ASR entrypoints."""

from typing import Any

from pyaesa.asocc.methods.lcia_inputs import normalize_lcia_methods
from pyaesa.asocc.orchestration.projection.config.config import _normalize_projection_mode
from pyaesa.asocc.runtime.request.defaults import DETERMINISTIC_ASOCC_OPTIONAL_DEFAULTS
from pyaesa.asocc.runtime.selection.normalize import normalize_l1_reg_mode_required
from pyaesa.asocc.runtime.selection.resolve import resolve_method_selection
from pyaesa.shared.selectors.time_selectors import (
    normalize_optional_reg_window_selector,
    normalize_optional_year_selector,
)

_ALLOWED_BASE_ASOCC_KEYS = {
    "method_plan",
    "l1_methods",
    "one_step_methods",
    "two_step_methods",
    "l1_l2_pairs",
    "l1_reg_aggreg",
    "reference_years",
    "ssp_scenario",
    "projection_mode",
    "reg_window",
    "l2_reuse_years",
    "include_lcia_based_allocation_methods",
}

_BASE_ASOCC_DEFAULTS: dict[str, Any] = {
    key: DETERMINISTIC_ASOCC_OPTIONAL_DEFAULTS[key]
    for key in (
        "method_plan",
        "l1_methods",
        "one_step_methods",
        "two_step_methods",
        "l1_l2_pairs",
        "l1_reg_aggreg",
        "reference_years",
        "ssp_scenario",
        "projection_mode",
        "reg_window",
        "l2_reuse_years",
    )
}
_BASE_ASOCC_DEFAULTS["include_lcia_based_allocation_methods"] = True


def _require_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"'{name}' must be a boolean. Received {type(value).__name__}.")


def _normalize_optional_selector_list(value: Any, *, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(
            f"'{name}' must be a string or list of strings. Received {type(value).__name__}."
        )
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"'{name}' must contain only strings.")
    if any(not item.strip() for item in values):
        raise ValueError(f"'{name}' must contain only non empty strings.")
    normalized = sorted({item.strip() for item in values})
    return normalized or None


def _normalize_required_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string. Received {type(value).__name__}.")
    text = value.strip()
    if not text:
        raise ValueError(f"'{name}' is required because it selects the requested composite scope.")
    return text


def _normalize_optional_text(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{name}' must be a string when provided.")
    text = value.strip()
    return text or None


def normalize_shared_lcia_methods(lcia_method: str | list[str]) -> list[str]:
    """Normalize the shared composite LCIA method selector."""
    methods = normalize_lcia_methods(lcia_method)
    if not methods:
        raise ValueError("'lcia_method' must include at least one LCIA method.")
    return list(methods)


def normalize_mrio_scope(
    *,
    source: str,
    agg_reg: bool = False,
    agg_sec: bool = False,
    agg_version: str = "",
    group_indices: bool = False,
) -> dict[str, Any]:
    """Normalize shared composite MRIO scope arguments."""
    source_norm = _normalize_required_text(source, name="source")
    agg_reg_norm = _require_bool(agg_reg, name="agg_reg")
    agg_sec_norm = _require_bool(agg_sec, name="agg_sec")
    group_indices_norm = _require_bool(group_indices, name="group_indices")
    return {
        "source": source_norm,
        "agg_reg": agg_reg_norm,
        "agg_sec": agg_sec_norm,
        "agg_version": _normalize_optional_text(
            agg_version,
            name="agg_version",
        ),
        "group_indices": group_indices_norm,
    }


def normalize_base_asocc_args(raw: dict[str, Any] | None, *, fu_code: str) -> dict[str, Any]:
    """Normalize the shared composite base aSoCC selector block."""
    if raw is not None and not isinstance(raw, dict):
        raise ValueError("'base_asocc_args' must be a dictionary when provided.")
    payload = {} if raw is None else dict(raw)
    unknown = sorted(set(payload) - _ALLOWED_BASE_ASOCC_KEYS)
    if unknown:
        raise ValueError(f"'base_asocc_args' contains unknown key(s): {unknown}.")
    out = dict(_BASE_ASOCC_DEFAULTS)
    out.update(payload)
    out["method_plan"] = _normalize_required_text(
        out.get("method_plan"),
        name="base_asocc_args.method_plan",
    ).lower()
    out["l1_methods"] = _normalize_optional_selector_list(
        out.get("l1_methods"),
        name="base_asocc_args.l1_methods",
    )
    out["one_step_methods"] = _normalize_optional_selector_list(
        out.get("one_step_methods"),
        name="base_asocc_args.one_step_methods",
    )
    out["two_step_methods"] = _normalize_optional_selector_list(
        out.get("two_step_methods"),
        name="base_asocc_args.two_step_methods",
    )
    out["l1_l2_pairs"] = _normalize_optional_selector_list(
        out.get("l1_l2_pairs"),
        name="base_asocc_args.l1_l2_pairs",
    )
    out["l1_reg_aggreg"] = normalize_l1_reg_mode_required(
        _normalize_required_text(
            out.get("l1_reg_aggreg"),
            name="base_asocc_args.l1_reg_aggreg",
        )
    )
    out["projection_mode"] = _normalize_projection_mode(out.get("projection_mode"))
    out["include_lcia_based_allocation_methods"] = _require_bool(
        out["include_lcia_based_allocation_methods"],
        name="base_asocc_args.include_lcia_based_allocation_methods",
    )
    out["reference_years"] = normalize_optional_year_selector(
        out.get("reference_years"),
        name="base_asocc_args.reference_years",
    )
    out["reg_window"] = normalize_optional_reg_window_selector(
        out.get("reg_window"),
        name="base_asocc_args.reg_window",
    )
    out["l2_reuse_years"] = normalize_optional_year_selector(
        out.get("l2_reuse_years"),
        name="base_asocc_args.l2_reuse_years",
    )
    resolve_method_selection(
        fu_code=fu_code,
        method_plan=out["method_plan"],
        l1_methods=out["l1_methods"],
        one_step_methods=out["one_step_methods"],
        two_step_methods=out["two_step_methods"],
        l1_l2_pairs=out["l1_l2_pairs"],
    )
    return out


def effective_asocc_lcia_methods(
    *,
    shared_lcia_methods: list[str],
    include_lcia_based_allocation_methods: bool,
) -> list[str] | None:
    """Return the effective aSoCC LCIA methods for one normalized shared request."""
    if not include_lcia_based_allocation_methods:
        return None
    return list(shared_lcia_methods)


def asocc_lcia_methods_from_allocate_args(
    *,
    base_allocate_args: dict[str, Any],
) -> list[str] | None:
    """Return the aSoCC LCIA scope stored in one composite request."""
    methods = base_allocate_args["lcia_method"]
    if methods is None:
        return None
    return [str(method) for method in methods]


def build_composite_base_allocate_args(
    *,
    project_name: str,
    years: int | list[int] | range,
    lcia_method: list[str],
    asocc_lcia_methods: list[str] | None = None,
    fu_code: str,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
    r_c: str | list[str] | None,
    r_f: str | list[str] | None,
    source: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
    group_indices: bool,
    base_asocc_args: dict[str, Any],
) -> dict[str, Any]:
    """Build the composite aCC/ASR deterministic request payload."""
    asocc_args = base_asocc_args
    effective_lcia_methods = list(lcia_method if asocc_lcia_methods is None else asocc_lcia_methods)
    return {
        "project_name": str(project_name).strip(),
        "source": source,
        "agg_reg": agg_reg,
        "agg_sec": agg_sec,
        "agg_version": agg_version,
        "years": years,
        "fu_code": fu_code,
        "r_p": r_p,
        "s_p": s_p,
        "r_c": r_c,
        "r_f": r_f,
        "group_indices": group_indices,
        "method_plan": asocc_args["method_plan"],
        "l1_methods": asocc_args["l1_methods"],
        "one_step_methods": asocc_args["one_step_methods"],
        "two_step_methods": asocc_args["two_step_methods"],
        "l1_l2_pairs": asocc_args["l1_l2_pairs"],
        "l1_reg_aggreg": asocc_args["l1_reg_aggreg"],
        "include_lcia_based_allocation_methods": bool(
            asocc_args["include_lcia_based_allocation_methods"]
        ),
        "lcia_method": effective_asocc_lcia_methods(
            shared_lcia_methods=effective_lcia_methods,
            include_lcia_based_allocation_methods=bool(
                asocc_args["include_lcia_based_allocation_methods"]
            ),
        ),
        "reference_years": asocc_args["reference_years"],
        "ssp_scenario": asocc_args["ssp_scenario"],
        "projection_mode": asocc_args["projection_mode"],
        "reg_window": asocc_args["reg_window"],
        "l2_reuse_years": asocc_args["l2_reuse_years"],
    }


def base_asocc_kwargs_from_allocate_args(*, base_allocate_args: dict[str, Any]) -> dict[str, Any]:
    """Return the public aSoCC request subset from a composite allocate payload."""
    keys = (
        "project_name",
        "source",
        "agg_reg",
        "agg_sec",
        "agg_version",
        "years",
        "fu_code",
        "r_p",
        "s_p",
        "r_c",
        "r_f",
        "group_indices",
        "method_plan",
        "l1_methods",
        "one_step_methods",
        "two_step_methods",
        "l1_l2_pairs",
        "l1_reg_aggreg",
        "lcia_method",
        "reference_years",
        "ssp_scenario",
        "projection_mode",
        "reg_window",
        "l2_reuse_years",
    )
    return {key: base_allocate_args[key] for key in keys}
