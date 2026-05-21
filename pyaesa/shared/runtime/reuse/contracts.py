"""Deterministic scope reuse checks shared by public families."""

from typing import Any, Protocol

from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens


class AsoccPersistedScopeView(Protocol):
    """Minimal deterministic aSoCC persisted scope view."""

    @property
    def compute_signature(self) -> Any: ...

    @property
    def ssp_scenarios(self) -> list[str]: ...


def normalize_selector_payload(
    values: Any,
    *,
    context: str = "Selector payload",
) -> dict[str, tuple[str, ...]]:
    """Return selector payload values as sorted tuples of strings."""
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"{context} must be a mapping.")
    normalized: dict[str, tuple[str, ...]] = {}
    for key, raw in values.items():
        column = str(key).strip()
        if not column:
            continue
        normalized[column] = _normalize_optional_str_tuple(raw)
    return normalized


def normalize_signature_selectors(signature: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    """Return selector payload stored inside a deterministic run signature."""
    values = signature.get("selectors") or {}
    return {
        str(key).strip(): _normalize_optional_str_tuple(raw)
        for key, raw in values.items()
        if str(key).strip()
    }


def asocc_signature_matches_request(
    *,
    requested_signature: dict[str, Any],
    scope: AsoccPersistedScopeView,
    run_ssp_scenarios: list[str] | None,
    check_ssp: bool = True,
) -> bool:
    """Return whether one deterministic aSoCC scope can satisfy one request."""
    return asocc_signature_payload_matches_request(
        requested_signature=requested_signature,
        persisted_signature=scope.compute_signature.as_dict(),
        ssp_scenarios=scope.ssp_scenarios,
        run_ssp_scenarios=run_ssp_scenarios,
        check_ssp=check_ssp,
    )


def asocc_signature_payload_matches_request(
    *,
    requested_signature: dict[str, Any],
    persisted_signature: dict[str, Any],
    ssp_scenarios: list[str] | None,
    run_ssp_scenarios: list[str] | None,
    check_ssp: bool = True,
) -> bool:
    """Return whether one persisted aSoCC signature can satisfy one request."""
    candidate_signature = persisted_signature
    exact_keys = (
        "source",
        "group_version",
        "group_reg",
        "group_sec",
        "fu_code",
        "studied_indices_tag",
        "l1_reg_aggreg",
        "variant_tag",
        "aggreg_indices",
        "projection_mode",
    )
    if _signature_view(candidate_signature, exact_keys) != _signature_view(
        requested_signature,
        exact_keys,
    ):
        return False
    # A None request follows the function default and may match a prior resolved reg_window.
    requested_reg_window = requested_signature.get("reg_window")
    if requested_reg_window is not None and _canonical(
        candidate_signature.get("reg_window")
    ) != _canonical(requested_reg_window):
        return False
    if not _optional_subset(
        requested=requested_signature.get("lcia_methods"),
        available=candidate_signature.get("lcia_methods"),
    ):
        return False
    # None year selectors follow function defaults; explicit values must be covered.
    if not _optional_subset(
        requested=requested_signature.get("reference_years_input"),
        available=candidate_signature.get("reference_years_input"),
        numeric=True,
    ):
        return False
    if not _optional_subset(
        requested=requested_signature.get("l2_reuse_years"),
        available=candidate_signature.get("l2_reuse_years"),
        numeric=True,
    ):
        return False
    if check_ssp:
        requested_ssp = _normalize_ssp_set(requested_signature.get("ssp_scenario_input"))
        candidate_ssp = _normalize_ssp_set(candidate_signature.get("ssp_scenario_input"))
        if candidate_ssp is None:
            candidate_ssp = _normalize_ssp_set(ssp_scenarios) or _normalize_ssp_set(
                run_ssp_scenarios
            )
        if requested_ssp is not None and not requested_ssp.issubset(candidate_ssp or set()):
            return False
    requested_methods = requested_signature.get("selected_methods")
    candidate_methods = candidate_signature.get("selected_methods")
    if isinstance(requested_methods, dict):
        if not isinstance(candidate_methods, dict):
            return False
        for bucket, requested_values in requested_methods.items():
            requested_set = _normalize_str_set(requested_values) or set()
            candidate_set = _normalize_str_set(candidate_methods.get(bucket)) or set()
            if not requested_set.issubset(candidate_set):
                return False
    return True


def io_lca_signature_compatible(
    *,
    signature: dict[str, Any],
    project_name: str,
    source: str,
    group_reg: bool,
    group_sec: bool,
    group_version: str | None,
    fu_code: str,
    aggreg_indices: bool,
    output_format: str,
    requested_years: set[int],
    requested_methods: set[str],
    requested_selectors: dict[str, tuple[str, ...]],
) -> tuple[bool, tuple[int, int]]:
    """Return compatibility and score for one deterministic IO-LCA signature."""
    exact = {
        "project_name": project_name,
        "source": source,
        "group_reg": bool(group_reg),
        "group_sec": bool(group_sec),
        "group_version": group_version,
        "fu_code": fu_code,
        "aggreg_indices": bool(aggreg_indices),
        "output_format": output_format,
    }
    for key, expected in exact.items():
        if signature.get(key) != expected:
            return False, (0, 0)
    years = {int(year) for year in signature.get("years", [])}
    methods = {str(method) for method in signature.get("lcia_methods", [])}
    if not requested_years.issubset(years) or not requested_methods.issubset(methods):
        return False, (0, 0)
    selectors = normalize_signature_selectors(signature)
    for column, requested in requested_selectors.items():
        available = set(selectors.get(column, tuple()))
        if requested and not set(requested).issubset(available):
            return False, (0, 0)
    excess_years = len(years - requested_years)
    excess_methods = len(methods - requested_methods)
    return True, (excess_years, excess_methods)


def _signature_view(signature: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: _canonical(signature.get(key)) for key in keys if key in signature}


def _canonical(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(sorted(_canonical(item) for item in value))
    if isinstance(value, dict):
        return {str(key): _canonical(val) for key, val in sorted(value.items())}
    return value


def _normalize_optional_str_tuple(value: Any) -> tuple[str, ...]:
    values = _normalize_str_set(value)
    return tuple(sorted(values or set()))


def _normalize_str_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return {text} if text else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    return {text} if text else set()


def _normalize_int_set(value: Any) -> set[int] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    return {int(value)}


def _optional_subset(*, requested: Any, available: Any, numeric: bool = False) -> bool:
    normalizer = _normalize_int_set if numeric else _normalize_str_set
    requested_set = normalizer(requested)
    available_set = normalizer(available)
    if requested_set is None:
        return True
    if available_set is None:
        return False
    return requested_set.issubset(available_set)


def _normalize_ssp_set(value: Any) -> set[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, (list, tuple, set)):
        raw = [str(item) for item in value]
    else:
        raw = [str(value)]
    normalized = normalize_ssp_tokens(raw)
    return set(normalized) if normalized else set()
