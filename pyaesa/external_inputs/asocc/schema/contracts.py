"""Shared selector and filename contracts for external aSoCC inputs."""

from dataclasses import dataclass
from typing import Any, cast

from pyaesa.asocc.methods.registry.registry import normalize_fu_code
from pyaesa.asocc.runtime.methods.labels import (
    parse_raw_asocc_method_label,
)

_EXTERNAL_METHOD_KEYS = {"l1_methods", "one_step_methods", "l1_l2_pairs"}


def _normalize_optional_string_list(value: Any, *, name: str) -> list[str] | None:
    """Normalize optional string or list-of-string input."""
    if value is None:
        return None
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(
            f"'{name}' must be a string or list of strings when provided. "
            f"Received {type(value).__name__}."
        )
    invalid_types = sorted({type(item).__name__ for item in values if not isinstance(item, str)})
    if invalid_types:
        raise ValueError(
            f"'{name}' must contain only string values when provided. "
            f"Received item type(s): {invalid_types}."
        )
    out = sorted({str(item).strip() for item in values if str(item).strip()})
    return out or None


def _validate_external_method_label(method_label: str, *, name: str) -> str:
    """Validate one external method label against the scientific label contract."""
    text = str(method_label).strip()
    if not text:
        raise ValueError(f"'{name}' must contain non empty method labels when provided.")
    if "::" in text:
        raise ValueError(
            f"'{name}' method labels must not contain '::'. Use "
            "'<l1_method>::<l2_method>' only inside the pair-list selector."
        )
    try:
        parse_raw_asocc_method_label(text)
    except ValueError as exc:
        raise ValueError(
            f"'{name}' method labels must use the scientific method-label form "
            "used by inter-method tree aggregation, for example 'CO(S)', "
            "'CO-HR(S,cum)', or 'AR(E^{CBA_FD})'. "
            f"Received {text!r}."
        ) from exc
    return text


def _normalize_external_method_names(
    value: list[str] | None,
    *,
    name: str,
) -> list[str] | None:
    """Validate one optional external method-name list."""
    if value is None:
        return None
    return sorted({_validate_external_method_label(item, name=name) for item in value})


def normalize_external_method_selector(
    raw: dict[str, Any] | None,
    *,
    fu_code: str,
    argument_name: str = "external_method",
) -> dict[str, Any] | None:
    """Normalize the shared external aSoCC method-selector block."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"'{argument_name}' must be a dictionary when provided. "
            "Use selector keys such as 'l1_methods', 'one_step_methods', or 'l1_l2_pairs'."
        )
    unknown = sorted(set(raw) - _EXTERNAL_METHOD_KEYS)
    if unknown:
        raise ValueError(
            f"'{argument_name}' contains unknown key(s): {unknown}. Supported keys are "
            "'l1_methods', 'one_step_methods', and 'l1_l2_pairs'."
        )
    out = {
        "l1_methods": _normalize_external_method_names(
            _normalize_optional_string_list(
                raw.get("l1_methods"),
                name=f"{argument_name}.l1_methods",
            ),
            name=f"{argument_name}.l1_methods",
        ),
        "one_step_methods": _normalize_external_method_names(
            _normalize_optional_string_list(
                raw.get("one_step_methods"),
                name=f"{argument_name}.one_step_methods",
            ),
            name=f"{argument_name}.one_step_methods",
        ),
        "l1_l2_pairs": _normalize_optional_string_list(
            raw.get("l1_l2_pairs"),
            name=f"{argument_name}.l1_l2_pairs",
        ),
    }
    fu_norm = normalize_fu_code(fu_code)
    if fu_norm.startswith("L1."):
        if out["one_step_methods"] or out["l1_l2_pairs"]:
            raise ValueError(
                f"'{argument_name}.one_step_methods' and '{argument_name}.l1_l2_pairs' are "
                "not accepted for L1 functional units. Use only "
                f"'{argument_name}.l1_methods'."
            )
    elif out["l1_methods"]:
        raise ValueError(
            f"'{argument_name}.l1_methods' is not accepted for L2 functional units. "
            f"Use '{argument_name}.one_step_methods' and/or "
            f"'{argument_name}.l1_l2_pairs' for final outputs."
        )
    for pair in out["l1_l2_pairs"] or []:
        if str(pair).count("::") != 1:
            raise ValueError(
                f"'{argument_name}.l1_l2_pairs' entries must use the form "
                "'<l1_method>::<l2_method>'."
            )
        left, right = [piece.strip() for piece in str(pair).split("::", 1)]
        if not left or not right:
            raise ValueError(
                f"'{argument_name}.l1_l2_pairs' entries must use the form "
                "'<l1_method>::<l2_method>'."
            )
    if not any(out.values()):
        raise ValueError(
            f"'{argument_name}' must include at least one selector key: "
            "'l1_methods', 'one_step_methods', or 'l1_l2_pairs'. "
            "Provide the selector that matches the requested functional-unit level."
        )
    return {key: value for key, value in out.items() if value is not None}


@dataclass(frozen=True)
class ExternalMethodSelection:
    """Normalized one external deterministic aSoCC method selection."""

    fu_code: str
    l2_method: str | None
    l1_method: str | None
    level: str

    @property
    def l1_l2_method(self) -> str:
        """Return the canonical combined method label."""
        if self.level == "level_1":
            return str(self.l1_method)
        if self.l1_method is None:
            return str(self.l2_method)
        return f"{self.l1_method}_{self.l2_method}"

    @property
    def file_method_token(self) -> str:
        """Return the single method token used in external file naming."""
        if self.level == "level_1":
            return str(self.l1_method)
        return str(self.l2_method)

    @property
    def asocc_method_label(self) -> str:
        """Return the internal combined label used in output columns and folder names."""
        return self.l1_l2_method

    @property
    def user_label(self) -> str:
        """Return the declaration-form label for error messages."""
        if self.level == "level_1":
            return str(self.l1_method)
        if self.l1_method is None:
            return str(self.l2_method)
        return f"{self.l1_method}::{self.l2_method}"


def external_selection_labels(
    *,
    external_method: dict[str, Any] | None,
    fu_code: str,
) -> list[str]:
    """Return normalized external selection labels."""
    if external_method is None:
        return []
    return [
        selection.asocc_method_label
        for selection in iter_external_method_selections(
            external_method=external_method,
            fu_code=fu_code,
        )
    ]


def validate_external_method_collisions(
    *,
    native_labels: list[str] | None,
    external_method: dict[str, Any] | None,
    fu_code: str,
    where: str,
) -> None:
    """Fail when effective native and external method labels overlap."""
    if not native_labels or external_method is None:
        return
    collisions = sorted(
        set(str(value) for value in native_labels)
        & set(external_selection_labels(external_method=external_method, fu_code=fu_code))
    )
    if collisions:
        raise ValueError(
            f"{where} resolves native and external methods with overlapping labels: "
            f"{collisions}. Rename the external method(s) or change the native selection "
            "so one run never contains duplicate final method labels."
        )


def iter_external_method_selections(
    *,
    external_method: dict[str, Any],
    fu_code: str,
) -> tuple[ExternalMethodSelection, ...]:
    """Return normalized external selections in deterministic iteration order."""
    normalized = cast(
        dict[str, Any],
        normalize_external_method_selector(
            external_method,
            fu_code=fu_code,
            argument_name="external_method",
        ),
    )
    fu_norm = normalize_fu_code(fu_code)
    if fu_norm.startswith("L1."):
        return tuple(
            ExternalMethodSelection(
                fu_code=fu_norm,
                l2_method=None,
                l1_method=l1_method_label,
                level="level_1",
            )
            for l1_method_label in normalized.get("l1_methods", [])
        )
    selections: list[ExternalMethodSelection] = []
    for l2_method_label in normalized.get("one_step_methods", []):
        selections.append(
            ExternalMethodSelection(
                fu_code=fu_norm,
                l2_method=l2_method_label,
                l1_method=None,
                level="level_2",
            )
        )
    for pair in normalized.get("l1_l2_pairs", []):
        pieces = [piece.strip() for piece in str(pair).split("::", 1)]
        l1_method = _validate_external_method_label(
            pieces[0],
            name="external_method.l1_l2_pairs",
        )
        l2_method = _validate_external_method_label(
            pieces[1],
            name="external_method.l1_l2_pairs",
        )
        selections.append(
            ExternalMethodSelection(
                fu_code=fu_norm,
                l2_method=l2_method,
                l1_method=l1_method,
                level="level_2",
            )
        )
    return tuple(selections)


def merge_external_selector_methods(
    *,
    target_selector: dict[str, Any] | None,
    external_method: dict[str, Any] | None,
    fu_code: str,
) -> dict[str, Any] | None:
    """Return a target selector whose method scope includes external selections."""
    if target_selector is None:
        return None
    out = dict(target_selector)
    if external_method is None:
        return out
    labels = [
        selection.asocc_method_label
        for selection in iter_external_method_selections(
            external_method=external_method,
            fu_code=fu_code,
        )
    ]
    methods = out.get("methods")
    if methods is None:
        return out
    merged = sorted({*(str(value) for value in methods), *labels})
    out["methods"] = merged
    return out
