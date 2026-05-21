"""Shared figure-request validation helpers."""

from typing import Any

from pyaesa.shared.figures.contracts import normalize_figure_output_format, validate_figure_dpi
from pyaesa.shared.figures.generation_policy import resolve_polar_years

_STYLE_VALUES = {"violin", "whisker", "both"}


def normalize_figure_format(figure_format: dict[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize one public figure-format dictionary."""
    payload = {} if figure_format is None else dict(figure_format)
    unknown = sorted(set(payload) - {"format", "dpi"})
    if unknown:
        raise ValueError(
            f"figure_format contains unsupported keys {unknown}. "
            "Supported keys are ['dpi', 'format']."
        )
    return {
        "format": normalize_figure_output_format(
            payload.get("format", "png"),
            argument_name="figure_format.format",
        ),
        "dpi": validate_figure_dpi(payload.get("dpi", 500)),
    }


def normalize_figure_options(
    figure_options: dict[str, Any] | None,
    *,
    allow_single_year_style: bool,
    allow_polar_years: bool,
    allow_polar_style: bool = False,
    allow_per_method: bool = False,
    allow_multi_method: bool = False,
    allow_inter_method: bool = False,
    allow_polar: bool = False,
    allow_nested_polar_style: bool = True,
    argument_name: str = "figure_options",
) -> dict[str, Any]:
    """Validate one public figure-options dictionary."""
    payload = {} if figure_options is None else dict(figure_options)
    allowed = set()
    if allow_single_year_style:
        allowed.add("single_year_style")
    if allow_polar_years:
        allowed.add("polar_years")
    if allow_polar_style:
        allowed.add("polar_style")
    if allow_per_method:
        allowed.add("per_method")
    if allow_multi_method:
        allowed.add("multi_method")
    if allow_inter_method:
        allowed.add("inter_method")
    if allow_polar:
        allowed.add("polar")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"{argument_name} contains unsupported keys {unknown}. "
            f"Supported keys are {sorted(allowed)}."
        )
    normalized: dict[str, Any] = {}
    for key, enabled in (
        ("per_method", allow_per_method),
        ("multi_method", allow_multi_method),
        ("inter_method", allow_inter_method),
    ):
        if enabled:
            normalized[key] = normalize_figure_bool(
                payload.get(key, True),
                argument_name=f"{argument_name}.{key}",
            )
    if allow_single_year_style and payload.get("single_year_style") is not None:
        normalized["single_year_style"] = normalize_single_year_style(
            payload["single_year_style"],
            argument_name=f"{argument_name}.single_year_style",
        )
    if allow_polar_years and payload.get("polar_years") is not None:
        normalized["polar_years"] = normalize_polar_years(
            payload["polar_years"],
            argument_name=f"{argument_name}.polar_years",
        )
    if allow_polar_style and payload.get("polar_style") is not None:
        normalized["polar_style"] = normalize_single_year_style(
            payload["polar_style"],
            argument_name=f"{argument_name}.polar_style",
        )
    if allow_polar:
        normalized["polar"] = normalize_polar_options(
            payload.get("polar"),
            argument_name=f"{argument_name}.polar",
            allow_polar_style=allow_nested_polar_style,
        )
    return normalized


def validate_consecutive_multi_year_figure_request(
    requested_years: list[int],
    *,
    family_label: str,
) -> None:
    """Reject multi-year figure requests whose studied years are not consecutive."""
    years = sorted({int(year) for year in requested_years})
    if len(years) <= 1:
        return
    expected = list(range(years[0], years[-1] + 1))
    if years == expected:
        return
    requested = ", ".join(str(year) for year in years)
    missing = ", ".join(str(year) for year in expected if year not in set(years))
    raise ValueError(
        f"{family_label} multi-year figures require consecutive requested years. "
        f"Requested years: [{requested}]. Missing years in the requested period: [{missing}]. "
        "Request a consecutive year range, or disable figure generation for this call."
    )


def normalize_subfigure_options(
    subfigure_options: dict[str, Any] | None,
    *,
    allow_single_year_style: bool = True,
) -> dict[str, Any]:
    """Validate one public subfigure-options dictionary."""
    return normalize_figure_options(
        subfigure_options,
        allow_single_year_style=allow_single_year_style,
        allow_polar_years=False,
        argument_name="subfigure_options",
    )


def normalize_single_year_style(value: Any, *, argument_name: str) -> str:
    """Validate one single-year style selector."""
    style = str(value).strip().lower()
    if style not in _STYLE_VALUES:
        raise ValueError(f"{argument_name} must be one of {sorted(_STYLE_VALUES)}, got '{value}'.")
    return style


def normalize_figure_bool(value: Any, *, argument_name: str) -> bool:
    """Validate one figure product boolean selector."""
    if not isinstance(value, bool):
        raise ValueError(f"{argument_name} must be a boolean. Use True or False.")
    return bool(value)


def normalize_polar_options(
    value: Any,
    *,
    argument_name: str,
    allow_polar_style: bool = True,
) -> dict[str, Any]:
    """Validate nested ASR polar figure options."""
    payload = {} if value is None else dict(value) if isinstance(value, dict) else None
    if payload is None:
        raise ValueError(f"{argument_name} must be a dictionary.")
    supported = {"active", "polar_years"}
    if allow_polar_style:
        supported.add("polar_style")
    unknown = sorted(set(payload) - supported)
    if unknown:
        raise ValueError(
            f"{argument_name} contains unsupported keys {unknown}. "
            f"Supported keys are {sorted(supported)}."
        )
    normalized = {
        "active": normalize_figure_bool(
            payload.get("active", True),
            argument_name=f"{argument_name}.active",
        ),
        "polar_years": None
        if payload.get("polar_years") is None
        else normalize_polar_years(
            payload["polar_years"],
            argument_name=f"{argument_name}.polar_years",
        ),
    }
    if allow_polar_style:
        normalized["polar_style"] = normalize_single_year_style(
            payload.get("polar_style", "violin"),
            argument_name=f"{argument_name}.polar_style",
        )
    return normalized


def resolve_nested_polar_years(
    *,
    studied_years: list[int],
    polar: dict[str, Any],
    argument_name: str,
) -> list[int]:
    """Resolve active nested polar years using the shared ASR polar policy."""
    if not bool(polar["active"]):
        return []
    return resolve_polar_years(
        studied_years=studied_years,
        user_override=polar["polar_years"],
        argument_name=f"{argument_name}.polar_years",
    )


def normalize_polar_years(value: Any, *, argument_name: str) -> list[int]:
    """Validate one explicit ASR polar-year list."""
    if not isinstance(value, list):
        raise ValueError(f"{argument_name} must be a list of years.")
    years = sorted({int(item) for item in value})
    return years
