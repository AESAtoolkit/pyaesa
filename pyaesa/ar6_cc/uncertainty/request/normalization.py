"""Public request normalization for AR6 CC uncertainty."""

from typing import Any, Mapping

from pyaesa.ar6_cc.deterministic.request.contracts import (
    normalize_emission_type,
    normalize_emissions_mode,
)
from pyaesa.ar6_cc.shared.runtime.signatures import (
    normalize_cc_category,
    normalize_cc_ssp_scenario,
)
from pyaesa.process.ar6.utils.pipeline.runtime_helpers import validate_harmonization_method
from pyaesa.process.ar6.utils.pipeline.study_period import resolve_study_period

from pyaesa.ar6_cc.uncertainty.runtime.models import AR6CCUncertaintyRequest

AR6_DYNAMIC_CC_SOURCE = "dynamic_ar6_cc_uncertainty"
AR6_CC_UNCERTAINTY_SOURCES = (AR6_DYNAMIC_CC_SOURCE,)
AR6_CC_SAMPLING_METHODS = ("srs", "lhs")

_ALLOWED_BASE_KEYS = {
    "years",
    "harmonization",
    "harmonization_method",
    "category",
    "ssp_scenario",
    "emission_type",
    "include_afolu",
    "emissions_mode",
    "subset_version",
}


def normalize_ar6_cc_uncertainty_request(
    *,
    base_ar6_cc_args: Mapping[str, Any],
    source_parameters: Mapping[str, Any],
) -> AR6CCUncertaintyRequest:
    """Normalize deterministic AR6 CC selectors and source parameters."""
    payload = dict(base_ar6_cc_args)
    unknown = sorted(set(payload) - _ALLOWED_BASE_KEYS)
    if unknown:
        raise ValueError(f"Unsupported base_ar6_cc_args keys for uncertainty_ar6_cc: {unknown}.")
    if "years" not in payload:
        raise ValueError("base_ar6_cc_args.years is required.")

    study_period = resolve_study_period(payload["years"])
    years = list(range(int(study_period[0]), int(study_period[1]) + 1))
    harmonization = _bool_value(
        payload.get("harmonization", True),
        field="base_ar6_cc_args.harmonization",
    )
    harmonization_method = validate_harmonization_method(
        harmonization=harmonization,
        harmonization_method=str(payload.get("harmonization_method", "offset")),
    )
    category = normalize_cc_category(payload.get("category"))
    ssp_scenario = normalize_cc_ssp_scenario(payload.get("ssp_scenario"))
    emission_type = normalize_emission_type(str(payload.get("emission_type", "kyoto_gases")))
    include_afolu = _bool_value(
        payload.get("include_afolu", False),
        field="base_ar6_cc_args.include_afolu",
    )
    emissions_mode = normalize_emissions_mode(str(payload.get("emissions_mode", "gross_alt")))
    subset_version = _optional_text(
        payload.get("subset_version"),
        field="base_ar6_cc_args.subset_version",
    )
    normalized_source = normalize_ar6_cc_source_parameters(parameters=source_parameters)
    base_args = {
        "years": range(study_period[0], study_period[1] + 1),
        "harmonization": harmonization,
        "harmonization_method": harmonization_method,
        "category": category,
        "ssp_scenario": ssp_scenario,
        "emission_type": emission_type,
        "include_afolu": include_afolu,
        "emissions_mode": emissions_mode,
        "subset_version": subset_version,
    }
    deterministic_args = {
        **base_args,
        "output_format": "csv",
        "figures": False,
        "figure_format": {"format": "png", "dpi": 500},
    }
    return AR6CCUncertaintyRequest(
        base_ar6_cc_args={**base_args, "years": list(years)},
        deterministic_args=deterministic_args,
        source_parameters=normalized_source,
        study_period=study_period,
        years=years,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        category=category,
        ssp_scenario=ssp_scenario,
        emission_type=emission_type,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode,
        subset_version=subset_version,
    )


def normalize_ar6_cc_source_parameters(*, parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the AR6 CC dynamic trajectory source parameters."""
    payload = dict(parameters)
    sampling_method = str(payload.pop("sampling_method", "srs")).strip().lower()
    if sampling_method not in AR6_CC_SAMPLING_METHODS:
        raise ValueError(f"{AR6_DYNAMIC_CC_SOURCE}.sampling_method must be 'srs' or 'lhs'.")
    category_uncertainty = _bool_value(
        payload.pop("category_uncertainty", False),
        field=f"{AR6_DYNAMIC_CC_SOURCE}.category_uncertainty",
    )
    if payload:
        raise ValueError(f"Unsupported {AR6_DYNAMIC_CC_SOURCE} keys: {sorted(payload)}.")
    return {
        "sampling_method": sampling_method,
        "category_uncertainty": category_uncertainty,
    }


def _bool_value(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean.")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "None":
        raise ValueError(f"{field} must be a non empty string when provided.")
    return text
