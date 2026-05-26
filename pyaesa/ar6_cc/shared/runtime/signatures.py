"""Canonical AR6 CC selector normalization and scope-signature ownership."""

from pyaesa.download.ar6.utils.config import (
    RECOMMENDED_AR6_CATEGORIES,
    normalize_ar6_categories,
)
from pyaesa.shared.selectors.scenarios import DEFAULT_SSP_SCENARIOS, normalize_ssp_tokens

from pyaesa.ar6_cc.deterministic.request.contracts import (
    normalize_emission_type,
    normalize_emissions_mode,
)

AR6_CC_OUTPUT_CONTRACT = "study_period_table_with_post_study_period_companion_v1"
DEFAULT_AR6_CATEGORIES: list[str] = list(RECOMMENDED_AR6_CATEGORIES)


def normalize_cc_category(category: str | list[str] | None) -> list[str]:
    """Normalize an AR6 category selector, applying the public default when omitted."""
    return normalize_ar6_categories(category)


def normalize_cc_ssp_scenario(ssp_scenario: str | list[str] | None) -> list[str]:
    """Normalize an AR6 SSP selector, applying the public default when omitted."""
    if ssp_scenario is None:
        return list(DEFAULT_SSP_SCENARIOS)
    tokens = normalize_ssp_tokens(ssp_scenario, context="'ssp_scenario'")
    if not tokens:
        raise ValueError("'ssp_scenario' must be a non-empty string or list.")
    return tokens


def build_cc_scope_signature(
    *,
    study_period: list[int],
    harmonization: bool,
    harmonization_method: str,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str,
    category: str | list[str] | None,
    ssp_scenario: str | list[str] | None,
    subset_version: str | None,
) -> dict[str, object]:
    """Build the canonical AR6 CC scope signature payload."""
    return {
        "study_period": list(study_period),
        "harmonization": bool(harmonization),
        "harmonization_method": str(harmonization_method),
        "emission_type": normalize_emission_type(emission_type),
        "include_afolu": bool(include_afolu),
        "emissions_mode": normalize_emissions_mode(emissions_mode),
        "output_contract": AR6_CC_OUTPUT_CONTRACT,
        "category": normalize_cc_category(category),
        "ssp_scenario": list(normalize_ssp_tokens(normalize_cc_ssp_scenario(ssp_scenario))),
        "subset_version": subset_version,
    }
