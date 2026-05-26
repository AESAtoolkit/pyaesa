"""Owned default contract for deterministic aSoCC request normalization."""

from typing import Any

from pyaesa.shared.selectors.scenarios import DEFAULT_SSP_SCENARIOS


DETERMINISTIC_ASOCC_OPTIONAL_DEFAULTS: dict[str, Any] = {
    "agg_reg": False,
    "agg_sec": False,
    "agg_version": "",
    "years": None,
    "r_p": None,
    "s_p": None,
    "r_c": None,
    "r_f": None,
    "group_indices": False,
    "method_plan": "default",
    "l1_methods": None,
    "one_step_methods": None,
    "two_step_methods": None,
    "l1_l2_pairs": None,
    "l1_reg_aggreg": "post",
    "lcia_method": None,
    "reference_years": None,
    "ssp_scenario": DEFAULT_SSP_SCENARIOS,
    "projection_mode": "regression",
    "reg_window": None,
    "l2_reuse_years": None,
    "output_format": "csv",
    "intermediate_outputs": False,
    "figures": True,
    "figure_format": {"format": "png", "dpi": 500},
    "figure_external_method": None,
    "refresh": False,
}


UNCERTAINTY_BASE_ALLOCATE_DEFAULTS: dict[str, Any] = {
    key: value
    for key, value in DETERMINISTIC_ASOCC_OPTIONAL_DEFAULTS.items()
    if key not in {"output_format", "intermediate_outputs", "refresh"}
}


DISAGGREGATION_BASE_ALLOCATE_DEFAULTS: dict[str, Any] = {
    key: DETERMINISTIC_ASOCC_OPTIONAL_DEFAULTS[key]
    for key in (
        "years",
        "r_p",
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
}
