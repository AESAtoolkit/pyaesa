"""Tests for disaggregation argument validation."""

import pytest

from pyaesa.asocc.disaggregation.config import parse_disaggregate_args


def _base_config() -> dict:
    return {
        "target_agg_run": {
            "source": "oecd_v2025",
            "agg_reg": False,
            "agg_sec": False,
            "agg_version": None,
            "s_p": ["D"],
        },
        "ref_agg_run": {
            "source": "exiobase_396_ixi",
            "agg_reg": False,
            "agg_sec": True,
            "agg_version": "oecd_d",
            "s_p": ["D"],
        },
        "ref_disagg_run": {
            "source": "exiobase_396_ixi",
            "agg_reg": False,
            "agg_sec": True,
            "agg_version": "elec",
            "s_p": ["Electricity_coal", "Electricity_gas"],
        },
        "disaggregation_specs": [
            {"agg_sector_label": "D", "disagg_sector_label": "Electricity_coal"},
            {"agg_sector_label": "D", "disagg_sector_label": "Electricity_gas"},
        ],
        "new_disagg_version_name": "disagg_oecd",
    }


def _base_allocate_args() -> dict:
    return {
        "project_name": "proj",
        "fu_code": "L2.c.b",
        "r_c": ["FR", "US"],
        "years": [2005, 2030],
        "ssp_scenario": ["SSP2", "SSP3"],
    }


def _runtime_args(**overrides) -> dict:
    args = {
        "output_format": "csv",
        "figures": False,
        "figure_options": None,
        "figure_format": None,
        "figure_external_method": None,
        "refresh": False,
    }
    args.update(overrides)
    return args


def test_parse_disaggregate_args_success_with_multiple_splits() -> None:
    parsed = parse_disaggregate_args(
        disaggregation_config=_base_config(),
        base_allocate_args=_base_allocate_args(),
        **_runtime_args(),
    )
    assert parsed.disaggregation.new_disagg_version_name == "disagg_oecd"
    assert len(parsed.disaggregation.disaggregation_specs) == 2
    assert parsed.base_allocate_args["fu_code"] == "L2.c.b"


def test_parse_disaggregate_args_rejects_disaggregate_mapped_to_two_aggregated() -> None:
    config = _base_config()
    config["disaggregation_specs"] = [
        {"agg_sector_label": "D", "disagg_sector_label": "Electricity_coal"},
        {"agg_sector_label": "E", "disagg_sector_label": "Electricity_coal"},
    ]
    with pytest.raises(ValueError):
        parse_disaggregate_args(
            disaggregation_config=config,
            base_allocate_args=_base_allocate_args(),
            **_runtime_args(),
        )


def test_parse_disaggregate_args_rejects_forbidden_base_args() -> None:
    args = _base_allocate_args()
    args["source"] = "oecd_v2025"
    with pytest.raises(ValueError):
        parse_disaggregate_args(
            disaggregation_config=_base_config(),
            base_allocate_args=args,
            **_runtime_args(),
        )
    args = _base_allocate_args()
    args["reference_years"] = [2005]
    with pytest.raises(ValueError):
        parse_disaggregate_args(
            disaggregation_config=_base_config(),
            base_allocate_args=args,
            **_runtime_args(),
        )
    args = _base_allocate_args()
    args["lcia_method"] = ["gwp100_lcia"]
    with pytest.raises(ValueError):
        parse_disaggregate_args(
            disaggregation_config=_base_config(),
            base_allocate_args=args,
            **_runtime_args(),
        )


def test_parse_disaggregate_args_rejects_selector_spec_mismatch() -> None:
    config = _base_config()
    config["target_agg_run"]["s_p"] = ["X"]
    with pytest.raises(ValueError, match="target_agg_run.s_p"):
        parse_disaggregate_args(
            disaggregation_config=config,
            base_allocate_args=_base_allocate_args(),
            **_runtime_args(),
        )


def test_parse_disaggregate_args_rejects_invalid_runtime_output_controls() -> None:
    with pytest.raises(ValueError, match="output_format"):
        parse_disaggregate_args(
            disaggregation_config=_base_config(),
            base_allocate_args=_base_allocate_args(),
            **_runtime_args(output_format="xlsx"),
        )


def test_parse_disaggregate_args_applies_allocate_cc_defaults() -> None:
    parsed = parse_disaggregate_args(
        disaggregation_config=_base_config(),
        base_allocate_args={"project_name": "proj", "fu_code": "L2.c.b"},
        **_runtime_args(),
    )
    args = parsed.base_allocate_args
    assert args["method_plan"] == "default"
    assert args["group_indices"] is False
    assert args["l1_reg_aggreg"] == "post"
    assert args["reg_window"] is None
