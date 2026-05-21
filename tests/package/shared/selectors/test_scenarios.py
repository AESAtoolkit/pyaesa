import pytest

from pyaesa.shared.selectors import scenarios as scenarios_mod


def test_scenario_selector_contracts_cover_empty_invalid_and_partition_contracts() -> None:
    assert scenarios_mod.normalize_ssp_token(" ssp2 ") == "SSP2"
    assert scenarios_mod.normalize_optional_ssp_selector(["SSP2", "ssp1"], argument_name="ssp") == [
        "SSP1",
        "SSP2",
    ]
    assert scenarios_mod.normalize_optional_ssp_selector(None, argument_name="ssp") is None
    assert scenarios_mod.partition_token_to_ssp_token("ssp3", context="partition") == "SSP3"

    with pytest.raises(ValueError):
        scenarios_mod.normalize_ssp_token(" ", context="Scenario")
    with pytest.raises(ValueError):
        scenarios_mod.normalize_ssp_token("bad", context="Scenario")
    assert scenarios_mod.normalize_optional_ssp_selector("SSP2", argument_name="ssp") == ["SSP2"]
    with pytest.raises(ValueError):
        scenarios_mod.normalize_optional_ssp_selector(("SSP2",), argument_name="ssp")
    with pytest.raises(ValueError):
        scenarios_mod.partition_token_to_ssp_token(" ", context="partition")
