import pytest

from pyaesa.shared.tabular.deterministic_companion_stems import (
    parse_deterministic_companion_stem,
)


def test_parse_deterministic_companion_stem_covers_regression_reuse_and_ssp() -> None:
    projected = parse_deterministic_companion_stem(
        "l2_UT(GVA)__ssp1",
        scenario_tokens=["SSP1", "SSP2"],
    )
    assert projected.base_stem == "l2_UT(GVA)"
    assert projected.ssp_scenario == "SSP1"

    projected_alt = parse_deterministic_companion_stem(
        "l2_UT(GVA)__ssp2",
        scenario_tokens=["SSP1", "SSP2"],
    )
    assert projected_alt.base_stem == "l2_UT(GVA)"
    assert projected_alt.ssp_scenario == "SSP2"

    external_projected = parse_deterministic_companion_stem(
        "external__CO(S)_gwp100_lcia__ssp2",
        scenario_tokens=["SSP1", "SSP2"],
    )
    assert external_projected.base_stem == "external__CO(S)_gwp100_lcia"
    assert external_projected.ssp_scenario == "SSP2"

    external_plain = parse_deterministic_companion_stem(
        "external__CO(S)_gwp100_lcia",
        scenario_tokens=["SSP1", "SSP2"],
    )
    assert external_plain.base_stem == "external__CO(S)_gwp100_lcia"
    assert external_plain.ssp_scenario is None

    plain = parse_deterministic_companion_stem("l2_UT(GVA)")
    assert plain.base_stem == "l2_UT(GVA)"
    assert plain.ssp_scenario is None

    with pytest.raises(ValueError):
        parse_deterministic_companion_stem(
            "l2_UT(GVA)__ssp1__ssp2",
            scenario_tokens=["SSP1", "SSP2"],
        )
