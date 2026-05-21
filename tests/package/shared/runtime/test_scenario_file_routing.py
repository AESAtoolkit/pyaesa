from pathlib import Path

import pytest

from pyaesa.shared.runtime.scenario.file_routing import (
    ScenarioTaggedFileSpec,
    allowed_scenarios_for_year,
    resolve_year_assignments,
    validate_scenario_inventory,
)


def _spec(path: str, scenario: str | None, years: tuple[int, ...]) -> ScenarioTaggedFileSpec:
    return ScenarioTaggedFileSpec(path=Path(path), scenario=scenario, years=years)


def test_allowed_scenarios_for_year() -> None:
    assert allowed_scenarios_for_year(year=2030, ssp_scenario_options_by_year=None) == set()
    assert allowed_scenarios_for_year(year=2030, ssp_scenario_options_by_year={}) == set()
    assert allowed_scenarios_for_year(
        year=2030,
        ssp_scenario_options_by_year={2030: [None, "SSP2", "SSP1"]},
    ) == {"SSP1", "SSP2"}


def test_validate_scenario_inventory_branches() -> None:
    validate_scenario_inventory(specs=(), family_label="inputs", item_label="x")

    with pytest.raises(ValueError):
        validate_scenario_inventory(
            specs=(
                _spec("a.csv", None, (2000,)),
                _spec("b.csv", None, (2001,)),
            ),
            family_label="inputs",
            item_label="x",
        )

    with pytest.raises(ValueError):
        validate_scenario_inventory(
            specs=(
                _spec("a.csv", "SSP1", (2030,)),
                _spec("b.csv", "SSP2", (2040,)),
            ),
            family_label="inputs",
            item_label="x",
        )

    with pytest.raises(ValueError):
        validate_scenario_inventory(
            specs=(
                _spec("hist.csv", None, (2030,)),
                _spec("ssp.csv", "SSP2", (2030,)),
            ),
            family_label="inputs",
            item_label="x",
        )


def test_resolve_year_assignments_success_and_failures() -> None:
    hist = _spec("hist.csv", None, (2000, 2001))
    ssp1 = _spec("ssp1.csv", "SSP1", (2030, 2040))
    ssp2 = _spec("ssp2.csv", "SSP2", (2030, 2040))

    assigned = resolve_year_assignments(
        specs=(hist, ssp1, ssp2),
        years=[2000, 2030, 2040],
        ssp_scenario_options_by_year={2030: ["SSP2"], 2040: ["SSP1"]},
        family_label="inputs",
        item_label="x",
        expected_stems=["hist", "ssp1", "ssp2"],
    )
    assert assigned == {
        Path("hist.csv"): [2000],
        Path("ssp2.csv"): [2030],
        Path("ssp1.csv"): [2040],
    }

    multi_assigned = resolve_year_assignments(
        specs=(ssp1, ssp2),
        years=[2030],
        ssp_scenario_options_by_year={2030: ["SSP1", "SSP2"]},
        family_label="inputs",
        item_label="x",
        expected_stems=["ssp1", "ssp2"],
    )
    assert multi_assigned == {
        Path("ssp1.csv"): [2030],
        Path("ssp2.csv"): [2030],
    }

    with pytest.raises(ValueError):
        resolve_year_assignments(
            specs=(hist,),
            years=[2050],
            ssp_scenario_options_by_year=None,
            family_label="inputs",
            item_label="x",
        )

    with pytest.raises(ValueError):
        resolve_year_assignments(
            specs=(ssp1, ssp2),
            years=[2030],
            ssp_scenario_options_by_year={2030: ["SSP9"]},
            family_label="inputs",
            item_label="x",
            expected_stems=["ssp1", "ssp2"],
        )
