import pandas as pd
import pytest

from pyaesa.asocc.orchestration.setup.request import scenarios as scen_mod


def test_assert_unique_scenarios_reports_duplicates_once() -> None:
    scen_mod._assert_unique_scenarios(  # noqa: SLF001
        scenarios=[None, "SSP1", "SSP2"],
        where="demo",
    )
    with pytest.raises(ValueError):
        scen_mod._assert_unique_scenarios(  # noqa: SLF001
            scenarios=["SSP1", None, "SSP1", None],
            where="demo",
        )


def test_build_scenario_plan_by_year_covers_wb_and_ssp_routes() -> None:
    wb_df = pd.DataFrame(columns=["meta", "2005", 2006, "not_a_year"])
    assert scen_mod._wb_year_set(wb_df=wb_df) == {2005, 2006}  # noqa: SLF001

    plan = scen_mod.build_scenario_plan_by_year(
        years=[2005, 2006, 2030],
        wb_df=wb_df,
        ssp_scenarios=["SSP2", None, "SSP1"],
    )
    assert plan == {
        2005: [None],
        2006: [None],
        2030: ["SSP2", "SSP1"],
    }

    no_ssp_plan = scen_mod.build_scenario_plan_by_year(
        years=[2030],
        wb_df=pd.DataFrame(columns=["2005"]),
        ssp_scenarios=[None],
    )
    assert no_ssp_plan == {2030: [None]}


def test_scenario_state_options_from_plan_covers_uniqueness_and_ordering() -> None:
    with pytest.raises(ValueError):
        scen_mod.scenario_state_options_from_plan(scenario_plan_by_year={2030: ["SSP2", "SSP2"]})

    assert scen_mod.scenario_state_options_from_plan(
        scenario_plan_by_year={2030: ["SSP2"], 2031: ["SSP1", None]}
    ) == [None, "SSP1", "SSP2"]
    assert scen_mod.scenario_state_options_from_plan(
        scenario_plan_by_year={2030: ["SSP2"], 2031: ["SSP1"]}
    ) == ["SSP1", "SSP2"]
