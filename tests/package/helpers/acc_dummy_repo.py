import pandas as pd

from pyaesa.shared.lcia.paths import responsibility_periods_csv_path

from .ar6_dummy_repo import build_ar6_dummy_repo


def prepare_exiobase_future_projection_repo(allocation_dummy_repo) -> None:
    """Populate one allocation dummy repo for minimal future exiobase projection tests."""
    prepare_exiobase_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2005, 2006, 2007],
        scenario_years=[2030],
    )


def prepare_exiobase_default_method_repo(allocation_dummy_repo) -> None:
    """Populate one allocation dummy repo for minimal default-method exiobase tests."""
    prepare_exiobase_repo_with_years(
        allocation_dummy_repo,
        historical_years=list(range(1995, 2006)),
        scenario_years=[2030],
    )


def prepare_dynamic_acc_repo(allocation_dummy_repo) -> None:
    """Populate one allocation dummy repo with the dynamic ACC prerequisites."""
    prepare_dynamic_acc_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2020, 2021],
        scenario_years=[2030],
    )


def prepare_dynamic_acc_repo_with_years(
    allocation_dummy_repo,
    *,
    historical_years: list[int],
    scenario_years: list[int],
) -> None:
    """Populate one allocation dummy repo with explicit dynamic ACC year coverage."""
    build_ar6_dummy_repo(allocation_dummy_repo.repo_root)
    prepare_exiobase_repo_with_years(
        allocation_dummy_repo,
        historical_years=historical_years,
        scenario_years=scenario_years,
    )
    rps_path = responsibility_periods_csv_path(
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
    )
    pd.DataFrame(
        [
            {
                "impact": "climate_child",
                "impact_parent": "GWP_100",
                "responsibility_period_years": 2,
            }
        ]
    ).to_csv(rps_path, index=False)


def prepare_exiobase_repo_with_years(
    allocation_dummy_repo,
    *,
    historical_years: list[int],
    scenario_years: list[int],
) -> None:
    """Populate one allocation dummy repo with explicit exiobase year coverage."""
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=historical_years,
        scenario_years=scenario_years,
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_396_ixi",
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=historical_years,
    )
    allocation_dummy_repo.write_mrio_history(
        source="exiobase_396_ixi",
        matrix_version=None,
        years=historical_years,
    )
    allocation_dummy_repo.write_lcia_support(
        source="exiobase_396_ixi",
        matrix_version=None,
        lcia_method="gwp100_lcia",
        available_years=historical_years,
    )
