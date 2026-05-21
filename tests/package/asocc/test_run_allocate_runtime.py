from typing import Any, cast

import pytest

from pyaesa import deterministic_asocc
from pyaesa.asocc.data.paths import _get_mrio_year_dir


def test_deterministic_asocc_processes_warning_notices_and_final_write(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    result = deterministic_asocc(
        project_name="run_allocate_warning_notice",
        source="oecd_v2025",
        years=[2005],
        fu_code="L1.a",
        method_plan="default",
        l1_methods=["AR(E^{CBA_FD})", "EG(Pop)"],
        figures=False,
        refresh=True,
    )

    assert result.reuse_status == "computed"
    summary = str(result)
    assert summary
    assert "2005" in summary


def test_deterministic_asocc_rejects_invalid_source_and_figure_external_method(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo

    with pytest.raises(ValueError):
        deterministic_asocc(
            project_name="invalid_source",
            source=cast(Any, None),
            fu_code="L2.a.a",
            figures=False,
        )
    with pytest.raises(ValueError):
        deterministic_asocc(
            project_name="invalid_source",
            source=" ",
            fu_code="L2.a.a",
            figures=False,
        )
    with pytest.raises(ValueError):
        deterministic_asocc(
            project_name="invalid_figure_external",
            source="oecd_v2025",
            years=[2005],
            fu_code="L2.a.a",
            figure_external_method={"one_step_methods": ["UT(FD)"]},
            figures=False,
        )


def test_deterministic_asocc_processes_projection_notice_and_future_years(
    allocation_dummy_repo,
) -> None:
    historical_years = [2020, 2021, 2022]
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=historical_years,
        scenario_years=[2030],
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=historical_years,
    )
    allocation_dummy_repo.write_mrio_history(
        source="oecd_v2025",
        matrix_version=None,
        years=historical_years,
    )

    result = deterministic_asocc(
        project_name="run_allocate_projection_notice",
        source="oecd_v2025",
        years=[2030],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        ssp_scenario="SSP2",
        reg_window=historical_years,
        figures=False,
        refresh=True,
    )

    assert result.reuse_status == "computed"
    summary = str(result)
    assert summary
    assert "2030" in summary


def test_deterministic_asocc_skips_missing_year_directory_and_keeps_processed_outputs(
    allocation_dummy_repo,
) -> None:
    missing_year_dir = _get_mrio_year_dir(
        source="exiobase_396_ixi",
        year=2006,
        group_version=None,
    )
    for child in sorted(missing_year_dir.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        else:
            child.rmdir()
    missing_year_dir.rmdir()

    result = deterministic_asocc(
        project_name="run_allocate_skip_year",
        source="exiobase_396_ixi",
        years=[2005, 2006],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        lcia_method="gwp100_lcia",
        figures=False,
        refresh=True,
    )

    summary = str(result)
    assert summary
    assert "2005" in summary
    assert "2006" in summary


def test_deterministic_asocc_figures_false_preserves_existing_figures(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_figure_preserve_only"
    first_report = deterministic_asocc(
        project_name=project_name,
        source="exiobase_396_ixi",
        years=[2005],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        lcia_method="gwp100_lcia",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    )
    assert first_report is not None
    assert first_report.figure_paths
    assert all(path.exists() for path in first_report.figure_paths)

    skipped_report = deterministic_asocc(
        project_name=project_name,
        source="exiobase_396_ixi",
        years=[2005],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        lcia_method="gwp100_lcia",
        figures=False,
        refresh=False,
    )

    assert skipped_report.reuse_status == "reused_exact"
    assert all(path.exists() for path in first_report.figure_paths)


def test_deterministic_asocc_reuse_without_figures_returns_report(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    project_name = "asocc_reuse_without_figures"
    first_report = deterministic_asocc(
        project_name=project_name,
        source="oecd_v2025",
        years=[2005],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        figures=False,
        refresh=True,
    )
    assert first_report is not None

    reused_report = deterministic_asocc(
        project_name=project_name,
        source="oecd_v2025",
        years=[2005],
        fu_code="L2.a.a",
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        figures=False,
        refresh=False,
    )
    assert reused_report.reuse_status == "reused_exact"


def test_deterministic_asocc_reports_partial_reuse_for_expanded_year_scope(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    project_name = "asocc_partial_year_reuse"
    base_kwargs = {
        "project_name": project_name,
        "source": "oecd_v2025",
        "fu_code": "L2.a.a",
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
        "figures": False,
    }
    deterministic_asocc(years=[2005], refresh=True, **base_kwargs)

    expanded_report = deterministic_asocc(years=[2005, 2006], refresh=False, **base_kwargs)

    assert expanded_report.reuse_status == "partially_reused"
