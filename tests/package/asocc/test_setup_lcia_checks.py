import shutil

import pytest

from pyaesa.asocc.orchestration.setup.validation import lcia_checks as mod


def test_validate_lcia_requirements() -> None:
    mod._validate_lcia_requirements(
        source="oecd_v2025",
        is_exio=False,
        needs_lcia_flag=False,
        lcia_methods=None,
    )
    mod._validate_lcia_requirements(
        source="exiobase_396_ixi",
        is_exio=True,
        needs_lcia_flag=True,
        lcia_methods=["pb_lcia"],
    )
    with pytest.raises(ValueError):
        mod._validate_lcia_requirements(
            source="oecd_v2025",
            is_exio=False,
            needs_lcia_flag=False,
            lcia_methods=["pb_lcia"],
        )


def test_validate_lcia_ready_for_domain_returns_early_without_methods(
    allocation_dummy_repo,
) -> None:
    mod._validate_lcia_ready_for_domain(
        source="oecd_v2025",
        years=[2005],
        lcia_methods=None,
        matrix_version=None,
        domain_label="original",
    )


def test_validate_lcia_ready_for_domain_raises_with_missing_years_and_methods(
    allocation_dummy_repo,
) -> None:
    allocation_dummy_repo.set_lcia_methods(
        source="oecd_v2025",
        matrix_version=None,
        methods=["pb_lcia"],
        available_years_by_method={"pb_lcia": []},
    )

    with pytest.raises(ValueError):
        mod._validate_lcia_ready_for_domain(
            source="oecd_v2025",
            years=[2005, 2030],
            lcia_methods=["pb_lcia"],
            matrix_version=None,
            domain_label="original",
        )


def test_validate_lcia_ready_for_domain_grouped_hint_and_success(allocation_dummy_repo) -> None:
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_396_ixi",
        matrix_version="oecd_d",
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=[2005, 2006],
    )
    allocation_dummy_repo.write_mrio_history(
        source="exiobase_396_ixi",
        matrix_version="oecd_d",
        years=[2005, 2006],
    )
    allocation_dummy_repo.set_lcia_methods(
        source="exiobase_396_ixi",
        matrix_version="oecd_d",
        methods=["pb_bad"],
        available_years_by_method={"pb_bad": []},
    )
    with pytest.raises(ValueError):
        mod._validate_grouped_lcia_ready(
            source="exiobase_396_ixi",
            years=[2005, 2006],
            lcia_methods=["pb_bad"],
            group_version="oecd_d",
            group_reg=True,
            group_sec=False,
        )

    allocation_dummy_repo.set_lcia_methods(
        source="exiobase_396_ixi",
        matrix_version="oecd_d",
        methods=["pb_ok"],
        available_years_by_method={"pb_ok": [2005, 2006]},
    )
    mod._validate_grouped_lcia_ready(
        source="exiobase_396_ixi",
        years=[2005, 2006],
        lcia_methods=["pb_ok"],
        group_version="oecd_d",
        group_reg=True,
        group_sec=False,
    )


def test_validate_lcia_ready_for_domain_missing_year_only_branch(allocation_dummy_repo) -> None:
    allocation_dummy_repo.set_lcia_methods(
        source="oecd_v2025",
        matrix_version=None,
        methods=["pb_lcia"],
        available_years_by_method={"pb_lcia": [2005, 2006]},
    )
    year_dir = mod._get_mrio_year_dir(source="oecd_v2025", year=2005, group_version=None)
    if year_dir.exists():
        shutil.rmtree(year_dir)

    with pytest.raises(ValueError):
        mod._validate_lcia_ready_for_domain(
            source="oecd_v2025",
            years=[2005, 2006],
            lcia_methods=["pb_lcia"],
            matrix_version=None,
            domain_label="original",
        )


def test_validate_lcia_ready_for_domain_missing_year_entry_branch(
    allocation_dummy_repo_factory,
) -> None:
    repo = allocation_dummy_repo_factory(name="allocation_lcia_missing_year_entry")
    repo.set_lcia_methods(
        source="oecd_v2025",
        matrix_version=None,
        methods=["pb_lcia"],
        available_years_by_method={"pb_lcia": [2005, 2006]},
    )
    payload = repo._read_mrio_metadata(source="oecd_v2025", matrix_version=None)
    payload["years"].pop("2005", None)
    repo._write_mrio_metadata_payload(
        source="oecd_v2025",
        matrix_version=None,
        payload=payload,
    )

    with pytest.raises(ValueError):
        mod._validate_lcia_ready_for_domain(
            source="oecd_v2025",
            years=[2005],
            lcia_methods=["pb_lcia"],
            matrix_version=None,
            domain_label="original",
        )


def test_validate_lcia_ready_for_domain_empty_years_hint_fallback() -> None:
    with pytest.raises(ValueError):
        mod._validate_lcia_ready_for_domain(
            source="oecd_v2025",
            years=[],
            lcia_methods=["pb_lcia"],
            matrix_version=None,
            domain_label="original",
        )


def test_validate_original_lcia_ready_wrapper_success(allocation_dummy_repo) -> None:
    allocation_dummy_repo.set_lcia_methods(
        source="exiobase_396_ixi",
        matrix_version=None,
        methods=["pb_ok"],
        available_years_by_method={"pb_ok": [2005, 2006]},
    )
    mod._validate_original_lcia_ready(
        source="exiobase_396_ixi",
        years=[2005, 2006],
        lcia_methods=["pb_ok"],
    )
