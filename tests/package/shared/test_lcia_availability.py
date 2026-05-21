from pathlib import Path

import pytest

from pyaesa.workspace_initialisation.workspace import clear_default_repo_root
from pyaesa.shared.lcia import availability as availability_mod
from pyaesa.shared.lcia import paths as paths_mod


def test_lcia_availability_contracts_cover_file_existence_and_discovery(
    project_repo: Path,
) -> None:
    matrix_path = paths_mod.characterization_matrix_path(
        source="exiobase_3102_ixi",
        lcia_method="demo_lcia",
    )
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text("impact,value\nA,1.0\n", encoding="utf-8")

    rps_path = paths_mod.responsibility_periods_csv_path(
        source="exiobase_3102_ixi",
        lcia_method="demo_lcia",
    )
    rps_path.parent.mkdir(parents=True, exist_ok=True)
    rps_path.write_text("impact,value\nA,1.0\n", encoding="utf-8")

    static_cc_path = paths_mod.static_cc_csv_path(lcia_method="demo_lcia")
    static_cc_path.parent.mkdir(parents=True, exist_ok=True)
    static_cc_path.write_text("impact,value\nA,1.0\n", encoding="utf-8")

    skipped_static_cc_path = paths_mod.static_cc_csv_path(lcia_method="name_demo_lcia")
    skipped_static_cc_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_static_cc_path.write_text("impact,value\nA,1.0\n", encoding="utf-8")

    discovered_methods = availability_mod.discover_static_cc_methods()

    assert (
        availability_mod.has_characterization_matrix(
            source="exiobase_3102_ixi",
            lcia_method="demo_lcia",
        )
        is True
    )
    assert (
        availability_mod.has_characterization_matrix(
            source="exiobase_3102_ixi",
            lcia_method="missing_lcia",
        )
        is False
    )
    assert availability_mod.has_rps(source="exiobase_3102_ixi", lcia_method="demo_lcia") is True
    assert availability_mod.has_rps(source="exiobase_3102_ixi", lcia_method="missing_lcia") is False
    assert availability_mod.has_static_cc(lcia_method="demo_lcia") is True
    assert availability_mod.has_static_cc(lcia_method="missing_lcia") is False
    assert availability_mod.require_static_cc_csv_path(lcia_method="demo_lcia") == static_cc_path

    assert "demo_lcia" in discovered_methods
    assert "name_demo_lcia" not in discovered_methods

    missing_path = paths_mod.static_cc_csv_path(lcia_method="missing_lcia")
    if missing_path.exists():
        missing_path.unlink()
    with pytest.raises(FileNotFoundError):
        availability_mod.require_static_cc_csv_path(lcia_method="missing_lcia")

    assert paths_mod._shared_lcia_subdir(source=" exiobase_3102_ixi ") == "exiobase_3"  # noqa: SLF001
    assert paths_mod._shared_lcia_subdir(source=" oecd_v2025 ") == "oecd_v2025"  # noqa: SLF001
    assert paths_mod.bundled_static_cc_dir() == project_repo / "data_raw" / "carrying_capacities"
    assert paths_mod.carbon_account_cov_dir() == (
        project_repo / "data_raw" / "mrio" / "exiobase_3" / "lcia" / "carbon_accounts_covs"
    )
    assert paths_mod.carbon_account_cov_path(asset_name=" sec_cbca_covs.csv ") == (
        paths_mod.carbon_account_cov_dir() / "sec_cbca_covs.csv"
    )


def test_discover_static_cc_methods_uses_packaged_prerequisites_without_workspace() -> None:
    clear_default_repo_root()
    try:
        discovered_methods = availability_mod.discover_static_cc_methods()
    finally:
        clear_default_repo_root()
    assert "gwp100_lcia" in discovered_methods
    assert "pb_lcia" in discovered_methods
