from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyaesa.download.mrios.utils.source_registry import get_mrio_entry
from pyaesa.shared.lcia import contracts as contracts_mod
from pyaesa.shared.lcia.cov_inputs import (
    LCIACoVInputs,
    country_cov_values,
    load_lcia_cov_inputs,
    normalize_lcia_uncertainty_parameters,
    sector_cov_keys,
    sector_cov_values,
)
from pyaesa.shared.lcia import paths as paths_mod
from pyaesa.shared.lcia.uncertainty_keys import build_lcia_shared_u_key
from pyaesa.shared.uncertainty_assessment.request.shared_u import (
    deterministic_shared_u_matrix,
)


def test_lcia_contracts_cover_rules_and_bundled_static_cc_loading(
    project_repo: Path,
) -> None:
    del project_repo

    with pytest.raises(ValueError):
        contracts_mod.normalize_lcia_method_name(" ")

    assert contracts_mod.normalize_lcia_method_name(" pb_lcia ") == "pb_lcia"

    cc_path, schema_kind, rows = contracts_mod.load_bundled_static_cc_rows(lcia_method="pb_lcia")
    assert cc_path == paths_mod.static_cc_csv_path(lcia_method="pb_lcia")
    assert cc_path.exists()
    assert schema_kind in {"planetary boundary", "standard"}
    assert rows

    units_path, expected_units = contracts_mod.bundled_cc_expected_impact_units(
        lcia_method="pb_lcia"
    )
    impacts_path, expected_impacts = contracts_mod.bundled_cc_expected_impacts(
        lcia_method="pb_lcia"
    )
    assert units_path == cc_path
    assert impacts_path == cc_path
    assert expected_units
    assert expected_impacts
    assert len(expected_units) == len(set(expected_units))
    assert len(expected_impacts) == len(set(expected_impacts))
    assert {impact for impact, _unit in expected_units} == set(expected_impacts)
    first_impact, first_unit = expected_units[0]
    impact_unit_path, impact_unit = contracts_mod.bundled_cc_impact_unit(
        lcia_method="pb_lcia",
        impact=first_impact,
    )
    assert impact_unit_path == cc_path
    assert impact_unit == first_unit
    with pytest.raises(ValueError):
        contracts_mod.bundled_cc_impact_unit(lcia_method="pb_lcia", impact="missing_impact")

    assert contracts_mod.dynamic_cc_match(lcia_method="gwp100_lcia") == {"impact": "GWP_100"}
    assert contracts_mod.dynamic_cc_match(lcia_method="pb_lcia") is None
    assert contracts_mod.dynamic_cc_compatible_methods(
        method_specs=("pb_lcia", "ef_3.1", "gwp100_lcia", "unknown_method")
    ) == ["ef_3.1", "gwp100_lcia"]
    assert (
        contracts_mod._normalize_dynamic_cc_match(  # noqa: SLF001
            lcia_method="pb_lcia",
            match=None,
        )
        is None
    )
    with pytest.raises(ValueError):
        contracts_mod._normalize_dynamic_cc_match(  # noqa: SLF001
            lcia_method="pb_lcia",
            match={"impact": " "},
        )


def test_lcia_path_contracts_cover_cleaning_and_cov_paths(
    project_repo: Path,
) -> None:
    del project_repo

    with pytest.raises(ValueError):
        paths_mod.static_cc_csv_path(lcia_method=" ")

    assert paths_mod.static_cc_csv_path(lcia_method=" gwp100_lcia ").name == (
        "gwp100_lcia_cc_steady_state.csv"
    )
    assert paths_mod._shared_lcia_subdir(source="oecd_v2025") == "oecd_v2025"  # noqa: SLF001
    assert (
        paths_mod._shared_lcia_subdir(source=" exiobase_3102_ixi ")  # noqa: SLF001
        == get_mrio_entry("exiobase_3102_ixi").shared_prereq_root
    )

    carbon_dir = paths_mod.carbon_account_cov_dir()
    carbon_path = paths_mod.carbon_account_cov_path(asset_name=" demo.csv ")
    assert carbon_dir.name == "carbon_accounts_covs"
    assert carbon_path == carbon_dir / "demo.csv"
    assert carbon_path.parent == carbon_dir

    covs = load_lcia_cov_inputs(sector_cov_mapping={"Electricity": "Electricity"})
    assert covs.country_covs["World"] == covs.world_cov
    assert covs.sector_cov_mapping == {"Electricity": "Electricity"}
    assert "Electricity" in covs.sector_covs
    grouped_covs = load_lcia_cov_inputs(
        sector_cov_mapping={"Electricity": "Electricity"},
        group_reg=True,
        group_version="eu27",
    )
    assert grouped_covs.country_covs["EU27"] == pytest.approx(0.0768)
    assert grouped_covs.country_covs["World"] == grouped_covs.world_cov
    aggregate_path = paths_mod.carbon_account_cov_path(
        asset_name="reg_cbca_covs_aggreg_indices.csv"
    )
    if aggregate_path.exists():
        aggregate_path.unlink()
    with pytest.raises(ValueError):
        load_lcia_cov_inputs(sector_cov_mapping={}, aggregate_region_covs=True)
    pd.DataFrame({"exio_code": ["FR, US", "World"], "cov": [0.31, 0.2]}).to_csv(
        aggregate_path,
        index=False,
    )
    aggregate_covs = load_lcia_cov_inputs(
        sector_cov_mapping={"Electricity": "Electricity"},
        aggregate_region_covs=True,
    )
    assert aggregate_covs.country_covs["FR, US"] == pytest.approx(0.31)
    pd.DataFrame({"exio_code": ["EU27, Nordic", "World"], "cov": [0.41, 0.2]}).to_csv(
        paths_mod.carbon_account_cov_path(asset_name="reg_cbca_covs_group_eu27_aggreg_indices.csv"),
        index=False,
    )
    grouped_aggregate_covs = load_lcia_cov_inputs(
        sector_cov_mapping={"Electricity": "Electricity"},
        group_reg=True,
        group_version="eu27",
        aggregate_region_covs=True,
    )
    assert grouped_aggregate_covs.country_covs["EU27, Nordic"] == pytest.approx(0.41)
    with pytest.raises(ValueError):
        load_lcia_cov_inputs(sector_cov_mapping={}, group_reg=True, group_version="missing")

    assert normalize_lcia_uncertainty_parameters(
        parameters={"sector_cov_mapping": {" Electricity ": " Electricity "}}
    ) == {"sector_cov_mapping": {"Electricity": "Electricity"}}
    with pytest.raises(ValueError):
        normalize_lcia_uncertainty_parameters(parameters={"unknown": True})
    with pytest.raises(ValueError):
        normalize_lcia_uncertainty_parameters(parameters={"sector_cov_mapping": ["bad"]})
    with pytest.raises(ValueError):
        normalize_lcia_uncertainty_parameters(parameters={"sector_cov_mapping": {" ": "x"}})
    with pytest.raises(ValueError):
        normalize_lcia_uncertainty_parameters(
            parameters={"sector_cov_mapping": {"Electricity": " "}}
        )
    bad_group_path = paths_mod.carbon_account_cov_path(asset_name="reg_cbca_covs_group_bad.csv")
    bad_group_path.write_text("exio_code,cov\nEU27,0.1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_lcia_cov_inputs(sector_cov_mapping={}, group_reg=True, group_version="bad")
    bad_group_path.write_text("exio_code,cov\nEU27,x\nWorld,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_lcia_cov_inputs(sector_cov_mapping={}, group_reg=True, group_version="bad")
    bad_group_path.write_text("exio_code,cov\nEU27,0.1\nWorld,\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_lcia_cov_inputs(sector_cov_mapping={}, group_reg=True, group_version="bad")
    bad_group_path.write_text(
        "exio_code,cov\nEU27,0.1\nEU27,0.2\nWorld,0.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_lcia_cov_inputs(sector_cov_mapping={}, group_reg=True, group_version="bad")
    bad_group_path.write_text("code,cov\nEU27,0.1\nWorld,0.2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_lcia_cov_inputs(sector_cov_mapping={}, group_reg=True, group_version="bad")

    small_covs = LCIACoVInputs(
        country_covs={"FR": 0.1, "World": 0.2},
        sector_covs={"Electricity": 0.3},
        world_cov=0.2,
        sector_cov_mapping={"Power": "Electricity"},
    )
    np.testing.assert_allclose(
        country_cov_values(covs=small_covs, country_key=pd.Series(["FR"])),
        [0.1],
    )
    with pytest.raises(ValueError):
        country_cov_values(covs=small_covs, country_key=pd.Series(["DE"]))
    with pytest.raises(ValueError):
        country_cov_values(covs=grouped_covs, country_key=pd.Series(["MissingRegion"]))
    sector_keys = sector_cov_keys(covs=small_covs, sector_label=pd.Series(["Power"]))
    assert sector_keys.tolist() == ["Electricity"]
    np.testing.assert_allclose(
        sector_cov_values(covs=small_covs, sector_key=sector_keys),
        [0.3],
    )
    with pytest.raises(ValueError):
        sector_cov_keys(covs=small_covs, sector_label=pd.Series(["Heat"]))
    with pytest.raises(ValueError):
        sector_cov_values(covs=small_covs, sector_key=pd.Series(["Unknown"]))


def test_lcia_shared_u_key_is_driver_scoped_by_run() -> None:
    country_key = build_lcia_shared_u_key(
        project_name="p",
        source="exiobase_396_ixi",
        group_reg=False,
        group_sec=False,
        group_version=None,
        driver_kind="country",
        driver_key="FR",
    )
    sector_key = build_lcia_shared_u_key(
        project_name="p",
        source="exiobase_396_ixi",
        group_reg=False,
        group_sec=False,
        group_version=None,
        driver_kind="sector",
        driver_key="Electricity",
    )
    values = deterministic_shared_u_matrix(
        shared_u_keys=np.array([country_key, country_key, sector_key], dtype=object),
        run_indices=np.array([0], dtype=np.int64),
    )

    assert values[0, 0] == values[0, 1]
    assert values[0, 0] != values[0, 2]
    assert (
        deterministic_shared_u_matrix(
            shared_u_keys=np.array([country_key], dtype=object),
            run_indices=np.array([1], dtype=np.int64),
        )[0, 0]
        != values[0, 0]
    )
    matrix = deterministic_shared_u_matrix(
        shared_u_keys=np.array([country_key, sector_key], dtype=object),
        run_indices=np.array([0, 1], dtype=np.int64),
    )

    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(matrix[0], values[0, [0, 2]])
