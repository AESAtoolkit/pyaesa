from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa import set_workspace
from pyaesa.shared.acc_asr_common.branches import config as config_mod
from pyaesa.shared.acc_asr_common.branches import expand as expand_mod
from pyaesa.shared.lcia.paths import static_cc_csv_path
from pyaesa.shared.lcia.static_cc import read_static_cc, require_static_cc_bounds_available


def test_normalize_static_cc_config_covers_valid_inputs() -> None:
    assert config_mod._normalize_static_cc_config(  # noqa: SLF001
        {"exclude_max_cc": False},
        allowed_keys={"exclude_max_cc"},
    ) == {"exclude_max_cc": False, "bounds": ["min_cc", "max_cc"]}
    assert config_mod._normalize_static_cc_config(  # noqa: SLF001
        {"exclude_max_cc": True},
        allowed_keys={"exclude_max_cc"},
    ) == {"exclude_max_cc": True, "bounds": ["min_cc"]}
    assert config_mod._normalize_static_cc_config(  # noqa: SLF001
        {},
        allowed_keys={"exclude_max_cc"},
    ) == {"exclude_max_cc": False, "bounds": ["min_cc", "max_cc"]}
    assert (
        config_mod._normalize_static_cc_config(  # noqa: SLF001
            {"active": False},
            allowed_keys={"active", "exclude_max_cc"},
        )
        is None
    )


@pytest.mark.parametrize("static_cc_config", [{"exclude_max_cc": "yes"}, {"extra": 1}])
def test_normalize_static_cc_config_rejects_invalid_inputs(
    static_cc_config,
) -> None:
    with pytest.raises(ValueError):
        config_mod._normalize_static_cc_config(  # noqa: SLF001
            static_cc_config,
            allowed_keys={"exclude_max_cc"},
        )


def test_normalize_static_cc_config_rejects_invalid_active_flag() -> None:
    with pytest.raises(ValueError):
        config_mod._normalize_static_cc_config(  # noqa: SLF001
            {"active": "yes"},
            allowed_keys={"active", "exclude_max_cc"},
        )


def test_normalize_dynamic_cc_config_covers_valid_and_invalid_inputs() -> None:
    assert config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
        {
            "harmonization": False,
            "harmonization_method": "offset",
            "category": ["Energy", "Energy"],
            "ssp_scenario": ["ssp2", "SSP1", "SSP2"],
            "emission_type": "co2",
            "include_afolu": True,
            "subset_version": None,
        }
    ) == {
        "harmonization": False,
        "harmonization_method": "offset",
        "category": ["Energy"],
        "ssp_scenario": ["SSP1", "SSP2"],
        "emission_type": "co2",
        "include_afolu": True,
        "emissions_mode": "gross_alt",
        "subset_version": None,
    }
    assert (
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"subset_version": " subset_a "}
        )["subset_version"]
        == "subset_a"
    )
    assert (
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"active": False}
        )
        == {}
    )
    assert config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
        {"category": ["Energy"], "ssp_scenario": None}
    ) == {
        "harmonization": True,
        "harmonization_method": "offset",
        "category": ["Energy"],
        "ssp_scenario": ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
        "emission_type": "kyoto_gases",
        "include_afolu": False,
        "emissions_mode": "gross_alt",
        "subset_version": None,
    }

    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"extra": True}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"active": "yes"}
        )
    assert config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
        {"category": "Energy", "ssp_scenario": ["SSP1"]}
    )["category"] == ["Energy"]
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"category": " "}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"category": []}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"category": [1]}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"category": [" "]}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"category": 1}
        )
    assert config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
        {"category": ["Energy"], "ssp_scenario": "SSP2"}
    )["ssp_scenario"] == ["SSP2"]
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"ssp_scenario": []}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"harmonization": "False"}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"include_afolu": "True"}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"harmonization_method": 1}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"harmonization_method": " "}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"subset_version": 1}
        )
    with pytest.raises(ValueError):
        config_mod._normalize_dynamic_cc_config(  # noqa: SLF001
            {"subset_version": ""}
        )


def test_normalize_base_cc_args_covers_static_dynamic_and_errors() -> None:
    default_static = {"static": {"exclude_max_cc": False, "bounds": ["min_cc", "max_cc"]}}
    expected_dynamic = {
        "dynamic_ar6": {
            "harmonization": True,
            "harmonization_method": "offset",
            "category": ["Energy"],
            "ssp_scenario": ["SSP1"],
            "emission_type": "kyoto_gases",
            "include_afolu": False,
            "emissions_mode": "gross_alt",
            "subset_version": None,
        }
    }

    assert config_mod.normalize_base_cc_args({}) == default_static  # noqa: SLF001
    assert (
        config_mod.normalize_base_cc_args(  # noqa: SLF001
            {"static": {"exclude_max_cc": False}}
        )
        == default_static
    )
    assert config_mod.normalize_base_cc_args(  # noqa: SLF001
        {"dynamic_ar6": {"category": ["Energy"], "ssp_scenario": ["SSP1"]}}
    ) == {
        **default_static,
        **expected_dynamic,
    }
    assert config_mod.normalize_base_cc_args(  # noqa: SLF001
        {"static": None, "dynamic_ar6": {"category": ["Energy"], "ssp_scenario": ["SSP1"]}}
    ) == {
        **default_static,
        **expected_dynamic,
    }
    assert (
        config_mod.normalize_base_cc_args(  # noqa: SLF001
            {
                "static": {"active": False},
                "dynamic_ar6": {"category": ["Energy"], "ssp_scenario": ["SSP1"]},
            }
        )
        == expected_dynamic
    )
    with pytest.raises(ValueError):
        config_mod.normalize_base_cc_args(  # noqa: SLF001
            {"static": {"active": False}, "dynamic_ar6": {"active": False}}
        )
    with pytest.raises(ValueError):
        config_mod.normalize_base_cc_args({"extra": {}})  # noqa: SLF001
    with pytest.raises(ValueError):
        config_mod.normalize_base_cc_args(cast(Any, None))  # noqa: SLF001
    with pytest.raises(ValueError):
        config_mod.normalize_base_cc_args({"static": True})  # noqa: SLF001
    with pytest.raises(ValueError):
        config_mod.normalize_base_cc_args({"dynamic_ar6": True})  # noqa: SLF001
    with pytest.raises(ValueError):
        config_mod.normalize_base_cc_args(  # noqa: SLF001
            {"static": {"exclude_max_cc": False, "extra": 1}}
        )


def test_require_asr_static_cc_source_compatibility_covers_skip_success_and_failure(
    tmp_path: Path,
) -> None:
    set_workspace(tmp_path / "workspace", refresh=True)
    static_path = static_cc_csv_path(lcia_method="gwp100_lcia")
    static_path.parent.mkdir(parents=True, exist_ok=True)

    # Bounds narrower than min/max skip disk validation by contract.
    config_mod.require_asr_static_cc_source_compatibility(  # noqa: SLF001
        cc_source="gwp100_lcia",
        static_cc_bounds=["min_cc"],
    )

    pd.DataFrame(
        [
            {
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "min_cc": 1.0,
                "max_cc": 2.0,
            }
        ]
    ).to_csv(static_path, index=False)
    config_mod.require_asr_static_cc_source_compatibility(  # noqa: SLF001
        cc_source="gwp100_lcia",
        static_cc_bounds=["min_cc", "max_cc"],
    )

    pd.DataFrame(
        [
            {
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "min_cc": 1.0,
            }
        ]
    ).to_csv(static_path, index=False)
    with pytest.raises(ValueError):
        config_mod.require_asr_static_cc_source_compatibility(  # noqa: SLF001
            cc_source="gwp100_lcia",
            static_cc_bounds=["min_cc", "max_cc"],
        )


def test_static_cc_reader_and_bounds_validate_malformed_inputs(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        read_static_cc(missing_path)

    malformed_path = tmp_path / "malformed.csv"
    pd.DataFrame([{"impact": "GWP_100", "min_cc": 1.0}]).to_csv(
        malformed_path,
        index=False,
    )
    with pytest.raises(ValueError):
        read_static_cc(malformed_path)

    pd.DataFrame([{"impact": "GWP_100", "impact_unit": "kg CO2-eq", "min_cc": "bad"}]).to_csv(
        malformed_path, index=False
    )
    with pytest.raises(ValueError):
        read_static_cc(malformed_path)

    pd.DataFrame(
        [
            {
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "min_cc": 1.0,
                "max_cc": "bad",
            }
        ]
    ).to_csv(malformed_path, index=False)
    with pytest.raises(ValueError):
        read_static_cc(malformed_path)

    valid = pd.DataFrame(
        [
            {
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "min_cc": 1.0,
                "max_cc": None,
            }
        ]
    )
    with pytest.raises(ValueError):
        require_static_cc_bounds_available(
            cc_df=valid,
            requested_bounds=["min_cc", "max_cc"],
            context="test",
        )


def test_cc_branch_expansion_covers_dynamic_rejection_and_branch_payloads() -> None:
    assert expand_mod.iter_cc_method_branches(  # noqa: SLF001
        lcia_methods=["gwp100_lcia"],
        base_cc_args={"static": {"bounds": ["min_cc"]}},
        years=[2005],
    ) == [
        {
            "cc_source": "gwp100_lcia",
            "cc_type": "static",
            "static_cc_bounds": ["min_cc"],
        }
    ]

    dynamic_base = {
        "dynamic_ar6": {
            "harmonization": True,
            "harmonization_method": "offset",
            "category": ["C1"],
            "ssp_scenario": ["SSP1"],
            "emission_type": "kyoto_gases",
            "include_afolu": False,
            "subset_version": None,
        }
    }
    dynamic_branches = expand_mod.iter_cc_method_branches(  # noqa: SLF001
        lcia_methods=["gwp100_lcia"],
        base_cc_args=dynamic_base,
        years=[2010, 2011],
    )
    assert dynamic_branches[0]["cc_type"] == "dynamic_ar6"

    with pytest.raises(ValueError):
        expand_mod.iter_cc_method_branches(  # noqa: SLF001
            lcia_methods=["gwp100_lcia"],
            base_cc_args=dynamic_base,
            years=[2010, 2012],
        )

    with pytest.raises(ValueError):
        expand_mod.iter_cc_method_branches(  # noqa: SLF001
            lcia_methods=["pb_lcia"],
            base_cc_args=dynamic_base,
            years=[2010, 2011],
        )
