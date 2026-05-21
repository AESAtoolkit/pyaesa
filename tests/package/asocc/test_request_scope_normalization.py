import json
from pathlib import Path
from typing import Any, cast

import pytest

from pyaesa.asocc.runtime.request import normalization as norm_mod
from pyaesa.asocc.runtime.request import scope as scope_mod
from pyaesa.asocc.runtime.scope import branch_resolution as branch_mod


def test_request_scope_normalizers_cover_success_paths() -> None:
    assert norm_mod._normalize_optional_string_list(None) is None
    assert norm_mod._normalize_optional_string_list("  FR  ") == ["FR"]
    assert norm_mod._normalize_optional_string_list([" US ", "", "FR", "US"]) == ["FR", "US"]
    assert norm_mod._normalize_optional_string_list(2030) == ["2030"]

    assert norm_mod._normalize_optional_string(" demo ") == "demo"
    assert norm_mod._normalize_optional_string("   ") is None
    assert norm_mod._require_non_empty_string(" demo ", name="demo") == "demo"
    assert norm_mod._require_bool(True, name="flag") is True

    assert norm_mod._normalize_optional_years([2030, 2031], name="years") == [2030, 2031]
    assert norm_mod._normalize_optional_years(None, name="years") is None
    assert norm_mod._normalize_aggreg_indices(False) is False
    assert norm_mod._normalize_l1_reg_aggreg(" POST ") == "post"
    assert norm_mod.normalize_source_grouping_scope(
        source=" oecd_v2025 ",
        group_reg=True,
        group_sec=False,
        group_version=" grp ",
    ) == ("oecd_v2025", True, False, "grp")
    assert norm_mod.normalize_source_grouping_scope(
        source=" disagg_demo ",
        group_reg=None,
        group_sec=None,
        group_version=None,
    ) == ("disagg_demo", False, False, None)
    assert scope_mod._selector_method_label(" plain ") == "plain"  # noqa: SLF001
    assert scope_mod._selector_method_label(" CO(S)::UT(FD) ") == "CO(S)_UT(FD)"  # noqa: SLF001
    assert scope_mod._selector_methods_for_scope(  # noqa: SLF001
        fu_code="L1.a",
        selected_methods={"l1": [" EG(Pop) ", ""]},
        combined=[],
    ) == {"EG(Pop)"}
    assert scope_mod._selector_methods_for_scope(  # noqa: SLF001
        fu_code="L2.a.a",
        selected_methods={
            "l2_vs_global": [" UT(FD) ", ""],
        },
        combined=[("AR(E^{CBA_FD})", "EG(Pop)")],
    ) == {"UT(FD)", "EG(Pop)_AR(E^{CBA_FD})"}


@pytest.mark.parametrize(
    ("callable_", "_match"),
    [
        (
            lambda: norm_mod._require_non_empty_string(None, name="demo"),
            "'demo' must be a non-empty string",
        ),
        (
            lambda: norm_mod._require_non_empty_string("   ", name="demo"),
            "'demo' must be a non-empty string",
        ),
        (
            lambda: norm_mod._require_bool("true", name="flag"),
            "'flag' must be a boolean",
        ),
        (
            lambda: norm_mod._normalize_aggreg_indices("both"),
            "Use True or False, not 'both'",
        ),
        (
            lambda: norm_mod._normalize_aggreg_indices("no"),
            "must be a boolean",
        ),
        (
            lambda: norm_mod._normalize_l1_reg_aggreg(["pre"]),
            "Use 'pre' or 'post', not a list",
        ),
        (
            lambda: norm_mod._normalize_l1_reg_aggreg("both"),
            "Use 'pre' or 'post', not 'both'",
        ),
        (
            lambda: norm_mod._normalize_l1_reg_aggreg("invalid"),
            "must be either 'pre' or 'post'",
        ),
    ],
)
def test_request_scope_normalizers_reject_invalid_values(callable_, _match: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        callable_()


def test_normalize_base_allocate_args_normalizes_full_public_scope() -> None:
    normalized = norm_mod.normalize_base_allocate_args(
        {
            "project_name": " demo ",
            "source": " oecd_v2025 ",
            "fu_code": " L2.a.a ",
            "group_reg": True,
            "group_sec": False,
            "group_version": " grp ",
            "years": 2030,
            "r_p": [" US ", "FR", "US"],
            "s_p": "D",
            "r_c": None,
            "r_f": ["FR"],
            "aggreg_indices": False,
            "method_plan": "one_step",
            "one_step_methods": ["AR(E^{CBA_FD})"],
            "l1_methods": None,
            "two_step_methods": None,
            "l1_l2_pairs": None,
            "l1_reg_aggreg": "pre",
            "lcia_method": "gwp100_lcia",
            "reference_years": [2005, 2006],
            "ssp_scenario": "SSP2",
            "projection_mode": "regression",
            "reg_window": range(2018, 2021),
            "l2_reuse_years": [2018, 2019],
        }
    )

    assert normalized["project_name"] == "demo"
    assert normalized["source"] == "oecd_v2025"
    assert normalized["fu_code"] == "L2.a.a"
    assert normalized["group_reg"] is True
    assert normalized["group_sec"] is False
    assert normalized["group_version"] == "grp"
    assert normalized["years"] == [2030]
    assert normalized["r_p"] == ["FR", "US"]
    assert normalized["s_p"] == ["D"]
    assert normalized["r_c"] is None
    assert normalized["r_f"] == ["FR"]
    assert normalized["aggreg_indices"] is False
    assert normalized["method_plan"] == "one_step"
    assert normalized["one_step_methods"] == ["AR(E^{CBA_FD})"]
    assert normalized["l1_methods"] is None
    assert normalized["two_step_methods"] is None
    assert normalized["l1_l2_pairs"] is None
    assert normalized["l1_reg_aggreg"] == "pre"
    assert normalized["lcia_method"] == ["gwp100_lcia"]
    assert normalized["reference_years"] == [2005, 2006]
    assert normalized["ssp_scenario"] == ["SSP2"]
    assert normalized["projection_mode"] == "regression"
    assert normalized["reg_window"] == [2018, 2019, 2020]
    assert normalized["l2_reuse_years"] == [2018, 2019]


def test_normalize_deterministic_scope_args_normalizes_shared_scope() -> None:
    normalized = norm_mod.normalize_deterministic_scope_args(
        {
            "project_name": " demo ",
            "source": " oecd_v2025 ",
            "fu_code": " L2.a.a ",
            "group_reg": True,
            "group_sec": False,
            "group_version": " grp ",
            "aggreg_indices": False,
            "l1_reg_aggreg": " PRE ",
        },
        payload_name="base_asocc_args",
    )

    assert normalized == {
        "project_name": "demo",
        "source": "oecd_v2025",
        "fu_code": "L2.a.a",
        "group_reg": True,
        "group_sec": False,
        "group_version": "grp",
        "aggreg_indices": False,
        "l1_reg_aggreg": "pre",
    }


@pytest.mark.parametrize(
    ("raw", "_match", "error_type"),
    [
        (
            "not a dict",
            "'base_asocc_args' must be a dictionary",
            ValueError,
        ),
        (
            {
                "project_name": "demo",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "unknown": 1,
            },
            "contains unknown key",
            ValueError,
        ),
        (
            {
                "project_name": "demo",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "refresh": True,
            },
            "contains forbidden key",
            ValueError,
        ),
        (
            {
                "project_name": "demo",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "group_reg": "yes",
            },
            "group_reg",
            TypeError,
        ),
        (
            {
                "project_name": "demo",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "method_plan": "one_step",
                "one_step_methods": ["not_a_method"],
            },
            "canonical scientific registry label",
            ValueError,
        ),
        (
            {
                "project_name": "demo",
                "source": "disagg_oecd",
                "fu_code": "L2.a.a",
                "group_version": "elec",
            },
            "must be called directly without grouping controls",
            ValueError,
        ),
        (
            {
                "project_name": "demo",
                "source": "disagg_oecd",
                "fu_code": "L2.a.a",
                "group_reg": True,
            },
            "must be called directly without grouping controls",
            ValueError,
        ),
        (
            {
                "project_name": "demo",
                "source": "disagg_oecd",
                "fu_code": "L2.a.a",
                "group_sec": True,
            },
            "must be called directly without grouping controls",
            ValueError,
        ),
    ],
)
def test_normalize_base_allocate_args_rejects_invalid_payloads(
    raw,
    _match: str,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        norm_mod.normalize_base_allocate_args(raw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "_match", "error_type"),
    [
        (
            "not a dict",
            "'base_asocc_args' must be a dictionary",
            ValueError,
        ),
        (
            {
                "project_name": "demo",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "aggreg_indices": "true",
            },
            "'base_asocc_args.aggreg_indices' must be a boolean",
            ValueError,
        ),
        (
            {
                "project_name": "demo",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "l1_reg_aggreg": "bad",
            },
            "'base_asocc_args.l1_reg_aggreg' must be either 'pre' or 'post'",
            ValueError,
        ),
    ],
)
def test_normalize_deterministic_scope_args_rejects_invalid_payloads(
    raw,
    _match: str,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        norm_mod.normalize_deterministic_scope_args(raw, payload_name="base_asocc_args")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("group_reg", "group_sec", "group_version"),
    [
        (True, False, None),
        (False, True, None),
        (False, False, "demo"),
    ],
)
def test_normalize_source_grouping_scope_rejects_grouping_for_disaggregated_sources(
    group_reg: bool,
    group_sec: bool,
    group_version: str | None,
) -> None:
    with pytest.raises(ValueError):
        norm_mod.normalize_source_grouping_scope(
            source="disagg_demo",
            group_reg=group_reg,
            group_sec=group_sec,
            group_version=group_version,
        )


def test_request_scope_wrappers_build_payloads_signatures_and_iso3_scope(
    allocation_dummy_repo,
) -> None:
    base_allocate_args = norm_mod.normalize_base_allocate_args(
        {
            "project_name": "demo_scope",
            "source": "iso3",
            "fu_code": "L1.a",
            "method_plan": "default",
            "l1_methods": ["EG(Pop)"],
            "years": [2005],
            "reference_years": [2005],
            "ssp_scenario": "SSP2",
            "lcia_method": "gwp100_lcia",
            "projection_mode": "historical_reuse",
            "reg_window": [2005],
            "l2_reuse_years": [2005],
        }
    )

    scope = scope_mod.build_asocc_scope(base_allocate_args=base_allocate_args)
    assert scope.selected_methods["l1"] == ["EG(Pop)"]
    assert scope.target_selector_payload == {
        "years": [2005],
        "reference_year": [2005],
        "l2_reuse_year": [2005],
        "ssp_values": ["SSP2"],
        "lcia_method": ["gwp100_lcia"],
        "methods": ["EG(Pop)"],
    }
    assert scope.resolve_path_scope().source_label == "iso3"

    requested_signature = scope.requested_signature()
    assert requested_signature == scope.requested_signature()
    assert requested_signature["reg_window"] is None
    assert requested_signature["projection_mode"] is None
    assert requested_signature["l2_reuse_years"] == []

    normalized_scope = scope_mod.build_asocc_scope(
        base_allocate_args=norm_mod.normalize_base_allocate_args(base_allocate_args)
    )
    assert normalized_scope.requested_signature() == requested_signature

    minimal_scope = scope_mod.build_asocc_scope(
        base_allocate_args=norm_mod.normalize_base_allocate_args(
            {
                "project_name": "demo_scope",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "years": [2005],
            }
        )
    )
    assert minimal_scope.target_selector_payload == {
        "years": [2005],
        "ssp_values": ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
        "methods": ["UT(FD)"],
    }
    assert minimal_scope.requested_signature()["l2_reuse_years"] == []
    minimal_compute_signature = minimal_scope.compute_signature(
        years=[2005],
        output_format="csv",
        intermediate_outputs=True,
        historical_year_cap=None,
    )
    assert minimal_compute_signature["l2_reuse_years"] == []
    pair_scope = scope_mod.build_asocc_scope(
        base_allocate_args=norm_mod.normalize_base_allocate_args(
            {
                "project_name": "demo_scope",
                "source": "oecd_v2025",
                "fu_code": "L2.a.a",
                "method_plan": "pairs",
                "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FD)"],
                "lcia_method": "gwp100_lcia",
                "years": [2005],
            }
        )
    )
    assert pair_scope.target_selector_payload == {
        "years": [2005],
        "ssp_values": ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
        "lcia_method": ["gwp100_lcia"],
        "methods": ["AR(E^{CBA_FD})_UT(FD)"],
    }
    assert (
        scope_mod.AsoccScope(
            base_allocate_args={
                **minimal_scope.base_allocate_args,
                "years": None,
                "reference_years": None,
                "l2_reuse_years": None,
                "ssp_scenario": None,
                "lcia_method": None,
            },
            selected_l1=[],
            combined=[],
            selected_l2_one_step=[],
            selected_methods={"l1": [], "l2_in_l1": [], "l2_vs_global": []},
            filters={},
            studied_indices_tag="all",
        ).target_selector_payload
        == {}
    )
    del allocation_dummy_repo


def test_path_resolution_cover_validation_and_paths(allocation_dummy_repo) -> None:
    with pytest.raises(ValueError):
        branch_mod.outputs_project_root(project_name=" ")

    with pytest.raises(ValueError):
        branch_mod.build_asocc_deterministic_path_scope(
            proj_base=allocation_dummy_repo.repo_root,
            source_label=" ",
            group_version=None,
        )

    base_allocate_args = norm_mod.normalize_base_allocate_args(
        {
            "project_name": "demo_path",
            "source": "oecd_v2025",
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
        }
    )
    scope = branch_mod.resolve_allocate_path_scope(base_allocate_args=base_allocate_args)
    assert scope.source_label == "oecd_v2025"
    roots = branch_mod.collect_asocc_roots(scope=scope, fu_code="L2.a.a")
    assert len(roots) == 4
    assert all("B1_asocc" in path for path in roots)
    assert any(Path(path).parts[-2:] == ("results", "level_1") for path in roots)
    assert any(Path(path).parts[-3:] == ("results", "level_2", "l2_vs_global") for path in roots)
    assert branch_mod.allocate_run_metadata_path(scope=scope).name.endswith(".json")

    disagg_scope_expected = branch_mod.build_asocc_deterministic_path_scope(
        proj_base=scope.proj_base,
        source_label="disagg_demo",
        group_version=None,
    )
    disagg_manifest = branch_mod.allocate_run_metadata_path(scope=disagg_scope_expected)
    disagg_manifest.parent.mkdir(parents=True, exist_ok=True)
    disagg_manifest.write_text(
        json.dumps({"function": "deterministic_asocc", "arguments": {}}),
        encoding="utf-8",
    )
    assert branch_mod.project_base_from_allocation_descendant(disagg_manifest) == scope.proj_base
    disagg_scope, resolved_path = branch_mod.resolve_disaggregation_path_scope(
        base_allocate_args=base_allocate_args,
        source_label="disagg_demo",
    )
    assert disagg_scope == disagg_scope_expected
    assert resolved_path == disagg_manifest
    scope_wrapper = scope_mod.build_asocc_scope(base_allocate_args=base_allocate_args)
    wrapped_scope, wrapped_path = scope_wrapper.resolve_disaggregation_scope(
        source_label="disagg_demo"
    )
    assert wrapped_scope == disagg_scope
    assert wrapped_path == disagg_manifest

    grouped_base_allocate_args = norm_mod.normalize_base_allocate_args(
        {
            "project_name": "demo_path",
            "source": "exiobase_396_ixi",
            "group_sec": True,
            "group_version": "elec",
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
        }
    )
    grouped_disagg_scope, grouped_resolved_path = branch_mod.resolve_disaggregation_path_scope(
        base_allocate_args=grouped_base_allocate_args,
        source_label="disagg_demo",
    )
    assert grouped_disagg_scope.source_label == "disagg_demo"
    assert grouped_disagg_scope.group_version is None
    assert grouped_resolved_path == disagg_manifest
    direct_disagg_scope = branch_mod.build_asocc_deterministic_path_scope(
        proj_base=scope.proj_base,
        source_label="disagg_demo",
        group_version="elec",
    )
    assert direct_disagg_scope.group_version is None

    with pytest.raises(ValueError):
        branch_mod.project_base_from_allocation_descendant(allocation_dummy_repo.repo_root)

    with pytest.raises(ValueError):
        branch_mod.resolve_disaggregation_path_scope(
            base_allocate_args=base_allocate_args,
            source_label="missing_disagg_demo",
        )

    with pytest.raises(ValueError):
        branch_mod.path_scope_from_signature(
            proj_base=scope.proj_base,
            source_label="oecd_v2025",
            run_signature={},
            context_label="test path",
        )
    assert branch_mod.path_scope_from_signature(
        proj_base=scope.proj_base,
        source_label="oecd_v2025",
        run_signature={"group_version": "elec"},
        context_label="test path",
    ) == branch_mod.build_asocc_deterministic_path_scope(
        proj_base=scope.proj_base,
        source_label="oecd_v2025",
        group_version="elec",
    )

    enacting_metric_dir = branch_mod.asocc_enacting_metric_dir(
        scope=scope, level="level_2", fu_code="L2.a.a"
    )
    logs_root = branch_mod.asocc_logs_root(scope=scope)
    assert "enacting_metrics" in enacting_metric_dir.parts
    assert "logs" in logs_root.parts


def test_request_scope_projection_signature_edges() -> None:
    assert scope_mod._normalize_year_selector_for_signature(2005) == [2005]  # noqa: SLF001
    assert scope_mod._normalize_year_selector_for_signature(range(2005, 2007)) == [  # noqa: SLF001
        2005,
        2006,
    ]
    assert scope_mod._normalize_year_selector_for_signature([2006, 2005, 2005]) == [  # noqa: SLF001
        2005,
        2006,
    ]
    with pytest.raises(ValueError):
        scope_mod._normalize_year_selector_for_signature(  # noqa: SLF001
            cast(Any, (2005, 2006))
        )

    non_native_args = norm_mod.normalize_base_allocate_args(
        {
            "project_name": "demo_non_native_scope",
            "source": "disagg_demo",
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "years": [2030],
            "projection_mode": "regression",
            "reg_window": [2018, 2019, 2020],
            "l2_reuse_years": range(2018, 2020),
        }
    )
    non_native_scope = scope_mod.AsoccScope(
        base_allocate_args=non_native_args,
        selected_l1=[],
        combined=[],
        selected_l2_one_step=["UT(FD)"],
        selected_methods={"l1": [], "l2_in_l1": [], "l2_vs_global": ["UT(FD)"]},
        filters={},
        studied_indices_tag="all",
    )

    signature = non_native_scope.requested_signature(years_hint=[2030])

    assert signature["projection_mode"] == "regression"
    assert signature["reg_window"] == [2018, 2019, 2020]
    assert signature["l2_reuse_years"] == [2018, 2019]

    base_args = {
        "source": "exiobase_3102_ixi",
        "fu_code": "L2.c.b",
        "years": [2025],
        "projection_mode": None,
        "reg_window": None,
        "l2_reuse_years": None,
    }
    combined = [("UT(FDa)", "EG(Pop)")]

    _, default_reg_window, default_reuse_years = (
        scope_mod.effective_projection_signature_for_source(
            base_allocate_args=base_args,
            selected_l2_one_step=["UT(TD)"],
            combined=combined,
            years_hint=[2025],
            projection_rule_source=None,
        )
    )
    assert default_reg_window == list(range(1995, 2023))
    assert default_reuse_years == list(range(1995, 2023))

    _, explicit_reg_window, explicit_reuse_years = (
        scope_mod.effective_projection_signature_for_source(
            base_allocate_args={
                **base_args,
                "reg_window": list(range(1995, 2025)),
                "l2_reuse_years": [2023, 2024],
            },
            selected_l2_one_step=["UT(TD)"],
            combined=combined,
            years_hint=[2025],
            projection_rule_source=None,
        )
    )
    assert explicit_reg_window == list(range(1995, 2025))
    assert explicit_reuse_years == [2023, 2024]
