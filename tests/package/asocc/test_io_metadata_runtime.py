from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.asocc.io import metadata as mod
from pyaesa.asocc.runtime.request import scope as scope_mod


def test_run_scope_key_and_build_run_metadata_shapes() -> None:
    sig = {"b": 2, "a": 1}
    key_1 = mod._run_scope_key(signature=sig)
    key_2 = mod._run_scope_key(signature={"a": 1, "b": 2})
    assert key_1 == key_2
    assert len(key_1) == 64

    payload = mod._build_run_metadata(
        requested_years=[2020],
        resolved_years=[2020],
        selected_methods={"L1": ["EG(Pop)"]},
        fu_code="L1.a",
        studied_indices_tag="all",
        skipped_years={},
        outputs=["a.csv"],
        signature={"sig": 1},
    )
    assert payload["function"] == "deterministic_asocc"
    assert payload["arguments"] == {"sig": 1}
    assert payload["execution"]["requested_years"] == [2020]
    assert payload["execution"]["resolved_years"] == [2020]
    assert payload["artifacts"]["outputs"] == ["a.csv"]
    assert payload["provenance"]["fu_code"] == "L1.a"
    assert payload["provenance"]["selected_methods"]["L1"] == ["EG(Pop)"]
    assert isinstance(payload["execution"]["timestamp"], str)

    context = mod.RunContext(
        project_name="demo",
        source="source_a",
        fu_code="L1.a",
        group_version=None,
        group_version_reg=None,
        group_reg=False,
        group_sec=False,
        lcia_method=None,
        years_input=None,
        reference_years_input=None,
        ssp_scenario=None,
        is_exio=False,
        l1_lcia_kind="CBA_FD",
        lcia_methods=None,
        selected_l1=[],
        combined=[],
        selected_l2_one_step=[],
        required_indices=set(),
        filters={},
        studied_indices_tag="all",
        proj_base=Path("."),
        logger=None,
        requested_years=[],
        resolved_years=[],
        persisted_years=[],
        historical_years=[],
        reference_years=None,
        ssp_scenario_options=[None],
        run_signature={},
        needs_lcia=False,
        repo_root=Path("."),
        wb_df=pd.DataFrame(),
        ssp_df=pd.DataFrame(),
        wb_df_raw=pd.DataFrame(),
        ssp_df_raw=pd.DataFrame(),
        selected_methods={},
        l1_kinds_needed=set(),
        l1_only_no_mrio=False,
        l1_reg_aggreg="post",
        use_original_l1_post_domain=False,
        variant_tag=None,
        aggreg_indices=False,
        output_format="csv",
        intermediate_outputs=False,
        output_source_label="published_source",
    )
    assert context.output_source == "published_source"


def test_load_run_metadata_and_build_run_signature(tmp_path: Path) -> None:
    path = tmp_path / "run_metadata.json"
    path.write_text(
        '{"function": "deterministic_asocc", "arguments": {}}',
        encoding="utf-8",
    )
    loaded = mod._load_run_metadata(path)
    assert loaded["function"] == "deterministic_asocc"
    assert loaded["arguments"] == {}

    scope = scope_mod.build_asocc_scope(
        base_allocate_args={
            "project_name": "demo",
            "source": "oecd_v2025",
            "group_version": "v1",
            "group_reg": True,
            "group_sec": False,
            "fu_code": "L2.a.a",
            "r_p": None,
            "s_p": ["A"],
            "r_c": None,
            "r_f": None,
            "method_plan": "default",
            "l1_methods": ["EG(Pop)"],
            "one_step_methods": ["UT(FD)"],
            "two_step_methods": None,
            "l1_l2_pairs": None,
            "lcia_method": ["IPCC"],
            "reference_years": [2018, 2019],
            "ssp_scenario": "SSP2",
            "l1_reg_aggreg": "pre",
            "aggreg_indices": False,
            "projection_mode": "regression",
            "reg_window": [2018, 2019, 2020, 2021],
            "l2_reuse_years": None,
        }
    )
    signature = scope.compute_signature(
        years=[2030],
        output_format="csv",
        intermediate_outputs=True,
        historical_year_cap=None,
        variant_tag="v1",
    )
    assert signature["source"] == "oecd_v2025"
    assert signature["reg_window"] == [2018, 2019, 2020, 2021]
    assert signature["l2_reuse_years"] == []
    assert signature["intermediate_outputs"] is True

    try:
        scope_mod.build_asocc_scope(
            base_allocate_args={
                "project_name": "demo",
                "source": "oecd_v2025",
                "group_version": "v1",
                "group_reg": True,
                "group_sec": False,
                "fu_code": "L2.a.a",
                "r_p": None,
                "s_p": ["A"],
                "r_c": None,
                "r_f": None,
                "method_plan": "default",
                "l1_methods": ["EG(Pop)"],
                "one_step_methods": ["UT(FD)"],
                "two_step_methods": None,
                "l1_l2_pairs": None,
                "lcia_method": ["IPCC"],
                "reference_years": cast(Any, (2018, 2019)),
                "ssp_scenario": "SSP2",
                "l1_reg_aggreg": "pre",
                "aggreg_indices": False,
                "projection_mode": "regression",
                "reg_window": [2018, 2019, 2020, 2021],
                "l2_reuse_years": None,
            }
        ).compute_signature(
            years=[2030],
            output_format="csv",
            intermediate_outputs=True,
            historical_year_cap=None,
            variant_tag="v1",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected tuple reference-year selector to fail.")


def test_normalize_signature_year_selector_covers_supported_and_invalid_types() -> None:
    assert scope_mod._normalize_year_selector_for_signature(None) is None  # noqa: SLF001
    assert scope_mod._normalize_year_selector_for_signature(2019) == [2019]  # noqa: SLF001
    assert scope_mod._normalize_year_selector_for_signature(range(2018, 2021)) == [  # noqa: SLF001
        2018,
        2019,
        2020,
    ]
    assert scope_mod._normalize_year_selector_for_signature([2020, 2019, 2019]) == [  # noqa: SLF001
        2019,
        2020,
    ]
    try:
        scope_mod._normalize_year_selector_for_signature(cast(Any, (2018, 2020)))  # noqa: SLF001
    except ValueError:
        pass
    else:
        raise AssertionError("Expected tuple selector to fail.")

    try:
        scope_mod._normalize_year_selector_for_signature(cast(Any, {"2019"}))  # noqa: SLF001
    except ValueError:
        pass
    else:
        raise AssertionError("Expected unsupported selector type to fail.")


def test_normalize_reg_window_for_storage_uses_one_canonical_full_year_list() -> None:
    assert mod._normalize_reg_window_for_storage(None) is None
    assert mod._normalize_reg_window_for_storage((1995, 1997)) == [1995, 1996, 1997]
    assert mod._normalize_reg_window_for_storage(range(1995, 1998)) == [1995, 1996, 1997]
    assert mod._normalize_reg_window_for_storage([1995, 1996, 1997]) == [1995, 1996, 1997]


def test_normalize_year_selector_for_storage_covers_supported_and_invalid_types() -> None:
    assert mod._normalize_year_selector_for_storage(None) is None  # noqa: SLF001
    assert mod._normalize_year_selector_for_storage(2019) == [2019]  # noqa: SLF001
    assert mod._normalize_year_selector_for_storage(range(2018, 2021)) == [  # noqa: SLF001
        2018,
        2019,
        2020,
    ]
    assert mod._normalize_year_selector_for_storage([2020, 2019, 2019]) == [  # noqa: SLF001
        2019,
        2020,
    ]

    try:
        mod._normalize_year_selector_for_storage(cast(Any, {"2019"}))  # noqa: SLF001
    except ValueError:
        pass
    else:
        raise AssertionError("Expected unsupported selector type to fail.")
