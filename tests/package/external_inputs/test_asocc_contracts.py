from typing import Any

import pytest

from pyaesa.external_inputs.asocc.schema import contracts


def test_optional_string_list_and_method_label_validation_fail_fast() -> None:
    assert contracts._normalize_optional_string_list(None, name="x") is None
    assert contracts._normalize_optional_string_list(" CO(S) ", name="x") == ["CO(S)"]
    assert contracts._normalize_optional_string_list(["CO(S)", " ", "CO(S)"], name="x") == ["CO(S)"]

    with pytest.raises(ValueError):
        contracts._normalize_optional_string_list(1, name="x")

    with pytest.raises(ValueError):
        contracts._normalize_optional_string_list(["CO(S)", 1], name="x")

    assert contracts._validate_external_method_label("CO(S)", name="x") == "CO(S)"
    with pytest.raises(ValueError):
        contracts._validate_external_method_label("   ", name="x")
    with pytest.raises(ValueError):
        contracts._validate_external_method_label("A::B", name="x")
    with pytest.raises(ValueError):
        contracts._validate_external_method_label("not-a-method", name="x")


def test_normalize_external_method_selector_covers_validation_surface() -> None:
    assert (
        contracts.normalize_external_method_selector(None, fu_code="L1.a", argument_name="ext")
        is None
    )

    raw_bad: Any = "bad"
    with pytest.raises(ValueError):
        contracts.normalize_external_method_selector(
            raw_bad,
            fu_code="L1.a",
            argument_name="ext",
        )

    with pytest.raises(ValueError):
        contracts.normalize_external_method_selector(
            {"bad": ["CO(S)"]},
            fu_code="L1.a",
            argument_name="ext",
        )

    assert contracts.normalize_external_method_selector(
        {"l1_methods": ["CO(S)", "CO(S)", " "]},
        fu_code="L1.a",
        argument_name="ext",
    ) == {"l1_methods": ["CO(S)"]}

    with pytest.raises(ValueError):
        contracts.normalize_external_method_selector(
            {"one_step_methods": ["UT(FD)"]},
            fu_code="L1.a",
            argument_name="ext",
        )

    with pytest.raises(ValueError):
        contracts.normalize_external_method_selector(
            {"l1_methods": ["CO(S)"]},
            fu_code="L2.a.a",
            argument_name="ext",
        )

    with pytest.raises(ValueError):
        contracts.normalize_external_method_selector(
            {"l1_l2_pairs": ["CO(S)"]},
            fu_code="L2.a.a",
            argument_name="ext",
        )
    with pytest.raises(ValueError):
        contracts.normalize_external_method_selector(
            {"l1_l2_pairs": ["::UT(FD)"]},
            fu_code="L2.a.a",
            argument_name="ext",
        )

    with pytest.raises(ValueError):
        contracts.normalize_external_method_selector({}, fu_code="L2.a.a", argument_name="ext")

    assert contracts.normalize_external_method_selector(
        {"one_step_methods": ["UT(FD)"], "l1_l2_pairs": ["CO(S)::UT(FD)"]},
        fu_code="L2.a.a",
        argument_name="ext",
    ) == {
        "one_step_methods": ["UT(FD)"],
        "l1_l2_pairs": ["CO(S)::UT(FD)"],
    }


def test_external_method_selection_and_iteration_cover_level_specific_contracts() -> None:
    level1 = contracts.ExternalMethodSelection(
        fu_code="L1.a",
        l2_method=None,
        l1_method="CO(S)",
        level="level_1",
    )
    assert level1.l1_l2_method == "CO(S)"
    assert level1.file_method_token == "CO(S)"
    assert level1.asocc_method_label == "CO(S)"
    assert level1.user_label == "CO(S)"

    level2_one_step = contracts.ExternalMethodSelection(
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        l1_method=None,
        level="level_2",
    )
    assert level2_one_step.l1_l2_method == "UT(FD)"
    assert level2_one_step.file_method_token == "UT(FD)"
    assert level2_one_step.user_label == "UT(FD)"

    level2_pair = contracts.ExternalMethodSelection(
        fu_code="L2.a.a",
        l2_method="UT(FD)",
        l1_method="CO(S)",
        level="level_2",
    )
    assert level2_pair.l1_l2_method == "CO(S)_UT(FD)"
    assert level2_pair.user_label == "CO(S)::UT(FD)"

    level1_selections = contracts.iter_external_method_selections(
        external_method={"l1_methods": ["CO(S)"]},
        fu_code="L1.a",
    )
    assert [item.asocc_method_label for item in level1_selections] == ["CO(S)"]

    level2_selections = contracts.iter_external_method_selections(
        external_method={
            "one_step_methods": ["UT(FD)"],
            "l1_l2_pairs": ["CO(S)::UT(FD)"],
        },
        fu_code="L2.a.a",
    )
    assert [item.asocc_method_label for item in level2_selections] == [
        "UT(FD)",
        "CO(S)_UT(FD)",
    ]


def test_external_selection_labels_collision_checks_and_selector_merge() -> None:
    assert contracts.external_selection_labels(external_method=None, fu_code="L2.a.a") == []
    assert contracts.external_selection_labels(
        external_method={
            "one_step_methods": ["UT(FD)"],
            "l1_l2_pairs": ["CO(S)::UT(FD)"],
        },
        fu_code="L2.a.a",
    ) == ["UT(FD)", "CO(S)_UT(FD)"]

    contracts.validate_external_method_collisions(
        native_labels=None,
        external_method={"one_step_methods": ["UT(FD)"]},
        fu_code="L2.a.a",
        where="here",
    )
    contracts.validate_external_method_collisions(
        native_labels=["native"],
        external_method=None,
        fu_code="L2.a.a",
        where="here",
    )
    contracts.validate_external_method_collisions(
        native_labels=["native"],
        external_method={"one_step_methods": ["UT(FD)"]},
        fu_code="L2.a.a",
        where="here",
    )
    with pytest.raises(ValueError):
        contracts.validate_external_method_collisions(
            native_labels=["UT(FD)"],
            external_method={"one_step_methods": ["UT(FD)"]},
            fu_code="L2.a.a",
            where="here",
        )

    assert (
        contracts.merge_external_selector_methods(
            target_selector=None,
            external_method={"one_step_methods": ["UT(FD)"]},
            fu_code="L2.a.a",
        )
        is None
    )
    assert contracts.merge_external_selector_methods(
        target_selector={"methods": ["native"]},
        external_method=None,
        fu_code="L2.a.a",
    ) == {"methods": ["native"]}
    assert contracts.merge_external_selector_methods(
        target_selector={"years": [2019]},
        external_method={"one_step_methods": ["UT(FD)"]},
        fu_code="L2.a.a",
    ) == {"years": [2019]}
    assert contracts.merge_external_selector_methods(
        target_selector={"methods": ["native", "UT(FD)"]},
        external_method={
            "one_step_methods": ["UT(FD)"],
            "l1_l2_pairs": ["CO(S)::UT(FD)"],
        },
        fu_code="L2.a.a",
    ) == {"methods": ["CO(S)_UT(FD)", "UT(FD)", "native"]}
