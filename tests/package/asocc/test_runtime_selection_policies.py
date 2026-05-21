import pytest

from pyaesa.asocc.runtime.selection import pair_policy


def test_pair_policy_neutral_and_kind_filter() -> None:
    assert pair_policy.is_l1_neutral_method("EG(Pop)")
    assert not pair_policy.is_l1_neutral_method("AR(E^{CBA_FD})")
    assert pair_policy.is_l1_compatible_with_kind(l1_method="EG(Pop)", required_kind="PBA")
    assert pair_policy.is_l1_compatible_with_kind(
        l1_method="AR(E^{CBA_FD})",
        required_kind="CBA_FD",
    )
    assert not pair_policy.is_l1_compatible_with_kind(
        l1_method="AR(E^{CBA_FD})",
        required_kind="PBA",
    )
    assert pair_policy.filter_l1_for_two_step_kinds(
        l2_methods=["UT(FD)"],
        l1_methods=["AR(E^{CBA_FD})", "AR(Ecap^{PBA})", "EG(Pop)"],
    ) == ["AR(E^{CBA_FD})", "EG(Pop)"]
    assert (
        pair_policy.filter_l1_for_two_step_kinds(
            l2_methods=[],
            l1_methods=["AR(E^{CBA_FD})"],
        )
        == []
    )


def test_pair_policy_validate_explicit_pairs_and_errors() -> None:
    l1, combined = pair_policy.validate_explicit_pairs(
        fu_norm="L2.a.a",
        pairs=[("UT(FD)", "AR(E^{CBA_FD})")],
    )
    assert l1 == ["AR(E^{CBA_FD})"]
    assert combined == [("UT(FD)", "AR(E^{CBA_FD})")]

    with pytest.raises(ValueError):
        pair_policy.validate_explicit_pairs(
            fu_norm="L2.a.a",
            pairs=[("BAD", "AR(E^{CBA_FD})")],
        )


def test_pair_policy_apply_ar_rule() -> None:
    combined = [("AR(E^{CBA_FD})", "AR(E^{CBA_FD})"), ("UT(FD)", "AR(E^{CBA_FD})")]
    filtered, one_step = pair_policy.apply_ar_pair_policy_by_plan(
        fu_norm="L2.a.a",
        plan="default",
        combined=combined,
        one_step=[],
    )
    assert ("AR(E^{CBA_FD})", "AR(E^{CBA_FD})") not in filtered
    assert ("UT(FD)", "AR(E^{CBA_FD})") in filtered
    assert "AR(E^{CBA_FD})" in one_step

    with pytest.raises(ValueError):
        pair_policy.apply_ar_pair_policy_by_plan(
            fu_norm="L2.a.a",
            plan="pairs",
            combined=[("AR(E^{CBA_FD})", "AR(E^{CBA_FD})")],
            one_step=[],
        )
    with pytest.raises(ValueError):
        pair_policy.apply_ar_pair_policy_by_plan(
            fu_norm="L2.a.a",
            plan="one_step_pairs",
            combined=[("AR(E^{CBA_FD})", "AR(E^{CBA_FD})")],
            one_step=[],
        )

    filtered_custom, one_step_custom = pair_policy.apply_ar_pair_policy_by_plan(
        fu_norm="L2.a.a",
        plan="custom_plan",
        combined=[("AR(E^{CBA_FD})", "AR(E^{CBA_FD})")],
        one_step=[],
    )
    assert filtered_custom == [("AR(E^{CBA_FD})", "AR(E^{CBA_FD})")]
    assert one_step_custom == []
