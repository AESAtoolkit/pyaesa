import pytest

from pyaesa.asocc.methods.registry.registry import REGISTRY
from pyaesa.asocc.runtime.selection import plans
from pyaesa.asocc.runtime.selection import resolve as resolve_mod


def test_plans_core() -> None:
    assert "AR(E^{PBA})" not in plans.compatible_l1_for_l1_fu(fu_norm="L1.a")
    assert "AR(E^{CBA_FD})" not in plans.compatible_l1_for_l1_fu(fu_norm="L1.b")
    assert plans.compatible_l1_for_l1_fu(fu_norm="L2.a.a") == REGISTRY.list_l1_methods()

    assert plans.validate_l2_selection(
        fu_norm="L2.a.a",
        l1_weighting=False,
        selection=None,
        resolved_selection=[],
        label="one-step",
    ) == sorted(REGISTRY.list_l2_methods(fu_code="L2.a.a", l1_weighting=False))

    with pytest.raises(ValueError):
        plans.validate_l2_selection(
            fu_norm="L2.a.a",
            l1_weighting=False,
            selection=["BAD"],
            resolved_selection=["BAD"],
            label="one-step",
        )

    assert plans.resolve_l1_for_two_step([]) == REGISTRY.list_l1_methods()
    assert plans.resolve_l1_for_two_step(["AR(E^{CBA_FD})", "AR(E^{CBA_FD})"]) == ["AR(E^{CBA_FD})"]

    with pytest.raises(ValueError):
        plans.build_cartesian_pairs(
            l2_methods=["UT(FD)"],
            l1_methods=["AR(Ecap^{PBA})"],
        )


def test_plans_l1_and_l2_modes() -> None:
    with pytest.raises(ValueError):
        plans.resolve_l1_plan(
            fu_norm="L1.a",
            plan="pairs",
            l1_user=[],
            one_step_methods=None,
            two_step_methods=None,
            l1_l2_pairs=None,
        )
    with pytest.raises(ValueError):
        plans.resolve_l1_plan(
            fu_norm="L1.a",
            plan="default",
            l1_user=[],
            one_step_methods=["UT(FD)"],
            two_step_methods=None,
            l1_l2_pairs=None,
        )
    assert plans.resolve_l1_plan(
        fu_norm="L1.a",
        plan="default",
        l1_user=[],
        one_step_methods=None,
        two_step_methods=None,
        l1_l2_pairs=None,
    ) == (plans.compatible_l1_for_l1_fu(fu_norm="L1.a"), [], [])
    assert plans.resolve_l1_plan(
        fu_norm="L1.a",
        plan="default",
        l1_user=["EG(Pop)"],
        one_step_methods=None,
        two_step_methods=None,
        l1_l2_pairs=None,
    ) == (["EG(Pop)"], [], [])

    l1, combined, one = plans.resolve_default_l2_plan(
        fu_norm="L2.a.a",
        l1_user=["AR(E^{CBA_FD})"],
        one_step_methods=["UT(FD)"],
        two_step_methods=["UT(FD)"],
        l1_l2_pairs=None,
        one_step_user=["UT(FD)"],
        two_step_user=["UT(FD)"],
    )
    assert l1 == ["AR(E^{CBA_FD})"]
    assert combined == [("UT(FD)", "AR(E^{CBA_FD})")]
    assert one == ["UT(FD)"]

    l1, combined, one = plans.resolve_default_l2_plan(
        fu_norm="L2.a.a",
        l1_user=[],
        one_step_methods=["UT(FD)"],
        two_step_methods=[],
        l1_l2_pairs=None,
        one_step_user=["UT(FD)"],
        two_step_user=[],
    )
    assert (l1, combined, one) == ([], [], ["UT(FD)"])
    with pytest.raises(ValueError):
        plans.resolve_default_l2_plan(
            fu_norm="L2.a.a",
            l1_user=[],
            one_step_methods=["UT(FD)"],
            two_step_methods=["UT(FD)"],
            l1_l2_pairs=["AR(E)::UT(FD)"],
            one_step_user=["UT(FD)"],
            two_step_user=["UT(FD)"],
        )

    with pytest.raises(ValueError):
        plans.resolve_one_step_l2_plan(
            fu_norm="L2.a.a",
            l1_methods=["AR(E^{CBA_FD})"],
            two_step_methods=None,
            l1_l2_pairs=None,
            one_step_methods=None,
            one_step_user=[],
        )

    with pytest.raises(ValueError):
        plans.resolve_two_steps_l2_plan(
            fu_norm="L2.a.a",
            l1_user=["AR(E^{CBA_FD})"],
            one_step_methods=["UT(FD)"],
            l1_l2_pairs=None,
            two_step_methods=["UT(FD)"],
            two_step_user=["UT(FD)"],
        )

    with pytest.raises(ValueError):
        plans.resolve_one_step_pairs_l2_plan(
            fu_norm="L2.a.a",
            l1_methods=None,
            two_step_methods=None,
            one_step_methods=None,
            one_step_user=[],
            pairs_user=[],
        )
    with pytest.raises(ValueError):
        plans.resolve_one_step_pairs_l2_plan(
            fu_norm="L2.a.a",
            l1_methods=["AR(E^{CBA_FD})"],
            two_step_methods=None,
            one_step_methods=["UT(FD)"],
            one_step_user=["UT(FD)"],
            pairs_user=[("UT(FD)", "AR(E^{CBA_FD})")],
        )

    with pytest.raises(ValueError):
        plans.resolve_pairs_l2_plan(
            fu_norm="L2.a.a",
            l1_methods=None,
            one_step_methods=None,
            two_step_methods=None,
            pairs_user=[],
        )
    with pytest.raises(ValueError):
        plans.resolve_pairs_l2_plan(
            fu_norm="L2.a.a",
            l1_methods=["AR(E^{CBA_FD})"],
            one_step_methods=None,
            two_step_methods=None,
            pairs_user=[("UT(FD)", "AR(E^{CBA_FD})")],
        )


def test_resolve_and_dispatch() -> None:
    with pytest.raises(ValueError):
        resolve_mod.resolve_l1_user(fu_norm="L2.x", l1_methods=["AR({.})"])
    assert resolve_mod.resolve_l1_user(
        fu_norm="L2.a.a",
        l1_methods=["AR(E^{CBA_FD})"],
    ) == ["AR(E^{CBA_FD})"]
    assert resolve_mod.resolve_l1_user(
        fu_norm="L1.a",
        l1_methods=["AR(E^{CBA_FD})"],
    ) == ["AR(E^{CBA_FD})"]
    assert resolve_mod.resolve_l1_user(
        fu_norm="bad_fu",
        l1_methods=["EG(Pop)"],
    ) == ["EG(Pop)"]
    assert resolve_mod.resolve_l2_user(fu_norm="L2.a.a", names=["UT(FD)"]) == ["UT(FD)"]
    assert resolve_mod.resolve_l2_user(fu_norm="L2.a.a", names=None) == []

    with pytest.raises(ValueError):
        resolve_mod.resolve_pairs_user(fu_norm="L2.a.a", l1_l2_pairs=["bad"])

    assert resolve_mod.resolve_pairs_user(
        fu_norm="L2.a.a",
        l1_l2_pairs=["AR(E^{CBA_FD})::UT(FD)"],
    ) == [("UT(FD)", "AR(E^{CBA_FD})")]

    assert resolve_mod.resolve_method_selection(
        fu_code="L2.a.a",
        method_plan="default",
        l1_methods=["AR(E^{CBA_FD})"],
        one_step_methods=["UT(FD)"],
        two_step_methods=["UT(FD)"],
        l1_l2_pairs=None,
    ) == (["AR(E^{CBA_FD})"], [("UT(FD)", "AR(E^{CBA_FD})")], ["UT(FD)"])

    assert resolve_mod.resolve_method_selection(
        fu_code="L2.a.a",
        method_plan="one_step",
        l1_methods=None,
        one_step_methods=["UT(FD)"],
        two_step_methods=None,
        l1_l2_pairs=None,
    ) == ([], [], ["UT(FD)"])

    assert resolve_mod.resolve_method_selection(
        fu_code="L2.a.a",
        method_plan="two_steps",
        l1_methods=["AR(E^{CBA_FD})"],
        one_step_methods=None,
        two_step_methods=["UT(FD)"],
        l1_l2_pairs=None,
    ) == (["AR(E^{CBA_FD})"], [("UT(FD)", "AR(E^{CBA_FD})")], [])

    assert resolve_mod.resolve_method_selection(
        fu_code="L2.a.a",
        method_plan="one_step_pairs",
        l1_methods=None,
        one_step_methods=["UT(FD)"],
        two_step_methods=None,
        l1_l2_pairs=["AR(E^{CBA_FD})::UT(FD)"],
    ) == (["AR(E^{CBA_FD})"], [("UT(FD)", "AR(E^{CBA_FD})")], ["UT(FD)"])

    assert resolve_mod.resolve_method_selection(
        fu_code="L2.a.a",
        method_plan="pairs",
        l1_methods=None,
        one_step_methods=None,
        two_step_methods=None,
        l1_l2_pairs=["AR(E^{CBA_FD})::UT(FD)"],
    ) == (["AR(E^{CBA_FD})"], [("UT(FD)", "AR(E^{CBA_FD})")], [])
