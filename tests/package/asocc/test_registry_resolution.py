import importlib

import pytest

mod = importlib.import_module("pyaesa.asocc.methods.registry.queries.resolve")


class _FakeRegistry:
    def required_indices(self, name, fu_code, l1_weighting=None):
        if name == "EG(Pop)":
            return ("base",)
        if l1_weighting is True:
            return ("r_c",)
        if l1_weighting is False:
            return ("s_p",)
        return ()

    def method_family(self, name, level=None):
        if name == "EG(Pop)" and level == "L1":
            return "EG_POP"
        return "UT_FD"


def test_resolve_user_l1_method_name() -> None:
    assert mod.resolve_user_l1_method_name(" AR(E^{PBA}) ", l1_kind="PBA") == "AR(E^{PBA})"
    assert mod.resolve_user_l1_method_name(" EG(Pop) ") == "EG(Pop)"
    with pytest.raises(ValueError):
        mod.resolve_user_l1_method_name("AR(E^{PBA})", l1_kind="CBA_FD")
    with pytest.raises(ValueError):
        mod.resolve_user_l1_method_name("UT(FD)")
    with pytest.raises(ValueError):
        mod.resolve_user_l1_method_name("AR(E)")


def test_resolve_user_l2_method_name() -> None:
    assert mod.resolve_user_l2_method_name(name="AR(E^{PBA})", fu_code="L2.a.c") == "AR(E^{PBA})"
    assert mod.resolve_user_l2_method_name(name="UT(FD)", fu_code="L2.a.a") == "UT(FD)"
    with pytest.raises(ValueError):
        mod.resolve_user_l2_method_name(name="AR(E^{PBA})", fu_code="L1.a")
    with pytest.raises(ValueError):
        mod.resolve_user_l2_method_name(name="AR(E)", fu_code="L2.a.c")


def test_resolve_required_indices() -> None:
    fake = _FakeRegistry()

    out_l2 = mod.resolve_required_indices(
        fu_code="L2.a.a",
        selected_l1=["EG(Pop)"],
        combined=[("UT(TD)", "AR(E^{CBA_FD})")],
        selected_l2_one_step=["UT(FD)"],
        l1_kinds_needed={"CBA_FD", "PBA"},
        registry=fake,
        normalize_fu_code=lambda fu: fu,
    )
    assert out_l2 == {"base", "r_c", "s_p", "r_f", "r_p"}

    out_l1a = mod.resolve_required_indices(
        fu_code="L1.a",
        selected_l1=["EG(Pop)"],
        combined=[],
        selected_l2_one_step=[],
        l1_kinds_needed=set(),
        registry=fake,
        normalize_fu_code=lambda fu: fu,
    )
    assert out_l1a == {"base", "r_f"}

    out_l1b = mod.resolve_required_indices(
        fu_code="L1.b",
        selected_l1=["EG(Pop)"],
        combined=[],
        selected_l2_one_step=[],
        l1_kinds_needed=set(),
        registry=fake,
        normalize_fu_code=lambda fu: fu,
    )
    assert out_l1b == {"base", "r_p"}

    # Non neutral L1 family should skip inferred side enrichment.
    out_non_neutral = mod.resolve_required_indices(
        fu_code="L2.a.a",
        selected_l1=["UT(FD)"],
        combined=[],
        selected_l2_one_step=[],
        l1_kinds_needed={"CBA_FD", "PBA"},
        registry=fake,
        normalize_fu_code=lambda fu: fu,
    )
    assert out_non_neutral == set()

    # L2-neutral inference with partial and empty l1_kinds coverage.
    out_l2_cba_only = mod.resolve_required_indices(
        fu_code="L2.a.a",
        selected_l1=["EG(Pop)"],
        combined=[],
        selected_l2_one_step=[],
        l1_kinds_needed={"CBA_FD"},
        registry=fake,
        normalize_fu_code=lambda fu: fu,
    )
    assert out_l2_cba_only == {"base", "r_f"}

    out_l2_no_kind = mod.resolve_required_indices(
        fu_code="L2.a.a",
        selected_l1=["EG(Pop)"],
        combined=[],
        selected_l2_one_step=[],
        l1_kinds_needed=set(),
        registry=fake,
        normalize_fu_code=lambda fu: fu,
    )
    assert out_l2_no_kind == {"base"}
