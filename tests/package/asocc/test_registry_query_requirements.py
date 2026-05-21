import importlib
from typing import Any

import pytest

queries = importlib.import_module("pyaesa.asocc.methods.registry.queries.queries")
requirements = importlib.import_module("pyaesa.asocc.methods.registry.model.input_requirements")
model = importlib.import_module("pyaesa.asocc.methods.registry.model.types")


def _spec(
    *,
    name: str,
    level: str,
    fu_code: str | None,
    l1_weighting: bool = False,
    needs_lcia: bool = False,
    needs_rp: bool = False,
    indices: tuple[str, ...] = (),
    l1_kind: str | None = None,
    l2_weight_axis: str | None = None,
    expand_ar_years: bool = True,
    family: str,
) -> Any:
    return model.MethodSpec(
        name=name,
        level=level,
        fu_code=fu_code,
        l1_weighting=l1_weighting,
        needs_lcia=needs_lcia,
        needs_pop=False,
        needs_gdp=False,
        needs_utility=False,
        needs_rp=needs_rp,
        indices=indices,
        l1_kind=l1_kind,
        l2_weight_axis=l2_weight_axis,
        expand_ar_years=expand_ar_years,
        family=family,
    )


def _registry() -> Any:
    methods = [
        _spec(name="EG(Pop)", level="L1", fu_code="L1.a", family="EG_POP"),
        _spec(name="EG(Pop)", level="L1", fu_code="L1.b", family="EG_POP"),
        _spec(name="EG(Pop)", level="L1", fu_code="L2.a.a", family="EG_POP"),
        _spec(
            name="AR(E^{CBA_FD})",
            level="L1",
            fu_code="L1.a",
            l1_kind="CBA_FD",
            family="AR_E",
        ),
        _spec(
            name="AR(Ecap^{PBA})",
            level="L1",
            fu_code="L1.b",
            l1_kind="PBA",
            needs_lcia=True,
            needs_rp=True,
            family="AR_ECAP",
        ),
        _spec(
            name="PR-HR(Ecap,cum^{PBA})",
            level="L1",
            fu_code="L1.b",
            l1_kind="PBA",
            needs_lcia=True,
            family="PR_HR",
        ),
        _spec(
            name="UT(FD)",
            level="L2",
            fu_code="L2.a.a",
            indices=("r_f",),
            l1_kind="CBA_FD",
            family="UT_FD",
        ),
        _spec(
            name="UT(TD)",
            level="L2",
            fu_code="L2.a.b",
            l1_weighting=True,
            indices=("r_c",),
            l1_kind="CBA_TD",
            l2_weight_axis="r_f",
            family="UT_TD",
        ),
        _spec(
            name="AR(E^{CBA_FD})",
            level="L2",
            fu_code="L2.a.a",
            l1_weighting=True,
            indices=("r_f",),
            l1_kind="CBA_FD",
            l2_weight_axis="r_f",
            expand_ar_years=False,
            family="AR_E",
        ),
    ]
    return queries.MethodRegistry(methods)


def test_registry_basic_queries_and_required_indices() -> None:
    reg = _registry()

    assert len(reg.all_methods()) == 9
    assert reg.list_l1_methods() == [
        "AR(E^{CBA_FD})",
        "AR(Ecap^{PBA})",
        "EG(Pop)",
        "PR-HR(Ecap,cum^{PBA})",
    ]
    assert reg.list_l2_methods(fu_code="L2.a.a", l1_weighting=None) == [
        "AR(E^{CBA_FD})",
        "UT(FD)",
    ]
    assert reg.list_l2_methods(fu_code="L2.a.a", l1_weighting=False) == ["UT(FD)"]
    assert len(reg.get_method("UT(FD)", level="L2")) == 1
    assert reg.has_method("UT(FD)", level="L2", fu_code="L2.a.a", l1_weighting=False)
    assert not reg.has_method("UT(FD)", level="L1")
    assert not reg.has_method("UT(FD)", level="L2", fu_code="L2.b.a")

    assert set(reg.required_indices("EG(Pop)", "L1.a")) == {"r_f"}
    assert set(reg.required_indices("EG(Pop)", "L1.b")) == {"r_p"}
    assert set(reg.required_indices("EG(Pop)", "L2.a.a")) == {"r_f"}
    assert set(reg.required_indices("UT(TD)", "L2.a.b", l1_weighting=True)) == {"r_c"}
    assert reg.required_indices("UT(TD)", "L2.a.b", l1_weighting=False) == ()
    assert reg.required_indices("EG(Pop)", None) == ()

    reg.validate_selection("L2.a.a", ["UT(FD)"], l1_weighting=False)
    with pytest.raises(ValueError):
        reg.validate_selection("L2.a.a", ["UT(FDa)"], l1_weighting=False)


def test_registry_flags_kinds_axis_and_family_checks() -> None:
    reg = _registry()

    assert reg.method_requires_lcia("AR(Ecap^{PBA})", fu_code="L1.b")
    assert reg.method_requires_rp("AR(Ecap^{PBA})", fu_code="L1.b")
    assert not reg.method_requires_lcia("EG(Pop)", fu_code="L1.a")
    assert not reg.method_requires_rp("EG(Pop)", fu_code="L1.a")

    assert reg.l1_kind_for_l2_method("UT(TD)") == "CBA_TD"
    with pytest.raises(ValueError):
        reg.l1_kind_for_l2_method("missing")
    with pytest.raises(ValueError):
        queries.MethodRegistry(
            [
                _spec(
                    name="UT(X)",
                    level="L2",
                    fu_code="L2.a.a",
                    l1_kind="CBA_FD",
                    family="UT_FD",
                ),
                _spec(
                    name="UT(X)",
                    level="L2",
                    fu_code="L2.a.b",
                    l1_kind="CBA_TD",
                    family="UT_FD",
                ),
            ]
        ).l1_kind_for_l2_method("UT(X)")

    assert (
        queries.MethodRegistry(
            [
                _spec(
                    name="AR(single)",
                    level="L2",
                    fu_code="L2.a.a",
                    l1_weighting=True,
                    l2_weight_axis="r_f",
                    expand_ar_years=False,
                    family="AR_E",
                )
            ]
        ).expand_ar_years_for_method("AR(single)")
        is False
    )
    with pytest.raises(ValueError):
        reg.expand_ar_years_for_method("missing")
    with pytest.raises(ValueError):
        queries.MethodRegistry(
            [
                _spec(
                    name="AR(X)",
                    level="L1",
                    fu_code="L1.a",
                    expand_ar_years=False,
                    family="AR_E",
                ),
                _spec(
                    name="AR(X)",
                    level="L2",
                    fu_code="L2.a.a",
                    expand_ar_years=True,
                    family="AR_E",
                ),
            ]
        ).expand_ar_years_for_method("AR(X)")

    assert reg.l2_weight_axis_for_method("UT(TD)", fu_code="L2.a.b") == "r_f"
    with pytest.raises(ValueError):
        reg.l2_weight_axis_for_method("UT(TD)", fu_code="L2.a.a")
    with pytest.raises(ValueError):
        queries.MethodRegistry(
            [
                _spec(
                    name="UT(X)",
                    level="L2",
                    fu_code="L2.a.a",
                    l1_weighting=True,
                    l2_weight_axis="r_f",
                    family="UT_FD",
                ),
                _spec(
                    name="UT(X)",
                    level="L2",
                    fu_code="L2.a.a",
                    l1_weighting=True,
                    l2_weight_axis="r_p",
                    family="UT_FD",
                ),
            ]
        ).l2_weight_axis_for_method("UT(X)", fu_code="L2.a.a")

    assert reg.method_family("UT(FD)", level="L2", fu_code="L2.a.a") == "UT_FD"
    assert reg.method_family("UT(FD)", fu_code="L2.a.a", l1_weighting=False) == "UT_FD"
    assert reg.method_family("AR(E^{CBA_FD})", level="L2") == "AR_E"
    assert reg.method_family("AR(E^{CBA_FD})", fu_code="L2.a.a") == "AR_E"
    assert reg.method_family("AR(E^{CBA_FD})", l1_weighting=True) == "AR_E"
    with pytest.raises(ValueError):
        reg.method_family("missing")
    with pytest.raises(ValueError):
        queries.MethodRegistry(
            [
                _spec(name="X", level="L1", fu_code="L1.a", family="AR_E"),
                _spec(name="X", level="L2", fu_code="L2.a.a", family="UT_FD"),
            ]
        ).method_family("X")

    assert reg.method_is_ar("AR(Ecap^{PBA})", level="L1")
    assert reg.method_is_ar_cap("AR(Ecap^{PBA})", level="L1")
    assert reg.method_is_ut("UT(FD)", level="L2", fu_code="L2.a.a")


def test_registry_input_requirement_wrappers() -> None:
    reg = _registry()

    assert reg.method_requires_contiguous_history("AR(Ecap^{PBA})", level="L1")
    assert reg.method_requires_lcia_percap("AR(Ecap^{PBA})", level="L1")
    assert reg.method_requires_pr_hr_cumulative(
        "PR-HR(Ecap,cum^{PBA})",
        level="L1",
    )
    assert reg.l2_base_enacting_metrics("UT(FD)", fu_code="L2.a.a") == (
        "fd_rf",
        "fd_rp_sp",
        "fd_rp_sp_rf",
    )
    assert reg.lcia_enacting_metric_l1_metrics("AR(Ecap^{PBA})", level="L1") == ("e_pba_reg",)
    assert reg.lcia_enacting_metric_l2_metrics(
        "AR(E^{CBA_FD})",
        level="L2",
        fu_code="L2.a.a",
        l1_weighting=True,
    ) == ("e_cba_fd_rp_sp_rf",)
    assert reg.l1_kinds_for_method("AR(Ecap^{PBA})") == ["PBA"]


def test_input_requirement_functions() -> None:
    methods = [
        _spec(name="M1", level="L1", fu_code="L1.a", l1_kind="CBA_FD", family="AR_E"),
        _spec(name="M1", level="L1", fu_code="L1.b", l1_kind="PBA", family="AR_E"),
        _spec(name="M2", level="L2", fu_code="L2.a.a", l1_kind="CBA_TD", family="UT_TD"),
        _spec(name="M3", level="L1", fu_code="L1.a", l1_kind=None, family="EG_POP"),
    ]
    assert requirements.lcia_kinds_for_method(
        methods=methods,
        name="M1",
        level="L1",
        fu_code=None,
        l1_weighting=None,
    ) == {"CBA_FD", "PBA"}
    assert requirements.method_requires_contiguous_history(family="AR_E")
    assert not requirements.method_requires_contiguous_history(family="UT_FD")
    assert requirements.method_requires_lcia_percap(family="AR_ECAP")
    assert not requirements.method_requires_lcia_percap(family="AR_E")
    assert requirements.method_requires_pr_hr_cumulative(family="PR_HR")
    assert not requirements.method_requires_pr_hr_cumulative(family="AR_E")

    assert requirements.l2_base_enacting_metrics(family="UT_FD", fu_code="L2.a.a") == (
        "fd_rf",
        "fd_rp_sp",
        "fd_rp_sp_rf",
    )
    assert requirements.l2_base_enacting_metrics(family="UT_FD", fu_code="missing") == ()
    assert requirements.lcia_enacting_metric_l1_metrics(lcia_kinds={"CBA_FD", "PBA"}) == (
        "e_cba_fd_reg",
        "e_pba_reg",
    )

    generic = requirements.lcia_enacting_metric_l2_metrics(lcia_kinds={"CBA_FD", "PBA"})
    assert "e_cba_fd_rp_sp" in generic
    assert "e_pba_rp_sp" in generic

    one_step = requirements.lcia_enacting_metric_l2_metrics(
        lcia_kinds={"CBA_FD"},
        fu_code="L2.a.a",
        l1_weighting=False,
    )
    assert one_step == ("e_cba_fd_rp_sp",)
    two_step = requirements.lcia_enacting_metric_l2_metrics(
        lcia_kinds={"CBA_FD"},
        fu_code="L2.a.a",
        l1_weighting=True,
    )
    assert two_step == ("e_cba_fd_rp_sp_rf",)
    both = requirements.lcia_enacting_metric_l2_metrics(
        lcia_kinds={"CBA_FD"},
        fu_code="L2.a.a",
        l1_weighting=None,
    )
    assert both == ("e_cba_fd_rp_sp", "e_cba_fd_rp_sp_rf")

    with pytest.raises(ValueError):
        requirements.lcia_enacting_metric_l2_metrics(
            lcia_kinds={"UNKNOWN"},
            fu_code="L2.a.a",
            l1_weighting=False,
        )

    # Cover fu_code and l1_weighting mismatch filter branches.
    assert (
        requirements.lcia_kinds_for_method(
            methods=methods,
            name="M2",
            level="L2",
            fu_code="L2.x.x",
            l1_weighting=None,
        )
        == set()
    )
    assert (
        requirements.lcia_kinds_for_method(
            methods=methods,
            name="M2",
            level="L2",
            fu_code="L2.a.a",
            l1_weighting=True,
        )
        == set()
    )
    assert (
        requirements.lcia_kinds_for_method(
            methods=methods,
            name="M3",
            level="L1",
            fu_code=None,
            l1_weighting=None,
        )
        == set()
    )
