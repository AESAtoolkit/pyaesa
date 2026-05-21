import importlib
from typing import Any

import pytest

facade = importlib.import_module("pyaesa.asocc.methods.registry.registry")
queries = importlib.import_module("pyaesa.asocc.methods.registry.queries.queries")
build = importlib.import_module("pyaesa.asocc.methods.registry.build.build")
model = importlib.import_module("pyaesa.asocc.methods.registry.model.types")


def test_normalize_fu_code_validation_paths() -> None:
    assert model.normalize_fu_code("L2..a..b") == "L2.a.b"
    with pytest.raises(ValueError):
        model.normalize_fu_code(None)
    with pytest.raises(ValueError):
        model.normalize_fu_code("invalid")


def test_registry_facade_uses_real_registry_contract() -> None:
    assert (
        facade.resolve_user_l1_method_name("AR(E^{CBA_FD})", l1_kind="CBA_FD") == "AR(E^{CBA_FD})"
    )
    assert facade.resolve_user_l2_method_name(name="UT(FD)", fu_code="L2.a.a") == "UT(FD)"
    assert facade.resolve_required_indices(
        fu_code="L2.a.a",
        selected_l1=["EG(Pop)"],
        combined=[("UT(FD)", "EG(Pop)")],
        selected_l2_one_step=["UT(FD)"],
        l1_kinds_needed={"CBA_FD"},
    ) == {"r_f", "r_p", "s_p"}


def test_registry_build_validation_errors() -> None:
    with pytest.raises(ValueError):
        build._required_str({}, "name")

    with pytest.raises(ValueError):
        build._required_indices({"indices": ["r_f"]}, method_name="M")

    with pytest.raises(ValueError):
        build._family_for_method(name="M", level="L9")

    with pytest.raises(ValueError):
        build._spec_from_raw(
            raw={"name": "UT(FD)", "level": "L2", "indices": (), "l1_weighting": False},
            normalize_fu_code=lambda fu: fu,
        )

    with pytest.raises(ValueError):
        build._spec_from_raw(
            raw={
                "name": "UT(FD)",
                "level": "L2",
                "fu_code": "L2.x.x",
                "indices": (),
                "l1_weighting": False,
            },
            normalize_fu_code=lambda fu: fu,
        )

    with pytest.raises(ValueError):
        build._spec_from_raw(
            raw={
                "name": "UT(FD)",
                "level": "L2",
                "fu_code": "L2.a.a",
                "indices": (),
                "l1_weighting": True,
            },
            normalize_fu_code=lambda fu: fu,
        )


def test_registry_query_additional_filter_paths() -> None:
    def _spec(
        *,
        name: str,
        level: str,
        fu_code: str | None,
        l1_weighting: bool,
        indices: tuple[str, ...],
        l1_kind: str | None,
        family: str,
        l2_weight_axis: str | None = None,
    ) -> Any:
        return model.MethodSpec(
            name=name,
            level=level,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
            needs_lcia=False,
            needs_pop=False,
            needs_gdp=False,
            needs_utility=False,
            needs_rp=False,
            indices=indices,
            l1_kind=l1_kind,
            l2_weight_axis=l2_weight_axis,
            expand_ar_years=True,
            family=family,
        )

    reg = queries.MethodRegistry(
        [
            _spec(
                name="M",
                level="L1",
                fu_code="L1.a",
                l1_weighting=False,
                indices=("idx",),
                l1_kind=None,
                family="AR_E",
            ),
            _spec(
                name="M",
                level="L1",
                fu_code="L2.a.c",
                l1_weighting=False,
                indices=(),
                l1_kind=None,
                family="AR_E",
            ),
            _spec(
                name="M",
                level="L1",
                fu_code="L2.a.b",
                l1_weighting=False,
                indices=(),
                l1_kind=None,
                family="AR_E",
            ),
            _spec(
                name="M",
                level="L2",
                fu_code="L2.a.c",
                l1_weighting=True,
                indices=("r_c",),
                l1_kind="PBA",
                family="UT_TD",
                l2_weight_axis="r_f",
            ),
            _spec(
                name="M",
                level="L1",
                fu_code="L2.a.a",
                l1_weighting=False,
                indices=(),
                l1_kind="OTHER",
                family="AR_E",
            ),
        ]
    )

    all_methods = reg.all_methods()
    assert len(all_methods) == 5
    all_methods.pop()
    assert len(reg.all_methods()) == 5

    assert reg.has_method("M", level="L9") is False
    assert reg.has_method("M", level="L1", fu_code="L9.x.x") is False
    assert reg.has_method("M", level="L2", fu_code="L2.a.c", l1_weighting=False) is False

    assert set(reg.required_indices("M", "L1.a")) == {"idx", "r_f"}
    assert set(reg.required_indices("M", "L2.a.c")) == {"r_p", "r_c"}
    assert set(reg.required_indices("M", "L2.a.b")) == {"r_f"}
    assert set(reg.required_indices("M", "L2.a.c", l1_weighting=True)) == {"r_c"}
    assert set(reg.required_indices("M", "L2.a.a")) == set()

    assert reg.method_family("M", level="L2", fu_code="L2.a.c", l1_weighting=True) == "UT_TD"
    assert reg.method_family("M", fu_code="L2.a.c", l1_weighting=True) == "UT_TD"
    with pytest.raises(ValueError):
        reg.method_family("M", level="L2", fu_code="L2.a.c", l1_weighting=False)
