import re

import pytest

from pyaesa.asocc.orchestration.setup.request import selection as mod
from pyaesa.shared.selectors.path_tokens import (
    build_selector_filter_segment,
    deduplicated_selector_value_tokens,
    safe_path_token,
    selector_axis_token,
    selector_scope_request_axes_token,
)


def test_sanitize_and_encode_cover_edge_tokens() -> None:
    assert safe_path_token("  ") == "missing"
    assert safe_path_token("CON") == "_CON"
    assert safe_path_token("A/B:C*D?") == "A_B_C_D"
    assert safe_path_token("abcdefghijklmnopqrstuvwxyz", max_len=16) == "abcdefghijklmnop"
    assert selector_axis_token("s_p") == "sp"
    assert deduplicated_selector_value_tokens(
        ["Manufacture of basic metals", "Manufacture of basic plastics"],
        max_len=16,
    ) == {
        "Manufacture of basic metals": "Manufacture_of_b",
        "Manufacture of basic plastics": "Manufacture_of_b_2",
    }
    assert selector_scope_request_axes_token((("r_p", ("FR",)), ("s_p", None))) == "rp_FR_sp_all"


def test_build_filter_segment_and_normalize_filter_paths() -> None:
    assert build_selector_filter_segment(key="r_p", values=[" ", ""]) == ""

    long_values = [f"very_very_long_sector_name_{i}_{'x' * 40}" for i in range(30)]
    segment = build_selector_filter_segment(key="s_p", values=long_values)
    assert segment.startswith("s_p-n")
    assert len(segment) <= 120

    assert mod.normalize_filter(None) is None
    assert mod.normalize_filter([" A ", "", "B", " "]) == ["A", "B"]
    assert mod.normalize_filter([" ", ""]) is None
    assert mod.build_indices_tag({"r_p": [" ", ""], "s_p": None, "r_c": None, "r_f": None}) == (
        "all_indices"
    )


def test_apply_filter_messages_valid_and_invalid_cases() -> None:
    out = mod.apply_filter_messages(
        fu_code="L2.a.a",
        expected_indices=("r_p", "s_p"),
        filters={"r_p": ["FR"], "s_p": None, "r_c": None, "r_f": None},
    )
    assert out == {"r_p": ["FR"], "s_p": None, "r_c": None, "r_f": None}

    expected = "Selector 's_p' is not valid for fu_code='L1.b'. Expected selectors: r_p."
    with pytest.raises(ValueError, match=re.escape(expected)):
        mod.apply_filter_messages(
            fu_code="L1.b",
            expected_indices=("r_p",),
            filters={"r_p": ["FR"], "s_p": ["A"], "r_c": None, "r_f": None},
        )


@pytest.mark.parametrize(
    ("fu_code", "selector", "expected"),
    [
        ("L1.a", "s_p", "r_f"),
        ("L1.b", "s_p", "r_p"),
        ("L2.a.a", "r_f", "r_p, s_p"),
        ("L2.a.b", "r_f", "r_p, s_p"),
        ("L2.a.c", "r_c", "r_p, s_p"),
        ("L2.b.a", "r_c", "r_p, s_p, r_f"),
        ("L2.b.b", "r_f", "r_p, s_p, r_c"),
        ("L2.c.a", "r_p", "s_p, r_f"),
        ("L2.c.b", "r_p", "s_p, r_c"),
    ],
)
def test_resolve_filters_rejects_unexpected_public_selector_by_fu(
    fu_code: str,
    selector: str,
    expected: str,
) -> None:
    values = {"r_p": None, "s_p": None, "r_c": None, "r_f": None}
    values[selector] = ["D"] if selector == "s_p" else ["FR"]
    message = (
        f"Selector '{selector}' is not valid for fu_code='{fu_code}'. "
        f"Expected selectors: {expected}."
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        mod._resolve_filters(fu_code=fu_code, **values)


def test_resolve_l1_kinds_and_needs_lcia() -> None:
    l1_for_l1_fu = mod.resolve_l1_kinds(
        fu_code="L1.a",
        l1_lcia_kind="CBA_TD",
        combined=[("UT(FD)", "EG(Pop)")],
    )
    assert l1_for_l1_fu == {"CBA_TD"}

    l1_for_l2_fu = mod.resolve_l1_kinds(
        fu_code="L2.a.b",
        l1_lcia_kind="CBA_TD",
        combined=[("UT(FD)", "EG(Pop)"), ("AR(E^{CBA_TD})", "AR(E^{CBA_TD})")],
    )
    assert l1_for_l2_fu == {"CBA_FD", "CBA_TD"}

    assert (
        mod.needs_lcia(
            fu_code="L2.a.a",
            selected_l1=["EG(Pop)"],
            combined=[("AR(E^{CBA_FD})", "EG(Pop)")],
            selected_l2_one_step=[],
        )
        is True
    )
    assert (
        mod.needs_lcia(
            fu_code="L2.a.b",
            selected_l1=["EG(Pop)"],
            combined=[],
            selected_l2_one_step=["AR(E^{CBA_TD})"],
        )
        is True
    )
    assert (
        mod.needs_lcia(
            fu_code="L2.a.a",
            selected_l1=["PR-HR(Ecap,cum^{CBA_FD})"],
            combined=[],
            selected_l2_one_step=[],
        )
        is True
    )
    assert (
        mod.needs_lcia(
            fu_code="L2.a.a",
            selected_l1=["EG(Pop)"],
            combined=[],
            selected_l2_one_step=["UT(FD)"],
        )
        is False
    )
