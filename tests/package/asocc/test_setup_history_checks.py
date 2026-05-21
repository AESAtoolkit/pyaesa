import pytest

from pyaesa.asocc.orchestration.setup.validation import history_checks as mod


def _selection(selected_l1=None, combined=None, selected_l2_one_step=None):
    return type(
        "Sel",
        (),
        {
            "selected_l1": list(selected_l1 or []),
            "combined": list(combined or []),
            "selected_l2_one_step": list(selected_l2_one_step or []),
        },
    )()


def test_requires_ar_or_prhr_history_detects_any_requiring_method() -> None:
    assert mod._requires_ar_or_prhr_history(
        selection=_selection(selected_l1=["AR(E^{CBA_FD})"]),
        fu_code="L2.a.a",
    )
    assert mod._requires_ar_or_prhr_history(
        selection=_selection(combined=[("UT(FD)", "AR(E^{CBA_FD})")]),
        fu_code="L2.a.a",
    )
    assert mod._requires_ar_or_prhr_history(
        selection=_selection(selected_l2_one_step=["AR(E^{CBA_FD})"]),
        fu_code="L2.a.a",
    )
    assert mod._requires_ar_or_prhr_history(
        selection=_selection(combined=[("AR(E^{CBA_FD})", "EG(Pop)")]),
        fu_code="L2.a.a",
    )
    assert not mod._requires_ar_or_prhr_history(
        selection=_selection(selected_l1=["EG(Pop)"]),
        fu_code="L2.a.a",
    )


def test_validate_history_since_baseline_branches() -> None:
    mod._validate_history_since_baseline(
        source="oecd_v2025",
        group_version=None,
        group_reg=False,
        group_sec=False,
        historical_years=[2019],
        selection=_selection(selected_l1=["AR(E^{CBA_FD})"]),
        fu_code="L2.a.a",
    )

    mod._validate_history_since_baseline(
        source="exiobase_396_ixi",
        group_version=None,
        group_reg=False,
        group_sec=False,
        historical_years=[2019],
        selection=_selection(selected_l1=["EG(Pop)"]),
        fu_code="L2.a.a",
    )


def test_validate_history_since_baseline_raises_when_missing_years() -> None:
    with pytest.raises(ValueError, match="1996"):
        mod._validate_history_since_baseline(
            source="exiobase_396_ixi",
            group_version=None,
            group_reg=False,
            group_sec=False,
            historical_years=[1995, 1997],
            selection=_selection(selected_l1=["AR(E^{CBA_FD})"]),
            fu_code="L2.a.a",
        )


def test_validate_history_since_baseline_passes_when_contiguous() -> None:
    mod._validate_history_since_baseline(
        source="exiobase_396_ixi",
        group_version=None,
        group_reg=False,
        group_sec=False,
        historical_years=[1995, 1996, 1997],
        selection=_selection(selected_l1=["AR(E^{CBA_FD})"]),
        fu_code="L2.a.a",
    )
