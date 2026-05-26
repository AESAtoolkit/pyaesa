from types import SimpleNamespace

import pandas as pd
import pytest

from pyaesa.asocc.data.paths import _get_mrio_year_dir
from pyaesa.asocc.data.reference_payloads import (
    _reference_lcia_reason_suffix,
    load_ar_l2_reference_lcia_payload,
    load_reference_lcia_reg,
    load_reference_lcia_reg_for_domain,
    ensure_pr_hr_child_impact_timeseries_loaded,
)


def _reference_context(*, fu_code: str = "L2.a.a") -> SimpleNamespace:
    return SimpleNamespace(
        source="oecd_v2025",
        agg_version=None,
        fu_code=fu_code,
        filters={
            "r_p": ["FR"],
            "s_p": ["D"],
            "r_c": ["FR"],
            "r_f": ["FR"],
            "r_u": ["FR"],
        },
        needs_lcia=True,
        lcia_methods=["gwp100_lcia"],
        selected_l1=["AR(E^{CBA_FD})", "AR(E^{PBA})"],
        selected_l2_one_step=[
            "AR(E^{CBA_FD})",
            "AR(E^{CBA_TD})",
            "AR(E^{PBA})",
        ],
        combined=[],
        historical_years=[2005, 2006],
    )


def _reference_state() -> SimpleNamespace:
    return SimpleNamespace(
        skipped_years={},
        lcia_metadata_cache={},
        lcia_available_years_cache={},
        lcia_method_payload_cache={},
        cf_by_method={},
        lcia_units={},
        lcia_timeseries={},
        lcia_timeseries_original={},
    )


def test_load_ar_l2_reference_lcia_payload_slices_real_payloads(
    allocation_dummy_repo,
) -> None:
    context = _reference_context(fu_code="L2.a.b")
    state = _reference_state()

    lcia_ref = load_ar_l2_reference_lcia_payload(
        context=context,
        state=state,
        ref_year=2005,
        lcia_key="gwp100_lcia",
    )

    assert list(lcia_ref["e_cba_td_rp_sp"].columns.get_level_values("r_p")) == ["FR"]
    assert list(lcia_ref["e_cba_td_rp_sp"].columns.get_level_values("s_p")) == ["D"]


def test_load_reference_lcia_reg_uses_skipped_year_reason_and_domain_wrapper(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    context = _reference_context(fu_code="L2.a.b")
    state = _reference_state()

    cba_fd = load_reference_lcia_reg(
        context=context,
        state=state,
        ref_year=2005,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        agg_version=None,
    )
    pba = load_reference_lcia_reg(
        context=context,
        state=state,
        ref_year=2005,
        lcia_method="gwp100_lcia",
        lcia_kind="PBA",
        agg_version=None,
    )
    wrapped = load_reference_lcia_reg_for_domain(
        context=context,
        state=state,
        ref_year=2005,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        use_original_domain=True,
    )
    assert cba_fd.index.name == "impact"
    assert pba.columns.name == "r_p"
    assert wrapped.equals(cba_fd)


def test_load_reference_lcia_reg_reports_skipped_year_reason(
    allocation_dummy_repo_factory,
) -> None:
    repo = allocation_dummy_repo_factory(name="reference_payloads_unavailable_lcia")
    repo.write_lcia_support(
        source="oecd_v2025",
        matrix_version=None,
        lcia_method="gwp100_lcia",
        available_years=[2006],
    )
    context = _reference_context()
    state = _reference_state()

    with pytest.raises(ValueError):
        load_reference_lcia_reg(
            context=context,
            state=state,
            ref_year=2005,
            lcia_method="gwp100_lcia",
            lcia_kind="CBA_FD",
            agg_version=None,
        )
    assert state.skipped_years[2005]["gwp100_lcia"] == "Dummy LCIA unavailable"


def test_reference_payload_contracts_cover_reason_suffix() -> None:
    state_with_non_dict_reason = _reference_state()
    state_with_non_dict_reason.skipped_years[2005] = "bad state"
    assert (
        _reference_lcia_reason_suffix(
            state=state_with_non_dict_reason,
            ref_year=2005,
            lcia_method="gwp100_lcia",
        )
        == ""
    )

    state_with_missing_method = _reference_state()
    state_with_missing_method.skipped_years[2005] = {"other_method": "unavailable"}
    assert (
        _reference_lcia_reason_suffix(
            state=state_with_missing_method,
            ref_year=2005,
            lcia_method="gwp100_lcia",
        )
        == ""
    )


def test_load_reference_lcia_payload_and_reg_cover_missing_method_and_missing_key(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    context = _reference_context()
    state = _reference_state()

    saved_dir = _get_mrio_year_dir(source=context.source, year=2005, agg_version=None)
    state.lcia_method_payload_cache[
        (
            None,
            str(saved_dir),
            "gwp100_lcia",
        )
    ] = {
        "e_cba_fd_reg": pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.Index(["climate_parent"], name="impact"),
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
    }
    state.skipped_years[2005] = {"gwp100_lcia": "cached payload incomplete"}

    with pytest.raises(ValueError):
        load_reference_lcia_reg(
            context=context,
            state=state,
            ref_year=2005,
            lcia_method="gwp100_lcia",
            lcia_kind="PBA",
            agg_version=None,
        )


def test_ensure_pr_hr_child_impact_timeseries_loaded_handles_missing_files(
    allocation_dummy_repo,
    allocation_dummy_repo_factory,
) -> None:
    context = _reference_context()
    state = _reference_state()
    saved_dir = _get_mrio_year_dir(source=context.source, year=2005, agg_version=None)
    missing_metric = (
        saved_dir / "enacting_metrics" / "level_1" / "gwp100_lcia" / "e_cba_fd_reg.pickle"
    )
    missing_metric.unlink()

    loaded = ensure_pr_hr_child_impact_timeseries_loaded(
        context=context,
        state=state,
        through_year=2006,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        use_original_domain=False,
    )

    assert 2005 not in loaded
    assert 2006 in loaded

    original_repo = allocation_dummy_repo_factory(name="reference_payloads_original_lcia")
    assert original_repo.repo_root.exists()
    original_context = SimpleNamespace(
        source="oecd_v2025",
        agg_version=None,
        fu_code="L2.a.a",
        filters=context.filters,
        needs_lcia=True,
        lcia_methods=["gwp100_lcia"],
        historical_years=[2005],
    )
    original_state = _reference_state()
    original_loaded = ensure_pr_hr_child_impact_timeseries_loaded(
        context=original_context,
        state=original_state,
        through_year=2005,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        use_original_domain=True,
    )

    assert 2005 in original_loaded
    assert (
        original_state.lcia_timeseries_original["gwp100_lcia"]["CBA_FD"][2005].index.name
        == "impact"
    )

    cached_state = _reference_state()
    cached_frame = pd.DataFrame(
        [[1.0, 2.0]],
        index=pd.Index(["climate_parent"], name="impact"),
        columns=pd.Index(["FR", "US"], name="r_f"),
    )
    cached_state.lcia_timeseries["gwp100_lcia"] = {"CBA_FD": {2005: cached_frame}, "PBA": {}}
    cached_loaded = ensure_pr_hr_child_impact_timeseries_loaded(
        context=context,
        state=cached_state,
        through_year=2006,
        lcia_method="gwp100_lcia",
        lcia_kind="CBA_FD",
        use_original_domain=False,
    )

    assert cached_loaded[2005] is cached_frame
    assert 2006 in cached_loaded


def test_load_ar_l2_reference_lcia_payload_uses_normal_filtered_axes(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    context = _reference_context(fu_code="L2.a.a")
    state = _reference_state()

    lcia_ref = load_ar_l2_reference_lcia_payload(
        context=context,
        state=state,
        ref_year=2005,
        lcia_key="gwp100_lcia",
    )

    assert list(lcia_ref["e_cba_fd_reg"].columns) == ["FR"]
    assert list(lcia_ref["e_cba_fd_rp_sp"].columns.get_level_values("r_p")) == ["FR"]
    assert list(lcia_ref["e_cba_fd_rp_sp"].columns.get_level_values("s_p")) == ["D"]
