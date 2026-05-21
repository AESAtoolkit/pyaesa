import json
from pathlib import Path

import pytest

from pyaesa.asocc.orchestration.setup.run_setup import _prepare_context
from pyaesa.asocc.runtime.paths.deterministic import _get_allocate_refresh_scope_root
from pyaesa.asocc.orchestration.setup.request import selection as selection_mod
from pyaesa.asocc.orchestration.setup.request.types import PrepareContextRequest
from pyaesa.process.mrios.utils.io.paths import _get_metadata_path


def _request(**overrides) -> PrepareContextRequest:
    data = {
        "project_name": "setup_prepare_context",
        "source": "oecd_v2025",
        "group_version": None,
        "group_reg": False,
        "group_sec": False,
        "years": [2005],
        "historical_year_cap": None,
        "refresh": False,
        "lcia_method": None,
        "fu_code": "L1.a",
        "r_p": ["FR"],
        "s_p": None,
        "r_c": None,
        "r_f": None,
        "l_1": ["EG(Pop)"],
        "l_2_combined_with_l_1": None,
        "l_2_one_step": None,
        "reference_years": None,
        "ssp_scenario": None,
        "projection_mode": None,
        "reg_window": None,
        "l2_reuse_years": None,
        "l1_reg_aggreg": "post",
        "variant_tag": None,
        "aggreg_indices": False,
        "output_format": "csv",
        "intermediate_outputs": False,
        "output_source_label": None,
    }
    data.update(overrides)
    return PrepareContextRequest(**data)


def test_setup_selection_cover_validation_and_pruning() -> None:
    selection_mod._validate_td_grouped_output(fu_code="L2.a.a", aggreg_indices=False)
    with pytest.raises(ValueError):
        selection_mod._validate_td_grouped_output(fu_code="L2.a.b", aggreg_indices=True)

    grouping = selection_mod._resolve_grouping(
        group_reg=True,
        group_sec=None,
        group_version="demo",
    )
    assert grouping.apply_group_reg is True
    assert grouping.apply_group_sec is False
    assert grouping.group_version_reg == "demo"

    selection = selection_mod._resolve_selection_bundle(
        fu_code="L2.a.a",
        l_1=["EG(Pop)", "EG(Pop)"],
        l_2_combined_with_l_1=[("UT(FD)", "EG(Pop)"), ("UT(FD)", "EG(Pop)")],
        l_2_one_step=["UT(FD)", "UT(FD)"],
        l1_lcia_kind="CBA_FD",
    )
    assert selection.selected_l1 == ["EG(Pop)"]
    assert selection.combined == [("UT(FD)", "EG(Pop)")]
    assert selection.selected_l2_one_step == ["UT(FD)"]
    assert selection.selected_methods["l2_in_l1"] == ["EG(Pop)_UT(FD)"]

    pruned, dropped = selection_mod._prune_lcia_methods_without_lcia_input(
        fu_code="L2.a.a",
        lcia_methods=None,
        selection=selection_mod._resolve_selection_bundle(
            fu_code="L2.a.a",
            l_1=["AR(E^{CBA_FD})", "EG(Pop)"],
            l_2_combined_with_l_1=[("UT(FD)", "AR(E^{CBA_FD})")],
            l_2_one_step=["AR(E^{CBA_FD})", "UT(FD)"],
            l1_lcia_kind="CBA_FD",
        ),
    )
    assert pruned.selected_l1 == ["EG(Pop)"]
    assert pruned.selected_l2_one_step == ["UT(FD)"]
    assert dropped == [
        "AR(E^{CBA_FD})",
        "UT(FD)::AR(E^{CBA_FD})",
    ]

    unchanged, dropped = selection_mod._prune_lcia_methods_without_lcia_input(
        fu_code="L1.a",
        lcia_methods=["gwp100_lcia"],
        selection=selection_mod._resolve_selection_bundle(
            fu_code="L1.a",
            l_1=["AR(E^{CBA_FD})"],
            l_2_combined_with_l_1=None,
            l_2_one_step=None,
            l1_lcia_kind="CBA_FD",
        ),
    )
    assert dropped == []
    assert unchanged.selected_l1 == ["AR(E^{CBA_FD})"]

    with pytest.raises(ValueError):
        selection_mod._restrict_selection_for_iso3_mode(
            fu_code="L2.a.a",
            selection=selection,
        )
    with pytest.raises(ValueError):
        selection_mod._restrict_selection_for_iso3_mode(
            fu_code="L1.a",
            selection=selection_mod._resolve_selection_bundle(
                fu_code="L1.a",
                l_1=["AR(E^{CBA_FD})"],
                l_2_combined_with_l_1=None,
                l_2_one_step=None,
                l1_lcia_kind="CBA_FD",
            ),
        )
    iso3_selection = selection_mod._restrict_selection_for_iso3_mode(
        fu_code="L1.a",
        selection=selection_mod._resolve_selection_bundle(
            fu_code="L1.a",
            l_1=["AR(E^{CBA_FD})", "EG(Pop)", "PR(GDPcap)"],
            l_2_combined_with_l_1=None,
            l_2_one_step=None,
            l1_lcia_kind="CBA_FD",
        ),
    )
    assert iso3_selection.selected_l1 == ["EG(Pop)", "PR(GDPcap)"]

    filters, indices_tag = selection_mod._resolve_filters(
        required_indices={"r_p"},
        r_p=[" FR "],
        s_p=None,
        r_c=None,
        r_f=None,
    )
    assert filters == {"r_p": ["FR"], "s_p": None, "r_c": None, "r_f": None}
    assert indices_tag == "r_p-FR"
    assert selection_mod._resolve_output_domain_tag(source="iso3", group_version=None) is None
    assert (
        selection_mod._resolve_output_domain_tag(source="oecd_v2025", group_version=None)
        == "original_classification"
    )
    assert (
        selection_mod._resolve_output_domain_tag(source="oecd_v2025", group_version="demo")
        == "custom_classification_demo"
    )
    assert selection_mod._l1_methods_in_scope(selection) == {"EG(Pop)"}
    assert not selection_mod._uses_l1_post_original_domain(
        selection=selection,
        grouping=selection_mod._resolve_grouping(
            group_reg=False,
            group_sec=False,
            group_version=None,
        ),
        l1_reg_aggreg="pre",
    )
    assert selection_mod._uses_l1_post_original_domain(
        selection=selection_mod._resolve_selection_bundle(
            fu_code="L1.a",
            l_1=["AR(Ecap^{CBA_FD})"],
            l_2_combined_with_l_1=None,
            l_2_one_step=None,
            l1_lcia_kind="CBA_FD",
        ),
        grouping=selection_mod._resolve_grouping(
            group_reg=True,
            group_sec=False,
            group_version="demo",
        ),
        l1_reg_aggreg="post",
    )


def test_prepare_context_rejects_invalid_source_and_iso3_contracts(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        _prepare_context(request=_request(source=None))
    with pytest.raises(ValueError):
        _prepare_context(request=_request(source=" "))
    with pytest.raises(ValueError):
        _prepare_context(
            request=_request(
                source="iso3",
                fu_code="L1.a",
                group_reg=True,
                group_version="demo",
            )
        )
    with pytest.raises(ValueError):
        _prepare_context(
            request=_request(
                source="iso3",
                fu_code="L1.a",
                lcia_method="gwp100_lcia",
            )
        )
    with pytest.raises(ValueError):
        _prepare_context(
            request=_request(
                source="iso3",
                fu_code="L1.a",
                reference_years=[2005],
            )
        )


def test_prepare_context_restricts_iso3_selection_inside_run_setup(allocation_dummy_repo) -> None:
    context, state, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_iso3",
            source="iso3",
            fu_code="L1.a",
            years=[2005],
            r_p=None,
            l_1=["AR(E^{CBA_FD})", "EG(Pop)", "PR(GDPcap)"],
        )
    )
    assert is_complete is False
    assert context.source == "iso3"
    assert context.selected_l1 == ["EG(Pop)", "PR(GDPcap)"]
    assert context.selected_methods["l1"] == ["EG(Pop)", "PR(GDPcap)"]
    assert state.lcia_units == {}
    del allocation_dummy_repo


def test_prepare_context_rejects_when_lcia_pruning_removes_all_methods(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        _prepare_context(
            request=_request(
                source="oecd_v2025",
                l_1=["AR(E^{CBA_FD})"],
            )
        )


def test_prepare_context_builds_state_and_reuses_complete_scope(allocation_dummy_repo) -> None:
    full_years = list(range(1995, 2007))
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=full_years,
        scenario_years=[2030],
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_396_ixi",
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=full_years,
    )
    allocation_dummy_repo.write_mrio_history(
        source="exiobase_396_ixi",
        matrix_version=None,
        years=full_years,
    )
    allocation_dummy_repo.write_lcia_support(
        source="exiobase_396_ixi",
        matrix_version=None,
        lcia_method="gwp100_lcia",
    )
    context, state, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_success",
            source="exiobase_396_ixi",
            lcia_method="gwp100_lcia",
            l_1=["AR(E^{CBA_FD})"],
        )
    )
    assert is_complete is False
    assert context.source == "exiobase_396_ixi"
    assert state.lcia_units["gwp100_lcia"]["climate_parent"] == "kg CO2-eq / year"

    output_path = Path(context.proj_base) / "existing.csv"
    output_path.write_text("x", encoding="utf-8")
    allocation_dummy_repo.write_scope_metadata(
        context=context,
        completed_years=[2005],
        outputs=[output_path],
    )
    reused_context, reused_state, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_success",
            source="exiobase_396_ixi",
            lcia_method="gwp100_lcia",
            l_1=["AR(E^{CBA_FD})"],
        )
    )
    assert is_complete is True
    assert reused_context.run_signature == context.run_signature
    assert not hasattr(reused_state, "startup_notices")


def test_prepare_context_allows_year_append_for_incomplete_scope_metadata(
    allocation_dummy_repo,
) -> None:
    context, _, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_partial_scope",
            source="oecd_v2025",
            years=[2005, 2006],
            l_1=["EG(Pop)"],
            r_p=None,
        )
    )
    assert is_complete is False
    output_path = Path(context.proj_base) / "existing.csv"
    output_path.write_text("x", encoding="utf-8")
    allocation_dummy_repo.write_scope_metadata(
        context=context,
        completed_years=[2005],
        outputs=[output_path],
    )

    appended_context, _, appended_is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_partial_scope",
            source="oecd_v2025",
            years=[2005, 2006],
            l_1=["EG(Pop)"],
            r_p=None,
        )
    )

    assert appended_context.requested_years == [2005, 2006]
    assert appended_is_complete is False


def test_prepare_context_prunes_selector_append_to_missing_methods(
    allocation_dummy_repo,
) -> None:
    context, _, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_method_append",
            source="oecd_v2025",
            years=[2005],
            l_1=["EG(Pop)"],
            r_p=None,
        )
    )
    assert is_complete is False
    output_path = Path(context.proj_base) / "existing.csv"
    output_path.write_text("x", encoding="utf-8")
    allocation_dummy_repo.write_scope_metadata(
        context=context,
        completed_years=[2005],
        outputs=[output_path],
    )

    appended_context, _, appended_is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_method_append",
            source="oecd_v2025",
            years=[2005],
            l_1=["EG(Pop)", "PR(GDPcap)"],
            r_p=None,
        )
    )

    assert appended_is_complete is False
    assert appended_context.selected_l1 == ["PR(GDPcap)"]
    assert appended_context.run_signature["selected_methods"]["l1"] == [
        "EG(Pop)",
        "PR(GDPcap)",
    ]


def test_prepare_context_refresh_clears_existing_scope_outputs(allocation_dummy_repo) -> None:
    context, _, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_refresh",
            source="oecd_v2025",
            years=[2005],
            l_1=["EG(Pop)"],
            r_p=None,
        )
    )
    assert is_complete is False
    refresh_root = _get_allocate_refresh_scope_root(
        proj_base=Path(context.proj_base),
        source=context.output_source_label or context.source,
        group_version=context.group_version,
    )
    stale_file = refresh_root / "stale_scope.txt"
    stale_file.parent.mkdir(parents=True, exist_ok=True)
    stale_file.write_text("stale", encoding="utf-8")

    refreshed_context, _, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_refresh",
            source="oecd_v2025",
            years=[2005],
            l_1=["EG(Pop)"],
            r_p=None,
            refresh=True,
        )
    )

    assert is_complete is False
    assert Path(refreshed_context.proj_base) == Path(context.proj_base)
    assert not stale_file.exists()
    del allocation_dummy_repo


def test_prepare_context_refresh_handles_missing_scope_root(allocation_dummy_repo) -> None:
    context, _, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_refresh_missing_scope",
            source="oecd_v2025",
            years=[2005],
            l_1=["EG(Pop)"],
            r_p=None,
            refresh=True,
        )
    )
    assert is_complete is False
    assert context.source == "oecd_v2025"
    del allocation_dummy_repo


def test_prepare_context_refresh_clears_stale_outputs_without_scope_metadata(
    allocation_dummy_repo,
) -> None:
    context, _, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_refresh_missing_metadata",
            source="oecd_v2025",
            years=[2005],
            l_1=["EG(Pop)"],
            r_p=None,
        )
    )
    assert is_complete is False
    refresh_root = _get_allocate_refresh_scope_root(
        proj_base=Path(context.proj_base),
        source=context.output_source_label or context.source,
        group_version=context.group_version,
    )
    stale_output = refresh_root / "l1" / "eg_pop.csv"
    stale_output.parent.mkdir(parents=True, exist_ok=True)
    stale_output.write_text("stale", encoding="utf-8")

    refreshed_context, _, refreshed_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_refresh_missing_metadata",
            source="oecd_v2025",
            years=[2005],
            l_1=["EG(Pop)"],
            r_p=None,
            refresh=True,
        )
    )

    assert refreshed_complete is False
    assert Path(refreshed_context.proj_base) == Path(context.proj_base)
    assert not stale_output.exists()
    del allocation_dummy_repo


def test_prepare_context_covers_original_domain_lcia_setup_branch(
    allocation_dummy_repo,
) -> None:
    full_years = list(range(1995, 2007))
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=full_years,
        scenario_years=[2030],
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_396_ixi",
        matrix_version="demo_reg",
        sectors_used=["D", "X"],
        regions_used=["EU", "NAM"],
        years=full_years,
    )
    allocation_dummy_repo.write_mrio_history(
        source="exiobase_396_ixi",
        matrix_version="demo_reg",
        years=full_years,
    )
    allocation_dummy_repo.write_lcia_support(
        source="exiobase_396_ixi",
        matrix_version="demo_reg",
        lcia_method="gwp100_lcia",
    )

    context, state, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_original_domain",
            source="exiobase_396_ixi",
            group_version="demo_reg",
            group_reg=True,
            years=[2005],
            lcia_method="gwp100_lcia",
            l_1=["AR(Ecap^{CBA_FD})"],
            fu_code="L1.a",
            r_p=None,
        )
    )

    assert is_complete is False
    assert context.use_original_l1_post_domain is True
    assert state.lcia_units["gwp100_lcia"]["climate_parent"] == "kg CO2-eq / year"


def test_prepare_context_records_dropped_lcia_notice(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    context, state, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_drop_notice",
            source="oecd_v2025",
            years=[2005],
            l_1=["AR(E^{CBA_FD})", "EG(Pop)"],
            lcia_method=None,
            r_p=None,
        )
    )
    assert is_complete is False
    assert context.source == "oecd_v2025"
    assert any(
        level == "WARNING" and "AR(E^{CBA_FD})" in message
        for level, message in getattr(
            state,
            "startup_notices",
        )
    )


def test_prepare_context_records_projection_resolution_notice(allocation_dummy_repo) -> None:
    historical_years = list(range(1995, 2023))
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=historical_years,
        scenario_years=[2030],
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=historical_years,
    )
    allocation_dummy_repo.write_mrio_history(
        source="oecd_v2025",
        matrix_version=None,
        years=historical_years,
    )

    context, state, is_complete = _prepare_context(
        request=_request(
            project_name="setup_prepare_context_projection_notice",
            source="oecd_v2025",
            fu_code="L2.a.a",
            years=[2030],
            reg_window=[2005, 2006],
            r_p=None,
            l_1=None,
            l_2_one_step=["UT(FD)"],
        )
    )
    assert is_complete is False
    assert context.projection_context is not None
    assert context.projection_context.enabled is True
    assert context.requested_years == [2030]
    assert context.resolved_years == [2030]
    assert context.persisted_years == [2030]
    assert context.compute_years == [2005, 2006, 2030]
    assert getattr(state, "startup_notices") == []


def test_prepare_context_rejects_oecd_lcia_method(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError) as exc_info:
        _prepare_context(
            request=_request(
                project_name="setup_prepare_context_oecd_lcia_invalid",
                source="oecd_v2025",
                lcia_method="gwp100_lcia",
                l_1=["AR(E^{CBA_FD})"],
            )
        )
    assert "oecd_v2025" in str(exc_info.value)
    assert "lcia_method" in str(exc_info.value)


def test_prepare_context_rejects_missing_lcia_unit_metadata(allocation_dummy_repo) -> None:
    full_years = list(range(1995, 2007))
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=full_years,
        scenario_years=[2030],
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_396_ixi",
        matrix_version=None,
        sectors_used=["FR_S1", "FR_S2"],
        regions_used=["FR", "US"],
        years=full_years,
    )
    allocation_dummy_repo.write_mrio_history(
        source="exiobase_396_ixi",
        matrix_version=None,
        years=full_years,
    )
    allocation_dummy_repo.write_lcia_support(
        source="exiobase_396_ixi",
        matrix_version=None,
        lcia_method="gwp100_lcia",
    )
    metadata_path = _get_metadata_path("exiobase_396_ixi", matrix_version=None)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    years_payload = payload["years"]
    assert isinstance(years_payload, dict)
    for year_payload in years_payload.values():
        assert isinstance(year_payload, dict)
        enacting_metrics = year_payload["enacting_metrics"]
        assert isinstance(enacting_metrics, dict)
        units = enacting_metrics["units"]
        assert isinstance(units, dict)
        lcia_by_method = units["lcia_by_method"]
        assert isinstance(lcia_by_method, dict)
        lcia_by_method.pop("gwp100_lcia", None)
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError) as exc_info:
        _prepare_context(
            request=_request(
                project_name="setup_prepare_context_missing_units",
                source="exiobase_396_ixi",
                lcia_method="gwp100_lcia",
                l_1=["AR(E^{CBA_FD})"],
            )
        )
    assert "gwp100_lcia" in str(exc_info.value)
