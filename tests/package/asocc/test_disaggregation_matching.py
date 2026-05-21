from types import SimpleNamespace

import pytest

from pyaesa.asocc.disaggregation import matching as mod
from pyaesa.asocc.io.metadata import _build_run_metadata, _save_run_metadata
from pyaesa.asocc.orchestration.setup.run_setup import _prepare_context
from pyaesa.asocc.orchestration.setup.request.types import PrepareContextRequest


class _Signature:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def as_dict(self) -> dict:
        return dict(self._payload)


class _Scope:
    def __init__(
        self,
        *,
        scope_key: str,
        signature: dict,
        completed_years: list[int],
        ssp_scenarios: list[str] | None = None,
    ) -> None:
        self.scope_key = scope_key
        self.compute_signature = _Signature(signature)
        self.completed_years = completed_years
        self.ssp_scenarios = [] if ssp_scenarios is None else ssp_scenarios

    def covers_years(self, years: list[int]) -> bool:
        return set(int(year) for year in years).issubset(self.completed_years)


def _signature(**overrides) -> dict:
    signature = {
        "source": "oecd_v2025",
        "group_version": None,
        "group_reg": False,
        "group_sec": False,
        "fu_code": "L2.a.a",
        "studied_indices_tag": "demo",
        "l1_reg_aggreg": "post",
        "variant_tag": None,
        "aggreg_indices": False,
        "projection_mode": None,
        "reg_window": None,
        "lcia_methods": ["gwp100_lcia"],
        "reference_years_input": [2005],
        "l2_reuse_years": [2005],
        "ssp_scenario_input": ["SSP2"],
        "selected_methods": {"L2": ["UT(FD)"]},
        "selectors": {},
    }
    signature.update(overrides)
    return signature


def _request() -> PrepareContextRequest:
    return PrepareContextRequest(
        project_name="disaggregation_matching",
        source="oecd_v2025",
        group_version=None,
        group_reg=False,
        group_sec=False,
        years=[2005],
        historical_year_cap=None,
        refresh=False,
        lcia_method=None,
        fu_code="L2.a.a",
        r_p=None,
        s_p=["D"],
        r_c=None,
        r_f=None,
        l_1=None,
        l_2_combined_with_l_1=None,
        l_2_one_step=["UT(FD)"],
        reference_years=None,
        ssp_scenario=None,
        projection_mode=None,
        reg_window=None,
        l2_reuse_years=None,
        l1_reg_aggreg="post",
        variant_tag=None,
        aggreg_indices=False,
        output_format="csv",
        intermediate_outputs=False,
        output_source_label=None,
    )


def test_pick_scope_prefers_exact_minimal_scope_and_reports_missing_years() -> None:
    request_signature = _signature()
    catalog = SimpleNamespace(
        scopes=[
            _Scope(
                scope_key="wider",
                signature=_signature(),
                completed_years=[2005, 2006, 2007],
                ssp_scenarios=["SSP2"],
            ),
            _Scope(
                scope_key="exact",
                signature=_signature(),
                completed_years=[2005, 2006],
                ssp_scenarios=["SSP2"],
            ),
            _Scope(
                scope_key="wrong_method",
                signature=_signature(selected_methods={"L2": ["UT(GVA)"]}),
                completed_years=[2005, 2006],
                ssp_scenarios=["SSP2"],
            ),
        ],
        run_ssp_scenarios=["SSP2"],
    )

    scope_key, scope_signature, completed_years = mod._pick_scope(  # noqa: SLF001
        request_signature=request_signature,
        catalog=catalog,
        requested_years=[2005, 2006],
    )

    assert scope_key == "exact"
    assert scope_signature["source"] == "oecd_v2025"
    assert completed_years == [2005, 2006]

    with pytest.raises(ValueError, match="2008 \\(1 year\\)"):
        mod._pick_scope(  # noqa: SLF001
            request_signature=request_signature,
            catalog=catalog,
            requested_years=[2005, 2008],
        )


def test_build_user_rerun_message_carries_selector_scope() -> None:
    request = SimpleNamespace(
        source="oecd_v2025",
        group_reg=False,
        group_sec=True,
        group_version="elec",
        s_p=["Electricity"],
        aggreg_indices=False,
        l1_reg_aggreg="post",
    )
    message = mod._build_user_rerun_message(  # noqa: SLF001
        selector_name="target",
        request=request,
        missing_years=[2005, 2006, 2008],
    )
    assert message
    assert "target" in message
    assert "elec" in message
    assert "2005" in message
    assert "2006" in message
    assert "2008" in message

    ungrouped_message = mod._build_user_rerun_message(  # noqa: SLF001
        selector_name="target",
        request=SimpleNamespace(
            source="oecd_v2025",
            group_reg=False,
            group_sec=False,
            group_version=None,
            s_p=["D"],
            aggreg_indices=True,
            l1_reg_aggreg="post",
        ),
        missing_years=[2005],
    )
    assert ungrouped_message
    assert "grouped" in ungrouped_message


def test_match_selector_scope_reports_missing_metadata(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError, match="No deterministic_asocc run metadata found"):
        mod.match_selector_scope(
            selector_name="target",
            request=_request(),
            requested_years=[2005],
        )


def test_match_selector_scope_reports_missing_requested_years(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    request = _request()
    context, _state, _skipped = _prepare_context(request=request)
    metadata_path = mod._run_metadata_path_for_request(  # noqa: SLF001
        selector_name="target",
        context=context,
    )
    existing_output = metadata_path.parent / "existing_output.csv"
    existing_output.write_text("year,value\n2005,1.0\n", encoding="utf-8")
    scope_payload = _build_run_metadata(
        requested_years=[2005],
        resolved_years=[2005],
        selected_methods=dict(context.selected_methods),
        fu_code=str(context.fu_code),
        studied_indices_tag=str(context.studied_indices_tag),
        skipped_years={},
        outputs=[str(existing_output)],
        signature=context.run_signature,
    )
    scope_payload["execution"]["completed_years"] = [2005]
    _save_run_metadata(metadata_path, scope_payload)

    with pytest.raises(ValueError, match="Missing prerequisite deterministic_asocc outputs"):
        mod.match_selector_scope(
            selector_name="target",
            request=request,
            requested_years=[2006],
        )
