from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pyaesa.asocc.io.metadata import _run_scope_key
from pyaesa.asocc.orchestration.reporting_records import deterministic_asocc_info_messages
from pyaesa.asocc.orchestration.write.metadata import payload as mod
from pyaesa.asocc.runtime.paths.deterministic import stats_path_for_format


def _projection_context(*, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        mode="regression",
        reg_window=(2005, 2007),
        l2_reuse_years=(2030,),
        future_years=(2030,),
        max_historical_year=2022,
        l2_method_route_by_name={"UT(FD)": "regression"},
    )


def _context(
    *,
    proj_base: Path,
    output_source_label: str | None = None,
    projection_context: SimpleNamespace | None = None,
) -> SimpleNamespace:
    published_source = output_source_label or "oecd_v2025"
    return SimpleNamespace(
        proj_base=proj_base,
        source="oecd_v2025",
        output_source_label=output_source_label,
        output_source=published_source,
        group_version=None,
        group_reg=False,
        aggreg_indices=False,
        l1_reg_aggreg="post",
        project_name="aSoCC",
        group_sec=False,
        fu_code="L2.a.a",
        lcia_method=None,
        years_input=[2005, 2030],
        reference_years_input=[2005],
        ssp_scenario=None,
        variant_tag=None,
        output_format="csv",
        projection_context=projection_context,
        requested_years=[2030],
        resolved_years=[2030],
        wb_df=pd.DataFrame(columns=["2005"]),
        selected_methods={"L1": ["EG(Pop)"]},
        selected_l1=["EG(Pop)"],
        combined=[],
        selected_l2_one_step=[],
        studied_indices_tag="all",
        reference_years=[2005],
        filters={"bucket": ["regression"]},
        ssp_scenario_options=["SSP2"],
        run_signature={"source": "oecd_v2025", "projection_mode": "regression"},
    )


def _state(
    *,
    processed_years: list[int] | None = None,
    outputs_all: list[str] | None = None,
    outputs_written: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        processed_years=list(processed_years or []),
        outputs_all=list(outputs_all or []),
        outputs_written=list(outputs_written or []),
        skipped_years={},
        empty_ref_years=[],
    )


def _scope_payload(
    *,
    signature: dict,
    completed_years: list[int],
    outputs: list[str],
) -> dict:
    return {
        "function": "deterministic_asocc",
        "arguments": signature,
        "execution": {"completed_years": completed_years},
        "reuse": {"identity_key": _run_scope_key(signature=signature)},
        "artifacts": {"outputs": outputs},
        "provenance": {},
    }


def test_build_metadata_payload_uses_overrides_and_single_stats_path(tmp_path: Path) -> None:
    context = _context(
        proj_base=tmp_path,
        output_source_label="oecd_label",
        projection_context=_projection_context(),
    )
    stats_path = stats_path_for_format(
        proj_base=context.proj_base,
        output_format=context.output_format,
        source=context.output_source_label,
        group_version=context.group_version,
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text("header\n", encoding="utf-8")

    state = _state(
        processed_years=[2020],
        outputs_all=["state_all.csv"],
        outputs_written=["state_written.csv"],
    )
    state.startup_notices = [
        ("INFO", "startup info"),
        ("WARNING", "startup warning"),
        ("ERROR", "ignored"),
        ("INFO", " "),
    ]
    state.summary_warnings = ["summary warning", "", "summary warning"]
    payload = mod.build_metadata_payload(  # noqa: SLF001
        context=context,
        state=state,
        completed_years_override=[2032, 2031],
        outputs_override=["  report.csv  ", "", "summary.csv"],
    )

    assert payload["execution"]["completed_years"] == [2031, 2032]
    assert payload["artifacts"]["outputs"] == ["report.csv", "summary.csv"]
    assert payload["artifacts"]["regression_stats_path"] == str(stats_path)
    assert payload["artifacts"]["regression_stats_paths"] == [str(stats_path)]
    assert payload["provenance"]["projection"] == {
        "enabled": True,
        "mode": "regression",
        "reg_window": [2005, 2006, 2007],
        "l2_reuse_years": [2030],
    }
    summary_records = payload["summary_records"]
    summary_severities = [record["severity"] for record in summary_records]
    assert "ERROR" not in summary_severities
    assert "INFO" in summary_severities
    assert summary_severities.count("WARNING") == 2
    assert all(record["message"].strip() for record in summary_records)


def test_build_metadata_payload_uses_processed_years_without_prior_state(tmp_path: Path) -> None:
    context = _context(proj_base=tmp_path, projection_context=None)
    payload = mod.build_metadata_payload(  # noqa: SLF001
        context=context,
        state=_state(
            processed_years=[2021, 2020],
            outputs_all=["state_all.csv"],
            outputs_written=["state_written.csv"],
        ),
    )

    assert payload["execution"]["completed_years"] == [2020, 2021]
    assert payload["artifacts"]["outputs"] == ["state_all.csv"]
    assert payload["artifacts"]["regression_stats_path"] is None
    assert payload["artifacts"]["regression_stats_paths"] == []
    assert payload["provenance"]["projection"] == {
        "enabled": False,
        "mode": None,
        "reg_window": None,
        "l2_reuse_years": [],
    }
    reference_messages = deterministic_asocc_info_messages(
        context=SimpleNamespace(
            source="iso3",
            fu_code="L2.a.a",
            reference_years=None,
            historical_years=[2000],
            selected_l1=[],
            combined=[],
            selected_l2_one_step=["AR(E^{CBA_FD})"],
            ssp_scenario_options=[],
            requested_years=[],
            wb_df=pd.DataFrame(),
        )
    )
    assert len(reference_messages) == 1
    mixed_reference_messages = deterministic_asocc_info_messages(
        context=SimpleNamespace(
            source="iso3",
            fu_code="L2.a.a",
            reference_years=[2000],
            historical_years=[2000],
            selected_l1=["AR(E^{CBA_FD})"],
            combined=[("UT(FD)", "AR(E^{CBA_FD})")],
            selected_l2_one_step=["AR(E^{CBA_FD})"],
            ssp_scenario_options=[],
            requested_years=[],
            wb_df=pd.DataFrame(),
        )
    )
    assert len(mixed_reference_messages) == 1
    assert (
        deterministic_asocc_info_messages(
            context=SimpleNamespace(
                source="oecd_v2025",
                fu_code="L2.a.a",
                reference_years=None,
                historical_years=[],
                selected_l1=[],
                combined=[],
                selected_l2_one_step=["AR(E^{CBA_FD})"],
                ssp_scenario_options=["SSP2"],
                requested_years=[2005],
                wb_df=pd.DataFrame(columns=["2005"]),
            )
        )
        == []
    )
    ssp_messages = deterministic_asocc_info_messages(
        context=SimpleNamespace(
            source="oecd_v2025",
            fu_code="L2.a.a",
            reference_years=None,
            historical_years=[],
            selected_l1=["EG(Pop)"],
            combined=[("UT(FD)", "EG(Pop)")],
            selected_l2_one_step=["UT(FD)"],
            ssp_scenario_options=["SSP2"],
            requested_years=[2005, 2030],
            wb_df=pd.DataFrame(columns=["2005"]),
        )
    )
    assert len(ssp_messages) == 1
    unknown_window_context = _context(
        proj_base=tmp_path,
        projection_context=SimpleNamespace(
            enabled=True,
            mode="regression",
            reg_window=None,
            l2_reuse_years=(),
            future_years=(2030,),
            max_historical_year=2022,
            l2_method_route_by_name={"UT(FD)": "regression"},
        ),
    )
    unknown_window_context.combined = [("UT(FD)", "EG(Pop)")]
    unknown_window_context.selected_l2_one_step = ["UT(FD)"]
    unknown_window_messages = deterministic_asocc_info_messages(context=unknown_window_context)
    assert len(unknown_window_messages) == 2


def test_build_metadata_payload_merges_prior_append_state(tmp_path: Path) -> None:
    context = _context(proj_base=tmp_path, projection_context=None)
    payload = mod.build_metadata_payload(  # noqa: SLF001
        context=context,
        state=_state(
            processed_years=[2021],
            outputs_all=["old.csv", "new.csv"],
        ),
        prior_metadata=_scope_payload(
            signature=context.run_signature,
            completed_years=[2020],
            outputs=["old.csv"],
        ),
        merge_prior_current_scope=True,
    )

    assert payload["execution"]["completed_years"] == [2020, 2021]
    assert payload["artifacts"]["outputs"] == ["old.csv", "new.csv"]
