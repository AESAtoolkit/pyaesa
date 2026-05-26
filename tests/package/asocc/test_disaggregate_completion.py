from dataclasses import replace
from pathlib import Path

import pytest

from pyaesa.asocc.disaggregation.completion import (
    is_disaggregation_branch_complete,
    write_branch_metadata,
    write_scope_manifest,
)
from pyaesa.asocc.disaggregation.config import parse_disaggregate_args
from pyaesa.asocc.disaggregation.models import MatchedRun, ParsedArgs
from pyaesa.asocc.disaggregation.paths import disaggregation_metadata_path
from pyaesa.asocc.runtime.scope.branch_resolution import (
    allocate_run_metadata_path,
    path_scope_from_signature,
)
from pyaesa.shared.runtime.metadata.json import read_json_dict, write_json_dict


def _disaggregation_config() -> dict:
    return {
        "target_agg_run": {"source": "oecd_v2025", "s_p": ["Energy"]},
        "ref_agg_run": {
            "source": "exiobase_396_ixi",
            "agg_sec": True,
            "agg_version": "energy_aggregate",
            "s_p": ["Energy"],
        },
        "ref_disagg_run": {"source": "exiobase_396_ixi", "s_p": ["Coal"]},
        "disaggregation_specs": [{"agg_sector_label": "Energy", "disagg_sector_label": "Coal"}],
        "new_disagg_version_name": "disagg_oecd_energy",
    }


def _parsed(*, refresh: bool = False) -> ParsedArgs:
    return parse_disaggregate_args(
        disaggregation_config=_disaggregation_config(),
        base_allocate_args={
            "project_name": "disagg_completion",
            "years": [2005],
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
        },
        output_format="csv",
        figures=False,
        figure_options=None,
        figure_format=None,
        figure_external_method=None,
        refresh=refresh,
    )


def _run_signature() -> dict:
    return {
        "source": "disagg_oecd_energy",
        "agg_version": None,
        "agg_reg": False,
        "agg_sec": False,
        "fu_code": "L2.a.a",
        "studied_indices_tag": "all",
        "lcia_methods": [],
        "ssp_scenario_input": None,
        "reference_years_input": None,
        "selected_methods": {"l2_vs_global": ["UT(FD)"]},
        "l1_reg_aggreg": "post",
        "variant_tag": None,
        "group_indices": False,
        "output_format": "csv",
        "intermediate_outputs": False,
        "projection_mode": None,
        "reg_window": None,
        "l2_reuse_years": None,
        "historical_year_cap": None,
        "years": [2005],
    }


def _matched_runs(tmp_path: Path) -> dict[str, MatchedRun]:
    return {
        name: MatchedRun(
            selector_name=name,
            proj_base=tmp_path,
            run_metadata_path=tmp_path / f"{name}.json",
            scope_key=f"{name}_scope",
            scope_signature={"source": name},
            completed_years=[2005],
            output_source_label=name,
        )
        for name in ("target_agg_run", "ref_agg_run", "ref_disagg_run")
    }


def _write_complete_branch(tmp_path: Path) -> tuple[ParsedArgs, dict, dict, Path, Path, Path]:
    parsed = _parsed()
    run_signature = _run_signature()
    matched_runs = _matched_runs(tmp_path)
    final_output = tmp_path / "published" / "UT(FD).csv"
    final_output.parent.mkdir(parents=True, exist_ok=True)
    final_output.write_text("s_p,2005\nCoal,0.5\n", encoding="utf-8")
    scope = path_scope_from_signature(
        proj_base=tmp_path,
        source_label=parsed.disaggregation.new_disagg_version_name,
        run_signature=run_signature,
        context_label="test disaggregation scope",
    )
    manifest_path = allocate_run_metadata_path(scope=scope)
    scope_key = write_scope_manifest(
        manifest_path=manifest_path,
        parsed=parsed,
        run_signature=run_signature,
        requested_years=[2005],
        final_output_files=[final_output],
        ssp_scenario_options_by_year={2005: [None]},
    )
    metadata_path = disaggregation_metadata_path(
        proj_base=tmp_path,
        source_label=parsed.disaggregation.new_disagg_version_name,
        mode="post",
        group_indices=False,
    )
    write_branch_metadata(
        parsed=parsed,
        proj_base=tmp_path,
        run_signature=run_signature,
        requested_years=[2005],
        matched_runs=matched_runs,
        final_output_files=[final_output],
        audit_path=tmp_path / "audit.csv",
        metadata_path=metadata_path,
        disaggregated_scope_key=scope_key,
    )
    return parsed, run_signature, matched_runs, final_output, metadata_path, manifest_path


def _branch_complete(
    *,
    parsed: ParsedArgs,
    run_signature: dict,
    matched_runs: dict[str, MatchedRun],
    tmp_path: Path,
) -> bool:
    return is_disaggregation_branch_complete(
        parsed=parsed,
        proj_base=tmp_path,
        source_label=parsed.disaggregation.new_disagg_version_name,
        run_signature=run_signature,
        requested_years=[2005],
        matched_runs=matched_runs,
    )


def test_disaggregation_completion_accepts_exact_persisted_branch(tmp_path: Path) -> None:
    parsed, run_signature, matched_runs, _output, _metadata, _manifest = _write_complete_branch(
        tmp_path
    )

    assert _branch_complete(
        parsed=parsed,
        run_signature=run_signature,
        matched_runs=matched_runs,
        tmp_path=tmp_path,
    )


def test_disaggregation_completion_recomputes_when_refresh_is_requested(tmp_path: Path) -> None:
    parsed, run_signature, matched_runs, _output, _metadata, _manifest = _write_complete_branch(
        tmp_path
    )

    assert not _branch_complete(
        parsed=replace(parsed, refresh=True),
        run_signature=run_signature,
        matched_runs=matched_runs,
        tmp_path=tmp_path,
    )


@pytest.mark.parametrize(
    "case",
    [
        "missing_metadata",
        "changed_config",
        "missing_scope_mapping",
        "missing_scope_entry",
        "invalid_scope_entry",
        "changed_scope_key",
        "invalid_final_outputs",
        "empty_final_outputs",
        "missing_final_output",
        "missing_scope_manifest",
        "missing_manifest_scope",
        "incomplete_manifest_years",
        "missing_manifest_output",
    ],
)
def test_disaggregation_completion_recomputes_stale_file_states(
    tmp_path: Path,
    case: str,
) -> None:
    parsed, run_signature, matched_runs, final_output, metadata_path, manifest_path = (
        _write_complete_branch(tmp_path)
    )

    if case == "missing_metadata":
        metadata_path.unlink()
    elif case == "changed_config":
        payload = read_json_dict(metadata_path)
        payload["frozen_config"]["runtime"]["output_format"] = "parquet"
        write_json_dict(metadata_path, payload)
    elif case == "missing_scope_mapping":
        payload = read_json_dict(metadata_path)
        payload["exact_prior_allocate_cc_scopes"] = None
        write_json_dict(metadata_path, payload)
    elif case == "missing_scope_entry":
        payload = read_json_dict(metadata_path)
        del payload["exact_prior_allocate_cc_scopes"]["ref_disagg_run"]
        write_json_dict(metadata_path, payload)
    elif case == "invalid_scope_entry":
        payload = read_json_dict(metadata_path)
        payload["exact_prior_allocate_cc_scopes"]["ref_disagg_run"] = []
        write_json_dict(metadata_path, payload)
    elif case == "changed_scope_key":
        payload = read_json_dict(metadata_path)
        payload["exact_prior_allocate_cc_scopes"]["target_agg_run"]["scope_key"] = "other_scope"
        write_json_dict(metadata_path, payload)
    elif case == "invalid_final_outputs":
        payload = read_json_dict(metadata_path)
        payload["final_output_files"] = None
        write_json_dict(metadata_path, payload)
    elif case == "empty_final_outputs":
        payload = read_json_dict(metadata_path)
        payload["final_output_files"] = []
        write_json_dict(metadata_path, payload)
    elif case == "missing_final_output":
        final_output.unlink()
    elif case == "missing_scope_manifest":
        manifest_path.unlink()
    elif case == "missing_manifest_scope":
        write_json_dict(manifest_path, {})
    elif case == "incomplete_manifest_years":
        scope_payload = read_json_dict(manifest_path)
        scope_payload["execution"]["completed_years"] = []
        write_json_dict(manifest_path, scope_payload)
    elif case == "missing_manifest_output":
        scope_payload = read_json_dict(manifest_path)
        scope_payload["artifacts"]["outputs"] = [str(tmp_path / "missing_manifest_output.csv")]
        write_json_dict(manifest_path, scope_payload)

    assert not _branch_complete(
        parsed=parsed,
        run_signature=run_signature,
        matched_runs=matched_runs,
        tmp_path=tmp_path,
    )
