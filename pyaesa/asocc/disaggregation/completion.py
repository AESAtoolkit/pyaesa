"""Metadata and completion helpers for published-output disaggregation."""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.metadata.json import read_optional_json_dict, write_json_dict
from pyaesa.shared.selectors.time_selectors import normalize_reg_window_for_storage

from ..io.metadata import (
    _build_run_metadata,
    _load_run_metadata,
    _run_scope_key,
    _save_run_metadata,
)
from pyaesa.asocc.runtime.scope.persisted_scope import load_asocc_persisted_run_catalog
from pyaesa.asocc.runtime.scope.branch_resolution import (
    allocate_run_metadata_path,
    path_scope_from_signature,
)
from .models import MatchedRun, ParsedArgs
from pyaesa.asocc.disaggregation.paths import disaggregation_metadata_path


def build_expected_disaggregation_frozen_config(*, parsed: ParsedArgs) -> dict[str, Any]:
    """Return the canonical frozen config stored for one disaggregation branch."""
    base_allocate_args = dict(parsed.base_allocate_args)
    base_allocate_args["reg_window"] = normalize_reg_window_for_storage(
        base_allocate_args.get("reg_window"),
        name="base_allocate_args.reg_window",
    )
    return {
        "disaggregation_config": asdict(parsed.disaggregation),
        "base_allocate_args": base_allocate_args,
        "runtime": {
            "output_format": parsed.output_format,
        },
    }


def _branch_metadata_path(
    *,
    proj_base: Path,
    source_label: str,
    run_signature: dict[str, Any],
) -> Path:
    """Return the canonical disaggregation metadata path for one branch."""
    return disaggregation_metadata_path(
        proj_base=proj_base,
        source_label=source_label,
        mode=str(run_signature["l1_reg_aggreg"]),
        group_indices=bool(run_signature["group_indices"]),
    )


def _scope_manifest_complete(
    *,
    manifest_path: Path,
    run_signature: dict[str, Any],
    requested_years: list[int],
) -> bool:
    """Return whether the persisted scope manifest matches the requested branch years."""
    if not manifest_path.exists():
        return False
    catalog = load_asocc_persisted_run_catalog(payload=_load_run_metadata(manifest_path))
    scope = catalog.scope_for_compute_signature(compute_signature=run_signature)
    if scope is None:
        return False
    if not scope.covers_years(requested_years):
        return False
    return all(Path(output).exists() for output in scope.outputs)


def is_disaggregation_branch_complete(
    *,
    parsed: ParsedArgs,
    proj_base: Path,
    source_label: str,
    run_signature: dict[str, Any],
    requested_years: list[int],
    matched_runs: dict[str, MatchedRun],
) -> bool:
    """Return whether the requested disaggregation branch is already complete."""
    if parsed.refresh:
        return False
    metadata_path = _branch_metadata_path(
        proj_base=proj_base,
        source_label=source_label,
        run_signature=run_signature,
    )
    payload = read_optional_json_dict(metadata_path)
    if not payload:
        return False
    if payload.get("frozen_config") != build_expected_disaggregation_frozen_config(parsed=parsed):
        return False
    stored_scopes = payload.get("exact_prior_allocate_cc_scopes")
    if not isinstance(stored_scopes, dict):
        return False
    for name, matched in matched_runs.items():
        entry = stored_scopes.get(name)
        if not isinstance(entry, dict):
            return False
        if str(entry.get("scope_key", "")).strip() != matched.scope_key:
            return False
    final_output_files = payload.get("final_output_files")
    if not isinstance(final_output_files, list) or not final_output_files:
        return False
    if not all(Path(str(path)).exists() for path in final_output_files):
        return False
    scope = path_scope_from_signature(
        proj_base=proj_base,
        source_label=source_label,
        run_signature=run_signature,
        context_label="Disaggregated scope manifest completion check",
    )
    return _scope_manifest_complete(
        manifest_path=allocate_run_metadata_path(scope=scope),
        run_signature=run_signature,
        requested_years=requested_years,
    )


def write_scope_manifest(
    *,
    manifest_path: Path,
    parsed: ParsedArgs,
    run_signature: dict[str, Any],
    requested_years: list[int],
    final_output_files: list[Path],
    ssp_scenario_options_by_year: dict[int, list[str | None]],
) -> str:
    """Write the disaggregated scope manifest and return its scope key."""
    scope_key = _run_scope_key(signature=run_signature)
    ssp_values = sorted(
        {
            str(value)
            for values in ssp_scenario_options_by_year.values()
            for value in values
            if value is not None and str(value).strip()
        }
    )
    scope_payload = _build_run_metadata(
        requested_years=list(requested_years),
        resolved_years=list(requested_years),
        selected_methods=dict(run_signature.get("selected_methods", {})),
        fu_code=str(parsed.base_allocate_args["fu_code"]),
        studied_indices_tag=str(run_signature.get("studied_indices_tag", "")),
        skipped_years={},
        outputs=[str(path) for path in final_output_files],
        signature=run_signature,
    )
    scope_payload["execution"]["completed_years"] = list(requested_years)
    scope_payload["execution"]["timestamp"] = datetime.now().isoformat()
    scope_payload["provenance"]["reference_years"] = parsed.base_allocate_args["reference_years"]
    scope_payload["provenance"]["ssp_scenarios"] = ssp_values
    scope_payload["provenance"]["projection"] = {
        "enabled": bool(run_signature.get("projection_mode")),
        "mode": run_signature.get("projection_mode"),
        "reg_window": run_signature.get("reg_window"),
        "l2_reuse_years": run_signature.get("l2_reuse_years"),
    }
    _save_run_metadata(
        manifest_path,
        scope_payload,
    )
    return scope_key


def write_branch_metadata(
    *,
    parsed: ParsedArgs,
    proj_base: Path,
    run_signature: dict[str, Any],
    requested_years: list[int],
    matched_runs: dict[str, MatchedRun],
    final_output_files: list[Path],
    audit_path: Path,
    metadata_path: Path,
    disaggregated_scope_key: str,
) -> None:
    """Write one disaggregation branch metadata payload."""
    payload = {
        "frozen_config": build_expected_disaggregation_frozen_config(parsed=parsed),
        "requested_years": list(requested_years),
        "exact_prior_allocate_cc_scopes": {
            name: {
                "proj_base": str(matched.proj_base),
                "output_source_label": matched.output_source_label,
                "scope_key": matched.scope_key,
                "scope_signature": matched.scope_signature,
                "completed_years": matched.completed_years,
            }
            for name, matched in sorted(matched_runs.items())
        },
        "final_output_files": [str(path) for path in final_output_files],
        "disaggregation_audit_path": str(audit_path),
        "disaggregated_allocate_scope_key": disaggregated_scope_key,
        "run_signature": run_signature,
    }
    write_json_dict(metadata_path, payload)
