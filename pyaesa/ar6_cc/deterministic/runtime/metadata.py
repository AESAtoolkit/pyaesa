"""Metadata ownership for deterministic AR6 CC runs."""

from pathlib import Path
from typing import Any, cast

from pyaesa.process.ar6.utils.io import metadata as ar6_metadata
from pyaesa.shared.acc_asr_common.deterministic.state.scope_guard import (
    coverage_signature_covers,
)
from pyaesa.shared.figures.persisted_outputs import delete_persisted_figure_paths
from pyaesa.shared.runtime.manifest_contract import manifest_digest, path_list
from pyaesa.shared.runtime.reuse.derived_state import request_state_matches, set_request_state

from pyaesa.ar6_cc.deterministic.request.contracts import cc_variable
from .reports import AR6CCPathwayCount, ComputeAR6CCReport

_FIGURE_STATE_KEY = "figure_state"


def load_run_metadata(path: Path) -> dict[str, Any] | None:
    """Load deterministic AR6 CC metadata when it exists."""
    return ar6_metadata.read_json(path)


def save_run_metadata(path: Path, payload: dict[str, Any]) -> Path:
    """Persist deterministic AR6 CC metadata."""
    return ar6_metadata.write_json(path, payload)


def figure_state_matches(
    *,
    payload: dict[str, Any],
    request_signature: dict[str, object],
    compute_signature: dict[str, Any],
) -> bool:
    """Return whether the stored deterministic figure request already matches."""
    return request_state_matches(
        payload=payload,
        state_key=_FIGURE_STATE_KEY,
        request_signature=request_signature,
        compute_signature=compute_signature,
        compute_compatible=coverage_signature_covers,
    )


def set_figure_state(
    *,
    payload: dict[str, Any],
    request_signature: dict[str, object],
    compute_signature: dict[str, Any],
    paths: list[Path],
) -> dict[str, Any]:
    """Store deterministic AR6 CC figure state in metadata."""
    set_request_state(
        payload=payload,
        state_key=_FIGURE_STATE_KEY,
        request_signature=request_signature,
        compute_signature=compute_signature,
        paths=paths,
    )
    payload["artifacts"]["figure_paths"] = path_list(paths)
    return payload


def clear_figure_state_paths(*, payload: dict[str, Any]) -> None:
    """Delete deterministic AR6 CC figure files recorded by metadata."""
    block = cast(dict[str, Any] | None, payload.get(_FIGURE_STATE_KEY))
    if block is not None:
        delete_persisted_figure_paths(
            raw_paths=block.get("paths"),
        )
    delete_persisted_figure_paths(
        raw_paths=payload["artifacts"].get("figure_paths"),
    )


def build_run_metadata_payload(
    *,
    signature: dict[str, Any],
    identity_payload: dict[str, Any],
    coverage: dict[str, list[str]],
    write_scope_identity: dict[str, Any],
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str,
    cc_categories: list[str],
    ssp_scenarios: list[str],
    total_model_scenario_pairs: int,
    pathway_counts: list[AR6CCPathwayCount],
    missing_pathway_combinations: list[AR6CCPathwayCount],
    output_file: Path,
    process_ar6: dict[str, object],
    post_study_output_file: Path | None = None,
) -> dict[str, Any]:
    """Build the canonical deterministic AR6 CC provenance payload."""
    variable = cc_variable(
        emission_type=emission_type,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode,
    )
    return {
        "function": "deterministic_ar6_cc",
        "arguments": dict(signature),
        "execution": {
            "status": "complete",
            "total_model_scenario_pairs": int(total_model_scenario_pairs),
            "pathway_counts": [_pathway_count_payload(item) for item in pathway_counts],
            "missing_pathway_combinations": [
                _pathway_count_payload(item) for item in missing_pathway_combinations
            ],
        },
        "reuse": {
            "identity_key": manifest_digest(identity_payload),
            "coverage": coverage,
            "write_scope_key": manifest_digest(write_scope_identity),
        },
        "artifacts": {
            "output_file": str(output_file),
            "post_study_output_file": (
                None if post_study_output_file is None else str(post_study_output_file)
            ),
            "figure_paths": [],
        },
        "provenance": {
            "emission_type": emission_type,
            "emissions_mode": emissions_mode,
            "variable": variable,
            "include_afolu": bool(include_afolu),
            "cc_categories": list(cc_categories),
            "ssp_scenarios": list(ssp_scenarios),
            "process_ar6": process_ar6,
        },
    }


def _pathway_count_payload(item: AR6CCPathwayCount) -> dict[str, object]:
    return {
        "category": item.category,
        "ssp_scenario": item.ssp_scenario,
        "model_scenario_pairs": int(item.model_scenario_pairs),
    }


def _pathway_counts_from_payload(payload: object) -> list[AR6CCPathwayCount]:
    return [
        AR6CCPathwayCount(
            category=str(item["category"]),
            ssp_scenario=str(item["ssp_scenario"]),
            model_scenario_pairs=int(cast(int, item["model_scenario_pairs"])),
        )
        for item in cast(list[dict[str, object]], payload)
    ]


def build_cached_report(
    *,
    payload: dict[str, Any],
    study_period: list[int],
    harmonization: bool,
    harmonization_method: str,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str,
    subset_version: str | None,
    meta_file: Path,
    cc_dir: Path,
    logs_dir: Path,
    figure_paths: list[Path],
    reuse_status: str = "reused_exact",
) -> ComputeAR6CCReport:
    """Build one cached deterministic AR6 CC report from persisted metadata."""
    return ComputeAR6CCReport(
        study_period=study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        emission_type=emission_type,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode,
        variable=cast(str, payload["provenance"]["variable"]),
        categories=list(cast(list[str], payload["provenance"]["cc_categories"])),
        ssp_scenarios=list(cast(list[str], payload["provenance"]["ssp_scenarios"])),
        subset_version=subset_version,
        total_model_scenario_pairs=cast(int, payload["execution"]["total_model_scenario_pairs"]),
        pathway_counts=_pathway_counts_from_payload(payload["execution"]["pathway_counts"]),
        missing_pathway_combinations=_pathway_counts_from_payload(
            payload["execution"]["missing_pathway_combinations"]
        ),
        output_file=Path(cast(str, payload["artifacts"]["output_file"])),
        post_study_output_file=(
            None
            if payload["artifacts"].get("post_study_output_file") is None
            else Path(cast(str, payload["artifacts"]["post_study_output_file"]))
        ),
        figure_paths=figure_paths,
        meta_file=meta_file,
        cc_dir=cc_dir,
        logs_dir=logs_dir,
        reuse_status=reuse_status,
        process_ar6=cast(dict[str, object], payload["provenance"]["process_ar6"]),
    )
