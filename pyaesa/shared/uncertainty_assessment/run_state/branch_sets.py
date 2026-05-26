"""Manifest ownership for uncertainty calls split by carrying capacity route."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from pyaesa.shared.acc_asr_common.branches.config import cc_branch_token, normalize_base_cc_args
from pyaesa.shared.acc_asr_common.persistence.requests import build_public_cc_branch_args
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    allocate_run_id,
    build_manifest,
    read_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest_payloads import (
    mc_parameters_payload,
)
from pyaesa.shared.runtime.manifest_contract import manifest_json_value
from pyaesa.shared.uncertainty_assessment.run_state.report import (
    UncertaintyRunReport,
)


def build_branch_set_manifest(
    *,
    family: str,
    run_id: str,
    runtime: UncertaintyRuntimeRequest,
    arguments: dict[str, Any],
    branch_reports: list[UncertaintyRunReport],
) -> UncertaintyManifest:
    """Return the in memory manifest that summarizes completed CC route manifests."""
    branch_manifests = [report.manifest for report in branch_reports]
    branch_scope_manifests = [
        str(manifest.artifacts["scope_manifest"]) for manifest in branch_manifests
    ]
    artifacts = {
        "branch_scope_manifests": branch_scope_manifests,
        "branch_run_roots": [
            str(Path(str(manifest.artifacts["scope_manifest"])).parents[1])
            for manifest in branch_manifests
        ],
        "public_output": {
            "branches": [_branch_public_output(manifest=manifest) for manifest in branch_manifests]
        },
    }
    return build_manifest(
        family=family,
        mode=runtime.mode,
        output_format=runtime.output_format,
        active_sources=tuple(
            dict.fromkeys(
                source for manifest in branch_manifests for source in manifest.active_sources
            )
        ),
        completed_runs=min((manifest.completed_runs for manifest in branch_manifests), default=0),
        status="complete",
        run_id=run_id,
        requested_runs=runtime.n_runs,
        mc_parameters=mc_parameters_payload(runtime=runtime),
        source_parameters={"branch_count": len(branch_manifests)},
        arguments=arguments,
        artifacts=artifacts,
        compatibility_context={
            "artifact_contract": f"{family}_branch_set",
            "branch_scope_manifests": branch_scope_manifests,
        },
    )


def run_branch_set_report(
    *,
    family: str,
    root: Path,
    runtime: UncertaintyRuntimeRequest,
    arguments: dict[str, Any],
    branches: list[dict[str, Any]],
    requested_run_id: str | None,
    refresh: bool,
    run_branch: Callable[[dict[str, Any], str], UncertaintyRunReport],
) -> UncertaintyRunReport:
    """Run each carrying capacity route with one shared run id."""
    run_id = branch_set_run_id(
        root=root,
        runtime=runtime,
        arguments=arguments,
        branches=branches,
        requested_run_id=requested_run_id,
        refresh=refresh,
    )
    branch_reports = [run_branch(branch, run_id) for branch in branches]
    manifest = build_branch_set_manifest(
        family=family,
        runtime=runtime,
        run_id=run_id,
        arguments=arguments,
        branch_reports=branch_reports,
    )
    reuse_statuses = {report.reuse_status for report in branch_reports}
    reuse_status = "reused_exact" if reuse_statuses == {"reused_exact"} else "computed"
    return UncertaintyRunReport(manifest=manifest, reuse_status=reuse_status)


def branch_set_run_id_for_request(
    *,
    root: Path,
    runtime: UncertaintyRuntimeRequest,
    arguments: dict[str, Any],
    branches: list[dict[str, Any]],
) -> str | None:
    """Return the run id for a complete multi route branch set matching the request."""
    branch_roots = [_branch_root(root=root, branch=branch) for branch in branches]
    expected_arguments = [
        _branch_arguments(arguments=arguments, branch=branch) for branch in branches
    ]
    expected_mc_parameters = mc_parameters_payload(runtime=runtime)
    for manifest_path in sorted(branch_roots[0].glob("mc_*/logs/scope_manifest.json")):
        run_id = manifest_path.parents[1].name
        manifest_paths = [
            branch_root / str(run_id) / "logs" / "scope_manifest.json"
            for branch_root in branch_roots
        ]
        if not all(path.exists() for path in manifest_paths):
            continue
        if all(
            (
                manifest.status,
                manifest.mode,
                manifest.output_format,
                manifest.requested_runs,
                manifest.mc_parameters,
                manifest.arguments,
            )
            == (
                "complete",
                runtime.mode,
                runtime.output_format,
                runtime.n_runs,
                expected_mc_parameters,
                branch_arguments,
            )
            for manifest, branch_arguments in zip(
                (read_manifest(path=path) for path in manifest_paths),
                expected_arguments,
                strict=True,
            )
        ):
            return run_id
    return None


def branch_set_run_id(
    *,
    root: Path,
    runtime: UncertaintyRuntimeRequest,
    arguments: dict[str, Any],
    branches: list[dict[str, Any]],
    requested_run_id: str | None,
    refresh: bool,
) -> str:
    """Return the run id shared by all carrying capacity routes."""
    if requested_run_id is not None:
        return requested_run_id
    if not refresh:
        reusable = branch_set_run_id_for_request(
            root=root,
            runtime=runtime,
            arguments=arguments,
            branches=branches,
        )
        if reusable is not None:
            return reusable
    return allocate_run_id()


def _branch_root(*, root: Path, branch: dict[str, Any]) -> Path:
    """Return the Monte Carlo root for one carrying capacity branch."""
    return Path(root) / cc_branch_token(
        cc_source=str(branch["cc_source"]),
        cc_type=str(branch["cc_type"]),
    )


def _branch_arguments(*, arguments: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
    """Return the persisted request arguments expected for one branch."""
    branch_arguments = dict(arguments)
    branch_arguments["lcia_method"] = [str(branch["cc_source"])]
    branch_arguments["base_cc_args"] = normalize_base_cc_args(
        build_public_cc_branch_args(branch=branch)
    )
    return manifest_json_value(branch_arguments)


def _branch_public_output(*, manifest: UncertaintyManifest) -> dict[str, Any]:
    arguments = cast(dict[str, Any], manifest.arguments)
    payload: dict[str, Any] = {
        "scope_manifest": str(manifest.artifacts["scope_manifest"]),
        "base_cc_args": arguments["base_cc_args"],
    }
    if manifest.compatibility_context is not None:
        has_cumulative = manifest.compatibility_context.get("has_cumulative_outputs")
        if has_cumulative is not None:
            payload["has_cumulative_outputs"] = bool(has_cumulative)
    return payload
