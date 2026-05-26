from pathlib import Path
from typing import Any

from pyaesa.shared.acc_asr_common.branches.config import cc_branch_token
from pyaesa.shared.uncertainty_assessment.request.core import (
    UncertaintyRuntimeRequest,
    normalize_uncertainty_request,
)
from pyaesa.shared.uncertainty_assessment.run_state.branch_sets import (
    branch_set_run_id,
    branch_set_run_id_for_request,
    run_branch_set_report,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    build_manifest,
    write_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest_payloads import (
    mc_parameters_payload,
)
from pyaesa.shared.uncertainty_assessment.run_state.report import UncertaintyRunReport


def test_branch_set_reuse_reads_branch_manifests_without_parent_index(
    tmp_path: Path,
) -> None:
    runtime = normalize_uncertainty_request(
        family="asr",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": True, "n_runs": 5},
            "convergence": {"active": False},
        },
    )
    arguments = {
        "project_name": "demo",
        "years": [2020],
        "lcia_method": ["gwp100_lcia", "pb_lcia"],
        "fu_code": "L2.a.a",
        "base_cc_args": {"static": {"bounds": ["min_cc"]}},
    }
    branches = [
        {"cc_source": "gwp100_lcia", "cc_type": "static", "static_cc_bounds": ["min_cc"]},
        {"cc_source": "pb_lcia", "cc_type": "static", "static_cc_bounds": ["min_cc"]},
    ]
    _write_branch_manifest(
        root=tmp_path,
        branch=branches[0],
        run_id="mc_incomplete",
        runtime=runtime,
        arguments=arguments,
    )
    for branch in branches:
        _write_branch_manifest(
            root=tmp_path,
            branch=branch,
            run_id="mc_mismatch",
            runtime=runtime,
            arguments={**arguments, "fu_code": "L2.c.b"},
        )
    report = run_branch_set_report(
        family="asr",
        root=tmp_path,
        runtime=runtime,
        arguments=arguments,
        branches=branches,
        requested_run_id="mc_reuse",
        refresh=True,
        run_branch=lambda branch, run_id: _write_branch_manifest(
            root=tmp_path,
            branch=branch,
            run_id=run_id,
            runtime=runtime,
            arguments=arguments,
        ),
    )

    assert "scope_manifest" not in report.manifest.artifacts
    assert "Run status:" in str(report)
    assert not (tmp_path / "mc_reuse").exists()
    assert (
        branch_set_run_id_for_request(
            root=tmp_path,
            runtime=runtime,
            arguments=arguments,
            branches=branches,
        )
        == "mc_reuse"
    )
    allocated = branch_set_run_id(
        root=tmp_path / "empty",
        runtime=runtime,
        arguments=arguments,
        branches=branches,
        requested_run_id=None,
        refresh=False,
    )
    assert allocated.startswith("mc_")


def _write_branch_manifest(
    *,
    root: Path,
    branch: dict[str, Any],
    run_id: str,
    runtime: UncertaintyRuntimeRequest,
    arguments: dict[str, Any],
) -> UncertaintyRunReport:
    """Write one branch manifest owned by a concrete branch folder."""
    manifest_path = _manifest_path(root=root, branch=branch, run_id=run_id)
    manifest = build_manifest(
        family="asr",
        mode=runtime.mode,
        output_format=runtime.output_format,
        active_sources=("asocc",),
        completed_runs=runtime.n_runs,
        status="complete",
        run_id=run_id,
        requested_runs=runtime.n_runs,
        mc_parameters=mc_parameters_payload(runtime=runtime),
        arguments={
            **arguments,
            "lcia_method": [str(branch["cc_source"])],
            "base_cc_args": {"static": {"exclude_max_cc": True, "bounds": ["min_cc"]}},
        },
        artifacts={"scope_manifest": str(manifest_path)},
    )
    write_manifest(path=manifest_path, manifest=manifest)
    return UncertaintyRunReport(manifest=manifest, reuse_status="computed")


def _manifest_path(*, root: Path, branch: dict[str, Any], run_id: str) -> Path:
    """Return the branch manifest path for one run id."""
    return (
        root
        / cc_branch_token(cc_source=str(branch["cc_source"]), cc_type=str(branch["cc_type"]))
        / run_id
        / "logs"
        / "scope_manifest.json"
    )
