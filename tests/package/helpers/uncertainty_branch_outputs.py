from pathlib import Path

from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)

MIXED_STATIC_DYNAMIC_CC_TOKENS = frozenset(
    {"static__pb_lcia", "static__gwp100_lcia", "dynamic_ar6__gwp100_lcia"}
)


def assert_mixed_static_dynamic_cc_outputs(
    *,
    manifest: UncertaintyManifest,
    output_format: str,
    expected_run_id: str | None = None,
) -> dict[str, UncertaintyManifest]:
    """Assert the shared static plus dynamic CC uncertainty branch contract."""
    if expected_run_id is not None:
        assert manifest.run_id == expected_run_id
    branch_paths = [Path(path) for path in manifest.artifacts["branch_scope_manifests"]]
    assert len(branch_paths) == len(MIXED_STATIC_DYNAMIC_CC_TOKENS)
    assert {path.parents[2].name for path in branch_paths} == MIXED_STATIC_DYNAMIC_CC_TOKENS
    monte_carlo_root = branch_paths[0].parents[3]
    assert not (monte_carlo_root / manifest.run_id).exists()
    branch_manifests = [read_manifest(path=path) for path in branch_paths]
    assert {item.run_id for item in branch_manifests} == {manifest.run_id}
    by_branch: dict[str, UncertaintyManifest] = {}
    for item in branch_manifests:
        identity = read_uncertainty_table(
            path=Path(item.artifacts["public_row_identity"]),
            output_format=output_format,
        )
        branch_token = Path(item.artifacts["scope_manifest"]).parents[2].name
        by_branch[branch_token] = item
        assert set(identity["cc_type"]) == {branch_token.rsplit("__", maxsplit=1)[0]}
    return by_branch
