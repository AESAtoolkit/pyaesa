"""Canonical deterministic aSoCC path and project base resolution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pyaesa.asocc.runtime.paths.deterministic import (
    _get_allocate_run_metadata_path,
    _get_allocate_logs_dir,
)
from pyaesa.asocc.runtime.paths.family_roots import effective_group_version_for_source
from pyaesa.asocc.runtime.paths.published import (
    _get_enacting_metric_dir,
    _get_asocc_l1_dir,
    _get_asocc_l2_dir,
    _owning_fu_level_for_code,
)
from pyaesa.workspace_initialisation.workspace import (
    project_outputs_root,
)
from pyaesa.shared.runtime.io.family_root_names import ASOCC_ROOT_DIRNAME


@dataclass(frozen=True)
class AsoccDeterministicPathScope:
    """Deterministic aSoCC output path scope reused across consumers."""

    proj_base: Path
    source_label: str
    group_version: str | None


def outputs_project_root(*, project_name: str) -> Path:
    """Return the canonical outputs root for one allocation project."""
    return project_outputs_root(project_name=project_name)


def resolve_allocate_project_base(*, base_allocate_args: Mapping[str, Any]) -> Path:
    """Resolve the deterministic branch local project base for one allocate request."""
    return outputs_project_root(project_name=str(base_allocate_args["project_name"]))


def build_asocc_deterministic_path_scope(
    *,
    proj_base: Path,
    source_label: str,
    group_version: str | None,
) -> AsoccDeterministicPathScope:
    """Build one normalized deterministic aSoCC output path scope object."""
    source_clean = str(source_label).strip()
    if not source_clean:
        raise ValueError("source_label must be a non-empty string.")
    published_group_version = effective_group_version_for_source(
        source=source_clean,
        group_version=group_version,
    )
    return AsoccDeterministicPathScope(
        proj_base=Path(proj_base),
        source_label=source_clean,
        group_version=published_group_version,
    )


def resolve_allocate_path_scope(
    *,
    base_allocate_args: Mapping[str, Any],
) -> AsoccDeterministicPathScope:
    """Resolve one normalized deterministic aSoCC path scope from allocate args."""
    return build_asocc_deterministic_path_scope(
        proj_base=resolve_allocate_project_base(base_allocate_args=base_allocate_args),
        source_label=str(base_allocate_args["source"]),
        group_version=effective_group_version_for_source(
            source=str(base_allocate_args["source"]),
            group_version=base_allocate_args["group_version"],
        ),
    )


def allocate_run_metadata_path(*, scope: AsoccDeterministicPathScope) -> Path:
    """Return deterministic run metadata path for one normalized path scope."""
    return _get_allocate_run_metadata_path(
        scope.proj_base,
        source=scope.source_label,
        group_version=scope.group_version,
    )


def asocc_l1_dir(
    *,
    scope: AsoccDeterministicPathScope,
    lcia_sub: str | None,
    fu_code: str | None = None,
) -> Path:
    """Return level-1 deterministic aSoCC directory for one normalized path scope."""
    return _get_asocc_l1_dir(
        proj_base=scope.proj_base,
        source=scope.source_label,
        group_version=scope.group_version,
        lcia_sub=lcia_sub,
        owning_fu_level=_owning_fu_level_for_code(fu_code=fu_code),
    )


def asocc_l2_dir(
    *,
    scope: AsoccDeterministicPathScope,
    bucket: str,
    lcia_sub: str | None,
) -> Path:
    """Return level-2 deterministic aSoCC directory for one normalized path scope."""
    return _get_asocc_l2_dir(
        proj_base=scope.proj_base,
        source=scope.source_label,
        group_version=scope.group_version,
        bucket=bucket,
        lcia_sub=lcia_sub,
    )


def asocc_enacting_metric_dir(
    *,
    scope: AsoccDeterministicPathScope,
    level: str,
    fu_code: str | None = None,
) -> Path:
    """Return enacting metric directory for one normalized path scope."""
    return _get_enacting_metric_dir(
        proj_base=scope.proj_base,
        source=scope.source_label,
        group_version=scope.group_version,
        level=level,
        lcia_sub=None,
        owning_fu_level=_owning_fu_level_for_code(fu_code=fu_code),
    )


def asocc_logs_root(*, scope: AsoccDeterministicPathScope) -> Path:
    """Return logs root for one normalized path scope."""
    return _get_allocate_logs_dir(
        scope.proj_base,
        source=scope.source_label,
        group_version=scope.group_version,
    )


def collect_asocc_roots(
    *,
    scope: AsoccDeterministicPathScope,
    fu_code: str | None = None,
) -> list[str]:
    """Return deterministic aSoCC data roots for one normalized path scope."""
    roots = [
        asocc_l1_dir(scope=scope, lcia_sub=None, fu_code=fu_code),
        asocc_l2_dir(scope=scope, bucket="l2_in_l1", lcia_sub=None),
        asocc_l2_dir(scope=scope, bucket="l2_vs_global", lcia_sub=None),
        asocc_l2_dir(scope=scope, bucket="utility_propagation_contrib", lcia_sub=None),
    ]
    return [str(path) for path in roots]


def project_base_from_allocation_descendant(path: Path) -> Path:
    """Return the project branch root from one descendant under the aSoCC root."""
    resolved = Path(path)
    parts = list(resolved.parts)
    if ASOCC_ROOT_DIRNAME not in parts:
        raise ValueError(
            f"Cannot reconstruct project base from a path outside the aSoCC branch: {resolved}"
        )
    return Path(*parts[: parts.index(ASOCC_ROOT_DIRNAME)])


def path_scope_from_signature(
    *,
    proj_base: Path,
    source_label: str,
    run_signature: Mapping[str, Any],
    context_label: str,
) -> AsoccDeterministicPathScope:
    """Build one deterministic path scope from a persisted signature payload."""
    required_keys = {"group_version"}
    missing = sorted(key for key in required_keys if key not in run_signature)
    if missing:
        raise ValueError(
            f"{context_label} is missing deterministic keys required for path "
            f"resolution: {missing}."
        )
    return build_asocc_deterministic_path_scope(
        proj_base=proj_base,
        source_label=source_label,
        group_version=run_signature.get("group_version"),
    )


def resolve_disaggregation_path_scope(
    *,
    base_allocate_args: Mapping[str, Any],
    source_label: str,
) -> tuple[AsoccDeterministicPathScope, Path]:
    """Resolve a disaggregated aSoCC source from its deterministic scope manifest."""
    scope = build_asocc_deterministic_path_scope(
        proj_base=outputs_project_root(project_name=str(base_allocate_args["project_name"])),
        source_label=source_label,
        group_version=None,
    )
    manifest_path = allocate_run_metadata_path(scope=scope)
    if not manifest_path.exists():
        raise ValueError(
            "No deterministic aSoCC scope_manifest.json was found for disaggregated source "
            f"'{scope.source_label}'. Expected: {manifest_path}"
        )
    return scope, manifest_path
