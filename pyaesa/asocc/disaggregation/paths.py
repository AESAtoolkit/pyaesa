"""Canonical path ownership for disaggregate_asocc branch artifacts."""

from pathlib import Path

from ..io.contracts import mode_label
from pyaesa.asocc.runtime.scope.branch_resolution import (
    asocc_logs_root,
    build_asocc_deterministic_path_scope,
)
from pyaesa.asocc.runtime.paths.published import _asocc_deterministic_scope_root


def mode_suffix(*, group_indices: bool) -> str:
    """Return stable grouped or ungrouped suffix for branch artifact names."""
    return mode_label(group_indices=group_indices)


def disaggregation_source_root(*, proj_base: Path, source_label: str) -> Path:
    """Return the deterministic output root owned by one disaggregated aSoCC source."""
    scope = build_asocc_deterministic_path_scope(
        proj_base=proj_base,
        source_label=source_label,
        agg_version=None,
    )
    return _asocc_deterministic_scope_root(
        proj_base=scope.proj_base,
        source=scope.source_label,
        agg_version=scope.agg_version,
    )


def disaggregation_logs_dir(
    *,
    proj_base: Path,
    source_label: str,
) -> Path:
    """Return the log directory owned by one disaggregated aSoCC source."""
    scope = build_asocc_deterministic_path_scope(
        proj_base=proj_base,
        source_label=source_label,
        agg_version=None,
    )
    return asocc_logs_root(scope=scope) / "disaggregate_asocc_log"


def disaggregation_metadata_path(
    *,
    proj_base: Path,
    source_label: str,
    mode: str,
    group_indices: bool,
) -> Path:
    """Return disaggregate_asocc audit metadata path."""
    return disaggregation_logs_dir(
        proj_base=proj_base,
        source_label=source_label,
    ) / disaggregation_metadata_filename(
        mode=mode,
        group_indices=group_indices,
    )


def disaggregation_audit_path(
    *,
    logs_dir: Path,
    mode: str,
    group_indices: bool,
) -> Path:
    """Return disaggregation audit CSV path under one branch log directory."""
    return Path(logs_dir) / disaggregation_audit_filename(
        mode=mode,
        group_indices=group_indices,
    )


def disaggregation_metadata_filename(*, mode: str, group_indices: bool) -> str:
    """Return disaggregation branch metadata filename."""
    return f"metadata_{mode}_{mode_suffix(group_indices=group_indices)}.json"


def disaggregation_audit_filename(*, mode: str, group_indices: bool) -> str:
    """Return disaggregation audit filename."""
    return f"disaggregation_audit_{mode}_{mode_suffix(group_indices=group_indices)}.csv"
