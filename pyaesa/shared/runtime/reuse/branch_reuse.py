"""Shared cached-branch lifecycle helpers for deterministic family coordinators."""

import shutil
from pathlib import Path
from typing import Any


def cleanup_branch_outputs_for_refresh(
    *,
    existing_metadata: dict[str, Any],
    meta_path: Path,
    artifact_keys: tuple[str, ...],
    scope_targets: tuple[Path, ...] = (),
) -> None:
    """Delete persisted branch artifacts recorded in metadata for one refresh request."""
    cleanup_refresh_scope_targets(targets=scope_targets)
    artifacts = existing_metadata.get("artifacts", {})
    for key in artifact_keys:
        for path_str in artifacts.get(key, []):
            path = Path(path_str)
            if path.exists():
                path.unlink()
    if meta_path.exists():
        meta_path.unlink()


def cleanup_refresh_scope_targets(
    *,
    targets: tuple[Path, ...],
) -> None:
    """Delete scoped refresh targets even when no metadata was persisted."""
    for target in targets:
        if not target.exists():
            continue
        if target.is_dir():
            shutil.rmtree(target)
            continue
        target.unlink()
