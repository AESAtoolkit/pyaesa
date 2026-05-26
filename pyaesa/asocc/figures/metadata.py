"""Deterministic aSoCC figure metadata owners."""

from pathlib import Path
from typing import Any

from pyaesa.shared.figures.persisted_outputs import delete_persisted_figure_paths

from pyaesa.asocc.runtime.paths.deterministic import (
    _get_allocate_run_metadata_path,
)
from pyaesa.asocc.io.metadata import _load_run_metadata, _save_run_metadata
from .scope_planner import RunScope


def delete_persisted_figure_state_paths(
    *,
    payload: dict[str, Any],
    state_key: str,
) -> None:
    """Delete figure files recorded for one deterministic aSoCC figure state."""
    block = payload.get(state_key)
    if block is None:
        return
    delete_persisted_figure_paths(raw_paths=block.get("paths"))


def write_run_figure_paths(
    *,
    scope: RunScope,
    figure_paths: list[Path],
) -> None:
    """Persist deterministic figure paths without deleting other figure scopes."""
    metadata_path = _get_allocate_run_metadata_path(
        scope.proj_base,
        source=scope.source,
        agg_version=scope.agg_version,
    )
    payload = _load_run_metadata(metadata_path)
    artifacts = dict(payload["artifacts"])
    artifacts["figure_paths"] = [str(path) for path in sorted(set(figure_paths))]
    payload["artifacts"] = artifacts
    _save_run_metadata(metadata_path, payload)
