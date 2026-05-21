"""Deterministic IO-LCA figure scope replacement ownership."""

from pathlib import Path

from pyaesa.shared.figures.persisted_outputs import delete_persisted_figure_paths
from pyaesa.shared.runtime.io.persisted_paths import normalize_persisted_paths


def _figure_paths_from_scope(*, scope_payload: dict[str, object]) -> list[Path]:
    """Return persisted figure paths recorded for one figure scope."""
    raw_paths = normalize_persisted_paths(raw_paths=scope_payload["paths_written"])
    return sorted({Path(path) for path in raw_paths})


def clear_existing_io_lca_figure_scope(
    *,
    payload: dict[str, object],
) -> None:
    """Clear persisted deterministic IO-LCA figure files for the current output scope."""
    paths_to_delete = (
        _figure_paths_from_scope(scope_payload=payload)
        if payload.get("arguments") is not None
        else []
    )
    delete_persisted_figure_paths(
        raw_paths=paths_to_delete,
    )
    payload.clear()
