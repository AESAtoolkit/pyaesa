"""Shared cleanup helpers for persisted figure outputs."""

from pathlib import Path

from pyaesa.shared.runtime.io.persisted_paths import normalize_persisted_paths


def delete_persisted_figure_paths(*, raw_paths: object) -> None:
    """Delete figure files recorded by pyaesa owned metadata."""
    for path in normalize_persisted_paths(raw_paths=raw_paths):
        Path(path).unlink(missing_ok=True)
