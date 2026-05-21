"""Shared filesystem helpers for directory creation and atomic file writes."""

import os
from pathlib import Path
import tempfile
from typing import Callable


def ensure_dir(path: Path) -> Path:
    """Create one directory path and return it."""
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_file_parent(path: Path) -> Path:
    """Create the parent directory for one file path and return the file path."""
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_via_atomic_temp(
    path: Path,
    *,
    writer: Callable[[Path], None],
) -> Path:
    """Write one file through a same-directory temp file and atomic replace."""
    resolved = ensure_file_parent(path)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(resolved.parent),
        prefix=f".{resolved.stem}.tmp_",
        suffix=resolved.suffix,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        writer(tmp_path)
        tmp_path.replace(resolved)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return resolved


def atomic_write_text(
    path: Path,
    *,
    text: str,
    encoding: str = "utf-8",
) -> Path:
    """Write text to one file through an atomic temp-replace cycle."""

    def _write_text(tmp_path: Path) -> None:
        tmp_path.write_text(text, encoding=encoding)

    return write_via_atomic_temp(path, writer=_write_text)
