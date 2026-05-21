"""Shared packaged asset copying for external inputs."""

from pathlib import Path
import shutil

from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent


def copy_packaged_files(*, source_dir: Path, target_dir: Path) -> Path:
    """Copy packaged asset files once and preserve staged project files."""
    resolved_source = source_dir.resolve(strict=True)
    out_dir = ensure_dir(target_dir)
    for source in sorted(resolved_source.rglob("*")):
        if not source.is_file():
            continue
        relative_path = source.relative_to(resolved_source)
        target = ensure_file_parent(out_dir / relative_path)
        if not target.exists():
            shutil.copy2(source, target)
    return out_dir


def copy_packaged_file(*, source_file: Path, target_file: Path) -> Path:
    """Copy one packaged asset file once and preserve a staged project file."""
    resolved_source = source_file.resolve(strict=True)
    target = ensure_file_parent(target_file)
    if not target.exists():
        shutil.copy2(resolved_source, target)
    return target
