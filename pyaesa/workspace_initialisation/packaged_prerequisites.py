"""Prerequisite import ownership for workspace setup."""

import importlib.resources as resources
from pathlib import Path
import shutil

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent


def packaged_prerequisites_root():
    """Return the packaged prerequisite root shipped inside the installed package."""
    return resources.files("pyaesa.workspace_initialisation").joinpath("prerequisites")


def _copy_prerequisite_file(*, src: Path, dest: Path, refresh: bool) -> bool:
    """Copy one prerequisite source file when missing or refreshed."""
    dest = ensure_file_parent(dest)
    if dest.exists() and dest.is_dir():
        raise IsADirectoryError(f"Cannot copy prerequisite file to directory path '{dest}'.")
    if refresh or not dest.exists():
        shutil.copyfile(src, dest)
        return True
    return False


def import_prerequisites(*, repo_root: str | Path, refresh: bool = False) -> bool:
    """Copy prerequisite assets into one workspace repository.

    Packaged prerequisite assets, including methodological notes and the
    functional unit selection guide, are copied from the installed package
    resource tree.

    Args:
        repo_root: Workspace repository root that owns the copied ``data_raw/`` tree.
        refresh: When ``True``, overwrite existing prerequisite files.

    Returns:
        ``True`` when at least one prerequisite file was copied or
        refreshed, otherwise ``False``.

    Raises:
        IsADirectoryError: If a prerequisite file path is occupied by a directory.
        OSError: If a source prerequisite cannot be copied.
    """
    prerequisite_root = Path(repo_root).expanduser().resolve() / "data_raw"
    package_prereq_root = packaged_prerequisites_root()
    changed = False
    with resources.as_file(package_prereq_root) as prereq_folders:
        prereq_path = Path(prereq_folders)
        for src in prereq_path.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(prereq_path)
            if _copy_prerequisite_file(src=src, dest=prerequisite_root / rel, refresh=refresh):
                changed = True
    return changed
