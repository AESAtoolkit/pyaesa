"""Public output root resolution for package generated reporting paths."""

from pathlib import Path

from pyaesa.workspace_initialisation.workspace import get_default_repo_root


def public_output_root_from_path(path: Path) -> Path:
    """Return the user facing output folder that owns one package generated path."""
    ancestors = (Path(path), *Path(path).parents)
    for candidate in ancestors:
        if candidate.parent.name == "monte_carlo":
            return candidate
    for candidate in ancestors:
        if candidate.name == "deterministic":
            return candidate
    for candidate in ancestors:
        if candidate.name in {"logs", "results", "figures"}:
            return candidate.parent
    repo_root = get_default_repo_root()
    return next(candidate for candidate in ancestors if candidate.parent == repo_root)
