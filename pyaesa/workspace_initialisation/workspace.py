"""Workspace state and repository path ownership for active workspaces."""

from pathlib import Path

_REPOSITORY_DIRNAME = "pyaesa"
_ACTIVE_REPO_ROOT: dict[str, Path | None] = {"repo_root": None}


def clear_default_repo_root() -> None:
    """Clear the active repository root for the current session."""
    _ACTIVE_REPO_ROOT["repo_root"] = None


def resolve_repo_root(top_path: str | Path) -> Path:
    """Return the canonical repository root resolved from one user top path.

    Args:
        top_path: Parent directory where the package workspace root is created.

    Returns:
        Absolute repository root path.

    Raises:
        ValueError: If ``top_path`` is a blank string.
    """
    if isinstance(top_path, str) and not top_path.strip():
        raise ValueError("top_path must be a non-empty path string.")
    return Path(top_path).expanduser().resolve() / _REPOSITORY_DIRNAME


def set_default_repo_root(repo_root: str | Path) -> None:
    """Store the active repository root after successful workspace setup."""
    _ACTIVE_REPO_ROOT["repo_root"] = Path(repo_root).expanduser().resolve()


def get_default_repo_root() -> Path:
    """Return the configured repository root for the active session."""
    repo_root = _ACTIVE_REPO_ROOT.get("repo_root")
    if repo_root is None:
        raise RuntimeError(
            "Workspace repository root not configured. "
            "Call `set_workspace()` during workspace initialisation."
        )
    return repo_root


def project_outputs_root(*, project_name: str) -> Path:
    """Return the canonical project output root for one project scope.

    Args:
        project_name: Project scoped analytical output name.

    Returns:
        Absolute path to the project named folder beneath the configured repository.

    Raises:
        ValueError: If ``project_name`` is blank.
    """
    project_name_clean = str(project_name).strip()
    if not project_name_clean:
        raise ValueError("project_name must be a non-empty string.")
    return get_default_repo_root() / project_name_clean
