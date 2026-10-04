"""Synchronize the software citation version with the package metadata."""

import json
from pathlib import Path
import re
import tomllib


def sync_citation_version(repo_root: Path) -> bool:
    """Copy the package version into the root software citation field.

    Args:
        repo_root: Repository directory containing pyproject.toml and CITATION.cff.

    Returns:
        True when CITATION.cff was updated, otherwise False.

    Raises:
        ValueError: If the package version is not a nonempty string or the citation
            file does not contain exactly one top-level version field.
    """
    with (repo_root / "pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    if not isinstance(version, str) or not version.strip():
        raise ValueError("pyproject.toml must define project.version as a nonempty string.")

    citation_path = repo_root / "CITATION.cff"
    original = citation_path.read_bytes().decode("utf-8")
    updated, count = re.subn(
        r"(?m)^version:[^\r\n]*",
        lambda match: f"version: {json.dumps(version)}",
        original,
    )
    if count != 1:
        raise ValueError("CITATION.cff must contain exactly one top-level version field.")
    if updated == original:
        return False
    citation_path.write_bytes(updated.encode("utf-8"))
    return True


if __name__ == "__main__":
    changed = sync_citation_version(Path(__file__).resolve().parents[1])
    print(
        "Updated CITATION.cff from pyproject.toml."
        if changed
        else "CITATION.cff already matches pyproject.toml."
    )
