"""Filename helpers for SSP partitioned table siblings."""

from pathlib import Path
import re

_SCENARIO_FILE_PREFIX = "__"
_SCENARIO_PARTITION_TOKEN_RE = re.compile(r"^ssp\d+$")


def scenario_partition_path(*, base_path: Path, token: str) -> Path:
    """Return one SSP partition sibling path for a logical table path."""
    normalized_token = _normalize_scenario_partition_token(token)
    return (
        base_path.parent
        / f"{base_path.stem}{_SCENARIO_FILE_PREFIX}{normalized_token}{base_path.suffix}"
    )


def scenario_partition_glob_pattern(*, base_path: Path) -> str:
    """Return the glob pattern for SSP partition siblings."""
    return f"{base_path.stem}{_SCENARIO_FILE_PREFIX}ssp*{base_path.suffix}"


def scenario_partition_token_from_path(*, base_path: Path, path: Path) -> str | None:
    """Return the SSP partition token encoded in a sibling path."""
    if path == base_path:
        return None
    prefix = f"{base_path.stem}{_SCENARIO_FILE_PREFIX}"
    if not path.stem.startswith(prefix):
        return None
    token = path.stem[len(prefix) :].strip().lower()
    if not _SCENARIO_PARTITION_TOKEN_RE.fullmatch(token):
        return None
    return token


def trailing_scenario_partition_token(*, path: Path) -> str | None:
    """Return a trailing SSP partition token encoded in a file stem."""
    token = path.stem.rsplit(_SCENARIO_FILE_PREFIX, 1)[-1].strip().lower()
    if not _SCENARIO_PARTITION_TOKEN_RE.fullmatch(token):
        return None
    return token


def _normalize_scenario_partition_token(token: str) -> str:
    compact = re.sub(r"[\s_]+", "", str(token).strip()).lower()
    suffix = compact[3:] if compact.startswith("ssp") else compact
    if suffix.isdigit() or suffix == "<n>":
        return f"ssp{suffix}"
    raise ValueError("Scenario partition filenames must use canonical SSP tokens like 'ssp2'.")
