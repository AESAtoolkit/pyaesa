"""Shared manifest payload helpers for package run scopes."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def manifest_json_value(value: Any) -> Any:
    """Return one deterministic JSON-compatible manifest value."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, range):
        return list(value)
    if isinstance(value, Mapping):
        return {str(key): manifest_json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [manifest_json_value(item) for item in value]
    if isinstance(value, set):
        return [manifest_json_value(item) for item in sorted(value)]
    return value


def manifest_digest(payload: Mapping[str, Any]) -> str:
    """Return one stable digest for a manifest identity payload."""
    canonical = json.dumps(
        manifest_json_value(dict(payload)),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def path_list(paths: list[Path] | tuple[Path, ...]) -> list[str]:
    """Return sorted unique path strings for manifest artifacts."""
    return [str(path) for path in sorted({Path(path) for path in paths})]
