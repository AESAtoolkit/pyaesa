"""Shared JSON metadata I/O helpers."""

import json
from pathlib import Path
from typing import Any, cast

from pyaesa.shared.runtime.io.filesystem import atomic_write_text


def read_json_dict(path: Path) -> dict[str, Any]:
    """Load one required JSON file as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Required JSON metadata file is missing at {path}.")
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def read_optional_json_dict(path: Path) -> dict[str, Any]:
    """Load one optional JSON file as a dictionary or return an empty mapping."""
    if not path.exists():
        return {}
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json_dict(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON dictionary payload."""
    atomic_write_text(path, text=json.dumps(payload, indent=2))
