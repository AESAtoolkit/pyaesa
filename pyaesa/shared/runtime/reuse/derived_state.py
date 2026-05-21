"""Persist and compare derived artifact request state in metadata payloads.

This module supports exact no op versus rerender decisions for figures and
other derived outputs by storing the request signature and persisted artifact
paths associated with one deterministic or MC compute state.
"""

from pathlib import Path
from typing import Any, Callable, cast

from pyaesa.shared.runtime.io.persisted_paths import normalize_persisted_paths


def _state_block(payload: dict[str, Any], *, state_key: str) -> dict[str, Any] | None:
    """Return one stored package-owned state block when present."""
    raw = payload.get(state_key)
    if raw is None:
        return None
    return cast(dict[str, Any], raw)


def _stored_paths(block: dict[str, Any]) -> list[Path]:
    """Return the strict persisted path list for one derived state block."""
    return normalize_persisted_paths(raw_paths=block.get("paths"))


def _paths_exist(paths: list[Path]) -> bool:
    """Return whether all persisted artifact paths still exist."""
    if not paths:
        return False
    return all(path.exists() for path in paths)


def request_state_matches(
    *,
    payload: dict[str, Any],
    state_key: str,
    request_signature: dict[str, Any],
    compute_signature: dict[str, Any] | None = None,
    request_compatible: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
    compute_compatible: Callable[[dict[str, Any], dict[str, Any]], bool] | None = None,
) -> bool:
    """Return whether one persisted derived artifact request is still compatible."""
    block = _state_block(payload, state_key=state_key)
    if block is None:
        return False
    stored_compute = block.get("compute_signature")
    if compute_signature is not None:
        if compute_compatible is None:
            if stored_compute != compute_signature:
                return False
        elif not isinstance(stored_compute, dict) or not compute_compatible(
            stored_compute, compute_signature
        ):
            return False
    stored_request = block.get("request_signature")
    if request_compatible is None:
        if stored_request != request_signature:
            return False
    elif not isinstance(stored_request, dict) or not request_compatible(
        stored_request, request_signature
    ):
        return False
    return _paths_exist(_stored_paths(block))


def set_request_state(
    *,
    payload: dict[str, Any],
    state_key: str,
    request_signature: dict[str, Any],
    paths: list[Path] | tuple[Path, ...],
    compute_signature: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist one derived artifact request block in metadata payload."""
    block: dict[str, Any] = {
        "request_signature": request_signature,
        "paths": [str(path) for path in sorted({Path(path) for path in paths})],
    }
    if compute_signature is not None:
        block["compute_signature"] = compute_signature
    if extra:
        block.update(extra)
    payload[state_key] = block
