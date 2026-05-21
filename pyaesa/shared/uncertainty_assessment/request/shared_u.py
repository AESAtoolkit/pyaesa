"""Deterministic shared percentile positions for linked uncertainty drivers."""

import hashlib
import json
from typing import Any

import numpy as np


def stable_json_key(*, payload: dict[str, Any]) -> str:
    """Return a compact deterministic JSON key for one payload."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def deterministic_shared_u_matrix(
    *,
    shared_u_keys: np.ndarray,
    run_indices: np.ndarray,
) -> np.ndarray:
    """Return deterministic percentile positions as run by shared key matrix."""
    key_bytes = [_shared_u_key_json_bytes(shared_u_key=str(key)) for key in shared_u_keys.tolist()]
    out = np.empty((len(run_indices), len(key_bytes)), dtype=np.float64)
    for run_position, run_index in enumerate(run_indices):
        for key_position, encoded_key in enumerate(key_bytes):
            out[run_position, key_position] = _deterministic_shared_u_from_key_bytes(
                shared_u_key_bytes=encoded_key,
                run_index=int(run_index),
            )
    return out


def _deterministic_shared_u_from_key_bytes(*, shared_u_key_bytes: bytes, run_index: int) -> float:
    digest = hashlib.sha256(
        b'{"run_index":'
        + str(int(run_index)).encode("ascii")
        + b',"shared_u_key":'
        + shared_u_key_bytes
        + b"}"
    ).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(integer / 2**64)


def _shared_u_key_json_bytes(*, shared_u_key: str) -> bytes:
    return json.dumps(str(shared_u_key)).encode("utf-8")
