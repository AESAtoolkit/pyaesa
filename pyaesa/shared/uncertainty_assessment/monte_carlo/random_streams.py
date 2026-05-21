"""Run indexed random values for uncertainty source owners."""

import hashlib

import numpy as np

from pyaesa.shared.uncertainty_assessment.request.shared_u import stable_json_key


def uniform_by_run_index(*, stream_name: str, run_indices: np.ndarray) -> np.ndarray:
    """Return deterministic uniform values keyed by stream name and run index."""
    runs = np.asarray(run_indices, dtype=np.int64)
    return np.fromiter(
        (
            _uniform_value(stream_name=str(stream_name), run_index=int(run_index))
            for run_index in runs
        ),
        dtype=np.float64,
        count=len(runs),
    )


def _uniform_value(*, stream_name: str, run_index: int) -> float:
    digest = hashlib.sha256(
        stable_json_key(
            payload={"stream_name": str(stream_name), "run_index": int(run_index)}
        ).encode("utf-8")
    ).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(integer / 2**64)
