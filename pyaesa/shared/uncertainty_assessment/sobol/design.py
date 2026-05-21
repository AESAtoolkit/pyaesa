"""Saltelli design chunks for Sobol variance decomposition."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

SOBOL_EVALUATION_MEMORY_BYTES = 3_000_000_000
# The chunk planner budgets for family evaluators that allocate temporary
# alignment arrays beyond the final float64 A, B, and A_Bi output matrices.
SOBOL_EVALUATION_TEMPORARY_FACTOR = 64


@dataclass(frozen=True)
class SobolDesign:
    """Sobol A and B base matrices."""

    a: np.ndarray
    b: np.ndarray


@dataclass(frozen=True)
class SobolEvaluationChunk:
    """One memory bounded Saltelli design chunk."""

    row_start: int
    a: np.ndarray
    b: np.ndarray
    ab: tuple[np.ndarray, ...]


def saltelli_design(*, n_base_samples: int, dimension_count: int) -> SobolDesign:
    """Return balanced Sobol A and B design matrices."""
    sampler = qmc.Sobol(d=2 * dimension_count, scramble=False)
    base = sampler.random_base2(m=int(np.log2(n_base_samples)))
    a = base[:, :dimension_count]
    b = base[:, dimension_count:]
    return SobolDesign(a=a, b=b)


def sobol_base_sequence(
    *,
    mode: str,
    n_base_samples: int,
    max_base_samples: int,
) -> tuple[int, ...]:
    """Return the fixed or convergence Sobol base sample sequence."""
    if mode == "fixed":
        return (n_base_samples,)
    values = []
    n_base = n_base_samples
    while n_base <= max_base_samples:
        values.append(n_base)
        n_base *= 2
    return tuple(values)


def sobol_chunk_rows(*, output_count: int, dimension_count: int) -> int:
    """Return evaluation rows per chunk under the uncertainty memory target."""
    saltelli_block_count = dimension_count + 2
    matrix_bytes_per_row = (
        max(1, output_count) * 8 * SOBOL_EVALUATION_TEMPORARY_FACTOR * saltelli_block_count
    )
    accumulator_bytes = max(1, dimension_count) * max(1, output_count) * 8 * 4
    available = max(matrix_bytes_per_row, SOBOL_EVALUATION_MEMORY_BYTES - accumulator_bytes)
    return max(1, int(available // matrix_bytes_per_row))


def iter_saltelli_chunks(
    *,
    design: SobolDesign,
    chunk_rows: int,
    start_row: int = 0,
    stop_row: int | None = None,
) -> tuple[SobolEvaluationChunk, ...]:
    """Return memory bounded Saltelli chunks in base sample order."""
    chunks = []
    final_row = design.a.shape[0] if stop_row is None else stop_row
    for start in range(start_row, final_row, chunk_rows):
        stop = min(start + chunk_rows, final_row)
        a = design.a[start:stop]
        b = design.b[start:stop]
        ab = []
        for index in range(a.shape[1]):
            mixed = a.copy()
            mixed[:, index] = b[:, index]
            ab.append(mixed)
        chunks.append(SobolEvaluationChunk(row_start=start, a=a, b=b, ab=tuple(ab)))
    return tuple(chunks)
