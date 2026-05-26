"""Saltelli design chunks for Sobol variance decomposition."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from pyaesa.shared.runtime.memory import runtime_working_budget_bytes


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


def sobol_chunk_rows(
    *,
    output_count: int,
    dimension_count: int,
    confidence_resamples: int,
) -> int:
    """Return evaluation rows per chunk under the uncertainty memory target."""
    bytes_per_row = _sobol_chunk_bytes_per_base_row(
        output_count=output_count,
        dimension_count=dimension_count,
    )
    accumulator_bytes = _sobol_accumulator_bytes(
        output_count=output_count,
        dimension_count=dimension_count,
        confidence_resamples=confidence_resamples,
    )
    budget = runtime_working_budget_bytes(
        memory_budget_bytes=None,
        minimal_working_block_bytes=bytes_per_row + accumulator_bytes,
    )
    return max(1, int((budget - accumulator_bytes) // bytes_per_row))


def _sobol_chunk_bytes_per_base_row(*, output_count: int, dimension_count: int) -> int:
    saltelli_blocks = len(("a", "b")) + int(dimension_count)
    float_bytes = np.dtype(np.float64).itemsize
    output_bytes = saltelli_blocks * max(1, int(output_count)) * float_bytes
    design_bytes = saltelli_blocks * max(1, int(dimension_count)) * float_bytes
    return max(1, output_bytes + design_bytes)


def _sobol_accumulator_bytes(
    *,
    output_count: int,
    dimension_count: int,
    confidence_resamples: int,
) -> int:
    float_bytes = np.dtype(np.float64).itemsize
    output_cells = max(1, int(output_count))
    source_output_cells = max(1, int(dimension_count)) * output_cells
    bootstrap_output_cells = max(1, int(confidence_resamples)) * output_cells
    bootstrap_source_output_cells = max(1, int(confidence_resamples)) * source_output_cells
    output_moments = len(
        ("variance_count", "variance_sum", "variance_sumsq", "center_count", "center_sum")
    )
    source_moments = len(("s1_count", "s1_b_delta_sum", "s1_delta_sum", "st_count", "st_sum"))
    return float_bytes * (
        output_moments * output_cells
        + source_moments * source_output_cells
        + output_moments * bootstrap_output_cells
        + source_moments * bootstrap_source_output_cells
    )


def iter_saltelli_chunks(
    *,
    design: SobolDesign,
    chunk_rows: int,
    start_row: int = 0,
    stop_row: int | None = None,
) -> Iterator[SobolEvaluationChunk]:
    """Yield memory bounded Saltelli chunks in base sample order."""
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
        yield SobolEvaluationChunk(row_start=start, a=a, b=b, ab=tuple(ab))
