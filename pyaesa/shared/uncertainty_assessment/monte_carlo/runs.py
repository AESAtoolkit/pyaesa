"""Run plan ownership for uncertainty runs."""

from dataclasses import dataclass
import hashlib
import secrets

import numpy as np


@dataclass(frozen=True)
class RunBatch:
    """One Monte Carlo run batch."""

    batch_index: int
    start_run_index: int
    stop_run_index: int
    rng_seed: int
    run_index_values: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.batch_index < 0:
            raise ValueError("Run batch index must be non-negative.")
        if self.start_run_index < 0 or self.stop_run_index <= self.start_run_index:
            raise ValueError("Run batch indices must define a non-empty positive range.")
        if self.run_index_values is not None:
            object.__setattr__(
                self,
                "run_index_values",
                tuple(int(value) for value in self.run_index_values),
            )

    @property
    def n_runs(self) -> int:
        """Return the number of runs in this batch."""
        if self.run_index_values is not None:
            return len(self.run_index_values)
        return self.stop_run_index - self.start_run_index

    def run_indices(self) -> np.ndarray:
        """Return run indices as one compact numeric array."""
        if self.run_index_values is not None:
            return np.fromiter(self.run_index_values, dtype=np.int64)
        return np.arange(self.start_run_index, self.stop_run_index, dtype=np.int64)

    def rng(self) -> np.random.Generator:
        """Return the random generator owned by this batch."""
        return np.random.default_rng(self.rng_seed)


@dataclass(frozen=True)
class FixedRunPlan:
    """Fixed-run Monte Carlo execution plan."""

    n_runs: int
    batch_size: int
    seed: int
    start_run_index: int = 0

    def __post_init__(self) -> None:
        if self.n_runs <= 0:
            raise ValueError("n_runs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.start_run_index < 0:
            raise ValueError("start_run_index must be non-negative.")

    @property
    def batch_count(self) -> int:
        """Return the number of batches in the plan."""
        return (
            self.stop_run_index + self.batch_size - 1
        ) // self.batch_size - self.start_run_index // self.batch_size

    @property
    def stop_run_index(self) -> int:
        """Return the exclusive stop run index for the plan."""
        return self.start_run_index + self.n_runs

    def batches(self) -> list[RunBatch]:
        """Return all run batches in execution order."""
        first_batch = self.start_run_index // self.batch_size
        absolute_batch_count = (self.stop_run_index + self.batch_size - 1) // self.batch_size
        child_sequences = np.random.SeedSequence(self.seed).spawn(absolute_batch_count)
        batches: list[RunBatch] = []
        for batch_index in range(first_batch, absolute_batch_count):
            child = child_sequences[batch_index]
            start = max(self.start_run_index, batch_index * self.batch_size)
            stop = min((batch_index + 1) * self.batch_size, self.stop_run_index)
            rng_seed = int(child.generate_state(1, dtype=np.uint32)[0])
            batches.append(
                RunBatch(
                    batch_index=batch_index,
                    start_run_index=start,
                    stop_run_index=stop,
                    rng_seed=rng_seed,
                )
            )
        return batches


def fixed_run_plan(
    *,
    n_runs: int,
    batch_size: int,
    seed: int | None = None,
    start_run_index: int = 0,
) -> FixedRunPlan:
    """Build a fixed-run execution plan.

    Args:
        n_runs: Number of new runs to execute.
        batch_size: Maximum number of runs per execution batch.
        seed: Optional reproducibility seed. When omitted, a fresh seed is
            allocated by the runtime.
        start_run_index: First run index for this plan.

    Returns:
        Fixed-run execution plan.
    """
    resolved_seed = secrets.randbits(63) if seed is None else int(seed)
    return FixedRunPlan(
        n_runs=int(n_runs),
        batch_size=int(batch_size),
        seed=resolved_seed,
        start_run_index=int(start_run_index),
    )


def run_seed_from_run_id(*, run_id: str) -> int:
    """Return the deterministic internal random seed for one Monte Carlo run id."""
    digest = hashlib.sha256(str(run_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
