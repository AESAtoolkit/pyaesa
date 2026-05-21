"""Family neutral Monte Carlo mean convergence helpers."""

from dataclasses import dataclass

import numpy as np

from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest


def convergence_run_checkpoints(*, runtime: UncertaintyRuntimeRequest) -> tuple[int, ...]:
    """Return stable run interval checkpoints for cumulative mean convergence."""
    if runtime.mode == "fixed":
        return (int(runtime.n_runs),)
    checkpoints: list[int] = []
    target = min(int(runtime.stable_runs), int(runtime.n_runs))
    while target < int(runtime.n_runs):
        checkpoints.append(target)
        target = min(target + int(runtime.stable_runs), int(runtime.n_runs))
    checkpoints.append(int(runtime.n_runs))
    return tuple(checkpoints)


@dataclass
class ConvergenceCheckpointCursor:
    """Ordered checkpoint cursor that fires when a batch reaches or crosses a checkpoint."""

    checkpoints: tuple[int, ...]
    index: int = 0

    @classmethod
    def from_runtime(cls, *, runtime: UncertaintyRuntimeRequest) -> "ConvergenceCheckpointCursor":
        return cls(checkpoints=convergence_run_checkpoints(runtime=runtime))

    def advance_to_completed(self, *, completed_runs: int) -> None:
        """Advance past checkpoints already covered by existing persisted runs."""
        completed = int(completed_runs)
        while self.index < len(self.checkpoints) and self.checkpoints[self.index] <= completed:
            self.index += 1

    def reached(self, *, completed_runs: int) -> bool:
        """Return whether completed runs reached the next pending checkpoint."""
        return (
            self.index < len(self.checkpoints)
            and int(completed_runs) >= self.checkpoints[self.index]
        )

    def mark_checked(self, *, completed_runs: int) -> None:
        """Advance past all checkpoints covered by the checked completed run count."""
        self.advance_to_completed(completed_runs=completed_runs)


@dataclass
class MeanConvergenceAccumulator:
    """Streaming cumulative mean state for convergence targets."""

    sums: np.ndarray
    counts: np.ndarray
    previous_means: np.ndarray | None = None
    last_check_runs: int = 0
    stable_run_count: int = 0

    @classmethod
    def empty(cls, *, row_count: int) -> "MeanConvergenceAccumulator":
        return cls(
            sums=np.zeros(int(row_count), dtype=np.float64),
            counts=np.zeros(int(row_count), dtype=np.int64),
        )

    def update(self, *, values: np.ndarray) -> None:
        observed = ~np.isnan(values)
        self.sums += np.where(observed, values, 0.0).sum(axis=0)
        self.counts += observed.sum(axis=0)

    def means(self) -> np.ndarray:
        """Return current cumulative means, treating only NaN as missing."""
        return np.divide(
            self.sums,
            self.counts,
            out=np.full(len(self.sums), np.nan, dtype=np.float64),
            where=self.counts > 0,
        )

    def record_baseline(self, *, completed_runs: int) -> None:
        """Record the current cumulative mean as the next convergence baseline."""
        self.previous_means = self.means().copy()
        self.last_check_runs = int(completed_runs)
        self.stable_run_count = 0

    def check(self, *, completed_runs: int, rtol: float) -> bool:
        current = self.means()
        if self.previous_means is None:
            self.previous_means = current.copy()
            self.last_check_runs = int(completed_runs)
            self.stable_run_count = 0
            return False
        reached = stable_relative(previous=self.previous_means, current=current, rtol=rtol)
        self.stable_run_count = int(completed_runs) - self.last_check_runs if reached else 0
        self.previous_means = current.copy()
        self.last_check_runs = int(completed_runs)
        return reached


def ordered_mean_convergence_reached(
    *,
    targets: tuple[MeanConvergenceAccumulator, ...],
    completed_runs: int,
    rtol: float,
) -> bool:
    """Return whether every ordered mean target is stable at one checkpoint."""
    reached = True
    for target in targets:
        if not target.check(completed_runs=completed_runs, rtol=rtol):
            reached = False
    return reached


def stable_relative(*, previous: np.ndarray, current: np.ndarray, rtol: float) -> bool:
    """Return whether all non NaN means are stable within relative tolerance."""
    both_nan = np.isnan(previous) & np.isnan(current)
    either_nan = np.isnan(previous) | np.isnan(current)
    same_infinite = (
        np.isinf(previous) & np.isinf(current) & (np.signbit(previous) == np.signbit(current))
    )
    finite_pair = ~(either_nan | np.isinf(previous) | np.isinf(current))
    stable = np.zeros(previous.shape, dtype=bool)
    scale = np.maximum(np.abs(previous[finite_pair]), np.finfo(np.float64).tiny)
    stable[finite_pair] = (
        np.abs(current[finite_pair] - previous[finite_pair]) <= float(rtol) * scale
    )
    return bool(np.all(both_nan | same_infinite | (finite_pair & stable)))
