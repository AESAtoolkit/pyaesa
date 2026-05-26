"""Family neutral Monte Carlo mean convergence helpers."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyaesa.shared.runtime.memory import memory_bounded_rows
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    sparse_group_run_means,
)
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

    def accumulate_group_observations(
        self,
        *,
        groups: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Accumulate one pre collapsed observation per convergence target.

        `groups` contains target positions in this accumulator. `values`
        contains one value for each position after the caller has already
        reduced the current run window to target level, for example sparse run
        means or ASR frequency indicators. NaN values are missing observations
        and do not contribute to sums or counts.
        """
        if groups.size == 0:
            return
        observed = ~np.isnan(values)
        observed_groups = np.asarray(groups, dtype=np.int64)[observed]
        observed_values = np.asarray(values, dtype=np.float64)[observed]
        if observed_groups.size == 0:
            return
        np.add.at(self.sums, observed_groups, observed_values)
        np.add.at(self.counts, observed_groups, 1)

    def accumulate_sparse_group_means(
        self,
        *,
        row_runs: np.ndarray,
        row_groups: np.ndarray,
        values: np.ndarray,
        memory_budget_bytes: int | None = None,
    ) -> None:
        """Accumulate sparse selected row values collapsed to summary group means."""
        group_count = len(self.sums)
        runs = np.asarray(row_runs, dtype=np.int64)
        row_group_values = np.asarray(row_groups, dtype=np.int64)
        observed_values = np.asarray(values, dtype=np.float64)
        if runs.size == 0:
            return
        for groups, means in iter_sparse_group_mean_updates(
            row_runs=runs,
            row_groups=row_group_values,
            values=observed_values,
            group_count=group_count,
            memory_budget_bytes=memory_budget_bytes,
        ):
            self.accumulate_group_observations(groups=groups, values=means)

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


def _sparse_group_mean_slices(
    *,
    row_runs: np.ndarray,
    group_count: int,
    memory_budget_bytes: int | None,
) -> Iterator[tuple[int, int]]:
    runs_per_block = memory_bounded_rows(
        bytes_per_row=_sparse_group_mean_working_bytes_per_run(group_count=group_count),
        memory_budget_bytes=memory_budget_bytes,
    )
    first_run = int(row_runs[0])
    final_run = int(row_runs[-1]) + 1
    for run_start in range(first_run, final_run, runs_per_block):
        run_stop = min(run_start + runs_per_block, final_run)
        start = int(np.searchsorted(row_runs, run_start, side="left"))
        stop = int(np.searchsorted(row_runs, run_stop, side="left"))
        yield start, stop


def iter_sparse_group_mean_updates(
    *,
    row_runs: np.ndarray,
    row_groups: np.ndarray,
    values: np.ndarray,
    group_count: int,
    memory_budget_bytes: int | None = None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield memory bounded sparse means keyed by target group."""
    runs = np.asarray(row_runs, dtype=np.int64)
    row_group_values = np.asarray(row_groups, dtype=np.int64)
    observed_values = np.asarray(values, dtype=np.float64)
    if runs.size == 0:
        return
    for start, stop in _sparse_group_mean_slices(
        row_runs=runs,
        group_count=group_count,
        memory_budget_bytes=memory_budget_bytes,
    ):
        yield sparse_group_run_means(
            row_runs=runs[start:stop],
            row_groups=row_group_values[start:stop],
            values=observed_values[start:stop],
            group_count=group_count,
        )


def _sparse_group_mean_working_bytes_per_run(*, group_count: int) -> int:
    """Estimate temporary arrays used to collapse one sparse run to group means."""
    return int(group_count) * (
        np.dtype(np.float64).itemsize
        + np.dtype(np.int64).itemsize
        + np.dtype(np.bool_).itemsize * len(("observed", "nan_mask"))
    )


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


def mean_convergence_payload(
    *,
    reached: bool,
    completed_runs: int,
    runtime: UncertaintyRuntimeRequest,
) -> dict[str, Any]:
    """Return the standard Monte Carlo mean convergence payload."""
    return {
        "reached": bool(reached),
        "completed_runs": int(completed_runs),
        "max_runs": int(runtime.max_runs),
        "rtol": float(runtime.rtol),
        "stable_runs": int(runtime.stable_runs),
        "statistics": list(runtime.convergence_statistics),
    }


def mean_convergence_payload_for_targets(
    *,
    targets: tuple[MeanConvergenceAccumulator, ...],
    completed_runs: int,
    runtime: UncertaintyRuntimeRequest,
    check_convergence: bool,
) -> dict[str, Any] | None:
    """Return a reached payload when every mean target is stable at a checkpoint."""
    if not check_convergence or runtime.mode != "convergence":
        return None
    if not ordered_mean_convergence_reached(
        targets=targets,
        completed_runs=completed_runs,
        rtol=runtime.rtol,
    ):
        return None
    return mean_convergence_payload(
        reached=True,
        completed_runs=completed_runs,
        runtime=runtime,
    )


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
