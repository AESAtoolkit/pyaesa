import numpy as np

from pyaesa.shared.uncertainty_assessment.monte_carlo.convergence import (
    ConvergenceCheckpointCursor,
    MeanConvergenceAccumulator,
    convergence_run_checkpoints,
    iter_sparse_group_mean_updates,
    ordered_mean_convergence_reached,
    stable_relative,
)
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    sparse_group_run_means,
    sparse_public_row_group_membership_index,
    sparse_rows_to_overlapping_group_values,
)
from pyaesa.shared.uncertainty_assessment.io.run_writers import SparseRunRows
from pyaesa.shared.uncertainty_assessment.request.core import normalize_uncertainty_request


def test_convergence_checkpoints_and_cursor_crossings() -> None:
    runtime = normalize_uncertainty_request(
        family="demo",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 10, "stable_runs": 3},
        },
    )
    fixed_runtime = normalize_uncertainty_request(
        family="demo",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": True, "n_runs": 5},
            "convergence": {"active": False},
        },
    )

    assert convergence_run_checkpoints(runtime=runtime) == (3, 6, 9, 10)
    assert convergence_run_checkpoints(runtime=fixed_runtime) == (5,)

    cursor = ConvergenceCheckpointCursor.from_runtime(runtime=runtime)
    assert not cursor.reached(completed_runs=2)
    assert cursor.reached(completed_runs=4)
    cursor.mark_checked(completed_runs=4)
    assert not cursor.reached(completed_runs=4)
    assert cursor.reached(completed_runs=6)
    cursor.mark_checked(completed_runs=10)
    assert not cursor.reached(completed_runs=10)

    resumed = ConvergenceCheckpointCursor.from_runtime(runtime=runtime)
    resumed.advance_to_completed(completed_runs=6)
    assert not resumed.reached(completed_runs=6)
    assert resumed.reached(completed_runs=9)


def test_mean_convergence_accumulator_treats_only_nan_as_missing() -> None:
    accumulator = MeanConvergenceAccumulator.empty(row_count=3)
    accumulator.update(
        values=np.array(
            [
                [1.0, np.inf, np.nan],
                [3.0, np.inf, 5.0],
            ],
            dtype=np.float64,
        )
    )

    np.testing.assert_array_equal(accumulator.counts, [2, 2, 1])
    np.testing.assert_allclose(accumulator.means(), [2.0, np.inf, 5.0])
    assert not accumulator.check(completed_runs=2, rtol=0.01)

    accumulator.update(
        values=np.array(
            [
                [2.0, np.inf, 5.0],
                [2.0, np.inf, np.nan],
            ],
            dtype=np.float64,
        )
    )
    assert accumulator.check(completed_runs=4, rtol=0.01)
    assert accumulator.stable_run_count == 2

    accumulator.record_baseline(completed_runs=4)
    assert accumulator.stable_run_count == 0
    accumulator.update(values=np.array([[3.0, np.inf, 5.0]], dtype=np.float64))
    assert not accumulator.check(completed_runs=5, rtol=0.01)


def test_mean_convergence_accumulator_updates_sparse_group_means() -> None:
    public_row_groups = (("0", "1"), ("1",), ("2",))
    rows = SparseRunRows(
        run_index=np.array([0, 0, 0, 1, 1, 2], dtype=np.int64),
        public_row_id=np.array([0, 1, 2, 1, 2, 2], dtype=np.int64),
        values=np.array([1.0, 3.0, 5.0, 7.0, np.nan, 11.0], dtype=np.float64),
        value_column="value",
    )
    run_indices = np.array([0, 1, 2], dtype=np.int64)
    membership = sparse_public_row_group_membership_index(public_row_groups=public_row_groups)
    dense_values = np.array(
        [
            [2.0, 3.0, 5.0],
            [7.0, 7.0, np.nan],
            [np.nan, np.nan, 11.0],
        ],
        dtype=np.float64,
    )
    dense = MeanConvergenceAccumulator.empty(row_count=len(public_row_groups))
    dense.update(values=dense_values)
    sparse = MeanConvergenceAccumulator.empty(row_count=len(public_row_groups))
    row_runs, row_groups, values = sparse_rows_to_overlapping_group_values(
        sparse_rows=rows,
        run_indices=run_indices,
        public_row_group_index=membership,
    )

    sparse.accumulate_sparse_group_means(
        row_runs=row_runs,
        row_groups=row_groups,
        values=values,
    )
    chunked = MeanConvergenceAccumulator.empty(row_count=len(public_row_groups))
    bytes_per_group = (
        np.dtype(np.float64).itemsize
        + np.dtype(np.int64).itemsize
        + np.dtype(np.bool_).itemsize * 2
    )
    chunked.accumulate_sparse_group_means(
        row_runs=row_runs,
        row_groups=row_groups,
        values=values,
        memory_budget_bytes=bytes_per_group * len(public_row_groups),
    )
    empty = MeanConvergenceAccumulator.empty(row_count=len(public_row_groups))
    empty.accumulate_sparse_group_means(
        row_runs=np.array([], dtype=np.int64),
        row_groups=np.array([], dtype=np.int64),
        values=np.array([], dtype=np.float64),
    )
    empty_groups, empty_means = sparse_group_run_means(
        row_runs=np.array([], dtype=np.int64),
        row_groups=np.array([], dtype=np.int64),
        values=np.array([], dtype=np.float64),
        group_count=len(public_row_groups),
    )
    assert (
        list(
            iter_sparse_group_mean_updates(
                row_runs=np.array([], dtype=np.int64),
                row_groups=np.array([], dtype=np.int64),
                values=np.array([], dtype=np.float64),
                group_count=len(public_row_groups),
            )
        )
        == []
    )
    sparse.accumulate_group_observations(
        groups=np.array([], dtype=np.int64),
        values=np.array([], dtype=np.float64),
    )
    sparse.accumulate_group_observations(
        groups=np.array([0], dtype=np.int64),
        values=np.array([np.nan], dtype=np.float64),
    )

    np.testing.assert_allclose(sparse.sums, dense.sums)
    np.testing.assert_array_equal(sparse.counts, dense.counts)
    np.testing.assert_allclose(chunked.sums, dense.sums)
    np.testing.assert_array_equal(chunked.counts, dense.counts)
    np.testing.assert_array_equal(empty.counts, np.zeros(len(public_row_groups), dtype=np.int64))
    np.testing.assert_array_equal(empty_groups, np.array([], dtype=np.int64))
    np.testing.assert_array_equal(empty_means, np.array([], dtype=np.float64))


def test_ordered_mean_convergence_updates_all_checkpoint_targets() -> None:
    yearly = MeanConvergenceAccumulator.empty(row_count=1)
    cumulative = MeanConvergenceAccumulator.empty(row_count=1)
    yearly.update(values=np.array([[1.0], [1.0]], dtype=np.float64))
    cumulative.update(values=np.array([[5.0], [5.0]], dtype=np.float64))

    assert not ordered_mean_convergence_reached(
        targets=(yearly, cumulative),
        completed_runs=2,
        rtol=0.01,
    )

    yearly.update(values=np.array([[10.0], [10.0]], dtype=np.float64))
    cumulative.update(values=np.array([[5.0], [5.0]], dtype=np.float64))
    assert not ordered_mean_convergence_reached(
        targets=(yearly, cumulative),
        completed_runs=4,
        rtol=0.01,
    )
    assert cumulative.last_check_runs == 4

    yearly.update(values=np.array([[5.5], [5.5]], dtype=np.float64))
    cumulative.update(values=np.array([[5.0], [5.0]], dtype=np.float64))
    assert ordered_mean_convergence_reached(
        targets=(yearly, cumulative),
        completed_runs=6,
        rtol=0.01,
    )


def test_stable_relative_handles_nan_and_infinity_explicitly() -> None:
    assert stable_relative(
        previous=np.array([[np.nan, np.inf, -np.inf, 100.0]], dtype=np.float64),
        current=np.array([[np.nan, np.inf, -np.inf, 101.0]], dtype=np.float64),
        rtol=0.02,
    )
    assert not stable_relative(
        previous=np.array([[np.inf]], dtype=np.float64),
        current=np.array([[-np.inf]], dtype=np.float64),
        rtol=0.02,
    )
    assert not stable_relative(
        previous=np.array([[np.nan]], dtype=np.float64),
        current=np.array([[1.0]], dtype=np.float64),
        rtol=0.02,
    )
    assert not stable_relative(
        previous=np.array([[100.0]], dtype=np.float64),
        current=np.array([[103.0]], dtype=np.float64),
        rtol=0.02,
    )
