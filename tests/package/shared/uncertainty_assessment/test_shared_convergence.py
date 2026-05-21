import numpy as np

from pyaesa.shared.uncertainty_assessment.monte_carlo.convergence import (
    ConvergenceCheckpointCursor,
    MeanConvergenceAccumulator,
    convergence_run_checkpoints,
    ordered_mean_convergence_reached,
    stable_relative,
)
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
