"""Public and internal settings for AR6 figure sampling convergence."""

from typing import TypedDict


class SamplingFigureConfig(TypedDict):
    """Validated AR6 figure sampling configuration."""

    relative_tolerance: float
    max_runs_per_bucket: int


RUN_BATCH_SIZE = 10000
STABLE_CHECKS_REQUIRED = 3
CONVERGENCE_LOG_COLUMNS = (
    "variable",
    "method",
    "distribution_kind",
    "category",
    "ssp_family",
    "rng_seed",
    "final_runs_per_bucket",
    "run_batch_size",
    "maximum_runs_per_bucket",
    "relative_tolerance",
    "stable_checks_required",
    "mean",
    "median",
    "p25",
    "p75",
    "p5",
    "p95",
)


def validate_sampling_figure_config(
    *,
    figure_convergence_tol: float,
    figure_convergence_max_runs: int,
) -> SamplingFigureConfig:
    """Validate and normalize user facing AR6 figure sampling controls."""
    tolerance = float(figure_convergence_tol)
    max_runs = int(figure_convergence_max_runs)
    if tolerance <= 0.0:
        raise ValueError("'figure_convergence_tol' must be a positive number.")
    if max_runs <= 0:
        raise ValueError("'figure_convergence_max_runs' must be a positive integer.")
    return SamplingFigureConfig(
        relative_tolerance=tolerance,
        max_runs_per_bucket=max_runs,
    )


def minimum_completed_runs_per_bucket_for_convergence(
    *,
    run_batch_size: int = RUN_BATCH_SIZE,
    stable_checks_required: int = STABLE_CHECKS_REQUIRED,
) -> int:
    """Return the earliest per bucket run count at which convergence can be accepted."""
    batch_size = int(run_batch_size)
    stable_checks = int(stable_checks_required)
    # The first completed batch establishes the baseline snapshot; each stable
    # checkpoint comparison then needs one additional batch.
    return batch_size * (stable_checks + 1)
