"""aSoCC Monte Carlo convergence start state helpers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.asocc.uncertainty.engine.evaluation.summary_identity import summary_identity_groups
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    sparse_public_row_group_membership_index,
)
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.asocc.uncertainty.schema.public_rows import (
    ASOCC_UNCERTAINTY_CSV_DTYPES,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import fixed_run_plan
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table


@dataclass(frozen=True)
class ConvergenceStartState:
    """Initial convergence identity state from an existing appendable run."""

    identity: pd.DataFrame | None
    identity_written: bool
    public_row_groups: tuple[tuple[str, ...], ...]
    public_row_group_index: np.ndarray | None
    row_count: int
    completed_runs: int
    last_check_runs: int
    stable_run_count: int


def remaining_run_plan(
    *,
    n_runs: int,
    batch_size: int,
    run_seed: int,
    start_run_index: int,
):
    """Return the missing run index interval, or None when the request is complete."""
    remaining = int(n_runs) - int(start_run_index)
    if remaining <= 0:
        return None
    return fixed_run_plan(
        n_runs=remaining,
        batch_size=batch_size,
        seed=run_seed,
        start_run_index=start_run_index,
    )


def initial_convergence_state(
    *,
    paths: AsoccUncertaintyRunPaths,
    output_format: str,
    sources: SourceActivationPlan,
    append_existing: bool,
    sparse: bool,
    completed_runs: int,
) -> ConvergenceStartState:
    """Return initial convergence identity state without private cache replay."""
    if not append_existing:
        return ConvergenceStartState(
            identity=None,
            identity_written=False,
            public_row_groups=(),
            public_row_group_index=None,
            row_count=0,
            completed_runs=0,
            last_check_runs=0,
            stable_run_count=0,
        )
    identity = read_uncertainty_table(
        path=paths.public_row_identity,
        output_format=output_format,
        csv_dtypes=ASOCC_UNCERTAINTY_CSV_DTYPES,
    )
    _summary_identity, public_row_groups = summary_identity_groups(
        identity=identity,
        sources=sources,
    )
    public_row_group_index = (
        sparse_public_row_group_membership_index(public_row_groups=public_row_groups)
        if sparse
        else None
    )
    row_count = len(public_row_groups)
    return ConvergenceStartState(
        identity=identity,
        identity_written=True,
        public_row_groups=public_row_groups,
        public_row_group_index=public_row_group_index,
        row_count=row_count,
        completed_runs=int(completed_runs),
        last_check_runs=int(completed_runs),
        stable_run_count=0,
    )
