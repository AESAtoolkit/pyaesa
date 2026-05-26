"""aCC compact run matrix evaluation."""

from typing import cast

import numpy as np

from pyaesa.acc.uncertainty.evaluation.run_inputs import (
    asocc_public_row_count,
    fixed_cc_values_for_runs,
    iter_asocc_values,
)
from pyaesa.acc.uncertainty.runtime.models import ACCBranchPlan, ACCUncertaintyPlan


def iter_acc_run_batches(
    *,
    plan: ACCUncertaintyPlan,
    output_format: str,
    start_run_index: int = 0,
    stop_run_index: int | None = None,
    batch_size: int | None = None,
):
    """Yield ACC compact run matrix batches."""
    for run_indices, asocc_values in iter_asocc_values(
        asocc_input=plan.asocc_input,
        output_format=output_format,
        public_row_count=asocc_public_row_count(plan=plan),
        start_run_index=start_run_index,
        stop_run_index=cast(int, stop_run_index),
        batch_size=batch_size,
    ):
        yield (
            run_indices,
            evaluate_acc_value_matrix(
                branch_plans=plan.branch_plans,
                asocc_values=asocc_values,
                cc_values=fixed_cc_values_for_runs(
                    run_indices=run_indices,
                    deterministic_cc_values=plan.deterministic_cc_values,
                ),
            ),
        )


def evaluate_acc_value_matrix(
    *,
    branch_plans: tuple[ACCBranchPlan, ...],
    asocc_values: np.ndarray,
    cc_values: np.ndarray | None,
) -> np.ndarray:
    """Evaluate ACC values for one compact run or Sobol batch."""
    blocks: list[np.ndarray] = []
    for branch in branch_plans:
        share = asocc_values[:, branch.asocc_positions]
        if branch.cc_type == "static":
            blocks.append(share * cast(np.ndarray, branch.static_cc_values)[None, :])
            continue
        dynamic_cc_values = cast(np.ndarray, cc_values)
        converted_cc = (
            dynamic_cc_values[:, cast(np.ndarray, branch.cc_positions)]
            * cast(np.ndarray, branch.dynamic_cc_factors)[None, :]
        )
        blocks.append(share * converted_cc)
    return np.concatenate(blocks, axis=1)
