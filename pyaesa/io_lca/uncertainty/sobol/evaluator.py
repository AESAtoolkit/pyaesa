"""IO-LCA source unit evaluator used by downstream ASR Sobol."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.io_lca.uncertainty.runtime.models import IOLCAUncertaintyPlan
from pyaesa.io_lca.uncertainty.evaluation.sampling import sample_io_lca_lcia_matrix
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch


@dataclass(frozen=True)
class IOLCASobolEvaluationContext:
    """Prepared IO-LCA Sobol evaluator context."""

    plan: IOLCAUncertaintyPlan


def build_io_lca_sobol_evaluation_context(
    *,
    plan: IOLCAUncertaintyPlan,
) -> IOLCASobolEvaluationContext:
    """Build the canonical IO-LCA source unit evaluator context."""
    return IOLCASobolEvaluationContext(plan=plan)


def evaluate_io_lca_sobol_units(
    *,
    context: IOLCASobolEvaluationContext,
    units: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate unit interval source values into IO-LCA public row values."""
    batch = RunBatch(
        batch_index=0,
        start_run_index=0,
        stop_run_index=units.shape[0],
        rng_seed=0,
        run_index_values=tuple(range(units.shape[0])),
    )
    return (
        context.plan.identity,
        sample_io_lca_lcia_matrix(
            plan=context.plan,
            batch=batch,
            unit_values=units[:, 0],
        ),
    )
