"""aCC uncertainty runtime memory block planning."""

from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyPlan
from pyaesa.shared.uncertainty_assessment.request.core import (
    BatchMemoryBlock,
    sparse_selected_run_memory_blocks,
)


def acc_batch_memory_blocks(*, plan: ACCUncertaintyPlan) -> tuple[BatchMemoryBlock, ...]:
    """Return aCC batch memory blocks derived from active plan dimensions."""
    if plan.acc_run_layout != "sparse_selected_rows":
        return (
            BatchMemoryBlock("acc_source_values", len(plan.identity)),
            BatchMemoryBlock("acc_summary_values", len(plan.summary_identity)),
        )
    return sparse_selected_run_memory_blocks(
        prefix="acc",
        public_row_count=len(plan.identity),
        summary_row_count=len(plan.summary_identity),
        filters_and_sorts_output=True,
    )
