"""ASR uncertainty runtime memory block planning."""

from typing import Any

from pyaesa.shared.uncertainty_assessment.request.core import (
    BatchMemoryBlock,
    sparse_selected_run_memory_blocks,
)


def asr_batch_memory_blocks(*, plan: Any) -> tuple[BatchMemoryBlock, ...]:
    """Return ASR batch memory blocks derived from active plan dimensions."""
    blocks = [
        BatchMemoryBlock("lca_input_values", len(plan.identity)),
        BatchMemoryBlock("acc_input_values", len(plan.identity)),
    ]
    if plan.asr_run_layout == "sparse_selected_rows":
        blocks.extend(
            sparse_selected_run_memory_blocks(
                prefix="asr",
                public_row_count=len(plan.identity),
                summary_row_count=len(plan.summary_public_row_groups),
                filters_and_sorts_output=False,
            )
        )
    else:
        blocks.append(BatchMemoryBlock("yearly_summary_values", len(plan.summary_identity)))
        selected_component_arrays = ("yearly_lca", "yearly_acc")
        if plan.has_cumulative_outputs:
            selected_component_arrays = (
                *selected_component_arrays,
                "cumulative_lca",
                "cumulative_acc",
            )
        blocks.append(
            BatchMemoryBlock(
                "asr_selected_component_values",
                len(plan.identity),
                len(selected_component_arrays),
            )
        )
    if plan.has_cumulative_outputs:
        blocks.extend(
            [
                BatchMemoryBlock("cumulative_numerator_sums", len(plan.cumulative_identity)),
                BatchMemoryBlock("cumulative_denominator_sums", len(plan.cumulative_identity)),
                BatchMemoryBlock("cumulative_output_values", len(plan.cumulative_identity)),
                BatchMemoryBlock(
                    "cumulative_summary_values",
                    len(plan.cumulative_summary_identity),
                ),
            ]
        )
    return tuple(blocks)
