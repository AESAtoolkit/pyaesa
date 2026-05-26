"""Family-neutral uncertainty public request normalization."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pyaesa.shared.runtime.memory import runtime_working_budget_bytes
from pyaesa.shared.uncertainty_assessment.io.formats import normalize_uncertainty_output_format

DEFAULT_CONVERGENCE_MAX_RUNS = 500_000
DEFAULT_CONVERGENCE_RTOL = 0.05
DEFAULT_CONVERGENCE_STABLE_RUNS = 10_000
CONVERGENCE_STATISTICS: tuple[str, ...] = ("mean",)
DEFAULT_CONVERGENCE_STATISTICS: tuple[str, ...] = ("mean",)
DEFAULT_FIXED_RUNS = 1000


@dataclass(frozen=True)
class UncertaintyRuntimeRequest:
    """Normalized family-neutral uncertainty runtime request."""

    family: str
    mode: str
    output_format: str
    n_runs: int
    max_runs: int
    batch_size: int
    rtol: float
    stable_runs: int
    convergence_statistics: tuple[str, ...]


@dataclass(frozen=True)
class BatchMemoryBlock:
    """One run scoped memory block used by the batch planner.

    The default block is one float64 array, matching public uncertainty value
    arrays. Owners add named blocks when a path retains additional arrays.
    """

    name: str
    row_count: int
    array_count: int = 1
    dtype_bytes: int = 8


def normalize_uncertainty_request(
    *,
    family: str,
    output_format: str,
    mc_parameters: dict[str, Any] | None,
) -> UncertaintyRuntimeRequest:
    """Normalize family-neutral public uncertainty runtime options.

    Args:
        family: Public function family identifier.
        output_format: Requested public uncertainty table format.
        mc_parameters: Optional public Monte Carlo parameter mapping with
            ``fixed`` and ``convergence`` blocks. Each block has an ``active``
            boolean, and exactly one block must be active. When omitted or
            empty, the request uses convergence mode defaults.

    Returns:
        Normalized runtime request.
    """
    fixed_params, convergence_params = _normalize_mode_parameters(mc_parameters)
    active_modes = tuple(
        mode
        for mode, params in (("fixed", fixed_params), ("convergence", convergence_params))
        if bool(params["active"])
    )
    if len(active_modes) != 1:
        active = ", ".join(active_modes) if active_modes else "none"
        raise ValueError(
            "mc_parameters must activate exactly one Monte Carlo mode: set either "
            "fixed.active or convergence.active to true. Active modes: "
            f"{active}."
        )
    mode = active_modes[0]

    fixed_n_runs = _positive_int(fixed_params["n_runs"], field="fixed.n_runs")
    convergence_n_runs = _positive_int(
        convergence_params["max_runs"],
        field="convergence.max_runs",
    )
    n_runs = convergence_n_runs if mode == "convergence" else fixed_n_runs
    max_runs = n_runs if mode == "convergence" else 0
    rtol = float(convergence_params["rtol"])
    if rtol < 0.0:
        raise ValueError("convergence.rtol must be non negative.")
    stable_runs = _positive_int(
        convergence_params["stable_runs"],
        field="convergence.stable_runs",
    )
    convergence_statistics = _normalize_convergence_statistics(
        convergence_params["convergence_statistics"]
    )
    batch_size = min(stable_runs, n_runs) if mode == "convergence" else n_runs
    return UncertaintyRuntimeRequest(
        family=str(family),
        mode=mode,
        output_format=normalize_uncertainty_output_format(output_format),
        n_runs=n_runs,
        max_runs=max_runs,
        batch_size=batch_size,
        rtol=rtol,
        stable_runs=stable_runs,
        convergence_statistics=convergence_statistics,
    )


def memory_bounded_batch_size(
    *,
    runtime: UncertaintyRuntimeRequest,
    primary_block: BatchMemoryBlock,
    extra_blocks: Iterable[BatchMemoryBlock] = (),
    memory_budget_bytes: int | None = None,
) -> int:
    """Return an internal run batch size bounded by owner supplied memory blocks."""
    bytes_per_run = _block_bytes_per_run(primary_block)
    bytes_per_run += sum(_block_bytes_per_run(block) for block in extra_blocks)
    budget = runtime_working_budget_bytes(
        memory_budget_bytes=memory_budget_bytes,
        minimal_working_block_bytes=bytes_per_run,
    )
    memory_batch_size = max(1, budget // bytes_per_run)
    return min(
        int(runtime.n_runs),
        int(runtime.batch_size),
        int(memory_batch_size),
    )


def sparse_selected_run_memory_blocks(
    *,
    prefix: str,
    public_row_count: int,
    summary_row_count: int,
    filters_and_sorts_output: bool,
) -> tuple[BatchMemoryBlock, ...]:
    """Return common retained memory blocks for sparse selected run outputs."""
    sparse_columns = ("run_index", "public_row_id", "value")
    sparse_column_count = len(sparse_columns)
    summary_arrays = ("sum", "count", "value")
    blocks = [
        BatchMemoryBlock(
            f"{prefix}_sparse_source_window_columns", public_row_count, sparse_column_count
        ),
        BatchMemoryBlock(
            f"{prefix}_sparse_source_decoded_columns", public_row_count, sparse_column_count
        ),
        BatchMemoryBlock(
            f"{prefix}_sparse_source_reader_work_columns", public_row_count, sparse_column_count
        ),
        BatchMemoryBlock(
            f"{prefix}_sparse_source_reader_masks",
            public_row_count,
            len(("ready", "pending", "range")),
            dtype_bytes=1,
        ),
        BatchMemoryBlock(
            f"{prefix}_sparse_output_window_columns", public_row_count, sparse_column_count
        ),
        BatchMemoryBlock(
            f"{prefix}_sparse_output_concat_columns", public_row_count, sparse_column_count
        ),
        BatchMemoryBlock(
            f"{prefix}_sparse_output_render_columns", public_row_count, sparse_column_count
        ),
        BatchMemoryBlock(f"{prefix}_sparse_summary_source_positions", public_row_count),
        BatchMemoryBlock(f"{prefix}_sparse_summary_group_ids", public_row_count),
        BatchMemoryBlock(f"{prefix}_sparse_summary_values", summary_row_count, len(summary_arrays)),
    ]
    if filters_and_sorts_output:
        blocks.extend(
            [
                BatchMemoryBlock(f"{prefix}_sparse_output_finite_mask", public_row_count, 1, 1),
                BatchMemoryBlock(f"{prefix}_sparse_output_sort_order", public_row_count),
                BatchMemoryBlock(
                    f"{prefix}_sparse_output_sorted_columns",
                    public_row_count,
                    sparse_column_count,
                ),
            ]
        )
    return tuple(blocks)


def _block_bytes_per_run(block: BatchMemoryBlock) -> int:
    return int(block.row_count) * int(block.array_count) * int(block.dtype_bytes)


def _normalize_mode_parameters(
    mc_parameters: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    params = dict(mc_parameters or {})
    fixed = _normalize_mode_block(
        params.pop(
            "fixed",
            {
                "active": False,
                "n_runs": DEFAULT_FIXED_RUNS,
            },
        ),
        mode="fixed",
        defaults={
            "active": False,
            "n_runs": DEFAULT_FIXED_RUNS,
        },
    )
    convergence = _normalize_mode_block(
        params.pop(
            "convergence",
            {
                "active": True,
                "max_runs": DEFAULT_CONVERGENCE_MAX_RUNS,
                "rtol": DEFAULT_CONVERGENCE_RTOL,
                "stable_runs": DEFAULT_CONVERGENCE_STABLE_RUNS,
                "convergence_statistics": DEFAULT_CONVERGENCE_STATISTICS,
            },
        ),
        mode="convergence",
        defaults={
            "active": True,
            "max_runs": DEFAULT_CONVERGENCE_MAX_RUNS,
            "rtol": DEFAULT_CONVERGENCE_RTOL,
            "stable_runs": DEFAULT_CONVERGENCE_STABLE_RUNS,
            "convergence_statistics": DEFAULT_CONVERGENCE_STATISTICS,
        },
    )
    if params:
        raise ValueError(f"Unsupported top level mc_parameters keys: {sorted(params)}.")
    return fixed, convergence


def _normalize_mode_block(
    value: object,
    *,
    mode: str,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"mc_parameters.{mode} must be a dictionary.")
    block = {**defaults, **value}
    unsupported = sorted(set(block) - set(defaults))
    if unsupported:
        raise ValueError(f"Unsupported mc_parameters.{mode} keys: {unsupported}.")
    active = block["active"]
    if not isinstance(active, bool):
        raise ValueError(f"mc_parameters.{mode}.active must be a boolean.")
    return block


def _positive_int(value: Any, *, field: str) -> int:
    parsed: int | None = None
    if not isinstance(value, bool):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = None
    if parsed is None or parsed <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return parsed


def _normalize_convergence_statistics(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Iterable):
        values = tuple(value)
    else:
        values = ()
    normalized = tuple(str(item).strip() for item in values)
    unsupported = sorted(set(normalized) - set(CONVERGENCE_STATISTICS))
    if unsupported or not normalized:
        allowed = ", ".join(CONVERGENCE_STATISTICS)
        raise ValueError(f"convergence_statistics must use supported statistics: {allowed}.")
    return normalized
