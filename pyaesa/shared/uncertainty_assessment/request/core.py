"""Family-neutral uncertainty public request normalization."""

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

from pyaesa.shared.uncertainty_assessment.io.formats import normalize_uncertainty_output_format

DEFAULT_UNCERTAINTY_BATCH_SIZE = 10_000
MAX_UNCERTAINTY_BATCH_SIZE = 10_000
UNCERTAINTY_BATCH_MEMORY_BYTES = 3_000_000_000
UNCERTAINTY_BATCH_TEMPORARY_FACTOR = 24
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
    """One run scoped memory block used by the batch planner."""

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
        raise ValueError("Exactly one Monte Carlo mode block must be active.")
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
    batch_size = min(
        DEFAULT_UNCERTAINTY_BATCH_SIZE,
        stable_runs if mode == "convergence" else n_runs,
        n_runs,
    )
    return UncertaintyRuntimeRequest(
        family=_non_empty_text(family, field="family"),
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
    row_count: int,
    extra_blocks: Iterable[BatchMemoryBlock] = (),
) -> int:
    """Return an internal run batch size bounded by the memory target."""
    bytes_per_run = _block_bytes_per_run(
        BatchMemoryBlock(
            name="primary_values",
            row_count=int(row_count),
            array_count=UNCERTAINTY_BATCH_TEMPORARY_FACTOR,
        )
    )
    bytes_per_run += sum(_block_bytes_per_run(block) for block in extra_blocks)
    if bytes_per_run <= 0:
        raise ValueError("At least one positive batch memory row count is required.")
    memory_batch_size = max(1, UNCERTAINTY_BATCH_MEMORY_BYTES // bytes_per_run)
    return min(
        int(runtime.n_runs),
        int(runtime.batch_size),
        MAX_UNCERTAINTY_BATCH_SIZE,
        int(memory_batch_size),
    )


def _block_bytes_per_run(block: BatchMemoryBlock) -> int:
    if int(block.row_count) <= 0:
        return 0
    if int(block.array_count) <= 0:
        raise ValueError(f"Batch memory block '{block.name}' must use a positive array count.")
    if int(block.dtype_bytes) <= 0:
        raise ValueError(f"Batch memory block '{block.name}' must use a positive dtype size.")
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
        raise ValueError(f"Unsupported Monte Carlo parameter keys: {sorted(params)}.")
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
        raise ValueError(f"Unsupported Monte Carlo parameter keys: {unsupported}.")
    active = block["active"]
    if not isinstance(active, bool):
        raise ValueError(f"mc_parameters.{mode}.active must be a boolean.")
    return block


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer.")
    parsed = int(value)
    if parsed <= 0:
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
        raise ValueError("convergence_statistics must use mean only.")
    return normalized


def _non_empty_text(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    return text
