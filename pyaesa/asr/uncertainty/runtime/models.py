"""Typed contracts for ASR uncertainty."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LCARunValueProvider = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ASRUncertaintyRunPaths:
    """Canonical paths for one ASR Monte Carlo run."""

    run_root: Path
    public_row_identity: Path
    public_runs: Path
    summary_stats_runs: Path
    cumulative_row_identity: Path
    cumulative_runs: Path
    cumulative_summary_stats_runs: Path
    results_readme: Path
    source_methods: Path
    sobol_indices: Path
    sobol_source_summary: Path
    sobol_readme: Path
    scope_manifest: Path


@dataclass(frozen=True)
class LCAUncertaintyInput:
    """Resolved LCA numerator values used by ASR uncertainty."""

    identity: pd.DataFrame
    fixed_values: np.ndarray | None
    manifest: Any | None
    external_inputs: tuple[dict[str, Any], ...]
    source_method_rows: pd.DataFrame
    active_sources: tuple[str, ...]
    lca_type: str
    run_values_for_runs: LCARunValueProvider | None = None
    run_values_for_units: LCARunValueProvider | None = None
    run_inventory_size: int | None = None
    phase_function: str = "external_lca"
    phase_reuse_status: str = "computed"
    phase_output_root: Path | None = None


@dataclass(frozen=True)
class ASRUncertaintyPlan:
    """Complete vectorized ASR uncertainty plan."""

    identity: pd.DataFrame
    summary_identity: pd.DataFrame
    summary_public_row_groups: tuple[tuple[str, ...], ...]
    cumulative_identity: pd.DataFrame
    cumulative_summary_identity: pd.DataFrame
    cumulative_summary_public_row_groups: tuple[tuple[str, ...], ...]
    cumulative_public_row_groups: tuple[tuple[str, ...], ...]
    acc_positions: np.ndarray
    lca_positions: np.ndarray
    lca_unit_factors: np.ndarray
    acc_manifest: Any
    lca_input: LCAUncertaintyInput
    asr_run_layout: str
    source_method_rows: pd.DataFrame
    active_sources: tuple[str, ...]
    acc_position_order: np.ndarray = field(init=False)
    acc_positions_sorted: np.ndarray = field(init=False)
    cumulative_member_public_row_id: np.ndarray = field(init=False)
    cumulative_member_group_id: np.ndarray = field(init=False)
    cumulative_member_public_row_id_sorted: np.ndarray = field(init=False)
    cumulative_member_group_id_sorted: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Build immutable lookup arrays reused by ASR batch evaluation."""
        acc_order = np.argsort(self.acc_positions, kind="mergesort")
        member_public_row_id, member_group_id = _cumulative_membership_arrays(
            public_row_groups=self.cumulative_public_row_groups
        )
        member_order = np.argsort(member_public_row_id, kind="mergesort")
        object.__setattr__(self, "acc_position_order", acc_order)
        object.__setattr__(self, "acc_positions_sorted", self.acc_positions[acc_order])
        object.__setattr__(self, "cumulative_member_public_row_id", member_public_row_id)
        object.__setattr__(self, "cumulative_member_group_id", member_group_id)
        object.__setattr__(
            self,
            "cumulative_member_public_row_id_sorted",
            member_public_row_id[member_order],
        )
        object.__setattr__(
            self,
            "cumulative_member_group_id_sorted",
            member_group_id[member_order],
        )

    @property
    def has_cumulative_outputs(self) -> bool:
        """Return whether this plan owns dynamic period cumulative ASR outputs."""
        return not self.cumulative_identity.empty


def _cumulative_membership_arrays(
    *,
    public_row_groups: tuple[tuple[str, ...], ...],
) -> tuple[np.ndarray, np.ndarray]:
    public_row_ids: list[int] = []
    group_ids: list[int] = []
    for index, group in enumerate(public_row_groups):
        public_row_ids.extend(int(public_row_id) for public_row_id in group)
        group_ids.extend([index] * len(group))
    return (
        np.asarray(public_row_ids, dtype=np.int64),
        np.asarray(group_ids, dtype=np.int64),
    )
