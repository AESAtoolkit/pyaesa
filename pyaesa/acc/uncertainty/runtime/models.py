"""Typed contracts for aCC uncertainty."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ACCUncertaintyRunPaths:
    """Canonical paths for one aCC Monte Carlo run."""

    run_root: Path
    public_row_identity: Path
    public_runs: Path
    summary_stats_runs: Path
    results_readme: Path
    source_methods: Path
    sobol_indices: Path
    sobol_source_summary: Path
    sobol_readme: Path
    scope_manifest: Path


@dataclass(frozen=True)
class ACCBranchPlan:
    """Vectorized evaluation plan for one aCC carrying capacity branch."""

    identity: pd.DataFrame
    asocc_positions: np.ndarray
    cc_positions: np.ndarray | None
    static_cc_values: np.ndarray | None
    dynamic_cc_factors: np.ndarray | None
    cc_type: str
    cc_source: str


@dataclass(frozen=True)
class ACCDynamicCCInput:
    """Dynamic carrying capacity rows used by aCC uncertainty."""

    identity: pd.DataFrame | None
    deterministic_values: np.ndarray | None
    manifest: Any | None
    deterministic_manifest_path: Path | None
    reuse_status: str
    process_ar6: dict[str, object] | None = None


@dataclass(frozen=True)
class ACCAsoccInput:
    """aSoCC rows used by aCC uncertainty."""

    identity: pd.DataFrame | None
    deterministic_values: np.ndarray | None
    manifest: Any | None
    deterministic_manifest_path: Path | None
    reuse_status: str


@dataclass(frozen=True)
class ACCUncertaintyPlan:
    """Complete vectorized aCC uncertainty plan."""

    identity: pd.DataFrame
    summary_identity: pd.DataFrame
    summary_public_row_groups: tuple[tuple[str, ...], ...]
    branch_plans: tuple[ACCBranchPlan, ...]
    asocc_input: ACCAsoccInput
    dynamic_cc_input: ACCDynamicCCInput | None
    acc_run_layout: str
    deterministic_cc_values: np.ndarray | None
    source_method_rows: pd.DataFrame
    active_sources: tuple[str, ...]
    dynamic_category_uncertainty_active: bool = False
