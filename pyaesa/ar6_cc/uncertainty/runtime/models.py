"""Typed contracts for AR6 CC uncertainty."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AR6CCUncertaintyRequest:
    """Normalized public AR6 CC uncertainty request."""

    base_ar6_cc_args: dict[str, Any]
    deterministic_args: dict[str, Any]
    source_parameters: dict[str, Any]
    study_period: list[int]
    years: list[int]
    harmonization: bool
    harmonization_method: str
    category: list[str]
    ssp_scenario: list[str]
    emission_type: str
    include_afolu: bool
    emissions_mode: str
    subset_version: str | None


@dataclass(frozen=True)
class AR6CCDeterministicScope:
    """Resolved deterministic AR6 CC prerequisite scope."""

    metadata_path: Path
    reuse_status: str
    output_file: Path
    post_study_output_file: Path | None
    output_format: str
    scope_key: str
    emission_type: str
    include_afolu: bool
    variable: str
    emissions_mode: str
    categories: tuple[str, ...]
    ssp_scenarios: tuple[str, ...]
    subset_version: str | None
    pathway_counts: tuple[dict[str, object], ...]
    missing_pathway_combinations: tuple[dict[str, object], ...]
    process_ar6: dict[str, object]


@dataclass(frozen=True)
class AR6CCUncertaintyRunPaths:
    """Canonical paths for one AR6 CC Monte Carlo run."""

    run_root: Path
    public_row_identity: Path
    public_runs: Path
    summary_stats_runs: Path
    post_study_public_row_identity: Path
    post_study_public_runs: Path
    post_study_summary_stats_runs: Path
    budget_row_identity: Path
    budget_runs: Path
    budget_summary_stats_runs: Path
    results_readme: Path
    source_methods: Path
    scope_manifest: Path


@dataclass(frozen=True)
class AR6CCSamplingGroup:
    """One candidate trajectory pool and its output matrix column span."""

    category: str
    ssp_scenario: str
    flow_count: int
    candidate_positions: np.ndarray
    model_candidate_positions: tuple[np.ndarray, ...]
    output_start: int
    output_stop: int


@dataclass(frozen=True)
class AR6CCCategoryPool:
    """Category run candidates for one retained SSP pool."""

    ssp_scenario: str
    group_indices: tuple[int, ...]


@dataclass(frozen=True)
class AR6CCUncertaintyPlan:
    """AR6 CC uncertainty source plan."""

    identity: pd.DataFrame
    group_identity: pd.DataFrame
    trajectory_values: np.ndarray
    groups: tuple[AR6CCSamplingGroup, ...]
    category_pools: tuple[AR6CCCategoryPool, ...]
    source_method_rows: pd.DataFrame
    source_parameters: dict[str, Any]
    availability_messages: tuple[str, ...]
