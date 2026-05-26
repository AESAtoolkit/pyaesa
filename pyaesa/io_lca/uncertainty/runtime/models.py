"""Typed contracts for IO-LCA uncertainty."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec
from pyaesa.io_lca.data.paths import IOLCAPaths


@dataclass(frozen=True)
class IOLCAUncertaintyRequest:
    """Normalized public IO-LCA uncertainty request."""

    base_io_lca_args: dict[str, Any]
    deterministic_args: dict[str, Any]
    source_parameters: dict[str, Any]
    project_name: str
    source: str
    agg_reg: bool
    agg_sec: bool
    agg_version: str | None
    years: list[int]
    lcia_methods: list[str]
    fu_spec: IOLCAFUSpec
    filters: dict[str, list[str] | None]
    group_indices: bool


@dataclass(frozen=True)
class IOLCADeterministicScope:
    """Resolved deterministic IO-LCA main result scope."""

    paths: IOLCAPaths
    source: str
    metadata_path: Path
    scope_key: str
    output_format: str
    completed_years_by_method: dict[str, tuple[int, ...]]
    deterministic_paths: tuple[str, ...]
    reuse_status: str


@dataclass(frozen=True)
class IOLCAUncertaintyRunPaths:
    """Canonical paths for one IO-LCA Monte Carlo run."""

    run_root: Path
    public_row_identity: Path
    public_runs: Path
    summary_stats_runs: Path
    results_readme: Path
    source_methods: Path
    scope_manifest: Path


@dataclass(frozen=True)
class IOLCAUncertaintyPlan:
    """Compact IO-LCA LCIA uncertainty source plan."""

    identity: pd.DataFrame
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    unique_shared_u_keys: np.ndarray
    shared_u_inverse: np.ndarray
    source_method_rows: pd.DataFrame
