"""aSoCC Monte Carlo output path ownership."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyaesa.asocc.inter_method_tools.tree import (
    inter_method_tree_version_name,
    inter_method_tree_csv_name,
    inter_method_tree_figure_stem,
)

from pyaesa.shared.runtime.metadata.contracts import SCOPE_MANIFEST_FILENAME
from pyaesa.shared.uncertainty_assessment.io.formats import suffix_for_uncertainty_output


@dataclass(frozen=True)
class AsoccUncertaintyRunPaths:
    """Canonical on disk paths for one aSoCC Monte Carlo run."""

    run_root: Path
    public_row_identity: Path
    public_runs: Path
    summary_stats_runs: Path
    results_readme: Path
    source_methods: Path
    inter_method_tree_csv: Path
    inter_method_tree_figure_base: Path
    sobol_indices: Path
    sobol_source_summary: Path
    sobol_readme: Path
    scope_manifest: Path


def build_asocc_uncertainty_run_paths(
    *,
    deterministic_manifest_path: Path,
    run_id: str,
    output_format: str,
    inter_method_parameters: dict[str, Any] | None = None,
) -> AsoccUncertaintyRunPaths:
    """Return canonical paths for one package allocated Monte Carlo run."""
    suffix = suffix_for_uncertainty_output(output_format)
    run_root = asocc_monte_carlo_root(
        deterministic_manifest_path=deterministic_manifest_path
    ) / str(run_id)
    sobol_root = run_root / "results" / "sobol"
    inter_method_version_name = _inter_method_version_name(parameters=inter_method_parameters)
    return AsoccUncertaintyRunPaths(
        run_root=run_root,
        public_row_identity=run_root / "results" / f"public_row_identity{suffix}",
        public_runs=run_root / "results" / f"asocc_runs{suffix}",
        summary_stats_runs=run_root / "results" / f"summary_stats_runs{suffix}",
        results_readme=run_root / "results" / "README.txt",
        source_methods=run_root / "logs" / "source_methods.csv",
        inter_method_tree_csv=run_root
        / "figures"
        / "inter_method_tree"
        / inter_method_tree_csv_name(version_name=inter_method_version_name),
        inter_method_tree_figure_base=run_root
        / "figures"
        / "inter_method_tree"
        / inter_method_tree_figure_stem(version_name=inter_method_version_name),
        sobol_indices=sobol_root / f"sobol_indices{suffix}",
        sobol_source_summary=sobol_root / f"sobol_source_summary{suffix}",
        sobol_readme=sobol_root / "README_sobol.txt",
        scope_manifest=run_root / "logs" / SCOPE_MANIFEST_FILENAME,
    )


def asocc_monte_carlo_root(*, deterministic_manifest_path: Path) -> Path:
    """Return the aSoCC Monte Carlo root beside deterministic result folders."""
    deterministic_root = Path(deterministic_manifest_path).parent.parent
    return deterministic_root.parent / "monte_carlo"


def asocc_uncertainty_figures_root(*, paths: AsoccUncertaintyRunPaths) -> Path:
    """Return the pyaesa owned uncertainty figure root for one aSoCC run."""
    return Path(paths.run_root) / "figures"


def _inter_method_version_name(*, parameters: dict[str, Any] | None) -> str:
    return inter_method_tree_version_name(parameters=parameters)
