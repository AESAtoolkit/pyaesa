"""Path ownership for IO-LCA uncertainty outputs."""

from pathlib import Path

from pyaesa.shared.runtime.metadata.contracts import SCOPE_MANIFEST_FILENAME
from pyaesa.shared.uncertainty_assessment.io.formats import suffix_for_uncertainty_output

from pyaesa.io_lca.uncertainty.runtime.models import IOLCAUncertaintyRunPaths


def io_lca_monte_carlo_root(*, deterministic_manifest_path: Path) -> Path:
    """Return the IO-LCA Monte Carlo root beside deterministic result folders."""
    deterministic_root = Path(deterministic_manifest_path).parent.parent
    return deterministic_root.parent / "monte_carlo"


def build_io_lca_uncertainty_run_paths(
    *,
    deterministic_manifest_path: Path,
    run_id: str,
    output_format: str,
) -> IOLCAUncertaintyRunPaths:
    """Return canonical paths for one IO-LCA uncertainty run."""
    suffix = suffix_for_uncertainty_output(output_format)
    run_root = io_lca_monte_carlo_root(
        deterministic_manifest_path=deterministic_manifest_path
    ) / str(run_id)
    return IOLCAUncertaintyRunPaths(
        run_root=run_root,
        public_row_identity=run_root / "results" / f"public_row_identity{suffix}",
        public_runs=run_root / "results" / f"lca_runs{suffix}",
        summary_stats_runs=run_root / "results" / f"summary_stats_runs{suffix}",
        results_readme=run_root / "results" / "README.txt",
        source_methods=run_root / "logs" / "source_methods.csv",
        scope_manifest=run_root / "logs" / SCOPE_MANIFEST_FILENAME,
    )


def io_lca_uncertainty_figures_root(*, paths: IOLCAUncertaintyRunPaths) -> Path:
    """Return the canonical figure root for one IO-LCA uncertainty run."""
    return paths.run_root / "figures"
