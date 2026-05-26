"""Path ownership for aCC uncertainty outputs."""

from pathlib import Path

from pyaesa.acc.shared.runtime.paths import get_acc_root
from pyaesa.shared.acc_asr_common.branches.config import cc_branch_token
from pyaesa.shared.runtime.metadata.contracts import SCOPE_MANIFEST_FILENAME
from pyaesa.shared.uncertainty_assessment.io.formats import suffix_for_uncertainty_output

from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyRunPaths


def acc_monte_carlo_root(
    *,
    proj_base: Path,
    source_label: str,
    agg_version: str | None,
) -> Path:
    """Return the aCC Monte Carlo root for one source branch."""
    return (
        get_acc_root(
            proj_base=proj_base,
            source_label=source_label,
            agg_version=agg_version,
        )
        / "monte_carlo"
    )


def acc_monte_carlo_branch_root(
    *,
    monte_carlo_root: Path,
    cc_source: str,
    cc_type: str,
) -> Path:
    """Return the aCC Monte Carlo root for one carrying capacity branch."""
    return Path(monte_carlo_root) / cc_branch_token(cc_source=cc_source, cc_type=cc_type)


def build_acc_uncertainty_run_paths(
    *,
    monte_carlo_root: Path,
    run_id: str,
    output_format: str,
) -> ACCUncertaintyRunPaths:
    """Return canonical paths for one aCC uncertainty run."""
    suffix = suffix_for_uncertainty_output(output_format)
    run_root = Path(monte_carlo_root) / str(run_id)
    sobol_root = run_root / "results" / "sobol"
    return ACCUncertaintyRunPaths(
        run_root=run_root,
        public_row_identity=run_root / "results" / f"public_row_identity{suffix}",
        public_runs=run_root / "results" / f"acc_runs{suffix}",
        summary_stats_runs=run_root / "results" / f"summary_stats_runs{suffix}",
        results_readme=run_root / "results" / "README.txt",
        source_methods=run_root / "logs" / "source_methods.csv",
        sobol_indices=sobol_root / f"sobol_indices{suffix}",
        sobol_source_summary=sobol_root / f"sobol_source_summary{suffix}",
        sobol_readme=sobol_root / "README_sobol.txt",
        scope_manifest=run_root / "logs" / SCOPE_MANIFEST_FILENAME,
    )


def acc_uncertainty_figures_root(*, paths: ACCUncertaintyRunPaths) -> Path:
    """Return the figure root for one aCC uncertainty run."""
    return Path(paths.run_root) / "figures"
