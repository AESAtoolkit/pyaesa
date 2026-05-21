"""Path ownership for AR6 CC uncertainty outputs."""

from pathlib import Path

from pyaesa.shared.runtime.metadata.contracts import SCOPE_MANIFEST_FILENAME
from pyaesa.shared.figures.paths import figures_root_for_run
from pyaesa.shared.uncertainty_assessment.io.formats import suffix_for_uncertainty_output

from pyaesa.ar6_cc.uncertainty.runtime.models import AR6CCUncertaintyRunPaths


def ar6_cc_monte_carlo_root(*, deterministic_manifest_path: Path) -> Path:
    """Return the AR6 CC Monte Carlo root beside deterministic outputs."""
    deterministic_root = Path(deterministic_manifest_path).parent.parent
    return deterministic_root.parent / "monte_carlo"


def ar6_cc_uncertainty_figures_root(*, paths: AR6CCUncertaintyRunPaths) -> Path:
    """Return the canonical figure root for one AR6 CC uncertainty run."""
    return figures_root_for_run(run_root=paths.run_root)


def build_ar6_cc_uncertainty_run_paths(
    *,
    deterministic_manifest_path: Path,
    run_id: str,
    output_format: str,
) -> AR6CCUncertaintyRunPaths:
    """Return canonical paths for one AR6 CC uncertainty run."""
    suffix = suffix_for_uncertainty_output(output_format)
    run_root = ar6_cc_monte_carlo_root(
        deterministic_manifest_path=deterministic_manifest_path
    ) / str(run_id)
    return AR6CCUncertaintyRunPaths(
        run_root=run_root,
        public_row_identity=run_root / "results" / f"public_row_identity{suffix}",
        public_runs=run_root / "results" / f"cc_runs{suffix}",
        summary_stats_runs=run_root / "results" / f"summary_stats_runs{suffix}",
        post_study_public_row_identity=(
            run_root / "results" / f"post_study_period_public_row_identity{suffix}"
        ),
        post_study_public_runs=run_root / "results" / f"post_study_period_cc_runs{suffix}",
        post_study_summary_stats_runs=(
            run_root / "results" / f"post_study_period_summary_stats_runs{suffix}"
        ),
        budget_row_identity=(
            run_root / "results" / f"study_and_post_study_period_budget_row_identity{suffix}"
        ),
        budget_runs=run_root / "results" / f"study_and_post_study_period_budget_runs{suffix}",
        budget_summary_stats_runs=(
            run_root / "results" / f"study_and_post_study_period_budget_summary_stats{suffix}"
        ),
        results_readme=run_root / "results" / "README.txt",
        source_methods=run_root / "logs" / "source_methods.csv",
        scope_manifest=run_root / "logs" / SCOPE_MANIFEST_FILENAME,
    )
