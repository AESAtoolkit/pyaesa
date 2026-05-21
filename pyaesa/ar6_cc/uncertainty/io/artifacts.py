"""Persisted AR6 CC uncertainty artifact contracts."""

from pathlib import Path
from typing import Any, cast

from pyaesa.ar6_cc.uncertainty.runtime.models import AR6CCUncertaintyRunPaths
from pyaesa.shared.uncertainty_assessment.io.formats import suffix_for_uncertainty_output
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def ar6_cc_run_paths_from_manifest(*, manifest: UncertaintyManifest) -> AR6CCUncertaintyRunPaths:
    """Return typed AR6 CC uncertainty artifact paths from a completed manifest."""
    artifacts = cast(dict[str, Any], manifest.artifacts)
    run_root = Path(artifacts["scope_manifest"]).parents[1]
    suffix = suffix_for_uncertainty_output(str(manifest.output_format))
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
        scope_manifest=Path(artifacts["scope_manifest"]),
    )


def ar6_cc_run_layout_from_manifest(*, manifest: UncertaintyManifest) -> str:
    """Return the persisted AR6 CC run table layout from a completed manifest."""
    public = cast(dict[str, Any], manifest.artifacts["public_output"])
    return str(cast(dict[str, Any], public["cc_runs"])["layout"])
