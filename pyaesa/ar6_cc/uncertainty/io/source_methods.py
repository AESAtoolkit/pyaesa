"""AR6 CC uncertainty source method and README writers."""

from pathlib import Path

import pandas as pd

from pyaesa.ar6_cc.uncertainty.runtime.models import (
    AR6CCUncertaintyRequest,
    AR6CCUncertaintyRunPaths,
)
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent, write_via_atomic_temp
from pyaesa.shared.runtime.text import join_user_text_lines
from pyaesa.shared.uncertainty_assessment.io.run_artifacts import public_run_artifact_readme_lines


def write_ar6_cc_source_methods(*, path: Path, rows: pd.DataFrame) -> None:
    """Write the AR6 CC source method inventory."""
    ordered = rows.sort_values(
        ["cc_category", "ssp_scenario", "impact_unit", "cc_model", "cc_scenario", "cc_flow"],
        kind="mergesort",
    ).reset_index(drop=True)
    write_via_atomic_temp(
        ensure_file_parent(path),
        writer=lambda tmp_path: ordered.to_csv(tmp_path, index=False),
    )


def write_ar6_cc_results_readme(
    *,
    paths: AR6CCUncertaintyRunPaths,
    request: AR6CCUncertaintyRequest,
    availability_messages: tuple[str, ...],
) -> None:
    """Write the public AR6 CC Monte Carlo result guide."""
    text = _readme_text(request=request, availability_messages=availability_messages)

    def _write_text(tmp_path: Path) -> None:
        tmp_path.write_text(text, encoding="utf-8")

    write_via_atomic_temp(
        ensure_file_parent(paths.results_readme),
        writer=_write_text,
    )


def _readme_text(
    *,
    request: AR6CCUncertaintyRequest,
    availability_messages: tuple[str, ...],
) -> str:
    source = request.source_parameters
    category_mode = "active" if bool(source["category_uncertainty"]) else "inactive"
    lines = [
        "AR6 CC Uncertainty Results",
        "",
        "This run evaluates Monte Carlo uncertainty in dynamic AR6 carrying",
        "capacity trajectories by sampling retained deterministic AR6",
        "model-scenario pathways.",
        "",
        "Artifacts",
        "- public_row_identity: public AR6 CC selected trajectory rows.",
        *public_run_artifact_readme_lines(run_name="cc_runs"),
        "  Layout: sparse selected rows with run_index, public_row_id, and cc.",
        "- summary_stats_runs: exact summary statistics computed from all runs.",
        "- post_study_period_public_row_identity, post_study_period_cc_runs,",
        "  and post_study_period_summary_stats_runs: same layout for years",
        "  after the study period when the deterministic scope contains them.",
        "- study_and_post_study_period_budget_row_identity,",
        "  study_and_post_study_period_budget_runs, and",
        "  study_and_post_study_period_budget_summary_stats: exact cumulative",
        "  budget runs and summaries for figure budget panels.",
        "  Additional run artifacts use the same interval index contract.",
        "- source_methods.csv: retained AR6 trajectory candidates and probabilities.",
        "- scope_manifest.json: request, prerequisite, output, reuse metadata,",
        "  and canonical public table schemas for this result scope.",
        "",
        "Public Row Identity",
        "Each public row is identified by cc_category, ssp_scenario,",
        "cc_flow, cc_variable, impact_unit, cc_model, cc_scenario, and year.",
        "cc_runs stores only the trajectory rows selected in each Monte Carlo run. Join",
        "cc_runs.public_row_id to public_row_identity.public_row_id to read",
        "the selected category, model, scenario, SSP, impact unit, and year.",
        "",
        "Sampling Method",
        f"- sampling_method: {source['sampling_method']}",
        f"- category_uncertainty: {category_mode}",
        "",
        "For sampling_method='srs', each retained deterministic model-scenario",
        "trajectory in a category and SSP pool has probability 1 / n_pool.",
        "",
        "For sampling_method='lhs': first an AR6 model is rendered uniformly from the",
        "pool, then one retained scenario for that model is rendered uniformly.",
        "The probability for a model-scenario row is therefore",
        "1 / (n_models_in_pool * n_scenarios_for_that_model).",
        "",
        "When category_uncertainty is active, the run runs one retained",
        "category independently inside each retained SSP pool. Category",
        "probabilities are therefore conditional on the categories available",
        "in that pool.",
        "",
        "Summary Statistics",
        "summary_stats_runs groups out cc_model and cc_scenario because they",
        "are sampled trajectory states. When category_uncertainty is active,",
        "summary_stats_runs also groups out cc_category so it reports the",
        "integrated category uncertainty distribution by ssp_scenario,",
        "cc_flow, cc_variable, impact_unit, and year.",
        "",
        "Scope",
        f"- years: {min(request.years)} to {max(request.years)}",
        f"- category selector: {_selector_text(request.category)}",
        f"- ssp_scenario selector: {_selector_text(request.ssp_scenario)}",
        f"- emission_type: {request.emission_type}",
        f"- include_afolu: {request.include_afolu}",
        f"- emissions_mode: {request.emissions_mode}",
        f"- subset_version: {request.subset_version or 'none'}",
        "",
        "Availability Notes",
        *_availability_lines(availability_messages),
        "",
    ]
    return join_user_text_lines(lines)


def _selector_text(values: list[str]) -> str:
    return ", ".join(values)


def _availability_lines(messages: tuple[str, ...]) -> list[str]:
    if not messages:
        return ["- All requested category and SSP selectors retained values."]
    return [f"- {message}" for message in messages]
