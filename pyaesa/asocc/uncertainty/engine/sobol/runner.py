"""aSoCC Sobol variance decomposition on compact source owners."""

from dataclasses import dataclass

from pyaesa.asocc.uncertainty.engine.sobol.evaluator import (
    asocc_sobol_base_chunk_rows,
    build_asocc_sobol_evaluation_context,
    evaluate_asocc_sobol_chunk,
)
from pyaesa.asocc.uncertainty.engine.sobol.reporting import (
    asocc_sobol_method_payload,
    write_asocc_sobol_readme,
)
from pyaesa.asocc.uncertainty.engine.sobol.scope import selected_sobol_years
from pyaesa.asocc.uncertainty.engine.sobol.summary import asocc_sobol_source_summary
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.asocc.uncertainty.sources.inter_mrio import InterMrioPlan
from pyaesa.shared.uncertainty_assessment.sobol.plan import (
    SobolPlan,
    sobol_plan_payload,
)
from pyaesa.shared.uncertainty_assessment.sobol.runner import run_sobol_analysis
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan
from pyaesa.shared.uncertainty_assessment.io.tables import write_uncertainty_table
from pyaesa.shared.runtime.reporting.status import StatusSink


@dataclass(frozen=True)
class SobolRunResult:
    """Completed Sobol status for one aSoCC run."""

    ran: bool
    status: dict[str, object] | None


def run_asocc_sobol(
    *,
    paths: AsoccUncertaintyRunPaths,
    loaded,
    inter_mrio_plan: InterMrioPlan | None,
    runtime,
    sources: SourceActivationPlan,
    external_plan,
    sobol_plan: SobolPlan,
    status: StatusSink | None = None,
) -> SobolRunResult:
    """Run optional Sobol variance decomposition for aSoCC public rows."""
    if not sobol_plan.enabled:
        return SobolRunResult(ran=False, status=None)
    selected_years = selected_sobol_years(
        plan=sobol_plan,
        requested_years=tuple(int(year) for year in loaded.requested_years),
    )
    context = build_asocc_sobol_evaluation_context(
        loaded=loaded,
        inter_mrio_plan=inter_mrio_plan,
        sources=sources,
        external_plan=external_plan,
        sobol_plan=sobol_plan,
        selected_years=selected_years,
    )
    if len(context.source_names) < 2:
        reason = (
            "Sobol variance decomposition was not run because at least two "
            "uncertainty source dimensions are required to decompose variance "
            "between sources."
        )
        return SobolRunResult(
            ran=False,
            status={
                "mode": sobol_plan.mode,
                "ran": False,
                "reason": reason,
                "active_source_count": len(context.source_names),
                "parameters": sobol_plan_payload(plan=sobol_plan),
            },
        )
    result = run_sobol_analysis(
        plan=sobol_plan,
        dimension_names=context.source_names,
        evaluate=lambda chunk: evaluate_asocc_sobol_chunk(context=context, chunk=chunk),
        max_base_chunk_rows=asocc_sobol_base_chunk_rows(
            context=context,
            dimension_count=len(context.source_names),
        ),
        source_summary_builder=lambda identity, dimensions, estimates: asocc_sobol_source_summary(
            identity=identity,
            dimension_names=dimensions,
            estimates=estimates,
            confidence_level=sobol_plan.confidence_level,
            requested_ssp_scenarios=context.requested_ssp_scenarios,
        ),
        progress_source="uncertainty_asocc",
        status=status,
    )
    write_uncertainty_table(
        path=paths.sobol_indices,
        frame=result.indices,
        output_format=runtime.output_format,
    )
    write_uncertainty_table(
        path=paths.sobol_source_summary,
        frame=result.source_summary,
        output_format=runtime.output_format,
    )
    write_asocc_sobol_readme(
        path=paths.sobol_readme,
        output_format=runtime.output_format,
        source_names=context.source_names,
        selected_years=context.selected_years,
        plan=sobol_plan,
    )
    sobol_status = dict(result.status)
    sobol_status["ran"] = True
    sobol_status["parameters"] = sobol_plan_payload(plan=sobol_plan)
    sobol_status["selected_output_years"] = list(context.selected_years)
    sobol_status["method"] = asocc_sobol_method_payload(
        source_names=context.source_names,
        plan=sobol_plan,
        selected_years=context.selected_years,
    )
    return SobolRunResult(ran=True, status=sobol_status)
