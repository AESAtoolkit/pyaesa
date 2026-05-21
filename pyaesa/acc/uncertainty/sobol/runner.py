"""aCC Sobol variance decomposition."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyRunPaths
from pyaesa.acc.uncertainty.evaluation.source_unit_evaluator import (
    ACCSobolEvaluationContext,
    build_acc_sobol_evaluation_context,
    evaluate_acc_sobol_units,
)
from pyaesa.ar6_cc.uncertainty.request.normalization import AR6_DYNAMIC_CC_SOURCE
from pyaesa.acc.uncertainty.sobol.summary import acc_sobol_source_summary
from pyaesa.shared.uncertainty_assessment.sobol.plan import (
    SobolPlan,
    sobol_plan_payload,
)
from pyaesa.shared.uncertainty_assessment.sobol.design import SobolEvaluationChunk
from pyaesa.shared.uncertainty_assessment.sobol.reporting import (
    sobol_method_payload,
    write_sobol_readme,
)
from pyaesa.shared.uncertainty_assessment.sobol.runner import (
    EvaluatedSobolChunk,
    run_sobol_analysis,
)
from pyaesa.shared.uncertainty_assessment.io.tables import write_uncertainty_table
from pyaesa.shared.runtime.reporting.status import StatusSink


@dataclass(frozen=True)
class ACCSobolRunResult:
    """Completed Sobol status for one aCC run."""

    ran: bool
    status: dict[str, object] | None


def run_acc_sobol(
    *,
    paths: ACCUncertaintyRunPaths,
    runtime,
    branches: list[dict[str, Any]],
    base_asocc_args: dict[str, Any],
    asocc_uncertainty_config: dict[str, Any],
    external_method: dict[str, Any] | None,
    dynamic_cc_config: dict[str, Any] | None,
    full_years: int | list[int] | range,
    sobol_plan: SobolPlan,
    status: StatusSink | None = None,
) -> ACCSobolRunResult:
    """Run optional Sobol variance decomposition for aCC public rows."""
    if not sobol_plan.enabled:
        return ACCSobolRunResult(ran=False, status=None)
    context = build_acc_sobol_evaluation_context(
        branches=branches,
        base_asocc_args=base_asocc_args,
        asocc_uncertainty_config=asocc_uncertainty_config,
        external_method=external_method,
        dynamic_cc_config=dynamic_cc_config,
        full_years=full_years,
        sobol_plan=sobol_plan,
    )
    if len(context.source_names) < 2:
        reason = (
            "Sobol variance decomposition was not run because at least two "
            "uncertainty source dimensions are required to decompose variance "
            "between sources."
        )
        return ACCSobolRunResult(
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
        evaluate=lambda chunk: _evaluate_chunk(context=context, chunk=chunk),
        source_summary_builder=lambda identity, dimensions, estimates: acc_sobol_source_summary(
            identity=identity,
            dimension_names=dimensions,
            estimates=estimates,
            confidence_level=sobol_plan.confidence_level,
            requested_ssp_scenarios=context.asocc_context.requested_ssp_scenarios,
        ),
        progress_source="uncertainty_acc",
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
    write_sobol_readme(
        path=paths.sobol_readme,
        output_format=runtime.output_format,
        family_label="aCC",
        source_names=context.source_names,
        selected_scope_line=(
            "Sobol outputs are evaluated only for the selected aCC output years: "
            + ", ".join(str(year) for year in context.asocc_context.selected_years)
            + "."
        ),
        plan=sobol_plan,
        source_summary_notes=(
            "Static carrying capacity bounds are deterministic row selectors, not "
            "uncertainty sources.",
            "Dynamic AR6 CC contributes one source dimension when dynamic aCC is "
            f"requested and {AR6_DYNAMIC_CC_SOURCE} is active.",
        ),
        indices_notes=(
            "aCC row identity is built from the matched aSoCC public row and the "
            "carrying capacity row: static impact and bound, or dynamic AR6 "
            "category, SSP, model, scenario, and year.",
        ),
        method_notes=(
            "aCC Sobol evaluates aCC = aSoCC * CC directly for each Saltelli design row.",
        ),
    )
    sobol_status = dict(result.status)
    sobol_status["ran"] = True
    sobol_status["parameters"] = sobol_plan_payload(plan=sobol_plan)
    sobol_status["selected_output_years"] = list(context.asocc_context.selected_years)
    sobol_status["method"] = sobol_method_payload(
        source_names=context.source_names,
        plan=sobol_plan,
        selected_scope={"selected_output_years": list(context.asocc_context.selected_years)},
    )
    return ACCSobolRunResult(ran=True, status=sobol_status)


def _evaluate_chunk(
    *,
    context: ACCSobolEvaluationContext,
    chunk: SobolEvaluationChunk,
) -> EvaluatedSobolChunk:
    units = np.vstack((chunk.a, chunk.b, *chunk.ab))
    _, values = evaluate_acc_sobol_units(context=context, units=units)
    a_stop = chunk.a.shape[0]
    b_stop = a_stop + chunk.b.shape[0]
    ab_values = []
    start = b_stop
    for block in chunk.ab:
        stop = start + block.shape[0]
        ab_values.append(values[start:stop])
        start = stop
    return EvaluatedSobolChunk(
        identity=context.identity,
        a_values=values[:a_stop],
        b_values=values[a_stop:b_stop],
        ab_values=tuple(ab_values),
    )
