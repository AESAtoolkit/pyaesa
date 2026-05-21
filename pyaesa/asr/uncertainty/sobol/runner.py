"""ASR Sobol variance decomposition."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.acc.uncertainty.request.normalization import (
    AR6_DYNAMIC_CC_SOURCE,
    asocc_uncertainty_config_for_acc,
    dynamic_cc_source_parameters,
    normalize_acc_uncertainty_config,
)
from pyaesa.acc.uncertainty.evaluation.source_unit_evaluator import (
    ACCSobolEvaluationContext,
    build_acc_sobol_evaluation_context,
    evaluate_acc_sobol_units,
)
from pyaesa.asr.uncertainty.sources.lca_inputs import lcia_uncertainty_source_active
from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyRunPaths, LCAUncertaintyInput
from pyaesa.asr.uncertainty.evaluation.alignment import build_asr_alignment
from pyaesa.asr.uncertainty.evaluation.runs import (
    evaluate_asr_value_matrix,
)
from pyaesa.asr.uncertainty.sources.source_keys import acc_source_name, io_lca_source_name
from pyaesa.io_lca.data.contracts import IO_LCA_FAMILY
from pyaesa.io_lca.uncertainty.runtime.prerequisites import (
    load_deterministic_public_rows,
    prepare_io_lca_deterministic_prerequisite,
)
from pyaesa.io_lca.uncertainty.request.normalization import normalize_io_lca_uncertainty_request
from pyaesa.io_lca.uncertainty.evaluation.sampling import build_io_lca_lcia_plan
from pyaesa.io_lca.uncertainty.sobol.evaluator import (
    IOLCASobolEvaluationContext,
    build_io_lca_sobol_evaluation_context,
    evaluate_io_lca_sobol_units,
)
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan, sobol_plan_payload
from pyaesa.shared.uncertainty_assessment.sobol.design import SobolEvaluationChunk
from pyaesa.shared.uncertainty_assessment.sobol.reporting import (
    sobol_method_payload,
    write_sobol_readme,
)
from pyaesa.shared.uncertainty_assessment.sobol.runner import (
    EvaluatedSobolChunk,
    run_sobol_analysis,
)
from pyaesa.shared.lcia.uncertainty_source import LCIA_SOURCE
from pyaesa.shared.uncertainty_assessment.io.tables import write_uncertainty_table
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter
from pyaesa.shared.runtime.reporting.status import StatusSink


@dataclass(frozen=True)
class ASRSobolRunResult:
    """Completed Sobol status for one ASR run."""

    ran: bool
    status: dict[str, object] | None


@dataclass(frozen=True)
class ASRSobolEvaluationContext:
    """Prepared ASR Sobol evaluator context."""

    identity: pd.DataFrame
    source_names: tuple[str, ...]
    acc_context: ACCSobolEvaluationContext
    lca_input: LCAUncertaintyInput
    lca_context: IOLCASobolEvaluationContext | None
    acc_positions: np.ndarray
    lca_positions: np.ndarray
    lca_unit_factors: np.ndarray


def run_asr_sobol(
    *,
    paths: ASRUncertaintyRunPaths,
    runtime,
    branches: list[dict[str, Any]],
    base_asocc_args: dict[str, Any],
    base_io_lca_args: dict[str, Any],
    acc_uncertainty_config: dict[str, Any],
    lca_uncertainty_config: dict[str, Any],
    external_method: dict[str, Any] | None,
    lca_input: LCAUncertaintyInput,
    full_years: int | list[int] | range,
    sobol_plan: SobolPlan,
    status: StatusSink | None = None,
) -> ASRSobolRunResult:
    """Run optional Sobol variance decomposition for ASR public rows."""
    if not sobol_plan.enabled:
        return ASRSobolRunResult(ran=False, status=None)
    context = build_asr_sobol_evaluation_context(
        branches=branches,
        base_asocc_args=base_asocc_args,
        base_io_lca_args=base_io_lca_args,
        acc_uncertainty_config=acc_uncertainty_config,
        lca_uncertainty_config=lca_uncertainty_config,
        external_method=external_method,
        lca_input=lca_input,
        full_years=full_years,
        sobol_plan=sobol_plan,
    )
    if len(context.source_names) < 2:
        reason = (
            "Sobol variance decomposition was not run because at least two "
            "uncertainty source dimensions are required to decompose variance "
            "between sources."
        )
        return ASRSobolRunResult(
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
        progress_source="uncertainty_asr",
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
        family_label="ASR",
        source_names=context.source_names,
        selected_scope_line="Sobol outputs are evaluated on the ASR matched public rows.",
        plan=sobol_plan,
        source_summary_notes=(
            "ACC sources include the nested aSoCC and dynamic AR6 CC sources selected "
            "by uncertainty_acc.",
            "LCA contributes one numerator source dimension when IO-LCA LCIA "
            "uncertainty is active or external LCA Monte Carlo rows are supplied.",
        ),
        indices_notes=(
            "ASR row identity is built from matched ACC denominator rows and LCA numerator rows.",
        ),
        method_notes=("ASR Sobol evaluates ASR = LCA / aCC directly for each Saltelli row.",),
    )
    sobol_status = dict(result.status)
    sobol_status["ran"] = True
    sobol_status["parameters"] = sobol_plan_payload(plan=sobol_plan)
    sobol_status["method"] = sobol_method_payload(
        source_names=context.source_names,
        plan=sobol_plan,
        selected_scope={},
    )
    return ASRSobolRunResult(ran=True, status=sobol_status)


def build_asr_sobol_evaluation_context(
    *,
    branches: list[dict[str, Any]],
    base_asocc_args: dict[str, Any],
    base_io_lca_args: dict[str, Any],
    acc_uncertainty_config: dict[str, Any],
    lca_uncertainty_config: dict[str, Any],
    external_method: dict[str, Any] | None,
    lca_input: LCAUncertaintyInput,
    full_years: int | list[int] | range,
    sobol_plan: SobolPlan,
) -> ASRSobolEvaluationContext:
    """Build a reusable ASR Sobol evaluator context."""
    acc_uncertainty_config = normalize_acc_uncertainty_config(dict(acc_uncertainty_config))
    acc_context = build_acc_sobol_evaluation_context(
        branches=branches,
        base_asocc_args=base_asocc_args,
        asocc_uncertainty_config=asocc_uncertainty_config_for_acc(acc_uncertainty_config),
        external_method=external_method,
        dynamic_cc_config=dynamic_cc_source_parameters(
            acc_uncertainty_config.get(AR6_DYNAMIC_CC_SOURCE)
        ),
        full_years=full_years,
        sobol_plan=sobol_plan,
    )
    lca_context = _io_lca_sobol_context(
        base_io_lca_args=base_io_lca_args,
        uncertainty_config=lca_uncertainty_config,
        lca_input=lca_input,
    )
    lca_identity = lca_context.plan.identity if lca_context is not None else lca_input.identity
    alignment = build_asr_alignment(
        acc_identity=acc_context.identity,
        lca_identity=lca_identity,
        lca_type=lca_input.lca_type,
    )
    return ASRSobolEvaluationContext(
        identity=alignment.identity,
        source_names=tuple(acc_source_name(name) for name in acc_context.source_names)
        + _lca_sobol_source_names(lca_input=lca_input, lca_context=lca_context),
        acc_context=acc_context,
        lca_input=lca_input,
        lca_context=lca_context,
        acc_positions=alignment.acc_positions,
        lca_positions=alignment.lca_positions,
        lca_unit_factors=alignment.lca_unit_factors,
    )


def _evaluate_chunk(
    *,
    context: ASRSobolEvaluationContext,
    chunk: SobolEvaluationChunk,
) -> EvaluatedSobolChunk:
    units = np.vstack((chunk.a, chunk.b, *chunk.ab))
    values = evaluate_asr_sobol_units(context=context, units=units)
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


def evaluate_asr_sobol_units(
    *,
    context: ASRSobolEvaluationContext,
    units: np.ndarray,
) -> np.ndarray:
    """Evaluate unit interval source values into ASR public row values."""
    acc_count = len(context.acc_context.source_names)
    _, acc_values = evaluate_acc_sobol_units(
        context=context.acc_context,
        units=units[:, :acc_count],
    )
    lca_values = _lca_values_for_units(
        context=context,
        units=units[:, acc_count:],
    )
    return evaluate_asr_value_matrix(
        acc_values=acc_values,
        lca_values=lca_values,
        acc_positions=context.acc_positions,
        lca_positions=context.lca_positions,
        lca_unit_factors=context.lca_unit_factors,
    )


def _lca_values_for_units(*, context: ASRSobolEvaluationContext, units: np.ndarray) -> np.ndarray:
    if context.lca_context is not None:
        _, values = evaluate_io_lca_sobol_units(context=context.lca_context, units=units[:, :1])
        return values
    if context.lca_input.run_values_for_units is not None:
        source_units = units[:, 0]
        return context.lca_input.run_values_for_units(source_units)
    fixed = cast(np.ndarray, context.lca_input.fixed_values)
    return np.broadcast_to(fixed, (units.shape[0], len(fixed)))


def _io_lca_sobol_context(
    *,
    base_io_lca_args: dict[str, Any],
    uncertainty_config: dict[str, Any],
    lca_input: LCAUncertaintyInput,
) -> IOLCASobolEvaluationContext | None:
    if lca_input.lca_type != IO_LCA_FAMILY or not lcia_uncertainty_source_active(
        uncertainty_config
    ):
        return None
    request = normalize_io_lca_uncertainty_request(
        base_io_lca_args=base_io_lca_args,
        lcia_parameters=cast(dict[str, Any], uncertainty_config.get(LCIA_SOURCE, {})),
    )
    prerequisite = prepare_io_lca_deterministic_prerequisite(
        request=request,
        refresh=False,
        status=NullPhasePrinter(),
    )
    public_rows = load_deterministic_public_rows(request=request, scope=prerequisite)
    plan = build_io_lca_lcia_plan(request=request, public_rows=public_rows)
    return build_io_lca_sobol_evaluation_context(plan=plan)


def _lca_sobol_source_names(
    *,
    lca_input: LCAUncertaintyInput,
    lca_context: IOLCASobolEvaluationContext | None,
) -> tuple[str, ...]:
    if lca_context is not None:
        return (io_lca_source_name(LCIA_SOURCE),)
    if lca_input.run_values_for_runs is not None:
        return lca_input.active_sources
    return ()
