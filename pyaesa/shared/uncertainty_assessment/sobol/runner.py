"""Family neutral Sobol execution loop."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.shared.uncertainty_assessment.sobol.plan import SOBOL_TARGETS, SobolPlan
from pyaesa.shared.uncertainty_assessment.sobol.accumulator import (
    SobolIndexEstimate,
    SobolMomentAccumulator,
)
from pyaesa.shared.uncertainty_assessment.sobol.design import (
    SobolEvaluationChunk,
    iter_saltelli_chunks,
    saltelli_design,
    sobol_base_sequence,
    sobol_chunk_rows,
)
from pyaesa.shared.uncertainty_assessment.sobol.diagnostics import (
    max_sobol_confidence_half_width,
    sobol_diagnostic_counts,
    sobol_source_summary_confidence_converged,
    sobol_source_summary_estimator_range_pass,
)
from pyaesa.shared.uncertainty_assessment.sobol.summary import (
    sobol_global_source_summary,
    sobol_index_table,
)
from pyaesa.shared.runtime.reporting.run_progress import sobol_progress
from pyaesa.shared.runtime.reporting.status import StatusSink

SourceSummaryBuilder = Callable[
    [pd.DataFrame, tuple[str, ...], SobolIndexEstimate],
    pd.DataFrame,
]


@dataclass(frozen=True)
class EvaluatedSobolChunk:
    """Evaluated model outputs for one Saltelli design chunk."""

    identity: pd.DataFrame
    a_values: np.ndarray
    b_values: np.ndarray
    ab_values: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class SobolAnalysisResult:
    """Generic Sobol output tables and status."""

    indices: pd.DataFrame
    source_summary: pd.DataFrame
    status: dict[str, object]


def run_sobol_analysis(
    *,
    plan: SobolPlan,
    dimension_names: tuple[str, ...],
    evaluate: Callable[[SobolEvaluationChunk], EvaluatedSobolChunk],
    source_summary_builder: SourceSummaryBuilder | None = None,
    progress_source: str = "Sobol",
    status: StatusSink | None = None,
) -> SobolAnalysisResult:
    """Run fixed or convergence Sobol analysis with one family evaluator."""
    checkpoints = sobol_base_sequence(
        mode=plan.mode,
        n_base_samples=plan.n_base_samples,
        max_base_samples=plan.max_base_samples,
    )
    dimension_count = len(dimension_names)
    design = saltelli_design(
        n_base_samples=checkpoints[-1],
        dimension_count=dimension_count,
    )
    first_chunk = iter_saltelli_chunks(
        design=design,
        chunk_rows=1,
        start_row=0,
        stop_row=1,
    )[0]
    first = evaluate(first_chunk)
    output_count = first.a_values.shape[1]
    chunk_rows = sobol_chunk_rows(
        output_count=output_count,
        dimension_count=dimension_count,
    )
    reached = False
    history: list[dict[str, object]] = []
    final_table = pd.DataFrame()
    final_source_summary = pd.DataFrame()
    progress = sobol_progress(source=progress_source, status=status)
    try:
        for checkpoint_index, n_base in enumerate(checkpoints):
            progress.begin(
                label=_sobol_progress_label(
                    n_base=n_base,
                    max_base=plan.max_base_samples,
                    dimension_count=dimension_count,
                    mode=plan.mode,
                )
            )
            estimates = _estimate_checkpoint(
                plan=plan,
                design=design,
                n_base=n_base,
                output_count=output_count,
                dimension_count=dimension_count,
                chunk_rows=chunk_rows,
                first=first,
                evaluate=evaluate,
            )
            final_table, final_source_summary, checkpoint_status = _tables_and_status(
                identity=first.identity,
                dimension_names=dimension_names,
                estimates=estimates,
                plan=plan,
                source_summary_builder=source_summary_builder,
            )
            reached = bool(
                checkpoint_index > 0
                and checkpoint_status["confidence_precision_pass"]
                and checkpoint_status["source_summary_range_pass"]
            )
            history.append({"n_base_samples": int(n_base), "reached": reached, **checkpoint_status})
            final_checkpoint = checkpoint_index == len(checkpoints) - 1
            progress.complete(
                label=_sobol_progress_label(
                    n_base=n_base,
                    max_base=plan.max_base_samples,
                    dimension_count=dimension_count,
                    mode=plan.mode,
                ),
                persistent=plan.mode == "fixed" or reached or final_checkpoint,
            )
            if plan.mode == "fixed" or reached:
                break
    finally:
        progress.finish()
    return SobolAnalysisResult(
        indices=final_table,
        source_summary=final_source_summary,
        status={
            "reached": bool(reached),
            "mode": plan.mode,
            "n_base_samples": int(cast(int, history[-1]["n_base_samples"])),
            "rtol": plan.rtol,
            "abs_tol": plan.abs_tol,
            "scale_floor": plan.scale_floor,
            "convergence_targets": list(SOBOL_TARGETS),
            "confidence_level": plan.confidence_level,
            "confidence_resamples": plan.confidence_resamples,
            "convergence_monitor": "selected_scope_source_confidence_interval",
            "confidence_precision_pass": bool(cast(bool, history[-1]["confidence_precision_pass"])),
            "source_summary_range_pass": bool(cast(bool, history[-1]["source_summary_range_pass"])),
            "estimator_diagnostics_pass": bool(
                cast(bool, history[-1]["estimator_diagnostics_pass"])
            ),
            "diagnostic_output_count": int(cast(int, history[-1]["diagnostic_output_count"])),
            "negative_S1_count": int(cast(int, history[-1]["negative_S1_count"])),
            "ST_below_S1_count": int(cast(int, history[-1]["ST_below_S1_count"])),
            "above_one_count": int(cast(int, history[-1]["above_one_count"])),
            "max_S1_confidence_half_width": float(
                cast(float, history[-1]["max_S1_confidence_half_width"])
            ),
            "max_ST_confidence_half_width": float(
                cast(float, history[-1]["max_ST_confidence_half_width"])
            ),
        },
    )


def _sobol_progress_label(
    *,
    n_base: int,
    max_base: int,
    dimension_count: int,
    mode: str,
) -> str:
    """Return a compact Sobol checkpoint progress label."""
    evaluations = int(n_base) * (int(dimension_count) + 2)
    if str(mode) == "fixed":
        return f"base samples {int(n_base)}; design evaluations {evaluations}"
    return (
        f"base samples {int(n_base)} (max {int(max_base)}); "
        f"design evaluations {evaluations}; latest completed checkpoint, "
        "next checkpoint running"
    )


def _estimate_checkpoint(
    *,
    plan: SobolPlan,
    design,
    n_base: int,
    output_count: int,
    dimension_count: int,
    chunk_rows: int,
    first: EvaluatedSobolChunk,
    evaluate: Callable[[SobolEvaluationChunk], EvaluatedSobolChunk],
) -> SobolIndexEstimate:
    weights = (
        _bootstrap_row_weights(
            base_count=n_base,
            confidence_resamples=plan.confidence_resamples,
        )
        if n_base > 1
        else None
    )
    accumulator = SobolMomentAccumulator(
        output_count=output_count,
        dimension_count=dimension_count,
        confidence_resamples=plan.confidence_resamples,
    )
    accumulator.add(
        a_values=first.a_values,
        b_values=first.b_values,
        mixed_values=first.ab_values,
        row_weights=None if weights is None else weights[:, :1],
    )
    for chunk in iter_saltelli_chunks(
        design=design,
        chunk_rows=chunk_rows,
        start_row=1,
        stop_row=n_base,
    ):
        evaluated = evaluate(chunk)
        row_stop = chunk.row_start + chunk.a.shape[0]
        accumulator.add(
            a_values=evaluated.a_values,
            b_values=evaluated.b_values,
            mixed_values=evaluated.ab_values,
            row_weights=None if weights is None else weights[:, chunk.row_start : row_stop],
        )
    return accumulator.estimates(confidence_level=plan.confidence_level)


def _bootstrap_row_weights(*, base_count: int, confidence_resamples: int) -> np.ndarray:
    """Return deterministic bootstrap row count weights for one Sobol checkpoint."""
    rng = np.random.default_rng(np.random.SeedSequence(int(base_count)))
    index_dtype = np.int32 if base_count <= np.iinfo(np.int32).max else np.int64
    weights = np.zeros((confidence_resamples, base_count), dtype=np.float32)
    for resample in range(confidence_resamples):
        row_indices = rng.integers(0, base_count, size=base_count, dtype=index_dtype)
        weights[resample, :] = np.bincount(row_indices, minlength=base_count)
    return weights


def _tables_and_status(
    *,
    identity: pd.DataFrame,
    dimension_names: tuple[str, ...],
    estimates: SobolIndexEstimate,
    plan: SobolPlan,
    source_summary_builder: SourceSummaryBuilder | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    table = sobol_index_table(
        identity=identity,
        dimension_names=dimension_names,
        estimates=estimates,
    )
    source_summary = (
        sobol_global_source_summary(
            dimension_names=dimension_names,
            estimates=estimates,
            confidence_level=plan.confidence_level,
        )
        if source_summary_builder is None
        else source_summary_builder(identity, dimension_names, estimates)
    )
    convergence_summary = sobol_global_source_summary(
        dimension_names=dimension_names,
        estimates=estimates,
        confidence_level=plan.confidence_level,
    )
    diagnostic_counts = sobol_diagnostic_counts(
        s1=estimates.s1,
        st=estimates.st,
        s1_confidence_half_width=estimates.s1_confidence_half_width,
        st_confidence_half_width=estimates.st_confidence_half_width,
        variance=estimates.variance,
    )
    confidence_pass = sobol_source_summary_confidence_converged(
        source_summary=convergence_summary,
        plan=plan,
    )
    range_pass = sobol_source_summary_estimator_range_pass(
        source_summary=convergence_summary,
        plan=plan,
    )
    return (
        table,
        source_summary,
        {
            "estimator_diagnostics_pass": int(diagnostic_counts["diagnostic_output_count"]) == 0,
            "confidence_precision_pass": bool(confidence_pass),
            "source_summary_range_pass": bool(range_pass),
            "max_S1_confidence_half_width": max_sobol_confidence_half_width(
                source_summary=convergence_summary,
                column="variance_weighted_S1_confidence_half_width",
            ),
            "max_ST_confidence_half_width": max_sobol_confidence_half_width(
                source_summary=convergence_summary,
                column="variance_weighted_ST_confidence_half_width",
            ),
            **diagnostic_counts,
        },
    )
