"""Build compact Sobol README text from shared and family supplied sections."""

from string import Template

from pyaesa.shared.runtime.text import join_user_text_lines
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan

README_LINE_WIDTH = 88

_README_TEMPLATE = Template(
    """Sobol variance decomposition outputs
====================================

Purpose
-------

This file describes the Sobol files produced by the $family_label uncertainty
analysis workflow. Sobol indices are variance shares for active uncertainty
sources. They are not signs of effect, probabilities, elasticities, regression
coefficients, or physical units.

The values are conditional on the active sources, their sampling rules, the
selected years, the retained output scope, and the evaluated model version.

Files
-----

- sobol_source_summary$suffix: source level summary over the selected output
  output scope.
- sobol_indices$suffix: row level Sobol values for each evaluated public output
  identity and active uncertainty source.
- ../../logs/scope_manifest.json: run status, Sobol method metadata, sample
  size, convergence status, active source dimensions, selected Sobol years, and
  normalized public Sobol parameters.

How to read sobol_source_summary
--------------------------------

Use this file first when the question is which uncertainty source explains
variance for a selected $family_label output scope.

Each row reports one source_name within one summarized output scope. Selector
columns identify the output scope retained for interpretation. The selected
Sobol output years are the only years for which these indices explain variance.

$selected_scope_note
$source_summary_notes

Main columns:

- variance_weighted_S1: variance weighted mean first order Sobol index across
  included public output rows.
- variance_weighted_ST: variance weighted mean total order Sobol index across
  the same output rows.
- variance_weighted_ST_minus_S1: variance_weighted_ST minus
  variance_weighted_S1. It summarizes interaction involvement of the source
  over the selected scope.
- variance_weight_sum: sum of Sobol output variances used as weights.
- variance_weighted_S1_confidence_half_width: confidence half width for the
  variance weighted first order index.
- variance_weighted_ST_confidence_half_width: confidence half width for the
  variance weighted total order index.
- undefined_output_count: number of public rows whose Sobol values are
  undefined for this source.
- diagnostic_output_count: number of row level estimates whose interval remains
  outside the expected finite variance range.

How to read sobol_indices
-------------------------

Use this file when you need the exact row level decomposition for one evaluated
output identity.

- Output identity columns identify the $family_label output value being
  explained. They are not sampled source values.
- source_name identifies the active uncertainty source whose Sobol value is
  reported.
- sobol_output_variance is the estimated variance of that public row under the
  Sobol A and B design matrices.
- S1 is the first order Sobol index. It estimates Var(E[Y_j | X_i]) / Var(Y_j).
- ST is the total order Sobol index for the same output row and source.
- ST_minus_S1 is ST minus S1. It estimates interaction involvement of that
  source.
- S1_confidence_half_width and ST_confidence_half_width are bootstrap confidence
  half widths for row level estimates.
- estimator_diagnostic records whether the finite sample estimate and its
  confidence interval are inside the expected range for that row and source.
- A finite S1 or ST value of 0 means the corresponding numerator estimator is 0
  while sobol_output_variance is positive.
- NaN means the value is undefined because sobol_output_variance is zero or not
  finite, or because the estimator has no finite numerator terms.
$indices_notes

Interpretation
--------------

First order indices sum to 1 only when the evaluated model is additive in the
active source dimensions and those dimensions explain all sampled variance.
Total order indices can sum above 1 because each interaction is counted once
for every source that participates in it. Do not normalize S1 or ST to sum to 1.

Finite sample estimates are reported without clipping. Slightly negative
values, values above 1, or ST below S1 are numerical diagnostics. Use the
reported confidence half widths when interpreting those rows.

Confidence intervals and convergence
------------------------------------

The confidence half width describes numerical uncertainty from finite Sobol
sample size. It does not describe uncertainty in the model output itself.

For an estimate I and confidence half width h, report the interval as I +/- h.
At confidence_level=$confidence_level, h is computed from deterministic
bootstrap resamples using a normal approximation to the bootstrap estimate
dispersion.

In convergence mode, the stopping check monitors selected scope variance
weighted S1 and ST estimates. Convergence requires:

half_width <= abs_tol + rtol * max(abs(index), scale_floor)

Detailed selector rows can have wider confidence intervals than the selected
scope monitor. Such rows remain valid estimates with their reported precision,
but they are not necessarily the stopping criterion.

Public Sobol parameters
-----------------------

When sobol_parameters is omitted or None, no Sobol outputs are written. When a
dictionary is provided, the public Sobol request is active. Accepted
sobol_parameters keys are mode, n_base_samples, max_base_samples, rtol, and
sobol_years.

- mode: sample size mode. Default when omitted is fixed.
- n_base_samples: base Sobol sample size N. Default when omitted is 128.
- max_base_samples: maximum base sample size in convergence mode. Default when
  omitted is 1048576.
- rtol: relative tolerance used in convergence mode. Default when omitted is
  0.05.
- sobol_years: optional studied output years for which Sobol indices are
  evaluated and reported. Values must be selected from the request years. When
  omitted, the family owner chooses its documented public scope.

Fixed method metadata recorded for reproducibility:

- abs_tol=$abs_tol
- scale_floor=$scale_floor
- confidence_level=$confidence_level
- confidence_resamples=$confidence_resamples
- S1/ST convergence targets: the estimates monitored by the stopping criterion.

Active source dimensions
------------------------

$active_source_dimensions
$method_notes
"""
)


def build_sobol_readme_lines(
    *,
    suffix: str,
    family_label: str,
    source_names: tuple[str, ...],
    selected_scope_line: str,
    plan: SobolPlan,
    source_summary_notes: tuple[str, ...],
    indices_notes: tuple[str, ...],
    method_notes: tuple[str, ...],
) -> list[str]:
    """Return wrapped Sobol README lines."""
    text = _README_TEMPLATE.substitute(
        suffix=suffix,
        family_label=family_label,
        selected_scope_note=_render_note(selected_scope_line),
        source_summary_notes=_render_notes(source_summary_notes),
        indices_notes=_render_notes(indices_notes),
        confidence_level=f"{plan.confidence_level:g}",
        abs_tol=f"{plan.abs_tol:g}",
        scale_floor=f"{plan.scale_floor:g}",
        confidence_resamples=str(plan.confidence_resamples),
        active_source_dimensions=_render_note(
            "Active Sobol source dimensions: " + "; ".join(source_names) + "."
        ),
        method_notes=_render_notes(method_notes),
    )
    return join_user_text_lines(
        text.splitlines(),
        width=README_LINE_WIDTH,
    ).splitlines()


def _render_notes(notes: tuple[str, ...]) -> str:
    return "\n".join(_render_note(note) for note in notes)


def _render_note(note: str) -> str:
    text = note[2:].strip() if note.startswith("- ") else note.strip()
    if not text:
        return ""
    return join_user_text_lines([f"- {text}"], width=README_LINE_WIDTH)
