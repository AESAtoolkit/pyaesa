"""Family neutral Sobol README and method metadata construction."""

from pathlib import Path

from pyaesa.shared.uncertainty_assessment.io.formats import suffix_for_uncertainty_output
from pyaesa.shared.uncertainty_assessment.sobol.plan import SOBOL_TARGETS, SobolPlan
from pyaesa.shared.uncertainty_assessment.sobol.readme_text import (
    build_sobol_readme_lines,
)


def sobol_method_payload(
    *,
    source_names: tuple[str, ...],
    plan: SobolPlan,
    selected_scope: dict[str, object],
) -> dict[str, object]:
    """Return common Sobol method metadata for a run manifest."""
    payload: dict[str, object] = {
        "analysis": "Sobol variance decomposition",
        "source_dimensions": list(source_names),
        "estimators": "Saltelli first order S1 and total order ST",
        "sample_design": "balanced Sobol sequence with Saltelli A, B, and A_Bi matrices",
        "first_order_centering": "centered output estimator",
        "confidence_method": "deterministic bootstrap over Sobol base rows",
        "confidence_level": plan.confidence_level,
        "confidence_resamples": plan.confidence_resamples,
        "convergence_monitor": "selected_scope_source_confidence_interval",
        "method_references": [
            "SALib Sobol analysis: "
            "https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/sobol.html",
        ],
        "mode": plan.mode,
        "rtol": plan.rtol,
        "abs_tol": plan.abs_tol,
        "scale_floor": plan.scale_floor,
        "convergence_targets": list(SOBOL_TARGETS),
    }
    payload.update(selected_scope)
    return payload


def write_sobol_readme(
    *,
    path: Path,
    output_format: str,
    family_label: str,
    source_names: tuple[str, ...],
    selected_scope_line: str,
    plan: SobolPlan,
    source_summary_notes: tuple[str, ...],
    indices_notes: tuple[str, ...],
    method_notes: tuple[str, ...],
) -> None:
    """Write the common Sobol result README with family specific notes."""
    lines = build_sobol_readme_lines(
        suffix=suffix_for_uncertainty_output(output_format),
        family_label=family_label,
        source_names=source_names,
        selected_scope_line=selected_scope_line,
        plan=plan,
        source_summary_notes=source_summary_notes,
        indices_notes=indices_notes,
        method_notes=method_notes,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
