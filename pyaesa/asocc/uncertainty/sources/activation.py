"""aSoCC uncertainty source applicability checks."""

from dataclasses import replace
from typing import Any

from pyaesa.asocc.uncertainty.sources.inter_method import (
    inter_method_uncertainty_has_targets,
)
from pyaesa.asocc.uncertainty.sources.inter_mrio import (
    InterMrioPlan,
    inter_mrio_uncertainty_has_targets,
)
from pyaesa.asocc.uncertainty.sources.lcia import lcia_uncertainty_has_targets
from pyaesa.asocc.uncertainty.sources.names import (
    INTER_METHOD_SOURCE,
    INTER_MRIO_SOURCE,
    LCIA_SOURCE,
    PROJECTION_SOURCE,
    REFERENCE_YEAR_SOURCE,
)
from pyaesa.asocc.uncertainty.sources.projection import (
    projection_uncertainty_has_targets,
)
from pyaesa.asocc.uncertainty.sources.reference_year import (
    reference_year_uncertainty_has_targets,
)
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan


def deactivate_sources_without_row_targets(
    *,
    loaded: Any,
    sources: SourceActivationPlan,
    external_plan: Any,
) -> SourceActivationPlan:
    """Remove requested sources that have no target rows in the selected scope."""
    if sources.is_active(INTER_METHOD_SOURCE) and not inter_method_uncertainty_has_targets(
        loaded=loaded,
        external_plan=external_plan,
    ):
        sources = _without_source(sources=sources, source_name=INTER_METHOD_SOURCE)
    if sources.is_active(LCIA_SOURCE) and not lcia_uncertainty_has_targets(
        loaded=loaded,
        external_method_labels=external_plan.method_labels,
    ):
        sources = _without_source(sources=sources, source_name=LCIA_SOURCE)
    if sources.is_active(PROJECTION_SOURCE) and not projection_uncertainty_has_targets(
        loaded=loaded,
        external_method_labels=external_plan.method_labels,
    ):
        sources = _without_source(sources=sources, source_name=PROJECTION_SOURCE)
    if sources.is_active(REFERENCE_YEAR_SOURCE) and not reference_year_uncertainty_has_targets(
        rows=loaded.rows,
    ):
        sources = _without_source(sources=sources, source_name=REFERENCE_YEAR_SOURCE)
    return sources


def deactivate_inter_mrio_without_targets(
    *,
    sources: SourceActivationPlan,
    plan: InterMrioPlan | None,
) -> tuple[SourceActivationPlan, InterMrioPlan | None]:
    """Remove inter-MRIO uncertainty when the alternate endpoint has no targets."""
    if plan is None or not sources.is_active(INTER_MRIO_SOURCE):
        return sources, plan
    if inter_mrio_uncertainty_has_targets(plan=plan):
        return sources, plan
    return _without_source(sources=sources, source_name=INTER_MRIO_SOURCE), None


def _without_source(*, sources: SourceActivationPlan, source_name: str) -> SourceActivationPlan:
    return replace(
        sources,
        sources=tuple(source for source in sources.sources if source.name != source_name),
    )
