"""Public logs and summaries for aSoCC Monte Carlo runs."""

from pyaesa.asocc.uncertainty.sources.lcia import LCIAPlan
from pyaesa.asocc.uncertainty.sources.inter_method import (
    INTER_METHOD_SOURCE,
    InterMethodPlan,
)
from pyaesa.asocc.uncertainty.sources.inter_mrio import (
    INTER_MRIO_SOURCE,
    InterMrioPlan,
)
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.asocc.uncertainty.sources.projection import (
    PROJECTION_SOURCE,
    ProjectionPlan,
)
from pyaesa.asocc.uncertainty.sources.reference_year import (
    REFERENCE_YEAR_SOURCE,
    reference_year_source_method_row,
)
from pyaesa.asocc.uncertainty.io.results_readme import write_results_readme
from pyaesa.asocc.uncertainty.engine.evaluation.summary_identity import summary_identity_groups
from pyaesa.asocc.uncertainty.schema.public_rows import ASOCC_UNCERTAINTY_CSV_DTYPES
from pyaesa.asocc.uncertainty.io.source_methods import write_source_methods
from pyaesa.shared.uncertainty_assessment.request.sources import SourceActivationPlan
from pyaesa.shared.uncertainty_assessment.io.public_summary import (
    exact_summary_from_public_runs,
)
from pyaesa.shared.uncertainty_assessment.io.tables import (
    read_uncertainty_table,
    write_uncertainty_table,
)


def write_run_logs(
    *,
    paths: AsoccUncertaintyRunPaths,
    loaded,
    inter_method_plan: InterMethodPlan | None,
    inter_mrio_plan: InterMrioPlan | None,
    lcia_plan: LCIAPlan | None,
    projection_plan: ProjectionPlan | None,
    output_format: str,
    sources: SourceActivationPlan,
    summary_run_count: int = 0,
    public_runs_sparse: bool = False,
    write_summary: bool = True,
) -> None:
    """Write exact summaries, source method log, and result reading guide."""
    if write_summary:
        identity = read_uncertainty_table(
            path=paths.public_row_identity,
            output_format=output_format,
            csv_dtypes=ASOCC_UNCERTAINTY_CSV_DTYPES,
        )
        summary_identity, public_row_groups = summary_identity_groups(
            identity=identity,
            sources=sources,
        )
        summary = exact_summary_from_public_runs(
            identity_frame=summary_identity,
            runs_path=paths.public_runs,
            output_format=output_format,
            run_count=summary_run_count,
            public_row_groups=public_row_groups,
            sparse=public_runs_sparse,
        )
        write_uncertainty_table(
            path=paths.summary_stats_runs,
            frame=summary,
            output_format=output_format,
        )
    write_source_methods(
        path=paths.source_methods,
        rows=_source_method_rows(
            loaded=loaded,
            inter_method_plan=inter_method_plan,
            inter_mrio_plan=inter_mrio_plan,
            lcia_plan=lcia_plan,
            projection_plan=projection_plan,
            sources=sources,
        ),
    )
    write_results_readme(paths=paths)


def _source_method_rows(
    *,
    loaded,
    inter_method_plan: InterMethodPlan | None,
    inter_mrio_plan: InterMrioPlan | None,
    lcia_plan: LCIAPlan | None,
    projection_plan: ProjectionPlan | None,
    sources: SourceActivationPlan,
):
    rows = []
    if sources.is_active(INTER_METHOD_SOURCE) and inter_method_plan is not None:
        rows.append(inter_method_plan.source_method_row)
    if sources.is_active(PROJECTION_SOURCE) and projection_plan is not None:
        rows.append(projection_plan.source_method_row)
    if lcia_plan is not None:
        rows.extend(lcia_plan.source_method_rows)
    if sources.is_active(INTER_MRIO_SOURCE) and inter_mrio_plan is not None:
        rows.append(inter_mrio_plan.source_method_row)
    if sources.is_active(REFERENCE_YEAR_SOURCE):
        rows.append(reference_year_source_method_row(loaded=loaded))
    return rows
