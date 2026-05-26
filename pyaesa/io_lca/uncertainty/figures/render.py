"""IO-LCA uncertainty figure orchestration."""

from functools import partial
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from pyaesa.io_lca.uncertainty.figures.metadata import clear_uncertainty_figure_scope
from pyaesa.io_lca.uncertainty.figures.product_renderers import (
    write_lca_uncertainty_band_figures,
    write_lca_uncertainty_violin_figures,
)
from pyaesa.io_lca.uncertainty.figures.row_reader import (
    FigureTables,
    prepared_identity_rows,
    prepared_summary_rows,
    read_figure_tables,
    violin_rows_from_compact_runs,
)
from pyaesa.io_lca.uncertainty.figures.scope_planner import (
    FigureContext,
    build_figure_context,
)
from pyaesa.io_lca.uncertainty.io.paths import io_lca_uncertainty_figures_root
from pyaesa.io_lca.uncertainty.runtime.models import (
    IOLCAUncertaintyRequest,
    IOLCAUncertaintyRunPaths,
)
from pyaesa.io_lca.figures.common import selector_groups
from pyaesa.shared.figures.checkpoints import default_checkpoint_years
from pyaesa.shared.figures.jobs import PlannedFigureJob, render_figure_jobs
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def render_io_lca_uncertainty_figures(
    *,
    manifest: UncertaintyManifest,
    paths: IOLCAUncertaintyRunPaths,
    request: IOLCAUncertaintyRequest,
    figure_options: dict | None,
    figure_format: dict | None,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render requested IO-LCA uncertainty figures from public run artifacts."""
    context = build_figure_context(
        manifest=manifest,
        paths=paths,
        request=request,
        figure_options=figure_options,
        figure_format=figure_format,
    )
    single_year = len(set(context.requested_years)) == 1
    tables = read_figure_tables(context=context, include_summary=not single_year)
    clear_uncertainty_figure_scope(paths=paths)

    def jobs() -> Iterator[PlannedFigureJob]:
        """Yield IO-LCA uncertainty figure jobs from prepared figure tables."""
        yield from (
            _single_year_jobs(context=context, tables=tables)
            if single_year
            else _multi_year_jobs(context=context, tables=tables)
        )

    return render_figure_jobs(source="uncertainty_io_lca", jobs=jobs, status=status)


def _single_year_jobs(
    *, context: FigureContext, tables: FigureTables
) -> Iterator[PlannedFigureJob]:
    identity_rows = prepared_identity_rows(context=context, identity=tables.identity)
    rows = violin_rows_from_compact_runs(context=context, identity_rows=identity_rows)
    output_dir = io_lca_uncertainty_figures_root(paths=context.paths)
    checkpoint_years = default_checkpoint_years(list(context.requested_years))
    return _lcia_method_jobs(
        rows=rows,
        context=context,
        output_dir=output_dir,
        kind="single_year",
        render_kind="violin",
        checkpoint_years=checkpoint_years,
    )


def _multi_year_jobs(*, context: FigureContext, tables: FigureTables) -> Iterator[PlannedFigureJob]:
    rows = prepared_summary_rows(context=context, summary=tables.summary)
    output_dir = io_lca_uncertainty_figures_root(paths=context.paths)
    return _lcia_method_jobs(
        rows=rows,
        context=context,
        output_dir=output_dir,
        kind="multi_year",
        render_kind="bands",
        checkpoint_years=[],
    )


def _lcia_method_jobs(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    output_dir: Path,
    kind: str,
    render_kind: str,
    checkpoint_years: list[int],
) -> Iterator[PlannedFigureJob]:
    selector_columns = (
        tuple() if context.request.group_indices else context.request.fu_spec.selector_axes
    )
    for lcia_method, group in rows.groupby("lcia_method", dropna=False, sort=True):
        method = str(lcia_method)
        frame = group.copy()
        _selector_cols, groups = selector_groups(frame=frame, selector_columns=selector_columns)
        if render_kind == "violin":
            for _group_key, group_frame in groups:
                for checkpoint_year in checkpoint_years:
                    yield PlannedFigureJob(
                        kind=kind,
                        label=f"{method} single year {int(checkpoint_year)}",
                        render=partial(
                            write_lca_uncertainty_violin_figures,
                            lcia_method_frame=group_frame,
                            reference_frame=frame,
                            figures_dir=output_dir,
                            lcia_method=method,
                            checkpoint_years=[int(checkpoint_year)],
                            dpi=context.figure_dpi,
                            output_format=context.figure_output_format,
                            selector_columns=selector_columns,
                        ),
                    )
            continue
        for _group_key, group_frame in groups:
            yield PlannedFigureJob(
                kind=kind,
                label=f"{method} multi-year uncertainty",
                render=partial(
                    write_lca_uncertainty_band_figures,
                    lcia_method_frame=group_frame,
                    reference_frame=frame,
                    figures_dir=output_dir,
                    lcia_method=method,
                    dpi=context.figure_dpi,
                    output_format=context.figure_output_format,
                    selector_columns=selector_columns,
                ),
            )
