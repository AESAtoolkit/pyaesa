"""AR6 CC uncertainty figure orchestration."""

from pathlib import Path
from collections.abc import Iterator
from typing import cast

import pandas as pd

from pyaesa.ar6_cc.uncertainty.figures.metadata import clear_uncertainty_figure_scope
from pyaesa.ar6_cc.uncertainty.figures.product_renderers import (
    plot_trajectory_band_scope,
)
from pyaesa.ar6_cc.uncertainty.figures.row_reader import (
    FigureTables,
    budget_rows_by_category,
    budget_rows_global,
    categories_by_common_scope,
    category_pair_counts,
    common_pair_counts,
    read_figure_tables,
    summary_rows_by_category,
    summary_rows_global,
)
from pyaesa.ar6_cc.uncertainty.figures.scope_planner import (
    FigureContext,
    build_figure_context,
    category_scope_stem,
    common_scope_stem,
)
from pyaesa.ar6_cc.uncertainty.io.paths import ar6_cc_uncertainty_figures_root
from pyaesa.ar6_cc.uncertainty.runtime.models import AR6CCUncertaintyRunPaths
from pyaesa.shared.figures.jobs import PlannedFigureJob, render_figure_jobs
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def render_ar6_cc_uncertainty_figures(
    *,
    manifest: UncertaintyManifest,
    paths: AR6CCUncertaintyRunPaths,
    figure_options: dict | None,
    figure_format: dict | None,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render requested AR6 CC uncertainty figures from public run artifacts."""
    context = build_figure_context(
        manifest=manifest,
        paths=paths,
        figure_options=figure_options,
        figure_format=figure_format,
    )
    tables = read_figure_tables(context=context)
    clear_uncertainty_figure_scope(paths=paths)

    def jobs() -> Iterator[PlannedFigureJob]:
        """Yield AR6 CC uncertainty figure jobs from prepared figure tables."""
        yield from (
            _active_category_jobs(context=context, tables=tables)
            if context.category_uncertainty
            else _inactive_category_jobs(context=context, tables=tables)
        )

    return render_figure_jobs(source="uncertainty_ar6_cc", jobs=jobs, status=status)


def _active_category_jobs(
    *,
    context: FigureContext,
    tables: FigureTables,
) -> Iterator[PlannedFigureJob]:
    rows = summary_rows_global(tables=tables)
    output_dir = ar6_cc_uncertainty_figures_root(paths=context.paths)
    pair_counts = common_pair_counts(tables=tables)
    categories_by_scope = categories_by_common_scope(tables=tables)
    budget_rows = budget_rows_global(tables=tables)
    for key, group in rows.groupby("ssp_scenario", dropna=False, sort=True):
        ssp_scenario = str(key)
        budget_group = budget_rows.loc[budget_rows["ssp_scenario"].astype(str) == ssp_scenario]
        output_stem = output_dir / common_scope_stem(
            ssp_scenario=ssp_scenario,
        )
        yield _band_job(
            kind="multi_year",
            label=str(ssp_scenario),
            group=group,
            budget_group=budget_group,
            output_stem=output_stem,
            title_categories=categories_by_scope[ssp_scenario],
            variable_name=context.variable_name,
            ssp_scenario=ssp_scenario,
            pair_count=pair_counts[ssp_scenario],
            sampling_method=context.sampling_method,
            study_years=list(context.requested_years),
            dpi=context.figure_dpi,
            output_format=context.figure_output_format,
        )


def _inactive_category_jobs(
    *,
    context: FigureContext,
    tables: FigureTables,
) -> Iterator[PlannedFigureJob]:
    rows = summary_rows_by_category(tables=tables)
    output_dir = ar6_cc_uncertainty_figures_root(paths=context.paths)
    pair_counts = category_pair_counts(tables=tables)
    budget_rows = budget_rows_by_category(tables=tables)
    group_columns = ["ssp_scenario", "cc_category"]
    for key, group in rows.groupby(group_columns, dropna=False, sort=True):
        ssp_scenario, category = _two_part_key(key)
        budget_group = budget_rows.loc[
            budget_rows["ssp_scenario"].astype(str).eq(ssp_scenario)
            & budget_rows["cc_category"].astype(str).eq(category)
        ]
        output_stem = output_dir / category_scope_stem(
            ssp_scenario=ssp_scenario,
            category=category,
        )
        yield _band_job(
            kind="multi_year",
            label=f"{ssp_scenario} {category}",
            group=group,
            budget_group=budget_group,
            output_stem=output_stem,
            title_categories=[category],
            variable_name=context.variable_name,
            ssp_scenario=ssp_scenario,
            pair_count=pair_counts[(ssp_scenario, category)],
            sampling_method=context.sampling_method,
            study_years=list(context.requested_years),
            dpi=context.figure_dpi,
            output_format=context.figure_output_format,
        )


def _band_job(
    *,
    kind: str,
    label: str,
    group: pd.DataFrame,
    budget_group: pd.DataFrame,
    output_stem: Path,
    title_categories: list[str],
    variable_name: str,
    ssp_scenario: str,
    pair_count: int,
    sampling_method: str,
    study_years: list[int],
    dpi: int,
    output_format: str,
) -> PlannedFigureJob:
    frame = group.copy()
    budget_frame = budget_group.copy()
    return PlannedFigureJob(
        kind=kind,
        label=label,
        render=lambda: plot_trajectory_band_scope(
            frame=frame,
            budget_frame=budget_frame,
            output_stem=output_stem,
            title_categories=title_categories,
            variable_name=variable_name,
            ssp_scenario=ssp_scenario,
            pair_count=pair_count,
            sampling_method=sampling_method,
            study_years=study_years,
            dpi=dpi,
            output_format=output_format,
        ),
    )


def _two_part_key(key: object) -> tuple[str, str]:
    values = cast(tuple[object, object], key)
    return str(values[0]), str(values[1])
