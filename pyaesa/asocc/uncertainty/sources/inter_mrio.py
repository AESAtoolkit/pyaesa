"""Inter-MRIO uncertainty as compact final row interpolation."""

from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.asocc.uncertainty.engine.reuse.prerequisites import (
    prepare_asocc_deterministic_prerequisite,
)
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    ASOCC_TIME_ROUTE_COLUMN,
    ASOCC_VALUE_COLUMN,
    LoadedAsoccFinalRows,
    load_final_deterministic_asocc_rows,
)
from pyaesa.asocc.uncertainty.inputs.external_rows import external_method_row_mask
from pyaesa.asocc.uncertainty.schema.public_rows import expand_rows_to_reference_lcia_axis
from pyaesa.asocc.uncertainty.sources.inter_mrio_reporting import (
    InterMrioRouteReport,
    inter_mrio_notes,
    route_pairs_from_skipped,
    scopes_from_skipped,
    years_from_report,
)
from pyaesa.asocc.uncertainty.sources.inter_mrio_eligibility import (
    non_lcia_final_mask,
    optional_column,
)
from pyaesa.asocc.uncertainty.sources.projection import (
    ProjectionPlan,
    build_projection_plan,
    projection_public_row_template,
    projection_value_matrix_for_indices,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.uncertainty_assessment.monte_carlo.random_streams import uniform_by_run_index
from pyaesa.asocc.uncertainty.io.source_methods import SourceMethodRow
from pyaesa.asocc.uncertainty.sources.names import INTER_MRIO_SOURCE

INTER_MRIO_ALPHA_RANDOM_STREAM = "asocc.inter_mrio.alpha"
_METHOD_COLUMNS = ("l1_l2_method", "l1_method", "l2_method")
# LCIA axes are public duplication axes for non LCIA methods, not endpoint identity.
_INTER_MRIO_MATCH_EXCLUDED_COLUMNS = {
    ASOCC_VALUE_COLUMN,
    ASOCC_TIME_ROUTE_COLUMN,
    "lcia_method",
    "impact",
}
_ALTERNATE_DROPPED_ARG_KEYS = {
    "agg_version",
    "agg_reg",
    "agg_sec",
    "lcia_method",
    "figures",
    "figure_format",
    "figure_external_method",
}


@dataclass(frozen=True)
class InterMrioPlan:
    """Resolved alternate endpoint for inter-MRIO uncertainty."""

    alternate_source: str
    alternate_loaded: LoadedAsoccFinalRows
    alternate_projection_plan: ProjectionPlan | None
    route_report: "InterMrioRouteReport"
    source_method_row: SourceMethodRow
    external_method_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class InterMrioInterpolationMatches:
    """Canonical row matches used by interpolation and route reporting."""

    main_positions: np.ndarray
    alternate_positions: np.ndarray
    interpolated_rows: pd.DataFrame
    skipped_rows: pd.DataFrame


def build_inter_mrio_plan(
    *,
    loaded: LoadedAsoccFinalRows,
    parameters: dict[str, Any],
    projection_active: bool,
    reference_year_uncertainty_active: bool,
    external_method_labels: tuple[str, ...] = (),
    phase: Any = None,
) -> InterMrioPlan:
    """Build the compact inter-MRIO plan from one prepared alternate source."""
    alternate_source = str(parameters["source"]).strip()
    alternate_prerequisite = prepare_asocc_deterministic_prerequisite(
        base_asocc_args=_alternate_base_args(
            base_asocc_args=loaded.base_asocc_args,
            alternate_source=alternate_source,
        ),
        refresh=False,
        reference_year_uncertainty_active=reference_year_uncertainty_active,
        phase=phase,
    )
    alternate_loaded = load_final_deterministic_asocc_rows(prerequisite=alternate_prerequisite)
    alternate_loaded = replace(
        alternate_loaded,
        rows=expand_rows_to_reference_lcia_axis(
            rows=alternate_loaded.rows,
            reference=loaded.rows,
        ),
    )
    alternate_projection_plan = (
        build_projection_plan(loaded=alternate_loaded) if projection_active else None
    )
    route_report = inter_mrio_route_report(
        main_rows=loaded.rows,
        alternate_rows=alternate_loaded.rows,
        external_method_labels=external_method_labels,
    )
    return InterMrioPlan(
        alternate_source=alternate_source,
        alternate_loaded=alternate_loaded,
        alternate_projection_plan=alternate_projection_plan,
        route_report=route_report,
        source_method_row=inter_mrio_source_method_row(
            loaded=loaded,
            alternate_source=alternate_source,
            route_report=route_report,
        ),
        external_method_labels=external_method_labels,
    )


def inter_mrio_uncertainty_has_targets(*, plan: InterMrioPlan) -> bool:
    """Return whether the alternate endpoint can interpolate any selected year."""
    return bool(plan.route_report.interpolated_years)


def _alternate_base_args(
    *,
    base_asocc_args: dict[str, Any],
    alternate_source: str,
) -> dict[str, Any]:
    return {
        **{
            key: value
            for key, value in base_asocc_args.items()
            if key not in _ALTERNATE_DROPPED_ARG_KEYS
        },
        "source": alternate_source,
    }


def apply_inter_mrio_uncertainty_to_matrix(
    *,
    template: pd.DataFrame,
    values: np.ndarray,
    plan: InterMrioPlan,
    batch: RunBatch,
    projection_selection: np.ndarray | None,
    unit_values: np.ndarray | None = None,
    matches: InterMrioInterpolationMatches | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Interpolate eligible final rows between main and alternate endpoint matrices."""
    if template.empty:
        # External method branches can reach this owner through inter-method
        # sampling with no pyaesa owned rows. Inter-MRIO uncertainty is skipped
        # for external methods; those values are already final for this source.
        return template, values
    alternate_values = _alternate_endpoint_values(
        plan=plan,
        batch=batch,
        projection_selection=projection_selection,
    )
    if matches is None:
        matches = inter_mrio_interpolation_matches(template=template, plan=plan)
    main_positions = matches.main_positions
    alternate_positions = matches.alternate_positions
    if main_positions.size == 0:
        return template, values
    alpha = _alpha(batch=batch, unit_values=unit_values)[:, None]
    out = values.copy()
    out[:, main_positions] = values[:, main_positions] + alpha * (
        alternate_values[:, alternate_positions] - values[:, main_positions]
    )
    return template, out


def inter_mrio_interpolation_matches(
    *,
    template: pd.DataFrame,
    plan: InterMrioPlan,
) -> InterMrioInterpolationMatches:
    """Return stable interpolation positions for one compact batch template."""
    return _interpolation_matches(
        template=template,
        alternate_template=_alternate_endpoint_template(plan=plan),
        external_method_labels=plan.external_method_labels,
    )


def inter_mrio_source_method_row(
    *,
    loaded: LoadedAsoccFinalRows,
    alternate_source: str,
    route_report: InterMrioRouteReport,
) -> SourceMethodRow:
    """Return the compact scientific log row for inter-MRIO uncertainty."""
    return SourceMethodRow(
        source_component="asocc",
        source_name=INTER_MRIO_SOURCE,
        scope=str(loaded.base_asocc_args["fu_code"]),
        applied_bucket=loaded.final_bucket,
        year_min=min(loaded.requested_years),
        year_max=max(loaded.requested_years),
        distribution="continuous uniform alpha on [0, 1]",
        shared_random_variable="run_index",
        formula="sampled row = main_value + alpha * (alternate_value - main_value)",
        notes=inter_mrio_notes(
            alternate_source=alternate_source,
            route_report=route_report,
        ),
    )


def inter_mrio_route_report(
    *,
    main_rows: pd.DataFrame,
    alternate_rows: pd.DataFrame,
    external_method_labels: tuple[str, ...] = (),
) -> InterMrioRouteReport:
    """Return years interpolated or skipped by deterministic time route compatibility."""
    matches = _interpolation_matches(
        template=main_rows,
        alternate_template=alternate_rows,
        external_method_labels=external_method_labels,
    )
    return InterMrioRouteReport(
        interpolated_years=years_from_report(frame=matches.interpolated_rows),
        skipped_years=years_from_report(frame=matches.skipped_rows),
        skipped_route_pairs=route_pairs_from_skipped(frame=matches.skipped_rows),
        skipped_scopes=scopes_from_skipped(frame=matches.skipped_rows),
    )


def _alternate_endpoint_template(*, plan: InterMrioPlan) -> pd.DataFrame:
    if plan.alternate_projection_plan is not None:
        return projection_public_row_template(plan=plan.alternate_projection_plan)
    return plan.alternate_loaded.rows


def _alternate_endpoint_values(
    *,
    plan: InterMrioPlan,
    batch: RunBatch,
    projection_selection: np.ndarray | None,
) -> np.ndarray:
    if plan.alternate_projection_plan is not None:
        return projection_value_matrix_for_indices(
            plan=plan.alternate_projection_plan,
            batch=batch,
            selected_indices=cast(np.ndarray, projection_selection),
        )
    values = plan.alternate_loaded.rows[ASOCC_VALUE_COLUMN].to_numpy(dtype="float64")
    return np.broadcast_to(values, (batch.n_runs, len(values)))


def _interpolation_positions(
    *,
    template: pd.DataFrame,
    alternate_template: pd.DataFrame,
    external_method_labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    matches = _interpolation_matches(
        template=template,
        alternate_template=alternate_template,
        external_method_labels=external_method_labels,
    )
    return matches.main_positions, matches.alternate_positions


def _interpolation_matches(
    *,
    template: pd.DataFrame,
    alternate_template: pd.DataFrame,
    external_method_labels: tuple[str, ...] = (),
) -> InterMrioInterpolationMatches:
    columns = _alignment_columns(template=template, alternate_template=alternate_template)
    # External methods have no pyaesa owned alternate MRIO endpoint. They
    # remain at the user supplied aSoCC value under inter-MRIO uncertainty.
    external_rows = external_method_row_mask(frame=template, method_labels=external_method_labels)
    main_mask = non_lcia_final_mask(frame=template) & ~external_rows
    alternate_mask = non_lcia_final_mask(frame=alternate_template)
    main_source = template.loc[main_mask].reset_index(drop=True)
    alternate_source = alternate_template.loc[alternate_mask].reset_index(drop=True)
    main = main_source.loc[:, columns].copy()
    main["_main_position"] = np.flatnonzero(main_mask.to_numpy(dtype=bool))
    main["_method_label"] = _method_labels(frame=main_source)
    main["_year"] = pd.Series(
        pd.to_numeric(main_source.loc[:, "year"], errors="raise"),
        index=main.index,
    ).astype("int64")
    main["_main_route"] = main_source.loc[:, ASOCC_TIME_ROUTE_COLUMN].to_numpy()
    alternate = alternate_source.loc[:, columns].copy()
    alternate["_alternate_position"] = np.flatnonzero(alternate_mask.to_numpy(dtype=bool))
    alternate["_alternate_route"] = alternate_source.loc[:, ASOCC_TIME_ROUTE_COLUMN].to_numpy()
    matched = main.merge(alternate, on=columns, how="left", sort=False)
    interpolated_years = _whole_year_interpolation_years(matched=matched)
    interpolated_rows = matched.loc[matched["_year"].isin(interpolated_years)].reset_index(
        drop=True
    )
    interpolated_main_positions = set(interpolated_rows["_main_position"].astype("int64").tolist())
    skipped_main = main.loc[~main["_main_position"].isin(interpolated_main_positions)]
    skipped_rows = _skipped_route_rows(
        skipped_main=skipped_main,
        alternate_source=alternate_source,
        columns=_diagnostic_alignment_columns(
            template=template,
            alternate_template=alternate_template,
        ),
    )
    return InterMrioInterpolationMatches(
        main_positions=interpolated_rows["_main_position"].to_numpy(dtype=np.int64),
        alternate_positions=interpolated_rows["_alternate_position"].to_numpy(dtype=np.int64),
        interpolated_rows=interpolated_rows,
        skipped_rows=skipped_rows,
    )


def _alignment_columns(*, template: pd.DataFrame, alternate_template: pd.DataFrame) -> list[str]:
    return [
        column
        for column in template.columns
        if column in alternate_template.columns and column not in _INTER_MRIO_MATCH_EXCLUDED_COLUMNS
    ]


def _diagnostic_alignment_columns(
    *,
    template: pd.DataFrame,
    alternate_template: pd.DataFrame,
) -> list[str]:
    blocked = {*_INTER_MRIO_MATCH_EXCLUDED_COLUMNS, "l2_reuse_year"}
    return [
        column
        for column in template.columns
        if column in alternate_template.columns and column not in blocked
    ]


def _skipped_route_rows(
    *,
    skipped_main: pd.DataFrame,
    alternate_source: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if skipped_main.empty:
        return skipped_main.assign(_alternate_route=pd.Series(dtype="object"))
    alternate = alternate_source.loc[:, columns].copy()
    alternate["_alternate_route"] = alternate_source.loc[:, ASOCC_TIME_ROUTE_COLUMN].to_numpy()
    report = skipped_main.merge(alternate, on=columns, how="left", sort=False)
    report["_alternate_route"] = report["_alternate_route"].fillna("missing")
    return report.loc[
        :,
        ["_method_label", "_year", "_main_route", "_alternate_route"],
    ].drop_duplicates(ignore_index=True)


def _whole_year_interpolation_years(*, matched: pd.DataFrame) -> set[int]:
    years: set[int] = set()
    for raw_year, year_rows in matched.groupby("_year", dropna=False, sort=False):
        alternate_positions = pd.Series(year_rows.loc[:, "_alternate_position"], copy=False)
        if bool(alternate_positions.isna().to_numpy(dtype=bool).any()):
            continue
        main_routes = pd.Series(year_rows.loc[:, "_main_route"], copy=False).astype(str)
        alternate_routes = pd.Series(year_rows.loc[:, "_alternate_route"], copy=False).astype(str)
        if bool(
            (main_routes.to_numpy(dtype=object) == alternate_routes.to_numpy(dtype=object)).all()
        ):
            years.add(cast(int, raw_year))
    return years


def _method_labels(*, frame: pd.DataFrame) -> pd.Series:
    labels = optional_column(frame=frame, column="l2_method").copy()
    for column in ("l1_l2_method", "l1_method"):
        values = optional_column(frame=frame, column=column)
        mask = labels.isna() & values.notna()
        labels.loc[mask] = values.loc[mask]
    return labels.astype(str)


def _alpha(*, batch: RunBatch, unit_values: np.ndarray | None = None) -> np.ndarray:
    return (
        np.asarray(unit_values, dtype=np.float64)
        if unit_values is not None
        else uniform_by_run_index(
            stream_name=INTER_MRIO_ALPHA_RANDOM_STREAM,
            run_indices=batch.run_indices(),
        )
    )
