"""Dynamic ASR component diagnostic data preparation."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.acc.uncertainty.io.artifacts import (
    acc_run_layout_from_manifest,
    acc_run_paths_from_manifest,
)
from pyaesa.acc.uncertainty.sources.source_keys import (
    AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE,
    ASOCC_PROJECTION_SOURCE,
    ASOCC_REFERENCE_YEAR_SOURCE,
)
from pyaesa.asr.figures.common import VALUE_ARRAY_COLUMN, attach_common_columns, visible_values
from pyaesa.asr.figures.transitions import ASR_TRANSITION_SERIES_EXCLUDED_COLUMNS
from pyaesa.asr.uncertainty.figures.row_reader import (
    collapsed_value_rows,
    summary_rows_from_collapsed_values,
)
from pyaesa.asr.uncertainty.figures.scope_planner import FigureContext
from pyaesa.external_inputs.lca.deterministic import (
    load_external_lca_deterministic_rows_from_paths,
)
from pyaesa.external_inputs.lca.monte_carlo import (
    ExternalLCAMonteCarloSource,
    external_lca_values_for_runs,
    load_external_lca_monte_carlo_source_from_path,
)
from pyaesa.io_lca.data.loaders import load_io_lca_method_table
from pyaesa.io_lca.data.paths import main_results_path, resolve_io_lca_paths
from pyaesa.io_lca.orchestration.pipeline.run_signatures import table_extension_for_output
from pyaesa.io_lca.uncertainty.io.artifacts import io_lca_run_paths_from_manifest
from pyaesa.shared.acc_asr_common.scope.composite import build_composite_base_allocate_args
from pyaesa.shared.figures.contracts import SELECTOR_COLUMNS
from pyaesa.shared.figures.trajectory_bands import SUMMARY_COLUMNS
from pyaesa.shared.figures.uncertainty_run_values import (
    RUN_INDEX_ARRAY_COLUMN,
    collect_selected_compact_run_values,
    collect_selected_sparse_run_indexed_values,
    sum_values_by_run_index,
)
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
    EXT_LCA_SSP_SCENARIO_COLUMN,
)
from pyaesa.shared.tabular.scalars import is_display_missing
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)

_SCENARIO_INVARIANT_COLUMNS = {
    ASOCC_SSP_SCENARIO_COLUMN,
    EXT_LCA_SSP_SCENARIO_COLUMN,
}


@dataclass(frozen=True)
class ComponentDiagnosticRows:
    """Prepared dynamic component rows for one ASR uncertainty run."""

    acc_method: pd.DataFrame
    acc_inter: pd.DataFrame
    lca: pd.DataFrame
    manifest: UncertaintyManifest
    requested_years: tuple[int, ...]
    emissions_mode: str | None


def load_component_diagnostic_rows(*, context: FigureContext) -> ComponentDiagnosticRows:
    """Read upstream component artifacts once for dynamic ASR diagnostics."""
    acc_manifest = cast(
        UncertaintyManifest,
        _prerequisite_manifest(context.manifest, source="uncertainty_acc"),
    )
    lca_rows = _lca_component_rows(context=context)
    acc_rows = _convert_acc_to_lca_unit(
        _acc_component_value_rows(manifest=acc_manifest, context=context),
        lca_rows,
    )
    return ComponentDiagnosticRows(
        acc_method=_component_summary_rows(
            rows=acc_rows,
            context=context,
            include_method_axis=True,
            component="acc",
        ),
        acc_inter=_component_summary_rows(
            rows=acc_rows,
            context=context,
            include_method_axis=False,
            component="acc",
        ),
        lca=lca_rows,
        manifest=context.manifest,
        requested_years=tuple(context.requested_years),
        emissions_mode=_acc_emissions_mode(manifest=acc_manifest),
    )


def _acc_emissions_mode(*, manifest: UncertaintyManifest) -> str | None:
    """Return the upstream aCC emission mode used by one ASR uncertainty run."""
    public_args = manifest.arguments or {}
    base_cc_args = public_args.get("base_cc_args", {})
    dynamic_ar6 = base_cc_args.get("dynamic_ar6", {}) if isinstance(base_cc_args, dict) else {}
    value = dynamic_ar6.get("emissions_mode") if isinstance(dynamic_ar6, dict) else None
    return None if is_display_missing(value) else str(value).strip()


def component_scope_rows(
    rows: pd.DataFrame,
    *,
    asr_frame: pd.DataFrame,
    include_method_axis: bool,
) -> pd.DataFrame:
    """Return prepared component rows matching one visible ASR figure scope."""
    out = rows.copy()
    columns = [
        "cc_type",
        "lcia_method",
        "impact",
        "cc_category",
        "cc_model",
        "cc_scenario",
        "ar6_cc_ssp_scenario",
        "asocc_ssp_scenario",
        EXT_LCA_SSP_SCENARIO_COLUMN,
        "s_p",
        "r_c",
    ]
    if include_method_axis:
        columns.append("__method")
    for column in columns:
        values = visible_values(asr_frame, column)
        if column in out.columns and values:
            out = out.loc[_scope_filter(cast(pd.Series, out[column]), values, column=column)].copy()
    return out


def cumulative_component_entries(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """Return cumulative component run arrays for visible component rows."""
    entries = []
    for _key, group in frame.groupby(component_series_columns(frame), dropna=False, sort=True):
        row = pd.Series(group.iloc[0], copy=False)
        values = np.asarray(row["__component_cumulative_values"], dtype=np.float64)
        entries.append((_component_label(row), values))
    return entries


def component_series_columns(frame: pd.DataFrame) -> list[str]:
    """Return component columns that define one visible component series."""
    excluded = _component_series_excluded_columns()
    return [
        column
        for column in frame.columns
        if column not in {*excluded, "year"} and not str(column).startswith("__figure")
    ]


def _component_series_excluded_columns() -> set[str]:
    return {
        "public_row_id",
        VALUE_ARRAY_COLUMN,
        RUN_INDEX_ARRAY_COLUMN,
        "__component_cumulative_values",
        *ASR_TRANSITION_SERIES_EXCLUDED_COLUMNS,
        *SUMMARY_COLUMNS,
        "std",
        "min",
        "max",
    }


def _acc_component_value_rows(
    *,
    manifest: UncertaintyManifest,
    context: FigureContext,
) -> pd.DataFrame:
    paths = acc_run_paths_from_manifest(manifest=manifest)
    identity = read_uncertainty_table(
        path=paths.public_row_identity,
        output_format=context.output_format,
    )
    rows = _identity_with_values(
        identity=_requested_year_rows(identity, context=context),
        path=paths.public_runs,
        output_format=context.output_format,
        layout=acc_run_layout_from_manifest(manifest=manifest),
        completed_runs=int(context.manifest.completed_runs),
    )
    return rows


def _convert_acc_to_lca_unit(acc: pd.DataFrame, lca: pd.DataFrame) -> pd.DataFrame:
    keys = [
        column for column in ("lcia_method", "impact") if column in acc.columns and column in lca
    ]
    lca_units = lca.loc[:, [*keys, "impact_unit"]].drop_duplicates()
    out = acc.merge(
        lca_units.rename(columns={"impact_unit": "__lca_impact_unit"}),
        on=keys,
        how="left",
    )
    out[VALUE_ARRAY_COLUMN] = [
        np.asarray(values, dtype=np.float64) for values in out[VALUE_ARRAY_COLUMN].tolist()
    ]
    out["impact_unit"] = out["__lca_impact_unit"]
    return out.drop(columns=["__lca_impact_unit"])


def _component_summary_rows(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    include_method_axis: bool,
    component: str,
) -> pd.DataFrame:
    summary_input = (
        rows.drop(columns=[RUN_INDEX_ARRAY_COLUMN])
        if RUN_INDEX_ARRAY_COLUMN in rows.columns
        else rows
    )
    collapsed = collapsed_value_rows(
        rows=summary_input,
        context=context,
        include_method_axis=include_method_axis,
    )
    summary = summary_rows_from_collapsed_values(collapsed)
    summary["__component"] = str(component)
    return _with_cumulative_values(
        summary,
        rows,
        context=context,
        include_method_axis=include_method_axis,
    )


def _lca_component_rows(*, context: FigureContext) -> pd.DataFrame:
    manifest = _prerequisite_manifest(context.manifest, source="uncertainty_io_lca")
    if manifest is not None:
        return _io_lca_component_rows(context=context, manifest=manifest)
    if _has_deterministic_io_lca_input(context=context):
        return _deterministic_io_lca_component_rows(context=context)
    return _external_lca_component_rows(context=context)


def _has_deterministic_io_lca_input(*, context: FigureContext) -> bool:
    return any(
        str(item["type"]) == "io_lca_deterministic" for item in context.manifest.external_inputs
    )


def _io_lca_component_rows(
    *,
    context: FigureContext,
    manifest: UncertaintyManifest,
) -> pd.DataFrame:
    paths = io_lca_run_paths_from_manifest(manifest=manifest)
    identity = read_uncertainty_table(
        path=paths.public_row_identity,
        output_format=context.output_format,
    )
    rows = _identity_with_values(
        identity=_requested_year_rows(identity, context=context),
        path=paths.public_runs,
        output_format=context.output_format,
        layout="compact_run_matrix",
        completed_runs=int(context.manifest.completed_runs),
    )
    collapsed = _collapse_lca_rows(rows=rows)
    summary = summary_rows_from_collapsed_values(collapsed)
    summary["__component"] = "lca"
    return _with_cumulative_values(
        summary,
        rows,
        context=context,
        include_method_axis=False,
    )


def _deterministic_io_lca_component_rows(*, context: FigureContext) -> pd.DataFrame:
    item = next(
        item
        for item in context.manifest.external_inputs
        if str(item["type"]) == "io_lca_deterministic"
    )
    args = cast(dict[str, object], context.manifest.arguments)
    paths = resolve_io_lca_paths(
        project_name=str(args["project_name"]),
        agg_reg=bool(args["agg_reg"]),
        agg_sec=bool(args["agg_sec"]),
        agg_version=cast(str | None, args["agg_version"]),
    )
    output_format = str(item["output_format"])
    extension = table_extension_for_output(output_format)
    frames: list[pd.DataFrame] = []
    for lcia_method in cast(list[object], args["lcia_method"]):
        method = str(lcia_method)
        path = main_results_path(
            paths=paths,
            source=str(item["source"]),
            lcia_method=method,
            extension=extension,
        )
        rows = load_io_lca_method_table(path=path)
        scoped = _requested_year_rows(rows, context=context).copy()
        scoped["lcia_method"] = method
        frames.append(scoped)
    loaded = pd.concat(frames, ignore_index=True)
    identity = _deterministic_io_lca_identity(rows=loaded)
    values = np.tile(
        loaded["lca_value"].to_numpy(dtype=np.float64),
        (int(context.manifest.completed_runs), 1),
    )
    out = attach_common_columns(identity)
    out[VALUE_ARRAY_COLUMN] = list(values.T)
    collapsed = _collapse_lca_rows(rows=out)
    summary = summary_rows_from_collapsed_values(collapsed)
    summary["__component"] = "lca"
    return _with_cumulative_values(
        summary,
        out,
        context=context,
        include_method_axis=False,
    )


def _deterministic_io_lca_identity(*, rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["lcia_method", "year", "impact", "impact_unit"]
    columns.extend(column for column in SELECTOR_COLUMNS if column in rows.columns)
    extras = [column for column in rows.columns if column not in {*columns, "lca_value"}]
    out = rows.loc[:, [*columns, *extras]].copy().reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    out.insert(0, "public_row_id", np.arange(len(out), dtype=np.int64))
    return out


def _external_lca_component_rows(*, context: FigureContext) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    base_allocate_args = _base_allocate_args(context=context)
    for item in context.manifest.external_inputs:
        if str(item["type"]) == "external_lca_deterministic":
            frames.append(
                _external_lca_deterministic_source_rows(
                    context=context,
                    item=item,
                    base_allocate_args=base_allocate_args,
                )
            )
            continue
        for raw_path in cast(list[str], item["paths"]):
            source = load_external_lca_monte_carlo_source_from_path(
                path=Path(raw_path),
                version_name=str(item["version_name"]),
                lcia_method=str(item["lcia_method"]),
                years=list(context.requested_years),
                base_allocate_args=base_allocate_args,
            )
            frames.append(_external_lca_source_rows(source=source, context=context))
    rows = pd.concat(frames, ignore_index=True)
    collapsed = _collapse_lca_rows(rows=rows)
    summary = summary_rows_from_collapsed_values(collapsed)
    summary["__component"] = "lca"
    return _with_cumulative_values(
        summary,
        rows,
        context=context,
        include_method_axis=False,
    )


def _external_lca_deterministic_source_rows(
    *,
    context: FigureContext,
    item: dict[str, object],
    base_allocate_args: dict[str, object],
) -> pd.DataFrame:
    rows = load_external_lca_deterministic_rows_from_paths(
        paths=tuple(Path(raw_path) for raw_path in cast(list[str], item["paths"])),
        lcia_method=str(item["lcia_method"]),
        years=list(context.requested_years),
        base_allocate_args=base_allocate_args,
    )
    loaded = cast(pd.DataFrame, rows).rename(columns={"value": "lca_value"}).copy()
    loaded["lcia_method"] = str(item["lcia_method"])
    identity = _external_lca_deterministic_identity(rows=loaded)
    values = np.tile(
        loaded["lca_value"].to_numpy(dtype=np.float64),
        (int(context.manifest.completed_runs), 1),
    )
    out = attach_common_columns(identity)
    out[VALUE_ARRAY_COLUMN] = list(values.T)
    return out


def _external_lca_deterministic_identity(*, rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["lcia_method", "year", "impact", "impact_unit"]
    columns.extend(column for column in SELECTOR_COLUMNS if column in rows.columns)
    columns.extend(column for column in (EXT_LCA_SSP_SCENARIO_COLUMN,) if column in rows.columns)
    extras = [column for column in rows.columns if column not in {*columns, "lca_value", "value"}]
    out = rows.loc[:, [*columns, *extras]].copy().reset_index(drop=True)
    out["year"] = out["year"].astype(int)
    out.insert(0, "public_row_id", np.arange(len(out), dtype=np.int64))
    return out


def _identity_with_values(
    *,
    identity: pd.DataFrame,
    path: Path,
    output_format: str,
    layout: str,
    completed_runs: int,
) -> pd.DataFrame:
    rows = attach_common_columns(identity)
    public_row_ids = pd.Series(identity.loc[:, "public_row_id"], copy=False)
    if layout == "sparse_selected_rows":
        indexed_values = collect_selected_sparse_run_indexed_values(
            path=path,
            output_format=output_format,
            public_row_ids=public_row_ids,
            stop_run_index=completed_runs,
        )
        empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64))
        rows[RUN_INDEX_ARRAY_COLUMN] = [
            indexed_values.get(int(public_id), empty)[0]
            for public_id in rows["public_row_id"].tolist()
        ]
        rows[VALUE_ARRAY_COLUMN] = [
            indexed_values.get(int(public_id), empty)[1]
            for public_id in rows["public_row_id"].tolist()
        ]
        return rows.loc[rows[VALUE_ARRAY_COLUMN].map(len).gt(0)].reset_index(drop=True)
    values = collect_selected_compact_run_values(
        path=path,
        output_format=output_format,
        public_row_ids=public_row_ids,
        stop_run_index=completed_runs,
    )
    rows[VALUE_ARRAY_COLUMN] = [
        values[int(public_id)] for public_id in rows["public_row_id"].tolist()
    ]
    return rows


def _external_lca_source_rows(
    *,
    source: ExternalLCAMonteCarloSource,
    context: FigureContext,
) -> pd.DataFrame:
    run_indices = np.arange(int(context.manifest.completed_runs), dtype=np.int64)
    values = external_lca_values_for_runs(source=source, run_indices=run_indices)
    rows = attach_common_columns(source.identity.copy())
    rows[VALUE_ARRAY_COLUMN] = list(values.T)
    return rows


def _base_allocate_args(*, context: FigureContext) -> dict:
    args = cast(dict[str, object], context.manifest.arguments)
    return build_composite_base_allocate_args(
        project_name=str(args["project_name"]),
        years=cast(int | list[int] | range, args["years"]),
        lcia_method=[str(value) for value in cast(list[object], args["lcia_method"])],
        fu_code=str(args["fu_code"]),
        r_p=cast(str | list[str] | None, args["r_p"]),
        s_p=cast(str | list[str] | None, args["s_p"]),
        r_c=cast(str | list[str] | None, args["r_c"]),
        r_f=cast(str | list[str] | None, args["r_f"]),
        source=str(args["source"]),
        agg_reg=bool(args["agg_reg"]),
        agg_sec=bool(args["agg_sec"]),
        agg_version=cast(str | None, args["agg_version"]),
        group_indices=bool(args["group_indices"]),
        base_asocc_args=cast(dict, args["base_asocc_args"]),
    )


def _collapse_lca_rows(*, rows: pd.DataFrame) -> pd.DataFrame:
    key_columns = [
        column
        for column in rows.columns
        if column not in {"public_row_id", VALUE_ARRAY_COLUMN, RUN_INDEX_ARRAY_COLUMN, "__method"}
    ]
    records = []
    for _key, group in rows.groupby(key_columns, dropna=False, sort=True):
        payload = {column: group.iloc[0][column] for column in key_columns}
        payload[VALUE_ARRAY_COLUMN] = np.concatenate(
            [np.asarray(values, dtype=np.float64) for values in group[VALUE_ARRAY_COLUMN]]
        )
        records.append(payload)
    return pd.DataFrame.from_records(records)


def _with_cumulative_values(
    summary: pd.DataFrame,
    value_rows: pd.DataFrame,
    *,
    context: FigureContext,
    include_method_axis: bool,
) -> pd.DataFrame:
    cumulative = _component_cumulative_value_rows(
        rows=value_rows,
        context=context,
        include_method_axis=include_method_axis,
    ).rename(columns={VALUE_ARRAY_COLUMN: "__component_cumulative_values"})
    merge_columns = [
        column
        for column in cumulative.columns
        if column in summary.columns and column != "__component_cumulative_values"
    ]
    return summary.merge(cumulative, on=merge_columns, how="left")


def _component_cumulative_value_rows(
    *,
    rows: pd.DataFrame,
    context: FigureContext,
    include_method_axis: bool,
) -> pd.DataFrame:
    drop_columns = _component_cumulative_drop_columns(
        context=context,
        include_method_axis=include_method_axis,
    )
    scenario_columns = {column for column in _SCENARIO_INVARIANT_COLUMNS if column in rows}
    key_columns = [
        column
        for column in rows.columns
        if column
        not in {
            *drop_columns,
            "public_row_id",
            "year",
            VALUE_ARRAY_COLUMN,
            RUN_INDEX_ARRAY_COLUMN,
            *scenario_columns,
        }
    ]
    records = []
    for _key, group in rows.groupby(key_columns, dropna=False, sort=True):
        payload = {column: group.iloc[0][column] for column in key_columns}
        if RUN_INDEX_ARRAY_COLUMN in group.columns:
            run_indices = np.concatenate(
                [np.asarray(values, dtype=np.int64) for values in group[RUN_INDEX_ARRAY_COLUMN]]
            )
            values = np.concatenate(
                [np.asarray(values, dtype=np.float64) for values in group[VALUE_ARRAY_COLUMN]]
            )
            _run_indices, values = sum_values_by_run_index(
                run_indices=run_indices,
                values=values,
            )
            payload[VALUE_ARRAY_COLUMN] = values
        else:
            arrays = [
                np.asarray(values, dtype=np.float64)
                for values in group.sort_values("year", kind="stable")[VALUE_ARRAY_COLUMN]
            ]
            payload[VALUE_ARRAY_COLUMN] = np.sum(np.vstack(arrays), axis=0)
        records.append(payload)
    return pd.DataFrame.from_records(records)


def _component_cumulative_drop_columns(
    *,
    context: FigureContext,
    include_method_axis: bool,
) -> set[str]:
    active = set(context.active_sources)
    dropped = {ASOCC_TIME_ROUTE_PUBLIC_COLUMN}
    if _has_source(active, ASOCC_REFERENCE_YEAR_SOURCE):
        dropped.add("reference_year")
    if _has_source(active, ASOCC_PROJECTION_SOURCE):
        dropped.add("l2_reuse_year")
    if _has_source(active, AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE):
        dropped.update({"cc_model", "cc_scenario"})
    if context.dynamic_category_uncertainty_active:
        dropped.add("cc_category")
    if not include_method_axis:
        dropped.update({"__method", "l1_l2_method", "l1_method", "l2_method"})
    return dropped


def _has_source(active: set[str], source: str) -> bool:
    return source in active or any(name.endswith(source) for name in active)


def _scope_filter(values: pd.Series, accepted: list[str], *, column: str) -> pd.Series:
    text = values.astype("string").str.strip()
    accepted_values = {str(value).strip() for value in accepted if str(value).strip()}
    mask = text.isin(accepted_values)
    if column in _SCENARIO_INVARIANT_COLUMNS:
        mask |= values.isna() | text.isna() | text.eq("")
    return mask


def _requested_year_rows(frame: pd.DataFrame, *, context: FigureContext) -> pd.DataFrame:
    years = pd.Series(pd.to_numeric(pd.Series(frame.loc[:, "year"], copy=False), errors="raise"))
    years = years.astype(int)
    return frame.loc[years.isin(context.requested_years)].copy()


def _component_label(row: pd.Series) -> str:
    method = str(row.get("__method", "")).strip()
    return method or "component"


def _prerequisite_manifest(
    manifest: UncertaintyManifest,
    *,
    source: str,
) -> UncertaintyManifest | None:
    for item in manifest.deterministic_prerequisites:
        if str(item.get("base_function_source")) == source:
            return read_manifest(path=Path(str(item["scope_manifest"])))
    return None
