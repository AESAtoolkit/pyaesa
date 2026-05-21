"""External aSoCC row inputs for public uncertainty runs."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pandas as pd

from pyaesa.asocc.methods.lcia_inputs import normalize_lcia_methods
from pyaesa.asocc.runtime.scope.context_rebuild import resolve_external_ssp_scenario_options_by_year
from pyaesa.asocc.runtime.paths.external import get_asocc_external_method_level_dir
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    ASOCC_VALUE_COLUMN,
    LoadedAsoccFinalRows,
)
from pyaesa.external_inputs.asocc.schema.contracts import (
    iter_external_method_selections,
    validate_external_method_collisions,
)
from pyaesa.external_inputs.asocc.deterministic.files import (
    ExternalDeterministicRowsSource,
    describe_expected_external_deterministic_stems,
    expected_external_deterministic_stems,
    load_external_deterministic_rows_with_source,
)
from pyaesa.external_inputs.asocc.schema.file_specs import external_asocc_expected_stems
from pyaesa.external_inputs.asocc.monte_carlo.files import (
    MaterializedExternalMonteCarloRowsSource,
    external_monte_carlo_source_for_years,
    materialize_external_monte_carlo_source,
    resolve_external_monte_carlo_source,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch

EXTERNAL_ASOCC_RUN_SOURCE = "external_asocc_run_uncertainty"


class ExternalAsoccRunInventoryExhausted(ValueError):
    """Raised when convergence consumes all supplied external aSoCC runs."""


@dataclass(frozen=True)
class ExternalAsoccRowsPlan:
    """External aSoCC method labels and optional Monte Carlo row sources."""

    method_labels: tuple[str, ...] = ()
    deterministic_sources: tuple[ExternalDeterministicRowsSource, ...] = ()
    monte_carlo_sources: tuple[MaterializedExternalMonteCarloRowsSource, ...] = ()


@dataclass(frozen=True)
class _ExternalRowsContext:
    proj_base: Path
    fu_code: str
    years: list[int]
    lcia_methods: list[str] | None
    ssp_scenario_options_by_year: dict[int, list[str | None]]


def resolve_external_asocc_rows(
    *,
    loaded: LoadedAsoccFinalRows,
    external_method: dict[str, Any] | None,
    required_runs: int | None,
    external_lcia_methods: list[str] | None = None,
) -> tuple[LoadedAsoccFinalRows, ExternalAsoccRowsPlan]:
    """Resolve external aSoCC row inputs for the current uncertainty run."""
    if external_method is None:
        return loaded, ExternalAsoccRowsPlan()
    context = _external_rows_context(
        loaded=loaded,
        external_lcia_methods=external_lcia_methods,
    )
    native_methods = loaded.asocc_scope.target_selector_payload.get("methods")
    validate_external_method_collisions(
        native_labels=None if native_methods is None else list(native_methods),
        external_method=external_method,
        fu_code=context.fu_code,
        where="uncertainty_asocc external_method",
    )
    deterministic_frames: list[pd.DataFrame] = []
    deterministic_sources: list[ExternalDeterministicRowsSource] = []
    monte_carlo_sources: list[MaterializedExternalMonteCarloRowsSource] = []
    method_labels: list[str] = []
    for selection in iter_external_method_selections(
        external_method=external_method,
        fu_code=context.fu_code,
    ):
        method_labels.append(selection.asocc_method_label)
        mc_source = resolve_external_monte_carlo_source(
            proj_base=context.proj_base,
            selection=selection,
            years=context.years,
            lcia_methods=context.lcia_methods,
            ssp_scenario_options_by_year=context.ssp_scenario_options_by_year,
        )
        if mc_source is not None:
            materialized = materialize_external_monte_carlo_source(source=mc_source)
            if required_runs is not None:
                _validate_render_capacity(source=materialized, n_runs=required_runs)
            monte_carlo_sources.append(materialized)
            continue
        deterministic = load_external_deterministic_rows_with_source(
            proj_base=context.proj_base,
            selection=selection,
            years=context.years,
            lcia_methods=context.lcia_methods,
            ssp_scenario_options_by_year=context.ssp_scenario_options_by_year,
        )
        if deterministic is None:
            _raise_missing_external_rows(context=context, selection=selection)
        deterministic_frames.append(_finalize_external_rows(frame=deterministic.rows))
        deterministic_sources.append(deterministic.source)
    if deterministic_frames:
        rows = _concat_rows([loaded.rows, *deterministic_frames])
        loaded = replace(loaded, rows=rows)
    return loaded, ExternalAsoccRowsPlan(
        method_labels=tuple(sorted(method_labels)),
        deterministic_sources=tuple(deterministic_sources),
        monte_carlo_sources=tuple(monte_carlo_sources),
    )


def external_asocc_has_monte_carlo_rows(
    *,
    loaded: LoadedAsoccFinalRows,
    external_method: dict[str, Any] | None,
    external_lcia_methods: list[str] | None = None,
) -> bool:
    """Return whether an external aSoCC selection provides stochastic run rows."""
    if external_method is None:
        return False
    lcia_methods = _external_lcia_methods(
        loaded=loaded,
        external_lcia_methods=external_lcia_methods,
    )
    years = [int(year) for year in loaded.requested_years]
    for selection in iter_external_method_selections(
        external_method=external_method,
        fu_code=str(loaded.base_asocc_args["fu_code"]),
    ):
        source = resolve_external_monte_carlo_source(
            proj_base=loaded.path_scope.proj_base,
            selection=selection,
            years=years,
            lcia_methods=lcia_methods,
            ssp_scenario_options_by_year=None,
        )
        if source is not None:
            return True
    return False


def append_external_monte_carlo_matrix(
    *,
    template: pd.DataFrame,
    values: np.ndarray,
    plan: ExternalAsoccRowsPlan,
    batch: RunBatch,
    unit_values: np.ndarray | None = None,
    external_run_indices_by_label: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Append external Monte Carlo rows to a compact run matrix batch."""
    if not plan.monte_carlo_sources:
        return template, values
    external_templates: list[pd.DataFrame] = []
    external_values: list[np.ndarray] = []
    for source in plan.monte_carlo_sources:
        run_indices = _external_source_run_indices(
            source=source,
            batch=batch,
            unit_values=unit_values,
            explicit_run_indices=None
            if external_run_indices_by_label is None
            else external_run_indices_by_label.get(source.selection.asocc_method_label),
        )
        external_template, source_values = _external_source_matrix(
            source=source,
            run_indices=run_indices,
        )
        external_templates.append(external_template)
        external_values.append(source_values)
    return (
        _concat_rows([template, *external_templates]),
        np.concatenate([values, *external_values], axis=1),
    )


def append_external_monte_carlo_template(
    *,
    template: pd.DataFrame,
    plan: ExternalAsoccRowsPlan,
) -> pd.DataFrame:
    """Append external Monte Carlo row templates without reading sampled values."""
    if not plan.monte_carlo_sources:
        return template
    external_templates = [
        _finalize_external_rows(frame=source.run_matrix.template)
        for source in plan.monte_carlo_sources
    ]
    return _concat_rows([template, *external_templates])


def external_plan_for_years(
    *,
    plan: ExternalAsoccRowsPlan,
    years: tuple[int, ...],
) -> ExternalAsoccRowsPlan:
    """Return an external row plan scoped to requested studied years."""
    if not plan.monte_carlo_sources:
        return plan
    scoped_sources = []
    for source in plan.monte_carlo_sources:
        scoped_source = external_monte_carlo_source_for_years(source=source, years=years)
        if scoped_source is not None:
            scoped_sources.append(scoped_source)
    return replace(plan, monte_carlo_sources=tuple(scoped_sources))


def _external_source_matrix(
    *,
    source: MaterializedExternalMonteCarloRowsSource,
    run_indices: tuple[int, ...],
) -> tuple[pd.DataFrame, np.ndarray]:
    requested_positions = np.asarray(run_indices, dtype=np.int64)
    return (
        _finalize_external_rows(frame=source.run_matrix.template),
        source.run_matrix.values[requested_positions, :],
    )


def _external_source_run_indices(
    *,
    source: MaterializedExternalMonteCarloRowsSource,
    batch: RunBatch,
    unit_values: np.ndarray | None,
    explicit_run_indices: np.ndarray | None,
) -> tuple[int, ...]:
    if explicit_run_indices is not None:
        return _validated_inventory_indices(
            source=source,
            run_indices=tuple(int(value) for value in explicit_run_indices.tolist()),
        )
    if unit_values is None:
        return _validated_inventory_indices(
            source=source,
            run_indices=tuple(int(value) for value in batch.run_indices()),
        )
    values = np.asarray(unit_values, dtype=np.float64)
    # Saltelli evaluation rows are design rows, not package Monte Carlo
    # run_index values. Treat external Monte Carlo runs as an empirical
    # source distribution by mapping Sobol unit values onto the available
    # run_index inventory.
    clipped = np.clip(values, 0.0, np.nextafter(1.0, 0.0))
    positions = np.floor(clipped * source.available_runs).astype(np.int64)
    inventory = np.asarray(source.run_indices, dtype=np.int64)
    return tuple(int(value) for value in inventory[positions])


def _validated_inventory_indices(
    *,
    source: MaterializedExternalMonteCarloRowsSource,
    run_indices: tuple[int, ...],
) -> tuple[int, ...]:
    available = source.available_runs
    max_requested = max(run_indices)
    if max_requested < available:
        return run_indices
    raise ExternalAsoccRunInventoryExhausted(
        "External aSoCC Monte Carlo run inventory was exhausted before Monte Carlo "
        "convergence was reached. "
        f"Selection='{source.selection.asocc_method_label}', available run_index range "
        f"0 to {available - 1}, first missing run_index={available}. "
        "Provide more external Monte Carlo runs or run a fixed run request within the "
        "available external inventory."
    )


def _external_rows_context(
    *,
    loaded: LoadedAsoccFinalRows,
    external_lcia_methods: list[str] | None = None,
) -> _ExternalRowsContext:
    years = [int(year) for year in loaded.requested_years]
    return _ExternalRowsContext(
        proj_base=loaded.path_scope.proj_base,
        fu_code=str(loaded.base_asocc_args["fu_code"]),
        years=years,
        lcia_methods=_external_lcia_methods(
            loaded=loaded,
            external_lcia_methods=external_lcia_methods,
        ),
        ssp_scenario_options_by_year=resolve_external_ssp_scenario_options_by_year(
            base_allocate_args=loaded.base_asocc_args,
            years=years,
            output_source_label=loaded.path_scope.source_label,
        ),
    )


def _external_lcia_methods(
    *,
    loaded: LoadedAsoccFinalRows,
    external_lcia_methods: list[str] | None,
) -> list[str] | None:
    """Return LCIA methods allowed for external aSoCC file tokens."""
    return normalize_lcia_methods(
        external_lcia_methods
        if external_lcia_methods is not None
        else loaded.base_asocc_args.get("lcia_method")
    )


def _finalize_external_rows(*, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out[ASOCC_VALUE_COLUMN] = pd.to_numeric(out.pop("value"), errors="raise")
    return out.drop(columns=[column for column in ("level",) if column in out.columns])


def _concat_rows(frames: list[pd.DataFrame]) -> pd.DataFrame:
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    return pd.concat(
        [frame.reindex(columns=columns) for frame in frames],
        ignore_index=True,
    )


def external_method_row_mask(
    *,
    frame: pd.DataFrame,
    method_labels: tuple[str, ...],
) -> pd.Series:
    """Return rows declared through external aSoCC method selectors."""
    if not method_labels or frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    labels = _row_method_labels(frame=frame)
    return labels.isin(set(method_labels))


def _row_method_labels(*, frame: pd.DataFrame) -> pd.Series:
    labels = (
        _string_or_empty(pd.Series(frame.loc[:, "l2_method"], copy=False))
        if "l2_method" in frame
        else _empty_labels(frame)
    )
    for column in ("l1_method", "l1_l2_method"):
        if column not in frame:
            continue
        values = _string_or_empty(pd.Series(frame.loc[:, column], copy=False))
        labels = labels.mask(values.ne(""), values)
    return labels


def _string_or_empty(series: pd.Series) -> pd.Series:
    out = series.astype("string").fillna("").str.strip()
    return out.astype(object)


def _empty_labels(frame: pd.DataFrame) -> pd.Series:
    return pd.Series([""] * len(frame), index=frame.index, dtype=object)


def _validate_render_capacity(*, source, n_runs: int) -> None:
    if source.available_runs < int(n_runs):
        first_missing = source.available_runs
        missing = list(range(first_missing, min(first_missing + 10, int(n_runs))))
        raise ValueError(
            "External aSoCC Monte Carlo files cannot cover the requested uncertainty runs. "
            f"Selection='{source.selection.asocc_method_label}', missing run_index values "
            f"{missing[:10]}."
        )


def _raise_missing_external_rows(*, context: _ExternalRowsContext, selection) -> NoReturn:
    deterministic_stems = expected_external_deterministic_stems(
        selection=selection,
        lcia_methods=context.lcia_methods,
        years=context.years,
        ssp_scenario_options_by_year=context.ssp_scenario_options_by_year,
    )
    deterministic_message = describe_expected_external_deterministic_stems(
        proj_base=context.proj_base,
        selection=selection,
        stems=deterministic_stems,
    )
    monte_carlo_stems = external_asocc_expected_stems(
        fu_code=selection.fu_code,
        file_method_token=selection.file_method_token,
        l1_method=selection.l1_method,
        lcia_methods=context.lcia_methods,
        years=context.years,
        ssp_scenario_options_by_year=context.ssp_scenario_options_by_year,
    )
    monte_carlo_dir = get_asocc_external_method_level_dir(
        proj_base=context.proj_base,
        storage_mode="monte_carlo",
        level=str(selection.level),
    )
    monte_carlo_files = [f"{stem}.csv" for stem in monte_carlo_stems if "__ssp" not in stem]
    raise ValueError(
        "Missing external aSoCC files for uncertainty_asocc. "
        f"Selection='{selection.asocc_method_label}'. "
        "The loader first searches for an external Monte Carlo file set and uses those "
        "run indexed rows when present. If no matching Monte Carlo file set is present, "
        "a deterministic external file set is accepted and repeated for every run. "
        f"Expected Monte Carlo file set in '{monte_carlo_dir}': {monte_carlo_files}. "
        f"Expected deterministic external file set: {deterministic_message}."
    )
