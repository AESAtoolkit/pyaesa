"""Monte Carlo external LCA loading for ASR uncertainty."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.shared.figures.contracts import SELECTOR_COLUMNS
from pyaesa.shared.runtime.scenario.columns import (
    EXT_LCA_SSP_SCENARIO_COLUMN,
    LCA_SSP_START_YEAR_COLUMN,
)
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens

from pyaesa.external_inputs.lca.io import (
    discover_external_lca_files,
    external_lca_no_requested_year_rows_error,
    parse_external_lca_filename,
    read_external_lca,
    validate_external_lca_contract,
    validate_external_lca_required_columns,
    validate_external_lca_reserved_columns,
    validate_external_lca_selector_columns,
)
from pyaesa.external_inputs.shared.compact_matrix import (
    is_compact_run_matrix_dir,
    load_compact_run_matrix_source,
)
from pyaesa.external_inputs.shared.matrix_identity import (
    IdentityLookup,
    assign_values,
    contiguous_inventory,
    positions_for_frame,
    require_complete_matrix,
    template_lookup,
)
from pyaesa.external_inputs.shared.tabular import none_for_missing_series
from .naming import normalize_external_lca_version_name
from pyaesa.external_inputs.lca.paths import external_lca_monte_carlo_dir
from pyaesa.external_inputs.lca.monte_carlo_stream import load_external_lca_long_matrix_source
from .scenario_metadata import with_lca_ssp_start_year

_MONTE_CARLO_REQUIRED_COLUMNS = {
    "run_index",
    "year",
    EXT_LCA_SSP_SCENARIO_COLUMN,
    "value",
}
_FILE_OWNED_COLUMNS = {"lcia_method", "ssp_scenario"}
_COMPACT_IDENTITY_REQUIRED_COLUMNS = {
    "year",
    EXT_LCA_SSP_SCENARIO_COLUMN,
    "impact",
    "impact_unit",
}


@dataclass(frozen=True)
class ExternalLCAMonteCarloSource:
    """External LCA Monte Carlo run source and public identity."""

    version_name: str
    lcia_method: str
    identity: pd.DataFrame
    run_indices: np.ndarray
    paths: tuple[Path, ...]
    values_for_runs: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class _FrameMatrixParts:
    identity: pd.DataFrame
    run_indices: np.ndarray
    lookup: IdentityLookup


class ExternalLCARunInventoryExhausted(ValueError):
    """Raised when an ASR convergence run consumes all supplied LCA runs."""


def load_external_lca_monte_carlo_source(
    *,
    proj_base: Path,
    version_name: str,
    lcia_method: str,
    years: list[int],
    base_allocate_args: dict[str, Any],
) -> ExternalLCAMonteCarloSource | None:
    """Load one external LCA Monte Carlo source, if staged for the request."""
    version = normalize_external_lca_version_name(
        version_name,
        argument_name="external LCA version_name",
    )
    path = _matching_path(
        directory=external_lca_monte_carlo_dir(project_base=proj_base),
        version_name=version,
        lcia_method=lcia_method,
    )
    if path is None:
        return None
    return load_external_lca_monte_carlo_source_from_path(
        path=path,
        version_name=version,
        lcia_method=lcia_method,
        years=years,
        base_allocate_args=base_allocate_args,
    )


def load_external_lca_monte_carlo_source_from_path(
    *,
    path: Path,
    version_name: str,
    lcia_method: str,
    years: list[int],
    base_allocate_args: dict[str, Any],
) -> ExternalLCAMonteCarloSource:
    """Materialize one external LCA Monte Carlo file into canonical identity and runs."""
    source_path = Path(path)
    version = normalize_external_lca_version_name(
        version_name,
        argument_name="external LCA version_name",
    )
    if source_path.is_dir():
        compact = load_compact_run_matrix_source(
            directory=source_path,
            run_file_name="lca_runs.csv",
            context=f"External LCA compact Monte Carlo source '{source_path.name}'",
        )
        identity = compact.identity.copy()
        validate_external_lca_required_columns(
            columns=identity.columns,
            path=source_path / "public_row_identity.csv",
            required_columns=_COMPACT_IDENTITY_REQUIRED_COLUMNS,
            file_label="External LCA compact public_row_identity.csv",
        )
        validate_external_lca_reserved_columns(
            columns=identity.columns,
            path=source_path / "public_row_identity.csv",
            forbidden_columns=_FILE_OWNED_COLUMNS,
            file_label="External LCA compact public_row_identity.csv",
        )
        validate_external_lca_contract(frame=identity, path=source_path, lcia_method=lcia_method)
        validate_external_lca_selector_columns(
            frame=identity,
            path=source_path,
            base_allocate_args=base_allocate_args,
        )
        scenario_series = pd.Series(
            identity.loc[:, EXT_LCA_SSP_SCENARIO_COLUMN],
            copy=False,
        )
        identity[EXT_LCA_SSP_SCENARIO_COLUMN] = none_for_missing_series(
            _normalized_scenarios(scenario_series)
        )
        year_mask = identity["year"].astype(int).isin([int(year) for year in years])
        if not bool(year_mask.any()):
            raise external_lca_no_requested_year_rows_error(path=source_path, years=years)
        selected_columns = year_mask.to_numpy(dtype=bool)
        identity = identity.loc[year_mask, :].reset_index(drop=True)
        identity["public_row_id"] = np.arange(len(identity), dtype=np.int64)
        identity = with_lca_ssp_start_year(identity)
        identity.insert(1, "lcia_method", str(lcia_method))
        return ExternalLCAMonteCarloSource(
            version_name=version,
            lcia_method=str(lcia_method),
            identity=identity,
            run_indices=compact.run_indices,
            paths=(source_path,),
            values_for_runs=lambda requested: compact.values_for_runs(
                np.asarray(requested, dtype=np.int64)
            )[:, selected_columns],
        )
    if source_path.suffix.lower() in {".csv", ".parquet"}:
        source = load_external_lca_long_matrix_source(
            path=source_path,
            lcia_method=lcia_method,
            years=years,
            base_allocate_args=base_allocate_args,
        )
        identity = source.identity.copy()
        run_indices = source.run_indices

        def values_for_runs(requested: np.ndarray) -> np.ndarray:
            return source.values_for_runs(np.asarray(requested, dtype=np.int64))
    else:
        rows = _normalize_rows(
            frame=read_external_lca(source_path),
            path=source_path,
            lcia_method=lcia_method,
            years=years,
            base_allocate_args=base_allocate_args,
        )
        parts = _frame_matrix_parts(
            rows=with_lca_ssp_start_year(rows),
            path=source_path,
            version_name=version,
            lcia_method=lcia_method,
        )
        identity = parts.identity.copy()
        run_indices = parts.run_indices

        def values_for_runs(requested: np.ndarray) -> np.ndarray:
            return _values_for_frame_path(
                path=source_path,
                years=years,
                base_allocate_args=base_allocate_args,
                lcia_method=lcia_method,
                parts=parts,
                requested_runs=np.asarray(requested, dtype=np.int64),
            )

    identity.insert(1, "lcia_method", str(lcia_method))
    return ExternalLCAMonteCarloSource(
        version_name=version,
        lcia_method=str(lcia_method),
        identity=identity,
        run_indices=run_indices,
        paths=(source_path,),
        values_for_runs=values_for_runs,
    )


def external_lca_values_for_runs(
    *,
    source: ExternalLCAMonteCarloSource,
    run_indices: np.ndarray,
) -> np.ndarray:
    """Return external LCA values for package run indices."""
    requested = np.asarray(run_indices, dtype=np.int64)
    if requested.size == 0:
        return np.empty((0, len(source.identity)), dtype=np.float64)
    available = len(source.run_indices)
    missing = requested[(requested < 0) | (requested >= available)]
    if missing.size == 0:
        return source.values_for_runs(requested)
    first_missing = len(source.run_indices)
    raise ExternalLCARunInventoryExhausted(
        "External LCA Monte Carlo run inventory was exhausted before ASR Monte Carlo "
        "convergence was reached. "
        f"version_name='{source.version_name}', lcia_method='{source.lcia_method}', "
        f"available run_index range 0 to {first_missing - 1}, first missing "
        f"run_index={int(missing.min())}. Provide more external LCA Monte Carlo runs "
        "or run a fixed run request within the available external inventory."
    )


def external_lca_values_for_units(
    *,
    source: ExternalLCAMonteCarloSource,
    unit_values: np.ndarray,
) -> np.ndarray:
    """Map Sobol unit interval values onto the empirical external LCA inventory."""
    values = np.asarray(unit_values, dtype=np.float64)
    clipped = np.clip(values, 0.0, np.nextafter(1.0, 0.0))
    positions = np.floor(clipped * len(source.run_indices)).astype(np.int64)
    return external_lca_values_for_runs(source=source, run_indices=positions)


def _matching_path(*, directory: Path, version_name: str, lcia_method: str) -> Path | None:
    compact_name = f"{version_name}__{lcia_method}"
    compact_dir = directory / compact_name
    if is_compact_run_matrix_dir(compact_dir, run_file_name="lca_runs.csv"):
        return compact_dir
    for path in discover_external_lca_files(directory):
        spec = parse_external_lca_filename(path=path)
        if spec.scenario is not None:
            raise ValueError(
                f"External LCA Monte Carlo file '{path.name}' must use "
                "'<version_name>__<lcia_method>' stems without SSP suffixes. The "
                "row column 'lca_ssp_scenario' owns the scenario axis."
            )
        if spec.version_name == version_name and spec.lcia_method == str(lcia_method):
            return path
    return None


def _normalize_rows(
    *,
    frame: pd.DataFrame,
    path: Path,
    lcia_method: str,
    years: list[int],
    base_allocate_args: dict[str, Any],
) -> pd.DataFrame:
    return _normalize_rows_core(
        frame=frame,
        path=path,
        lcia_method=lcia_method,
        years=years,
        base_allocate_args=base_allocate_args,
        validate_lcia_inventory=True,
    )


def _normalize_rows_core(
    *,
    frame: pd.DataFrame,
    path: Path,
    lcia_method: str,
    years: list[int],
    base_allocate_args: dict[str, Any],
    validate_lcia_inventory: bool,
) -> pd.DataFrame:
    validate_external_lca_required_columns(
        columns=frame.columns,
        path=path,
        required_columns=_MONTE_CARLO_REQUIRED_COLUMNS,
        file_label="External LCA Monte Carlo file",
    )
    validate_external_lca_reserved_columns(
        columns=frame.columns,
        path=path,
        forbidden_columns=_FILE_OWNED_COLUMNS,
        file_label="External LCA Monte Carlo file",
    )
    if validate_lcia_inventory:
        validate_external_lca_contract(frame=frame, path=path, lcia_method=lcia_method)
    validate_external_lca_selector_columns(
        frame=frame,
        path=path,
        base_allocate_args=base_allocate_args,
    )
    out = frame.copy()
    out["run_index"] = _numeric_series(out, "run_index").astype("int64")
    if bool((out["run_index"] < 0).any()):
        raise ValueError(f"External LCA Monte Carlo file '{path}' contains negative run_index.")
    out["year"] = _numeric_series(out, "year").astype("int64")
    out = out.loc[out["year"].isin([int(year) for year in years])].copy()
    out["impact"] = out["impact"].astype(str).str.strip()
    out["impact_unit"] = out["impact_unit"].astype(str).str.strip()
    out["value"] = _numeric_series(out, "value").astype("float64")
    out[EXT_LCA_SSP_SCENARIO_COLUMN] = _normalized_scenarios(out[EXT_LCA_SSP_SCENARIO_COLUMN])
    if out.empty:
        raise external_lca_no_requested_year_rows_error(path=path, years=years)
    return out.reset_index(drop=True)


def _normalized_scenarios(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    missing = raw.isna() | raw.eq("")
    out = raw.astype(object)
    unique = sorted({str(value) for value in out.loc[~missing].tolist()})
    mapping = {
        value: normalize_ssp_tokens(
            [value],
            context="External LCA Monte Carlo lca_ssp_scenario column",
        )[0]
        for value in unique
    }
    out.loc[~missing] = out.loc[~missing].map(mapping)
    out.loc[missing] = None
    return pd.Series(out, index=series.index, dtype=object)


def _materialize_matrix(
    *,
    rows: pd.DataFrame,
    path: Path,
    version_name: str,
    lcia_method: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    parts = _frame_matrix_parts(
        rows=rows,
        path=path,
        version_name=version_name,
        lcia_method=lcia_method,
    )
    values = _values_for_rows(rows=rows, parts=parts, requested_runs=parts.run_indices)
    return parts.identity, values, parts.run_indices


def _frame_matrix_parts(
    *,
    rows: pd.DataFrame,
    path: Path,
    version_name: str,
    lcia_method: str,
) -> _FrameMatrixParts:
    identity_columns = _identity_columns(frame=rows)
    context = (
        f"External LCA Monte Carlo file '{path}' "
        f"(version_name='{version_name}', lcia_method='{lcia_method}')"
    )
    if bool(rows.duplicated(["run_index", *identity_columns], keep=False).any()):
        raise ValueError(
            f"{context} contains duplicate values for the same run_index and public LCA identity."
        )
    run_indices = np.asarray(
        contiguous_inventory(
            values=rows["run_index"].to_numpy(dtype=np.int64),
            context=context,
        ),
        dtype=np.int64,
    )
    template = (
        rows.loc[rows["run_index"].eq(0), [*identity_columns, "value"]]
        .sort_values(identity_columns, kind="mergesort")
        .reset_index(drop=True)
    )
    matrix_template, lookup = template_lookup(template=template)
    identity = matrix_template.drop(columns=["value"]).reset_index(drop=True)
    parts = _FrameMatrixParts(identity, run_indices, lookup)
    _values_for_rows(rows=rows, parts=parts, requested_runs=run_indices)
    identity.insert(0, "public_row_id", np.arange(len(identity), dtype=np.int64))
    return _FrameMatrixParts(identity=identity, run_indices=run_indices, lookup=lookup)


def _values_for_frame_path(
    *,
    path: Path,
    years: list[int],
    base_allocate_args: dict[str, Any],
    lcia_method: str,
    parts: _FrameMatrixParts,
    requested_runs: np.ndarray,
) -> np.ndarray:
    rows = _normalize_rows_core(
        frame=read_external_lca(path),
        path=path,
        lcia_method=lcia_method,
        years=years,
        base_allocate_args=base_allocate_args,
        validate_lcia_inventory=False,
    )
    return _values_for_rows(
        rows=with_lca_ssp_start_year(rows),
        parts=parts,
        requested_runs=requested_runs,
    )


def _values_for_rows(
    *,
    rows: pd.DataFrame,
    parts: _FrameMatrixParts,
    requested_runs: np.ndarray,
) -> np.ndarray:
    requested = np.asarray(requested_runs, dtype=np.int64)
    if requested.size == 0:
        return np.empty((0, len(parts.identity)), dtype=np.float64)
    unique_runs = np.unique(requested)
    selected = rows.loc[rows["run_index"].isin(unique_runs.tolist()), :]
    values = np.full((len(unique_runs), len(parts.identity)), np.nan, dtype=np.float64)
    filled = np.zeros(values.shape, dtype=bool)
    source_runs = selected["run_index"].to_numpy(dtype=np.int64)
    assign_values(
        values=values,
        filled=filled,
        row_positions=np.searchsorted(unique_runs, source_runs),
        column_positions=positions_for_frame(frame=selected, lookup=parts.lookup),
        source_values=selected["value"].to_numpy(dtype=np.float64),
    )
    require_complete_matrix(filled=filled)
    return values[np.searchsorted(unique_runs, requested)]


def _identity_columns(*, frame: pd.DataFrame) -> list[str]:
    base = [
        column
        for column in [
            "year",
            "impact",
            "impact_unit",
            EXT_LCA_SSP_SCENARIO_COLUMN,
            LCA_SSP_START_YEAR_COLUMN,
        ]
        if column in frame.columns
    ]
    selectors = [column for column in SELECTOR_COLUMNS if column in frame.columns]
    excluded = {"run_index", "value", *base, *selectors}
    extras = sorted(column for column in frame.columns if column not in excluded)
    return [*base, *selectors, *extras]


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(pd.Series(frame.loc[:, column], copy=False)))
