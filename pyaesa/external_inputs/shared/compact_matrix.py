"""Compact external Monte Carlo run matrix reader."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


@dataclass(frozen=True)
class CompactRunMatrix:
    """External compact run matrix aligned to a public row identity."""

    identity: pd.DataFrame
    values: np.ndarray
    run_indices: np.ndarray
    paths: tuple[Path, Path]


@dataclass(frozen=True)
class CompactRunMatrixSource:
    """External compact run matrix source with bounded value access."""

    identity: pd.DataFrame
    run_indices: np.ndarray
    paths: tuple[Path, Path]
    values_for_runs: Callable[[np.ndarray], np.ndarray]


def is_compact_run_matrix_dir(path: Path, *, run_file_name: str) -> bool:
    """Return whether a directory exposes the compact run matrix contract."""
    return (
        path.is_dir()
        and (path / "public_row_identity.csv").is_file()
        and (path / run_file_name).is_file()
    )


def load_compact_run_matrix(
    *,
    directory: Path,
    run_file_name: str,
    context: str,
) -> CompactRunMatrix:
    """Load a compact external Monte Carlo identity and numeric run matrix."""
    source = load_compact_run_matrix_source(
        directory=directory,
        run_file_name=run_file_name,
        context=context,
    )
    return CompactRunMatrix(
        identity=source.identity,
        values=source.values_for_runs(source.run_indices),
        run_indices=source.run_indices,
        paths=source.paths,
    )


def load_compact_run_matrix_source(
    *,
    directory: Path,
    run_file_name: str,
    context: str,
) -> CompactRunMatrixSource:
    """Load compact external Monte Carlo identity and bounded value access."""
    identity_path = directory / "public_row_identity.csv"
    runs_path = directory / run_file_name
    identity = pd.read_csv(identity_path)
    _validate_identity(identity=identity, context=context)
    run_columns = list(pd.read_csv(runs_path, nrows=0).columns)
    _validate_run_columns(columns=run_columns, identity=identity, context=context)
    table = pacsv.read_csv(
        runs_path,
        convert_options=pacsv.ConvertOptions(include_columns=["run_index"]),
    )
    run_indices = _int64_column(table["run_index"])
    expected_runs = np.arange(len(run_indices), dtype=np.int64)
    if not np.array_equal(run_indices, expected_runs):
        raise ValueError(
            f"{context} compact run matrix run_index values must start at 0 and be contiguous."
        )
    public_row_ids = identity["public_row_id"].to_numpy(dtype=np.int64)
    return CompactRunMatrixSource(
        identity=identity.reset_index(drop=True),
        run_indices=run_indices,
        paths=(identity_path, runs_path),
        values_for_runs=lambda requested: compact_run_matrix_values_for_runs(
            runs_path=runs_path,
            public_row_ids=public_row_ids,
            run_indices=run_indices,
            requested_runs=np.asarray(requested, dtype=np.int64),
        ),
    )


def compact_run_matrix_values_for_runs(
    *,
    runs_path: Path,
    public_row_ids: np.ndarray,
    run_indices: np.ndarray,
    requested_runs: np.ndarray,
) -> np.ndarray:
    """Return selected compact run matrix values without loading every run."""
    requested = np.asarray(requested_runs, dtype=np.int64)
    if requested.size == 0:
        return np.empty((0, len(public_row_ids)), dtype=np.float64)
    unique_runs = np.unique(requested)
    _validate_requested_runs(
        run_indices=run_indices,
        requested_runs=unique_runs,
        context=f"Compact run matrix '{runs_path}'",
    )
    start = int(unique_runs[0])
    stop = int(unique_runs[-1]) + 1
    column_names = [str(int(public_row_id)) for public_row_id in public_row_ids]

    def skip_prefix(row_number: object) -> bool:
        return 0 < int(str(row_number)) <= start

    frame = pd.read_csv(
        runs_path,
        usecols=["run_index", *column_names],
        skiprows=skip_prefix,
        nrows=stop - start,
    )
    source_runs = frame["run_index"].to_numpy(dtype=np.int64)
    source_values = frame.loc[:, column_names].to_numpy(dtype=np.float64)
    unique_positions = np.searchsorted(source_runs, unique_runs)
    inside = unique_positions < len(source_runs)
    if not bool(inside.all()) or not np.array_equal(
        source_runs[unique_positions[inside]],
        unique_runs[inside],
    ):
        raise ValueError(f"Compact run matrix '{runs_path}' is missing requested run rows.")
    unique_values = source_values[unique_positions]
    return unique_values[np.searchsorted(unique_runs, requested)]


def _validate_identity(*, identity: pd.DataFrame, context: str) -> None:
    if "public_row_id" not in identity.columns:
        raise ValueError(f"{context} compact public_row_identity.csv must include public_row_id.")
    public_ids = identity["public_row_id"].to_numpy(dtype=np.int64)
    expected = np.arange(len(public_ids), dtype=np.int64)
    if not np.array_equal(public_ids, expected):
        raise ValueError(
            f"{context} compact public_row_id values must start at 0 and be contiguous."
        )


def _validate_run_columns(*, columns: list[str], identity: pd.DataFrame, context: str) -> None:
    expected = ["run_index", *[str(int(value)) for value in identity["public_row_id"].tolist()]]
    if columns != expected:
        raise ValueError(
            f"{context} compact run matrix columns must be {expected}. Observed={columns}."
        )


def _int64_column(column: pa.ChunkedArray) -> np.ndarray:
    return np.asarray(pc.cast(column, pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64)


def _validate_requested_runs(
    *,
    run_indices: np.ndarray,
    requested_runs: np.ndarray,
    context: str,
) -> None:
    if int(requested_runs[0]) >= 0 and int(requested_runs[-1]) < len(run_indices):
        return
    missing = requested_runs[(requested_runs < 0) | (requested_runs >= len(run_indices))]
    missing_values = missing.astype(int).tolist()
    raise ValueError(f"{context} is missing requested run_index values {missing_values}.")
