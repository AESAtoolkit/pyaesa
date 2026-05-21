"""Deterministic ASR figure state ownership."""

from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow.parquet as pq

from pyaesa.asr.deterministic.state.metadata import load_run_metadata
from pyaesa.asr.shared.runtime.paths import ASRDeterministicPathContext, get_asr_meta_path
from pyaesa.shared.figures.persisted_outputs import delete_persisted_figure_paths
from pyaesa.shared.tabular.wide_tables import resolve_single_allocation_method_identity


def l1_l2_methods_by_path(paths: list[Path], *, family_label: str) -> dict[Path, str]:
    """Return canonical allocation identity from persisted deterministic ASR tables."""
    out: dict[Path, str] = {}
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path, nrows=1)
        elif suffix == ".parquet":
            frame = _first_parquet_row(path=path)
        else:
            frame = cast(pd.DataFrame, pd.read_pickle(path)).head(1)
        out[path] = resolve_single_allocation_method_identity(
            frame,
            where=f"{family_label} figure input '{path.name}'",
        )
    return out


def _first_parquet_row(*, path: Path) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    batch = next(parquet.iter_batches(batch_size=1))
    return batch.to_pandas()


def clear_persisted_asr_figures(*, path_context: ASRDeterministicPathContext) -> None:
    """Delete deterministic ASR figure files recorded in branch metadata."""
    meta_path = get_asr_meta_path(context=path_context)
    if not meta_path.exists():
        return
    payload = load_run_metadata(meta_path)
    delete_persisted_figure_paths(
        raw_paths=payload["artifacts"].get("figure_paths"),
    )
