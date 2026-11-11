"""File identity payloads for manifest compatibility keys."""

from pathlib import Path

import pandas as pd

from pyaesa.shared.runtime.manifest_contract import manifest_digest
from pyaesa.shared.tabular.table_io import read_table


def table_sha256(frame: pd.DataFrame, *, labels: tuple[str, ...], numbers: tuple[str, ...]) -> str:
    """Fingerprint calculation fields, independently of row order and table formatting."""
    selected = frame.loc[:, list(labels)].astype("string")
    for column in numbers:
        selected[column] = (
            pd.Series(pd.to_numeric(frame[column], errors="raise"), dtype=float) + 0.0
        )
    selected = selected.sort_values(list(selected.columns)).astype(object)
    return manifest_digest(
        {
            "columns": list(selected.columns),
            "rows": selected.where(selected.notna(), None).values.tolist(),
        }
    )


def file_identity_payload(
    *, path: Path, labels: tuple[str, ...], numbers: tuple[str, ...]
) -> dict[str, object]:
    """Fingerprint explicit calculation columns, excluding file location and annotations."""
    frame = read_table(path=path).reindex(columns=[*labels, *numbers])
    return {
        "sha256": table_sha256(frame, labels=labels, numbers=numbers),
        "input_provenance": {"path": str(path)},
    }
