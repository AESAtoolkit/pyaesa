"""aSoCC uncertainty scientific source method logs."""

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from pyaesa.shared.runtime.io.filesystem import write_via_atomic_temp

ASOCC_SOURCE_METHOD_COLUMNS: tuple[str, ...] = (
    "source_component",
    "source_name",
    "scope",
    "applied_bucket",
    "allocation_method",
    "lcia_method",
    "impact_categories",
    "year_min",
    "year_max",
    "primary_cov_kind",
    "primary_cov_key",
    "primary_cov_value",
    "reference_cov_kind",
    "reference_cov_key",
    "reference_cov_value",
    "distribution",
    "shared_random_variable",
    "formula",
    "notes",
)


@dataclass(frozen=True)
class SourceMethodRow:
    """One compact aSoCC source method explanation row."""

    source_component: str
    source_name: str
    formula: str
    scope: str | None = None
    applied_bucket: str | None = None
    allocation_method: str | None = None
    lcia_method: str | None = None
    impact_categories: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    primary_cov_kind: str | None = None
    primary_cov_key: str | None = None
    primary_cov_value: float | None = None
    reference_cov_kind: str | None = None
    reference_cov_key: str | None = None
    reference_cov_value: float | None = None
    distribution: str | None = None
    shared_random_variable: str | None = None
    notes: str | None = None


def source_methods_frame(*, rows: list[SourceMethodRow]) -> pd.DataFrame:
    """Return aSoCC source method rows with canonical columns."""
    if not rows:
        return pd.DataFrame(columns=ASOCC_SOURCE_METHOD_COLUMNS)
    frame = pd.DataFrame([asdict(row) for row in rows])
    return frame.loc[:, list(ASOCC_SOURCE_METHOD_COLUMNS)]


def write_source_methods(*, path: Path, rows: list[SourceMethodRow]) -> None:
    """Write aSoCC source method rows as CSV."""
    frame = source_methods_frame(rows=rows)
    write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_csv(tmp_path, index=False))
