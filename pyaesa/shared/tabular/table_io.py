"""Small shared table IO helpers for package generated tabular artifacts."""

from pathlib import Path
from typing import cast

import pandas as pd

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent, write_via_atomic_temp
from pyaesa.shared.runtime.scenario.partitions import (
    scenario_partition_glob_pattern,
    scenario_partition_token_from_path,
)
from pyaesa.shared.tabular.contracts import TABULAR_SUFFIX_SET


def read_table(*, path: Path) -> pd.DataFrame:
    """Read one CSV, Parquet, or pickle table."""
    suffix = path.suffix.lower()
    if suffix not in TABULAR_SUFFIX_SET:
        raise ValueError(
            f"Unsupported table format '{path.suffix}' for {path}. "
            f"Use one of: {sorted(TABULAR_SUFFIX_SET)}."
        )
    if not path.exists():
        return pd.DataFrame()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        frame.attrs["source_path"] = str(path)
        return frame
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        frame.attrs["source_path"] = str(path)
        return frame
    frame = cast(pd.DataFrame, pd.read_pickle(path))
    frame.attrs["source_path"] = str(path)
    return frame


def write_table(*, path: Path, frame: pd.DataFrame) -> None:
    """Write one CSV, Parquet, or pickle table."""
    path = ensure_file_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_csv(tmp_path, index=False))
        return
    if suffix == ".parquet":
        write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_parquet(tmp_path, index=False))
        return
    if suffix == ".pickle":
        write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_pickle(tmp_path))
        return
    raise ValueError(
        f"Unsupported table format '{path.suffix}' for {path}. "
        f"Use one of: {sorted(TABULAR_SUFFIX_SET)}."
    )


def partitioned_output_paths(*, base_path: Path) -> list[Path]:
    """Return the existing concrete paths for one logical partitioned table."""
    partition_paths = sorted(
        base_path.parent.glob(scenario_partition_glob_pattern(base_path=base_path))
    )
    candidates = [base_path, *partition_paths]
    ordered: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        if (
            path != base_path
            and scenario_partition_token_from_path(base_path=base_path, path=path) is None
        ):
            continue
        seen.add(path)
        ordered.append(path)
    return ordered
