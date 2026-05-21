"""LCIA method filename ownership helpers."""

from pathlib import Path
import re

from pyaesa.shared.tabular.table_io import partitioned_output_paths

_SCENARIO_SUFFIX_RE = re.compile(r"__ssp\d+$")


def lcia_method_partition_path(*, base_path: Path, lcia_method: str | None) -> Path:
    """Return the logical table path for one file owned LCIA method."""
    if lcia_method is None or not str(lcia_method).strip():
        return base_path
    token = str(lcia_method).strip()
    return base_path.parent / f"{base_path.stem}__{token}{base_path.suffix}"


def lcia_method_from_table_path(*, path: Path, file_stem: str) -> str | None:
    """Return the LCIA method token encoded in one table path."""
    stem = _SCENARIO_SUFFIX_RE.sub("", path.stem)
    if stem == file_stem:
        return None
    prefix = f"{file_stem}__"
    if not stem.startswith(prefix):
        raise ValueError(f"Table '{path}' does not match LCIA owned stem '{file_stem}'.")
    token = stem[len(prefix) :].strip()
    return token or None


def expected_lcia_method_table_paths(*, base_path: Path, lcia_methods: list[str]) -> list[Path]:
    """Return expected logical table paths for resolved LCIA methods."""
    methods = sorted({str(value).strip() for value in lcia_methods if str(value).strip()})
    return [
        lcia_method_partition_path(base_path=base_path, lcia_method=method) for method in methods
    ]


def resolved_lcia_method_table_paths(*, base_path: Path) -> list[Path]:
    """Return persisted concrete table paths for all LCIA method partitions."""
    logical_paths = {base_path}
    prefix = f"{base_path.stem}__"
    for path in sorted(base_path.parent.glob(f"{base_path.stem}*{base_path.suffix}")):
        stem = _SCENARIO_SUFFIX_RE.sub("", path.stem)
        if stem == base_path.stem or stem.startswith(prefix):
            method = lcia_method_from_table_path(path=path, file_stem=base_path.stem)
            logical_paths.add(lcia_method_partition_path(base_path=base_path, lcia_method=method))
    paths: list[Path] = []
    for logical_path in sorted(logical_paths):
        paths.extend(partitioned_output_paths(base_path=logical_path))
    return paths
