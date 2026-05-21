"""Deterministic path and filename resolution for IO-LCA workflows."""

import re
from dataclasses import dataclass
from pathlib import Path

from pyaesa.workspace_initialisation.workspace import (
    project_outputs_root,
)
from pyaesa.shared.runtime.io.family_root_names import LCA_ROOT_DIRNAME
from pyaesa.shared.runtime.metadata.contracts import (
    FIGURE_MANIFEST_FILENAME,
    SCOPE_MANIFEST_FILENAME,
)
from pyaesa.shared.lcia.file_owned_tables import (
    expected_lcia_method_table_paths,
    lcia_method_from_table_path,
    lcia_method_partition_path,
    resolved_lcia_method_table_paths,
)

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_piece(value: str) -> str:
    """Return deterministic filename safe token."""
    text = _SAFE_CHARS_RE.sub("_", str(value).strip())
    text = text.strip("._-")
    return text or "item"


@dataclass(frozen=True)
class IOLCAPaths:
    """Resolved output and metadata paths for one IO-LCA scope."""

    project_base: Path
    lca_root: Path
    source_version_token: str


def resolve_io_lca_paths(
    *,
    project_name: str,
    group_reg: bool,
    group_sec: bool,
    group_version: str | None,
) -> IOLCAPaths:
    """Resolve deterministic project paths for IO-LCA and figures."""
    grouped = bool(group_reg or group_sec)
    project_base = project_outputs_root(project_name=project_name)
    source_version_token = (
        _sanitize_piece(str(group_version))
        if grouped and str(group_version).strip()
        else "original_version"
    )
    return IOLCAPaths(
        project_base=project_base,
        lca_root=project_base / LCA_ROOT_DIRNAME / "io_lca",
        source_version_token=source_version_token,
    )


def _lcia_method_tag(lcia_method: str) -> str:
    """Return deterministic LCIA method tag used in paths."""
    return _sanitize_piece(lcia_method)


def _source_tag(source: str) -> str:
    """Return deterministic source tag used in paths."""
    return _sanitize_piece(source)


def _source_scope_root(*, paths: IOLCAPaths, source: str) -> Path:
    """Return source scoped IO-LCA root under ``io_lca``."""
    return paths.lca_root / f"{_source_tag(source)}__{paths.source_version_token}"


def source_scope_root_for_source(*, paths: IOLCAPaths, source: str) -> Path:
    """Return the canonical native IO-LCA family root for one source/version scope."""
    return _source_scope_root(paths=paths, source=source)


def deterministic_scope_metadata_paths(*, paths: IOLCAPaths) -> list[Path]:
    """Return all persisted deterministic IO-LCA scope metadata paths in one project tree."""
    if not paths.lca_root.exists():
        return []
    return sorted(
        metadata_path
        for metadata_path in paths.lca_root.glob(f"*/deterministic/logs/{SCOPE_MANIFEST_FILENAME}")
        if metadata_path.is_file()
    )


def _deterministic_scope_root(*, paths: IOLCAPaths, source: str) -> Path:
    """Return source scoped deterministic IO-LCA root."""
    return _source_scope_root(paths=paths, source=source) / "deterministic"


def lca_results_dir_for_source(*, paths: IOLCAPaths, source: str) -> Path:
    """Return source scoped main results folder."""
    return _deterministic_scope_root(paths=paths, source=source) / "results"


def origin_dir_for_source(*, paths: IOLCAPaths, source: str) -> Path:
    """Return source scoped upstream origin folder."""
    return _deterministic_scope_root(paths=paths, source=source) / "results" / "origin"


def stages_dir_for_source(*, paths: IOLCAPaths, source: str) -> Path:
    """Return source scoped upstream stage folder."""
    return _deterministic_scope_root(paths=paths, source=source) / "results" / "stages"


def log_dir_for_source(*, paths: IOLCAPaths, source: str) -> Path:
    """Return source scoped IO-LCA logs folder."""
    return _deterministic_scope_root(paths=paths, source=source) / "logs"


def figures_dir_for_source(*, paths: IOLCAPaths, source: str) -> Path:
    """Return source scoped IO-LCA figures folder."""
    return _deterministic_scope_root(paths=paths, source=source) / "figures"


def origin_columns_defs_path(*, paths: IOLCAPaths, source: str) -> Path:
    """Return shared upstream origin column definitions text path."""
    return origin_dir_for_source(paths=paths, source=source) / "columns_defs.txt"


def stage_columns_defs_path(*, paths: IOLCAPaths, source: str) -> Path:
    """Return shared upstream stage column definitions text path."""
    return stages_dir_for_source(paths=paths, source=source) / "columns_defs.txt"


def main_results_path(
    *,
    paths: IOLCAPaths,
    source: str,
    lcia_method: str,
    extension: str,
) -> Path:
    """Return deterministic main results file path."""
    out_dir = lca_results_dir_for_source(
        paths=paths,
        source=source,
    )
    return out_dir / f"{_lcia_method_tag(lcia_method)}.{extension}"


def origin_results_path(
    *,
    paths: IOLCAPaths,
    source: str,
    lcia_method: str,
    extension: str,
) -> Path:
    """Return deterministic upstream origin file path."""
    out_dir = origin_dir_for_source(
        paths=paths,
        source=source,
    )
    return out_dir / f"origins__{_lcia_method_tag(lcia_method)}.{extension}"


def origin_ratio_results_path(
    *,
    paths: IOLCAPaths,
    source: str,
    lcia_method: str,
    extension: str,
) -> Path:
    """Return deterministic upstream origin ratio file path."""
    out_dir = origin_dir_for_source(
        paths=paths,
        source=source,
    )
    return out_dir / f"origins_ratio__{_lcia_method_tag(lcia_method)}.{extension}"


def stage_results_path(
    *,
    paths: IOLCAPaths,
    source: str,
    lcia_method: str,
    year: int,
    extension: str,
) -> Path:
    """Return deterministic staged upstream file path for one LCIA method/year."""
    out_dir = stages_dir_for_source(
        paths=paths,
        source=source,
    )
    return out_dir / f"stages__{_lcia_method_tag(lcia_method)}__{int(year)}.{extension}"


def io_metadata_path_for_source(*, paths: IOLCAPaths, source: str) -> Path:
    """Return source scoped deterministic IO-LCA scope manifest path."""
    return log_dir_for_source(paths=paths, source=source) / SCOPE_MANIFEST_FILENAME


def figure_metadata_path_for_source(*, paths: IOLCAPaths, source: str) -> Path:
    """Return source scoped deterministic IO-LCA figure metadata path."""
    return log_dir_for_source(paths=paths, source=source) / FIGURE_MANIFEST_FILENAME


def io_lca_expected_method_table_paths(*, base_path: Path, lcia_methods: list[str]) -> list[Path]:
    """Return the canonical expected IO-LCA partition paths for one method set."""
    return expected_lcia_method_table_paths(base_path=base_path, lcia_methods=lcia_methods)


def io_lca_lcia_method_from_path(*, path: Path, file_stem: str) -> str | None:
    """Return the canonical LCIA method token from one partitioned IO-LCA path."""
    return lcia_method_from_table_path(path=path, file_stem=file_stem)


def io_lca_method_table_path(*, base_path: Path, lcia_method: str) -> Path:
    """Return the canonical method partitioned IO-LCA table path."""
    return lcia_method_partition_path(base_path=base_path, lcia_method=lcia_method)


def io_lca_method_table_paths(*, base_path: Path) -> list[Path]:
    """Return the existing method partitioned IO-LCA table paths under one base path."""
    return resolved_lcia_method_table_paths(base_path=base_path)
