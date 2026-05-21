"""Deterministic ASR numerator source discovery and loading."""

from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.external_inputs.lca.deterministic import load_external_lca_deterministic_rows
from pyaesa.external_inputs.lca.paths import external_lca_root
from pyaesa.io_lca.data.contracts import IO_LCA_FAMILY
from pyaesa.io_lca.data.paths import (
    io_metadata_path_for_source,
    lca_results_dir_for_source,
    resolve_io_lca_paths,
)
from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.shared.tabular.contracts import TABULAR_SUFFIX_SET
from pyaesa.shared.tabular.table_io import read_table


def _finalize_lca_rows(lca_rows: pd.DataFrame) -> pd.DataFrame:
    """Return ASR numerator rows in the canonical local shape."""
    out = lca_rows.copy()
    out["year"] = out["year"].astype(int).astype(str)
    out["impact"] = out["impact"].astype(str)
    out["lca_value"] = pd.to_numeric(out["lca_value"], errors="raise")
    out["impact_unit"] = out["impact_unit"].astype(str)
    return out.reset_index(drop=True)


def _discover_io_lca_result_files(
    *,
    source_label: str,
    base_allocate_args: dict[str, Any],
) -> list[Path]:
    """Return persisted deterministic IO-LCA result files for one source scope."""
    paths = resolve_io_lca_paths(
        project_name=str(base_allocate_args["project_name"]),
        group_reg=bool(base_allocate_args["group_reg"]),
        group_sec=bool(base_allocate_args["group_sec"]),
        group_version=base_allocate_args["group_version"],
    )
    results_dir = lca_results_dir_for_source(
        paths=paths,
        source=source_label,
    )
    return sorted(
        path
        for path in results_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in TABULAR_SUFFIX_SET
    )


def _load_io_lca_rows(
    *,
    source_label: str,
    lcia_method: str,
    base_allocate_args: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    result_files = _discover_io_lca_result_files(
        source_label=source_label,
        base_allocate_args=base_allocate_args,
    )
    for path in result_files:
        frame = read_table(path=path)
        subset = frame.loc[frame["lcia_method"].astype(str) == str(lcia_method)].copy()
        if not subset.empty:
            frames.append(subset)
    if not frames:
        paths = resolve_io_lca_paths(
            project_name=str(base_allocate_args["project_name"]),
            group_reg=bool(base_allocate_args["group_reg"]),
            group_sec=bool(base_allocate_args["group_sec"]),
            group_version=base_allocate_args["group_version"],
        )
        results_dir = lca_results_dir_for_source(paths=paths, source=source_label)
        raise FileNotFoundError(
            "deterministic_asr could not find deterministic IO-LCA numerator rows for "
            f"lcia_method='{lcia_method}' in '{results_dir}'. The IO-LCA prerequisite "
            "is resolved by deterministic_asr for the same project, MRIO source, "
            "grouping scope, functional unit, years, and LCIA method; check the "
            "deterministic IO-LCA result scope and skipped method years."
        )
    return _finalize_lca_rows(pd.concat(frames, ignore_index=True))


def _load_contextual_external_rows(
    *,
    proj_base: Path,
    version_name: str,
    lcia_method: str,
    base_allocate_args: dict[str, Any],
    years: list[int],
) -> pd.DataFrame:
    external_dir = external_lca_root(project_base=proj_base)
    templates = external_dir / "templates"
    out, _paths = load_external_lca_deterministic_rows(
        proj_base=proj_base,
        version_name=version_name,
        lcia_method=lcia_method,
        years=years,
        ssp_scenario_options_by_year=None,
        base_allocate_args=base_allocate_args,
    )
    if out is None:
        raise FileNotFoundError(
            "No deterministic external LCA files were found for "
            f"version_name='{version_name}', lcia_method='{lcia_method}' under "
            f"'{external_dir}'. Run prepare_external_inputs(...) to import "
            f"README guidance under '{templates}' and runnable examples under "
            "'deterministic/'. Provide a deterministic file named "
            f"'{version_name}__{lcia_method}.csv' or "
            f"'{version_name}__{lcia_method}__<ssp_scenario>.csv'."
        )
    out = out.rename(columns={"value": "lca_value"})
    return _finalize_lca_rows(out)


def load_lca_rows(
    *,
    proj_base: Path,
    source_label: str,
    lca_type: str,
    lcia_method: str,
    lca_version_name: str | None,
    base_allocate_args: dict[str, Any],
    years: list[int],
) -> pd.DataFrame:
    """Load deterministic ASR numerator rows for one branch."""
    if lca_type == IO_LCA_FAMILY:
        return _load_io_lca_rows(
            source_label=source_label,
            lcia_method=lcia_method,
            base_allocate_args=base_allocate_args,
        )
    return _load_contextual_external_rows(
        proj_base=proj_base,
        version_name=cast(str, lca_version_name),
        lcia_method=lcia_method,
        base_allocate_args=base_allocate_args,
        years=years,
    )


def lca_public_output_root(
    *,
    proj_base: Path,
    source_label: str,
    lca_type: str,
    base_allocate_args: dict[str, Any],
) -> Path:
    """Return the public output folder for the ASR numerator LCA route."""
    if lca_type != IO_LCA_FAMILY:
        return external_lca_root(project_base=proj_base)
    paths = resolve_io_lca_paths(
        project_name=str(base_allocate_args["project_name"]),
        group_reg=bool(base_allocate_args["group_reg"]),
        group_sec=bool(base_allocate_args["group_sec"]),
        group_version=base_allocate_args["group_version"],
    )
    return public_output_root_from_path(
        io_metadata_path_for_source(paths=paths, source=source_label)
    )
