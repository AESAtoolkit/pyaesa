"""Workflow helpers for allocation validation orchestration."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pyaesa.asocc.runtime.scope.branch_resolution import (
    allocate_run_metadata_path,
    asocc_l1_dir,
    asocc_l2_dir,
    outputs_project_root,
    resolve_allocate_path_scope,
)
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.asocc.runtime.request.scope import build_asocc_scope


def resolve_lcia_methods(
    value: str | tuple[str, ...] | list[str],
) -> str | list[str]:
    """Return deterministic_asocc compatible LCIA methods from config."""
    if isinstance(value, str):
        return value
    return list(value)


def _slugify(text: str) -> str:
    """Return one filesystem safe token for validation project names."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    return cleaned.strip("_") or "value"


def validation_project_name(
    *,
    base_project_name: str,
    source: str,
    fu_code: str,
    l1_reg_aggreg: str,
) -> str:
    """Return one deterministic validation project name for a source/FU/mode target."""
    return "__".join(
        (
            str(base_project_name).strip(),
            _slugify(source),
            _slugify(fu_code),
            f"l1_{_slugify(l1_reg_aggreg)}",
        )
    )


def _scope_compatible_args(base_allocate_args: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic scope subset accepted by current scope helpers."""
    keys = (
        "project_name",
        "source",
        "agg_reg",
        "agg_sec",
        "agg_version",
        "years",
        "fu_code",
        "r_p",
        "s_p",
        "r_c",
        "r_f",
        "group_indices",
        "method_plan",
        "l1_methods",
        "one_step_methods",
        "two_step_methods",
        "l1_l2_pairs",
        "l1_reg_aggreg",
        "lcia_method",
        "reference_years",
        "ssp_scenario",
        "projection_mode",
        "reg_window",
        "reuse_years",
    )
    return {key: base_allocate_args.get(key) for key in keys}


@dataclass(frozen=True)
class ValidationOutputPaths:
    """Published deterministic output locations for one validation target."""

    project_name: str
    project_root: Path
    metadata_path: Path
    l1_share_dir: Path
    l2_share_root: Path
    bucket_dirs: dict[str, Path]


def resolve_validation_output_paths(
    *,
    base_allocate_args: Mapping[str, Any],
    buckets: Sequence[str],
) -> ValidationOutputPaths:
    """Resolve current deterministic output paths for one validation request."""
    normalized_scope = build_asocc_scope(
        base_allocate_args=normalize_base_allocate_args(_scope_compatible_args(base_allocate_args))
    )
    path_scope = resolve_allocate_path_scope(base_allocate_args=normalized_scope.base_allocate_args)
    l2_in_l1_dir = asocc_l2_dir(
        scope=path_scope,
        bucket="l2_in_l1",
        lcia_sub=None,
    )
    return ValidationOutputPaths(
        project_name=str(normalized_scope.base_allocate_args["project_name"]),
        project_root=outputs_project_root(
            project_name=str(normalized_scope.base_allocate_args["project_name"])
        ),
        metadata_path=allocate_run_metadata_path(scope=path_scope),
        l1_share_dir=asocc_l1_dir(
            scope=path_scope,
            lcia_sub=None,
        ),
        l2_share_root=l2_in_l1_dir.parent,
        bucket_dirs={
            str(bucket): asocc_l2_dir(
                scope=path_scope,
                bucket=str(bucket),
                lcia_sub=None,
            )
            for bucket in buckets
        },
    )
