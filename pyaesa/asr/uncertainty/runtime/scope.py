"""Public request scope normalization for uncertainty ASR runs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.asocc.runtime.scope.branch_resolution import resolve_allocate_project_base
from pyaesa.asr.shared.lca.request import (
    normalize_lca_args,
    selected_lca_type,
    selected_lca_version_name,
)
from pyaesa.asr.uncertainty.io.paths import asr_monte_carlo_root
from pyaesa.external_inputs.asocc.schema.contracts import normalize_external_method_selector
from pyaesa.shared.acc_asr_common.branches.config import normalize_base_cc_args
from pyaesa.shared.acc_asr_common.branches.expand import iter_cc_method_branches
from pyaesa.shared.acc_asr_common.scope.composite import (
    base_asocc_kwargs_from_allocate_args,
    build_composite_base_allocate_args,
    normalize_base_asocc_args,
    normalize_mrio_scope,
    normalize_shared_lcia_methods,
)


@dataclass(frozen=True)
class ASRUncertaintyScope:
    """Normalized public scope for one uncertainty ASR call."""

    shared_methods: list[str]
    mrio_scope: dict[str, Any]
    asocc_config: dict[str, Any]
    cc_config: dict[str, Any]
    lca_config: dict[str, Any]
    lca_type: str
    lca_version_name: str | None
    external_method: dict[str, Any] | None
    base_allocate_args: dict[str, Any]
    proj_base: Path
    source_label: str
    root: Path
    branches: list[dict[str, Any]]
    base_args: dict[str, Any]


def build_asr_uncertainty_scope(
    *,
    project_name: str,
    source: str,
    group_reg: bool,
    group_sec: bool,
    group_version: str,
    years: int | list[int] | range,
    fu_code: str,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
    r_c: str | list[str] | None,
    r_f: str | list[str] | None,
    aggreg_indices: bool,
    lcia_method: str | list[str],
    base_asocc_args: dict[str, Any] | None,
    external_method: dict[str, Any] | None,
    base_cc_args: dict[str, Any],
    lca_args: dict[str, Any],
    sobol_enabled: bool,
) -> ASRUncertaintyScope:
    """Normalize public ASR arguments and derive component scope paths."""
    shared_methods = normalize_shared_lcia_methods(lcia_method)
    mrio_scope = normalize_mrio_scope(
        source=source,
        group_reg=group_reg,
        group_sec=group_sec,
        group_version=group_version,
        aggreg_indices=aggreg_indices,
    )
    asocc_config = normalize_base_asocc_args(base_asocc_args, fu_code=fu_code)
    cc_config = normalize_base_cc_args(base_cc_args)
    lca_config = normalize_lca_args(lca_args)
    external_method_norm = normalize_external_method_selector(
        external_method,
        fu_code=fu_code,
        argument_name="external_method",
    )
    base_allocate_args = build_composite_base_allocate_args(
        project_name=project_name,
        years=years,
        lcia_method=shared_methods,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        source=mrio_scope["source"],
        group_reg=mrio_scope["group_reg"],
        group_sec=mrio_scope["group_sec"],
        group_version=mrio_scope["group_version"],
        aggreg_indices=mrio_scope["aggreg_indices"],
        base_asocc_args=asocc_config,
    )
    proj_base = resolve_allocate_project_base(
        base_allocate_args=normalize_base_allocate_args(
            base_asocc_kwargs_from_allocate_args(base_allocate_args=base_allocate_args)
        )
    )
    lca_type = selected_lca_type(lca_args=lca_config)
    lca_version_name = selected_lca_version_name(lca_args=lca_config)
    lca_route = lca_type if lca_version_name is None else f"{lca_type} ({lca_version_name})"
    source_label = str(base_allocate_args["source"])
    branches = (
        iter_cc_method_branches(
            lcia_methods=shared_methods,
            base_cc_args=cc_config,
            years=years,
        )
        if sobol_enabled
        else []
    )
    base_args = {
        "project_name": project_name,
        "years": years,
        "lcia_method": shared_methods,
        "fu_code": fu_code,
        "r_p": r_p,
        "s_p": s_p,
        "r_c": r_c,
        "r_f": r_f,
        "source": mrio_scope["source"],
        "group_reg": mrio_scope["group_reg"],
        "group_sec": mrio_scope["group_sec"],
        "group_version": mrio_scope["group_version"],
        "aggreg_indices": mrio_scope["aggreg_indices"],
        "base_asocc_args": asocc_config,
        "base_cc_args": cc_config,
        "lca_args": lca_config,
        "lca_route": lca_route,
        "external_method": external_method_norm,
    }
    return ASRUncertaintyScope(
        shared_methods=shared_methods,
        mrio_scope=mrio_scope,
        asocc_config=asocc_config,
        cc_config=cc_config,
        lca_config=lca_config,
        lca_type=lca_type,
        lca_version_name=lca_version_name,
        external_method=external_method_norm,
        base_allocate_args=base_allocate_args,
        proj_base=proj_base,
        source_label=source_label,
        root=asr_monte_carlo_root(
            proj_base=proj_base,
            source_label=source_label,
            group_version=base_allocate_args["group_version"],
            lca_type=lca_type,
            lca_version_name=lca_version_name,
        ),
        branches=branches,
        base_args=base_args,
    )
