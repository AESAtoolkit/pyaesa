"""Public request normalization for IO-LCA uncertainty."""

from typing import Any, Mapping

from pyaesa.io_lca.contracts.fu_mapping import resolve_fu_spec
from pyaesa.io_lca.orchestration.request.domain_checks import (
    validate_group_indices_requires_multi_selection,
    validate_group_indices_supported,
)
from pyaesa.io_lca.orchestration.request.selectors import (
    has_multi_selected_indices,
    resolve_selectors,
    validate_selector_labels,
)
from pyaesa.io_lca.orchestration.request.validation import (
    normalize_aggregation,
    normalize_lcia_method_list,
    normalize_supported_source,
)
from pyaesa.io_lca.orchestration.request.year_resolution import resolve_years_strict
from pyaesa.shared.lcia.cov_inputs import normalize_lcia_uncertainty_parameters

from pyaesa.io_lca.uncertainty.runtime.models import IOLCAUncertaintyRequest

_ALLOWED_BASE_KEYS = {
    "project_name",
    "source",
    "agg_reg",
    "agg_sec",
    "agg_version",
    "years",
    "lcia_method",
    "fu_code",
    "r_f",
    "r_c",
    "r_p",
    "s_p",
    "group_indices",
}


def normalize_io_lca_uncertainty_request(
    *,
    base_io_lca_args: Mapping[str, Any],
    lcia_parameters: Mapping[str, Any],
) -> IOLCAUncertaintyRequest:
    """Normalize the IO-LCA scientific request and LCIA source parameters."""
    payload = dict(base_io_lca_args)
    unknown = sorted(set(payload) - _ALLOWED_BASE_KEYS)
    if unknown:
        raise ValueError(f"Unsupported base_io_lca_args keys for uncertainty_io_lca: {unknown}.")
    for required in ("project_name", "source", "lcia_method", "fu_code"):
        if required not in payload:
            raise ValueError(f"base_io_lca_args.{required} is required.")
    source = normalize_supported_source(
        source=str(payload["source"]),
        caller="uncertainty_io_lca",
    )
    agg_reg = _bool_value(payload.get("agg_reg", False), field="base_io_lca_args.agg_reg")
    agg_sec = _bool_value(payload.get("agg_sec", False), field="base_io_lca_args.agg_sec")
    agg_reg, agg_sec, agg_version = normalize_aggregation(
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=payload.get("agg_version"),
    )
    methods = normalize_lcia_method_list(lcia_method=payload["lcia_method"])
    fu_spec = resolve_fu_spec(fu_code=str(payload["fu_code"]))
    filters, _studied_indices_tag = resolve_selectors(
        spec=fu_spec,
        r_f=payload.get("r_f"),
        r_c=payload.get("r_c"),
        r_p=payload.get("r_p"),
        s_p=payload.get("s_p"),
    )
    validate_selector_labels(
        source=source,
        agg_version=agg_version,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        filters=filters,
    )
    years = resolve_years_strict(
        years=payload.get("years"),
        source=source,
        agg_version=agg_version,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        upstream_analysis=False,
    )
    group_indices = _bool_value(
        payload.get("group_indices", False),
        field="base_io_lca_args.group_indices",
    )
    validate_group_indices_requires_multi_selection(
        group_indices=group_indices,
        has_multi_indices=has_multi_selected_indices(filters),
    )
    validate_group_indices_supported(spec=fu_spec, group_indices=group_indices)
    source_parameters = normalize_lcia_uncertainty_parameters(parameters=lcia_parameters)
    project_name = _non_empty_text(payload["project_name"], field="base_io_lca_args.project_name")
    base_args = {
        "project_name": project_name,
        "source": source,
        "agg_reg": agg_reg,
        "agg_sec": agg_sec,
        "agg_version": agg_version,
        "years": list(years),
        "lcia_method": list(methods),
        "fu_code": fu_spec.fu_code,
        "r_f": filters.get("r_f"),
        "r_c": filters.get("r_c"),
        "r_p": filters.get("r_p"),
        "s_p": filters.get("s_p"),
        "group_indices": group_indices,
    }
    deterministic_args = {
        **base_args,
        "upstream_analysis": False,
        "output_format": "csv",
        "figures": False,
    }
    return IOLCAUncertaintyRequest(
        base_io_lca_args=base_args,
        deterministic_args=deterministic_args,
        source_parameters=source_parameters,
        project_name=project_name,
        source=source,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
        years=list(years),
        lcia_methods=list(methods),
        fu_spec=fu_spec,
        filters=filters,
        group_indices=group_indices,
    )


def _bool_value(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean.")
    return value


def _non_empty_text(value: object, *, field: str) -> str:
    text = str(value).strip()
    if not text or text == "None":
        raise ValueError(f"{field} must be a non empty string.")
    return text
