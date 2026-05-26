"""Helpers for user visible composite persisted payloads."""

from typing import Any, cast

from pyaesa.shared.selectors.time_selectors import normalize_time_selector_mapping


def build_public_cc_branch_args(*, branch: dict[str, Any]) -> dict[str, Any]:
    """Build the public CC family block for one composite branch."""
    if branch["cc_type"] == "static":
        bounds = list(branch["static_cc_bounds"])
        return {"static": {"exclude_max_cc": bounds == ["min_cc"]}}
    return {
        "static": {"active": False},
        "dynamic_ar6": {
            "harmonization": cast(bool, branch["harmonization"]),
            "harmonization_method": str(branch["harmonization_method"]),
            "category": branch["category"],
            "ssp_scenario": branch["ssp_scenario"],
            "emission_type": str(branch["emission_type"]),
            "include_afolu": cast(bool, branch["include_afolu"]),
            "emissions_mode": str(branch["emissions_mode"]),
            "subset_version": branch["subset_version"],
        },
    }


def build_public_composite_request_payload(
    *,
    project_name: str,
    years: int | list[int] | range,
    lcia_method: list[str],
    fu_code: str,
    r_p: str | list[str] | None,
    s_p: str | list[str] | None,
    r_c: str | list[str] | None,
    r_f: str | list[str] | None,
    source: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
    group_indices: bool,
    base_asocc_args: dict[str, Any],
    base_cc_args: dict[str, Any] | None = None,
    lca_args: dict[str, Any] | None = None,
    external_method: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public persisted composite payload for aCC/ASR runs."""
    return cast(
        dict[str, Any],
        normalize_time_selector_mapping(
            {
                "project_name": str(project_name).strip(),
                "years": years,
                "lcia_method": list(lcia_method),
                "fu_code": str(fu_code).strip(),
                "r_p": r_p,
                "s_p": s_p,
                "r_c": r_c,
                "r_f": r_f,
                "source": source,
                "agg_reg": agg_reg,
                "agg_sec": agg_sec,
                "agg_version": agg_version,
                "group_indices": group_indices,
                "base_asocc_args": normalize_time_selector_mapping(dict(base_asocc_args)),
                "base_cc_args": None if base_cc_args is None else dict(base_cc_args),
                "lca_args": None if lca_args is None else dict(lca_args),
                "external_method": None if external_method is None else dict(external_method),
            }
        ),
    )
