"""LCIA status and unit map ownership for MRIO processing."""

from typing import Any, Sequence, cast

import pandas as pd

from pyaesa.process.mrios.utils.parsers.exio_parser import (
    ExioCharacterizationOptions,
)
from pyaesa.process.mrios.utils.parsers.exio_characterization import (
    _OPTIONAL_EXTENSIONS,
    _normalize_label,
)


def normalize_lcia_method_list(method_specs: Sequence[str] | None) -> list[str]:
    """Return cleaned LCIA method names preserving order."""
    if method_specs is None:
        return []
    cleaned: list[str] = []
    for method_spec in method_specs:
        value = str(method_spec).strip()
        if value:
            cleaned.append(value)
    return cleaned


def merge_lcia_method_lists(*method_spec_lists: Sequence[str] | None) -> list[str]:
    """Return deduplicated LCIA methods preserving first seen order."""
    merged: list[str] = []
    seen: set[str] = set()
    for method_specs in method_spec_lists:
        for lcia_method in normalize_lcia_method_list(method_specs):
            if lcia_method in seen:
                continue
            seen.add(lcia_method)
            merged.append(lcia_method)
    return merged


def year_entry_lcia_methods(year_meta: dict[str, Any] | None) -> list[str]:
    """Return LCIA methods declared in one metadata year entry."""
    if year_meta is None:
        return []
    enacting_metric_meta = cast(dict[str, Any], year_meta.get("enacting_metrics", {}))
    method_specs = enacting_metric_meta.get("lcia_methods")
    if method_specs is None:
        return []
    return normalize_lcia_method_list(cast(list[str], method_specs))


def year_entry_unavailable_lcia_methods(year_meta: dict[str, Any] | None) -> set[str]:
    """Return LCIA methods recorded as unavailable in one metadata year entry."""
    if year_meta is None:
        return set()
    raw_status = cast(dict[str, Any] | None, year_meta.get("lcia_status"))
    if raw_status is None:
        return set()
    unavailable: set[str] = set()
    for raw_method, raw_meta in raw_status.items():
        lcia_method = raw_method.strip()
        if not lcia_method:
            continue
        raw_meta = cast(dict[str, Any], raw_meta)
        if (
            raw_meta.get("available") is False
            and {_normalize_label(name) for name in raw_meta.get("missing", ())}
            != _OPTIONAL_EXTENSIONS
        ):
            unavailable.add(lcia_method)
    return unavailable


def expected_pyaesa_lcia_methods(
    *,
    year_meta: dict[str, Any] | None,
    requested_methods: Sequence[str] | None,
) -> list[str] | None:
    """Return LCIA methods expected in pyaesa completeness checks."""
    meta_methods = year_entry_lcia_methods(year_meta)
    expected = merge_lcia_method_lists(
        meta_methods,
        requested_methods,
    )
    unavailable = year_entry_unavailable_lcia_methods(year_meta)
    if unavailable:
        expected = [
            lcia_method
            for lcia_method in expected
            if lcia_method in meta_methods or lcia_method not in unavailable
        ]
    return expected or None


def extract_lcia_units_from_char_matrix(
    *,
    lcia_method: str,
    char_matrix: pd.DataFrame,
) -> dict[str, str]:
    """Build ``impact_parent -> impact_unit`` mapping from a CF matrix."""
    del lcia_method
    mapping = cast(pd.DataFrame, char_matrix[["impact_parent", "impact_unit"]].copy())
    dedup = cast(
        pd.DataFrame,
        mapping.drop_duplicates(subset=["impact_parent", "impact_unit"]).sort_values(
            ["impact_parent", "impact_unit"]
        ),
    )
    result: dict[str, str] = {}
    for _, row in dedup.iterrows():
        parent = str(row["impact_parent"]).strip()
        unit = str(row["impact_unit"]).strip()
        result[parent] = unit
    return dict(sorted(result.items()))


def extract_lcia_units_from_jobs(
    jobs: dict[str, ExioCharacterizationOptions],
) -> dict[str, dict[str, str]]:
    """Build LCIA unit maps for characterization jobs."""
    out: dict[str, dict[str, str]] = {}
    for lcia_method, options in jobs.items():
        out[lcia_method] = extract_lcia_units_from_char_matrix(
            lcia_method=lcia_method,
            char_matrix=options.char_matrix,
        )
    return out


def resolve_lcia_units_for_methods(
    *,
    lcia_method_names: Sequence[str] | None,
    units_by_method: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    """Return ``method -> impact unit map`` for processed methods."""
    resolved: dict[str, dict[str, str]] = {}
    for lcia_method in normalize_lcia_method_list(lcia_method_names):
        resolved[lcia_method] = units_by_method[lcia_method]
    return resolved


def build_lcia_status_payload(
    *,
    lcia_method_names: Sequence[str],
    applied_methods: Sequence[str] | None,
    missing_by_method: dict[str, list[str]] | None,
) -> dict[str, dict[str, Any]]:
    """Build stable LCIA availability payload for metadata."""
    applied = set(normalize_lcia_method_list(applied_methods))
    missing_map = missing_by_method or {}
    payload: dict[str, dict[str, Any]] = {}
    for lcia_method in normalize_lcia_method_list(lcia_method_names):
        missing = missing_map.get(lcia_method)
        if missing:
            payload[lcia_method] = {
                "available": lcia_method in applied,
                "missing": sorted(set(missing)),
            }
            continue
        if lcia_method in applied:
            payload[lcia_method] = {"available": True}
            continue
        payload[lcia_method] = {"available": False, "missing": []}
    return payload


def build_year_entry_payload(
    *,
    saved_dir_name: str,
    core_matrices: list[str],
    extension_payload: dict[str, Any],
    updated_iso: str,
    uncasext_only: bool,
    preclip_core_matrices: list[str],
    preclip_extension_payload: dict[str, list[str]],
    pymrio_calc_all: bool,
    enacting_metric_units: dict[str, Any],
    applied_methods: Sequence[str] | None,
    is_exio: bool,
    requires_characterization: bool,
    year_char_jobs: dict[str, ExioCharacterizationOptions],
    missing_by_method: dict[str, list[str]] | None,
    runtime_env: dict[str, str],
    raw_correction_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one ``metadata['years'][year]`` payload entry."""
    year_entry: dict[str, Any] = {
        "saved_dir": saved_dir_name,
        "core": core_matrices,
        "extensions": extension_payload,
        "updated": updated_iso,
        "runtime_env": dict(runtime_env),
    }
    if uncasext_only:
        year_entry["uncasext_only"] = True
    else:
        year_entry["preclip"] = {
            "dir": "preclip",
            "core": list(preclip_core_matrices),
            "extensions_dir": "extensions",
            "extensions": dict(preclip_extension_payload),
            "pymrio_calc_all": bool(pymrio_calc_all),
        }
    year_entry["utility_propag_uncasext"] = {
        "dir": "utility_propag_uncasext",
        "matrices": ["x_to_rc", "kappa", "omega_reg"],
    }
    year_entry["enacting_metrics"] = {
        "dir": "enacting_metrics",
        "level_1": ["fd_rf", "gva_rp"],
        "level_2": ["fd_rp_sp_rf", "fd_rp_sp", "fd_rf_sp", "gva_rp_sp"],
        "units_file": "units.json",
        "units": enacting_metric_units,
        "lcia_methods": list(applied_methods or []),
        "lcia_subdir": True,
        "lcia_level_1": (["e_cba_fd_reg", "e_pba_reg"] if is_exio else []),
        "lcia_level_2": (
            [
                "e_pba_rp_sp",
                "e_cba_fd_rp_sp",
                "e_cba_fd_rp_sp_rf",
                "e_cba_td_rp_sp_rc",
                "e_cba_td_rp_sp",
                "e_cba_fd_rf_sp",
                "e_cba_td_rc_sp",
            ]
            if is_exio
            else []
        ),
    }
    if requires_characterization and year_char_jobs:
        year_entry["characterization_inputs"] = {
            method: options.matrix_sha256 for method, options in year_char_jobs.items()
        }
        lcia_status_payload = build_lcia_status_payload(
            lcia_method_names=list(year_char_jobs.keys()),
            applied_methods=applied_methods,
            missing_by_method=missing_by_method,
        )
        year_entry["lcia_status"] = lcia_status_payload
    if raw_correction_payload is not None:
        year_entry["raw_corrected_values"] = dict(raw_correction_payload)
    return year_entry
