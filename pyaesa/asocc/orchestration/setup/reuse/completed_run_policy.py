"""Completed run policy for deterministic aSoCC setup."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from pyaesa.asocc.orchestration.setup.request.types import _YearBundle
from pyaesa.asocc.runtime.scope.persisted_scope import (
    AsoccPersistedRunScope,
    load_asocc_persisted_run_catalog,
)
from pyaesa.shared.tabular.contracts import TABULAR_SUFFIXES

from ....io.metadata import _load_run_metadata
from ....runtime.paths.deterministic import _get_allocate_run_metadata_path
from ....runtime.paths.published import _asocc_deterministic_scope_root, _get_asocc_root

_SET_AWARE_SIGNATURE_KEYS = {
    "years",
    "lcia_methods",
    "ssp_scenario_input",
    "reference_years_input",
    "selected_methods",
    "l2_reuse_years",
}


def _exact_identity_mismatches(
    *,
    requested_signature: Mapping[str, Any],
    persisted_signature: Mapping[str, Any],
) -> list[tuple[str, Any, Any]]:
    """Return fixed identity fields that differ inside one deterministic output folder."""
    exact_keys = sorted(
        (set(requested_signature) | set(persisted_signature)) - _SET_AWARE_SIGNATURE_KEYS
    )
    # A None request follows the function default and may match a prior resolved reg_window.
    return [
        (
            key,
            requested_signature.get(key),
            persisted_signature.get(key),
        )
        for key in exact_keys
        if requested_signature.get(key) != persisted_signature.get(key)
        and not (key == "reg_window" and requested_signature.get(key) is None)
    ]


def _format_exact_identity_mismatches(
    *,
    mismatches: list[tuple[str, Any, Any]],
) -> str:
    """Return a concise user facing exact identity mismatch diagnostic."""
    return " | ".join(
        f"{key}: requested={requested!r}; persisted={persisted!r}"
        for key, requested, persisted in mismatches
    )


@dataclass(frozen=True)
class AppendComputeScope:
    """Subset of requested axes that must be computed for an append run."""

    years: list[int]
    lcia_methods: list[str] | None
    ssp_scenario_input: list[str] | str | None
    reference_years_input: list[int] | None
    selected_methods: dict[str, list[str]] | None
    l2_reuse_years: list[int] | None


def _missing_scope_outputs(*, scope: AsoccPersistedRunScope) -> list[str]:
    """Return output paths recorded by one completed scope that are not on disk."""
    return [str(path) for path in scope.outputs if not Path(str(path)).exists()]


def _enforce_project_fu_scope(
    *,
    proj_base: Path,
    requested_fu_code: str,
) -> None:
    """Enforce one deterministic aSoCC functional unit per project.

    Deterministic aSoCC outputs share method-owned files inside the project
    aSoCC tree. Keeping a single FU per project prevents incompatible public
    identity axes such as ``r_p`` and ``r_c`` from being appended into the same
    output tables.
    """
    asocc_root = _get_asocc_root(proj_base=proj_base)
    if not asocc_root.exists():
        return
    existing_fu_codes: set[str] = set()
    for manifest_path in asocc_root.glob("*/deterministic/logs/scope_manifest.json"):
        metadata = _load_run_metadata(manifest_path)
        catalog = load_asocc_persisted_run_catalog(payload=metadata)
        for scope in catalog.scopes:
            signature = scope.compute_signature.as_dict()
            existing_fu_codes.add(str(signature["fu_code"]))
    incompatible = sorted(fu for fu in existing_fu_codes if fu != requested_fu_code)
    if incompatible:
        raise ValueError(
            "deterministic_asocc uses one functional unit per project_name. "
            "This project already contains deterministic aSoCC outputs for "
            f"fu_code={incompatible}; "
            f"requested fu_code='{requested_fu_code}'. Use a new project_name for a different "
            "functional unit, or manually remove the existing aSoCC outputs for this project."
        )


def _as_set(value: Any) -> set[str]:
    """Return a normalized set for one signature value compared as a set."""
    if value is None:
        return set()
    if isinstance(value, Mapping):
        out: set[str] = set()
        for key, values in value.items():
            for item in _as_set(values):
                out.add(f"{key}:{item}")
        return out
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value if item is not None}
    return {str(value)}


def _as_ordered_missing_values(*, requested: Any, persisted: Any) -> list[Any]:
    """Return requested values absent from a persisted set, preserving request order."""
    persisted_set = _as_set(persisted)
    if requested is None:
        return []
    if isinstance(requested, (list, tuple)):
        return [value for value in requested if str(value) not in persisted_set]
    return [] if str(requested) in persisted_set else [requested]


def _missing_selected_methods(
    *,
    requested: Mapping[str, Any] | None,
    persisted: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    """Return selected method labels absent from a persisted selected-method mapping."""
    requested_map = requested or {}
    persisted_map = persisted or {}
    out: dict[str, list[str]] = {}
    for key in ("l1", "l2_in_l1", "l2_vs_global"):
        missing = _as_ordered_missing_values(
            requested=requested_map.get(key),
            persisted=persisted_map.get(key),
        )
        out[key] = [str(value) for value in missing]
    return out


def _axis_relation(*, requested: set[str], persisted: set[str]) -> str:
    """Classify one deterministic axis relation using set semantics."""
    if requested == persisted:
        return "exact"
    if requested.issubset(persisted):
        return "subset"
    if requested.issuperset(persisted):
        return "superset"
    return "partial"


def _scope_set_relations(
    *,
    requested_signature: Mapping[str, Any],
    persisted_signature: Mapping[str, Any],
    requested_years: list[int],
    completed_years: set[int],
) -> list[str]:
    """Return set based axis relations for one compatible deterministic scope."""
    relations = [
        _axis_relation(
            requested={str(int(year)) for year in requested_years},
            persisted={str(int(year)) for year in completed_years},
        )
    ]
    for key in sorted(_SET_AWARE_SIGNATURE_KEYS - {"years"}):
        relations.append(
            _axis_relation(
                requested=_as_set(requested_signature.get(key)),
                persisted=_as_set(persisted_signature.get(key)),
            )
        )
    return relations


def _scope_set_relation_details(
    *,
    requested_signature: Mapping[str, Any],
    persisted_signature: Mapping[str, Any],
    requested_years: list[int],
    completed_years: set[int],
) -> list[tuple[str, str, list[str], list[str]]]:
    """Return axis relation details for a matching deterministic scope."""
    axis_values = [
        (
            "years",
            {str(int(year)) for year in requested_years},
            {str(int(year)) for year in completed_years},
        )
    ]
    for key in sorted(_SET_AWARE_SIGNATURE_KEYS - {"years"}):
        axis_values.append(
            (
                key,
                _as_set(requested_signature.get(key)),
                _as_set(persisted_signature.get(key)),
            )
        )
    return [
        (
            key,
            _axis_relation(requested=requested, persisted=persisted),
            sorted(requested - persisted),
            sorted(persisted - requested),
        )
        for key, requested, persisted in axis_values
        if _axis_relation(requested=requested, persisted=persisted) != "exact"
    ]


def _format_relation_details(
    *,
    details: list[tuple[str, str, list[str], list[str]]],
) -> str:
    """Return a compact deterministic scope compatibility diagnostic."""
    parts = []
    for axis, relation, requested_extra, persisted_extra in details:
        parts.append(
            f"{axis}={relation}; requested_not_persisted={requested_extra}; "
            f"persisted_not_requested={persisted_extra}"
        )
    return " | ".join(parts)


def _scope_is_complete(*, relations: list[str]) -> bool:
    """Return whether a persisted scope fully covers a request."""
    return all(relation in {"exact", "subset"} for relation in relations)


def _scope_is_appendable(*, relations: list[str]) -> bool:
    """Return whether a request can append to a persisted scope."""
    return (
        "superset" in relations
        and "subset" not in relations
        and all(relation in {"exact", "superset"} for relation in relations)
    )


def _append_year_bundle(*, year_bundle: _YearBundle, missing_years: list[int]) -> _YearBundle:
    """Return a compute year bundle limited to missing public years."""
    return replace(year_bundle, resolved_years=missing_years)


def _scope_outputs(*, scope: AsoccPersistedRunScope) -> list[str]:
    """Return persisted output paths from one metadata scope."""
    return [str(path).strip() for path in scope.outputs if str(path).strip()]


def _append_compute_scope(
    *,
    requested_signature: Mapping[str, Any],
    persisted_signature: Mapping[str, Any],
    requested_years: list[int],
    completed_years: set[int],
) -> AppendComputeScope:
    """Return compute-only axes absent from the persisted deterministic scope."""
    missing_years = sorted({int(year) for year in requested_years} - completed_years)
    missing_lcia = _as_ordered_missing_values(
        requested=requested_signature.get("lcia_methods"),
        persisted=persisted_signature.get("lcia_methods"),
    )
    missing_ssp = _as_ordered_missing_values(
        requested=requested_signature.get("ssp_scenario_input"),
        persisted=persisted_signature.get("ssp_scenario_input"),
    )
    missing_refs = _as_ordered_missing_values(
        requested=requested_signature.get("reference_years_input"),
        persisted=persisted_signature.get("reference_years_input"),
    )
    missing_reuse = _as_ordered_missing_values(
        requested=requested_signature.get("l2_reuse_years"),
        persisted=persisted_signature.get("l2_reuse_years"),
    )
    missing_methods = _missing_selected_methods(
        requested=requested_signature.get("selected_methods"),
        persisted=persisted_signature.get("selected_methods"),
    )
    return AppendComputeScope(
        years=missing_years,
        lcia_methods=[str(value) for value in missing_lcia] or None,
        ssp_scenario_input=[str(value) for value in missing_ssp] or None,
        reference_years_input=[int(value) for value in missing_refs] or None,
        selected_methods=missing_methods if any(missing_methods.values()) else None,
        l2_reuse_years=[int(value) for value in missing_reuse] or None,
    )


def _append_compute_years(
    *,
    append_scope: AppendComputeScope,
    requested_years: list[int],
) -> list[int]:
    """Return year compute scope for one append run."""
    if (
        append_scope.lcia_methods
        or append_scope.ssp_scenario_input
        or append_scope.reference_years_input
        or append_scope.selected_methods
        or append_scope.l2_reuse_years
    ):
        # Mixed year plus selector append uses one rectangular compute scope.
        # Exact missing cell execution would require splitting this public
        # request into multiple internal run contexts before yearly compute.
        return list(requested_years)
    return list(append_scope.years)


def _completed_years(*, scope: AsoccPersistedRunScope) -> set[int]:
    """Return completed years from canonical metadata."""
    return set(scope.completed_years)


def _has_existing_output_files(
    *,
    proj_base: Path,
    output_source: str,
    group_version: str | None,
) -> bool:
    """Return whether persisted deterministic output files exist for one source."""
    patterns = tuple(f"*{suffix}" for suffix in TABULAR_SUFFIXES)
    source_root = _asocc_deterministic_scope_root(
        proj_base=proj_base,
        source=output_source,
        group_version=group_version,
    )
    if not source_root.exists():
        return False
    return any(next(source_root.rglob(pattern), None) is not None for pattern in patterns)


def apply_completed_run_policy(
    *,
    refresh: bool,
    proj_base: Path,
    output_source: str,
    run_signature: dict,
    year_bundle: _YearBundle,
    reference_years: list[int] | None,
    requested_years: list[int],
    ssp_scenario_options: list[str | None],
) -> tuple[
    _YearBundle,
    list[int] | None,
    list[str | None],
    bool,
    list[int] | None,
    list[str] | None,
    AppendComputeScope | None,
]:
    """Return whether a compatible deterministic scope is already complete."""
    _enforce_project_fu_scope(
        proj_base=proj_base,
        requested_fu_code=str(run_signature["fu_code"]),
    )
    if refresh:
        return year_bundle, reference_years, ssp_scenario_options, False, None, None, None

    meta_path = _get_allocate_run_metadata_path(
        proj_base,
        source=output_source,
        group_version=run_signature.get("group_version"),
    )
    outputs_exist = _has_existing_output_files(
        proj_base=proj_base,
        output_source=output_source,
        group_version=run_signature.get("group_version"),
    )
    if not meta_path.exists():
        if outputs_exist:
            raise ValueError(
                "Existing deterministic aSoCC output files were found but run metadata is missing: "
                f"{meta_path}. Use refresh=True or a new project_name."
            )
        return year_bundle, reference_years, ssp_scenario_options, False, None, None, None

    prior = _load_run_metadata(meta_path)
    catalog = load_asocc_persisted_run_catalog(payload=prior)

    incompatible_exact_details: list[str] = []
    incompatible_overlap_details: list[str] = []
    for scope in catalog.scopes:
        persisted_signature = scope.compute_signature.as_dict()
        exact_mismatches = _exact_identity_mismatches(
            requested_signature=run_signature,
            persisted_signature=persisted_signature,
        )
        if exact_mismatches:
            incompatible_exact_details.append(
                _format_exact_identity_mismatches(mismatches=exact_mismatches)
            )
            continue
        missing_outputs = _missing_scope_outputs(scope=scope)
        if missing_outputs:
            raise ValueError(
                "Existing deterministic aSoCC metadata marks this request as completed, "
                "but one or more referenced output files are missing. "
                f"Metadata file: {meta_path}. "
                f"Missing output files: {missing_outputs[:5]}. "
                "Use refresh=True or a new project_name."
            )
        relations = _scope_set_relations(
            requested_signature=run_signature,
            persisted_signature=persisted_signature,
            requested_years=requested_years,
            completed_years=_completed_years(scope=scope),
        )
        if _scope_is_complete(relations=relations):
            return (
                year_bundle,
                reference_years,
                ssp_scenario_options,
                True,
                sorted(_completed_years(scope=scope)),
                _scope_outputs(scope=scope),
                None,
            )
        if _scope_is_appendable(relations=relations):
            completed_years = _completed_years(scope=scope)
            metadata_completed_years = sorted(
                completed_years | {int(year) for year in requested_years}
            )
            append_scope = _append_compute_scope(
                requested_signature=run_signature,
                persisted_signature=persisted_signature,
                requested_years=requested_years,
                completed_years=completed_years,
            )
            return (
                _append_year_bundle(
                    year_bundle=year_bundle,
                    missing_years=_append_compute_years(
                        append_scope=append_scope,
                        requested_years=requested_years,
                    ),
                ),
                reference_years,
                ssp_scenario_options,
                False,
                metadata_completed_years,
                _scope_outputs(scope=scope),
                append_scope,
            )
        incompatible_overlap_details.append(
            _format_relation_details(
                details=_scope_set_relation_details(
                    requested_signature=run_signature,
                    persisted_signature=persisted_signature,
                    requested_years=requested_years,
                    completed_years=_completed_years(scope=scope),
                ),
            )
        )
    if incompatible_exact_details:
        raise ValueError(
            "Existing deterministic aSoCC metadata for this source and group_version uses "
            "fixed identity fields that differ from the current request. These fields share "
            "one deterministic output folder and cannot coexist there. Use a new "
            "project_name, or use refresh=True to replace the deterministic aSoCC outputs "
            "for this source and group_version. Incompatible identity details: "
            + " ; ".join(incompatible_exact_details)
        )
    raise ValueError(
        "Existing deterministic aSoCC metadata partially overlaps the requested scope. "
        "Append requires every differing extendable axis to be requested as a superset "
        "of the persisted scope. To continue, either use a new project_name, use "
        "refresh=True for this deterministic aSoCC scope, add the persisted-only "
        "values to the request, or remove the request-only values so every differing "
        "axis is exact or a request superset. Incompatible axis details: "
        + " ; ".join(incompatible_overlap_details)
    )
