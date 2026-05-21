"""Deterministic ACC and ASR branch scope signature guard."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.manifest_contract import manifest_digest
from pyaesa.shared.runtime.scenario.columns import AR6_CC_SSP_SCENARIO_COLUMN


def branch_identity_payload(
    *, public_request_payload: dict[str, Any], cc_type: str
) -> dict[str, Any]:
    """Return immutable deterministic branch identity excluding appendable coverage axes."""
    payload = deepcopy(public_request_payload)
    cc_args = dict(payload.get("base_cc_args") or {})
    payload.pop("years", None)
    if cc_type == "static":
        cc_args["static"] = {}
    else:
        dynamic = dict(cc_args.get("dynamic_ar6") or {})
        dynamic.pop("category", None)
        dynamic.pop("ssp_scenario", None)
        cc_args["dynamic_ar6"] = dynamic
    payload["base_cc_args"] = cc_args
    return payload


def branch_coverage(
    *,
    cc_type: str,
    requested_years: list[int],
    static_cc_bounds: list[str],
    category: list[str] | None,
    ssp_scenario: list[str] | None,
) -> dict[str, list[Any]]:
    """Return deterministic branch coverage axes represented by one request."""
    coverage: dict[str, list[Any]] = {"years": [int(year) for year in requested_years]}
    if cc_type == "static":
        coverage["cc_bound"] = [str(value) for value in static_cc_bounds]
    else:
        coverage["cc_category"] = [str(value) for value in (category or [])]
        coverage[AR6_CC_SSP_SCENARIO_COLUMN] = [str(value) for value in (ssp_scenario or [])]
    return coverage


def branch_reuse_mode_or_raise(
    *,
    existing_metadata: dict[str, Any] | None,
    requested_identity: dict[str, Any],
    requested_coverage: dict[str, list[Any]],
    scope_label: str,
    function_name: str,
) -> str:
    """Return ``reuse``, ``append``, or ``compute`` for one branch manifest."""
    if not existing_metadata:
        return "compute"
    ensure_same_branch_identity_or_raise(
        existing_metadata=existing_metadata,
        requested_identity=requested_identity,
        scope_label=scope_label,
        function_name=function_name,
    )
    existing_coverage = existing_metadata["reuse"]["coverage"]
    return _coverage_relation_or_raise(
        existing_coverage=existing_coverage,
        requested_coverage=requested_coverage,
        scope_label=scope_label,
        function_name=function_name,
    )


def ensure_recorded_output_files_exist(
    *,
    existing_metadata: dict[str, Any],
    scope_label: str,
    function_name: str,
) -> None:
    """Fail when a complete deterministic branch manifest points to missing outputs."""
    missing = [
        str(path)
        for path in (
            Path(str(value))
            for value in existing_metadata["artifacts"].get("output_files", [])
            if str(value).strip()
        )
        if not path.exists()
    ]
    if missing:
        raise ValueError(
            f"Existing {function_name} metadata marks deterministic scope '{scope_label}' as "
            "complete, but one or more output files are missing. "
            f"Missing output files: {missing[:5]}. Use refresh=True or a new project_name."
        )


def ensure_same_branch_identity_or_raise(
    *,
    existing_metadata: dict[str, Any] | None,
    requested_identity: dict[str, Any],
    scope_label: str,
    function_name: str,
) -> None:
    """Fail when a persisted deterministic branch belongs to a different identity."""
    if not existing_metadata:
        return
    existing_key = str(existing_metadata["reuse"]["identity_key"])
    requested_key = manifest_digest(requested_identity)
    if existing_key == requested_key:
        return
    raise ValueError(
        f"{function_name} cannot append to deterministic scope '{scope_label}' because "
        "branch identity changed. Use a distinct project_name."
    )


def _coverage_relation_or_raise(
    *,
    existing_coverage: dict[str, Any],
    requested_coverage: dict[str, list[Any]],
    scope_label: str,
    function_name: str,
) -> str:
    relations: list[str] = []
    for axis, requested_values in requested_coverage.items():
        existing_values = _persisted_coverage_set(existing_coverage.get(axis))
        requested_set = set(requested_values)
        if requested_set == existing_values:
            relations.append("exact")
        elif requested_set.issubset(existing_values):
            relations.append("subset")
        elif requested_set.issuperset(existing_values):
            relations.append("superset")
        else:
            raise ValueError(
                f"{function_name} cannot append request coverage to deterministic scope "
                f"'{scope_label}' because axis '{axis}' partially overlaps or is disjoint. "
                "Use a distinct project scope or refresh this branch."
            )
    if all(relation in {"exact", "subset"} for relation in relations):
        return "reuse"
    if "subset" in relations:
        raise ValueError(
            f"{function_name} cannot append request coverage to deterministic scope "
            f"'{scope_label}' because one coverage axis is a subset while another is a "
            "superset. Use a distinct project scope or refresh this branch."
        )
    return "append"


def merged_coverage(
    *,
    existing_metadata: dict[str, Any] | None,
    requested_coverage: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Return sorted branch coverage after writing one request."""
    existing = None if existing_metadata is None else existing_metadata["reuse"]["coverage"]
    merged: dict[str, list[Any]] = {}
    for axis, requested_values in requested_coverage.items():
        values = set(requested_values)
        if isinstance(existing, dict) and axis in existing:
            values.update(_persisted_coverage_set(existing[axis]))
        merged[axis] = sorted(values, key=lambda item: str(item))
    return merged


def coverage_signature_covers(stored: dict[str, Any], requested: dict[str, Any]) -> bool:
    """Return whether one stored branch coverage signature covers a figure request."""
    return stored["identity_key"] == requested["identity_key"] and all(
        set(values).issubset(_persisted_coverage_set(stored["coverage"][axis]))
        for axis, values in requested["coverage"].items()
    )


def _persisted_coverage_set(values: Any) -> set[Any]:
    return set(values)
