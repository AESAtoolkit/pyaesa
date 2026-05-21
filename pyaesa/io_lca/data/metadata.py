"""Scope manifest ownership for deterministic IO-LCA and figure workflows."""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, cast
from pyaesa.shared.runtime.manifest_contract import manifest_digest
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.reuse.contracts import (
    io_lca_signature_compatible,
)


def _scope_key(*, signature: dict[str, Any]) -> str:
    """Return deterministic SHA-256 digest for a run signature."""
    return manifest_digest(signature)


def _now_iso() -> str:
    """Return local timestamp string."""
    return datetime.now().isoformat()


def _empty_scope_payload(*, function_name: str) -> dict[str, Any]:
    """Return an empty deterministic IO-LCA manifest payload."""
    timestamp = _now_iso()
    return {
        "function": function_name,
        "arguments": None,
        "timestamp": timestamp,
        "complete": False,
        "paths_written": [],
        "status": {
            "main": {},
            "origin": {},
            "stages": {},
            "figures": {},
        },
        "identity_key": None,
    }


def _scope_manifest_payload(*, key: str, scope: dict[str, Any]) -> dict[str, Any]:
    """Return the persisted deterministic IO-LCA manifest."""
    return {
        "function": scope["function"],
        "arguments": scope["arguments"],
        "execution": {
            "status": "complete" if bool(scope["complete"]) else "running",
            "complete": bool(scope["complete"]),
            "updated": scope["timestamp"],
            "sections": scope["status"],
        },
        "reuse": {"identity_key": key},
        "artifacts": {"paths_written": list(scope["paths_written"])},
        "provenance": {},
    }


def _scope_from_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the mutable runtime scope from one persisted IO-LCA manifest."""
    execution = cast(dict[str, Any], payload["execution"])
    artifacts = cast(dict[str, Any], payload["artifacts"])
    reuse = cast(dict[str, Any], payload["reuse"])
    return {
        "function": payload["function"],
        "arguments": payload["arguments"],
        "timestamp": execution["updated"],
        "complete": bool(execution["complete"]),
        "paths_written": list(cast(list[Any], artifacts["paths_written"])),
        "status": cast(dict[str, Any], execution["sections"]),
        "identity_key": reuse["identity_key"],
    }


def load_scope_manifest(*, path: Path, function_name: str) -> dict[str, Any]:
    """Load or initialize the deterministic IO-LCA scope manifest.

    Args:
        path: Scope manifest JSON file.
        function_name: Function label (for example ``"deterministic_io_lca"`` or
            ``"deterministic_io_lca_figures"``).

    Returns:
        Parsed payload with required top level keys.
    """
    if not path.exists():
        return _empty_scope_payload(function_name=function_name)
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return _scope_from_manifest(payload)


def save_scope_manifest(*, path: Path, payload: dict[str, Any]) -> None:
    """Persist one deterministic IO-LCA scope manifest payload."""
    path = ensure_file_parent(path)
    signature = cast(dict[str, Any], payload["arguments"])
    key = str(payload.get("identity_key") or _scope_key(signature=signature))
    manifest_payload = _scope_manifest_payload(key=key, scope=payload)
    path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")


def get_scope(
    *,
    payload: dict[str, Any],
    signature: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    """Return scope key and existing scope entry for signature."""
    key = _scope_key(signature=signature)
    if payload.get("arguments") is None or payload.get("identity_key") != key:
        return key, None
    return key, payload


def iter_scope_entries(*, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical deterministic IO-LCA scope entries from one metadata payload."""
    return [] if payload.get("arguments") is None else [payload]


def require_scope_signature(*, scope: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical persisted deterministic IO-LCA scope signature."""
    return cast(dict[str, Any], scope["arguments"])


def compatible_scope(
    *,
    payload: dict[str, Any],
    project_name: str,
    source: str,
    group_reg: bool,
    group_sec: bool,
    group_version: str | None,
    fu_code: str,
    aggreg_indices: bool,
    output_format: str,
    requested_years: set[int],
    requested_methods: set[str],
    requested_selectors: dict[str, tuple[str, ...]],
) -> dict[str, Any] | None:
    """Return the best deterministic IO-LCA scope compatible with one request."""
    candidates: list[tuple[tuple[int, int], dict[str, Any]]] = []
    for scope in iter_scope_entries(payload=payload):
        signature = require_scope_signature(scope=scope)
        compatible, score = io_lca_signature_compatible(
            signature=signature,
            project_name=project_name,
            source=source,
            group_reg=group_reg,
            group_sec=group_sec,
            group_version=group_version,
            fu_code=fu_code,
            aggreg_indices=aggreg_indices,
            output_format=output_format,
            requested_years=requested_years,
            requested_methods=requested_methods,
            requested_selectors=requested_selectors,
        )
        if compatible:
            candidates.append((score, scope))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def ensure_scope(
    *,
    payload: dict[str, Any],
    key: str,
    signature: dict[str, Any],
    function_name: str,
) -> dict[str, Any]:
    """Create or return mutable scope payload for one signature."""
    if payload.get("arguments") is not None and payload.get("identity_key") == key:
        return payload
    payload.clear()
    payload.update(
        {
            "function": function_name,
            "arguments": signature,
            "timestamp": _now_iso(),
            "complete": False,
            "paths_written": [],
            "status": {
                "main": {},
                "origin": {},
                "stages": {},
                "figures": {},
            },
            "identity_key": key,
        }
    )
    return payload


def scope_complete_and_existing(scope: dict[str, Any]) -> bool:
    """Return ``True`` when scope is complete and all outputs still exist."""
    if not bool(scope.get("complete")):
        return False
    paths = cast(list[Any], scope.get("paths_written", []))
    if not paths:
        return False
    return all(Path(str(path)).exists() for path in paths)


def _lcia_method_status_entry(
    *,
    scope: dict[str, Any],
    section: str,
    lcia_method: str,
) -> dict[str, Any]:
    """Return one persisted status entry for a deterministic IO-LCA method."""
    status = cast(dict[str, Any], scope.get("status", {}))
    section_map = cast(dict[str, Any], status.get(section, {}))
    return cast(dict[str, Any], section_map.get(lcia_method, {}))


def _year_values_from_status_entry(
    *,
    entry: dict[str, Any],
    field_name: str,
) -> list[int]:
    """Return sorted integer years from one list-valued method status field."""
    raw_years = entry.get(field_name, [])
    return sorted({int(raw_year) for raw_year in raw_years})


def get_lcia_method_years(
    *,
    scope: dict[str, Any],
    section: str,
    lcia_method: str,
) -> list[int]:
    """Return completed years for one LCIA method in one status section."""
    lcia_method_entry = _lcia_method_status_entry(
        scope=scope,
        section=section,
        lcia_method=lcia_method,
    )
    return _year_values_from_status_entry(
        entry=lcia_method_entry,
        field_name="years_done",
    )


def get_lcia_method_done_and_skipped_years(
    *,
    scope: dict[str, Any],
    section: str,
    lcia_method: str,
) -> tuple[set[int], set[int]]:
    """Return completed and skipped years for one deterministic IO-LCA method."""
    lcia_method_entry = _lcia_method_status_entry(
        scope=scope,
        section=section,
        lcia_method=lcia_method,
    )
    done = set(
        _year_values_from_status_entry(
            entry=lcia_method_entry,
            field_name="years_done",
        )
    )
    skipped_raw = cast(dict[Any, Any], lcia_method_entry.get("years_skipped", {}))
    skipped = {int(raw_year) for raw_year in skipped_raw}
    return done, skipped


def set_lcia_method_years(
    *,
    scope: dict[str, Any],
    section: str,
    lcia_method: str,
    years_done: list[int],
    skipped_by_year: dict[int, str] | None = None,
) -> None:
    """Write LCIA method level year status in scope payload."""
    status = cast(dict[str, Any], scope["status"])
    section_map = cast(dict[str, Any] | None, status.get(section))
    if section_map is None:
        section_map = {}
        status[section] = section_map
    entry = cast(dict[str, Any] | None, section_map.get(lcia_method))
    if entry is None:
        entry = {}
        section_map[lcia_method] = entry
    entry["years_done"] = sorted({int(year) for year in years_done})
    if skipped_by_year is not None:
        entry["years_skipped"] = {
            str(int(year)): str(reason) for year, reason in sorted(skipped_by_year.items())
        }


def set_figure_paths(
    *,
    scope: dict[str, Any],
    lcia_method: str,
    figure_paths: list[Path],
) -> None:
    """Persist figure path status for one LCIA method."""
    status = cast(dict[str, Any], scope["status"])
    figure_status = cast(dict[str, Any] | None, status.get("figures"))
    if figure_status is None:
        figure_status = {}
        status["figures"] = figure_status
    figure_status[lcia_method] = {
        "paths": [str(path) for path in sorted({Path(path) for path in figure_paths})],
    }


def merge_written_paths(*, scope: dict[str, Any], paths: list[Path]) -> None:
    """Merge deduplicated written paths into scope payload."""
    existing = scope.get("paths_written", [])
    all_paths = {str(path) for path in cast(list[Any], existing)}
    all_paths.update(str(path) for path in paths)
    scope["paths_written"] = sorted(all_paths)


def set_scope_complete(*, scope: dict[str, Any], complete: bool) -> None:
    """Set completion flag and refresh timestamp."""
    scope["complete"] = bool(complete)
    scope["timestamp"] = _now_iso()
