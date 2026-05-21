"""Typed readers for persisted deterministic aSoCC run-scope metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from pyaesa.shared.figures.title_contract import (
    SelectorScopeRequest,
    selector_scope_request_from_filters,
)

from ...io.metadata import _run_scope_key


def _require_str_list(*, values: Any) -> list[str]:
    """Return one non-empty string list from package-owned persisted metadata."""
    out: list[str] = []
    for value in values:
        out.append(str(value).strip())
    return out


def _require_int_list(*, values: Any) -> list[int]:
    """Return one integer list from package-owned persisted metadata."""
    return sorted({int(value) for value in values})


def _normalize_optional_string_list(*, values: Any) -> list[str]:
    """Return one normalized optional string list from persisted metadata."""
    if values is None:
        return []
    out = sorted(
        {str(value).strip() for value in values if value is not None and str(value).strip()}
    )
    return out


def _normalize_selector_filters(
    raw_filters: Any,
) -> dict[str, list[str] | None] | None:
    """Normalize top-level persisted selector filters."""
    if not isinstance(raw_filters, dict):
        return None
    normalized_filters: dict[str, list[str] | None] = {}
    for key, values in raw_filters.items():
        column = str(key).strip()
        if not column:
            continue
        if values is None:
            normalized_filters[column] = None
            continue
        normalized_filters[column] = [str(value) for value in values]
    return normalized_filters


@dataclass(frozen=True)
class AsoccPersistedComputeSignature:
    """Typed owner for one persisted deterministic aSoCC compute signature."""

    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return one shallow copy of the persisted signature payload."""
        return dict(self.payload)

    @property
    def source(self) -> str:
        """Return the persisted original source for this deterministic scope."""
        return str(self.payload["source"]).strip()

    @property
    def variant_tag(self) -> str | None:
        """Return the optional persisted variant tag for this deterministic scope."""
        raw = self.payload.get("variant_tag")
        if raw is None:
            return None
        text = str(raw).strip()
        return text or None

    @property
    def aggreg_indices(self) -> bool:
        """Return whether this persisted scope represents grouped outputs."""
        return bool(self.payload["aggreg_indices"])

    @property
    def output_format(self) -> str:
        """Return the persisted tabular output format for this deterministic scope."""
        return str(self.payload["output_format"]).strip()

    @property
    def intermediate_outputs(self) -> bool:
        """Return whether intermediate outputs were enabled for this deterministic scope."""
        return bool(self.payload.get("intermediate_outputs", True))

    def prerequisite_identity_payload(
        self,
        *,
        base_function_source: str,
        output_source_label: str,
        proj_base: Path,
        scope_key: str,
    ) -> dict[str, Any]:
        """Return the exact persisted signature digest payload for one reused prerequisite."""
        return {
            "base_function_source": base_function_source,
            "output_source_label": output_source_label,
            "proj_base": str(proj_base),
            "scope_key": str(scope_key),
            "scope_signature": self.as_dict(),
        }


@dataclass(frozen=True)
class AsoccPersistedRunScope:
    """Typed view of one persisted deterministic aSoCC run scope."""

    scope_key: str
    compute_signature: AsoccPersistedComputeSignature
    completed_years: list[int]
    outputs: list[str]
    ssp_scenarios: list[str]
    timestamp: str | None = None

    @property
    def source(self) -> str:
        """Return the persisted original source for this deterministic scope."""
        return self.compute_signature.source

    @property
    def variant_tag(self) -> str | None:
        """Return the optional persisted variant tag for this deterministic scope."""
        return self.compute_signature.variant_tag

    @property
    def aggreg_indices(self) -> bool:
        """Return whether this persisted scope represents grouped outputs."""
        return self.compute_signature.aggreg_indices

    @property
    def output_format(self) -> str:
        """Return the persisted tabular output format for this deterministic scope."""
        return self.compute_signature.output_format

    @property
    def intermediate_outputs(self) -> bool:
        """Return whether intermediate outputs were enabled for this deterministic scope."""
        return self.compute_signature.intermediate_outputs

    def covers_years(self, years: list[int] | None) -> bool:
        """Return whether this persisted scope covers all requested years."""
        if not years:
            return True
        return set(int(year) for year in years).issubset(set(self.completed_years))

    def prerequisite_identity_payload(
        self,
        *,
        base_function_source: str,
        output_source_label: str,
        proj_base: Path,
    ) -> dict[str, Any]:
        """Return the exact persisted prerequisite identity payload for this scope."""
        return self.compute_signature.prerequisite_identity_payload(
            base_function_source=base_function_source,
            output_source_label=output_source_label,
            proj_base=proj_base,
            scope_key=self.scope_key,
        )


@dataclass(frozen=True)
class AsoccPersistedRunCatalog:
    """Typed view of persisted deterministic aSoCC run metadata."""

    scopes: tuple[AsoccPersistedRunScope, ...]
    selector_filters: dict[str, list[str] | None] | None
    run_ssp_scenarios: list[str]

    def selector_scope_request(self) -> SelectorScopeRequest | None:
        """Return selector scope metadata preserved in deterministic run metadata."""
        if self.selector_filters is None:
            return None
        return selector_scope_request_from_filters(filters=self.selector_filters)

    def scope_for_compute_signature(
        self,
        *,
        compute_signature: Mapping[str, Any],
    ) -> AsoccPersistedRunScope | None:
        """Return the exact persisted scope for one deterministic signature when present."""
        scope_key = _run_scope_key(signature=dict(compute_signature))
        for scope in self.scopes:
            if scope.scope_key == scope_key:
                return scope
        return None


def _parse_persisted_scope(*, raw_scope: Any) -> AsoccPersistedRunScope:
    """Parse one package-owned persisted deterministic aSoCC scope payload."""
    raw_scope = dict(raw_scope)
    signature = dict(raw_scope["arguments"])
    execution = cast(Mapping[str, Any], raw_scope["execution"])
    artifacts = cast(Mapping[str, Any], raw_scope["artifacts"])
    provenance = cast(Mapping[str, Any], raw_scope["provenance"])
    reuse = cast(Mapping[str, Any], raw_scope["reuse"])
    return AsoccPersistedRunScope(
        scope_key=str(reuse["identity_key"]),
        compute_signature=AsoccPersistedComputeSignature(payload=dict(signature)),
        completed_years=_require_int_list(
            values=execution["completed_years"],
        ),
        outputs=_require_str_list(
            values=artifacts["outputs"],
        ),
        ssp_scenarios=_normalize_optional_string_list(values=provenance.get("ssp_scenarios")),
        timestamp=(
            None
            if execution.get("timestamp") is None
            else str(execution.get("timestamp")).strip() or None
        ),
    )


def load_asocc_persisted_run_catalog(
    *,
    payload: Mapping[str, Any],
) -> AsoccPersistedRunCatalog:
    """Parse package-owned deterministic aSoCC run metadata into typed scope views."""
    scopes: tuple[AsoccPersistedRunScope, ...]
    if not payload:
        scopes = ()
        provenance: Mapping[str, Any] = {}
    else:
        scopes = (_parse_persisted_scope(raw_scope=payload),)
        provenance = cast(Mapping[str, Any], payload.get("provenance", {}))
    return AsoccPersistedRunCatalog(
        scopes=scopes,
        selector_filters=_normalize_selector_filters(provenance.get("filters")),
        run_ssp_scenarios=_normalize_optional_string_list(values=provenance.get("ssp_scenarios")),
    )
