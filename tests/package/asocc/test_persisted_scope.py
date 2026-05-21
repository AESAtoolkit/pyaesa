from pathlib import Path

from pyaesa.asocc.io.metadata import _run_scope_key
from pyaesa.asocc.runtime.scope.persisted_scope import (
    AsoccPersistedComputeSignature,
    AsoccPersistedRunCatalog,
    AsoccPersistedRunScope,
    load_asocc_persisted_run_catalog,
)


def _scope_payload(
    *,
    signature: dict[str, object],
    completed_years: list[int],
    outputs: list[str],
    ssp_scenarios: list[object] | None = None,
    filters: dict[str, object] | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    execution: dict[str, object] = {"completed_years": completed_years}
    if timestamp is not None:
        execution["timestamp"] = timestamp
    return {
        "function": "deterministic_asocc",
        "arguments": signature,
        "execution": execution,
        "reuse": {"identity_key": _run_scope_key(signature=signature)},
        "artifacts": {"outputs": outputs},
        "provenance": {
            "ssp_scenarios": [] if ssp_scenarios is None else ssp_scenarios,
            "filters": filters,
        },
    }


def test_persisted_scope_contracts_cover_success_paths() -> None:
    signature_payload = {
        "source": "oecd_v2025",
        "variant_tag": " demo_variant ",
        "aggreg_indices": True,
        "output_format": "csv",
        "intermediate_outputs": False,
    }
    persisted_signature = AsoccPersistedComputeSignature(payload=signature_payload)
    scope_key = _run_scope_key(signature=signature_payload)
    persisted_scope = AsoccPersistedRunScope(
        scope_key=scope_key,
        compute_signature=persisted_signature,
        completed_years=[2005, 2001, 2005],
        outputs=["/tmp/a.csv", "/tmp/b.csv"],
        ssp_scenarios=["SSP2", "SSP1"],
        timestamp=" 2025-01-02T03:04:05Z ",
    )
    catalog = AsoccPersistedRunCatalog(
        scopes=(persisted_scope,),
        selector_filters={
            "r_p": [" FR ", "FR"],
            "s_p": None,
            "": ["ignored"],
            "custom": ["value"],
        },
        run_ssp_scenarios=["SSP2", "SSP1"],
    )

    assert persisted_signature.as_dict() == signature_payload
    assert persisted_signature.source == "oecd_v2025"
    assert persisted_signature.variant_tag == "demo_variant"
    assert persisted_signature.aggreg_indices is True
    assert persisted_signature.output_format == "csv"
    assert persisted_signature.intermediate_outputs is False
    assert (
        AsoccPersistedComputeSignature(
            payload={
                "source": "oecd_v2025",
                "aggreg_indices": False,
                "output_format": "csv",
            }
        ).intermediate_outputs
        is True
    )
    assert (
        AsoccPersistedComputeSignature(
            payload={
                "source": "oecd_v2025",
                "aggreg_indices": False,
                "output_format": "csv",
            }
        ).variant_tag
        is None
    )
    assert (
        AsoccPersistedComputeSignature(
            payload={
                "source": "oecd_v2025",
                "aggreg_indices": False,
                "output_format": "csv",
                "variant_tag": "   ",
            }
        ).variant_tag
        is None
    )

    identity_payload = persisted_signature.prerequisite_identity_payload(
        base_function_source="deterministic_asocc",
        output_source_label="oecd_v2025",
        proj_base=Path("/tmp/project"),
        scope_key=scope_key,
    )
    assert identity_payload == {
        "base_function_source": "deterministic_asocc",
        "output_source_label": "oecd_v2025",
        "proj_base": str(Path("/tmp/project")),
        "scope_key": scope_key,
        "scope_signature": signature_payload,
    }

    assert persisted_scope.compute_signature.as_dict() == signature_payload
    assert persisted_scope.source == "oecd_v2025"
    assert persisted_scope.variant_tag == "demo_variant"
    assert persisted_scope.aggreg_indices is True
    assert persisted_scope.output_format == "csv"
    assert persisted_scope.intermediate_outputs is False
    assert persisted_scope.covers_years(None) is True
    assert persisted_scope.covers_years([]) is True
    assert persisted_scope.covers_years([2001]) is True
    assert persisted_scope.covers_years([1999]) is False
    assert (
        persisted_scope.prerequisite_identity_payload(
            base_function_source="deterministic_asocc",
            output_source_label="oecd_v2025",
            proj_base=Path("/tmp/project"),
        )
        == identity_payload
    )

    selector_request = catalog.selector_scope_request()
    assert selector_request is not None
    assert selector_request.axes == (
        ("r_p", ("FR",)),
        ("s_p", None),
    )
    assert (
        catalog.scope_for_compute_signature(compute_signature=signature_payload) == persisted_scope
    )
    assert catalog.scope_for_compute_signature(compute_signature={"source": "other"}) is None

    catalog_no_filters = AsoccPersistedRunCatalog(
        scopes=(persisted_scope,),
        selector_filters=None,
        run_ssp_scenarios=[],
    )
    assert catalog_no_filters.selector_scope_request() is None


def test_persisted_scope_contracts_parse_catalog_scope_fields() -> None:
    filters = {
        "r_p": [" FR ", "FR"],
        "s_p": None,
        "": ["skip"],
    }
    signature = {
        "source": "oecd_v2025",
        "aggreg_indices": False,
        "output_format": "csv",
    }
    catalog = load_asocc_persisted_run_catalog(
        payload=_scope_payload(
            signature=signature,
            completed_years=[2005],
            outputs=["/tmp/a.csv"],
            ssp_scenarios=["SSP2", " SSP1 ", None],
            filters=filters,
        )
    )
    assert [scope.scope_key for scope in catalog.scopes] == [_run_scope_key(signature=signature)]
    assert catalog.run_ssp_scenarios == ["SSP1", "SSP2"]
    assert catalog.selector_scope_request() is not None
    assert catalog.scopes[0].timestamp is None
    assert catalog.scopes[0].ssp_scenarios == ["SSP1", "SSP2"]
    assert catalog.scopes[0].compute_signature.intermediate_outputs is True

    parsed_timestamp = load_asocc_persisted_run_catalog(
        payload=_scope_payload(
            signature=signature,
            completed_years=[2005],
            outputs=["/tmp/a.csv"],
            ssp_scenarios=["SSP2", " SSP1 ", None],
            timestamp=" 2025-01-02T03:04:05Z ",
        )
    )
    assert parsed_timestamp.scopes[0].timestamp == "2025-01-02T03:04:05Z"
    assert parsed_timestamp.scopes[0].ssp_scenarios == ["SSP1", "SSP2"]
