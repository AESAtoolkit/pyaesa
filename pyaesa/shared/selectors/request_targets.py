"""Canonical request-target selector builders reused across package families."""

from typing import Any

from pyaesa.asocc.runtime.request.scope import build_asocc_scope
from pyaesa.external_inputs.asocc.schema.contracts import merge_external_selector_methods


def build_asocc_target_selector(
    *,
    base_asocc_args: dict[str, Any],
    external_method: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the merged native plus deterministic external aSoCC target selector."""
    return (
        merge_external_selector_methods(
            target_selector=build_asocc_scope(
                base_allocate_args=base_asocc_args
            ).target_selector_payload,
            external_method=external_method,
            fu_code=str(base_asocc_args["fu_code"]),
        )
        or {}
    )


def build_io_lca_target_selector(*, base_io_lca_args: dict[str, Any]) -> dict[str, Any]:
    """Build the IO-LCA target selector from one normalized public request block."""
    selector: dict[str, Any] = {}
    if base_io_lca_args.get("years") is not None:
        selector["years"] = base_io_lca_args["years"]
    if base_io_lca_args.get("lcia_method") is not None:
        selector["methods"] = base_io_lca_args["lcia_method"]
    return selector
