"""LCIA dependent compute dispatcher for L1 orchestration."""

from ....methods.registry.registry import REGISTRY
from .l1_lcia_ar import _compute_l1_ar_lcia_method
from .l1_lcia_inputs import _iter_lcia_method_inputs
from .l1_lcia_standard import _compute_l1_standard_lcia_method
from .l1_types import _L1RunContext


def _compute_l1_lcia_method(
    *,
    run: _L1RunContext,
    l1_method: str,
    lcia_by_method: dict[str, dict],
    lcia_by_method_original: dict[str, dict] | None,
    region_label_override: str | None = None,
) -> None:
    """Compute and store one LCIA dependent L1 method across payloads."""
    for lcia_inputs in _iter_lcia_method_inputs(
        run=run,
        l1_method=l1_method,
        lcia_by_method=lcia_by_method,
        lcia_by_method_original=lcia_by_method_original,
    ):
        if REGISTRY.method_family(l1_method, level="L1") in {"AR_E", "AR_ECAP"}:
            _compute_l1_ar_lcia_method(
                run=run,
                l1_method=l1_method,
                lcia_inputs=lcia_inputs,
                region_label_override=region_label_override,
            )
            continue
        _compute_l1_standard_lcia_method(
            run=run,
            l1_method=l1_method,
            lcia_inputs=lcia_inputs,
            region_label_override=region_label_override,
        )
