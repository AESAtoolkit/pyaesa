"""Static method spec composition for the allocation method registry."""

from pyaesa.asocc.methods.registry.specs.l1 import build_raw_l1_method_specs
from pyaesa.asocc.methods.registry.specs.l2 import build_raw_l2_method_specs


def build_raw_method_specs() -> list[dict[str, object]]:
    """Build all raw method spec payloads as dictionaries."""
    # Keep composition explicit so L1/L2 spec files can evolve independently.
    return build_raw_l1_method_specs() + build_raw_l2_method_specs()
