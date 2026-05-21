"""Public registry facade for allocation methods."""

from pyaesa.asocc.methods.registry.queries.queries import MethodRegistry
from pyaesa.asocc.methods.registry.build.build import build_method_specs
from pyaesa.asocc.methods.registry.model.types import MethodSpec, normalize_fu_code
from pyaesa.asocc.methods.registry.queries.resolve import (
    resolve_required_indices as _resolve_required_indices,
    resolve_user_l1_method_name as _resolve_user_l1_method_name,
    resolve_user_l2_method_name as _resolve_user_l2_method_name,
)

_RAW_METHOD_SPECS = build_method_specs(
    normalize_fu_code=normalize_fu_code,
)
# Freeze declarative specs into typed MethodSpec rows once at import time.
METHOD_SPECS: list[MethodSpec] = [MethodSpec(**spec) for spec in _RAW_METHOD_SPECS]

REGISTRY = MethodRegistry(METHOD_SPECS)


def resolve_user_l1_method_name(
    name: str,
    *,
    l1_kind: str | None = None,
) -> str:
    """Validate and return one canonical L1 registry method label."""
    return _resolve_user_l1_method_name(name, l1_kind=l1_kind)


def resolve_user_l2_method_name(
    *,
    name: str,
    fu_code: str,
) -> str:
    """Validate and return one canonical L2 registry method label."""
    return _resolve_user_l2_method_name(name=name, fu_code=fu_code)


def resolve_required_indices(
    *,
    fu_code: str,
    selected_l1: list[str],
    combined: list[tuple[str, str]],
    selected_l2_one_step: list[str],
    l1_kinds_needed: set[str],
) -> set[str]:
    """Resolve required indices over all selected methods."""
    return _resolve_required_indices(
        fu_code=fu_code,
        selected_l1=selected_l1,
        combined=combined,
        selected_l2_one_step=selected_l2_one_step,
        l1_kinds_needed=l1_kinds_needed,
        registry=REGISTRY,
        normalize_fu_code=normalize_fu_code,
    )
