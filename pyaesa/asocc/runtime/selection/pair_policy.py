"""Pair compatibility policy for method selection."""

from ...methods.registry.registry import REGISTRY


def is_l1_neutral_method(l1_method: str) -> bool:
    """Return whether an L1 method has no fixed boundary side in its name."""
    family = REGISTRY.method_family(l1_method, level="L1")
    return family in {"EG_POP", "PR_GDPCAP"}


def is_l1_compatible_with_kind(*, l1_method: str, required_kind: str) -> bool:
    """Return whether one L1 method is compatible with one required kind."""
    if is_l1_neutral_method(l1_method):
        return True
    kinds = REGISTRY.l1_kinds_for_method(l1_method)
    return required_kind in kinds


def filter_l1_for_two_step_kinds(
    *,
    l2_methods: list[str],
    l1_methods: list[str],
) -> list[str]:
    """Filter L1 set for selected two step kinds."""
    if not l2_methods:
        return []
    required_kinds = {REGISTRY.l1_kind_for_l2_method(name) for name in l2_methods}
    out: list[str] = []
    for l1_name in l1_methods:
        if any(
            is_l1_compatible_with_kind(l1_method=l1_name, required_kind=kind)
            for kind in required_kinds
        ):
            out.append(l1_name)
    return out


def validate_explicit_pairs(
    *,
    fu_norm: str,
    pairs: list[tuple[str, str]],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Validate explicit two step pairs and return normalized L1/combined lists."""
    l1_set: set[str] = set()
    combined: list[tuple[str, str]] = []
    for l2_name, l1_name in pairs:
        if not REGISTRY.has_method(
            l2_name,
            level="L2",
            fu_code=fu_norm,
            l1_weighting=True,
        ):
            raise ValueError(f"Invalid two-step L2 method in pair for {fu_norm}: '{l2_name}'.")
        l1_set.add(l1_name)
        combined.append((l2_name, l1_name))
    return sorted(l1_set), list(dict.fromkeys(combined))


def apply_ar_pair_policy_by_plan(
    *,
    fu_norm: str,
    plan: str,
    combined: list[tuple[str, str]],
    one_step: list[str],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Apply mode specific AR pair policy."""
    one_step_set = set(one_step)
    filtered_pairs: list[tuple[str, str]] = []

    for l2_name, l1_name in combined:
        l2_family = REGISTRY.method_family(l2_name, level="L2", fu_code=fu_norm)
        l1_family = REGISTRY.method_family(l1_name, level="L1")
        same_ar_pair = l2_family in {"AR_E", "AR_ECAP"} and l1_family in {
            "AR_E",
            "AR_ECAP",
        }
        if not same_ar_pair:
            filtered_pairs.append((l2_name, l1_name))
            continue
        # Canonical rule: AR(L1)::AR(L2) is represented as one step AR(L2),
        # never as a two step pair output.
        if plan in {"default", "two_steps"}:
            one_step_set.add(l2_name)
            continue
        if plan == "pairs":
            raise ValueError(
                "AR(L1)::AR(L2) is not allowed in method_plan='pairs'. "
                f"Got '{l1_name}::{l2_name}'. Use method_plan='one_step_pairs' "
                f"and put '{l2_name}' in one_step_methods."
            )
        if plan == "one_step_pairs":
            raise ValueError(
                "AR(L1)::AR(L2) is not allowed in method_plan='one_step_pairs'. "
                f"Got '{l1_name}::{l2_name}'. Remove it from l1_l2_pairs and "
                f"put '{l2_name}' in one_step_methods."
            )
        filtered_pairs.append((l2_name, l1_name))

    return filtered_pairs, sorted(one_step_set)
