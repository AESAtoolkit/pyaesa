"""Plan-specific method selection resolution."""

from ...methods.registry.registry import REGISTRY
from .pair_policy import (
    filter_l1_for_two_step_kinds,
    is_l1_compatible_with_kind,
    validate_explicit_pairs,
)


def compatible_l1_for_l1_fu(*, fu_norm: str) -> list[str]:
    """Return L1 methods compatible with an L1 functional unit."""
    all_l1 = REGISTRY.list_l1_methods()
    if fu_norm == "L1.a":
        return sorted(
            m for m in all_l1 if ("PBA" not in m) or m.startswith("EG(") or m.startswith("PR(")
        )
    if fu_norm == "L1.b":
        return sorted(
            m for m in all_l1 if ("CBA_FD" not in m) or m.startswith("EG(") or m.startswith("PR(")
        )
    return all_l1


def validate_l2_selection(
    *,
    fu_norm: str,
    l1_weighting: bool,
    selection: list[str] | None,
    resolved_selection: list[str],
    label: str,
) -> list[str]:
    """Validate L2 selection list (or return full compatible set)."""
    if selection is None:
        return sorted(REGISTRY.list_l2_methods(fu_code=fu_norm, l1_weighting=l1_weighting))
    valid: list[str] = []
    for name in resolved_selection:
        if not REGISTRY.has_method(
            name,
            level="L2",
            fu_code=fu_norm,
            l1_weighting=l1_weighting,
        ):
            raise ValueError(f"L2 {label} method '{name}' is not valid for {fu_norm}.")
        valid.append(name)
    return sorted(set(valid))


def resolve_l1_for_two_step(l1_user: list[str]) -> list[str]:
    """Resolve L1 method set to use on two step side."""
    if l1_user:
        return sorted(set(l1_user))
    return REGISTRY.list_l1_methods()


def build_cartesian_pairs(
    *,
    l2_methods: list[str],
    l1_methods: list[str],
) -> list[tuple[str, str]]:
    """Build kind compatible L2xL1 two step selection."""
    out: list[tuple[str, str]] = []
    for l2_name in l2_methods:
        required_kind = REGISTRY.l1_kind_for_l2_method(l2_name)
        compatible = [
            l1_name
            for l1_name in l1_methods
            if is_l1_compatible_with_kind(
                l1_method=l1_name,
                required_kind=required_kind,
            )
        ]
        if not compatible:
            raise ValueError(
                "No compatible L1 methods found for two-step L2 method "
                f"'{l2_name}' (requires L1 kind '{required_kind}')."
            )
        out.extend((l2_name, l1_name) for l1_name in compatible)
    return out


def resolve_l1_plan(
    *,
    fu_norm: str,
    plan: str,
    l1_user: list[str],
    one_step_methods: list[str] | None,
    two_step_methods: list[str] | None,
    l1_l2_pairs: list[str] | None,
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Resolve method selection for L1 functional units."""
    if plan != "default":
        raise ValueError(
            "method_plan is not applicable for L1 fu_code. "
            "Use method_plan='default' and control L1 subset with l1_methods."
        )
    if any(value is not None for value in (one_step_methods, two_step_methods, l1_l2_pairs)):
        raise ValueError("L2 selector arguments are not valid for L1 fu_code.")
    if l1_user:
        return sorted(set(l1_user)), [], []
    return compatible_l1_for_l1_fu(fu_norm=fu_norm), [], []


def resolve_default_l2_plan(
    *,
    fu_norm: str,
    l1_user: list[str],
    one_step_methods: list[str] | None,
    two_step_methods: list[str] | None,
    l1_l2_pairs: list[str] | None,
    one_step_user: list[str],
    two_step_user: list[str],
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Resolve default L2 plan."""
    if l1_l2_pairs is not None:
        raise ValueError(
            "method_plan='default' does not accept l1_l2_pairs. "
            "Use method_plan='pairs' or method_plan='one_step_pairs'."
        )
    one_step = validate_l2_selection(
        fu_norm=fu_norm,
        l1_weighting=False,
        selection=one_step_methods,
        resolved_selection=one_step_user,
        label="one-step",
    )
    two_step = validate_l2_selection(
        fu_norm=fu_norm,
        l1_weighting=True,
        selection=two_step_methods,
        resolved_selection=two_step_user,
        label="two-step",
    )
    if not two_step:
        return [], [], one_step
    # In default mode, l1_methods only constrains the two step cartesian side.
    l1_for_two = resolve_l1_for_two_step(l1_user)
    l1_for_two = filter_l1_for_two_step_kinds(
        l2_methods=two_step,
        l1_methods=l1_for_two,
    )
    combined = build_cartesian_pairs(l2_methods=two_step, l1_methods=l1_for_two)
    l1_effective = sorted({pair[1] for pair in combined})
    return (l1_effective, combined, one_step)


def resolve_one_step_l2_plan(
    *,
    fu_norm: str,
    l1_methods: list[str] | None,
    two_step_methods: list[str] | None,
    l1_l2_pairs: list[str] | None,
    one_step_methods: list[str] | None,
    one_step_user: list[str],
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Resolve one step only L2 plan."""
    if any(value is not None for value in (l1_methods, two_step_methods, l1_l2_pairs)):
        raise ValueError("method_plan='one_step' only accepts one_step_methods.")
    one_step = validate_l2_selection(
        fu_norm=fu_norm,
        l1_weighting=False,
        selection=one_step_methods,
        resolved_selection=one_step_user,
        label="one-step",
    )
    return [], [], one_step


def resolve_two_steps_l2_plan(
    *,
    fu_norm: str,
    l1_user: list[str],
    one_step_methods: list[str] | None,
    l1_l2_pairs: list[str] | None,
    two_step_methods: list[str] | None,
    two_step_user: list[str],
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Resolve two steps only L2 plan."""
    if one_step_methods is not None or l1_l2_pairs is not None:
        raise ValueError("method_plan='two_steps' does not accept one_step_methods or l1_l2_pairs.")
    two_step = validate_l2_selection(
        fu_norm=fu_norm,
        l1_weighting=True,
        selection=two_step_methods,
        resolved_selection=two_step_user,
        label="two-step",
    )
    l1_for_two = resolve_l1_for_two_step(l1_user)
    l1_for_two = filter_l1_for_two_step_kinds(
        l2_methods=two_step,
        l1_methods=l1_for_two,
    )
    combined = build_cartesian_pairs(l2_methods=two_step, l1_methods=l1_for_two)
    l1_effective = sorted({pair[1] for pair in combined})
    return l1_effective, combined, []


def resolve_one_step_pairs_l2_plan(
    *,
    fu_norm: str,
    l1_methods: list[str] | None,
    two_step_methods: list[str] | None,
    one_step_methods: list[str] | None,
    one_step_user: list[str],
    pairs_user: list[tuple[str, str]],
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Resolve L2 plan with one step plus explicit pairs."""
    if two_step_methods is not None or l1_methods is not None:
        raise ValueError(
            "method_plan='one_step_pairs' accepts only one_step_methods and l1_l2_pairs."
        )
    if not pairs_user:
        raise ValueError(
            "method_plan='one_step_pairs' requires at least one 'L1::L2' entry in l1_l2_pairs."
        )
    one_step = validate_l2_selection(
        fu_norm=fu_norm,
        l1_weighting=False,
        selection=one_step_methods,
        resolved_selection=one_step_user,
        label="one-step",
    )
    l1_set, combined = validate_explicit_pairs(fu_norm=fu_norm, pairs=pairs_user)
    return l1_set, combined, one_step


def resolve_pairs_l2_plan(
    *,
    fu_norm: str,
    l1_methods: list[str] | None,
    one_step_methods: list[str] | None,
    two_step_methods: list[str] | None,
    pairs_user: list[tuple[str, str]],
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Resolve explicit pairs only L2 plan."""
    if any(value is not None for value in (l1_methods, one_step_methods, two_step_methods)):
        raise ValueError("method_plan='pairs' only accepts l1_l2_pairs.")
    if not pairs_user:
        raise ValueError("method_plan='pairs' requires at least one 'L1::L2' entry in l1_l2_pairs.")
    l1_set, combined = validate_explicit_pairs(fu_norm=fu_norm, pairs=pairs_user)
    return l1_set, combined, []
