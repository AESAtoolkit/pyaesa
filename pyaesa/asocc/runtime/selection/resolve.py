"""Selection resolution for deterministic_asocc method plans."""

from ...methods.registry.registry import (
    REGISTRY,
    normalize_fu_code,
    resolve_user_l1_method_name,
    resolve_user_l2_method_name,
)
from .normalize import normalize_plan, resolve_level
from .pair_policy import apply_ar_pair_policy_by_plan
from .plans import (
    resolve_default_l2_plan,
    resolve_l1_plan,
    resolve_one_step_l2_plan,
    resolve_one_step_pairs_l2_plan,
    resolve_pairs_l2_plan,
    resolve_two_steps_l2_plan,
)


def resolve_l1_user(*, fu_norm: str, l1_methods: list[str] | None) -> list[str]:
    """Validate user provided canonical L1 labels."""
    if not l1_methods:
        return []
    fu_l1_kind = None
    if fu_norm.startswith("L1."):
        fu_l1_kind = "PBA" if fu_norm == "L1.b" else "CBA_FD"
    return list(
        dict.fromkeys(resolve_user_l1_method_name(name, l1_kind=fu_l1_kind) for name in l1_methods)
    )


def resolve_l2_user(*, fu_norm: str, names: list[str] | None) -> list[str]:
    """Validate user provided canonical L2 labels."""
    return [resolve_user_l2_method_name(name=name, fu_code=fu_norm) for name in (names or [])]


def resolve_pairs_user(
    *,
    fu_norm: str,
    l1_l2_pairs: list[str] | None,
) -> list[tuple[str, str]]:
    """Resolve and validate explicit L1::L2 pairs from user input."""
    resolved: list[tuple[str, str]] = []
    for pair in l1_l2_pairs or []:
        if "::" not in pair:
            raise ValueError("Each l1_l2_pairs entry must be formatted 'L1METHOD::L2METHOD'.")
        l1_name, l2_name = [p.strip() for p in pair.split("::", 1)]
        l2_res = resolve_user_l2_method_name(name=l2_name, fu_code=fu_norm)
        l1_kind = REGISTRY.l1_kind_for_l2_method(l2_res)
        l1_res = resolve_user_l1_method_name(l1_name, l1_kind=l1_kind)
        resolved.append((l2_res, l1_res))
    return resolved


def resolve_method_selection(
    *,
    fu_code: str,
    method_plan: str,
    l1_methods: list[str] | None,
    one_step_methods: list[str] | None,
    two_step_methods: list[str] | None,
    l1_l2_pairs: list[str] | None,
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """Resolve user selectors to normalized L1, two step pairs, and L2 one step."""
    fu_norm = normalize_fu_code(fu_code)
    plan = normalize_plan(method_plan)
    level = resolve_level(fu_norm=fu_norm)

    l1_user = resolve_l1_user(fu_norm=fu_norm, l1_methods=l1_methods)
    if level == "l1":
        # L1 functional units do not accept any L2-specific selectors.
        return resolve_l1_plan(
            fu_norm=fu_norm,
            plan=plan,
            l1_user=l1_user,
            one_step_methods=one_step_methods,
            two_step_methods=two_step_methods,
            l1_l2_pairs=l1_l2_pairs,
        )

    one_step_user = resolve_l2_user(fu_norm=fu_norm, names=one_step_methods)
    two_step_user = resolve_l2_user(fu_norm=fu_norm, names=two_step_methods)
    pairs_user = resolve_pairs_user(fu_norm=fu_norm, l1_l2_pairs=l1_l2_pairs)

    if plan == "default":
        l1_out, combined_out, one_step_out = resolve_default_l2_plan(
            fu_norm=fu_norm,
            l1_user=l1_user,
            one_step_methods=one_step_methods,
            two_step_methods=two_step_methods,
            l1_l2_pairs=l1_l2_pairs,
            one_step_user=one_step_user,
            two_step_user=two_step_user,
        )
    elif plan == "one_step":
        l1_out, combined_out, one_step_out = resolve_one_step_l2_plan(
            fu_norm=fu_norm,
            l1_methods=l1_methods,
            two_step_methods=two_step_methods,
            l1_l2_pairs=l1_l2_pairs,
            one_step_methods=one_step_methods,
            one_step_user=one_step_user,
        )
    elif plan == "two_steps":
        l1_out, combined_out, one_step_out = resolve_two_steps_l2_plan(
            fu_norm=fu_norm,
            l1_user=l1_user,
            one_step_methods=one_step_methods,
            l1_l2_pairs=l1_l2_pairs,
            two_step_methods=two_step_methods,
            two_step_user=two_step_user,
        )
    elif plan == "one_step_pairs":
        l1_out, combined_out, one_step_out = resolve_one_step_pairs_l2_plan(
            fu_norm=fu_norm,
            l1_methods=l1_methods,
            two_step_methods=two_step_methods,
            one_step_methods=one_step_methods,
            one_step_user=one_step_user,
            pairs_user=pairs_user,
        )
    else:
        l1_out, combined_out, one_step_out = resolve_pairs_l2_plan(
            fu_norm=fu_norm,
            l1_methods=l1_methods,
            one_step_methods=one_step_methods,
            two_step_methods=two_step_methods,
            pairs_user=pairs_user,
        )

    combined_out, one_step_out = apply_ar_pair_policy_by_plan(
        fu_norm=fu_norm,
        plan=plan,
        combined=combined_out,
        one_step=one_step_out,
    )
    return l1_out, combined_out, one_step_out
