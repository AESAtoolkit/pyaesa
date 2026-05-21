"""Method family requirements for historical coverage and enacting metric inputs."""

from typing import Optional

from pyaesa.asocc.methods.registry.model.types import MethodSpec

_HISTORY_REQUIRED_FAMILIES = {"AR_E", "AR_ECAP", "PR_HR"}
_LCIA_PERCAP_REQUIRED_FAMILIES = {"AR_ECAP", "PR_HR"}
_PR_HR_CUMULATIVE_FAMILIES = {"PR_HR"}
_L2_BASE_enacting_metric_KEYS_BY_FAMILY_FU: dict[str, dict[str, set[str]]] = {
    "UT_FD": {
        "L2.a.a": {"fd_rf", "fd_rp_sp", "fd_rp_sp_rf"},
        "L2.b.a": {"fd_rf", "fd_rp_sp_rf"},
        "L2.c.a": {"fd_rf", "fd_rf_sp"},
    },
    "UT_FDA": {
        "L2.a.b": {"fd_rf"},
        "L2.b.b": {"fd_rf"},
        "L2.c.b": {"fd_rf"},
    },
    "UT_GVAA": {
        "L2.a.b": {"gva_rp"},
        "L2.b.b": {"gva_rp"},
        "L2.c.b": {"gva_rp"},
    },
    "UT_GVA": {
        "L2.a.c": {"gva_rp", "gva_rp_sp"},
    },
    "UT_TD": {
        "L2.a.b": {"fd_rf", "x_rp_sp"},
        "L2.b.b": {"fd_rf", "x_rp_sp_rc"},
        "L2.c.b": {"fd_rf", "x_rc_sp"},
    },
}
_LCIA_L1_KEYS_BY_KIND: dict[str, set[str]] = {
    "CBA_FD": {"e_cba_fd_reg"},
    "CBA_TD": {"e_cba_fd_reg"},
    "PBA": {"e_pba_reg"},
}
_LCIA_L2_KEYS_BY_KIND: dict[str, set[str]] = {
    "CBA_FD": {"e_cba_fd_rp_sp", "e_cba_fd_rp_sp_rf", "e_cba_fd_rf_sp"},
    "CBA_TD": {"e_cba_td_rp_sp_rc", "e_cba_td_rp_sp", "e_cba_td_rc_sp"},
    "PBA": {"e_pba_rp_sp"},
}
# FU scoped LCIA L2 key routing for one step outputs (l1_weighting=False).
# Each FU uses a single canonical LCIA matrix shape for deterministic writing.
_LCIA_L2_KEYS_BY_KIND_FU_ONE_STEP: dict[str, dict[str, str]] = {
    "CBA_FD": {
        "L2.a.a": "e_cba_fd_rp_sp",
        "L2.b.a": "e_cba_fd_rp_sp_rf",
        "L2.c.a": "e_cba_fd_rf_sp",
    },
    "CBA_TD": {
        "L2.a.b": "e_cba_td_rp_sp",
        "L2.b.b": "e_cba_td_rp_sp_rc",
        "L2.c.b": "e_cba_td_rc_sp",
    },
    "PBA": {
        "L2.a.c": "e_pba_rp_sp",
    },
}
# Two step overrides (l1_weighting=True) when the LCIA matrix shape differs
# from one step for the same (kind, FU) pair.
_LCIA_L2_KEYS_BY_KIND_FU_TWO_STEP_OVERRIDE: dict[tuple[str, str], str] = {
    ("CBA_FD", "L2.a.a"): "e_cba_fd_rp_sp_rf",
}


def lcia_kinds_for_method(
    *,
    methods: list[MethodSpec],
    name: str,
    level: str,
    fu_code: Optional[str],
    l1_weighting: Optional[bool],
) -> set[str]:
    """Resolve canonical LCIA boundary kinds for one method selection."""
    kinds: set[str] = set()
    for method_spec in methods:
        if method_spec.name != name or method_spec.level != level:
            continue
        if fu_code is not None and method_spec.fu_code != fu_code:
            continue
        if l1_weighting is not None and method_spec.l1_weighting != l1_weighting:
            continue
        if method_spec.l1_kind is not None:
            kinds.add(method_spec.l1_kind)
    return kinds


def method_requires_contiguous_history(*, family: str) -> bool:
    """Return whether this family needs contiguous historical MRIO coverage."""
    return family in _HISTORY_REQUIRED_FAMILIES


def method_requires_lcia_percap(*, family: str) -> bool:
    """Return whether this family needs LCIA per capita enacting metrics."""
    return family in _LCIA_PERCAP_REQUIRED_FAMILIES


def method_requires_pr_hr_cumulative(*, family: str) -> bool:
    """Return whether this family needs PR-HR cumulative per capita metrics."""
    return family in _PR_HR_CUMULATIVE_FAMILIES


def l2_base_enacting_metrics(*, family: str, fu_code: str) -> tuple[str, ...]:
    """Return base (non LCIA) enacting metric keys required by one L2 method family."""
    family_map = _L2_BASE_enacting_metric_KEYS_BY_FAMILY_FU.get(family, {})
    keys = family_map.get(fu_code, set())
    return tuple(sorted(keys))


def lcia_enacting_metric_l1_metrics(*, lcia_kinds: set[str]) -> tuple[str, ...]:
    """Return level-1 LCIA enacting metric keys required by LCIA boundary kinds."""
    keys: set[str] = set()
    for kind in lcia_kinds:
        keys.update(_LCIA_L1_KEYS_BY_KIND.get(kind, set()))
    return tuple(sorted(keys))


def lcia_enacting_metric_l2_metrics(
    *,
    lcia_kinds: set[str],
    fu_code: str | None = None,
    l1_weighting: bool | None = None,
) -> tuple[str, ...]:
    """Return level-2 LCIA enacting metric keys required by LCIA boundary kinds.

    Resolution modes:
    - `fu_code is None`: return the union for each LCIA kind (generic mode).
    - `fu_code is not None` and `l1_weighting is False`: return the FU one step key.
    - `fu_code is not None` and `l1_weighting is True`: return the FU two step key
      (or the one step key when no override is defined).
    - `fu_code is not None` and `l1_weighting is None`: return both keys for that FU,
      used by callers that need the full FU scoped LCIA L2 key set.
    """
    keys: set[str] = set()
    for kind in lcia_kinds:
        if fu_code is None:
            keys.update(_LCIA_L2_KEYS_BY_KIND.get(kind, set()))
            continue
        fu_map = _LCIA_L2_KEYS_BY_KIND_FU_ONE_STEP.get(kind, {})
        if fu_code not in fu_map:
            raise ValueError(
                f"Missing LCIA L2 enacting metric mapping for kind='{kind}', fu_code='{fu_code}'."
            )
        one_step_key = fu_map[fu_code]
        two_step_key = _LCIA_L2_KEYS_BY_KIND_FU_TWO_STEP_OVERRIDE.get(
            (kind, fu_code),
            one_step_key,
        )
        if l1_weighting is None:
            keys.add(one_step_key)
            keys.add(two_step_key)
            continue
        keys.add(two_step_key if bool(l1_weighting) else one_step_key)
    return tuple(sorted(keys))
