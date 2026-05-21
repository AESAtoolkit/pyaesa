"""LCIA input resolution used by L1 LCIA compute."""

from ....data.reference_payloads import ensure_pr_hr_child_impact_timeseries_loaded
from ....methods.registry.registry import REGISTRY
from .l1_lcia_reference_policy import (
    emit_pr_hr_lcia_freeze_notice_if_needed,
    resolve_latest_lcia_year_for_method_kind,
)
from .l1_types import _L1RunContext, _LciaMethodInputs


def _iter_lcia_kinds(l1_method: str, context) -> list[str]:
    """Return applicable LCIA kinds for one method."""
    del context
    return REGISTRY.l1_kinds_for_method(l1_method)


def _iter_lcia_method_inputs(
    *,
    run: _L1RunContext,
    l1_method: str,
    lcia_by_method: dict[str, dict],
    lcia_by_method_original: dict[str, dict] | None,
) -> list[_LciaMethodInputs]:
    """Build typed LCIA payload records for one method."""
    out: list[_LciaMethodInputs] = []
    use_original_domain = run.context.use_original_l1_post_domain and REGISTRY.method_family(
        l1_method, level="L1"
    ) in {"PR_HR", "AR_ECAP"}
    active_lcia_by_method = lcia_by_method_original if use_original_domain else lcia_by_method
    if active_lcia_by_method is None:
        active_lcia_by_method = {}
    lcia_methods = list(run.context.lcia_methods or active_lcia_by_method.keys())
    for lcia_method in lcia_methods:
        method_payload = active_lcia_by_method.get(lcia_method, {})
        for lcia_kind in _iter_lcia_kinds(l1_method, run.context):
            payload_key = "e_cba_fd_reg" if lcia_kind == "CBA_FD" else "e_pba_reg"
            lcia_reg = method_payload.get(payload_key) if isinstance(method_payload, dict) else None
            latest_year, latest_reg, _last_error = resolve_latest_lcia_year_for_method_kind(
                run=run,
                lcia_method=lcia_method,
                lcia_kind=lcia_kind,
                use_original_domain=use_original_domain,
            )
            if latest_reg is None or latest_year is None:
                continue
            impact_year = latest_year
            if lcia_reg is None:
                lcia_reg = latest_reg
            lcia_reg_by_year = None
            rps_df = None
            impact_parent_map = None
            if "PR-HR" in l1_method:
                lcia_reg_by_year = ensure_pr_hr_child_impact_timeseries_loaded(
                    context=run.context,
                    state=run.state,
                    through_year=run.year,
                    lcia_method=lcia_method,
                    lcia_kind=lcia_kind,
                    use_original_domain=use_original_domain,
                )
                last_timeseries_year = max(int(y) for y in lcia_reg_by_year.keys())
                emit_pr_hr_lcia_freeze_notice_if_needed(
                    run=run,
                    lcia_method=lcia_method,
                    last_timeseries_year=last_timeseries_year,
                )
                rps_df = run.state.rps_by_method.get(lcia_method)
                impact_parent_map = run.state.cf_by_method.get(lcia_method)
            out.append(
                _LciaMethodInputs(
                    lcia_method=lcia_method,
                    lcia_kind=lcia_kind,
                    lcia_reg=lcia_reg,
                    lcia_reg_by_year=lcia_reg_by_year,
                    rps_df=rps_df,
                    impact_parent_map=impact_parent_map,
                    resolved_name=l1_method,
                    impact_year=impact_year,
                )
            )
    return out
