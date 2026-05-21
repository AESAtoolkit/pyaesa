"""L1 LCIA reference-availability policy and notice flow."""

import pandas as pd

from ....data.reference_payloads import load_reference_lcia_reg
from ....data.source_schema import default_historical_cutoff_for_source
from ...method_scope import _max_historical_mrio_year
from ...common_formatting import format_year_scope
from ..shared.scenario_routing import emit_notice
from .l1_types import _L1RunContext, _LciaMethodInputs


def resolve_latest_lcia_year_for_method_kind(
    *,
    run: _L1RunContext,
    lcia_method: str,
    lcia_kind: str,
    use_original_domain: bool,
) -> tuple[int | None, pd.DataFrame | None, str | None]:
    """Return latest year<=studied year with available LCIA for one method/kind."""
    last_error: str | None = None
    group_version = None if use_original_domain else run.context.group_version
    candidates = sorted(
        (y for y in run.context.historical_years if y <= run.year),
        reverse=True,
    )
    for candidate_year in candidates:
        try:
            lcia_reg = load_reference_lcia_reg(
                context=run.context,
                state=run.state,
                ref_year=int(candidate_year),
                lcia_method=lcia_method,
                lcia_kind=lcia_kind,
                group_version=group_version,
            )
            return int(candidate_year), lcia_reg, None
        except ValueError as exc:
            last_error = str(exc)
            continue
    return None, None, last_error


def emit_pr_hr_lcia_freeze_notice_if_needed(
    *,
    run: _L1RunContext,
    lcia_method: str,
    last_timeseries_year: int,
) -> None:
    """Emit the PR-HR LCIA freeze notice when historical LCIA does not cover the year."""
    if int(run.year) <= int(last_timeseries_year):
        return

    max_hist_year = _max_historical_mrio_year(historical_years=run.context.historical_years)
    max_requested_year = max((int(y) for y in run.context.resolved_years), default=run.year)
    has_future_mrio_gap = max_hist_year is not None and max_requested_year > int(max_hist_year)
    lcia_gap_key_prefix = f"pr-hr-lcia-gap:{lcia_method}:"
    if max_hist_year is not None and int(run.year) > int(max_hist_year):
        if any(str(key).startswith(lcia_gap_key_prefix) for key in run.state.notices_emitted):
            return
        emit_notice(
            context=run.context,
            state=run.state,
            key=f"pr-hr-no-mrio:{lcia_method}:{max_hist_year}",
            message=(
                "Historical responsibility (PR-HR) cumulative impact per capita is frozen at "
                f"last available LCIA year {last_timeseries_year}. Method={lcia_method}. "
                f"Reason: MRIO data is missing after {max_hist_year}."
            ),
        )
        return

    gap_year = int(last_timeseries_year) + 1
    year_reason = run.state.skipped_years.get(gap_year, {})
    reason = year_reason.get(lcia_method) if isinstance(year_reason, dict) else None
    lcia_missing_years = sorted(
        int(y)
        for y in run.context.resolved_years
        if int(y) >= int(gap_year) and (max_hist_year is None or int(y) <= int(max_hist_year))
    )
    scope = format_year_scope(lcia_missing_years)
    reason_msg = f"{reason} {scope}" if reason else f"LCIA data is missing {scope}"
    mrio_suffix = (
        f" MRIO data is missing after {max_hist_year}."
        if has_future_mrio_gap and max_hist_year is not None
        else ""
    )
    emit_notice(
        context=run.context,
        state=run.state,
        key=f"pr-hr-lcia-gap:{lcia_method}:{gap_year}",
        message=(
            "Historical responsibility (PR-HR) cumulative impact per capita is frozen at last "
            f"available LCIA year {last_timeseries_year}. Method={lcia_method}. "
            f"Reason: {reason_msg}.{mrio_suffix}"
        ),
    )


def resolve_reference_years_for_ar(
    *,
    run: _L1RunContext,
    lcia_inputs: _LciaMethodInputs,
    use_original_domain: bool,
) -> list[int]:
    """Return AR reference years clipped to available LCIA years."""
    default_cutoff = default_historical_cutoff_for_source(run.context.source)
    refs_raw = run.context.reference_years or [
        y
        for y in run.context.historical_years
        if default_cutoff is None or int(y) <= default_cutoff
    ]
    group_version = None if use_original_domain else run.context.group_version
    cache_key = (
        str(lcia_inputs.lcia_method),
        str(lcia_inputs.lcia_kind),
        bool(use_original_domain),
        group_version,
        tuple(int(y) for y in refs_raw),
    )
    cached = run.state.ar_valid_refs_cache.get(cache_key)
    if cached is None:
        refs_valid: list[int] = []
        refs_dropped: list[tuple[int, str]] = []
        for ref_year in refs_raw:
            try:
                load_reference_lcia_reg(
                    context=run.context,
                    state=run.state,
                    ref_year=int(ref_year),
                    lcia_method=lcia_inputs.lcia_method,
                    lcia_kind=lcia_inputs.lcia_kind,
                    group_version=group_version,
                )
                refs_valid.append(int(ref_year))
            except ValueError as exc:
                refs_dropped.append((int(ref_year), str(exc)))
        run.state.ar_valid_refs_cache[cache_key] = (refs_valid, refs_dropped)
    else:
        refs_valid, refs_dropped = cached

    max_available_ref = max(refs_valid) if refs_valid else None
    display_max_ref = (
        max_available_ref
        if max_available_ref is not None
        else _max_historical_mrio_year(historical_years=run.context.historical_years)
    )
    max_hist_year = _max_historical_mrio_year(historical_years=run.context.historical_years)
    max_requested_year = max((int(y) for y in run.context.resolved_years), default=run.year)
    has_future_mrio_gap = max_hist_year is not None and max_requested_year > int(max_hist_year)

    if (
        run.context.reference_years is None
        and display_max_ref is not None
        and run.year > display_max_ref
    ):
        first_affected_year = int(display_max_ref) + 1
        if max_hist_year is not None and run.year > max_hist_year:
            emit_notice(
                context=run.context,
                state=run.state,
                key=f"ar-ref-no-mrio:{lcia_inputs.lcia_method}:{max_hist_year}",
                message=(
                    "Max acquired rights (AR) reference year is "
                    f"last available LCIA year {display_max_ref}. "
                    f"Method={lcia_inputs.lcia_method}. "
                    f"Reason: MRIO data is missing after {max_hist_year}."
                ),
            )
        else:
            first_dropped_msg = next(
                (
                    str(dropped_msg)
                    for dropped_year, dropped_msg in refs_dropped
                    if int(dropped_year) == first_affected_year
                ),
                None,
            )
            dropped_reason = (
                first_dropped_msg.split("Reason:", 1)[1].strip()
                if first_dropped_msg is not None and "Reason:" in first_dropped_msg
                else None
            )
            if dropped_reason:
                reason_years = sorted(
                    int(y)
                    for y, msg in refs_dropped
                    if "Reason:" in msg and msg.split("Reason:", 1)[1].strip() == dropped_reason
                )
                reason = f"{dropped_reason} {format_year_scope(reason_years)}"
            else:
                reason = f"LCIA data is missing {format_year_scope([first_affected_year])}"
            mrio_suffix = (
                f" MRIO data is missing after {max_hist_year}."
                if has_future_mrio_gap and max_hist_year is not None
                else ""
            )
            emit_notice(
                context=run.context,
                state=run.state,
                key=f"ar-ref-lcia-gap:{lcia_inputs.lcia_method}:{first_affected_year}",
                message=(
                    "Max acquired rights (AR) reference year is "
                    f"last available LCIA year {display_max_ref}. "
                    f"Method={lcia_inputs.lcia_method}. Reason: {reason}.{mrio_suffix}"
                ),
            )
    elif run.context.reference_years is not None and refs_dropped and display_max_ref is not None:
        emit_notice(
            context=run.context,
            state=run.state,
            key=f"ar-ref-clipped-requested:{lcia_inputs.lcia_method}",
            message=(
                "Requested acquired rights (AR) reference_year values above available LCIA "
                f"years were ignored. Max available reference year is {display_max_ref}. "
                f"Method={lcia_inputs.lcia_method}"
            ),
        )
    return refs_valid


def emit_ar_no_reference_years_notice(
    *,
    run: _L1RunContext,
    l1_method: str,
    lcia_inputs: _LciaMethodInputs,
) -> None:
    """Emit the AR no-reference-years notice for one studied year."""
    emit_notice(
        context=run.context,
        state=run.state,
        key=f"ar-no-refs:{lcia_inputs.lcia_method}",
        message=(
            "No LCIA reference years available for AR computation "
            f"({l1_method}, lcia_method={lcia_inputs.lcia_method}, kind={lcia_inputs.lcia_kind}) "
            f"at studied year {run.year}."
        ),
    )
