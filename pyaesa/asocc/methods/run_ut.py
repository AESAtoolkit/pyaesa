"""UT orchestration for L2 yearly execution."""

import pandas as pd

from .compute_l2 import compute_l2_method
from .registry.registry import REGISTRY


def _ut_preweight_cache_key(
    *,
    l2_method: str,
    fu_code: str,
    year: int,
) -> tuple[str, str, int]:
    """Return the scientific cache key for one adjusted UT preweight."""
    return (str(l2_method), str(fu_code), int(year))


def _compute_ut_l2_preweight(
    *,
    context,
    l2_method: str,
    year: int,
    enacting_metric_l1: dict,
    enacting_metric_l2: dict,
    utility: dict,
) -> pd.DataFrame:
    """Compute one adjusted UT preweight from its actual formula inputs."""
    return compute_l2_method(
        l2_method=l2_method,
        fu_code=context.fu_code,
        year=year,
        l1_weights=None,
        fd_rf=enacting_metric_l1["fd_rf"],
        gva_rp=enacting_metric_l1["gva_rp"],
        fd_rp_sp_rf=enacting_metric_l2["fd_rp_sp_rf"],
        fd_rp_sp=enacting_metric_l2["fd_rp_sp"],
        fd_rf_sp=enacting_metric_l2["fd_rf_sp"],
        gva_rp_sp=enacting_metric_l2["gva_rp_sp"],
        x_to_rc=utility["x_to_rc"],
        kappa=utility["kappa"],
        omega_reg=utility["omega_reg"],
        lcia=None,
        reference_year=None,
        pre_weighting=True,
    )


def _get_ut_l2_preweight(
    *,
    context,
    state,
    ssp_scenario: str | None,
    l2_method: str,
    year: int,
    lcia_data,
    lcia_key: str | None,
    ref_year: int | None,
    enacting_metric_l1: dict,
    enacting_metric_l2: dict,
    utility: dict,
) -> pd.DataFrame:
    """Return cached UT two step preweight matrix for one method/year."""
    pw_cache_key = _ut_preweight_cache_key(
        l2_method=l2_method,
        fu_code=context.fu_code,
        year=year,
    )
    pw_cache = state.preweight_cache_by_ssp_scenario[ssp_scenario]
    if pw_cache_key not in pw_cache:
        pre = _compute_ut_l2_preweight(
            context=context,
            l2_method=l2_method,
            year=year,
            enacting_metric_l1=enacting_metric_l1,
            enacting_metric_l2=enacting_metric_l2,
            utility=utility,
        )
        pw_cache[pw_cache_key] = pre
    return pw_cache[pw_cache_key]


def _weight_ut_contribution_from_preweight(
    *,
    context,
    l2_method: str,
    year: int,
    weights: pd.Series,
    pre_weighted: pd.DataFrame,
    weight_axis: str | None = None,
) -> pd.DataFrame:
    """Apply L1 weights to a preweighted UT MRIO enacting metric without recomputing it."""
    pre_series = pre_weighted.iloc[:, 0]
    axis_name = (
        weight_axis
        if weight_axis is not None
        else REGISTRY.l2_weight_axis_for_method(l2_method, context.fu_code)
    )
    weighted = pre_series.mul(weights, level=axis_name)
    return weighted.to_frame(int(year))


def _compute_ut_weighted_contribution_from_preweight(
    *,
    context,
    state,
    ssp_scenario: str | None,
    l2_method: str,
    year: int,
    lcia_data,
    lcia_key: str | None,
    ref_year: int | None,
    weights,
    enacting_metric_l1: dict,
    enacting_metric_l2: dict,
    utility: dict,
) -> pd.DataFrame:
    """Compute unsummed two step UT contribution before L1-axis aggregation."""
    pre = _get_ut_l2_preweight(
        context=context,
        state=state,
        ssp_scenario=ssp_scenario,
        l2_method=l2_method,
        year=year,
        lcia_data=lcia_data,
        lcia_key=lcia_key,
        ref_year=ref_year,
        enacting_metric_l1=enacting_metric_l1,
        enacting_metric_l2=enacting_metric_l2,
        utility=utility,
    )
    return _weight_ut_contribution_from_preweight(
        context=context,
        l2_method=l2_method,
        year=year,
        weights=weights,
        pre_weighted=pre,
    )


def _compute_ut_weighted_from_preweight(
    *,
    context,
    state,
    ssp_scenario: str | None,
    l2_method: str,
    year: int,
    lcia_data,
    lcia_key: str | None,
    ref_year: int | None,
    weights,
    enacting_metric_l1: dict,
    enacting_metric_l2: dict,
    utility: dict,
    precomputed_contrib: pd.DataFrame | None = None,
):
    """Compute two step UT output via cached pre weight matrices.

    Args:
        context: Run context.
        state: Mutable run state.
        scenario: SSP scenario key.
        l2_method: UT(FDa) or UT(GVAa).
        year: Studied year.
        lcia_data: Optional LCIA payload.
        lcia_key: LCIA key for cache separation.
        ref_year: Optional reference year.
        weights: L1 weights series.
        enacting_metric_l1: L1 enacting metric inputs.
        enacting_metric_l2: L2 enacting metric inputs.
        utility: Utility propagation inputs.

    Returns:
        Wide DataFrame with one studied year column.
    """
    weighted_df = precomputed_contrib
    if weighted_df is None:
        weighted_df = _compute_ut_weighted_contribution_from_preweight(
            context=context,
            state=state,
            ssp_scenario=ssp_scenario,
            l2_method=l2_method,
            year=year,
            lcia_data=lcia_data,
            lcia_key=lcia_key,
            ref_year=ref_year,
            weights=weights,
            enacting_metric_l1=enacting_metric_l1,
            enacting_metric_l2=enacting_metric_l2,
            utility=utility,
        )
    weighted = weighted_df.iloc[:, 0]
    required_indices = REGISTRY.required_indices(
        l2_method,
        context.fu_code,
        l1_weighting=True,
    )
    out_series = weighted.groupby(
        level=list(required_indices),
        sort=False,
    ).sum(min_count=1)
    return out_series.to_frame(int(year))
