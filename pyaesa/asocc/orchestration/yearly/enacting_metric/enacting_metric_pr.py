"""PR enacting metric recording."""

import pandas as pd

from ....io.metadata import EnactingMetricKey, RunContext, RunState
from .enacting_metric_common import _record_enacting_metric_input


def record_pr_enacting_metrics(
    *,
    context: RunContext,
    state: RunState,
    year: int,
    ssp_scenario: str | None,
    reg_group_map: dict[str, str],
    pop_iso: pd.Series,
    gdp_iso: pd.Series,
    iso_to_mrio: pd.Series,
) -> None:
    """Record population and GDP capita enacting metrics for PR methods."""
    # Keep WB backed years in the base file and SSP backed years in scenario files.
    # This avoids mixing heterogeneous WB/SSP units in a single CSV.
    scenario_key = None if str(int(year)) in context.wb_df.columns else ssp_scenario
    pop_key = EnactingMetricKey(metric="population", ssp_scenario=scenario_key)
    gdpcap_iso = gdp_iso.div(pop_iso.replace(0, pd.NA))
    gdpcap_iso = gdpcap_iso.replace([float("inf"), -float("inf")], pd.NA)
    gdpcap_key = EnactingMetricKey(metric="gdp_capita", ssp_scenario=scenario_key)
    if context.l1_only_no_mrio:
        _record_enacting_metric_input(
            context=context,
            state=state,
            key=pop_key,
            year=year,
            series=pop_iso.copy(),
            level="level_1",
        )
        _record_enacting_metric_input(
            context=context,
            state=state,
            key=gdpcap_key,
            year=year,
            series=gdpcap_iso.copy(),
            level="level_1",
        )
        return

    iso_idx = _build_pr_index(
        pop_index=pop_iso.index,
        iso_to_mrio=iso_to_mrio,
        reg_group_map=reg_group_map,
        include_grouped_col=bool(context.group_version_reg),
    )
    pop_iso_series = pd.Series(pop_iso.to_numpy(), index=iso_idx)
    gdpcap_series = pd.Series(gdpcap_iso.to_numpy(), index=iso_idx)
    _record_enacting_metric_input(
        context=context,
        state=state,
        key=pop_key,
        year=year,
        series=pop_iso_series,
        level="level_1",
    )
    _record_enacting_metric_input(
        context=context,
        state=state,
        key=gdpcap_key,
        year=year,
        series=gdpcap_series,
        level="level_1",
    )


def _build_pr_index(
    *,
    pop_index: pd.Index,
    iso_to_mrio: pd.Series,
    reg_group_map: dict[str, str],
    include_grouped_col: bool,
) -> pd.MultiIndex:
    """Build canonical PR enacting metric index."""
    iso_index = pop_index.astype(str)
    iso_to_mrio_map = {
        str(key): str(value) for key, value in iso_to_mrio.items() if pd.notna(value)
    }
    mapped_values = [iso_to_mrio_map.get(code, pd.NA) for code in iso_index.to_list()]
    mrio_mapped = pd.Series(mapped_values, index=iso_index, dtype="object")
    missing_mask = mrio_mapped.isna().to_numpy(dtype=bool)
    if bool(missing_mask.any()):
        missing_iso = iso_index[missing_mask].tolist()
        sample = [str(v) for v in missing_iso[:10]]
        raise ValueError(
            "Missing iso3->MRIO mapping while building PR enacting metric index. "
            f"Missing iso3 labels (sample): {sample}"
        )
    mrio_codes_original = pd.Index(mrio_mapped.astype(str), name="mrio_code")
    if include_grouped_col and reg_group_map:
        missing = sorted({str(code) for code in mrio_codes_original if code not in reg_group_map})
        if missing:
            raise ValueError(
                "Regional grouping map is missing MRIO labels referenced by PR inputs. "
                f"Missing labels (sample): {missing[:10]}"
            )
        mrio_codes_grouped = pd.Index(
            [reg_group_map[str(code)] for code in mrio_codes_original],
            name="grouped_mrio_code",
        ).astype(str)
        return pd.MultiIndex.from_arrays(
            [iso_index, mrio_codes_original, mrio_codes_grouped],
            names=["iso3_code", "mrio_code", "grouped_mrio_code"],
        )
    return pd.MultiIndex.from_arrays(
        [iso_index, mrio_codes_original],
        names=["iso3_code", "mrio_code"],
    )
