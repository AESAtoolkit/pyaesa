"""Pure shaping for enacting metric LCIA-derived metrics."""

import pandas as pd

from ....data.load_mrio import _metric_to_series
from .enacting_metric_common import _append_aggregated_mrio_code_level
from .enacting_metric_lcia_routing import _PrHrCumulativeMetricContract


def _align_population_for_lcia_frame(
    *,
    population: pd.Series,
    lcia_frame: pd.DataFrame,
) -> pd.Series:
    """Align one population series index naming to one LCIA frame."""
    aligned = population.copy()
    col_names = [str(name) for name in lcia_frame.columns.names]
    if len(col_names) == 1 and col_names[0] != "None":
        aligned.index = aligned.index.set_names(col_names[0])
    return aligned


def _shape_lcia_percap_series(
    *,
    lcia_frame: pd.DataFrame,
    population: pd.Series,
    output_metric: str,
    region_label: str,
    use_original_domain: bool,
    source_key: str,
    agg_version: str | None,
) -> pd.Series:
    """Return one shaped enacting metric LCIA per-capita series."""
    pop_aligned = _align_population_for_lcia_frame(
        population=population,
        lcia_frame=lcia_frame,
    )
    per_cap = lcia_frame.div(pop_aligned.replace(0, pd.NA), axis=1)
    per_cap = per_cap.replace([float("inf"), -float("inf")], pd.NA)
    series = _metric_to_series(output_metric, per_cap)
    if use_original_domain:
        series = _append_aggregated_mrio_code_level(
            series=series,
            region_label=region_label,
            source_key=source_key,
            agg_version=agg_version,
        )
    return series


def _shape_pr_hr_cumulative_series(
    *,
    parent_cum: dict[str, pd.Series],
    contract: _PrHrCumulativeMetricContract,
    use_original_domain: bool,
    source_key: str,
    agg_version: str | None,
) -> pd.Series:
    """Return one shaped PR-HR cumulative enacting metric series."""
    cumulative_df = pd.DataFrame.from_dict(parent_cum, orient="index")
    cumulative_df.index.name = "impact"
    cumulative_df.columns.name = contract.region_label
    series = _metric_to_series(
        contract.output_metric,
        cumulative_df,
    )
    if use_original_domain:
        series = _append_aggregated_mrio_code_level(
            series=series,
            region_label=contract.region_label,
            source_key=source_key,
            agg_version=agg_version,
        )
    return series
