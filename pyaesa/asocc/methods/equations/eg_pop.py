"""EG(Pop) allocation method (L1)."""

import pandas as pd


def compute_eg_pop(
    *,
    population: pd.Series,
    year: int,
    region_label: str = "region",
) -> pd.DataFrame:
    """Compute EG(Pop) shares for a year.

    Args:
        population: Population series by region.
        year: Year of computation.

    Returns:
        DataFrame of shares indexed by region.
    """
    if population.index.name != region_label:
        population = population.copy()
        population.index = population.index.set_names(region_label)
    total = population.sum()
    if total == 0:
        share = population * 0.0
    else:
        share = population / total
    return share.to_frame(int(year))
