"""AR(Ecap) allocation method (L1)."""

import numpy as np
import pandas as pd

from .ar_e import _stack_array_to_year


def _duplicated_label_sample(series: pd.Series) -> list[str]:
    """Return a short sample of duplicated labels from one Series index."""
    duplicates = pd.Index(series.index[series.index.duplicated()]).unique().tolist()
    return [str(value) for value in duplicates[:10]]


def compute_ar_ecap_l1(
    *,
    year: int,
    population: pd.Series,
    population_ref: pd.Series | None,
    lcia_reg: pd.DataFrame | None,
    lcia_reg_by_year: dict[int, pd.DataFrame] | None,
    reference_year: int | None,
    region_label: str = "region",
    index_cache: dict[tuple[object, ...], object] | None = None,
) -> pd.DataFrame:
    """Compute AR(Ecap) for L1.

    Args:
        year: Studied year (output column label).
        population: Population series by region (studied year).
        population_ref: Population series by region (reference year).
        lcia_reg: LCIA impacts by region for current year.
        lcia_reg_by_year: LCIA time series by year.
        reference_year: Reference year.

    Returns:
        DataFrame of AR shares indexed by impact and region.
    """
    if reference_year is None:
        raise ValueError("reference_year is required for AR methods.")
    if lcia_reg is None and lcia_reg_by_year is None:
        raise ValueError("LCIA regional impacts required for AR.")
    impacts = lcia_reg_by_year.get(reference_year) if lcia_reg_by_year else lcia_reg
    if impacts is None:
        raise ValueError("LCIA impacts missing for reference year.")
    if population_ref is None:
        raise ValueError("reference-year population required for AR(Ecap).")
    if not population_ref.index.is_unique:
        sample = _duplicated_label_sample(population_ref)
        raise ValueError(
            f"reference-year population has duplicate region labels. Duplicate labels: {sample}"
        )
    if not population.index.is_unique:
        sample = _duplicated_label_sample(population)
        raise ValueError(
            f"studied-year population has duplicate region labels. Duplicate labels: {sample}"
        )
    impact_values = impacts.to_numpy(dtype="float64", copy=False)
    population_ref_values = population_ref.reindex(impacts.columns).to_numpy(
        dtype="float64",
        na_value=float("nan"),
        copy=False,
    )
    population_values = population.reindex(impacts.columns).to_numpy(
        dtype="float64",
        na_value=float("nan"),
        copy=False,
    )
    per_cap = np.full(impact_values.shape, np.nan, dtype=np.float64)
    valid_per_cap = ~np.isnan(impact_values) & ~np.isnan(population_ref_values[np.newaxis, :])
    valid_per_cap &= population_ref_values[np.newaxis, :] != 0.0
    np.divide(
        impact_values,
        population_ref_values[np.newaxis, :],
        out=per_cap,
        where=valid_per_cap,
    )
    scaled = per_cap * population_values[np.newaxis, :]
    denominator = np.nansum(scaled, axis=1)
    share_values = np.full(scaled.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(scaled) & (denominator[:, np.newaxis] != 0.0)
    np.divide(scaled, denominator[:, np.newaxis], out=share_values, where=valid)
    share_values[~np.isfinite(share_values)] = np.nan
    return _stack_array_to_year(
        share_values,
        row_index=impacts.index,
        columns=impacts.columns,
        col_names=[region_label],
        year=year,
        index_cache=index_cache,
    )
