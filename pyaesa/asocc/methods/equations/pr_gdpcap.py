"""PR(GDPcap) allocation method (L1)."""

import pandas as pd

from pyaesa.asocc.data.region_group_mapping import load_region_group_mapping
from pyaesa.asocc.runtime.selection.normalize import (
    normalize_l1_reg_mode_required,
)
from .share_math import normalize_share, safe_divide_series


def _map_region_code(code: str, mapping: dict[str, str]) -> str:
    """Map a MRIO code with identity fallback."""
    mapped = mapping.get(code)
    return code if mapped is None else mapped


def _to_str_mapping(series: pd.Series, *, label: str) -> dict[str, str]:
    """Convert a Series mapping to a strict str->str dictionary."""
    out: dict[str, str] = {}
    for key, value in series.items():
        out[str(key)] = str(value)
    return out


def _compute_pre_aggregated_share(
    *,
    pop_iso: pd.Series,
    gdp_iso: pd.Series,
    iso_to_mrio: pd.Series,
    source_key: str,
    group_version: str | None,
) -> pd.Series:
    """Compute PR(GDPcap) MRIO shares in pre aggregation mode."""
    pop_iso = pop_iso.copy()
    gdp_iso = gdp_iso.copy()
    pop_iso.index = pop_iso.index.map(str)
    gdp_iso.index = gdp_iso.index.map(str)
    iso_to_mrio_by_iso = _to_str_mapping(
        iso_to_mrio.reindex(pop_iso.index),
        label="ISO3->MRIO mapping",
    )
    final_mrio_by_iso = dict(iso_to_mrio_by_iso)
    if group_version:
        mapping = load_region_group_mapping(
            source_key=source_key,
            group_version=group_version,
        )
        final_mrio_by_iso = {
            iso: _map_region_code(mrio, mapping) for iso, mrio in iso_to_mrio_by_iso.items()
        }
        mapped_unique = [
            _map_region_code(mrio, mapping) for mrio in sorted(set(iso_to_mrio_by_iso.values()))
        ]
        grouped_sizes = {
            str(code): int(count)
            for code, count in pd.Series(mapped_unique, dtype="object").value_counts().items()
        }
    else:
        grouped_sizes = {}

    # Always start from ISO3 rows; only pool ISO3 upfront when their MRIO
    # regions are explicitly grouped together by the grouping map.
    pr_key_items: list[tuple[str, str]] = []
    for iso3_raw in pop_iso.index:
        iso3 = str(iso3_raw)
        grouped_code = final_mrio_by_iso[iso3]
        grouped_size = int(grouped_sizes.get(grouped_code, 0))
        if group_version and grouped_size > 1:
            pr_key_items.append(("grp", grouped_code))
        else:
            pr_key_items.append(("iso", iso3))
    pr_key = pd.MultiIndex.from_tuples(pr_key_items, names=["key_type", "key_value"])

    pop_pr = pd.Series(pop_iso.groupby(pr_key).sum(min_count=1), copy=False)
    gdp_pr = pd.Series(gdp_iso.groupby(pr_key).sum(min_count=1), copy=False)
    gdp_cap_pr = safe_divide_series(gdp_pr, pop_pr)
    inv_pr = safe_divide_series(pop_pr, gdp_cap_pr)
    share_pr = normalize_share(inv_pr).rename("share").to_frame()
    mrio_codes: list[str] = []
    for index_item in share_pr.index.tolist():
        kind = str(index_item[0])
        key = str(index_item[1])
        mrio_codes.append(key if kind == "grp" else final_mrio_by_iso[key])
    share_pr["mrio_code"] = mrio_codes
    grouped_share = share_pr.groupby("mrio_code")["share"].sum(min_count=1)
    return pd.Series(grouped_share, copy=False)


def _compute_post_aggregated_share(
    *,
    pop_iso: pd.Series,
    gdp_iso: pd.Series,
    iso_to_mrio: pd.Series,
    source_key: str,
    group_version: str | None,
) -> pd.Series:
    """Compute PR(GDPcap) MRIO shares in post aggregation mode."""
    pop_iso = pop_iso.copy()
    gdp_iso = gdp_iso.copy()
    pop_iso.index = pop_iso.index.map(str)
    gdp_iso.index = gdp_iso.index.map(str)
    iso_to_mrio_by_iso = _to_str_mapping(
        iso_to_mrio.reindex(pop_iso.index),
        label="ISO3->MRIO mapping",
    )
    gdp_cap_iso = safe_divide_series(gdp_iso, pop_iso)
    inv_iso = safe_divide_series(pop_iso, gdp_cap_iso)
    share_iso = normalize_share(inv_iso)
    mrio_codes = pd.Index(
        [iso_to_mrio_by_iso[str(iso3)] for iso3 in share_iso.index],
        name="mrio_code",
    )
    share_mrio = pd.Series(share_iso.groupby(mrio_codes).sum(min_count=1), copy=False)
    if group_version:
        mapping = load_region_group_mapping(
            source_key=source_key,
            group_version=group_version,
        )
        grouped_index = pd.Index(
            [_map_region_code(str(code), mapping) for code in share_mrio.index],
            name=share_mrio.index.name,
        )
        share_mrio = pd.Series(
            share_mrio.groupby(grouped_index).sum(min_count=1),
            copy=False,
        )
    return share_mrio


def compute_pr_gdpcap(
    *,
    pop_iso: pd.Series,
    gdp_iso: pd.Series,
    iso_to_mrio: pd.Series,
    year: int,
    source_key: str,
    group_version: str | None,
    aggregation_mode: str,
    region_label: str = "region",
) -> pd.DataFrame:
    """Compute PR(GDPcap) shares for a year (ISO3 handling).

    Args:
        pop_iso: Population by ISO3 code.
        gdp_iso: GDP by ISO3 code.
        iso_to_mrio: Mapping from ISO3 code to MRIO region code.
        year: Year of computation.
        source_key: MRIO source key.
        group_version: Grouping version tag.

    Returns:
        DataFrame of shares indexed by MRIO region code.
    """
    mode = normalize_l1_reg_mode_required(aggregation_mode)

    if mode == "pre":
        share_mrio = _compute_pre_aggregated_share(
            pop_iso=pop_iso,
            gdp_iso=gdp_iso,
            iso_to_mrio=iso_to_mrio,
            source_key=source_key,
            group_version=group_version,
        )
    else:
        share_mrio = _compute_post_aggregated_share(
            pop_iso=pop_iso,
            gdp_iso=gdp_iso,
            iso_to_mrio=iso_to_mrio,
            source_key=source_key,
            group_version=group_version,
        )

    share_mrio.index = share_mrio.index.set_names(region_label)
    return share_mrio.to_frame(int(year))
