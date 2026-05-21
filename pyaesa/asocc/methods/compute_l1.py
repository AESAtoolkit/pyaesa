"""Compute L1 allocation methods."""

from collections.abc import Callable
from typing import cast

import pandas as pd

from ..data.region_group_mapping import load_region_group_mapping
from .equations.ar_e import compute_ar_e_l1
from .equations.ar_ecap import compute_ar_ecap_l1
from .equations.eg_pop import compute_eg_pop
from .equations.pr_gdpcap import compute_pr_gdpcap
from .equations.pr_hr_ecap_cum import compute_pr_hr
from .registry.registry import REGISTRY
from ..runtime.selection.normalize import normalize_l1_reg_mode_required


def _aggregate_l1_regions_post(
    *,
    frame: pd.DataFrame,
    source_key: str,
    group_version_reg: str,
    region_label: str,
) -> pd.DataFrame:
    """Map original MRIO regions to grouped labels and sum duplicate rows."""
    # Post aggregation is only used for methods where shares are computed on
    # original regions first, then rolled up to grouped regions.
    mapping = load_region_group_mapping(
        source_key=source_key,
        group_version=group_version_reg,
    )
    if isinstance(frame.index, pd.MultiIndex):
        if region_label not in frame.index.names:
            raise ValueError(
                f"Cannot post-aggregate L1 output: missing region level '{region_label}'."
            )
        regions = frame.index.get_level_values(region_label).map(
            lambda code: mapping.get(code, code)
        )
        new_levels: list[pd.Index] = []
        new_names: list[str] = []
        for level_pos, idx_name in enumerate(frame.index.names):
            new_names.append(str(idx_name))
            if str(idx_name) == region_label:
                new_levels.append(pd.Index(regions, name=region_label))
            else:
                new_levels.append(frame.index.get_level_values(level_pos))
        out = frame.copy()
        out.index = pd.MultiIndex.from_arrays(new_levels, names=new_names)
        grouped = out.groupby(level=new_names, sort=False).sum(min_count=1)
        return cast(pd.DataFrame, grouped)

    if str(frame.index.name) != region_label:
        raise ValueError(
            "Cannot post-aggregate L1 output: unexpected index name "
            f"'{frame.index.name}', expected '{region_label}'."
        )
    out = frame.copy()
    out.index = out.index.map(lambda code: mapping.get(code, code))
    grouped = out.groupby(level=0, sort=False).sum(min_count=1)
    return cast(pd.DataFrame, grouped)


def resolve_l1_region_label(*, l1_method: str, fu_code: str) -> str:
    """Resolve canonical L1 region axis name for a method/FU context."""
    family = REGISTRY.method_family(l1_method, level="L1")
    if family in {"AR_E", "AR_ECAP", "PR_HR"}:
        if "PBA" in l1_method:
            return "r_p"
        return "r_f"
    if fu_code == "L1.b":
        return "r_p"
    if fu_code.startswith("L2."):
        two_step = REGISTRY.list_l2_methods(fu_code=fu_code, l1_weighting=True)
        kind = REGISTRY.l1_kind_for_l2_method(two_step[0])
        return "r_p" if kind == "PBA" else "r_f"
    return "r_f"


def compute_l1_method(
    *,
    l1_method: str,
    fu_code: str,
    year: int,
    population: pd.Series,
    population_by_year: dict[int, pd.Series] | None,
    population_ref: pd.Series | None,
    pr_pop: pd.Series | None,
    pr_gdp: pd.Series | None,
    pr_to_mrio: pd.Series | None,
    source_key: str,
    group_version_reg: str | None,
    l1_reg_aggreg: str,
    region_label_override: str | None = None,
    lcia_reg: pd.DataFrame | None,
    lcia_reg_by_year: dict[int, pd.DataFrame] | None,
    rps_df: pd.DataFrame | None,
    impact_parent_map: pd.Series | None,
    available_years: list[int],
    reference_year: int | None,
    impact_year: int | None = None,
    pr_hr_parent_cum_cache: dict[int, dict[str, pd.Series]] | None = None,
    pr_hr_fallback_callback: Callable[[list[str], int, int], None] | None = None,
    index_cache: dict[tuple[object, ...], object] | None = None,
) -> pd.DataFrame:
    """Compute one L1 method result for one studied year.

    Args:
        l1_method: L1 method name.
        fu_code: Functional unit code.
        year: Studied year.
        population: Population series by region.
        population_by_year: Historical population series by year.
        population_ref: Population series at AR reference year.
        pr_pop: Population by ISO3 code.
        pr_gdp: GDP by ISO3 code.
        pr_to_mrio: Mapping from ISO3 code to MRIO region.
        source_key: MRIO source key.
        group_version_reg: Region grouping tag.
        l1_reg_aggreg: L1 aggregation mode ("pre" or "post") for methods
            where grouping timing matters.
        region_label_override: Explicit region axis label override.
        lcia_reg: LCIA regional impacts for current year.
        lcia_reg_by_year: LCIA time series by year.
        rps_df: Responsibility period settings.
        impact_parent_map: Mapping from impact child to parent.
        available_years: Available LCIA years for responsibility windows.
        reference_year: Reference year for AR methods.
        impact_year: Optional LCIA year to use for PR-HR cumulative impacts.

    Returns:
        Wide DataFrame indexed by region or (impact, region) with one year column.
    """
    region_label = (
        str(region_label_override)
        if region_label_override is not None
        else resolve_l1_region_label(
            l1_method=l1_method,
            fu_code=fu_code,
        )
    )

    family = REGISTRY.method_family(l1_method, level="L1")
    # Family dispatch keeps method name branching centralized in registry metadata.
    if family == "EG_POP":
        return compute_eg_pop(
            population=population,
            year=year,
            region_label=region_label,
        )

    if family == "PR_GDPCAP":
        if pr_pop is None or pr_gdp is None or pr_to_mrio is None:
            raise ValueError(
                "PR(GDPcap) requires PR population, PR GDP, and ISO to MRIO region mapping inputs."
            )
        return compute_pr_gdpcap(
            pop_iso=pr_pop,
            gdp_iso=pr_gdp,
            iso_to_mrio=pr_to_mrio,
            year=year,
            source_key=source_key,
            group_version=group_version_reg,
            aggregation_mode=l1_reg_aggreg,
            region_label=region_label,
        )

    if family == "PR_HR":
        if lcia_reg_by_year is None or rps_df is None or impact_parent_map is None:
            raise ValueError(
                "PR-HR requires LCIA time series, responsibility period settings, "
                "and impact parent mapping inputs."
            )
        if population_by_year is None:
            raise ValueError("PR-HR requires population history by year.")
        return compute_pr_hr(
            year=year,
            impact_year=impact_year,
            population=population,
            population_by_year=population_by_year,
            lcia_reg_by_year=lcia_reg_by_year,
            rps_df=rps_df,
            impact_parent_map=impact_parent_map,
            available_years=available_years,
            source_key=source_key,
            group_version=group_version_reg,
            aggregation_mode=l1_reg_aggreg,
            region_label=region_label,
            parent_cum_cache=pr_hr_parent_cum_cache,
            fallback_callback=pr_hr_fallback_callback,
        )

    if family == "AR_ECAP":
        result = compute_ar_ecap_l1(
            year=year,
            population=population,
            population_ref=population_ref,
            lcia_reg=lcia_reg,
            lcia_reg_by_year=lcia_reg_by_year,
            reference_year=reference_year,
            region_label=region_label,
            index_cache=index_cache,
        )
        mode = normalize_l1_reg_mode_required(l1_reg_aggreg)
        if mode == "post" and group_version_reg:
            # For post mode, compute on original labels then aggregate to grouped labels.
            result = _aggregate_l1_regions_post(
                frame=result,
                source_key=source_key,
                group_version_reg=group_version_reg,
                region_label=region_label,
            )
        return result
    return compute_ar_e_l1(
        year=year,
        lcia_reg=lcia_reg,
        lcia_reg_by_year=lcia_reg_by_year,
        reference_year=reference_year,
        region_label=region_label,
    )
