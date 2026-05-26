"""Selector request and region-compatibility helpers for disaggregation."""

from typing import Any

from pyaesa.process.mrios.utils.io.metadata import read_processed_mrio_regions

from ..entrypoints.argument_contracts import ensure_list_str
from pyaesa.asocc.orchestration.setup.request.types import PrepareContextRequest


def _build_selector_request(
    *,
    selector,
    base_allocate_args: dict[str, Any],
    l1_methods: list[str] | None,
    combined_methods: list[tuple[str, str]] | None,
    one_step_methods: list[str] | None,
    l1_reg_aggreg: str,
    group_indices: bool,
    variant_tag: str | None,
    output_format: str = "csv",
    output_source_label: str | None = None,
) -> PrepareContextRequest:
    """Build one deterministic setup request for a selected prerequisite scope."""
    return PrepareContextRequest(
        project_name=str(base_allocate_args["project_name"]),
        source=str(selector.source),
        agg_version=selector.agg_version,
        agg_reg=bool(selector.agg_reg),
        agg_sec=bool(selector.agg_sec),
        years=base_allocate_args["years"],
        historical_year_cap=None,
        refresh=False,
        lcia_method=None,
        fu_code=str(base_allocate_args["fu_code"]),
        r_p=ensure_list_str(base_allocate_args["r_p"]),
        s_p=list(selector.s_p),
        r_c=ensure_list_str(base_allocate_args["r_c"]),
        r_f=ensure_list_str(base_allocate_args["r_f"]),
        l_1=l1_methods,
        l_2_combined_with_l_1=combined_methods,
        l_2_one_step=one_step_methods,
        reference_years=base_allocate_args["reference_years"],
        ssp_scenario=base_allocate_args["ssp_scenario"],
        projection_mode=base_allocate_args["projection_mode"],
        reg_window=base_allocate_args["reg_window"],
        l2_reuse_years=base_allocate_args["l2_reuse_years"],
        l1_reg_aggreg=str(l1_reg_aggreg),
        variant_tag=variant_tag,
        group_indices=bool(group_indices),
        output_format=str(output_format),
        intermediate_outputs=False,
        output_source_label=output_source_label,
    )


def _region_filters(value: str | list[str] | None) -> list[str]:
    """Return normalized region labels from the public disaggregation run plan."""
    values = ensure_list_str(value)
    return [] if values is None else [item.strip() for item in values if item.strip()]


def _load_selector_regions(selector) -> list[str]:
    """Return the processed region labels declared for one selector."""
    matrix_version = selector.agg_version if (selector.agg_reg or selector.agg_sec) else None
    return read_processed_mrio_regions(str(selector.source), matrix_version=matrix_version)


def validate_region_compatibility(
    *,
    target_selector,
    ref_aggregated_selector,
    ref_disaggregate_selector,
    base_allocate_args: dict[str, Any],
    combined_methods: list[tuple[str, str]],
) -> None:
    """Validate same-label studied-region compatibility for disaggregation."""
    target_regions = _load_selector_regions(target_selector)
    ref_aggregated_regions = _load_selector_regions(ref_aggregated_selector)
    ref_disaggregate_regions = _load_selector_regions(ref_disaggregate_selector)
    del combined_methods
    studied_regions = sorted(
        set(
            _region_filters(base_allocate_args["r_p"])
            + _region_filters(base_allocate_args["r_c"])
            + _region_filters(base_allocate_args["r_f"])
        )
    )
    if not studied_regions:
        return
    missing_target = [item for item in studied_regions if item not in target_regions]
    missing_ref_aggregated = [
        item for item in studied_regions if item not in ref_aggregated_regions
    ]
    missing_ref_disaggregate = [
        item for item in studied_regions if item not in ref_disaggregate_regions
    ]
    if missing_target or missing_ref_aggregated or missing_ref_disaggregate:
        raise ValueError(
            "Studied regions must be present by the same label in all selected sources. "
            f"missing_target={missing_target}, "
            f"missing_ref_aggregated={missing_ref_aggregated}, "
            f"missing_ref_disaggregate={missing_ref_disaggregate}."
        )
