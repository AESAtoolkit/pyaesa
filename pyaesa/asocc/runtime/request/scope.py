"""Neutral aSoCC scope ownership shared across downstream consumers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from pyaesa.asocc.data.source_schema import (
    default_historical_cutoff_for_source,
    is_iso3_source,
    max_modeled_year_for_source,
    min_modeled_year_for_source,
)
from pyaesa.asocc.methods.lcia_inputs import normalize_lcia_methods
from pyaesa.asocc.orchestration.projection.config.config import resolve_projection_context
from pyaesa.asocc.orchestration.setup.request.selection import (
    _prune_lcia_methods_without_lcia_input,
    _resolve_filters,
    _resolve_selection_bundle,
    _restrict_selection_for_iso3_mode,
)
from pyaesa.asocc.runtime.methods.labels import l1_l2_method_label
from pyaesa.asocc.runtime.paths.family_roots import is_native_asocc_source
from pyaesa.asocc.runtime.request.normalization import (
    _normalize_optional_string_list,
)
from pyaesa.asocc.runtime.scope.branch_resolution import (
    AsoccDeterministicPathScope,
    resolve_allocate_path_scope,
    resolve_disaggregation_path_scope,
)
from pyaesa.asocc.runtime.selection.resolve import resolve_method_selection
from pyaesa.shared.selectors.time_selectors import (
    normalize_optional_year_selector,
    normalize_reg_window_for_storage,
)


def _selector_method_label(raw_label: str) -> str:
    """Return the canonical target selector label for one selected method."""
    if "::" not in raw_label:
        return str(raw_label).strip()
    l1_method, l2_method = [piece.strip() for piece in str(raw_label).split("::", 1)]
    return l1_l2_method_label(l1_method=l1_method, l2_method=l2_method)


def _selector_methods_for_scope(
    *,
    fu_code: str,
    selected_methods: dict[str, list[str]],
    combined: list[tuple[str, str]],
) -> set[str]:
    """Return the final public method labels reachable in one aSoCC scope."""
    if str(fu_code).startswith("L1."):
        return {
            _selector_method_label(method_label)
            for method_label in selected_methods.get("l1", [])
            if str(method_label).strip()
        }
    final_methods = {
        str(method_label).strip()
        for method_label in selected_methods.get("l2_vs_global", [])
        if str(method_label).strip()
    }
    for l2_method, l1_method in combined:
        final_methods.add(l1_l2_method_label(l1_method=l1_method, l2_method=l2_method))
    return final_methods


@dataclass(frozen=True)
class AsoccScope:
    """Owned family-local identity for one normalized deterministic aSoCC scope."""

    base_allocate_args: dict[str, Any]
    selected_l1: list[str]
    combined: list[tuple[str, str]]
    selected_l2_one_step: list[str]
    selected_methods: dict[str, list[str]]
    filters: dict[str, list[str] | None]
    studied_indices_tag: str

    def requested_signature(
        self,
        *,
        years_hint: int | list[int] | range | None = None,
    ) -> dict[str, Any]:
        """Return the effective deterministic aSoCC signature used for reuse matching.

        The uncertainty prerequisite must match the same effective projection
        contract that ``deterministic_asocc(...)`` would resolve for the target
        years. Omitted projection arguments therefore resolve here to the same
        canonical defaults used by deterministic setup before later reuse
        matching compares them against persisted deterministic signatures.
        """
        projection_mode, reg_window, l2_reuse_years = effective_projection_signature_for_source(
            base_allocate_args=self.base_allocate_args,
            selected_l2_one_step=self.selected_l2_one_step,
            combined=self.combined,
            years_hint=years_hint,
            projection_rule_source=None,
        )
        return {
            "source": self.base_allocate_args["source"],
            "agg_version": self.base_allocate_args["agg_version"],
            "agg_reg": bool(self.base_allocate_args["agg_reg"]),
            "agg_sec": bool(self.base_allocate_args["agg_sec"]),
            "fu_code": self.base_allocate_args["fu_code"],
            "studied_indices_tag": self.studied_indices_tag,
            "lcia_methods": list(self.base_allocate_args["lcia_method"] or []),
            "ssp_scenario_input": self.base_allocate_args["ssp_scenario"],
            "reference_years_input": self.base_allocate_args["reference_years"],
            "selected_methods": self.selected_methods,
            "l1_reg_aggreg": self.base_allocate_args["l1_reg_aggreg"],
            "variant_tag": None,
            "group_indices": self.base_allocate_args["group_indices"],
            "projection_mode": projection_mode,
            "reg_window": reg_window,
            "l2_reuse_years": l2_reuse_years,
        }

    @property
    def target_selector_payload(self) -> dict[str, Any]:
        """Return the internal target selector payload for one aSoCC scope."""
        selector: dict[str, Any] = {}
        years = normalize_optional_year_selector(self.base_allocate_args.get("years"), name="years")
        reference_years = normalize_optional_year_selector(
            self.base_allocate_args.get("reference_years"),
            name="reference_years",
        )
        l2_reuse_years = normalize_optional_year_selector(
            self.base_allocate_args.get("l2_reuse_years"),
            name="l2_reuse_years",
        )
        ssp_values = _normalize_optional_string_list(self.base_allocate_args.get("ssp_scenario"))
        lcia_methods = normalize_lcia_methods(self.base_allocate_args.get("lcia_method"))
        methods = _selector_methods_for_scope(
            fu_code=str(self.base_allocate_args["fu_code"]),
            selected_methods=self.selected_methods,
            combined=self.combined,
        )
        if years is not None:
            selector["years"] = years
        if reference_years is not None:
            selector["reference_year"] = reference_years
        if l2_reuse_years is not None:
            selector["l2_reuse_year"] = l2_reuse_years
        if ssp_values is not None:
            selector["ssp_values"] = ssp_values
        if lcia_methods:
            selector["lcia_method"] = list(lcia_methods)
        if methods:
            selector["methods"] = sorted(methods)
        return selector

    def resolve_path_scope(self) -> AsoccDeterministicPathScope:
        """Resolve the canonical deterministic aSoCC path scope for this request."""
        return resolve_allocate_path_scope(base_allocate_args=self.base_allocate_args)

    def resolve_disaggregation_scope(
        self,
        *,
        source_label: str,
    ) -> tuple[AsoccDeterministicPathScope, Path]:
        """Resolve one disaggregated deterministic path scope and scope manifest."""
        return resolve_disaggregation_path_scope(
            base_allocate_args=self.base_allocate_args,
            source_label=source_label,
        )

    def compute_signature(
        self,
        *,
        years: list[int],
        output_format: str,
        intermediate_outputs: bool,
        historical_year_cap: int | None,
        variant_tag: str | None = None,
    ) -> dict[str, Any]:
        """Return the canonical persisted deterministic aSoCC compute signature."""
        projection_mode, reg_window, l2_reuse_years = effective_projection_signature_for_source(
            base_allocate_args=self.base_allocate_args,
            selected_l2_one_step=self.selected_l2_one_step,
            combined=self.combined,
            years_hint=years,
            projection_rule_source=None,
        )
        return {
            "source": self.base_allocate_args["source"],
            "agg_version": self.base_allocate_args["agg_version"],
            "agg_reg": self.base_allocate_args["agg_reg"],
            "agg_sec": self.base_allocate_args["agg_sec"],
            "fu_code": self.base_allocate_args["fu_code"],
            "studied_indices_tag": self.studied_indices_tag,
            "years": list(years),
            "lcia_methods": list(self.base_allocate_args["lcia_method"] or []),
            "ssp_scenario_input": self.base_allocate_args["ssp_scenario"],
            "reference_years_input": _normalize_year_selector_for_signature(
                self.base_allocate_args["reference_years"]
            ),
            "selected_methods": dict(self.selected_methods),
            "l1_reg_aggreg": self.base_allocate_args["l1_reg_aggreg"],
            "variant_tag": variant_tag,
            "group_indices": self.base_allocate_args["group_indices"],
            "output_format": output_format,
            "intermediate_outputs": bool(intermediate_outputs),
            "projection_mode": projection_mode,
            "reg_window": reg_window,
            "l2_reuse_years": l2_reuse_years,
            "historical_year_cap": historical_year_cap,
        }


def _normalize_year_selector_for_signature(
    value: int | list[int] | range | None,
) -> list[int] | None:
    """Normalize one optional year selector for persisted deterministic signatures."""
    return normalize_optional_year_selector(value, name="year")


def effective_projection_signature_for_source(
    *,
    base_allocate_args: dict[str, Any],
    selected_l2_one_step: list[str],
    combined: list[tuple[str, str]],
    years_hint: int | list[int] | range | None,
    projection_rule_source: str | None,
) -> tuple[str | None, list[int] | None, list[int] | None]:
    """Return effective projection signature fields for prerequisite matching."""
    source = str(projection_rule_source or base_allocate_args["source"])
    if not is_native_asocc_source(source=source):
        return (
            base_allocate_args["projection_mode"],
            normalize_reg_window_for_storage(base_allocate_args["reg_window"]),
            _normalize_year_selector_for_signature(base_allocate_args["l2_reuse_years"]),
        )
    if is_iso3_source(source):
        return None, None, []
    requested_years = normalize_optional_year_selector(
        base_allocate_args["years"] if years_hint is None else years_hint,
        name="years_hint",
    )
    modeled_min = cast(int, min_modeled_year_for_source(source))
    modeled_max = cast(int, max_modeled_year_for_source(source))
    resolved_years = (
        list(range(int(modeled_min), int(modeled_max) + 1))
        if requested_years is None
        else list(requested_years)
    )
    historical_cutoff = default_historical_cutoff_for_source(source)
    # Explicit selectors override the default max regression/reuse window
    # set before nowcasting starts while remaining capped by MRIO availability.
    historical_max = (
        int(modeled_max)
        if historical_cutoff is None
        or base_allocate_args["reg_window"] is not None
        or base_allocate_args["l2_reuse_years"] is not None
        else int(historical_cutoff)
    )
    requested_max = max(int(year) for year in resolved_years)
    historical_years = list(range(int(modeled_min), min(requested_max, historical_max) + 1))
    projection_context = resolve_projection_context(
        source=source,
        fu_code=str(base_allocate_args["fu_code"]),
        resolved_years=resolved_years,
        historical_years=historical_years,
        selected_l2_one_step=selected_l2_one_step,
        combined=combined,
        projection_mode=base_allocate_args["projection_mode"],
        reg_window=normalize_reg_window_for_storage(base_allocate_args["reg_window"]),
        l2_reuse_years=base_allocate_args["l2_reuse_years"],
    )
    return (
        projection_context.mode,
        normalize_reg_window_for_storage(projection_context.reg_window),
        list(projection_context.l2_reuse_years),
    )


def build_asocc_scope(*, base_allocate_args: Mapping[str, Any]) -> AsoccScope:
    """Build one explicit family-local aSoCC scope object from normalized args."""
    l1_full, combined_full, one_step_full = resolve_method_selection(
        fu_code=base_allocate_args["fu_code"],
        method_plan=base_allocate_args["method_plan"],
        l1_methods=base_allocate_args["l1_methods"],
        one_step_methods=base_allocate_args["one_step_methods"],
        two_step_methods=base_allocate_args["two_step_methods"],
        l1_l2_pairs=base_allocate_args["l1_l2_pairs"],
    )
    l1_lcia_kind = "PBA" if base_allocate_args["fu_code"] == "L1.b" else "CBA_FD"
    selection = _resolve_selection_bundle(
        fu_code=base_allocate_args["fu_code"],
        l_1=l1_full,
        l_2_combined_with_l_1=combined_full,
        l_2_one_step=one_step_full,
        l1_lcia_kind=l1_lcia_kind,
    )
    if (
        base_allocate_args.get("source")
        and is_native_asocc_source(source=str(base_allocate_args["source"]))
        and is_iso3_source(str(base_allocate_args["source"]))
    ):
        selection = _restrict_selection_for_iso3_mode(
            fu_code=base_allocate_args["fu_code"],
            selection=selection,
        )
    selection, _ = _prune_lcia_methods_without_lcia_input(
        fu_code=base_allocate_args["fu_code"],
        lcia_methods=base_allocate_args["lcia_method"],
        selection=selection,
    )
    filters, studied_indices_tag = _resolve_filters(
        required_indices=selection.required_indices,
        r_p=base_allocate_args["r_p"],
        s_p=base_allocate_args["s_p"],
        r_c=base_allocate_args["r_c"],
        r_f=base_allocate_args["r_f"],
    )
    return AsoccScope(
        base_allocate_args=dict(base_allocate_args),
        selected_l1=list(selection.selected_l1),
        combined=list(selection.combined),
        selected_l2_one_step=list(selection.selected_l2_one_step),
        selected_methods=selection.selected_methods,
        filters=filters,
        studied_indices_tag=studied_indices_tag,
    )
