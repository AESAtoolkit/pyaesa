"""Shared dataclasses for setup orchestration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _SelectionBundle:
    """Resolved method selections and derived selection metadata."""

    selected_l1: list[str]
    combined: list[tuple[str, str]]
    selected_l2_one_step: list[str]
    l1_kinds_needed: set[str]
    required_indices: set[str]
    needs_lcia_flag: bool
    l1_only_no_mrio: bool
    selected_methods: dict[str, list[str]]


@dataclass(frozen=True)
class _AggregationBundle:
    """Resolved aggregation controls."""

    apply_agg_reg: bool
    apply_agg_sec: bool
    agg_version_reg: str | None


@dataclass(frozen=True)
class _YearBundle:
    """Resolved studied and historical year sets."""

    resolved_years: list[int]
    historical_years: list[int]
    max_year: int
    out_of_range_years: list[int]


@dataclass(frozen=True)
class PrepareContextRequest:
    """Input payload for setup context preparation."""

    project_name: str
    source: str
    agg_version: str | None
    agg_reg: bool | None
    agg_sec: bool | None
    years: int | list[int] | range | None
    historical_year_cap: int | None
    refresh: bool
    lcia_method: str | list[str] | None
    fu_code: str
    r_p: list[str] | None
    s_p: list[str] | None
    r_c: list[str] | None
    r_f: list[str] | None
    l_1: list[str] | None
    l_2_combined_with_l_1: list[tuple[str, str]] | None
    l_2_one_step: list[str] | None
    reference_years: int | list[int] | range | None
    ssp_scenario: str | list[str] | None
    projection_mode: str | None
    reg_window: list[int] | range | None
    l2_reuse_years: int | list[int] | range | None
    l1_reg_aggreg: str
    variant_tag: str | None
    group_indices: bool
    output_format: str
    intermediate_outputs: bool
    output_source_label: str | None = None

    @property
    def output_source(self) -> str:
        """Return the single canonical published source label for this request."""
        return (
            str(self.output_source_label).strip()
            if self.output_source_label is not None and str(self.output_source_label).strip()
            else str(self.source).strip()
        )
