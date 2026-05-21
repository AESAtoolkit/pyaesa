"""Metadata ownership for allocation runs."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from pyaesa.asocc.runtime.output.contracts import OutputSpec
from pyaesa.shared.selectors.time_selectors import (
    normalize_optional_year_selector,
    normalize_reg_window_for_storage,
)
from pyaesa.shared.runtime.manifest_contract import manifest_digest
from pyaesa.shared.runtime.metadata.json import (
    read_optional_json_dict,
    write_json_dict,
)


def _run_scope_key(*, signature: dict[str, Any]) -> str:
    """Return deterministic key for one run signature scope.

    The key is a stable SHA-256 digest of the canonical JSON representation
    of ``signature``. It is used to index exact completed run metadata.
    """
    return manifest_digest(signature)


def _now_iso() -> str:
    """Return a local timestamp string for persisted deterministic metadata."""
    return datetime.now().isoformat()


def _build_run_metadata(
    *,
    requested_years: list[int],
    resolved_years: list[int],
    selected_methods: dict[str, list[str]],
    fu_code: str,
    studied_indices_tag: str,
    skipped_years: dict[int, str | dict[str, str]],
    outputs: list[str],
    signature: dict[str, Any],
) -> dict[str, Any]:
    """Build metadata payload for an allocation run.

    Args:
        resolved_years: Years processed.
        selected_methods: Selected methods by level.
        fu_code: Functional unit code.
        studied_indices_tag: Indices tag.
        skipped_years: Skipped years and reasons.
        outputs: Written outputs.

    Returns:
        Metadata dictionary.
    """
    timestamp = _now_iso()
    return {
        "function": "deterministic_asocc",
        "arguments": signature,
        "execution": {
            "status": "complete",
            "timestamp": timestamp,
            "requested_years": requested_years,
            "resolved_years": resolved_years,
            "skipped_years": skipped_years,
        },
        "reuse": {"identity_key": _run_scope_key(signature=signature)},
        "artifacts": {"outputs": outputs, "figure_paths": []},
        "provenance": {
            "fu_code": fu_code,
            "studied_indices_tag": studied_indices_tag,
            "selected_methods": selected_methods,
        },
    }


@dataclass
class RunContext:
    """Immutable per branch run configuration.

    This object is built once during setup and passed through yearly compute
    and write stages. It intentionally contains only deterministic run
    configuration and read only loaded inputs.
    """

    project_name: str
    source: str
    fu_code: str
    group_version: str | None
    group_version_reg: str | None
    group_reg: bool | None
    group_sec: bool | None
    lcia_method: str | list[str] | None
    years_input: int | list[int] | range | None
    reference_years_input: int | list[int] | range | None
    ssp_scenario: str | list[str] | None
    is_exio: bool
    l1_lcia_kind: str
    lcia_methods: list[str] | None
    selected_l1: list[str]
    combined: list[tuple[str, str]]
    selected_l2_one_step: list[str]
    required_indices: set[str]
    filters: dict[str, list[str] | None]
    studied_indices_tag: str
    proj_base: Path
    logger: Any
    requested_years: list[int]
    resolved_years: list[int]
    persisted_years: list[int]
    historical_years: list[int]
    reference_years: list[int] | None
    ssp_scenario_options: list[str | None]
    run_signature: dict[str, Any]
    needs_lcia: bool
    repo_root: Path
    wb_df: pd.DataFrame
    ssp_df: pd.DataFrame
    wb_df_raw: pd.DataFrame
    ssp_df_raw: pd.DataFrame
    selected_methods: dict[str, list[str]]
    l1_kinds_needed: set[str]
    l1_only_no_mrio: bool
    l1_reg_aggreg: str
    use_original_l1_post_domain: bool
    variant_tag: str | None
    aggreg_indices: bool
    output_format: str
    intermediate_outputs: bool
    output_source_label: str | None = None
    projection_context: Any | None = None
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None = None
    metadata_completed_years: list[int] | None = None
    metadata_prior_outputs: list[str] | None = None
    compute_years: list[int] = field(default_factory=list)

    @property
    def output_source(self) -> str:
        """Return the published source label for this run scope."""
        return self.output_source_label or self.source


@dataclass
class RunState:
    """Mutable in memory state for one run branch.

    The state is progressively updated while processing years and then used
    by the write stage. It tracks outputs, caches, warnings, and enacting metric
    metric buffers.
    """

    # State is branch local: each (l1_reg_aggreg x aggreg_indices) branch gets
    # its own buffers/caches to keep writes deterministic.

    processed_years: list[int] = field(default_factory=list)
    skipped_years: dict[int, str | dict[str, str]] = field(default_factory=dict)
    empty_ref_years: dict[int, list[int]] = field(default_factory=dict)
    outputs_written: list[str] = field(default_factory=list)
    outputs_all: list[str] = field(default_factory=list)
    output_files_created: list[str] = field(default_factory=list)
    output_files_updated: list[str] = field(default_factory=list)
    notices_emitted: set[str] = field(default_factory=set)
    l1_results_by_ssp_scenario: dict[str | None, dict[OutputSpec, list[pd.DataFrame]]] = field(
        default_factory=dict
    )
    l2_results_by_ssp_scenario: dict[str | None, dict[OutputSpec, list[pd.DataFrame]]] = field(
        default_factory=dict
    )
    pre_weighting_written_by_ssp_scenario: dict[
        str | None,
        set[tuple[str, str | None, int | None, int]],
    ] = field(default_factory=dict)
    pop_series_by_ssp_scenario: dict[str | None, dict[int, pd.Series]] = field(default_factory=dict)
    pr_post_pop_series_by_ssp_scenario: dict[str | None, dict[int, pd.Series]] = field(
        default_factory=dict
    )
    gdp_series_by_ssp_scenario: dict[str | None, dict[int, pd.Series]] = field(default_factory=dict)
    ar_l1_cache_by_ssp_scenario: dict[str | None, dict[tuple, pd.DataFrame]] = field(
        default_factory=dict
    )
    ar_l2_cache_by_ssp_scenario: dict[str | None, dict[tuple, pd.DataFrame]] = field(
        default_factory=dict
    )
    preweight_cache_by_ssp_scenario: dict[str | None, dict[tuple, pd.DataFrame]] = field(
        default_factory=dict
    )
    enacting_metric_inputs: dict["EnactingMetricKey", dict[int, pd.Series]] = field(
        default_factory=dict
    )
    enacting_metric_levels: dict["EnactingMetricKey", str] = field(default_factory=dict)
    mrio_default_monetary_unit: str | None = None
    mrio_units: dict[str, str] = field(default_factory=dict)
    lcia_units: dict[str, pd.Series] = field(default_factory=dict)
    lcia_timeseries: dict[str, dict[str, dict[int, pd.DataFrame]]] = field(default_factory=dict)
    lcia_timeseries_original: dict[str, dict[str, dict[int, pd.DataFrame]]] = field(
        default_factory=dict
    )
    rps_by_method: dict[str, pd.DataFrame] = field(default_factory=dict)
    cf_by_method: dict[str, pd.Series] = field(default_factory=dict)
    ar_valid_refs_cache: dict[tuple, tuple[list[int], list[tuple[int, str]]]] = field(
        default_factory=dict
    )
    lcia_metadata_cache: dict[tuple[str, str | None], tuple[dict[str, Any], str]] = field(
        default_factory=dict
    )
    lcia_available_years_cache: dict[tuple[str, str | None, str], list[int]] = field(
        default_factory=dict
    )
    lcia_method_payload_cache: dict[tuple[str | None, str, str], dict[str, pd.DataFrame]] = field(
        default_factory=dict
    )
    pr_hr_parent_cum_cache: dict[tuple, dict[int, dict[str, pd.Series]]] = field(
        default_factory=dict
    )
    pr_hr_rp1_zero_fallback_pending: dict[
        tuple[str, tuple[str, ...], int, int, bool, str | None],
        set[str],
    ] = field(default_factory=dict)
    l1_invariant_cache: dict[tuple, tuple[pd.DataFrame, pd.DataFrame | None]] = field(
        default_factory=dict
    )
    lcia_sliced_payload_cache: dict[tuple, dict] = field(default_factory=dict)
    l1_year_invariant_cache: dict[int, dict[str, pd.DataFrame]] = field(default_factory=dict)
    reg_group_map_cache: dict[tuple[str, str | None], dict[str, str]] = field(default_factory=dict)
    projection_payload_cache: dict[tuple[int, str | None], Any] = field(default_factory=dict)
    projection_history_cache: dict[tuple, Any] = field(default_factory=dict)
    ut_reuse_preweight_cache: dict[tuple, pd.DataFrame] = field(default_factory=dict)
    ut_reuse_one_step_cache: dict[tuple, pd.DataFrame] = field(default_factory=dict)
    # Projection fit cache stores heterogeneous model payloads:
    # OLS level fits and share logit fits have different value shapes.
    regression_fit_cache: dict[tuple, Any] = field(default_factory=dict)
    projection_regression_basis_cache: dict[tuple, Any] = field(default_factory=dict)
    regression_stats_rows: list[dict[str, Any]] = field(default_factory=list)
    regression_fit_inputs_rows: list[dict[str, Any]] = field(default_factory=list)
    regression_uncertainty_rows: list[dict[str, Any]] = field(default_factory=list)
    ut_gvaa_identity_closure_rows: list[dict[str, Any]] = field(default_factory=list)
    write_progress_total: int = 0
    write_progress_current: int = 0
    write_progress_last_width: int = 0
    write_progress_label: str | None = None
    write_progress_prefix: str | None = None
    output_spec_cache: dict[tuple, OutputSpec] = field(default_factory=dict)
    l2_batch_weighting_plan_cache: dict[tuple, Any] = field(default_factory=dict)
    output_index_level_cache: dict[tuple[object, ...], Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnactingMetricKey:
    """Structured key for one enacting metric series family."""

    metric: str
    lcia_method: str | None = None
    ssp_scenario: str | None = None


def _load_run_metadata(path: Path) -> dict[str, Any]:
    """Load deterministic aSoCC scope manifest metadata if present.

    Args:
        path: ``scope_manifest.json`` path.

    Returns:
        Parsed metadata payload, or an empty dictionary when missing.
    """
    payload = read_optional_json_dict(path)
    if not payload:
        return {}
    return dict(payload)


def _save_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Persist one deterministic aSoCC scope manifest."""
    write_json_dict(path, dict(payload))


def _normalize_year_selector_for_storage(
    value: int | list[int] | range | None,
) -> list[int] | None:
    """Normalize year selector values for deterministic metadata storage."""
    return normalize_optional_year_selector(value, name="year")


def _normalize_reg_window_for_storage(
    value: list[int] | range | tuple[int, int] | None,
) -> list[int] | None:
    """Normalize reg_window to one persisted consecutive year list."""
    return normalize_reg_window_for_storage(value)
