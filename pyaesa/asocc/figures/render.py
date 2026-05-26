"""Deterministic aSoCC figure orchestration."""

from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.reuse.derived_state import request_state_matches, set_request_state
from pyaesa.shared.runtime.reuse.contracts import asocc_signature_payload_matches_request
from pyaesa.shared.runtime.metadata.json import read_optional_json_dict, write_json_dict
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.figures.request_validation import (
    validate_consecutive_multi_year_figure_request,
)

from pyaesa.asocc.runtime.paths.deterministic import _get_asocc_figure_metadata_path
from pyaesa.asocc.runtime.paths.published import _get_asocc_figures_root
from .metadata import delete_persisted_figure_state_paths, write_run_figure_paths
from .product_renderers import render_products
from .row_reader import load_figure_rows
from .scope_planner import (
    RunScope,
    figure_signature,
    figure_state_key,
    requested_ssp_scenarios,
)


def render_asocc_figures(
    *,
    proj_base: Path,
    source: str,
    fu_code: str,
    requested_years: list[int],
    lcia_methods: list[str] | None,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
    compute_signature: dict[str, Any],
    output_paths: list[str],
    figure_external_method: dict[str, Any] | None,
    figure_options: dict[str, bool],
    dpi: int,
    output_format: str,
    refresh: bool,
    skip_if_exact: bool,
    status_source: str = "deterministic_asocc",
    status: StatusSink | None = None,
) -> list[Path] | None:
    """Render requested deterministic aSoCC figures from persisted public outputs."""
    validate_consecutive_multi_year_figure_request(
        requested_years=requested_years,
        family_label="deterministic aSoCC",
    )
    scope = RunScope.from_signature(
        proj_base=proj_base,
        source=source,
        signature=compute_signature,
    )
    figures_root = _get_asocc_figures_root(
        proj_base=proj_base,
        level="level_1" if str(fu_code).startswith("L1.") else "level_2",
        source=source,
        agg_version=scope.agg_version,
    )
    metadata_path = _get_asocc_figure_metadata_path(
        proj_base=proj_base,
        source=source,
        agg_version=scope.agg_version,
    )
    signature = figure_signature(
        requested_years=requested_years,
        lcia_methods=lcia_methods,
        dpi=dpi,
        output_format=output_format,
        figure_external_method=figure_external_method,
        figure_options=figure_options,
    )
    payload = read_optional_json_dict(metadata_path)
    state_key = figure_state_key(fu_code=fu_code)
    if (
        skip_if_exact
        and not refresh
        and request_state_matches(
            payload=payload,
            state_key=state_key,
            request_signature=signature,
            compute_signature=compute_signature,
            request_compatible=_asocc_figure_request_covers,
            compute_compatible=_asocc_figure_compute_covers,
        )
    ):
        return None
    delete_persisted_figure_state_paths(
        payload=payload,
        state_key=state_key,
    )
    if not (bool(figure_options["per_method"]) or bool(figure_options["multi_method"])):
        payload.pop(state_key, None)
        write_json_dict(metadata_path, payload)
        write_run_figure_paths(scope=scope, figure_paths=[])
        return []
    rows = load_figure_rows(
        scope=scope,
        fu_code=fu_code,
        requested_years=requested_years,
        lcia_methods=lcia_methods,
        ssp_scenarios=requested_ssp_scenarios(
            options_by_year=ssp_scenario_options_by_year,
            compute_signature=compute_signature,
        ),
        compute_signature=compute_signature,
        output_paths=output_paths,
        figure_external_method=figure_external_method,
    )
    figure_paths = render_products(
        rows=rows,
        figures_root=figures_root,
        requested_years=requested_years,
        dpi=dpi,
        output_format=output_format,
        status_source=status_source,
        per_method=bool(figure_options["per_method"]),
        multi_method=bool(figure_options["multi_method"]),
        status=status,
    )
    unique_paths = sorted({Path(path) for path in figure_paths})
    set_request_state(
        payload=payload,
        state_key=state_key,
        request_signature=signature,
        compute_signature=compute_signature,
        paths=unique_paths,
    )
    write_json_dict(metadata_path, payload)
    write_run_figure_paths(
        scope=scope,
        figure_paths=unique_paths,
    )
    return unique_paths


def _asocc_figure_request_covers(stored: dict[str, Any], requested: dict[str, Any]) -> bool:
    exact_keys = (
        "function",
        "contract",
        "figure_output_format",
        "figure_dpi",
        "figure_external_method",
        "figure_options",
    )
    return (
        all(stored.get(key) == requested.get(key) for key in exact_keys)
        and set(map(str, requested.get("years") or ())).issubset(
            set(map(str, stored.get("years") or ()))
        )
        and set(map(str, requested.get("lcia_methods") or ())).issubset(
            set(map(str, stored.get("lcia_methods") or ()))
        )
    )


def _asocc_figure_compute_covers(stored: dict[str, Any], requested: dict[str, Any]) -> bool:
    return asocc_signature_payload_matches_request(
        requested_signature=dict(requested),
        persisted_signature=dict(stored),
        ssp_scenarios=None,
        run_ssp_scenarios=None,
    )
