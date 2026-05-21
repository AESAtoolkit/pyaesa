"""Scope planning helpers for ASR uncertainty figures."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyRunPaths
from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
    normalize_figure_options,
    resolve_nested_polar_years,
)
from pyaesa.shared.selectors.scenarios import DEFAULT_SSP_SCENARIOS
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


@dataclass(frozen=True)
class FigureContext:
    """Normalized context for one completed ASR uncertainty run."""

    manifest: UncertaintyManifest
    paths: ASRUncertaintyRunPaths
    figures_root: Path
    requested_years: tuple[int, ...]
    requested_asocc_ssps: tuple[str, ...]
    fu_code: str
    output_format: str
    figure_output_format: str
    figure_dpi: int
    per_method: bool
    multi_method: bool
    inter_method: bool
    active_sources: tuple[str, ...]
    run_layout: str
    dynamic_category_uncertainty_active: bool
    polar_years: tuple[int, ...]
    polar_style: str
    dynamic_cc_sampling_method: str | None = None


def build_figure_context(
    *,
    manifest: UncertaintyManifest,
    paths: ASRUncertaintyRunPaths,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> FigureContext:
    """Return normalized figure context for a completed ASR uncertainty run."""
    options = normalize_figure_options(
        figure_options,
        allow_single_year_style=False,
        allow_polar_years=False,
        allow_per_method=True,
        allow_multi_method=True,
        allow_inter_method=True,
        allow_polar=True,
    )
    polar = dict(options["polar"])
    figure = normalize_figure_format(figure_format)
    args = dict(manifest.arguments or {})
    public_output = dict(manifest.artifacts["public_output"] or {})
    runs = dict(public_output.get("asr_runs") or {})
    source_parameters = dict(manifest.source_parameters or {})
    years = _years_from_args(args)
    return FigureContext(
        manifest=manifest,
        paths=paths,
        figures_root=paths.run_root / "figures",
        requested_years=tuple(years),
        requested_asocc_ssps=tuple(_asocc_ssps_from_args(args)),
        fu_code=str(args["fu_code"]),
        output_format=str(manifest.output_format),
        figure_output_format=str(figure["format"]),
        figure_dpi=int(figure["dpi"]),
        per_method=bool(options["per_method"]),
        multi_method=bool(options["multi_method"]),
        inter_method=bool(options["inter_method"]),
        active_sources=tuple(str(source) for source in manifest.active_sources),
        run_layout=str(runs.get("layout", "compact_run_matrix")),
        dynamic_category_uncertainty_active=bool(
            source_parameters.get("dynamic_cc_category_uncertainty", False)
        ),
        dynamic_cc_sampling_method=(
            None
            if source_parameters.get("dynamic_cc_sampling_method") is None
            else str(source_parameters["dynamic_cc_sampling_method"]).strip().lower()
        ),
        polar_years=tuple(
            resolve_nested_polar_years(
                studied_years=years,
                polar=polar,
                argument_name="figure_options.polar",
            )
        ),
        polar_style=str(polar["polar_style"]),
    )


def single_requested_year(context: FigureContext) -> int | None:
    """Return the single requested year if this is a one year product."""
    years = sorted({int(year) for year in context.requested_years})
    return years[0] if len(years) == 1 else None


def _years_from_args(args: dict[str, Any]) -> list[int]:
    raw = args.get("years")
    if isinstance(raw, int):
        return [int(raw)]
    return sorted({int(value) for value in cast(Iterable[Any], raw)})


def _asocc_ssps_from_args(args: dict[str, Any]) -> list[str]:
    base_asocc_args = args.get("base_asocc_args") or {}
    raw = base_asocc_args.get("ssp_scenario") or DEFAULT_SSP_SCENARIOS
    values = [raw] if isinstance(raw, str) else list(cast(Iterable[Any], raw))
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})
