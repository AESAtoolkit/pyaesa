"""Scope planning helpers for aCC uncertainty figures."""

from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from pyaesa.acc.uncertainty.io.paths import (
    acc_uncertainty_figures_root,
)
from pyaesa.acc.uncertainty.runtime.models import ACCUncertaintyRunPaths
from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
    normalize_figure_options,
)
from pyaesa.shared.selectors.scenarios import DEFAULT_SSP_SCENARIOS
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


@dataclass(frozen=True)
class FigureContext:
    """Normalized context for one completed aCC uncertainty run."""

    manifest: UncertaintyManifest
    paths: ACCUncertaintyRunPaths
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
    dynamic_cc_sampling_method: str | None = None


def build_figure_context(
    *,
    manifest: UncertaintyManifest,
    paths: ACCUncertaintyRunPaths,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> FigureContext:
    """Return normalized figure context for a completed aCC uncertainty run."""
    options = normalize_figure_options(
        figure_options,
        allow_single_year_style=False,
        allow_polar_years=False,
        allow_per_method=True,
        allow_multi_method=True,
        allow_inter_method=True,
    )
    figure = normalize_figure_format(figure_format)
    args = dict(manifest.arguments or {})
    public_output = dict(manifest.artifacts["public_output"] or {})
    runs = dict(public_output.get("acc_runs") or {})
    source_parameters = dict(manifest.source_parameters or {})
    return FigureContext(
        manifest=manifest,
        paths=paths,
        figures_root=acc_uncertainty_figures_root(paths=paths),
        requested_years=tuple(_years_from_args(args)),
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
