"""Scope planning helpers for AR6 CC uncertainty figures."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pyaesa.ar6_cc.uncertainty.io.paths import (
    ar6_cc_uncertainty_figures_root,
)
from pyaesa.ar6_cc.uncertainty.request.normalization import AR6_DYNAMIC_CC_SOURCE
from pyaesa.ar6_cc.uncertainty.runtime.models import AR6CCUncertaintyRunPaths
from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
    normalize_figure_options,
)
from pyaesa.shared.selectors.scenarios import ssp_partition_token
from pyaesa.shared.tabular.scalars import sanitize_token
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest

SUMMARY_STAT_COLUMNS = ("mean", "std", "min", "p5", "p25", "median", "p75", "p95", "max")
TRAJECTORY_BAND_COLUMNS = ("mean", "median", "p25", "p75", "p5", "p95")
IDENTITY_COLUMNS = (
    "public_row_id",
    "cc_category",
    "ssp_scenario",
    "cc_flow",
    "cc_variable",
    "impact_unit",
    "year",
)


@dataclass(frozen=True)
class FigureContext:
    """Normalized context for one completed AR6 CC uncertainty run."""

    manifest: UncertaintyManifest
    paths: AR6CCUncertaintyRunPaths
    figures_root: Path
    requested_years: tuple[int, ...]
    requested_ssps: tuple[str, ...]
    variable_name: str
    categories: tuple[str, ...]
    category_uncertainty: bool
    sampling_method: str
    has_post_study_period: bool
    output_format: str
    figure_output_format: str
    figure_dpi: int


def build_figure_context(
    *,
    manifest: UncertaintyManifest,
    paths: AR6CCUncertaintyRunPaths,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> FigureContext:
    """Return normalized AR6 CC uncertainty figure context."""
    normalize_figure_options(
        figure_options,
        allow_single_year_style=False,
        allow_polar_years=False,
    )
    figure = normalize_figure_format(figure_format)
    args = _base_ar6_cc_args(manifest=manifest)
    years = _years_from_args(args)
    prerequisite = _single_prerequisite(manifest=manifest)
    source_parameters = _source_parameters(manifest=manifest)
    artifacts = cast(dict[str, Any], manifest.artifacts)
    return FigureContext(
        manifest=manifest,
        paths=paths,
        figures_root=ar6_cc_uncertainty_figures_root(paths=paths),
        requested_years=tuple(years),
        requested_ssps=tuple(_ssp_from_args(args)),
        variable_name=str(prerequisite.get("variable", "AR6 CC")).strip() or "AR6 CC",
        categories=tuple(str(value) for value in prerequisite.get("categories", []) or []),
        category_uncertainty=bool(source_parameters.get("category_uncertainty", False)),
        sampling_method=str(source_parameters["sampling_method"]).strip().lower(),
        has_post_study_period="post_study_period_summary_stats_runs" in artifacts,
        output_format=str(manifest.output_format),
        figure_output_format=str(figure["format"]),
        figure_dpi=int(figure["dpi"]),
    )


def common_scope_stem(
    *,
    ssp_scenario: str,
) -> str:
    """Return a deterministic file stem for one common category ensemble figure."""
    return ssp_partition_token(ssp_scenario)


def category_scope_stem(
    *,
    ssp_scenario: str,
    category: str,
) -> str:
    """Return a deterministic file stem for one category band figure."""
    parts = [ssp_partition_token(ssp_scenario)]
    parts.append(f"cat_{sanitize_token(category)}")
    return "__".join(parts)


def _base_ar6_cc_args(*, manifest: UncertaintyManifest) -> dict[str, Any]:
    payload = cast(dict[str, Any], manifest.arguments or {})
    return dict(cast(dict[str, Any], payload.get("base_ar6_cc_args", {})))


def _single_prerequisite(*, manifest: UncertaintyManifest) -> dict[str, Any]:
    prerequisites = list(manifest.deterministic_prerequisites)
    return dict(cast(dict[str, Any], prerequisites[0]))


def _source_parameters(*, manifest: UncertaintyManifest) -> dict[str, Any]:
    source_parameters = cast(dict[str, Any], manifest.source_parameters or {})
    return dict(cast(dict[str, Any], source_parameters.get(AR6_DYNAMIC_CC_SOURCE, {})))


def _years_from_args(args: dict[str, Any]) -> list[int]:
    raw = args["years"]
    if isinstance(raw, int):
        return [int(raw)]
    return [int(value) for value in raw]


def _ssp_from_args(args: dict[str, Any]) -> list[str]:
    raw = args.get("ssp_scenario")
    if raw is None:
        return []
    values = [raw] if isinstance(raw, str) else list(raw)
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})
