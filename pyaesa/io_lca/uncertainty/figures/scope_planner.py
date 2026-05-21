"""Figure request planning for IO-LCA uncertainty outputs."""

from dataclasses import dataclass
from typing import Any

from pyaesa.io_lca.uncertainty.runtime.models import (
    IOLCAUncertaintyRequest,
    IOLCAUncertaintyRunPaths,
)
from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
    normalize_figure_options,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest

SUMMARY_STAT_COLUMNS = ("mean", "std", "min", "p5", "p25", "median", "p75", "p95", "max")
VALUE_ARRAY_COLUMN = "__run_values"


@dataclass(frozen=True)
class FigureContext:
    """Resolved IO-LCA uncertainty figure request."""

    manifest: UncertaintyManifest
    paths: IOLCAUncertaintyRunPaths
    request: IOLCAUncertaintyRequest
    output_format: str
    requested_years: tuple[int, ...]
    figure_output_format: str
    figure_dpi: int


def build_figure_context(
    *,
    manifest: UncertaintyManifest,
    paths: IOLCAUncertaintyRunPaths,
    request: IOLCAUncertaintyRequest,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> FigureContext:
    """Return normalized IO-LCA uncertainty figure context."""
    normalize_figure_options(
        figure_options,
        allow_single_year_style=False,
        allow_polar_years=False,
    )
    normalized_format = normalize_figure_format(figure_format)
    return FigureContext(
        manifest=manifest,
        paths=paths,
        request=request,
        output_format=manifest.output_format,
        requested_years=tuple(int(year) for year in request.years),
        figure_output_format=str(normalized_format["format"]),
        figure_dpi=int(normalized_format["dpi"]),
    )
