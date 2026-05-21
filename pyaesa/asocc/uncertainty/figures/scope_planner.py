"""Scope planning helpers for aSoCC uncertainty figures."""

from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.asocc.uncertainty.io.paths import (
    AsoccUncertaintyRunPaths,
    asocc_uncertainty_figures_root,
)
from pyaesa.asocc.figures.file_stems import asocc_scope_stem, visible_scope_values
from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
    normalize_figure_options,
)
from pyaesa.shared.figures.lcia_metadata import lcia_title_parts
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.scalars import is_display_missing
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest

METHOD_COLUMNS = ("l1_l2_method", "l1_method", "l2_method")
SELECTOR_COLUMNS = ("r_c", "r_p", "r_f", "s_p")
LCIA_COLUMNS = ("lcia_method", "impact")
SUMMARY_STAT_COLUMNS = ("mean", "std", "min", "p5", "p25", "median", "p75", "p95", "max")
VALUE_ARRAY_COLUMN = "__values"


@dataclass(frozen=True)
class FigureContext:
    """Normalized context for one completed aSoCC uncertainty run."""

    manifest: UncertaintyManifest
    paths: AsoccUncertaintyRunPaths
    figures_root: Path
    requested_years: tuple[int, ...]
    requested_ssps: tuple[str, ...]
    fu_code: str
    output_format: str
    figure_output_format: str
    figure_dpi: int
    per_method: bool
    multi_method: bool
    inter_method: bool
    active_sources: tuple[str, ...]


def build_figure_context(
    *,
    manifest: UncertaintyManifest,
    paths: AsoccUncertaintyRunPaths,
    figure_options: dict[str, Any] | None,
    figure_format: dict[str, Any] | None,
) -> FigureContext:
    """Return normalized figure context for a completed uncertainty run."""
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
    return FigureContext(
        manifest=manifest,
        paths=paths,
        figures_root=asocc_uncertainty_figures_root(paths=paths),
        requested_years=tuple(_years_from_args(args)),
        requested_ssps=tuple(_ssp_from_args(args)),
        fu_code=str(args.get("fu_code", "")).strip(),
        output_format=str(manifest.output_format),
        figure_output_format=str(figure["format"]),
        figure_dpi=int(figure["dpi"]),
        per_method=bool(options["per_method"]),
        multi_method=bool(options["multi_method"]),
        inter_method=bool(options["inter_method"]),
        active_sources=tuple(str(source) for source in manifest.active_sources),
    )


def attach_common_plot_columns(*, frame: pd.DataFrame, context: FigureContext) -> pd.DataFrame:
    """Attach method, functional unit, and normalized row owned SSP columns."""
    work = frame.copy()
    work["fu_code"] = context.fu_code
    work["__method"] = method_labels(work)
    return with_normalized_row_ssp(frame=work)


def with_normalized_row_ssp(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize row owned SSP scenario labels without planning figure scopes."""
    work = frame.copy()
    if ASOCC_SSP_SCENARIO_COLUMN not in work.columns:
        return work
    series = pd.Series(work[ASOCC_SSP_SCENARIO_COLUMN], copy=False)
    scenario_mask = ~series.map(is_display_missing)
    work.loc[scenario_mask, ASOCC_SSP_SCENARIO_COLUMN] = (
        series.loc[scenario_mask].astype(str).str.upper()
    )
    return work


def method_labels(frame: pd.DataFrame) -> pd.Series:
    """Return one visible method label per row."""
    labels = pd.Series(["aSoCC"] * len(frame), index=frame.index, dtype="object")
    for column in reversed(METHOD_COLUMNS):
        if column not in frame.columns:
            continue
        values = pd.Series(frame[column], copy=False)
        mask = ~values.map(is_display_missing)
        labels.loc[mask] = values.loc[mask].astype(str).str.strip()
    return labels


def visible_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Return sorted nonmissing display values for one column."""
    return visible_scope_values(frame, column)


def single_requested_year(context: FigureContext) -> int | None:
    """Return the single requested year if this is a single year product."""
    years = sorted({int(year) for year in context.requested_years})
    return years[0] if len(years) == 1 else None


def scoped_stem(
    label: str,
    frame: pd.DataFrame,
    *,
    include_impact: bool,
    selector_token: str = "all",
    studied_year: int | None = None,
) -> str:
    """Return one deterministic file stem for a figure scope."""
    return asocc_scope_stem(
        label,
        frame,
        include_impact=include_impact,
        selector_token=selector_token,
        studied_year=studied_year,
    )


def scope_title(
    label: str | None,
    frame: pd.DataFrame,
    *,
    selector_title: str | None = None,
    studied_year: int | None,
) -> str:
    """Return one compact aSoCC uncertainty figure title."""
    parts = ["aSoCC uncertainty"]
    if label is not None:
        parts.append(label)
    if selector_title is not None and str(selector_title).strip():
        parts.append(str(selector_title).strip())
    parts.extend(lcia_title_parts(frame, include_impact=False))
    if studied_year is not None:
        parts.append(str(int(studied_year)))
    ssp_values = visible_values(frame, ASOCC_SSP_SCENARIO_COLUMN)
    if ssp_values:
        parts.append(f"Prospective: {ssp_values[0]}")
    return " | ".join(parts)


def _years_from_args(args: dict[str, Any]) -> list[int]:
    raw = args.get("years")
    if isinstance(raw, int):
        return [int(raw)]
    return sorted({int(value) for value in cast(Iterable[Any], raw)})


def _ssp_from_args(args: dict[str, Any]) -> list[str]:
    raw = args.get("ssp_scenario")
    if raw is None:
        return []
    values = [raw] if isinstance(raw, str) else list(raw)
    return sorted({str(value).strip().upper() for value in values if str(value).strip()})
