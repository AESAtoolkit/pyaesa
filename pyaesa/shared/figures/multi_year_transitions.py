"""Shared multi-year transition helpers for historical and SSP figure routes."""

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import cast

import matplotlib.transforms as transforms
import pandas as pd

from pyaesa.shared.tabular.wide_tables import first_non_null_scenario_year

_MARKER_YEAR_COLUMN = "__transition_marker_year"
_MARKER_LABEL_COLUMN = "__transition_marker_label"
_MARKER_COLOR_COLUMN = "__transition_marker_color"
_PROSPECTIVE_SHADE_COLOR = "#d9d9d9"
_PROSPECTIVE_SHADE_ALPHA = 0.28
_GENERIC_SSP_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<scenario>SSP[0-9]+)$", re.IGNORECASE)
_GENERIC_TRANSITION_LABELS = frozenset(
    {
        "retrospective/prospective transition",
    }
)
_COMPONENT_TRANSITION_LABELS = frozenset({"aSoCC", "LCA"})
_GENERIC_TRANSITION_LABEL_Y_OFFSET_PT = 4.0
_COMPONENT_TRANSITION_LABEL_Y_OFFSET_PT = 4.0
_COMPONENT_TRANSITION_HEADER_Y_OFFSET_PT = 13.0
_SPECIFIC_TRANSITION_LABEL_Y_OFFSET_PT = 4.0
_SPECIFIC_TRANSITION_LABEL_LEVEL_STEP_PT = 9.0


@dataclass(frozen=True)
class TransitionMarker:
    """One labeled vertical marker rendered on a multi-year panel."""

    year: int
    label: str
    color: str


def marker_year_column() -> str:
    """Return the internal marker year column name."""
    return _MARKER_YEAR_COLUMN


def marker_label_column() -> str:
    """Return the internal marker label column name."""
    return _MARKER_LABEL_COLUMN


def marker_color_column() -> str:
    """Return the internal marker color column name."""
    return _MARKER_COLOR_COLUMN


def is_missing_scalar(value: object) -> bool:
    """Return whether one grouped scalar should be treated as missing."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, float):
        return pd.isna(value)
    return False


def normalized_requested_years(requested_years: Sequence[int]) -> list[int]:
    """Return one sorted, unique requested year list."""
    normalized = sorted({int(year) for year in requested_years})
    return normalized


def ssp_tokens_from_plan(
    *,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
    requested_years: Sequence[int],
) -> list[str]:
    """Return sorted non null SSP tokens present in the requested year plan."""
    if ssp_scenario_options_by_year is None:
        return []
    tokens = {
        str(value)
        for year in requested_years
        for value in (ssp_scenario_options_by_year.get(int(year), [None]) or [None])
        if value is not None
    }
    return sorted(tokens)


def stem_ssp_suffix(
    *,
    stem: str,
    ssp_tokens: Sequence[str],
) -> tuple[str, str | None]:
    """Split one file stem into base stem and trailing SSP suffix when present."""
    normalized = sorted({str(token) for token in ssp_tokens if str(token).strip()}, key=len)
    for token in reversed(normalized):
        suffix = f"_{token}"
        if str(stem).endswith(suffix):
            return str(stem)[: -len(suffix)], str(token)
    return str(stem), None


def generic_ssp_suffix(stem: str) -> tuple[str, str | None]:
    """Split one stem on a generic trailing ``SSP<n>`` suffix when present."""
    match = _GENERIC_SSP_SUFFIX_RE.fullmatch(str(stem).strip())
    if match is None:
        return str(stem), None
    return str(match.group("base")), str(match.group("scenario")).upper()


def non_null_prospective_values(
    frame: pd.DataFrame,
    *,
    prospective_column: str = "scenario",
) -> list[str]:
    """Return sorted non null prospective values from one long form frame."""
    if prospective_column not in frame.columns:
        return []
    series = pd.Series(frame.loc[:, prospective_column], copy=False)
    return sorted({str(value) for value in series.dropna().astype(str).tolist() if str(value)})


def expand_historical_rows_for_prospective_series(
    frame: pd.DataFrame,
    *,
    grouping_columns: list[str],
    propagate_columns: list[str] | None = None,
    year_column: str = "year",
    prospective_column: str = "scenario",
    marker_label: str = "prospective transition",
    marker_color: str = "#7d7d7d",
) -> pd.DataFrame:
    """Repeat historical rows into each prospective group and annotate switch markers.

    Groups that contain only historical rows are left unchanged and receive no
    marker metadata. Groups that contain only one or more prospective tagged rows
    are also left unchanged, because they do not represent a historical to SSP
    transition.
    """
    if year_column not in frame.columns or prospective_column not in frame.columns:
        out = frame.copy()
        out[_MARKER_YEAR_COLUMN] = pd.NA
        out[_MARKER_LABEL_COLUMN] = None
        out[_MARKER_COLOR_COLUMN] = None
        return out

    work = frame.copy()
    numeric_years = pd.to_numeric(pd.Series(work.loc[:, year_column], copy=False), errors="raise")
    work[year_column] = pd.Series(numeric_years, copy=False).astype(int)
    groups = (
        [((), work)]
        if not grouping_columns
        else list(work.groupby(grouping_columns, dropna=False, sort=True))
    )
    expanded: list[pd.DataFrame] = []
    for _group_key, group in groups:
        prospective_series = pd.Series(group.loc[:, prospective_column], copy=False)
        historical = group.loc[prospective_series.isna()].copy()
        propagation_columns = [
            column for column in (propagate_columns or []) if column in group.columns
        ]
        prospective_values = sorted(
            {str(value) for value in prospective_series.dropna().astype(str).tolist() if str(value)}
        )
        if not prospective_values:
            unchanged = group.copy()
            unchanged[_MARKER_YEAR_COLUMN] = pd.NA
            unchanged[_MARKER_LABEL_COLUMN] = None
            unchanged[_MARKER_COLOR_COLUMN] = None
            expanded.append(unchanged)
            continue
        for prospective_value in prospective_values:
            prospective_rows = group.loc[
                prospective_series.astype(str) == str(prospective_value)
            ].copy()
            switch_year = first_non_null_scenario_year(
                prospective_rows,
                scenario_column=prospective_column,
                year_column=year_column,
            )
            if historical.empty:
                prospective_rows[_MARKER_YEAR_COLUMN] = pd.NA
                prospective_rows[_MARKER_LABEL_COLUMN] = None
                prospective_rows[_MARKER_COLOR_COLUMN] = None
                expanded.append(prospective_rows)
                continue
            repeated_historical = historical.copy()
            repeated_historical.loc[:, prospective_column] = str(prospective_value)
            if propagation_columns:
                prospective_seed = prospective_rows.iloc[0]
                for column in propagation_columns:
                    repeated_historical.loc[:, column] = prospective_seed[column]
            combined = pd.concat([repeated_historical, prospective_rows], ignore_index=True)
            combined[_MARKER_YEAR_COLUMN] = int(cast(int, switch_year))
            combined[_MARKER_LABEL_COLUMN] = str(marker_label)
            combined[_MARKER_COLOR_COLUMN] = str(marker_color)
            expanded.append(combined)
    out = pd.concat(expanded, ignore_index=True) if expanded else work.iloc[0:0].copy()
    return out


def markers_from_frame(frame: pd.DataFrame) -> list[TransitionMarker]:
    """Return unique panel markers from one prepared frame."""
    required = {_MARKER_YEAR_COLUMN, _MARKER_LABEL_COLUMN, _MARKER_COLOR_COLUMN}
    if not required.issubset(frame.columns):
        return []
    if frame.empty:
        return []
    markers: dict[tuple[int, str, str], TransitionMarker] = {}
    marker_frame = frame.loc[:, [_MARKER_YEAR_COLUMN, _MARKER_LABEL_COLUMN, _MARKER_COLOR_COLUMN]]
    for year, label, color in marker_frame.itertuples(index=False, name=None):
        if year is None or pd.isna(year) or label in {None, ""} or bool(pd.isna(label)):
            continue
        key = (int(year), str(label), str(color or "#7d7d7d"))
        markers[key] = TransitionMarker(
            year=int(year),
            label=str(label),
            color=str(color or "#7d7d7d"),
        )
    return sorted(markers.values(), key=lambda item: (item.year, item.label, item.color))


def render_transition_markers(
    axis,
    *,
    markers: Sequence[TransitionMarker],
    shade_right: float | None = None,
) -> None:
    """Render shared retrospective/prospective transition markers on one axis."""
    if not markers:
        return
    trans = transforms.blended_transform_factory(axis.transData, axis.transAxes)
    _render_prospective_shade(axis, markers=markers, shade_right=shade_right)
    for marker in markers:
        x = transition_boundary_x(marker.year)
        axis.axvline(
            x,
            color=str(marker.color),
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            zorder=0,
        )
    unique_markers = {
        (float(marker.year), str(marker.color), str(marker.label)): marker for marker in markers
    }
    marker_values = tuple(unique_markers.values())
    if _has_component_transition_pair(marker_values):
        _render_component_transition_labels(
            axis,
            markers=sorted(marker_values, key=lambda item: item.year),
            transform=trans,
        )
        return
    specific_label_levels: dict[int, float] = {}
    for marker in sorted(unique_markers.values(), key=lambda item: (item.year, item.label)):
        x = transition_boundary_x(marker.year)
        if str(marker.label) in _GENERIC_TRANSITION_LABELS:
            _render_generic_transition_labels(axis, marker=marker, x=x, transform=trans)
        else:
            level = _specific_label_level(x=x, last_x_by_level=specific_label_levels)
            axis.text(
                x,
                1.0,
                str(marker.label),
                color=str(marker.color),
                ha="center",
                va="bottom",
                fontsize=8,
                transform=_offset_transform(
                    axis,
                    transform=trans,
                    y_offset_pt=(
                        _SPECIFIC_TRANSITION_LABEL_Y_OFFSET_PT
                        + _SPECIFIC_TRANSITION_LABEL_LEVEL_STEP_PT * float(level)
                    ),
                ),
                clip_on=False,
                linespacing=0.9,
            )


def transition_title_pad(
    markers: Sequence[TransitionMarker],
    *,
    no_transition: int,
    single_transition: int,
    component_transition: int,
) -> int:
    """Return title padding matched to the visible transition label layout."""
    if not markers:
        return int(no_transition)
    if _has_component_transition_pair(markers):
        return int(component_transition)
    return int(single_transition)


def _specific_label_level(*, x: float, last_x_by_level: dict[int, float]) -> int:
    level = 0
    while level in last_x_by_level and abs(float(x) - last_x_by_level[level]) < 1.25:
        level += 1
    last_x_by_level[level] = float(x)
    return level


def _has_component_transition_pair(markers: Sequence[TransitionMarker]) -> bool:
    labels = {str(marker.label) for marker in markers}
    return len(markers) == 2 and labels == _COMPONENT_TRANSITION_LABELS


def _render_component_transition_labels(
    axis,
    *,
    markers: Sequence[TransitionMarker],
    transform,
) -> None:
    ordered = sorted(markers, key=lambda marker: transition_boundary_x(marker.year))
    left = ordered[0]
    right = ordered[-1]
    left_x = transition_boundary_x(left.year)
    right_x = transition_boundary_x(right.year)
    color = str(left.color)
    axis.text(
        (left_x + right_x) / 2.0,
        1.0,
        "Prospective start year",
        color=color,
        ha="center",
        va="bottom",
        fontsize=8,
        transform=_offset_transform(
            axis,
            transform=transform,
            y_offset_pt=_COMPONENT_TRANSITION_HEADER_Y_OFFSET_PT,
        ),
        clip_on=False,
    )
    for marker, x, horizontal_alignment in (
        (left, left_x, "right"),
        (right, right_x, "left"),
    ):
        axis.text(
            x,
            1.0,
            str(marker.label),
            color=str(marker.color),
            ha=horizontal_alignment,
            va="bottom",
            fontsize=8,
            transform=_offset_transform(
                axis,
                transform=transform,
                y_offset_pt=_COMPONENT_TRANSITION_LABEL_Y_OFFSET_PT,
            ),
            clip_on=False,
        )


def _render_generic_transition_labels(
    axis, *, marker: TransitionMarker, x: float, transform
) -> None:
    axis.text(
        x - 0.12,
        1.0,
        "retrospective",
        color=str(marker.color),
        ha="right",
        va="bottom",
        fontsize=8,
        transform=_offset_transform(
            axis,
            transform=transform,
            y_offset_pt=_GENERIC_TRANSITION_LABEL_Y_OFFSET_PT,
        ),
        clip_on=False,
    )
    axis.text(
        x + 0.12,
        1.0,
        "prospective",
        color=str(marker.color),
        ha="left",
        va="bottom",
        fontsize=8,
        transform=_offset_transform(
            axis,
            transform=transform,
            y_offset_pt=_GENERIC_TRANSITION_LABEL_Y_OFFSET_PT,
        ),
        clip_on=False,
    )


def _offset_transform(axis, *, transform, y_offset_pt: float):
    return transform + transforms.ScaledTranslation(
        0.0,
        float(y_offset_pt) / 72.0,
        axis.figure.dpi_scale_trans,
    )


def _render_prospective_shade(
    axis,
    *,
    markers: Sequence[TransitionMarker],
    shade_right: float | None,
) -> None:
    boundaries = [transition_boundary_x(marker.year) for marker in markers]
    xmin = min(boundaries)
    left, right = axis.get_xlim()
    if float(right) <= float(xmin):
        return
    span_right = (
        float(shade_right)
        if shade_right is not None
        else float(right) + max(1e-9, abs(float(right) - float(left)) * 0.02)
    )
    axis.axvspan(
        xmin,
        span_right,
        facecolor=_PROSPECTIVE_SHADE_COLOR,
        alpha=_PROSPECTIVE_SHADE_ALPHA,
        linewidth=0,
        zorder=0,
        clip_on=True,
    )
    axis.set_xlim(left, right)


def transition_boundary_x(year: int) -> float:
    """Return the visible x position of one retrospective/prospective divider."""
    return float(year)
