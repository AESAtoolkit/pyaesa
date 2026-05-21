"""Deterministic ASR figure row preparation."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyaesa.shared.figures.deterministic_transition_groups import (
    GroupedTransitionFile,
    grouped_transition_method_identity,
    long_frame_from_group,
    title_stem,
)
from pyaesa.shared.figures.lcia_scope import resolve_unique_lcia_method


@dataclass(frozen=True)
class PreparedAsrFigureGroup:
    """One ASR figure family prepared from persisted deterministic result tables."""

    relative_parent: Path
    base_stem: str
    grouped_files: list[GroupedTransitionFile]
    rows: pd.DataFrame
    title_label: str
    method_label: str
    marker_label: str
    marker_color: str


RawAsrFigureGroup = tuple[Path, str, list[GroupedTransitionFile]]


def prepare_asr_figure_groups(
    *,
    groups: list[RawAsrFigureGroup],
    requested_years: list[int],
    fu_code: str,
) -> list[PreparedAsrFigureGroup]:
    """Read and normalize deterministic ASR figure rows once per grouped family."""
    prepared: list[PreparedAsrFigureGroup] = []
    for relative_parent, base_stem, grouped_files in groups:
        method_label = grouped_transition_method_identity(
            relative_parent=relative_parent,
            base_stem=base_stem,
            grouped_files=grouped_files,
        )
        rows = _validated_asr_rows(
            long_frame_from_group(
                grouped_files=grouped_files,
                requested_years=requested_years,
            )
        )
        rows["__method"] = method_label
        rows["fu_code"] = str(fu_code)
        prepared.append(
            PreparedAsrFigureGroup(
                relative_parent=relative_parent,
                base_stem=base_stem,
                grouped_files=grouped_files,
                rows=rows,
                title_label=title_stem(
                    relative_parent=relative_parent,
                    base_stem=base_stem,
                    lcia_method=resolve_unique_lcia_method(rows),
                ),
                method_label=method_label,
                marker_label=grouped_files[0].marker_label,
                marker_color=grouped_files[0].marker_color,
            )
        )
    return _merge_dynamic_groups(prepared)


def _merge_dynamic_groups(groups: list[PreparedAsrFigureGroup]) -> list[PreparedAsrFigureGroup]:
    if not any("cumulative_asr" in group.rows.columns for group in groups):
        return groups
    merged: list[PreparedAsrFigureGroup] = []
    for method_label in dict.fromkeys(group.method_label for group in groups):
        method_groups = [group for group in groups if group.method_label == method_label]
        first = method_groups[0]
        rows = pd.concat([group.rows for group in method_groups], ignore_index=True)
        grouped_files = [
            grouped_file for group in method_groups for grouped_file in group.grouped_files
        ]
        merged.append(
            PreparedAsrFigureGroup(
                relative_parent=first.relative_parent,
                base_stem=first.base_stem,
                grouped_files=grouped_files,
                rows=rows,
                title_label=title_stem(
                    relative_parent=first.relative_parent,
                    base_stem=first.base_stem,
                    lcia_method=resolve_unique_lcia_method(rows),
                ),
                method_label=first.method_label,
                marker_label=first.marker_label,
                marker_color=first.marker_color,
            )
        )
    return merged


def has_multiple_prepared_asr_groups(groups: list[PreparedAsrFigureGroup]) -> bool:
    """Return whether prepared ASR rows span more than one allocation method."""
    return len({group.method_label for group in groups}) > 1


def combined_prepared_asr_rows(groups: list[PreparedAsrFigureGroup]) -> pd.DataFrame:
    """Return one combined deterministic ASR figure frame from prepared groups."""
    frames: list[pd.DataFrame] = []
    for group in groups:
        rows = group.rows.copy()
        rows["combined_group_label"] = group.method_label
        rows["__method"] = group.method_label
        frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def all_prepared_asr_rows(groups: list[PreparedAsrFigureGroup]) -> pd.DataFrame:
    """Return all prepared ASR rows without rereading result tables."""
    frames = [group.rows for group in groups]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _validated_asr_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = pd.Series(pd.to_numeric(frame["value"], errors="raise"), copy=False)
    out = frame.copy()
    out["value"] = values
    return out
