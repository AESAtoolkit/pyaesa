"""Deterministic ASR figure path and grouping ownership."""

from pathlib import Path

from pyaesa.shared.figures.deterministic_transition_groups import (
    GroupedTransitionFile,
    group_files_by_base,
)
from pyaesa.shared.runtime.io.persisted_paths import scoped_existing_table_paths

from .state import l1_l2_methods_by_path


def resolve_scoped_figure_paths(
    *,
    root: Path,
    output_paths: list[Path],
    field_name: str,
    family_label: str,
) -> list[Path]:
    """Return persisted ASR figure input paths scoped to one deterministic root."""
    scoped = scoped_existing_table_paths(
        raw_paths=output_paths,
        root=root,
        field_name=field_name,
    )
    if not scoped:
        raise ValueError(f"{family_label} figure scope has no persisted inputs under '{root}'.")
    return scoped


def resolve_grouped_figure_inputs(
    *,
    root: Path,
    output_paths: list[Path],
    field_name: str,
    share_transition_meta: dict[str, dict[str, object]],
    family_label: str,
) -> list[tuple[Path, str, list[GroupedTransitionFile]]]:
    """Return deterministic grouped figure inputs for one ASR output root."""
    scoped_paths = resolve_scoped_figure_paths(
        root=root,
        output_paths=output_paths,
        field_name=field_name,
        family_label=family_label,
    )
    return group_files_by_base(
        root=root,
        paths=scoped_paths,
        share_transition_meta=share_transition_meta,
        l1_l2_methods_by_path=l1_l2_methods_by_path(scoped_paths, family_label=family_label),
    )
