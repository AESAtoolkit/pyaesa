"""Shared grouping helpers for deterministic wide table transition figures."""

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.deterministic_companion_stems import (
    parse_deterministic_companion_stem,
)
from pyaesa.shared.tabular.scalars import sanitize_token
from pyaesa.shared.tabular.l2_reuse_years import (
    canonicalize_l2_reuse_year_column,
    frame_l2_reuse_years as resolved_frame_l2_reuse_years,
)
from pyaesa.shared.tabular.table_io import read_table
from pyaesa.shared.tabular.wide_tables import melt_requested_year_value_rows
from .lcia_metadata import ensure_frame_lcia_method_metadata

_OUTPUT_SUFFIX_RE = re.compile(
    r"__(?:min_cc|max_cc|[A-Za-z0-9_.-]+__min_cc|[A-Za-z0-9_.-]+__max_cc|.+__.+)$"
)


@dataclass(frozen=True)
class GroupedTransitionFile:
    """One deterministic file grouped by its normalized share base stem."""

    path: Path
    l1_l2_method: str
    base_stem: str
    marker_label: str
    marker_color: str


def group_files_by_base(
    *,
    root: Path,
    paths: list[Path],
    share_transition_meta: dict[str, dict[str, object]],
    l1_l2_methods_by_path: dict[Path, str],
) -> list[tuple[Path, str, list[GroupedTransitionFile]]]:
    """Group aCC and ASR files by normalized share base stem."""
    grouped: dict[tuple[Path, str], list[GroupedTransitionFile]] = {}
    for path in paths:
        l1_l2_method = l1_l2_methods_by_path[path]
        payload = _share_transition_payload(
            output_path=path,
            share_transition_meta=share_transition_meta,
            l1_l2_method=l1_l2_method,
        )
        base_stem = str(payload.get("base_stem", l1_l2_method))
        grouped.setdefault(
            (normalize_companion_relative_parent(path.relative_to(root).parent), base_stem),
            [],
        ).append(
            GroupedTransitionFile(
                path=path,
                l1_l2_method=str(l1_l2_method),
                base_stem=base_stem,
                marker_label=str(payload.get("marker_label", "prospective transition")),
                marker_color=str(payload.get("marker_color", "#7d7d7d")),
            )
        )
    return [
        (parent, base_stem, sorted(grouped[(parent, base_stem)], key=lambda item: item.path.name))
        for parent, base_stem in sorted(grouped)
    ]


def long_frame_from_group(
    *,
    grouped_files: list[GroupedTransitionFile],
    requested_years: list[int],
) -> pd.DataFrame:
    """Return one long deterministic frame for one grouped share family."""
    return long_frame_from_loaded_group(
        grouped_frames=[(item, read_table(path=item.path)) for item in grouped_files],
        requested_years=requested_years,
    )


def long_frame_from_loaded_group(
    *,
    grouped_frames: list[tuple[GroupedTransitionFile, pd.DataFrame]],
    requested_years: list[int],
) -> pd.DataFrame:
    """Return one long deterministic frame from already loaded grouped wide tables."""
    normalized_group_frames: list[tuple[GroupedTransitionFile, pd.DataFrame, tuple[int, ...]]] = []
    l2_reuse_years: set[int] = set()
    for item, raw_frame in grouped_frames:
        frame = canonicalize_l2_reuse_year_column(raw_frame, path=item.path)
        item_l2_reuse_years = resolved_frame_l2_reuse_years(frame)
        normalized_group_frames.append((item, frame, item_l2_reuse_years))
        l2_reuse_years.update(item_l2_reuse_years)
    sorted_l2_reuse_years = sorted(l2_reuse_years)
    long_frames: list[pd.DataFrame] = []
    for item, frame, frame_l2_reuse_years in normalized_group_frames:
        if frame_l2_reuse_years:
            long_frames.append(
                _wide_to_long(
                    frame=frame,
                    requested_years=requested_years,
                )
            )
            continue
        if not _frame_has_active_asocc_scenario(frame, requested_years) and sorted_l2_reuse_years:
            for l2_reuse_year in sorted_l2_reuse_years:
                long_frames.append(
                    _wide_to_long(
                        frame=_with_l2_reuse_year(frame=frame, l2_reuse_year=l2_reuse_year),
                        requested_years=requested_years,
                    )
                )
            continue
        long_frames.append(
            _wide_to_long(
                frame=frame,
                requested_years=requested_years,
            )
        )
    return pd.concat(long_frames, ignore_index=True)


def combined_long_frame_from_groups(
    *,
    groups: list[tuple[Path, str, list[GroupedTransitionFile]]],
    requested_years: list[int],
    fu_code: str,
) -> pd.DataFrame:
    """Return one combined long frame for grouped deterministic transition families."""
    frames: list[pd.DataFrame] = []
    for _relative_parent, base_stem, grouped_files in groups:
        family = long_frame_from_group(
            grouped_files=grouped_files,
            requested_years=requested_years,
        )
        family["fu_code"] = str(fu_code).strip()
        family["combined_group_label"] = grouped_transition_method_identity(
            relative_parent=_relative_parent,
            base_stem=base_stem,
            grouped_files=grouped_files,
        )
        frames.append(family)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def has_multiple_grouped_transition_groups(
    groups: list[tuple[Path, str, list[GroupedTransitionFile]]],
) -> bool:
    """Return whether grouped deterministic figure inputs span more than one method family."""
    return (
        len(
            {
                grouped_transition_method_identity(
                    relative_parent=relative_parent,
                    base_stem=base_stem,
                    grouped_files=grouped_files,
                )
                for relative_parent, base_stem, grouped_files in groups
            }
        )
        > 1
    )


def grouped_transition_method_identity(
    *,
    relative_parent: Path,
    base_stem: str,
    grouped_files: list[GroupedTransitionFile],
) -> str:
    """Return the canonical grouped deterministic method identity."""
    del relative_parent, base_stem
    return str(grouped_files[0].l1_l2_method).strip()


def normalize_companion_base_stem(stem: str) -> str:
    """Return the shared base stem for historical and projected companion files."""
    return parse_deterministic_companion_stem(stem).base_stem


def normalize_companion_relative_parent(relative_parent: Path) -> Path:
    """Return one grouped family parent without companion route subfolders."""
    parts = [part for part in relative_parent.parts if part not in {"", "."}]
    normalized: list[str] = []
    index = 0
    while index < len(parts):
        part = parts[index]
        if part in {"results", "level_1", "level_2", "l2_vs_global"}:
            index += 1
            continue
        if part == "l2_in_l1":
            normalized.append("l2_in_l1")
            index += 1
            continue
        if part == "utility_propagation_contrib":
            normalized.append("utility_propagation_contrib")
            index += 1
            continue
        if part == "regression_proj":
            index += 1
            continue
        if part == "historical_reuse":
            index += 1
            continue
        normalized.append(part)
        index += 1
    return Path(*normalized) if normalized else Path(".")


def origin_share_stem_from_output_stem(
    *,
    output_stem: str,
    share_transition_meta: dict[str, dict[str, object]],
) -> str | None:
    """Return the originating deterministic share stem for one downstream output stem."""
    stem = str(output_stem).strip()
    if not stem:
        return None
    if stem in share_transition_meta:
        return stem
    candidates = [
        key
        for key in share_transition_meta
        if str(key).strip()
        and (stem == str(key).strip() or stem.startswith(f"{str(key).strip()}__"))
    ]
    if candidates:
        return max(candidates, key=len)
    normalized_stem = _normalized_transition_stem(stem)
    normalized_candidates = [
        key
        for key in share_transition_meta
        if str(key).strip()
        and (
            normalized_stem == _normalized_transition_stem(str(key).strip())
            or normalized_stem.startswith(f"{_normalized_transition_stem(str(key).strip())}__")
        )
    ]
    if normalized_candidates:
        return max(normalized_candidates, key=lambda key: len(_normalized_transition_stem(key)))
    return None


def title_stem(
    *,
    relative_parent: Path,
    base_stem: str,
    lcia_method: str | None = None,
) -> str:
    """Return one human-readable stem label from a grouped output path."""
    stem = str(base_stem)
    if lcia_method is not None:
        method_text = str(lcia_method).strip()
        for suffix in (f"__{method_text}", f"_{method_text}"):
            if method_text and stem.endswith(suffix):
                candidate = stem[: -len(suffix)].strip("_")
                stem = candidate or stem
                break
    parts = [part for part in relative_parent.parts if part not in {".", ""}]
    return " / ".join([*parts, stem]) if parts else stem


def _wide_to_long(
    *,
    frame: pd.DataFrame,
    requested_years: list[int],
) -> pd.DataFrame:
    frame = ensure_frame_lcia_method_metadata(frame)
    long_frame = melt_requested_year_value_rows(
        frame,
        requested_years=requested_years,
    )
    return long_frame.reset_index(drop=True)


def _frame_has_active_asocc_scenario(frame: pd.DataFrame, requested_years: list[int]) -> bool:
    """Return whether one wide frame owns an active aSoCC SSP label."""
    if ASOCC_SSP_SCENARIO_COLUMN not in frame.columns:
        return False
    year_columns = [str(year) for year in requested_years if str(year) in frame.columns]
    active_rows = frame.loc[:, year_columns].notna().any(axis=1)
    labels = pd.Series(frame.loc[active_rows, ASOCC_SSP_SCENARIO_COLUMN], copy=False).dropna()
    return bool(labels.astype(str).str.strip().ne("").any())


def _share_transition_payload(
    *,
    output_path: Path,
    share_transition_meta: dict[str, dict[str, object]],
    l1_l2_method: str,
) -> dict[str, object]:
    """Return transition metadata for one downstream output path."""
    origin_share_stem = origin_share_stem_from_output_stem(
        output_stem=output_path.stem,
        share_transition_meta=share_transition_meta,
    )
    if origin_share_stem is not None:
        return share_transition_meta.get(origin_share_stem, {})
    return share_transition_meta.get(str(l1_l2_method), {})


def _candidate_output_share_stem(output_stem: str) -> str:
    """Return the likely originating share stem encoded in one downstream output stem."""
    return normalize_companion_base_stem(_raw_candidate_output_share_stem(output_stem))


def _raw_candidate_output_share_stem(output_stem: str) -> str:
    """Return the likely originating share stem before reuse normalization."""
    stem = str(output_stem).strip()
    if "__" not in stem:
        return stem
    head, _sep, _tail = stem.rpartition("__")
    while "__" in head and _OUTPUT_SUFFIX_RE.search(head):
        head, _sep, _tail = head.rpartition("__")
    return head


def _with_l2_reuse_year(*, frame: pd.DataFrame, l2_reuse_year: int) -> pd.DataFrame:
    """Return one frame copy annotated with its historical l2_reuse_year."""
    out = frame.copy()
    out["l2_reuse_year"] = int(l2_reuse_year)
    return out


def _normalized_transition_stem(stem: str) -> str:
    """Return one sanitized deterministic stem for cross-owner path matching."""
    return "__".join(sanitize_token(piece) for piece in str(stem).split("__") if str(piece).strip())
