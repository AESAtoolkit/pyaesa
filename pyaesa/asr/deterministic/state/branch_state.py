"""Deterministic ASR branch state."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .metadata import cached_path_list, cached_text_list


@dataclass(frozen=True)
class DeterministicBranchState:
    """Branch summary state produced by deterministic ASR."""

    cc_source: str
    cc_type: str
    cc_bounds: list[str]
    impacts_used: list[str]
    figure_paths: list[Path]
    output_dirs: list[Path]
    meta_file: Path | None


def cached_branch_state(
    *,
    existing_metadata: dict[str, Any],
    figure_paths: list[Path],
    meta_path: Path,
) -> DeterministicBranchState:
    """Return ASR branch state derived from cached metadata."""
    provenance = existing_metadata["provenance"]
    return DeterministicBranchState(
        cc_source=str(provenance["cc_source"]),
        cc_type=str(provenance["cc_type"]),
        cc_bounds=cached_text_list(
            existing_metadata=existing_metadata,
            field_name="cc_bounds",
        ),
        impacts_used=cached_text_list(
            existing_metadata=existing_metadata,
            field_name="impacts",
        ),
        figure_paths=figure_paths,
        output_dirs=cached_path_list(
            existing_metadata=existing_metadata,
            field_name="output_dirs",
        ),
        meta_file=meta_path,
    )


def written_branch_state(
    *,
    cc_source: str,
    cc_type: str,
    cc_bounds: list[str],
    impacts_used: list[str],
    figure_paths: list[Path],
    output_dirs: list[Path],
    meta_path: Path,
) -> DeterministicBranchState:
    """Return ASR branch state derived from a freshly written branch."""
    return DeterministicBranchState(
        cc_source=cc_source,
        cc_type=cc_type,
        cc_bounds=list(cc_bounds),
        impacts_used=list(impacts_used),
        figure_paths=list(figure_paths),
        output_dirs=list(output_dirs),
        meta_file=meta_path,
    )
