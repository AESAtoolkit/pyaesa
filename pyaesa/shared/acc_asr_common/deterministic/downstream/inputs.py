"""Shared downstream aSoCC share loading for deterministic aCC and ASR."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.asocc.runtime.scope.branch_resolution import (
    asocc_l1_dir,
    asocc_l2_dir,
    build_asocc_deterministic_path_scope,
)
from pyaesa.external_inputs.asocc.deterministic.downstream_shares import load_external_asocc_shares
from pyaesa.shared.tabular.contracts import TABULAR_SUFFIX_SET
from pyaesa.shared.tabular.table_io import read_table


def read_share_file(path: Path) -> pd.DataFrame:
    """Read one allocation share output file."""
    return read_table(path=path)


def collect_share_files(root: Path) -> list[Path]:
    """Collect all share output files under one root recursively."""
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in TABULAR_SUFFIX_SET
    )


def _l1_share_files(root: Path) -> list[Path]:
    """Collect deterministic L1 share files without descending into nested L2 routes."""
    return [path for path in collect_share_files(root) if "level_2" not in path.parts]


@dataclass(frozen=True)
class AsoccShare:
    """One aSoCC share source used by downstream aCC and ASR."""

    file_stem: str
    relative_dir: Path
    impacts: tuple[str, ...]
    source_label: str
    path: Path | None = None
    frame_wide: pd.DataFrame | None = None

    def read(self) -> pd.DataFrame:
        """Return the concrete wide share table."""
        if self.frame_wide is not None:
            return self.frame_wide.copy()
        return read_share_file(cast(Path, self.path))

    @property
    def display_name(self) -> str:
        """Return the user facing source label for status messages."""
        if self.path is not None:
            return self.path.name
        return f"{self.file_stem}.<external>"


@dataclass(frozen=True)
class LoadedAsoccShare:
    """One branch local downstream aSoCC share table loaded exactly once."""

    file_stem: str
    relative_dir: Path
    impacts: tuple[str, ...]
    source_label: str
    display_name: str
    reference_path: Path
    frame_wide: pd.DataFrame


def load_asocc_share(asocc_share: AsoccShare) -> LoadedAsoccShare:
    """Materialize one downstream aSoCC share for branch local reuse."""
    frame = asocc_share.read()
    return LoadedAsoccShare(
        file_stem=asocc_share.file_stem,
        relative_dir=asocc_share.relative_dir,
        impacts=asocc_share.impacts,
        source_label=asocc_share.source_label,
        display_name=asocc_share.display_name,
        reference_path=asocc_share_reference_path(asocc_share),
        frame_wide=frame,
    )


def native_asocc_shares(
    *,
    proj_base: Path,
    source_label: str,
    fu_code: str,
    base_allocate_args: dict[str, Any],
) -> list[AsoccShare]:
    """Return all native aSoCC share files for one source scope."""
    scope = build_asocc_deterministic_path_scope(
        proj_base=proj_base,
        source_label=source_label,
        agg_version=base_allocate_args["agg_version"],
    )
    l1_root = asocc_l1_dir(scope=scope, lcia_sub=None, fu_code=fu_code)
    l2_vs_global_root = asocc_l2_dir(scope=scope, bucket="l2_vs_global", lcia_sub=None)
    fu_text = str(fu_code).strip()
    if fu_text.startswith("L1."):
        share_files = _l1_share_files(l1_root)
    elif fu_text.startswith("L2."):
        share_files = collect_share_files(l2_vs_global_root)
    else:
        share_files = _l1_share_files(l1_root) + collect_share_files(l2_vs_global_root)
    out: list[AsoccShare] = []
    for path in share_files:
        out.append(
            AsoccShare(
                file_stem=path.stem,
                relative_dir=_relative_share_path(path),
                impacts=tuple(),
                source_label="native",
                path=path,
            )
        )
    return out


def external_asocc_shares(
    *,
    proj_base: Path,
    base_allocate_args: dict[str, Any],
    fu_code: str,
    external_method: dict[str, Any] | None,
    years: list[int],
    lcia_method: str | None,
    output_source_label: str,
) -> list[AsoccShare]:
    """Return deterministic external aSoCC share tables for one branch scope."""
    if external_method is None:
        return []
    resolved = load_external_asocc_shares(
        proj_base=proj_base,
        fu_code=fu_code,
        external_method=external_method,
        years=years,
        lcia_method=lcia_method,
        base_allocate_args=base_allocate_args,
        output_source_label=output_source_label,
    )
    return [
        AsoccShare(
            file_stem=item.file_stem,
            relative_dir=item.relative_dir,
            impacts=item.impacts,
            source_label="external",
            frame_wide=item.frame_wide,
        )
        for item in resolved
    ]


def combined_asocc_shares(
    *,
    proj_base: Path,
    source_label: str,
    base_allocate_args: dict[str, Any],
    fu_code: str,
    external_method: dict[str, Any] | None,
    years: list[int],
    lcia_method: str | None,
    output_source_label: str,
) -> list[AsoccShare]:
    """Return native plus deterministic external aSoCC shares."""
    return [
        *native_asocc_shares(
            proj_base=proj_base,
            source_label=source_label,
            fu_code=fu_code,
            base_allocate_args=base_allocate_args,
        ),
        *external_asocc_shares(
            proj_base=proj_base,
            base_allocate_args=base_allocate_args,
            fu_code=fu_code,
            external_method=external_method,
            years=years,
            lcia_method=lcia_method,
            output_source_label=output_source_label,
        ),
    ]


def asocc_share_reference_path(asocc_share: AsoccShare) -> Path:
    """Return one path-like stem reference for deterministic share identity parsing."""
    if asocc_share.path is not None:
        return asocc_share.path
    return Path(f"{asocc_share.file_stem}.csv")


def _relative_share_path(share_path: Path) -> Path:
    """Return the canonical downstream public relative path for one aSoCC table."""
    parts = list(share_path.parts)
    if "level_1" in parts:
        idx = parts.index("level_1")
        rel_parts = parts[idx + 1 : -1]
        return Path(*rel_parts) if rel_parts else Path(".")
    if "level_2" in parts:
        idx = parts.index("level_2")
        rel_parts = parts[idx + 1 : -1]
        if rel_parts[:1] == ["l2_vs_global"]:
            rel_parts = rel_parts[1:]
        return Path(*rel_parts) if rel_parts else Path(".")
    return Path(".")
