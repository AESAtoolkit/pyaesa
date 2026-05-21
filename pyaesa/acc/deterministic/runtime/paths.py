"""Deterministic aCC path ownership aligned to the deterministic aSoCC branch scope."""

from dataclasses import dataclass
from pathlib import Path

from pyaesa.asocc.runtime.paths.family_roots import asocc_source_version_token
from pyaesa.acc.shared.runtime.paths import get_acc_root
from pyaesa.shared.acc_asr_common.branches.config import cc_branch_token
from pyaesa.shared.runtime.metadata.contracts import (
    FIGURE_MANIFEST_FILENAME,
    SCOPE_MANIFEST_FILENAME,
)
from pyaesa.shared.runtime.io.filesystem import ensure_dir


@dataclass(frozen=True)
class ACCDeterministicPathContext:
    """Canonical deterministic aCC branch path context."""

    proj_base: Path
    source_label: str
    group_version: str | None
    cc_source: str
    cc_type: str
    public_result_root_name: str | None = None


def build_acc_path_context(
    *,
    proj_base: Path,
    source_label: str,
    group_version: str | None,
    cc_source: str,
    cc_type: str,
    public_result_root_name: str | None = None,
) -> ACCDeterministicPathContext:
    """Build one canonical deterministic aCC path context."""
    return ACCDeterministicPathContext(
        proj_base=Path(proj_base),
        source_label=str(source_label).strip(),
        group_version=group_version,
        cc_source=str(cc_source).strip(),
        cc_type=str(cc_type).strip().lower(),
        public_result_root_name=(
            None if public_result_root_name is None else str(public_result_root_name).strip()
        ),
    )


def build_acc_scope_label(
    *,
    source_label: str,
    group_version: str | None,
    cc_source: str,
    cc_type: str,
) -> str:
    """Return the deterministic aCC metadata scope label."""
    source_token = asocc_source_version_token(source=source_label, group_version=group_version)
    return f"{source_token}__{cc_branch_token(cc_source=cc_source, cc_type=cc_type)}"


def get_acc_branch_root(*, context: ACCDeterministicPathContext) -> Path:
    """Return the canonical deterministic aCC branch root."""
    return ensure_dir(
        get_acc_root(
            proj_base=context.proj_base,
            source_label=context.source_label,
            group_version=context.group_version,
        )
        / "deterministic"
        / cc_branch_token(cc_source=context.cc_source, cc_type=context.cc_type)
    )


def get_acc_output_dir(
    *,
    context: ACCDeterministicPathContext,
    public_result_root_name: str | None = None,
) -> Path:
    """Return the deterministic aCC output root for one resolved branch."""
    public_root = str(public_result_root_name or context.public_result_root_name or "").strip()
    if public_root:
        return ensure_dir(get_acc_branch_root(context=context) / public_root)
    return get_acc_branch_root(context=context)


def get_acc_logs_dir(*, context: ACCDeterministicPathContext) -> Path:
    """Return the deterministic aCC logs directory."""
    return get_acc_branch_root(context=context) / "logs"


def get_acc_meta_path(*, context: ACCDeterministicPathContext) -> Path:
    """Return the deterministic aCC scope-manifest path."""
    return get_acc_logs_dir(context=context) / SCOPE_MANIFEST_FILENAME


def get_acc_figure_metadata_path(*, context: ACCDeterministicPathContext) -> Path:
    """Return the deterministic aCC figure request metadata path."""
    return get_acc_logs_dir(context=context) / FIGURE_MANIFEST_FILENAME


def acc_output_relative_dir(*, upstream_relative_dir: Path) -> Path:
    """Return the canonical deterministic aCC relative output path from one aSoCC share."""
    parts = list(Path(upstream_relative_dir).parts)
    if not parts:
        return Path(".")
    rel_parts = parts
    while rel_parts and rel_parts[0] in {"level_1", "level_2"}:
        rel_parts = rel_parts[1:]
        if rel_parts and rel_parts[0] in {"results", "l2_vs_global"}:
            rel_parts = rel_parts[1:]
    if rel_parts and rel_parts[0] == "results":
        rel_parts = rel_parts[1:]
    return Path(*rel_parts) if rel_parts else Path(".")
