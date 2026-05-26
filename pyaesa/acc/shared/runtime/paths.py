"""Shared runtime path ownership for aCC deterministic flows."""

from pathlib import Path

from pyaesa.asocc.runtime.paths.family_roots import (
    asocc_source_version_token,
)
from pyaesa.asocc.runtime.output.contracts import join_file_owned_tokens
from pyaesa.shared.runtime.io.family_root_names import ACC_ROOT_DIRNAME


def public_result_root_name_for_fu_code(*, fu_code: str) -> str:
    """Return the canonical public result root for one FU code."""
    return "results_l2_vs_global" if str(fu_code).strip().startswith("L2.") else "results"


def build_acc_output_stem(
    *,
    base_stem: str,
    cc_source: str,
    cc_type: str,
) -> str:
    """Return one deterministic or uncertainty aCC file stem."""
    normalized_base = str(base_stem).strip()
    normalized_cc_source = str(cc_source).strip()
    base_pieces = [piece.strip() for piece in normalized_base.split("__") if piece.strip()]
    cc_type_norm = str(cc_type).strip().lower()
    source_present = normalized_cc_source in base_pieces
    stem = list(base_pieces)
    if not source_present:
        stem.append(normalized_cc_source)
    if cc_type_norm == "dynamic_ar6":
        stem.append("dynamic_ar6")
    return join_file_owned_tokens(*stem)


def get_acc_root(
    *,
    proj_base: Path,
    source_label: str,
    agg_version: str | None,
) -> Path:
    """Return the source-scoped aCC root shared by deterministic and MC branches."""
    return (
        Path(proj_base)
        / ACC_ROOT_DIRNAME
        / asocc_source_version_token(
            source=source_label,
            agg_version=agg_version,
        )
    )
