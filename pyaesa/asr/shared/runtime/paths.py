"""Deterministic ASR path ownership aligned to the deterministic aSoCC branch scope."""

from dataclasses import dataclass
from pathlib import Path

from pyaesa.asocc.runtime.paths.family_roots import asocc_source_version_token
from pyaesa.external_inputs.lca.naming import normalize_external_lca_version_name
from pyaesa.shared.acc_asr_common.branches.config import cc_branch_token
from pyaesa.shared.runtime.metadata.contracts import (
    FIGURE_MANIFEST_FILENAME,
    SCOPE_MANIFEST_FILENAME,
)
from pyaesa.shared.runtime.io.family_root_names import ASR_ROOT_DIRNAME
from pyaesa.shared.runtime.io.filesystem import ensure_dir


def _lca_route_token(*, lca_type: str, lca_version_name: str | None) -> str:
    """Return the ASR numerator route token."""
    if str(lca_type).strip().lower() == "io_lca":
        return "io_lca"
    version = normalize_external_lca_version_name(
        lca_version_name,
        argument_name="lca_args.external_lca.version_name",
    )
    return f"external_lca__{version}"


@dataclass(frozen=True)
class ASRDeterministicPathContext:
    """Canonical deterministic ASR branch path context."""

    proj_base: Path
    source_label: str
    group_version: str | None
    fu_code: str
    lca_type: str
    cc_source: str
    cc_type: str
    lca_version_name: str | None = None


def build_asr_path_context(
    *,
    proj_base: Path,
    source_label: str,
    group_version: str | None,
    fu_code: str,
    lca_type: str,
    cc_source: str,
    cc_type: str,
    lca_version_name: str | None = None,
) -> ASRDeterministicPathContext:
    """Build one canonical deterministic ASR path context."""
    return ASRDeterministicPathContext(
        proj_base=Path(proj_base),
        source_label=str(source_label).strip(),
        group_version=group_version,
        fu_code=str(fu_code).strip(),
        lca_type=str(lca_type).strip().lower(),
        cc_source=str(cc_source).strip(),
        cc_type=str(cc_type).strip().lower(),
        lca_version_name=lca_version_name,
    )


def build_asr_scope_label(
    *,
    source_label: str,
    group_version: str | None,
    lca_type: str,
    cc_source: str,
    cc_type: str,
    lca_version_name: str | None = None,
) -> str:
    """Return the deterministic ASR metadata scope label."""
    source_token = asocc_source_version_token(source=source_label, group_version=group_version)
    lca_token = _lca_route_token(lca_type=lca_type, lca_version_name=lca_version_name)
    return f"{source_token}__{lca_token}__{cc_branch_token(cc_source=cc_source, cc_type=cc_type)}"


def get_asr_root(*, proj_base: Path) -> Path:
    """Return the branch-local ASR root."""
    return Path(proj_base) / ASR_ROOT_DIRNAME


def get_asr_route_root(
    *,
    proj_base: Path,
    source_label: str,
    group_version: str | None,
    lca_type: str,
    lca_version_name: str | None = None,
) -> Path:
    """Return the source and numerator-route scoped ASR root."""
    lca_token = _lca_route_token(lca_type=lca_type, lca_version_name=lca_version_name)
    return (
        get_asr_root(proj_base=proj_base)
        / asocc_source_version_token(
            source=source_label,
            group_version=group_version,
        )
        / lca_token
    )


def get_asr_branch_root(*, context: ASRDeterministicPathContext) -> Path:
    """Return the canonical deterministic ASR branch root for logs and figures."""
    return (
        get_asr_route_root(
            proj_base=context.proj_base,
            source_label=context.source_label,
            group_version=context.group_version,
            lca_type=context.lca_type,
            lca_version_name=context.lca_version_name,
        )
        / "deterministic"
        / cc_branch_token(cc_source=context.cc_source, cc_type=context.cc_type)
    )


def _public_surface_root(
    *,
    context: ASRDeterministicPathContext,
    family: str,
    create: bool,
) -> Path:
    """Return one FU specific public surface root."""
    if str(context.fu_code).strip().upper().startswith("L2."):
        root = get_asr_branch_root(context=context) / f"{family}_l2_vs_global"
    else:
        root = get_asr_branch_root(context=context) / family
    return ensure_dir(root) if create else root


def get_asr_results_dir(*, context: ASRDeterministicPathContext) -> Path:
    """Return the deterministic ASR results root for one FU scope."""
    return _public_surface_root(context=context, family="results", create=True)


def get_asr_logs_dir(*, context: ASRDeterministicPathContext) -> Path:
    """Return the deterministic ASR logs directory."""
    return get_asr_branch_root(context=context) / "logs"


def get_asr_meta_path(*, context: ASRDeterministicPathContext) -> Path:
    """Return the deterministic ASR scope-manifest path."""
    return get_asr_logs_dir(context=context) / SCOPE_MANIFEST_FILENAME


def get_asr_figure_metadata_path(*, context: ASRDeterministicPathContext) -> Path:
    """Return the deterministic ASR figure request metadata path."""
    return get_asr_logs_dir(context=context) / FIGURE_MANIFEST_FILENAME


def get_asr_figures_dir(*, context: ASRDeterministicPathContext) -> Path:
    """Return the deterministic ASR figures root."""
    return _public_surface_root(context=context, family="figures", create=False)


def get_asr_figure_inputs_dir(*, context: ASRDeterministicPathContext) -> Path:
    """Return the deterministic ASR figure input root."""
    return _public_surface_root(context=context, family="figure_inputs", create=True)


def get_asr_dynamic_component_rows_path(
    *,
    context: ASRDeterministicPathContext,
    fmt: str,
) -> Path:
    """Return the dynamic ASR component figure input table path."""
    return get_asr_figure_inputs_dir(context=context) / f"dynamic_component_rows.{fmt}"
