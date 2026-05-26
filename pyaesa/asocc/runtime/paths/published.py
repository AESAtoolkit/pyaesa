"""Family-local deterministic aSoCC published output and figure paths."""

from pathlib import Path

from pyaesa.asocc.runtime.paths.family_roots import (
    _get_asocc_root,
    asocc_source_version_token,
)


def _asocc_deterministic_scope_root(
    *,
    proj_base: Path,
    source: str,
    agg_version: str | None,
) -> Path:
    """Return the deterministic aSoCC scope root for one explicit branch."""
    return (
        _get_asocc_root(proj_base=proj_base)
        / asocc_source_version_token(source=source, agg_version=agg_version)
        / "deterministic"
    )


def _owning_fu_level_for_code(*, fu_code: str | None) -> str:
    """Return the canonical owning deterministic FU level for one request code."""
    return "level_2" if str(fu_code).strip().startswith("L2.") else "level_1"


def _normalize_owning_fu_level(*, owning_fu_level: str) -> str:
    """Return one validated deterministic owning FU level token."""
    return str(owning_fu_level).strip()


def _get_asocc_results_root(
    *,
    proj_base: Path,
    source: str,
    agg_version: str | None,
    owning_fu_level: str = "level_1",
) -> Path:
    """Return the deterministic L1 public results root for one branch."""
    level_token = _normalize_owning_fu_level(owning_fu_level=owning_fu_level)
    base = (
        _asocc_deterministic_scope_root(
            proj_base=proj_base,
            source=source,
            agg_version=agg_version,
        )
        / "results"
    )
    return base if level_token == "level_1" else base / "level_1"


def _canonical_l2_results_relative_dir(*, bucket: str) -> Path:
    """Return the canonical deterministic L2 results relative directory."""
    bucket_clean = str(bucket).strip() or "l2_vs_global"
    relative_dirs = {
        "l2_vs_global": Path("results") / "level_2" / "l2_vs_global",
        "l2_in_l1": Path("results") / "level_2" / "l2_in_l1",
        "utility_propagation_contrib": Path("results") / "level_2" / "utility_propagation_contrib",
    }
    return relative_dirs[bucket_clean]


def _get_asocc_l2_results_root(
    *,
    proj_base: Path,
    source: str,
    agg_version: str | None,
    bucket: str,
) -> Path:
    """Return the deterministic L2 route root for one bucket."""
    return _asocc_deterministic_scope_root(
        proj_base=proj_base,
        source=source,
        agg_version=agg_version,
    ) / _canonical_l2_results_relative_dir(bucket=bucket)


def _get_asocc_l1_dir(
    *,
    proj_base: Path,
    source: str,
    agg_version: str | None,
    lcia_sub: str | None,
    owning_fu_level: str = "level_1",
) -> Path:
    """Return the deterministic L1 public results directory."""
    base = _get_asocc_results_root(
        proj_base=proj_base,
        source=source,
        agg_version=agg_version,
        owning_fu_level=owning_fu_level,
    )
    return base / lcia_sub if lcia_sub else base


def _get_asocc_l2_dir(
    *,
    proj_base: Path,
    source: str,
    agg_version: str | None,
    bucket: str,
    lcia_sub: str | None,
) -> Path:
    """Return the deterministic L2 public route directory."""
    base = _get_asocc_l2_results_root(
        proj_base=proj_base,
        source=source,
        agg_version=agg_version,
        bucket=bucket,
    )
    return base / lcia_sub if lcia_sub else base


def _get_enacting_metric_dir(
    *,
    proj_base: Path,
    source: str,
    agg_version: str | None,
    level: str,
    lcia_sub: str | None = None,
    owning_fu_level: str = "level_1",
) -> Path:
    """Return the deterministic enacting metrics directory for one scope."""
    if level == "level_1":
        base = _get_asocc_results_root(
            proj_base=proj_base,
            source=source,
            agg_version=agg_version,
            owning_fu_level=owning_fu_level,
        )
    else:
        base = (
            _asocc_deterministic_scope_root(
                proj_base=proj_base,
                source=source,
                agg_version=agg_version,
            )
            / "results"
            / "level_2"
        )
    out = base / "enacting_metrics"
    return out / lcia_sub if lcia_sub else out


def _get_enacting_metric_output_path(
    *,
    proj_base: Path,
    source: str,
    agg_version: str | None,
    level: str,
    key_metric: str,
    key_method: str | None,
    key_scenario: str | None,
    output_format: str,
    lcia_sub: str | None = None,
    owning_fu_level: str = "level_1",
) -> Path:
    """Return output path for one enacting metric artifact."""
    from pyaesa.asocc.runtime.paths.deterministic import suffix_for_output_format

    out_base = _get_enacting_metric_dir(
        proj_base=proj_base,
        source=source,
        agg_version=agg_version,
        level=level,
        lcia_sub=lcia_sub,
        owning_fu_level=owning_fu_level,
    )
    stem = str(key_metric).replace("_capita", "_cap")
    if key_scenario is not None:
        stem = f"{stem}_{key_scenario}"
    if key_method:
        stem = f"{stem}_{key_method}"
    return out_base / f"{stem}{suffix_for_output_format(output_format=output_format)}"


def _get_asocc_figures_root(
    *,
    proj_base: Path,
    level: str,
    source: str,
    agg_version: str | None,
) -> Path:
    """Return the deterministic aSoCC figures root for one FU scope."""
    if level == "level_1":
        dirname = "figures"
    else:
        dirname = "figures_l2_vs_global"
    return (
        _asocc_deterministic_scope_root(
            proj_base=proj_base,
            source=source,
            agg_version=agg_version,
        )
        / dirname
    )


def reuse_output_path_for(
    *,
    context,
    bucket: str,
    file_stem: str,
) -> Path:
    """Return one persisted reuse source output path for a bucket and stem."""
    from pyaesa.asocc.runtime.paths.deterministic import suffix_for_output_format

    base = _get_asocc_l2_dir(
        proj_base=context.proj_base,
        source=context.source,
        agg_version=context.agg_version,
        bucket=bucket,
        lcia_sub=None,
    )
    return base / f"{file_stem}{suffix_for_output_format(output_format=context.output_format)}"
