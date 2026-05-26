"""Shared domain level validation checks for IO-LCA entrypoints."""

from pathlib import Path

from pyaesa.asocc.orchestration.setup.formatting.formatting import _process_mrio_hint
from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec


def _years_for_process_hint(years: int | list[int] | range | None) -> list[int]:
    """Return request years suitable for a process_mrio hint."""
    if years is None:
        return []
    if isinstance(years, int):
        return [int(years)]
    return [int(year) for year in years]


def require_aggregated_branch(
    *,
    source: str,
    agg_version: str | None,
    agg_reg: bool,
    agg_sec: bool,
    metadata_path: Path,
    methods: list[str],
    years: int | list[int] | range | None,
) -> None:
    """Fail fast when aggregated MRIO metadata branch is missing."""
    if metadata_path.exists() or not (agg_reg or agg_sec):
        return
    hint = _process_mrio_hint(
        source=source,
        years=_years_for_process_hint(years),
        agg_version=agg_version,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        lcia_methods=methods,
    )
    raise ValueError(
        "Aggregated processed MRIO branch is missing for this source/agg_version. "
        f"Expected metadata at {metadata_path}. Run: {hint}"
    )


def validate_upstream_supported(*, spec: IOLCAFUSpec, upstream_analysis: bool) -> None:
    """Fail when upstream decomposition is requested for unsupported FU families."""
    if not upstream_analysis:
        return
    if spec.upstream_supported:
        return
    raise ValueError(
        f"upstream_analysis=True is not supported for fu_code='{spec.fu_code}' because "
        "it already represents direct PBA (Scope 1) sector-level impacts."
    )


def validate_group_indices_supported(*, spec: IOLCAFUSpec, group_indices: bool) -> None:
    """Reject grouped index outputs for TD L2 FUs, aligned with deterministic_asocc."""
    if not group_indices:
        return
    if spec.fu_code not in {"L2.a.b", "L2.b.b", "L2.c.b"}:
        return
    raise ValueError(
        "group_indices=True is not allowed for L2.a.b/L2.b.b/L2.c.b because "
        "CBA_TD perimeters can introduce double counting when aggregation outputs."
    )


def validate_group_indices_requires_multi_selection(
    *,
    group_indices: bool,
    has_multi_indices: bool,
) -> None:
    """Reject grouped index mode when no multi value selector is provided."""
    if not group_indices:
        return
    if has_multi_indices:
        return
    raise ValueError(
        "group_indices=True requires at least one selector with multiple values "
        "(r_f/r_c/r_p/s_p). For single-index selections, use group_indices=False."
    )
