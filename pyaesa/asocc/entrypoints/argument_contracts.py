"""Shared public entrypoint argument contracts for allocation methods.

This module owns the validation and normalization rules that apply before
execution enters either deterministic aSoCC computation or deterministic aSoCC
disaggregation. It owns only argument level contracts that are shared by more
than one public entrypoint, namely:

- deterministic output format normalization
- normalization of selector arguments that accept one string or many strings
- admissibility checks for grouped output requests
- grouped output eligibility checks

It does not resolve MRIO inputs, method plans, output paths, or runtime state.
Those responsibilities stay in the orchestration and runtime packages.
"""

from pyaesa.shared.tabular.contracts import normalize_tabular_output_format


def normalize_allocate_output_format(output_format: str) -> str:
    """Validate and normalize deterministic allocation output format."""
    return normalize_tabular_output_format(output_format)


def ensure_list_str(value: str | list[str] | None) -> list[str] | None:
    """Normalize API string or list filters to list form."""
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return value


def validate_grouped_request(
    *,
    fu_norm: str,
    grouped_requested: bool,
    r_p: list[str] | None,
    s_p: list[str] | None,
    r_c: list[str] | None,
    r_f: list[str] | None,
) -> None:
    """Validate grouped output eligibility for provided filters."""
    if not grouped_requested:
        return
    multi_region = any(
        region_list is not None and len(set(region_list)) >= 2 for region_list in (r_p, r_c, r_f)
    )
    multi_sector = s_p is not None and len(set(s_p)) >= 2
    if fu_norm.startswith("L1."):
        if not multi_region:
            raise ValueError(
                "For L1 functional units, aggreg_indices=True requires a "
                "multi-region filter (at least two values in r_p/r_c/r_f)."
            )
        return
    if not fu_norm.startswith("L2."):
        return
    if not (multi_region or multi_sector):
        raise ValueError(
            "For L2 functional units, aggreg_indices=True requires either "
            "a multi-region filter (r_p/r_c/r_f) or multi-sector filter (s_p)."
        )
