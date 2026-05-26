"""Run signature builders shared by IO-LCA computation and figure generation."""

from typing import Any


def build_io_lca_signature(
    *,
    project_name: str,
    source: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
    years: list[int],
    methods: list[str],
    fu_code: str,
    filters: dict[str, list[str] | None],
    upstream_analysis: bool,
    upstream_stages: int,
    group_indices: bool,
    output_format: str,
) -> dict[str, Any]:
    """Build deterministic signature payload for one ``deterministic_io_lca`` run scope."""
    return {
        "project_name": project_name,
        "source": source,
        "agg_reg": bool(agg_reg),
        "agg_sec": bool(agg_sec),
        "agg_version": agg_version,
        "years": years,
        "lcia_methods": methods,
        "fu_code": fu_code,
        "selectors": {
            "r_f": filters.get("r_f"),
            "r_c": filters.get("r_c"),
            "r_p": filters.get("r_p"),
            "s_p": filters.get("s_p"),
        },
        "upstream_analysis": bool(upstream_analysis),
        "upstream_stages": int(upstream_stages),
        "group_indices": bool(group_indices),
        "output_format": output_format,
    }


def build_io_lca_figure_signature(
    *,
    project_name: str,
    source: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: str | None,
    years: list[int],
    methods: list[str],
    fu_code: str,
    filters: dict[str, list[str] | None],
    group_indices: bool,
    dpi: int,
    output_format: str,
    io_output_format: str,
) -> dict[str, Any]:
    """Build deterministic signature payload for one IO-LCA figure run scope."""
    return {
        "project_name": project_name,
        "source": source,
        "agg_reg": bool(agg_reg),
        "agg_sec": bool(agg_sec),
        "agg_version": agg_version,
        "years": years,
        "lcia_methods": methods,
        "fu_code": fu_code,
        "selectors": {
            "r_f": filters.get("r_f"),
            "r_c": filters.get("r_c"),
            "r_p": filters.get("r_p"),
            "s_p": filters.get("s_p"),
        },
        "group_indices": bool(group_indices),
        "dpi": int(dpi),
        "output_format": output_format,
        "io_output_format": io_output_format,
    }


def table_extension_for_output(output_format: str) -> str:
    """Map table output format to deterministic file extension."""
    return {"csv": "csv", "pickle": "pickle", "parquet": "parquet"}[output_format]
