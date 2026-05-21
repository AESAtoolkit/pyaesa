"""Run signature builders shared by IO-LCA computation and figure generation."""

from typing import Any


def build_io_lca_signature(
    *,
    project_name: str,
    source: str,
    group_reg: bool,
    group_sec: bool,
    group_version: str | None,
    years: list[int],
    methods: list[str],
    fu_code: str,
    filters: dict[str, list[str] | None],
    upstream_analysis: bool,
    upstream_stages: int,
    aggreg_indices: bool,
    output_format: str,
) -> dict[str, Any]:
    """Build deterministic signature payload for one ``deterministic_io_lca`` run scope."""
    return {
        "project_name": project_name,
        "source": source,
        "group_reg": bool(group_reg),
        "group_sec": bool(group_sec),
        "group_version": group_version,
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
        "aggreg_indices": bool(aggreg_indices),
        "output_format": output_format,
    }


def build_io_lca_figure_signature(
    *,
    project_name: str,
    source: str,
    group_reg: bool,
    group_sec: bool,
    group_version: str | None,
    years: list[int],
    methods: list[str],
    fu_code: str,
    filters: dict[str, list[str] | None],
    aggreg_indices: bool,
    dpi: int,
    output_format: str,
    io_output_format: str,
) -> dict[str, Any]:
    """Build deterministic signature payload for one IO-LCA figure run scope."""
    return {
        "project_name": project_name,
        "source": source,
        "group_reg": bool(group_reg),
        "group_sec": bool(group_sec),
        "group_version": group_version,
        "years": years,
        "lcia_methods": methods,
        "fu_code": fu_code,
        "selectors": {
            "r_f": filters.get("r_f"),
            "r_c": filters.get("r_c"),
            "r_p": filters.get("r_p"),
            "s_p": filters.get("s_p"),
        },
        "aggreg_indices": bool(aggreg_indices),
        "dpi": int(dpi),
        "output_format": output_format,
        "io_output_format": io_output_format,
    }


def table_extension_for_output(output_format: str) -> str:
    """Map table output format to deterministic file extension."""
    return {"csv": "csv", "pickle": "pickle", "parquet": "parquet"}[output_format]
