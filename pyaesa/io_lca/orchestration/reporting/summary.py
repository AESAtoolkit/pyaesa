"""Reporting ownership for deterministic IO-LCA orchestration."""

from pathlib import Path
from typing import Any

from pyaesa.shared.runtime.reporting.composite_phase_index import PHASE_A_LCA
from pyaesa.shared.runtime.reporting.labels import (
    figures_available_line,
    labelled_values_line,
    output_files_available_line,
)
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.reporting.reuse_status import public_reuse_status
from pyaesa.shared.runtime.reporting.summary import document, render_summary, section
from pyaesa.shared.runtime.text import extend_user_text_lines

from pyaesa.io_lca.orchestration.figure_generation import render_io_lca_figures
from pyaesa.io_lca.orchestration.pipeline.progress import (
    format_year_ranges_with_count,
)


def generate_io_lca_figures(
    *,
    project_name: str,
    source: str,
    group_reg: bool,
    group_sec: bool,
    group_version: str,
    years: list[int],
    lcia_method: list[str],
    fu_code: str,
    r_f,
    r_c,
    r_p,
    s_p,
    aggreg_indices,
    dpi: int,
    output_format: str,
    resolved_io_scope: tuple[str, dict[str, Any]],
    status: StatusSink | None = None,
) -> list[Path]:
    """Render deterministic IO-LCA figures from persisted main outputs."""
    figure_report = render_io_lca_figures(
        project_name=project_name,
        source=source,
        group_reg=group_reg,
        group_sec=group_sec,
        group_version=group_version,
        years=years,
        lcia_method=lcia_method,
        fu_code=fu_code,
        r_f=r_f,
        r_c=r_c,
        r_p=r_p,
        s_p=s_p,
        aggreg_indices=aggreg_indices,
        dpi=dpi,
        output_format=output_format,
        refresh=False,
        resolved_io_scope=resolved_io_scope,
        status=status,
    )
    if figure_report is None:
        return []
    return sorted({Path(path) for path in figure_report})


def build_io_lca_summary(
    *,
    source: str,
    output_root: Path,
    resolved_years: list[int],
    covered_main_years: set[int],
    covered_origin_years: set[int],
    covered_stage_years: set[int],
    skipped_method_years: dict[str, dict[int, str]],
    aggreg_indices: bool,
    upstream_analysis: bool,
    stage_outputs_enabled: bool,
    reuse_status: str,
    lca_results_dirs: set[Path],
    origin_dirs: set[Path],
    stages_dirs: set[Path],
    figure_paths: list[Path],
    project_name: str | None = None,
    lcia_methods: list[str] | None = None,
    fu_code: str | None = None,
    group_reg: bool | None = None,
    group_sec: bool | None = None,
    group_version: str | None = None,
    main_result_paths: list[Path] | None = None,
    origin_paths: list[Path] | None = None,
    stage_paths: list[Path] | None = None,
) -> list[str]:
    """Build the final user facing summary block for one IO-LCA run."""
    function_lines = [
        f"Source: {source}",
        "MRIO scope: "
        + _format_mrio_scope(
            group_reg=group_reg,
            group_sec=group_sec,
            group_version=group_version,
            aggreg_indices=aggreg_indices,
        ),
        f"Processed (main): {format_year_ranges_with_count(sorted(covered_main_years))}",
    ]
    if upstream_analysis:
        function_lines.append(
            "Processed upstream origin rows: "
            f"{format_year_ranges_with_count(sorted(covered_origin_years))}"
        )
    if stage_outputs_enabled:
        function_lines.append(
            "Processed upstream stage rows: "
            f"{format_year_ranges_with_count(sorted(covered_stage_years))}"
        )
    skipped_years = sorted(
        {int(year) for skipped_by_year in skipped_method_years.values() for year in skipped_by_year}
    )
    if skipped_years:
        function_lines.append(
            f"Skipped unavailable years: {format_year_ranges_with_count(skipped_years)}"
        )
        for lcia_method_mode in sorted(skipped_method_years):
            skipped_by_year = skipped_method_years.get(lcia_method_mode, {})
            lcia_method, mode_name = lcia_method_mode.rsplit("__", 1)
            lcia_method_text = (
                f"{lcia_method} [{mode_name}]" if (aggreg_indices and mode_name) else lcia_method
            )
            for year in sorted(skipped_by_year):
                extend_user_text_lines(
                    function_lines,
                    "Skipped year detail: "
                    f"year={int(year)}, lcia_method={lcia_method_text}, "
                    f"reason={skipped_by_year[year]}",
                )
    function_lines.append(f"Output folder: {output_root}")
    output_file_count = _public_output_file_count(
        main_result_paths=main_result_paths or [],
        origin_paths=origin_paths or [],
        stage_paths=stage_paths or [],
        figure_paths=figure_paths,
        output_root=output_root,
    )
    if output_file_count:
        function_lines.append(output_files_available_line(output_file_count))
    function_lines.extend(
        inventory_lines(
            [
                *(
                    [inventory_item(folder="results", content="main LCA tables")]
                    if lca_results_dirs
                    else []
                ),
                *(
                    [inventory_item(folder="results", content="upstream origin tables")]
                    if upstream_analysis and origin_dirs
                    else []
                ),
                *(
                    [inventory_item(folder="results", content="upstream stage tables")]
                    if stage_outputs_enabled and stages_dirs
                    else []
                ),
                inventory_item(folder="logs", content="summary log"),
            ]
        )
    )
    if figure_paths:
        function_lines.append(figures_available_line(len(figure_paths)))
    common_lines = [f"Run status: {public_reuse_status(reuse_status)}"]
    if project_name is not None:
        common_lines.append(f"Project: {project_name}")
    common_lines.append(
        labelled_values_line(
            "Studied year",
            "Studied years",
            tuple(resolved_years),
            format_year_ranges_with_count(resolved_years),
        )
    )
    if lcia_methods:
        common_lines.append(
            labelled_values_line(
                "LCIA method",
                "LCIA methods",
                tuple(lcia_methods),
                ", ".join(lcia_methods),
            )
        )
    if fu_code is not None:
        common_lines.append(f"Functional unit: {fu_code}")
    summary = render_summary(
        document(
            "deterministic_io_lca",
            lines=common_lines,
            sections=(
                section(
                    PHASE_A_LCA,
                    children=(section("deterministic_io_lca", lines=function_lines),),
                ),
            ),
        )
    )
    return summary.splitlines()


def _format_mrio_scope(
    *,
    group_reg: bool | None,
    group_sec: bool | None,
    group_version: str | None,
    aggreg_indices: bool,
) -> str:
    parts = [
        f"group_reg={bool(group_reg)}",
        f"group_sec={bool(group_sec)}",
        f"group_version={group_version or 'none'}",
        f"aggreg_indices={bool(aggreg_indices)}",
    ]
    return ", ".join(parts)


def _public_output_file_count(
    *,
    main_result_paths: list[Path],
    origin_paths: list[Path],
    stage_paths: list[Path],
    figure_paths: list[Path],
    output_root: Path,
) -> int:
    paths = {
        str(path)
        for path in [*main_result_paths, *origin_paths, *stage_paths, *figure_paths]
        if str(path).strip()
    }
    summary_log = output_root / "logs" / "summary.log"
    if summary_log.exists() and summary_log.stat().st_size > 0:
        paths.add(str(summary_log))
    return len(paths)
