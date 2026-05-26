"""Per-method IO-LCA execution runner."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec
from pyaesa.io_lca.compute.main_results import build_main_results_rows
from pyaesa.io_lca.compute.upstream_origin import finalize_origin_rows
from pyaesa.io_lca.compute.upstream_stages import compute_upstream_rows
from pyaesa.io_lca.data.loaders import load_main_payload, load_upstream_payload
from pyaesa.io_lca.data.metadata import get_lcia_method_years
from pyaesa.io_lca.data.paths import (
    IOLCAPaths,
    main_results_path,
    origin_ratio_results_path,
    origin_results_path,
    stage_results_path,
)
from pyaesa.io_lca.data.writers import read_table, write_table

from pyaesa.io_lca.orchestration.io.method_support import (
    aggregate_main,
    aggregate_origin,
    aggregate_stage,
    pending_stage_years,
    selector_combos,
    to_origin_ratio_wide,
)
from pyaesa.io_lca.orchestration.io.method_writes import (
    write_main_year,
    write_origin_year,
    write_stage_year,
)
from pyaesa.io_lca.orchestration.pipeline.progress import year_progress
from pyaesa.io_lca.orchestration.pipeline.run_signatures import table_extension_for_output


@dataclass(frozen=True)
class IOLCAMethodResult:
    """Per method run results used to update run metadata and reports."""

    main_paths: list[Path]
    origin_paths: list[Path]
    stage_paths: list[Path]
    done_main_years: list[int]
    done_origin_years: list[int]
    done_stage_years: list[int]
    skipped_years: dict[int, str]


def run_io_lca_method(
    *,
    lcia_method: str,
    source: str,
    agg_version: str | None,
    agg_reg: bool,
    agg_sec: bool,
    spec: IOLCAFUSpec,
    filters: dict[str, list[str] | None],
    metadata: dict,
    domain_metadata_path: Path,
    paths: IOLCAPaths,
    scope: dict,
    resolved_years: list[int],
    upstream_analysis: bool,
    upstream_stages: int,
    group_indices: bool,
    output_format: str,
    refresh: bool,
    method_progress=None,
) -> IOLCAMethodResult:
    """Run one LCIA method for IO-LCA main/origin/stage outputs."""
    extension = table_extension_for_output(output_format)
    stage_outputs_enabled = upstream_analysis and spec.fu_code != "L1.b"
    main_path = main_results_path(
        paths=paths,
        source=source,
        lcia_method=lcia_method,
        extension=extension,
    )
    origin_path = origin_results_path(
        paths=paths,
        source=source,
        lcia_method=lcia_method,
        extension=extension,
    )
    existing_main = (
        []
        if refresh
        else get_lcia_method_years(scope=scope, section="main", lcia_method=lcia_method)
    )
    existing_origin = (
        []
        if refresh
        else get_lcia_method_years(scope=scope, section="origin", lcia_method=lcia_method)
    )
    existing_stage = (
        []
        if (refresh or not stage_outputs_enabled)
        else get_lcia_method_years(scope=scope, section="stages", lcia_method=lcia_method)
    )

    pending_main = sorted(set(resolved_years) - set(existing_main))
    pending_origin = sorted(set(resolved_years) - set(existing_origin)) if upstream_analysis else []
    pending_stage = (
        pending_stage_years(
            years=resolved_years,
            existing_years=existing_stage,
            paths=paths,
            source=source,
            lcia_method=lcia_method,
            extension=extension,
        )
        if stage_outputs_enabled
        else []
    )
    pending_stage = [year for year in pending_stage if year in pending_origin]
    needed_years = sorted(set(pending_main) | set(pending_origin) | set(pending_stage))

    written_main: list[Path] = []
    written_origin: list[Path] = []
    written_stage: list[Path] = []
    done_main = set(existing_main)
    done_origin = set(existing_origin)
    done_stage = set(existing_stage)
    skipped_years: dict[int, str] = {}
    main_selector_axes = spec.selector_axes
    downstream_selector_axes = tuple() if group_indices else spec.selector_axes
    owns_progress = method_progress is None
    if method_progress is None:
        method_progress = year_progress(
            source="deterministic_io_lca",
            action=f"processing[{lcia_method}]",
            total=len(needed_years),
        )
    else:
        method_progress.action = f"processing[{lcia_method}]"
        method_progress.total = int(method_progress.total) + int(len(needed_years))
    try:
        for year in needed_years:
            method_progress.begin_year(year)
            payload, unavailable_reason = load_main_payload(
                source=source,
                agg_version=agg_version,
                agg_reg=agg_reg,
                agg_sec=agg_sec,
                metadata=metadata,
                metadata_path=domain_metadata_path,
                year=year,
                lcia_method=lcia_method,
                fu_spec=spec,
            )
            if payload is None:
                skipped_years[int(year)] = str(unavailable_reason)
                method_progress.complete_year(year)
                continue

            year_main_rows = pd.DataFrame()
            year_origin_rows = pd.DataFrame()
            year_stage_rows = pd.DataFrame()

            if year in pending_main:
                year_main_rows = build_main_results_rows(
                    payload=payload,
                    spec=spec,
                    filters=filters,
                )
                if group_indices and not year_main_rows.empty:
                    year_main_rows = aggregate_main(
                        year_main_rows,
                        selector_axes=main_selector_axes,
                    )
                done_main.add(int(year))

            need_upstream = upstream_analysis and year in pending_origin
            if need_upstream:
                upstream_payload = load_upstream_payload(
                    source=source,
                    saved_dir=payload.saved_dir,
                    lcia_method=lcia_method,
                    fu_spec=spec,
                )
                combos = selector_combos(
                    payload=payload,
                    spec=spec,
                    lcia_method=lcia_method,
                    filters=filters,
                )
                stage_rows, origin_rows_for_year = compute_upstream_rows(
                    year=year,
                    spec=spec,
                    combos=combos,
                    payload=upstream_payload,
                    upstream_stages=(upstream_stages if stage_outputs_enabled else 0),
                    unit_by_impact=payload.unit_by_impact,
                    emit_stage_rows=stage_outputs_enabled,
                )
                year_origin_rows = finalize_origin_rows(
                    origin_rows=origin_rows_for_year.reset_index(drop=True),
                    selector_axes=spec.selector_axes,
                )
                if group_indices and not year_origin_rows.empty:
                    year_origin_rows = aggregate_origin(year_origin_rows)
                done_origin.add(int(year))
                if year in pending_stage:
                    year_stage_rows = aggregate_stage(stage_rows) if group_indices else stage_rows
                    done_stage.add(int(year))
            merged_main: pd.DataFrame | None = None
            if year in pending_main and not year_main_rows.empty:
                merged_main = write_main_year(
                    year_main_rows=year_main_rows,
                    paths=paths,
                    source=source,
                    lcia_method=lcia_method,
                    extension=extension,
                    output_format=output_format,
                    effective_selector_axes=main_selector_axes,
                    written_main=written_main,
                )

            if upstream_analysis and year in pending_origin and not year_origin_rows.empty:
                if merged_main is not None:
                    main_for_check = merged_main
                else:
                    main_for_check = read_table(main_path)
                write_origin_year(
                    year_origin_rows=year_origin_rows,
                    main_for_check=main_for_check,
                    lcia_method=lcia_method,
                    paths=paths,
                    source=source,
                    extension=extension,
                    output_format=output_format,
                    effective_selector_axes=downstream_selector_axes,
                    written_origin=written_origin,
                )

            if stage_outputs_enabled and year in pending_stage and not year_stage_rows.empty:
                write_stage_year(
                    year=year,
                    year_stage_rows=year_stage_rows,
                    paths=paths,
                    source=source,
                    lcia_method=lcia_method,
                    extension=extension,
                    output_format=output_format,
                    effective_selector_axes=downstream_selector_axes,
                    written_stage=written_stage,
                )
            method_progress.complete_year(year)
    finally:
        if owns_progress:
            method_progress.finish()

    if main_path.exists() and main_path not in written_main:
        written_main.append(main_path)
    if upstream_analysis and origin_path.exists():
        origin_written_this_run = origin_path in written_origin
        if origin_path not in written_origin:
            written_origin.append(origin_path)
        default_ratio_path = origin_ratio_results_path(
            paths=paths,
            source=source,
            lcia_method=lcia_method,
            extension=extension,
        )
        if origin_written_this_run:
            origin_frame = read_table(origin_path)
            ratio_frame = to_origin_ratio_wide(
                frame=origin_frame,
                selector_axes=downstream_selector_axes,
            )
            write_table(path=default_ratio_path, frame=ratio_frame, output_format=output_format)
        if default_ratio_path.exists() and default_ratio_path not in written_origin:
            written_origin.append(default_ratio_path)
    if stage_outputs_enabled:
        for year in sorted(done_stage):
            out_path = stage_results_path(
                paths=paths,
                source=source,
                lcia_method=lcia_method,
                year=year,
                extension=extension,
            )
            if out_path.exists() and out_path not in written_stage:
                written_stage.append(out_path)

    return IOLCAMethodResult(
        main_paths=written_main,
        origin_paths=written_origin,
        stage_paths=written_stage,
        done_main_years=sorted(done_main),
        done_origin_years=sorted(done_origin),
        done_stage_years=sorted(done_stage),
        skipped_years=skipped_years,
    )
