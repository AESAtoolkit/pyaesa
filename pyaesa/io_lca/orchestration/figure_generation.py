"""Internal figure generation orchestration for deterministic IO-LCA outputs."""

from functools import partial
from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.shared.figures.checkpoints import default_checkpoint_years
from pyaesa.shared.figures.jobs import PlannedFigureJob, render_figure_jobs
from pyaesa.shared.figures.title_contract import selector_scope_request_from_filters
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.reuse.contracts import normalize_selector_payload
from pyaesa.shared.tabular.scalars import sanitize_token
from pyaesa.io_lca.figures.common import normalize_plot_years, selector_groups
from pyaesa.io_lca.contracts.fu_mapping import resolve_fu_spec

from ..data.loaders import load_domain_metadata, load_io_lca_method_table
from ..data.metadata import (
    compatible_scope,
    ensure_scope,
    get_scope,
    load_scope_manifest,
    merge_written_paths,
    require_scope_signature,
    save_scope_manifest,
    scope_complete_and_existing,
    set_figure_paths,
    set_scope_complete,
)
from ..data.paths import (
    figure_metadata_path_for_source,
    figures_dir_for_source,
    main_results_path,
    resolve_io_lca_paths,
)
from ..plot.figure_writers import (
    write_lcia_method_checkpoint_figures,
    write_lcia_method_figures,
)
from .figure_scope import (
    clear_existing_io_lca_figure_scope,
)
from pyaesa.io_lca.orchestration.request.domain_checks import (
    require_grouped_branch,
    validate_aggreg_indices_requires_multi_selection,
    validate_aggreg_indices_supported,
)
from .figure_support import (
    done_and_skipped_lcia_years,
    require_main_result_columns,
    validate_lcia_method_coverage,
)
from pyaesa.io_lca.orchestration.pipeline.run_signatures import (
    build_io_lca_figure_signature,
    table_extension_for_output,
)
from pyaesa.io_lca.orchestration.request.selectors import (
    has_multi_selected_indices,
    resolve_selectors,
    validate_selector_labels,
)
from pyaesa.io_lca.orchestration.request.validation import (
    normalize_aggreg_indices_modes,
    normalize_figure_output_format,
    normalize_grouping,
    normalize_lcia_method_list,
    normalize_supported_source,
    validate_dpi,
)
from pyaesa.io_lca.orchestration.request.year_resolution import resolve_years_strict


def render_io_lca_figures(
    *,
    project_name: str,
    source: str,
    group_reg: bool = False,
    group_sec: bool = False,
    group_version: str = "",
    years: int | list[int] | range | None = None,
    lcia_method: str | list[str],
    fu_code: str,
    r_f: str | list[str] | None = None,
    r_c: str | list[str] | None = None,
    r_p: str | list[str] | None = None,
    s_p: str | list[str] | None = None,
    aggreg_indices: bool = False,
    dpi: int = 500,
    output_format: str = "png",
    refresh: bool = False,
    resolved_io_scope: tuple[str, dict[str, Any]],
    status: StatusSink | None = None,
) -> list[Path] | None:
    """Generate deterministic environmental burden figures from existing IO-LCA outputs."""
    source_norm = normalize_supported_source(
        source=source,
        caller="deterministic_io_lca figure generation",
    )
    group_reg_norm, group_sec_norm, group_version_norm = normalize_grouping(
        group_reg=group_reg,
        group_sec=group_sec,
        group_version=group_version,
    )
    methods = normalize_lcia_method_list(lcia_method=lcia_method)
    spec = resolve_fu_spec(fu_code=fu_code)
    aggreg_indices = normalize_aggreg_indices_modes(aggreg_indices)[0]
    dpi_norm = validate_dpi(dpi)
    figure_output_norm = normalize_figure_output_format(output_format)

    filters, _studied_indices_tag = resolve_selectors(
        spec=spec,
        r_f=r_f,
        r_c=r_c,
        r_p=r_p,
        s_p=s_p,
    )
    has_multi_indices = has_multi_selected_indices(filters)
    validate_selector_labels(
        source=source_norm,
        group_version=group_version_norm,
        group_reg=group_reg_norm,
        group_sec=group_sec_norm,
        filters=filters,
    )
    _metadata_preview, metadata_path = load_domain_metadata(
        source=source_norm,
        group_version=group_version_norm,
    )
    require_grouped_branch(
        source=source_norm,
        group_version=group_version_norm,
        group_reg=group_reg_norm,
        group_sec=group_sec_norm,
        metadata_path=metadata_path,
        methods=methods,
        years=years,
    )
    resolved_years = resolve_years_strict(
        years=years,
        source=source_norm,
        group_version=group_version_norm,
        group_reg=group_reg_norm,
        group_sec=group_sec_norm,
        upstream_analysis=False,
    )
    validate_aggreg_indices_requires_multi_selection(
        aggreg_indices=aggreg_indices,
        has_multi_indices=has_multi_indices,
    )
    validate_aggreg_indices_supported(spec=spec, aggreg_indices=aggreg_indices)
    paths = resolve_io_lca_paths(
        project_name=project_name,
        group_reg=group_reg_norm,
        group_sec=group_sec_norm,
        group_version=group_version_norm,
    )
    source_figures_dir = figures_dir_for_source(
        paths=paths,
        source=source_norm,
    )
    figure_metadata_path = figure_metadata_path_for_source(paths=paths, source=source_norm)
    io_output_format, io_scope = resolved_io_scope
    figure_payload = load_scope_manifest(
        path=figure_metadata_path,
        function_name="deterministic_io_lca_figures",
    )
    selector_scope_request = selector_scope_request_from_filters(
        filters=filters,
        selector_columns=spec.selector_axes,
    )
    figure_signature = build_io_lca_figure_signature(
        project_name=project_name,
        source=source_norm,
        group_reg=group_reg_norm,
        group_sec=group_sec_norm,
        group_version=group_version_norm,
        years=resolved_years,
        methods=methods,
        fu_code=spec.fu_code,
        filters=filters,
        aggreg_indices=aggreg_indices,
        dpi=dpi_norm,
        output_format=figure_output_norm,
        io_output_format=io_output_format,
    )
    scope_key, scope_existing = get_scope(payload=figure_payload, signature=figure_signature)
    if scope_existing is None and figure_payload.get("arguments") is not None:
        stored_figure_signature = require_scope_signature(scope=figure_payload)
        if (
            stored_figure_signature.get("output_format") == figure_output_norm
            and stored_figure_signature.get("dpi") == dpi_norm
            and stored_figure_signature.get("io_output_format") == io_output_format
            and compatible_scope(
                payload={
                    **figure_payload,
                    "arguments": {**stored_figure_signature, "output_format": io_output_format},
                },
                project_name=project_name,
                source=source_norm,
                group_reg=group_reg_norm,
                group_sec=group_sec_norm,
                group_version=group_version_norm,
                fu_code=spec.fu_code,
                aggreg_indices=aggreg_indices,
                output_format=io_output_format,
                requested_years={int(year) for year in resolved_years},
                requested_methods=set(methods),
                requested_selectors=normalize_selector_payload(filters),
            )
            is not None
        ):
            scope_existing = figure_payload
    if not refresh and scope_existing is not None and scope_complete_and_existing(scope_existing):
        return sorted({Path(str(path)) for path in scope_existing["paths_written"]})
    clear_existing_io_lca_figure_scope(payload=figure_payload)
    scope = ensure_scope(
        payload=figure_payload,
        key=scope_key,
        signature=figure_signature,
        function_name="deterministic_io_lca_figures",
    )
    scope["paths_written"] = []
    scope["complete"] = False
    scope["status"]["figures"] = {}

    table_extension = table_extension_for_output(io_output_format)
    figure_paths: list[Path] = []
    checkpoint_years = default_checkpoint_years(resolved_years)
    exact_single_year_scope = len({int(year) for year in resolved_years}) == 1
    lcia_method_frames: dict[str, pd.DataFrame] = {}
    for lcia_method in methods:
        validate_lcia_method_coverage(
            io_scope=io_scope,
            lcia_method=lcia_method,
            years=resolved_years,
        )
        done_years, _skipped_years = done_and_skipped_lcia_years(
            scope=io_scope,
            lcia_method=lcia_method,
        )
        if not ({int(year) for year in resolved_years} & set(done_years)):
            lcia_method_frames[lcia_method] = pd.DataFrame()
            continue
        main_path = main_results_path(
            paths=paths,
            source=source_norm,
            lcia_method=lcia_method,
            extension=table_extension,
        )
        lcia_method_frame = cast(pd.DataFrame, load_io_lca_method_table(path=main_path))
        require_main_result_columns(
            frame=lcia_method_frame,
            lcia_method=lcia_method,
            selector_axes=tuple() if aggreg_indices else spec.selector_axes,
        )
        lcia_method_frame = normalize_plot_years(frame=lcia_method_frame)
        lcia_method_frame = cast(
            pd.DataFrame,
            lcia_method_frame.loc[
                lcia_method_frame["year"].astype(int).isin(resolved_years)
            ].copy(),
        )
        lcia_method_frames[lcia_method] = lcia_method_frame
    jobs: list[PlannedFigureJob] = []
    for lcia_method in methods:
        frame = lcia_method_frames[lcia_method]
        if frame.empty:
            continue
        _selector_cols, groups = selector_groups(frame=frame, selector_columns=None)
        if exact_single_year_scope:
            for checkpoint_year in checkpoint_years:
                for _group_key, group_frame in groups:
                    jobs.append(
                        PlannedFigureJob(
                            kind="checkpoint",
                            label=f"{lcia_method} {int(checkpoint_year)}",
                            render=partial(
                                write_lcia_method_checkpoint_figures,
                                lcia_method_frame=group_frame,
                                reference_frame=frame,
                                figures_dir=source_figures_dir,
                                lcia_method=lcia_method,
                                checkpoint_years=[int(checkpoint_year)],
                                dpi=dpi_norm,
                                output_format=figure_output_norm,
                                selector_scope_request=selector_scope_request,
                            ),
                        )
                    )
            continue
        for _group_key, group_frame in groups:
            jobs.append(
                PlannedFigureJob(
                    kind="multi_year",
                    label=f"{lcia_method} multi-year",
                    render=partial(
                        write_lcia_method_figures,
                        lcia_method_frame=group_frame,
                        reference_frame=frame,
                        figures_dir=source_figures_dir,
                        lcia_method=lcia_method,
                        dpi=dpi_norm,
                        output_format=figure_output_norm,
                        selector_scope_request=selector_scope_request,
                    ),
                )
            )
    rendered = render_figure_jobs(source="deterministic_io_lca", jobs=jobs, status=status)
    for lcia_method in methods:
        method_token = sanitize_token(lcia_method)
        lcia_method_paths = [path for path in rendered if path.name.startswith(f"{method_token}__")]
        set_figure_paths(
            scope=scope,
            lcia_method=lcia_method,
            figure_paths=lcia_method_paths,
        )
        figure_paths.extend(lcia_method_paths)
    merge_written_paths(scope=scope, paths=figure_paths)
    set_scope_complete(scope=scope, complete=bool(figure_paths))
    save_scope_manifest(path=figure_metadata_path, payload=figure_payload)
    if not figure_paths:
        return None
    return sorted({Path(path) for path in figure_paths})
