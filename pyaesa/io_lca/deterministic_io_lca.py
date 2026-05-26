"""Public IO-LCA entrypoint based on processed MRIO outputs."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from pyaesa.io_lca.contracts.fu_mapping import resolve_fu_spec
from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
)
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_A_LCA,
    phase_ready_detail,
    phase_reused_detail,
)
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.reporting.summary_log import summary_log_path, write_summary_log

from .contracts.runtime_types import IOLCAReport
from .data.loaders import load_domain_metadata
from .data.paths import (
    lca_results_dir_for_source,
    log_dir_for_source,
    origin_dir_for_source,
    resolve_io_lca_paths,
    stages_dir_for_source,
)
from .orchestration.pipeline.mode_runner import execute_io_lca_mode
from .orchestration.reporting.summary import build_io_lca_summary, generate_io_lca_figures
from .orchestration.request.domain_checks import (
    require_aggregated_branch,
    validate_upstream_supported,
)
from .orchestration.request.selectors import (
    has_multi_selected_indices,
    resolve_selectors,
    validate_selector_labels,
)
from .orchestration.request.validation import (
    normalize_group_indices_modes,
    normalize_aggregation,
    normalize_io_output_format,
    normalize_lcia_method_list,
    normalize_supported_source,
    validate_upstream_stages,
)
from .orchestration.request.year_resolution import resolve_years_strict


def deterministic_io_lca(
    *,
    project_name: str,
    source: str,
    agg_reg: bool = False,
    agg_sec: bool = False,
    agg_version: str = "",
    years: int | list[int] | range | None = None,
    lcia_method: str | list[str],
    fu_code: str,
    s_p: str | list[str] | None = None,
    r_p: str | list[str] | None = None,
    r_c: str | list[str] | None = None,
    r_f: str | list[str] | None = None,
    upstream_analysis: bool = False,
    upstream_stages: int = 3,
    group_indices: bool = False,
    output_format: str = "csv",
    figures: bool = True,
    figure_format: dict = {"format": "png", "dpi": 500},
    refresh: bool = False,
    _status: StatusSink | None = None,
) -> IOLCAReport:
    """Compute deterministic IO-LCA outputs from processed MRIO tables.

    This function reads processed MRIO outputs produced by
    ``process_mrio(...)`` and writes deterministic IO-LCA result tables under
    the selected ``<project_name>``.
    Optional upstream supply chain decomposition outputs are written only when
    ``upstream_analysis=True``. It renders figures when requested.
    Omit arguments to use their default.

    Args:
        project_name: Required project name used to build
            ``<repo>/<project_name>``.
        source: MRIO source key (``"exiobase_396_ixi"``,
            ``"exiobase_396_pxp"``, ``"exiobase_3102_ixi"``,
            ``"exiobase_3102_pxp"``). pyaesa currently only supports
            EXIOBASE for LCIA characterization.
        agg_reg: If ``True``, reclassify MRIO regions with the
            ``agg_reg_<agg_version>.csv`` MRIO aggregation and disaggregation mapping.
            The mapping can keep native labels, aggregate several native regions
            into one target label, or disaggregate one native region across several
            target labels when a ``weight`` column is provided.
            Default ``False`` keeps native source regions.
        agg_sec: If ``True``, reclassify MRIO sectors with the
            ``agg_sec_<agg_version>.csv`` MRIO aggregation and disaggregation mapping.
            The mapping can keep native labels, aggregate several native sectors
            into one target label, or disaggregate one native sector across several
            target labels when a ``weight`` column is provided.
            Default ``False`` keeps native source sectors.
        agg_version: Name token used to resolve the matching
            ``agg_reg_<agg_version>.csv`` and/or
            ``agg_sec_<agg_version>.csv`` MRIO aggregation and disaggregation
            mapping files in ``data_raw/mrio/<source>/aggregation``.
            Required when ``agg_reg`` or ``agg_sec`` is True. Defaults to
            an empty string for native source classification. Use the same
            token in downstream calls that should reuse the processed
            classification. When a mapping file has a ``weight``
            column, weights must sum to ``1`` for each original label.
        years: Studied years. Accepts a single year, list, or range. If
            omitted, all available MRIO
            years for the selected source and ``agg_version`` are used.
        lcia_method: Required LCIA method(s) selected for IO-LCA results
            (for example ``"pb_lcia"`` or ``["pb_lcia", "gwp100_lcia"]``).
            The method(s) must have been processed for the same MRIO source
            with ``process_mrio(...)``. pyaesa currently supports IO-LCA only
            for EXIOBASE sources. To add a custom LCIA method with which run
            ``process_mrio(...)``, follow
            ``README_add_custom_lcia_characterization_matrices.txt`` in
            ``data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/``
            and pass the custom method file stem here.
        fu_code: Required functional unit code (for example ``"L1.a"``,
            ``"L2.c.b"``). See
            ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
            for all available functional unit codes and the system
            boundaries each represents.
        s_p: Producing sector filter(s), single string or list. If this is a
            required axis for ``fu_code`` and the argument is omitted, the run
            expands to all valid producing sectors. To identify valid sector
            names, see the first column of the relevant
            ``data_raw/mrio/.../aggregation/.../agg_sec_template.csv`` file. For
            EXIOBASE sector definitions, see
            ``data_raw/mrio/exiobase_3/sector_classification.xlsx``; EXIOBASE
            ixi and pxp use different sector lists.
        r_p: Producing region filter(s), single string or list. If this is a
            required axis for ``fu_code`` and the argument is omitted, the run
            expands to all valid producing regions. To identify valid region
            names, see the first column of the relevant
            ``data_raw/mrio/.../aggregation/agg_reg_template.csv`` file.
        r_c: Consuming region filter(s), single string or list. If this is a
            required axis for ``fu_code`` and the argument is omitted, the run
            expands to all valid consuming regions. To identify valid region
            names, see the first column of the relevant
            ``data_raw/mrio/.../aggregation/agg_reg_template.csv`` file.
        r_f: Final demand region filter(s), single string or list. If this is
            a required axis for ``fu_code`` and the argument is omitted, the
            run expands to all valid final demand regions. To identify valid
            region names, see the first column of the relevant
            ``data_raw/mrio/.../aggregation/agg_reg_template.csv`` file.
        upstream_analysis: Whether upstream diagnostic outputs are written.
            These outputs are for user audit only and are not used by any
            downstream package function. Default is ``False``.

            - ``False``: write only main IO-LCA result tables.
            - ``True``: also write origin tables attributing the footprint to
              producer sector-region pairs where impacts occur, and when
              supported by the FU, stage tables showing each upstream supply
              chain step with direct, embedded, and total impacts.
        upstream_stages: Number of upstream supply chain steps written when
            ``upstream_analysis=True``. Default ``3`` writes ``n`` to
            ``n-3``.
        group_indices: Whether multiple selected region or sector filter values
            are kept as separate result rows or summed into one result row after
            the function calculation has been performed.
            - ``False`` (default): keep selected values as independent rows.
            - ``True``: sum selected values into one result row.
            The function refuses to run when ``group_indices=True`` is used
            with ``L2.a.b``, ``L2.b.b``, or ``L2.c.b`` because summing output
            rows for CBA total demand boundaries can double count. For these
            functional units, change the upstream MRIO aggregation and disaggregation
            scope with ``agg_reg``, ``agg_sec``, and ``agg_version`` before
            running the study.
        output_format: Persisted output file format: ``"csv"`` (default),
            ``"pickle"``, or ``"parquet"``.
        figures: Whether to render figures.
            Default is ``True``.
        figure_format: Figure render settings mapping. Defaults to
            ``{"format": "png", "dpi": 500}``.

            Nested keys:

            - ``format``: Figure file format. Accepted values are ``"png"``,
              ``"pdf"``, and ``"svg"``.
            - ``dpi``: Positive integer figure resolution used for raster
              outputs.
        refresh: If ``True``, clear and recompute the resolved deterministic
            IO-LCA source and version output scope under
            ``<project>/A_lca/io_lca``. For example, for
            ``project_name="demo"``, ``source="exiobase_3102_ixi"``, and
            ``agg_version="elec"``, the refreshed path is
            ``<repo>/demo/A_lca/io_lca/exiobase_3102_ixi__elec/deterministic``.
            It is not limited to one LCIA method inside that output scope.
            Processed MRIO inputs, processed population and GDP, raw downloads,
            and downstream ASR outputs are not refreshed. Defaults to
            ``False``.

    Returns:
        IOLCAReport describing deterministic IO-LCA table outputs and figure
        outputs when figures are requested.

    Raises:
        ValueError: If the source, aggregation scope, years, FU code, selectors,
            LCIA methods, aggregation mode, output format, or upstream settings
            are invalid; if aggregated prerequisites are missing; if processed MRIO
            assets required by the request are unavailable; or if upstream
            analysis is requested outside the FU routes with upstream outputs.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

    Example:
        Compute deterministic IO-LCA for ``L2.c.b`` producing sector
        ``Paper`` and consuming region ``FR``::

            from pyaesa import deterministic_io_lca

            deterministic_io_lca(
                project_name="demo",
                source="exiobase_3102_ixi",
                years=2019,
                lcia_method="gwp100_lcia",
                fu_code="L2.c.b",
                s_p=["Paper"],
                r_c=["FR"],
            )
    """
    source_norm = normalize_supported_source(source=source, caller="deterministic_io_lca")
    agg_reg_norm, agg_sec_norm, agg_version_norm = normalize_aggregation(
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
    )
    output_format_norm = normalize_io_output_format(output_format)
    figure_format_norm = normalize_figure_format(figure_format)
    group_indices = normalize_group_indices_modes(group_indices)[0]
    methods = normalize_lcia_method_list(lcia_method=lcia_method)
    spec = resolve_fu_spec(fu_code=fu_code)
    stages = validate_upstream_stages(upstream_stages)
    scope_paths = resolve_io_lca_paths(
        project_name=project_name,
        agg_reg=agg_reg_norm,
        agg_sec=agg_sec_norm,
        agg_version=agg_version_norm,
    )
    validate_upstream_supported(spec=spec, upstream_analysis=upstream_analysis)
    stage_outputs_enabled = upstream_analysis and spec.fu_code != "L1.b"
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
        agg_version=agg_version_norm,
        agg_reg=agg_reg_norm,
        agg_sec=agg_sec_norm,
        filters=filters,
    )
    metadata, domain_metadata_path = load_domain_metadata(
        source=source_norm,
        agg_version=agg_version_norm,
    )
    require_aggregated_branch(
        source=source_norm,
        agg_version=agg_version_norm,
        agg_reg=agg_reg_norm,
        agg_sec=agg_sec_norm,
        metadata_path=domain_metadata_path,
        methods=methods,
        years=years,
    )
    resolved_years = resolve_years_strict(
        years=years,
        source=source_norm,
        agg_version=agg_version_norm,
        agg_reg=agg_reg_norm,
        agg_sec=agg_sec_norm,
        upstream_analysis=upstream_analysis,
    )

    all_main_paths: list[Path] = []
    all_origin_paths: list[Path] = []
    all_stage_paths: list[Path] = []
    skipped_method_years: dict[str, dict[int, str]] = {}
    metadata_paths: list[Path] = []
    resolved_io_scope: tuple[str, dict[str, Any]] | None = None
    lca_results_dirs: set[Path] = set()
    origin_dirs: set[Path] = set()
    stages_dirs: set[Path] = set()
    covered_main_years: set[int] = set()
    covered_origin_years: set[int] = set()
    covered_stage_years: set[int] = set()

    with _status_scope(_status) as status:
        paths = scope_paths
        lca_results_dirs.add(
            lca_results_dir_for_source(
                paths=paths,
                source=source_norm,
            )
        )
        if upstream_analysis:
            origin_dirs.add(
                origin_dir_for_source(
                    paths=paths,
                    source=source_norm,
                )
            )
        if stage_outputs_enabled:
            stages_dirs.add(
                stages_dir_for_source(
                    paths=paths,
                    source=source_norm,
                )
            )
        mode_result = execute_io_lca_mode(
            project_name=project_name,
            source=source_norm,
            agg_reg=agg_reg_norm,
            agg_sec=agg_sec_norm,
            agg_version=agg_version_norm,
            methods=methods,
            spec=spec,
            filters=filters,
            metadata=metadata,
            domain_metadata_path=domain_metadata_path,
            resolved_years=resolved_years,
            upstream_analysis=upstream_analysis,
            stages=stages,
            stage_outputs_enabled=stage_outputs_enabled,
            group_indices=group_indices,
            output_format=output_format_norm,
            refresh=refresh,
            has_multi_indices=has_multi_indices,
            status=status,
        )
        metadata_paths.append(mode_result.metadata_path)
        resolved_io_scope = (
            str(mode_result.output_format),
            dict(mode_result.scope_payload),
        )
        all_main_paths.extend(mode_result.main_paths)
        all_origin_paths.extend(mode_result.origin_paths)
        all_stage_paths.extend(mode_result.stage_paths)
        covered_main_years.update(mode_result.covered_main_years)
        covered_origin_years.update(mode_result.covered_origin_years)
        covered_stage_years.update(mode_result.covered_stage_years)
        skipped_method_years.update(mode_result.skipped_method_years)
        figure_paths: list[Path]
        if figures:
            figure_paths = generate_io_lca_figures(
                project_name=project_name,
                source=source_norm,
                agg_reg=agg_reg_norm,
                agg_sec=agg_sec_norm,
                agg_version=agg_version_norm or "",
                years=resolved_years,
                lcia_method=methods,
                fu_code=spec.fu_code,
                r_f=r_f,
                r_c=r_c,
                r_p=r_p,
                s_p=s_p,
                group_indices=group_indices,
                dpi=int(figure_format_norm["dpi"]),
                output_format=str(figure_format_norm["format"]),
                resolved_io_scope=resolved_io_scope,
                status=status,
            )
        else:
            figure_paths = []
        if _status is None:
            detail = (
                phase_reused_detail
                if mode_result.reuse_status == "reused_exact"
                else phase_ready_detail
            )
            phase_status = cast(PhasePrinter, status)
            phase_status.complete(
                detail(
                    scope_name="LCA",
                    output_root=log_dir_for_source(paths=scope_paths, source=source_norm).parent,
                ),
                owner="deterministic_io_lca",
            )
    summary_block = build_io_lca_summary(
        source=source_norm,
        output_root=log_dir_for_source(paths=scope_paths, source=source_norm).parent,
        resolved_years=resolved_years,
        covered_main_years=covered_main_years,
        covered_origin_years=covered_origin_years,
        covered_stage_years=covered_stage_years,
        skipped_method_years=skipped_method_years,
        group_indices=group_indices,
        upstream_analysis=upstream_analysis,
        stage_outputs_enabled=stage_outputs_enabled,
        reuse_status=mode_result.reuse_status,
        lca_results_dirs=lca_results_dirs,
        origin_dirs=origin_dirs,
        stages_dirs=stages_dirs,
        figure_paths=figure_paths,
        project_name=project_name,
        lcia_methods=methods,
        fu_code=spec.fu_code,
        agg_reg=agg_reg_norm,
        agg_sec=agg_sec_norm,
        agg_version=agg_version_norm,
        main_result_paths=all_main_paths,
        origin_paths=all_origin_paths,
        stage_paths=all_stage_paths,
    )
    report = IOLCAReport(
        source=source_norm,
        fu_code=spec.fu_code,
        years=resolved_years,
        lcia_methods=methods,
        main_result_paths=sorted({Path(path) for path in all_main_paths}),
        origin_paths=sorted({Path(path) for path in all_origin_paths}),
        stage_paths=sorted({Path(path) for path in all_stage_paths}),
        figure_paths=figure_paths,
        skipped_method_years=skipped_method_years,
        metadata_path=sorted({Path(path) for path in metadata_paths})[0],
        summary_lines=summary_block,
        reuse_status=mode_result.reuse_status,
    )
    write_summary_log(
        path=summary_log_path(logs_dir=log_dir_for_source(paths=scope_paths, source=source_norm)),
        summary=str(report),
    )
    return report


@contextmanager
def _status_scope(status: StatusSink | None) -> Iterator[StatusSink]:
    """Yield a caller supplied status sink or own a direct run status sink."""
    if status is not None:
        yield status
        return
    owned = PhasePrinter("deterministic_io_lca")
    owned.announce(PHASE_A_LCA, "deterministic_io_lca")
    try:
        yield owned
    finally:
        owned.finish()
