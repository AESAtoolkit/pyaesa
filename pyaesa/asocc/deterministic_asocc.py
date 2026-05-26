"""Allocate shares of carrying capacities (aSoCC) using precomputed MRIO/Pop/GDP inputs."""

from pyaesa.asocc.runtime.scope.branch_resolution import outputs_project_root
from pyaesa.external_inputs.asocc.schema.contracts import normalize_external_method_selector
from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
    normalize_figure_options,
)
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    phase_ready_detail,
    phase_reused_detail,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.selectors.time_selectors import (
    normalize_optional_reg_window_selector,
    normalize_optional_year_selector,
)

from .entrypoints.argument_contracts import (
    ensure_list_str,
    normalize_allocate_output_format,
    validate_grouped_request,
)
from .io.logging import close_loggers_for_scope
from .methods.registry.registry import normalize_fu_code
from .orchestration.run_allocate import _run_allocate_family, _RunCommonInputs
from .orchestration.run_allocate_support import AllocateReport
from .runtime.selection.normalize import normalize_l1_reg_mode, normalize_output_mode
from .runtime.selection.resolve import resolve_method_selection


def deterministic_asocc(
    *,
    project_name: str,
    source: str,
    agg_reg: bool = False,
    agg_sec: bool = False,
    agg_version: str = "",
    years: int | list[int] | range | None = None,
    fu_code: str,
    s_p: str | list[str] | None = None,
    r_p: str | list[str] | None = None,
    r_c: str | list[str] | None = None,
    r_f: str | list[str] | None = None,
    group_indices: bool = False,
    method_plan: str = "default",
    l1_methods: list[str] | None = None,
    one_step_methods: list[str] | None = None,
    two_step_methods: list[str] | None = None,
    l1_l2_pairs: list[str] | None = None,
    l1_reg_aggreg: str = "post",
    lcia_method: str | list[str] | None = None,
    reference_years: int | list[int] | range | None = None,
    ssp_scenario: str | list[str] = [
        "SSP1",
        "SSP2",
        "SSP3",
        "SSP4",
        "SSP5",
    ],
    projection_mode: str = "regression",
    reg_window: list[int] | range | None = None,
    l2_reuse_years: int | list[int] | range | None = None,
    output_format: str = "csv",
    intermediate_outputs: bool = False,
    figures: bool = True,
    figure_format: dict = {"format": "png", "dpi": 500},
    figure_options: dict[str, bool] = {"per_method": True, "multi_method": True},
    figure_external_method: dict[str, list[str]] | None = None,
    refresh: bool = False,
    _phase: PhasePrinter | NullPhasePrinter | None = None,
) -> AllocateReport:
    """Compute deterministic allocated shares of carrying capacities (aSoCC).

    This function consumes processed MRIO matrices and processed population/GDP
    inputs, computes L1 or L2 allocated shares, and writes deterministic
    outputs under ``<project_name>``. It renders figures when requested.
    Omit arguments to use their default.

    Args:
        project_name: Required project name used to build
            ``<repo>/<project_name>``.
        source: MRIO source key (``"exiobase_396_ixi"``,
            ``"exiobase_396_pxp"``, ``"exiobase_3102_ixi"``,
            ``"exiobase_3102_pxp"``, or ``"oecd_v2025"``), or ``"iso3"``
            for ISO3 only mode (L1 EG/PR(GDPcap) only).
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
        method_plan: ``method_plan`` defaults to ``"default"`` and accepts
            ``"default"``, ``"one_step"``, ``"two_steps"``, ``"pairs"``, or
            ``"one_step_pairs"``. When omitted, all pyaesa allocation methods
            available for the selected ``fu_code`` are applied. See
            ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
            for the allocation methods available per functional unit,
            including definitions and mathematical expressions.
        l1_methods: Optional L1 subset. Omit it to keep all L1 methods allowed
            by ``method_plan``. In ``"default"``, this filters only L1 weights
            used by two step methods. In ``"two_steps"``, it filters the two
            step cartesian L1 side.
        one_step_methods: Optional one step L2 subset. Omit it to keep all one
            step methods allowed by ``method_plan``.
        two_step_methods: Optional two step L2 subset. Omit it to keep all two
            step L2 methods allowed by ``method_plan``.
        l1_l2_pairs: Explicit pair list formatted as ``"L1METHOD::L2METHOD"``.
            Omit it unless ``method_plan`` is ``"pairs"`` or
            ``"one_step_pairs"``.
        l1_reg_aggreg: L1 aggregation mode for methods where timing matters
            (``PR(GDPcap)``, ``PR-HR(Ecap)`` and ``AR(Ecap)``).

            - ``"pre"``: aggregate regions before L1 computation.
            - ``"post"`` (default): compute on original regions, then
              aggregate.
        lcia_method: LCIA method(s) selected for LCIA based allocation
            methods (acquired rights (AR) methods at L1 and L2 and historical
            responsibility (PR-HR) at L1). Options are for example
            ``"pb_lcia"`` or ``["pb_lcia", "gwp100_lcia"]``. ``None`` skips
            LCIA characterization and excludes LCIA based allocation methods.
            Defaults to ``None``. pyaesa currently supports LCIA based
            allocation methods only for EXIOBASE sources. To add a custom
            LCIA method with which run ``process_mrio(...)``, follow
            ``README_add_custom_lcia_characterization_matrices.txt`` in
            ``data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/``
            and pass the custom method file stem here.
        reference_years: Acquired rights (AR) methods reference year selector.
            Accepts a single year, list, or range. If omitted, AR routes use
            all historical years in the studied range up to the source registry
            historical cutoff. For EXIOBASE 3.10.2 and OECD ICIO v2025, the
            cutoff is 2022; other supported MRIO sources use their own
            registry cutoff.
        ssp_scenario: SSP scenario name or list. Defaults to
            ``["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]`` and is applied
            when scenario dependent inputs are required.
        projection_mode: Projection policy for post historical years of L2
            utilitarian (UT) methods (MRIO economic enacting metrics).
            Defaults to ``"regression"``.

            - ``"regression"``: project UT inputs for future years.
            - ``"historical_reuse"``: reuse historical UT structures.
        reg_window: Historical regression fit window for regression mode.
            Provide it as ``range(start_year, end_year + 1)`` or as an
            explicit list of consecutive years in fit window order. When
            omitted, the source registry supplies the default fit window from
            the modeled year minimum through the source historical cutoff. For
            EXIOBASE 3.10.2 and OECD ICIO v2025, this resolves to 1995 to
            2022; other supported MRIO sources use their own registry window.
        l2_reuse_years: Historical L2 reuse year selector used by all UT
            historical reuse routes. In ``projection_mode="historical_reuse"``
            it applies to all UT methods; in ``projection_mode="regression"``
            it applies to adjusted UT routes (``UT(FDa)``, ``UT(GVAa)``),
            which always use historical reuse as regression is not supported
            (would require regression on full MRIO). If omitted, defaults to
            ``reg_window`` when required.
        output_format: Persisted output file format: ``"csv"`` (default),
            ``"pickle"``, or ``"parquet"``.
        intermediate_outputs: Whether to write intermediate output families.
            These outputs are for user audit and method inspection only; they
            are not used by downstream package functions.

            - ``False`` (default): skip writing enacting metrics and
              ``utility_propagation_contrib`` (L2*b for FUs) outputs.
            - ``True``: write all output families.
        figures: Whether to render figures.
            Default is ``True``.
        figure_format: Figure render settings mapping. Defaults to
            ``{"format": "png", "dpi": 500}``.

            Nested keys:

            - ``format``: Figure file format. Accepted values are ``"png"``,
              ``"pdf"``, and ``"svg"``.
            - ``dpi``: Positive integer figure resolution used for raster
              outputs.
        figure_options: Figure product selector mapping. Defaults to
            ``{"per_method": True, "multi_method": True}``.

            Nested keys:

            - ``per_method``: Whether to render method specific figures, with
              one separate figure for each allocation method.
            - ``multi_method``: Whether to render cross method comparison
              figures, with multiple allocation methods shown in the same
              figure.
        figure_external_method: Optional external deterministic aSoCC selector
            block used only for figure rendering. Use
            ``prepare_external_inputs(...)`` to import the external aSoCC
            README guidance and runnable CSV examples, then follow the
            imported guide for method syntax and data input format. This
            argument is valid only when ``figures=True``. Omit it to render only native
            deterministic aSoCC method rows. Defaults to ``None``.

        refresh: If ``True``, remove and rebuild the resolved deterministic
            aSoCC source and version output scope for this project, source
            label, and aggregate version. The cleared scope is the source and
            version ``deterministic`` folder under
            ``<project>/B1_asocc/<source_or_source__agg_version>``. For
            example, for ``project_name="demo"``,
            ``source="exiobase_3102_ixi"``, and ``agg_version="elec"``, the
            refreshed path is
            ``<repo>/demo/B1_asocc/exiobase_3102_ixi__elec/deterministic``.
            Processed MRIO inputs, processed population and GDP, raw
            downloads, and downstream aCC or ASR outputs are not refreshed.
            Defaults to ``False``.

    Returns:
        AllocateReport describing deterministic aSoCC table outputs and figure
        outputs when figures are requested.

    Raises:
        ValueError: If FU code is invalid, selectors are incompatible with
            ``method_plan``, required indices are missing, disallowed indices
            are provided, LCIA inputs are required but missing, aggregated output
            is requested in disallowed cases, or required MRIO/population/GDP
            years are unavailable.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

    Example:
        Compute deterministic aSoCC for ``L2.c.b`` producing sector ``Paper``
        and consuming region ``FR``, using defaults where omitted::

            deterministic_asocc(
                project_name="demo",
                source="exiobase_3102_ixi",
                years=2022,
                fu_code="L2.c.b",
                s_p=["Paper"],
                r_c=["FR"],
            )
    """
    project_root = outputs_project_root(project_name=project_name)
    phase = _phase if _phase is not None else PhasePrinter("deterministic_asocc")
    owns_phase = _phase is None
    # Ensure per project log handlers are always released, so subsequent
    # refresh runs can remove output files.
    try:
        phase.announce("Phase B.1: aSoCC", "deterministic_asocc")
        fu_norm = normalize_fu_code(fu_code)
        output_format_norm = normalize_allocate_output_format(output_format)
        figure_format = normalize_figure_format(figure_format)
        figure_options = normalize_figure_options(
            figure_options,
            allow_single_year_style=False,
            allow_polar_years=False,
            allow_per_method=True,
            allow_multi_method=True,
        )
        figure_output_format = str(figure_format["format"])
        figure_dpi = int(figure_format["dpi"])
        if figure_external_method is not None and not figures:
            raise ValueError("figure_external_method is only valid when figures=True.")
        years = normalize_optional_year_selector(years, name="years")
        reference_years = normalize_optional_year_selector(
            reference_years,
            name="reference_years",
        )
        reg_window = normalize_optional_reg_window_selector(reg_window, name="reg_window")
        l2_reuse_years = normalize_optional_year_selector(l2_reuse_years, name="l2_reuse_years")
        figure_external_method = normalize_external_method_selector(
            figure_external_method,
            fu_code=fu_norm,
            argument_name="figure_external_method",
        )

        base_l1, base_combined, base_one_step = resolve_method_selection(
            fu_code=fu_code,
            method_plan=method_plan,
            l1_methods=l1_methods,
            one_step_methods=one_step_methods,
            two_step_methods=two_step_methods,
            l1_l2_pairs=l1_l2_pairs,
        )
        mode = normalize_l1_reg_mode(l1_reg_aggreg)
        group_indices = normalize_output_mode(group_indices)
        r_p_list = ensure_list_str(r_p)
        s_p_list = ensure_list_str(s_p)
        r_c_list = ensure_list_str(r_c)
        r_f_list = ensure_list_str(r_f)
        common = _RunCommonInputs(
            project_name=project_name,
            source=source,
            agg_version=agg_version,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
            years=years,
            historical_year_cap=None,
            refresh=refresh,
            lcia_method=lcia_method,
            fu_code=fu_code,
            r_p=r_p_list,
            s_p=s_p_list,
            r_c=r_c_list,
            r_f=r_f_list,
            reference_years=reference_years,
            ssp_scenario=ssp_scenario,
            projection_mode=projection_mode,
            reg_window=reg_window,
            l2_reuse_years=l2_reuse_years,
            output_format=output_format_norm,
            intermediate_outputs=bool(intermediate_outputs),
        )
        validate_grouped_request(
            fu_norm=fu_norm,
            grouped_requested=group_indices,
            r_p=r_p_list,
            s_p=s_p_list,
            r_c=r_c_list,
            r_f=r_f_list,
        )

        report = _run_allocate_family(
            common=common,
            mode=mode,
            group_indices=group_indices,
            l1_override=base_l1,
            combined_override=base_combined,
            l2_one_step_override=base_one_step,
            figures=figures,
            refresh=refresh,
            figure_external_method=figure_external_method,
            figure_options=figure_options,
            figure_output_format=figure_output_format,
            figure_dpi=figure_dpi,
            phase=phase,
        )
        if owns_phase and report.output_root is not None:
            detail = (
                phase_reused_detail if report.reuse_status == "reused_exact" else phase_ready_detail
            )
            phase.complete(
                detail(scope_name="aSoCC", output_root=report.output_root),
                owner="deterministic_asocc",
            )
        return report
    finally:
        if owns_phase:
            phase.finish()
        close_loggers_for_scope(project_root)
