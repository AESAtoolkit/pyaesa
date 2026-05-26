"""Public entrypoint for allocated carrying capacity (aCC)."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from pyaesa.acc.deterministic.runtime.prerequisites import ensure_acc_branch_prerequisites
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.asocc.runtime.scope.branch_resolution import resolve_allocate_project_base
from pyaesa.external_inputs.asocc.schema.contracts import (
    normalize_external_method_selector,
    validate_external_method_collisions,
)
from pyaesa.shared.acc_asr_common.branches.config import normalize_base_cc_args
from pyaesa.shared.acc_asr_common.branches.expand import iter_cc_method_branches
from pyaesa.shared.acc_asr_common.deterministic.downstream.tabular_io import (
    normalize_downstream_output_format,
)
from pyaesa.shared.acc_asr_common.persistence.requests import (
    build_public_cc_branch_args,
    build_public_composite_request_payload,
)
from pyaesa.shared.acc_asr_common.reporting import build_downstream_common_scope_lines
from pyaesa.shared.acc_asr_common.scope.composite import (
    base_asocc_kwargs_from_allocate_args,
    build_composite_base_allocate_args,
    normalize_base_asocc_args,
    normalize_mrio_scope,
    normalize_shared_lcia_methods,
)
from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
    normalize_figure_options,
)
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_B2_ACC,
    CompositePhaseIndexEntry,
    phase_ready_detail,
    phase_reused_detail,
    public_phase_reuse_status,
    write_phase_index,
)
from pyaesa.shared.runtime.reporting.output_roots import public_output_root_from_path
from pyaesa.shared.runtime.reporting.phase import PhasePrinter
from pyaesa.shared.runtime.reporting.summary_log import summary_log_path, write_summary_log
from pyaesa.shared.selectors.request_targets import build_asocc_target_selector
from pyaesa.shared.selectors.time_selectors import (
    normalize_requested_years,
    normalize_time_selector_mapping,
)

from .deterministic.runner import run_single_acc
from .deterministic.state.reports import (
    ACCBranchReport,
    ComputeACCReport,
    acc_phase_inventory_lines,
    acc_phase_summary_lines,
)


def _build_branch_signature(
    *,
    public_request_payload: dict[str, Any],
) -> dict[str, Any]:
    normalized_payload = cast(
        dict[str, Any],
        normalize_time_selector_mapping(public_request_payload),
    )
    return {
        "function": "deterministic_acc",
        **normalized_payload,
    }


def deterministic_acc(
    *,
    project_name: str,
    source: str,
    agg_reg: bool = False,
    agg_sec: bool = False,
    agg_version: str = "",
    years: int | list[int] | range,
    fu_code: str,
    s_p: str | list[str] | None = None,
    r_p: str | list[str] | None = None,
    r_c: str | list[str] | None = None,
    r_f: str | list[str] | None = None,
    group_indices: bool = False,
    lcia_method: str | list[str],
    base_asocc_args: dict = {
        "method_plan": "default",
        "l1_methods": None,
        "one_step_methods": None,
        "two_step_methods": None,
        "l1_l2_pairs": None,
        "l1_reg_aggreg": "post",
        "reference_years": None,
        "ssp_scenario": ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
        "projection_mode": "regression",
        "reg_window": None,
        "l2_reuse_years": None,
        "include_lcia_based_allocation_methods": True,
    },
    external_method: dict | None = None,
    base_cc_args: dict = {
        "static": {"active": True, "exclude_max_cc": False},
        "dynamic_ar6": {
            "active": False,
            "harmonization": True,
            "harmonization_method": "offset",
            "category": ["C1", "C2", "C3", "C4"],
            "ssp_scenario": ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
            "emission_type": "kyoto_gases",
            "include_afolu": False,
            "emissions_mode": "gross_alt",
            "subset_version": None,
        },
    },
    output_format: str = "csv",
    figures: bool = True,
    figure_format: dict[str, Any] = {"format": "png", "dpi": 500},
    figure_options: dict[str, bool] = {"per_method": True, "multi_method": True},
    subfigures: bool = True,
    refresh: bool = False,
    _phase: PhasePrinter | None = None,
    _shared_asocc_lcia_methods: list[str] | None = None,
) -> ComputeACCReport:
    """Compute deterministic allocated carrying capacities (aCC).

    The function creates or reuses deterministic allocated shares of carrying
    capacities (aSoCC) and dynamic AR6 carrying capacity (CC) prerequisites
    when required, then computes ``aCC = aSoCC * CC``. It writes deterministic
    aCC tables and renders figures when requested. Omit arguments to use their
    default.

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
        lcia_method: Carrying capacity reference method(s) selected for aCC
            (for example ``"pb_lcia"``, ``"gwp100_lcia"``, or ``"ef_3.1"``).
            Static CC requires a matching reference file in
            ``data_raw/carrying_capacities/``; to add a custom carrying
            capacity reference, follow
            ``README_add_custom_carrying_capacities.txt`` in that folder and
            pass the custom method file stem here. Some references support
            only steady state CC. Dynamic AR6 CC is available for any
            selected method whose static carrying capacity CSV contains an
            impact row equal to ``"GWP_100"``.
            ``base_asocc_args["include_lcia_based_allocation_methods"]``
            controls whether LCIA based allocation methods are included; it
            defaults to ``True``. When included, the selected ``lcia_method``
            must also be available as a processed MRIO LCIA method. pyaesa
            currently supports LCIA based allocation methods only for EXIOBASE
            sources.
            To add a custom LCIA method with which run ``process_mrio(...)``,
            follow ``README_add_custom_lcia_characterization_matrices.txt`` in
            ``data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/``
            and pass the custom method file stem there.
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
        base_asocc_args: Optional aSoCC only envelope used to resolve the
            upstream deterministic ``deterministic_asocc(...)`` scope. Write
            nested arguments as ``base_asocc_args={"method_plan": "default"}``.
            Omit the envelope or any accepted key to use its default.

            Nested keys:

            - ``method_plan``: ``method_plan`` defaults to ``"default"`` and
              accepts ``"default"``, ``"one_step"``, ``"two_steps"``,
              ``"pairs"``, or ``"one_step_pairs"``. When omitted, all pyaesa
              allocation methods available for the selected ``fu_code`` are
              applied. See
              ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
              for the allocation methods available per functional unit,
              including definitions and mathematical expressions.
            - ``l1_methods``: Optional L1 subset. Omit it to keep all L1
              methods allowed by ``method_plan``. In ``"default"``, this
              filters only L1 weights used by two step methods. In
              ``"two_steps"``, it filters the two step cartesian L1 side.
            - ``one_step_methods``: Optional one step L2 subset. Omit it to
              keep all one step methods allowed by ``method_plan``.
            - ``two_step_methods``: Optional two step L2 subset. Omit it to
              keep all two step L2 methods allowed by ``method_plan``.
            - ``l1_l2_pairs``: Explicit pair list formatted as
              ``"L1METHOD::L2METHOD"``. Omit it unless ``method_plan`` is
              ``"pairs"`` or ``"one_step_pairs"``.
            - ``l1_reg_aggreg``: L1 aggregation mode for methods where timing
              matters (``PR(GDPcap)``, ``PR-HR(Ecap)`` and ``AR(Ecap)``).
              ``"pre"`` aggregates regions before L1 computation. ``"post"``
              (default) computes on original regions, then aggregates.
            - ``reference_years``: Acquired rights (AR) methods reference
              year selector. Accepts a single year, list, or range. If
              omitted, AR routes use all historical years in the studied range
              up to the source registry historical cutoff. For EXIOBASE
              3.10.2 and OECD ICIO v2025, the cutoff is 2022; other supported
              MRIO sources use their own registry cutoff.
            - ``ssp_scenario``: SSP scenario name or list. Defaults to
              ``["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]`` and is applied
              when scenario dependent inputs are required.
            - ``projection_mode``: Projection policy for post historical
              years of L2 utilitarian (UT) methods (MRIO economic enacting
              metrics). Defaults to ``"regression"``. ``"regression"``
              projects UT inputs for future years. ``"historical_reuse"``
              reuses historical UT structures.
            - ``reg_window``: Historical regression fit window for regression
              mode. Provide it as ``range(start_year, end_year + 1)`` or as
              an explicit list of consecutive years in fit window order. When
              omitted, the source registry supplies the default fit window
              from the modeled year minimum through the source historical
              cutoff. For EXIOBASE 3.10.2 and OECD ICIO v2025, this resolves
              to 1995 to 2022; other supported MRIO sources use their own
              registry window.
            - ``l2_reuse_years``: Historical L2 reuse year selector used by
              all UT historical reuse routes. In
              ``projection_mode="historical_reuse"`` it applies to all UT
              methods; in ``projection_mode="regression"`` it applies to
              adjusted UT routes (``UT(FDa)``, ``UT(GVAa)``), which always
              use historical reuse as regression is not supported (would
              require regression on full MRIO). If omitted, defaults to
              ``reg_window`` when required.
            - ``include_lcia_based_allocation_methods``: Whether to include
              LCIA based allocation methods (e.g.: acquired rights - AR, or
              historical responsibility - PR-HR). Defaults to ``True``.
              ``False`` keeps only non LCIA dependent allocation methods.
        base_cc_args: Carrying capacity family envelope. The package default
            is static active and dynamic AR6 inactive. Provide a
            ``dynamic_ar6`` block to add dynamic AR6 carrying capacities. Set
            ``static.active=False`` for dynamic only runs.

            Nested keys:

            - ``static``: Static carrying capacity branch. It is active by
              default unless ``active=False`` is provided.

              Nested keys:

              - ``active``: Whether the static branch is active. Defaults to
                ``True``.
              - ``exclude_max_cc``: Whether to use only ``min_cc``. Defaults
                to ``False``. ``False`` keeps the paired
                ``min_cc`` plus ``max_cc`` interpretation when ``max_cc`` is
                present. ``True`` uses only ``min_cc``.

            - ``dynamic_ar6``: Dynamic AR6 carrying capacity branch. It is
              inactive when omitted. When the block is provided, it is active
              unless ``active=False`` is provided. It uses top level
              ``years`` and requires at least two consecutive years.

              Nested keys:

              - ``active``: Whether the dynamic AR6 branch is active.
              - ``harmonization``: Whether to harmonize retained AR6 pathways
                to the historical baseline. Defaults to ``True``.
              - ``harmonization_method``: Harmonization method applied only
                when ``harmonization=True``. Defaults to ``"offset"``. The
                only supported value is currently ``"offset"``. Ignored when
                ``harmonization=False``.
              - ``category``: AR6 category classification selector for global
                warming trajectories, as a string or list, such as ``"C3"``
                or ``["C1", "C2"]``. Valid values are C1 through C8.
                Defaults to C1 to C4, the categories aligned with the
                2015 Paris Agreement.

              - ``ssp_scenario``: Canonical SSP selector as a string, list,
                or ``None``, such as ``"SSP2"`` or ``["SSP1", "SSP2"]``.
                Defaults to SSP1 to SSP5.

              - ``emission_type``: Dynamic AR6 emission type. Accepted values
                are ``"kyoto_gases"`` (default) and ``"co2"``.
                ``emission_type="kyoto_gases"`` uses the GWP100 Kyoto Gases
                aggregate; ``emission_type="co2"`` uses direct CO2 pathways.

              - ``include_afolu``: Whether AFOLU emissions are included inside
                the selected ``emission_type``. Defaults to ``False``.

              - ``emissions_mode``: Dynamic AR6 emissions mode. Accepted
                values are ``"net"``, ``"gross"``, and ``"gross_alt"``.
                Defaults to ``"gross_alt"``. ``"net"`` uses net AR6 emissions
                pathways directly. ``"gross"`` removes all sequestration
                sources from net emissions. ``"gross_alt"`` removes all
                sequestration sources except CCS. CCS is retained because IPCC
                AR6 treats CCS as capture at fossil or industrial point
                sources rather than direct removal of CO2 from the atmosphere,
                so it is kept separate from net negative sequestration. Gross
                modes write positive emissions rows and signed negative
                sequestration companion rows; downstream aCC and ASR consume
                only the positive emissions rows. See
                ``data_raw/methodological_notes/methodological_note__steady_state__dynamic_cc.pdf``
                for the methodological explanation.

              - ``subset_version``: Optional selector for a subset of AR6
                model-scenario pairs. Follow
                ``data_processed/ar6/<processed_scope>/README_model_scenario_subset.txt``
                to create the subset CSV.
        external_method: Optional external aSoCC method selector. Use
            ``{"l1_methods": [...]}`` for L1 functional units. For L2
            functional units use ``{"one_step_methods": [...]}`` and/or
            ``{"l1_l2_pairs": ["<l1_method>::<l2_method>", ...]}``.
            Omit this argument or pass ``None`` when using only native pyaesa
            deterministic aSoCC methods. Use ``prepare_external_inputs(...)``
            to import the external aSoCC README guidance and runnable CSV examples, then
            follow the imported README guidance for external method names, selector
            syntax, and deterministic or Monte Carlo external aSoCC CSV
            preparation.
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
        figure_options: Figure product selector mapping. Defaults to
            ``{"per_method": True, "multi_method": True}``.

            Nested keys:

            - ``per_method``: Whether to render method specific figures, with
              one separate figure for each allocation method.
            - ``multi_method``: Whether to render cross method comparison
              figures, with multiple allocation methods shown in the same
              figure.
        subfigures: Whether prerequisite deterministic aSoCC and dynamic
            AR6 CC calls should render their own figures when ``figures=True``.
            Default: ``True``.

        refresh: If ``True``, clear and recompute the resolved deterministic
            aCC output scope for this project, source and aggregation version,
            carrying capacity source, and carrying capacity type, plus the
            matching deterministic aSoCC prerequisite used by that request.
            When dynamic AR6 CC is used, this also refreshes the matching
            processed AR6 output scope selected by ``process_ar6(...)`` and
            the matching ``deterministic_ar6_cc(...)`` output scope.
            For example, for ``project_name="demo"``,
            ``source="exiobase_3102_ixi"``, ``agg_version="elec"``, and
            static ``cc_source="gwp100_lcia"``, the refreshed scope is
            ``<repo>/demo/B2_acc/exiobase_3102_ixi__elec/deterministic/static__gwp100_lcia``.
            Processed MRIO inputs, processed population and GDP, raw
            downloads, and downstream ASR outputs are not refreshed. Defaults
            to ``False``.

    Returns:
        ``ComputeACCReport`` describing deterministic aCC table outputs and
        figure outputs when figures are requested.

    Raises:
        ValueError: If the shared scope is invalid, a prerequisite contract is
            not satisfied, a required envelope key is missing, an envelope
            key or value falls outside the accepted contract, or a
            scenario dependent aSoCC branch is paired with a
            different dynamic AR6 SSP set.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

    Example:
        Compute static aCC for ``L2.c.b`` producing sector ``Paper`` and
        consuming region ``FR``, using defaults where omitted::

            deterministic_acc(
                project_name="demo",
                source="exiobase_3102_ixi",
                years=range(2020, 2031),
                lcia_method="gwp100_lcia",
                fu_code="L2.c.b",
                s_p=["Paper"],
                r_c=["FR"],
            )
    """
    shared_methods = normalize_shared_lcia_methods(lcia_method)
    requested_years = normalize_requested_years(years)
    mrio_scope = normalize_mrio_scope(
        source=source,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
        group_indices=group_indices,
    )
    asocc_config = normalize_base_asocc_args(base_asocc_args, fu_code=fu_code)
    cc_config = normalize_base_cc_args({} if base_cc_args is None else base_cc_args)
    external_method_norm = normalize_external_method_selector(
        external_method,
        fu_code=fu_code,
        argument_name="external_method",
    )
    base_allocate_args = build_composite_base_allocate_args(
        project_name=project_name,
        years=years,
        lcia_method=shared_methods,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        source=cast(str, mrio_scope["source"]),
        agg_reg=cast(bool, mrio_scope["agg_reg"]),
        agg_sec=cast(bool, mrio_scope["agg_sec"]),
        agg_version=cast(str | None, mrio_scope["agg_version"]),
        group_indices=cast(bool, mrio_scope["group_indices"]),
        base_asocc_args=asocc_config,
        asocc_lcia_methods=_shared_asocc_lcia_methods,
    )
    native_selector = build_asocc_target_selector(base_asocc_args=base_allocate_args)
    validate_external_method_collisions(
        native_labels=native_selector.get("methods"),
        external_method=external_method_norm,
        fu_code=fu_code,
        where="deterministic_acc",
    )
    proj_base = resolve_allocate_project_base(
        base_allocate_args=normalize_base_allocate_args(
            base_asocc_kwargs_from_allocate_args(base_allocate_args=base_allocate_args)
        )
    )
    fmt = normalize_downstream_output_format(output_format)
    figure_format_norm = normalize_figure_format(figure_format)
    figure_options_norm = normalize_figure_options(
        figure_options,
        allow_single_year_style=False,
        allow_polar_years=False,
        allow_per_method=True,
        allow_multi_method=True,
    )
    subfigures_effective = all((figures, subfigures))
    branches: list[ACCBranchReport] = []
    for branch in iter_cc_method_branches(
        lcia_methods=shared_methods,
        base_cc_args=cc_config,
        years=years,
    ):
        is_dynamic_branch = branch["cc_type"] == "dynamic_ar6"
        branch_harmonization = cast(bool, branch["harmonization"]) if is_dynamic_branch else True
        branch_harmonization_method = (
            cast(str, branch["harmonization_method"]) if is_dynamic_branch else "offset"
        )
        branch_category = cast(list[str] | None, branch["category"]) if is_dynamic_branch else None
        branch_ssp_scenario = (
            cast(list[str] | None, branch["ssp_scenario"]) if is_dynamic_branch else None
        )
        branch_emission_type = (
            cast(str, branch["emission_type"]) if is_dynamic_branch else "kyoto_gases"
        )
        branch_include_afolu = cast(bool, branch["include_afolu"]) if is_dynamic_branch else False
        branch_emissions_mode = (
            cast(str, branch["emissions_mode"]) if is_dynamic_branch else "gross_alt"
        )
        branch_subset_version = (
            cast(str | None, branch["subset_version"]) if is_dynamic_branch else None
        )
        owns_phase = _phase is None
        phase = PhasePrinter("deterministic_acc") if _phase is None else _phase
        prerequisite_kwargs: dict[str, Any] = {
            "phase": phase,
            "base_allocate_args": base_allocate_args,
            "cc_source": branch["cc_source"],
            "cc_type": branch["cc_type"],
            "years": requested_years,
            "harmonization": branch_harmonization,
            "harmonization_method": branch_harmonization_method,
            "category": branch_category,
            "ssp_scenario": branch_ssp_scenario,
            "emission_type": branch_emission_type,
            "include_afolu": branch_include_afolu,
            "emissions_mode": branch_emissions_mode,
            "subset_version": branch_subset_version,
            "output_format": fmt,
            "figure_format": figure_format_norm,
            "figure_options": figure_options_norm,
        }
        prerequisites = ensure_acc_branch_prerequisites(
            **prerequisite_kwargs,
            figures=False,
            refresh=refresh,
        )
        phase_entries = prerequisites.phase_entries
        phase.announce(PHASE_B2_ACC, "deterministic_acc")
        result = run_single_acc(
            proj_base=proj_base,
            build_branch_signature=_build_branch_signature,
            public_request_payload=build_public_composite_request_payload(
                project_name=project_name,
                years=years,
                lcia_method=[str(branch["cc_source"])],
                fu_code=fu_code,
                r_p=r_p,
                s_p=s_p,
                r_c=r_c,
                r_f=r_f,
                source=cast(str, mrio_scope["source"]),
                agg_reg=cast(bool, mrio_scope["agg_reg"]),
                agg_sec=cast(bool, mrio_scope["agg_sec"]),
                agg_version=cast(str | None, mrio_scope["agg_version"]),
                group_indices=cast(bool, mrio_scope["group_indices"]),
                base_asocc_args=asocc_config,
                base_cc_args=build_public_cc_branch_args(branch=branch),
                external_method=external_method_norm,
            ),
            source_label=str(base_allocate_args["source"]),
            base_allocate_args=base_allocate_args,
            fu_code=fu_code,
            external_method=external_method_norm,
            cc_source=branch["cc_source"],
            cc_type=branch["cc_type"],
            years=requested_years,
            static_cc_bounds=(
                list(branch["static_cc_bounds"])
                if branch["cc_type"] == "static"
                else list(branch_category or [])
            ),
            harmonization=branch_harmonization,
            harmonization_method=branch_harmonization_method,
            category=branch_category,
            ssp_scenario=branch_ssp_scenario,
            emission_type=branch_emission_type,
            include_afolu=branch_include_afolu,
            emissions_mode=branch_emissions_mode,
            subset_version=branch_subset_version,
            output_format=fmt,
            figures=figures,
            figure_options=figure_options_norm,
            figure_output_format=str(figure_format_norm["format"]),
            figure_dpi=int(figure_format_norm["dpi"]),
            refresh=refresh,
            status=phase,
        )
        meta_file = cast(Path, result.meta_file)
        branch_output_root = public_output_root_from_path(meta_file)
        if result.reuse_status == "reused_exact":
            phase.complete(
                phase_reused_detail(
                    scope_name="aCC",
                    output_root=branch_output_root,
                )
            )
        else:
            phase.complete(
                phase_ready_detail(
                    scope_name="aCC",
                    output_root=branch_output_root,
                )
            )
        if subfigures_effective:
            prerequisites = ensure_acc_branch_prerequisites(
                **prerequisite_kwargs,
                figures=True,
                refresh=False,
            )
        result = replace(result, dynamic_ar6_summary=prerequisites.dynamic_ar6_summary)
        phase_entries.append(
            CompositePhaseIndexEntry(
                phase=PHASE_B2_ACC,
                function="deterministic_acc",
                status="complete",
                reuse_status=public_phase_reuse_status(run_status=result.reuse_status),
                output_root=branch_output_root,
                summary_lines=acc_phase_summary_lines(branch=result),
                inventory_lines=acc_phase_inventory_lines(branch=result),
            )
        )
        result = replace(
            result,
            phase_entries=tuple(phase_entries),
            phase_index_path=write_phase_index(
                metadata_path=meta_file,
                entries=phase_entries,
            ),
        )
        branches.append(result)
        if owns_phase:
            phase.finish()
    report = ComputeACCReport(
        branches=branches,
        output_root=proj_base,
        common_lines=build_downstream_common_scope_lines(
            project_name=project_name,
            years=requested_years,
            lcia_methods=shared_methods,
            fu_code=fu_code,
            agg_reg=cast(bool, mrio_scope["agg_reg"]),
            agg_sec=cast(bool, mrio_scope["agg_sec"]),
            agg_version=cast(str | None, mrio_scope["agg_version"]),
            group_indices=cast(bool, mrio_scope["group_indices"]),
            ssp_scenarios=cast(list[str] | None, asocc_config.get("ssp_scenario")),
        ),
    )
    for branch in branches:
        meta_file = cast(Path, branch.meta_file)
        write_summary_log(
            path=summary_log_path(logs_dir=meta_file.parent),
            summary=str(report),
        )
    return report
