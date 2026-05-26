"""Public entrypoint for allocated carrying capacity (aCC) uncertainty."""

from typing import Any

from pyaesa.acc.uncertainty.runner import run_uncertainty_acc
from pyaesa.shared.figures.request_validation import (
    normalize_figure_format,
    normalize_figure_options,
)
from pyaesa.shared.uncertainty_assessment.run_state.report import UncertaintyRunReport


def uncertainty_acc(
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
    base_asocc_args: dict[str, Any] = {
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
    external_method: dict[str, Any] | None = None,
    base_cc_args: dict[str, Any] = {
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
    uncertainty_config: dict[str, Any] = {
        "mc_parameters": {
            "fixed": {"active": False, "n_runs": 1000},
            "convergence": {
                "active": True,
                "max_runs": 500000,
                "rtol": 0.05,
                "stable_runs": 10000,
                "convergence_statistics": ["mean"],
            },
        },
        "lcia_uncertainty": {"active": False, "sector_cov_mapping": {}},
        "projection_uncertainty": {"active": True},
        "reference_year_uncertainty": {"active": True},
        "inter_mrio_uncertainty": {"active": False, "alternate_source": None},
        "inter_method_uncertainty": {"active": True, "mode": "equal_weight"},
        "dynamic_ar6_cc_uncertainty": {
            "active": True,
            "sampling_method": "srs",
            "category_uncertainty": False,
        },
    },
    sobol_parameters: dict[str, Any] = {
        "active": False,
        "fixed": {"active": False, "n_base_samples": 128},
        "convergence": {
            "active": True,
            "max_base_samples": 1048576,
            "rtol": 0.05,
        },
        "sobol_years": None,
    },
    output_format: str = "csv_compact",
    figures: bool = True,
    figure_format: dict[str, Any] = {"format": "png", "dpi": 500},
    figure_options: dict[str, bool] = {
        "per_method": True,
        "multi_method": True,
        "inter_method": True,
    },
    subfigures: bool = True,
    refresh: bool = False,
) -> UncertaintyRunReport:
    """Run allocated carrying capacity (aCC) Monte Carlo uncertainty.

    The function creates or reuses upstream allocated shares of carrying
    capacities (aSoCC) and dynamic AR6 carrying capacity (CC) uncertainty
    outputs, samples only the uncertainty sources requested in
    ``uncertainty_config``, and computes ``aCC = aSoCC * CC``. It writes run
    values, summary statistics, uncertainty source parameters, Sobol variance
    decomposition when enabled, and figures when requested under the aCC Monte
    Carlo output folder. Omit arguments to use their default.

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
        uncertainty_config: Monte Carlo configuration dictionary. The default
            signature activates projection, reference year, inter-method, and
            dynamic AR6 CC uncertainty for dynamic carrying capacity branches.
            LCIA uncertainty is inactive by default because L2 LCIA rows
            require ``sector_cov_mapping``: keys are output ``s_p`` labels and
            values are sector CoV codes from ``sec_cbca_covs.csv``. Inter
            MRIO uncertainty is inactive by default because it requires an
            alternate published disaggregated aSoCC source. Source blocks use
            an ``active`` boolean; write ``active=False`` to disable a default
            active source. See
            ``data_raw/methodological_notes/methodological_note__acc_uncertainty_sources.pdf``
            for uncertainty source definitions and mathematical expressions.

            Accepted keys:

            - ``mc_parameters``: optional dictionary with ``convergence`` and
              ``fixed`` mode blocks. Exactly one mode must be active.

              Nested mode blocks:

              - ``convergence``: convergence mode block. This is the default
                active mode.

                Nested keys:

                - ``active``: Whether convergence mode is active.
                - ``max_runs``: Maximum number of Monte Carlo runs allowed
                  before stopping.
                - ``rtol``: Relative tolerance used to decide whether
                  monitored summary statistics have converged.
                - ``stable_runs``: Number of consecutive accepted runs that
                  must remain within tolerance before the run stops.
                - ``convergence_statistics``: Statistic monitored for
                  convergence. Monte Carlo convergence is mean only and
                  defaults to ``["mean"]``.

              - ``fixed``: fixed run count mode block.

                Nested keys:

                - ``active``: Whether fixed mode is active.
                - ``n_runs``: Exact number of Monte Carlo runs.

            - ``lcia_uncertainty``: optional LCIA source block. It defaults
              to ``{"active": False, "sector_cov_mapping": {}}``. Country
              level LCIA CoVs are resolved automatically. L2 sector resolved
              LCIA rows require ``sector_cov_mapping`` to map output ``s_p``
              labels to sector CoV codes from ``sec_cbca_covs.csv``, for
              example
              ``{"active": True, "sector_cov_mapping": {"Paper": "Paper"}}``.
              Carbon consumption based accounts
              coefficients of variation (CoV) files are available under
              ``data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/``. Users
              can inspect ``sec_cbca_covs.csv`` for sector CoV codes before
              choosing ``sector_cov_mapping`` values. CoV keys must match the
              LCIA uncertainty output domain. If ``agg_reg=True``, region
              keys use ``reg_cbca_covs_agg_<agg_version>.csv``; otherwise
              they use ``reg_cbca_covs.csv``. If ``group_indices=True``
              sums a region selector axis after calculation, put the full
              combined output region label in
              ``reg_cbca_covs_agg_<agg_version>_group_indices.csv`` when
              ``agg_reg=True``, otherwise ``reg_cbca_covs_group_indices.csv``.
              If ``agg_sec=True`` or ``group_indices=True``, use the
              corresponding ``agg_sec`` output labels or combined output
              ``s_p`` labels as ``sector_cov_mapping`` keys. For
              example, with ``group_indices=True`` and ``s_p=["A", "B"]``,
              write ``sector_cov_mapping={"A, B": "Electricity"}`` when
              ``Electricity`` is the sector CoV code selected from
              ``sec_cbca_covs.csv``.

              Nested keys:

              - ``active``: Whether LCIA uncertainty is active.
              - ``sector_cov_mapping``: Mapping from output ``s_p`` labels to
                sector CoV codes from ``sec_cbca_covs.csv``.

            - ``projection_uncertainty``: optional source block. It defaults
              to ``{"active": True}``. For prospective rows using L2
              historical reuse, each Monte Carlo run samples one L2 reuse
              year uniformly from the deterministic ``l2_reuse_years``
              candidates requested for the years where reuse applies.

              Nested key:

              - ``active``: Whether projection uncertainty is active.

            - ``reference_year_uncertainty``: optional source block. It
              defaults to ``{"active": True}``. For acquired rights (AR)
              routes, each Monte Carlo run samples uniformly among requested
              reference years admissible for the studied year
              (``reference_year <= year``). The same sampled reference year is
              shared across the run when admissible; years for which it is not
              admissible resample among their admissible reference years.

              Nested key:

              - ``active``: Whether reference year uncertainty is active.

            - ``inter_mrio_uncertainty``: optional source block. To activate
              it, write ``{"active": True, "alternate_source":
              "<disaggregated label>"}``, for example
              ``{"active": True, "alternate_source": "oecd_electricity"}``.
              It applies continuous uniform interpolation between the main
              MRIO source and an alternate published disaggregated aSoCC
              source created by ``disaggregate_asocc(...)``. It applies only
              to non LCIA methods.

              Nested keys:

              - ``active``: Whether inter-MRIO uncertainty is active.
              - ``alternate_source``: Published disaggregated aSoCC source
                label used as the alternate MRIO source.

            - ``inter_method_uncertainty``: optional source block. It
              defaults to ``{"active": True, "mode": "equal_weight"}``.
              Each Monte Carlo run samples one method leaf among the selected
              deterministic and external methods. Equal weight mode writes the
              tree CSV, README, and rendered probability tree under the
              run folder ``figures/inter_method_tree/``. To prepare custom
              weights before running uncertainty, use
              ``write_asocc_weight_template(...)``; it writes
              ``equal_weights.csv``, ``README_inter_method_weights.txt``, and
              ``probability_tree__equal_weights.<ext>`` under
              ``B1_asocc/preview_inter_method_weights/``. Use
              ``preview_asocc_weight_tree(...)`` to validate and render a
              custom probability tree before using
              ``{"mode": "custom", "version_name": "..."}``.

              Nested keys:

              - ``active``: Whether inter-method uncertainty is active.
              - ``mode``: Inter-method sampling mode. Accepted values are
                ``"equal_weight"`` and ``"custom"``.
              - ``version_name``: Custom weight version used when
                ``mode="custom"``.

            - ``dynamic_ar6_cc_uncertainty``: optional AR6 CC source block. It
              defaults to ``{"active": True, "sampling_method": "srs",
              "category_uncertainty": False}``. ``sampling_method`` accepts
              ``"srs"`` for simple random sampling (samples across retained
              model-scenario pairs matching the requested category and SSP) or
              ``"lhs"`` for Latin hypercube sampling (samples among retained
              models first, then among retained scenarios for the selected
              model, category, and SSP to limit over representation of models
              with more AR6 submissions). The effect of this choice is visible
              in ``process_ar6(...)`` sampling diagnostic figures.
              ``category_uncertainty`` is inactive by default. If ``True``,
              each Monte Carlo run first samples one retained AR6 category
              with equal probability in the studied SSP pool. It then applies
              ``sampling_method`` inside that selected category; with
              ``sampling_method="lhs"``, this means model first, then scenario
              inside that model.

              Nested keys:

              - ``active``: Whether dynamic AR6 carrying capacity uncertainty
                is active.
              - ``sampling_method``: AR6 pathway sampling method. Accepted
                values are ``"srs"`` and ``"lhs"``.
              - ``category_uncertainty``: Whether each Monte Carlo run samples
                one retained AR6 category before applying ``sampling_method``
                inside that category.
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
        base_asocc_args: Optional upstream aSoCC selector envelope. Write
            nested arguments as ``base_asocc_args={"method_plan": "default"}``.
            Accepted keys are method and projection controls only. Omit the
            envelope or any accepted key to use its default.

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
        output_format: Public uncertainty table format, either
            ``"csv_compact"`` or ``"parquet"``. Defaults to
            ``"csv_compact"``.
        sobol_parameters: Sobol sensitivity settings. Sobol analysis estimates
            the contribution of active uncertainty sources to output variance
            and writes ``README_sobol.txt`` under ``results/sobol/`` for
            interpretation. The default has ``active=False`` and writes no
            Sobol artifacts. To run Sobol, set ``active=True``. With the
            default mode blocks, enabled Sobol uses convergence mode. To run
            a fixed Sobol design, set ``fixed.active=True`` and
            ``convergence.active=False``. Sobol base sizes must be powers of
            two.

            Nested keys:

            - ``active``: Whether Sobol sensitivity analysis is active.
            - ``convergence``: convergence mode block. When Sobol is active,
              exactly one of ``convergence`` or ``fixed`` is active.

              Nested keys:

              - ``active``: Whether convergence mode is active.
              - ``max_base_samples``: Maximum Sobol base size.
              - ``rtol``: Relative tolerance for monitored ``S1`` and ``ST``
                indices.

            - ``fixed``: fixed Sobol base size mode block.

              Nested keys:

              - ``active``: Whether fixed mode is active.
              - ``n_base_samples``: Exact Sobol base size.

            - ``sobol_years``: Studied output years evaluated by Sobol for
              static carrying capacity branches. When omitted, static Sobol
              evaluates only the first and last studied years in the requested
              studied year set. Dynamic AR6 carrying capacity Sobol ignores
              this yearly selector and evaluates cumulative aCC over the full
              studied period.
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
            ``{"per_method": True, "multi_method": True, "inter_method": True}``.

            Nested keys:

            - ``per_method``: Whether to render method specific figures, with
              one separate figure for each allocation method.
            - ``multi_method``: Whether to render cross method comparison
              figures, with multiple allocation methods shown in the same
              figure.
            - ``inter_method``: Whether to render inter-method uncertainty
              figures. These figures use the same method specific layout as
              ``per_method``, but represent uncertainty induced by the inter
              method uncertainty setting rather than comparing individual
              allocation methods. This option is ignored when inter-method
              uncertainty is inactive.
        subfigures: Whether prerequisite uncertainty aSoCC and active
            uncertainty AR6 CC calls should render their own figures when
            ``figures=True``. Default: ``True``.

        refresh: If ``True``, refresh the resolved aCC Monte Carlo outputs and
            every upstream component output scope called by this aCC
            uncertainty request. This can refresh deterministic and Monte
            Carlo aSoCC output scopes, and when dynamic AR6 CC is used, the
            matching processed AR6 output scope selected by
            ``process_ar6(...)`` and the matching
            ``deterministic_ar6_cc(...)`` output scope. When dynamic AR6 CC
            uncertainty is active, this can also refresh the matching
            ``uncertainty_ar6_cc(...)`` output scope. For example, matching
            aCC Monte Carlo run folders are refreshed under
            ``<repo>/demo/B2_acc/exiobase_3102_ixi__elec/monte_carlo/mc_<generated_id>``.
            External aSoCC inputs, static carrying capacity files, processed
            MRIO inputs, processed population and GDP, raw downloads, and
            downstream ASR outputs are not refreshed. Defaults to ``False``.

    Returns:
        UncertaintyRunReport describing aCC uncertainty table outputs and
        figure outputs when figures are requested.

    Raises:
        ValueError: If the request, source configuration, or persisted input
            files are inconsistent.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

    Example:
        Run static and dynamic aCC for ``L2.c.b`` producing sector ``Paper``
        and consuming region ``FR`` with LCIA uncertainty added, using
        defaults where omitted::

            from pyaesa import uncertainty_acc

            uncertainty_acc(
                project_name="demo",
                source="exiobase_3102_ixi",
                years=range(2020, 2031),
                lcia_method="gwp100_lcia",
                fu_code="L2.c.b",
                s_p=["Paper"],
                r_c=["FR"],
                base_cc_args={
                    "static": {"active": True},
                    "dynamic_ar6": {"active": True},
                },
                uncertainty_config={
                    "lcia_uncertainty": {
                        "active": True,
                        "sector_cov_mapping": {"Paper": "Paper"},
                    },
                },
            )
    """
    figure_format_norm = normalize_figure_format(figure_format)
    figure_options_norm = normalize_figure_options(
        figure_options,
        allow_single_year_style=False,
        allow_polar_years=False,
        allow_per_method=True,
        allow_multi_method=True,
        allow_inter_method=True,
    )
    subfigures_effective = all((figures, subfigures))
    return run_uncertainty_acc(
        project_name=project_name,
        source=source,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
        years=years,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        group_indices=group_indices,
        lcia_method=lcia_method,
        base_asocc_args=base_asocc_args,
        external_method=external_method,
        base_cc_args=base_cc_args,
        uncertainty_config=uncertainty_config,
        sobol_parameters=sobol_parameters,
        output_format=output_format,
        figures=figures,
        figure_options=figure_options_norm,
        figure_format=figure_format_norm,
        subfigures=subfigures_effective,
        refresh=refresh,
    )
