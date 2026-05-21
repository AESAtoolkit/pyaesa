"""Public absolute sustainability ratio (ASR) uncertainty entrypoint."""

from typing import Any

from pyaesa.asr.uncertainty.runner import run_uncertainty_asr
from pyaesa.shared.uncertainty_assessment.run_state.report import UncertaintyRunReport


def uncertainty_asr(
    *,
    project_name: str,
    source: str,
    group_reg: bool = False,
    group_sec: bool = False,
    group_version: str = "",
    years: int | list[int] | range,
    fu_code: str,
    s_p: str | list[str] | None = None,
    r_p: str | list[str] | None = None,
    r_c: str | list[str] | None = None,
    r_f: str | list[str] | None = None,
    aggreg_indices: bool = False,
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
    lca_args: dict[str, Any] = {
        "external_lca": {"active": True, "version_name": None},
        "io_lca": {"active": False},
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
        "asocc_uncertainty_sources": {
            "lcia_uncertainty": {"active": False, "sector_cov_mapping": {}},
            "projection_uncertainty": {"active": True},
            "reference_year_uncertainty": {"active": True},
            "inter_mrio_uncertainty": {"active": False, "alternate_source": None},
            "inter_method_uncertainty": {"active": True, "mode": "equal_weight"},
        },
        "ar6_cc_uncertainty_sources": {
            "dynamic_ar6_cc_uncertainty": {
                "active": True,
                "sampling_method": "srs",
                "category_uncertainty": False,
            }
        },
        "io_lca_uncertainty_sources": {
            "lcia_uncertainty": {"active": False, "sector_cov_mapping": {}}
        },
    },
    sobol_parameters: dict[str, Any] = {
        "active": True,
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
    figure_options: dict[str, Any] = {
        "per_method": True,
        "multi_method": True,
        "inter_method": True,
        "polar": {"active": True, "polar_years": None, "polar_style": "violin"},
    },
    figure_format: dict[str, Any] = {"format": "png", "dpi": 500},
    subfigures: bool = True,
    refresh: bool = False,
) -> UncertaintyRunReport:
    """Run absolute sustainability ratio (ASR) Monte Carlo uncertainty.

    The function creates or reuses upstream allocated carrying capacity (aCC)
    and IO-LCA uncertainty outputs, consumes staged external
    LCA when selected, and samples only the uncertainty sources requested in
    ``uncertainty_config``. It writes run values, summary statistics,
    uncertainty source parameters, Sobol variance decomposition when enabled,
    and figures when requested under the ASR Monte Carlo output folder. Omit
    arguments to use their default.

    Args:
        project_name: Required project name used to build
            ``<repo>/<project_name>``.
        source: MRIO source key (``"exiobase_396_ixi"``,
            ``"exiobase_396_pxp"``, ``"exiobase_3102_ixi"``,
            ``"exiobase_3102_pxp"``, or ``"oecd_v2025"``), or ``"iso3"``
            for ISO3 only mode (L1 EG/PR(GDPcap) only).
        group_reg: If ``True``, aggregate regions using a grouping file.
            Default ``False`` keeps native source regions.
        group_sec: If ``True``, aggregate sectors using a grouping file.
            Default ``False`` keeps native source sectors.
        group_version: Grouping version tag used to resolve the region/sector
            mapping CSVs. Required when ``group_reg`` or ``group_sec`` is True.
            Defaults to an empty string for ungrouped processing. Follow
            ``README_grouping.txt`` in the active
            ``data_raw/mrio/<source>/grouping`` folder to name grouping
            versions and place the matching mapping CSVs.
        years: Studied years. Accepts a single year, list, or range. If
            omitted, all available MRIO
            years for the selected source/group version are used.
        lcia_method: Required LCIA method name or list of names. When upstream
            static carrying capacity (CC) is active, the denominator requires
            a matching carrying capacity CSV. The package includes static CC files for
            ``"pb_lcia"``, ``"gwp100_lcia"``, and ``"ef_3.1"``; among these,
            ``"ef_3.1"`` is the default carrying capacity method that
            currently has no MRIO LCIA characterization matrix. It is still
            dynamic AR6 compatible for its ``"GWP_100"`` impact category.
            Dynamic AR6 CC is supported for ``"gwp100_lcia"`` and for
            ``"ef_3.1"`` impact ``"GWP_100"`` only. Custom static CC methods
            also require a matching file in
            ``data_raw/carrying_capacities/``; follow
            ``README_add_custom_carrying_capacities.txt``. When upstream
            allocated shares of carrying capacities (aSoCC) LCIA based methods
            or IO-LCA generation is in scope, custom EXIOBASE MRIO LCIA
            methods must be prepared with
            ``data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/README_add_custom_lcia_characterization_matrices.txt``
            and processed with ``process_mrio(...)``.
            ``base_asocc_args["include_lcia_based_allocation_methods"]``
            controls whether LCIA based allocation methods are included. When
            it is ``False``, upstream aSoCC keeps only non LCIA dependent
            methods, so the requested ``lcia_method`` may be non MRIO when the
            matching carrying capacity and selected LCA numerator prerequisites
            exist.
        fu_code: Required functional unit code (for example ``"L1.a"``,
            ``"L2.c.b"``). See
            ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
            for all available functional unit codes and the system
            boundaries each represents.
        aggreg_indices: Whether multiple selected region/sector indices are
            reported as separate rows or summed into one row after the
            selected MRIO scope is computed.
            - ``False`` (default): keep selected values as independent rows.
            - ``True``: sum selected values into one row.
            Not allowed for ``L2.a.b``, ``L2.b.b``, and ``L2.c.b`` because
            aggregating CBA total demand system boundaries can double count.
            For these functional units, define the aggregation from
            ``process_mrio(...)`` onward with
            ``group_reg``/``group_sec``/``group_version``.
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
                warming trajectories, as a string or list, such as ``"C2"``
                or ``["C1", "C2"]``. Defaults to C1 to C4.
              - ``ssp_scenario``: Canonical SSP selector as a string, list,
                or ``None``, such as ``"SSP2"`` or ``["SSP1", "SSP2"]``.
                Defaults to SSP1 to SSP5.
              - ``emission_type``: Dynamic AR6 emission type. Accepted values
                are ``"kyoto_gases"`` (default) and ``"co2"``.
                ``emission_type="kyoto_gases"`` uses the GWP100 Kyoto Gases
                aggregate; ``emission_type="co2"`` uses direct CO2 pathways.
              - ``include_afolu``: Whether AFOLU is included inside the
                selected ``emission_type``. Defaults to ``False``.
              - ``emissions_mode``: Dynamic AR6 emissions mode. Accepted
                values are ``"net"``, ``"gross"``, and ``"gross_alt"``.
                Defaults to ``"gross_alt"``. Gross modes write positive
                emissions denominator rows and signed negative sequestration
                companion rows; downstream aCC and ASR consume only the
                denominator gross positive rows. ``"gross"`` removes all
                sequestration sources from net emissions. ``"gross_alt"``
                removes all sequestration sources except CCS, as it does not
                directly capture CO2 from the atmosphere; IPCC AR6 recommends
                treating CCS separately from net negative sequestration. See
                ``data_raw/methodological_notes/methodological_note__steady_state__dynamic_cc.pdf``
                for the methodological explanation.
              - ``subset_version``: Optional selector for a subset of AR6
                model-scenario pairs. Follow
                ``data_processed/ar6/<processed_scope>/README_model_scenario_subset.txt``
                to create the subset CSV.
        lca_args: LCA numerator route envelope. Exactly one route block must
            have ``active=True``. The default signature selects
            ``external_lca``. Set ``external_lca.active=False`` and
            ``io_lca.active=True`` to use IO-LCA generation.

            Nested keys:

            - ``external_lca``: External LCA route block.

              Nested keys:

              - ``active``: Whether staged external LCA files are used.
              - ``version_name``: External LCA version selected from staged
                files. Use ``prepare_external_inputs(...)`` to import the
                external LCA real input folders, README guidance, and runnable CSV
                examples, then follow the imported README guidance for version syntax and
                data input format.

            - ``io_lca``: IO-LCA generation route block.

              Nested key:

              - ``active``: Whether IO-LCA generation is used.
        uncertainty_config: Monte Carlo configuration dictionary. The default
            signature activates projection, reference year, and inter method
            uncertainty for the aSoCC denominator, and dynamic AR6 CC
            uncertainty for dynamic carrying capacity branches. LCIA
            uncertainty is inactive by default because L2 LCIA rows require
            ``sector_cov_mapping``: keys are output ``s_p`` labels and values
            are sector CoV codes from ``sec_cbca_covs.csv``. Inter MRIO
            uncertainty is inactive by default because it requires an
            alternate published disaggregated aSoCC source. IO-LCA LCIA
            uncertainty is inactive until requested. Source blocks
            use an ``active`` boolean; write ``active=False`` to disable a
            default active source. For external LCA Monte Carlo inputs, use
            ``prepare_external_inputs(...)`` and follow the imported external
            LCA README guidance and runnable examples. See
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

            - ``asocc_uncertainty_sources``: optional dictionary for aSoCC
              denominator sources. Write ``active=False`` to disable a default
              active source block.

              Nested source blocks:

              - ``lcia_uncertainty``: optional LCIA source block. It defaults
                to ``{"active": False, "sector_cov_mapping": {}}``. Country
                level LCIA CoVs are resolved automatically. L2 sector
                resolved LCIA rows require ``sector_cov_mapping`` to map
                output ``s_p`` labels to sector CoV codes from
                ``sec_cbca_covs.csv``, for example
                ``{"active": True, "sector_cov_mapping":
                {"Paper": "Paper"}}``. Carbon consumption based
                accounts coefficients of variation (CoV) files are available
                under
                ``data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/``.
                Users can inspect ``sec_cbca_covs.csv`` for sector CoV codes
                before choosing ``sector_cov_mapping`` values. CoV keys must
                match the LCIA uncertainty output domain. If
                ``group_reg=True``, region keys use
                ``reg_cbca_covs_group_<group_version>.csv``; otherwise they
                use ``reg_cbca_covs.csv``. If ``aggreg_indices=True``
                collapses a region axis, put the full aggregate region label
                in
                ``reg_cbca_covs_group_<group_version>_aggreg_indices.csv``
                when grouped, otherwise ``reg_cbca_covs_aggreg_indices.csv``.
                If ``group_sec=True`` or ``aggreg_indices=True``, use the
                corresponding grouped or aggregate ``s_p`` labels as
                ``sector_cov_mapping`` keys. For
                example, with ``aggreg_indices=True`` and ``s_p=["A", "B"]``,
                write ``sector_cov_mapping={"A, B": "Electricity"}`` when
                ``Electricity`` is the sector CoV code selected from
                ``sec_cbca_covs.csv``.

                Nested keys:

                - ``active``: Whether LCIA uncertainty is active.
                - ``sector_cov_mapping``: Mapping from output ``s_p`` labels
                  to sector CoV codes from ``sec_cbca_covs.csv``.

              - ``projection_uncertainty``: optional source block. It
                defaults to ``{"active": True}``. For prospective rows using
                L2 historical reuse, each Monte Carlo run samples one L2
                reuse year uniformly from the deterministic
                ``l2_reuse_years`` candidates requested for the years where
                reuse applies.

                Nested key:

                - ``active``: Whether projection uncertainty is active.

              - ``reference_year_uncertainty``: optional source block. It
                defaults to ``{"active": True}``. For acquired rights (AR)
                routes, each Monte Carlo run samples uniformly among
                requested reference years admissible for the studied year
                (``reference_year <= year``). The same sampled reference year
                is shared across the run when admissible; years for which it
                is not admissible resample among their admissible reference
                years.

                Nested key:

                - ``active``: Whether reference year uncertainty is active.

              - ``inter_mrio_uncertainty``: optional source block. To
                activate it, write ``{"active": True, "alternate_source":
                "<disaggregated label>"}``, for example
                ``{"active": True, "alternate_source": "oecd_electricity"}``.
                It applies continuous uniform interpolation between the main
                MRIO source and an alternate published disaggregated aSoCC
                source created by ``disaggregate_asocc(...)``. It applies
                only to non LCIA methods.

                Nested keys:

                - ``active``: Whether inter MRIO uncertainty is active.
                - ``alternate_source``: Published disaggregated aSoCC source
                  label used as the alternate MRIO source.

              - ``inter_method_uncertainty``: optional source block. It
                defaults to ``{"active": True, "mode": "equal_weight"}``.
                Each Monte Carlo run samples one method leaf among the
                selected deterministic and external methods. Equal weight
                mode writes the tree CSV, README, and rendered probability
                tree under the run folder ``figures/inter_method_tree/``. To
                prepare custom weights before running uncertainty, use
                ``write_asocc_weight_template(...)``; it writes
                ``equal_weights.csv``, ``README_inter_method_weights.txt``,
                and ``probability_tree__equal_weights.<ext>`` under
                ``B1_asocc/preview_inter_method_weights/``. Use
                ``preview_asocc_weight_tree(...)`` to validate and render a
                custom probability tree before using
                ``{"mode": "custom", "version_name": "..."}``.

                Nested keys:

                - ``active``: Whether inter method uncertainty is active.
                - ``mode``: Inter method sampling mode. Accepted values are
                  ``"equal_weight"`` and ``"custom"``.
                - ``version_name``: Custom weight version used when
                  ``mode="custom"``.

            - ``ar6_cc_uncertainty_sources``: optional dictionary for dynamic
              AR6 CC denominator sources. Write ``active=False`` to disable a
              default active source block.

              Nested source block:

              - ``dynamic_ar6_cc_uncertainty``: optional AR6 CC source block.
                It defaults to ``{"active": True, "sampling_method": "srs",
                "category_uncertainty": False}``. ``sampling_method`` accepts
                ``"srs"`` for simple random sampling (samples across retained
                model-scenario pairs matching the requested category and SSP)
                or ``"lhs"`` for Latin hypercube sampling (samples among
                retained models first, then among retained scenarios for the
                selected model, category, and SSP to limit over representation
                of models with more AR6 submissions). The effect of this
                choice is visible in ``process_ar6(...)`` sampling diagnostic
                figures. ``category_uncertainty`` is inactive by default. If
                ``True``, each Monte Carlo run first samples one retained AR6
                category with equal probability in the studied SSP pool. It
                then applies ``sampling_method`` inside that selected
                category; with ``sampling_method="lhs"``, this means model
                first, then scenario inside that model.

                Nested keys:

                - ``active``: Whether dynamic AR6 carrying capacity
                  uncertainty is active.
                - ``sampling_method``: AR6 pathway sampling method. Accepted
                  values are ``"srs"`` and ``"lhs"``.
                - ``category_uncertainty``: Whether each Monte Carlo run
                  samples one retained AR6 category before applying
                  ``sampling_method`` inside that category.

            - ``io_lca_uncertainty_sources``: optional dictionary for package
              managed IO-LCA numerator sources.

              Nested source block:

              - ``lcia_uncertainty``: optional LCIA source block. It defaults
                to ``{"active": False, "sector_cov_mapping": {}}``. Country
                level LCIA CoVs are resolved automatically. L2 sector
                resolved LCIA rows require ``sector_cov_mapping`` to map
                output ``s_p`` labels to sector CoV codes from
                ``sec_cbca_covs.csv``, for example
                ``{"active": True, "sector_cov_mapping":
                {"Paper": "Paper"}}``. Carbon consumption based
                accounts coefficients of variation (CoV) files are available
                under
                ``data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/``.
                Users can inspect ``sec_cbca_covs.csv`` for sector CoV codes
                before choosing ``sector_cov_mapping`` values. CoV keys must
                match the LCIA uncertainty output domain. If
                ``group_reg=True``, region keys use
                ``reg_cbca_covs_group_<group_version>.csv``; otherwise they
                use ``reg_cbca_covs.csv``. If ``aggreg_indices=True``
                collapses a region axis, put the full aggregate region label
                in
                ``reg_cbca_covs_group_<group_version>_aggreg_indices.csv``
                when grouped, otherwise ``reg_cbca_covs_aggreg_indices.csv``.
                If ``group_sec=True`` or ``aggreg_indices=True``, use the
                corresponding grouped or aggregate ``s_p`` labels as
                ``sector_cov_mapping`` keys. For
                example, with ``aggreg_indices=True`` and ``s_p=["A", "B"]``,
                write ``sector_cov_mapping={"A, B": "Electricity"}`` when
                ``Electricity`` is the sector CoV code selected from
                ``sec_cbca_covs.csv``.

                Nested keys:

                - ``active``: Whether LCIA uncertainty is active.
                - ``sector_cov_mapping``: Mapping from output ``s_p`` labels
                  to sector CoV codes from ``sec_cbca_covs.csv``.
        s_p: Producing sector filter(s), single string or list. If this is a
            required axis for ``fu_code`` and the argument is omitted, the run
            expands to all valid producing sectors. To identify valid sector
            names, see the first column of the relevant
            ``data_raw/mrio/.../grouping/.../group_sec_template.csv`` file. For
            EXIOBASE sector definitions, see
            ``data_raw/mrio/exiobase_3/sector_classification.xlsx``; EXIOBASE
            ixi and pxp use different sector lists.
        r_p: Producing region filter(s), single string or list. If this is a
            required axis for ``fu_code`` and the argument is omitted, the run
            expands to all valid producing regions. To identify valid region
            names, see the first column of the relevant
            ``data_raw/mrio/.../grouping/group_reg_template.csv`` file.
        r_c: Consuming region filter(s), single string or list. If this is a
            required axis for ``fu_code`` and the argument is omitted, the run
            expands to all valid consuming regions. To identify valid region
            names, see the first column of the relevant
            ``data_raw/mrio/.../grouping/group_reg_template.csv`` file.
        r_f: Final demand region filter(s), single string or list. If this is
            a required axis for ``fu_code`` and the argument is omitted, the
            run expands to all valid final demand regions. To identify valid
            region names, see the first column of the relevant
            ``data_raw/mrio/.../grouping/group_reg_template.csv`` file.
        base_asocc_args: Optional aSoCC denominator envelope. Write nested
            arguments as ``base_asocc_args={"method_plan": "default"}``.
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
            interpretation. The default has ``active=True`` and runs Sobol in
            convergence mode. Sobol base sizes must be powers of two.

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

            - ``sobol_years``: Studied output years evaluated by Sobol. When
              omitted, Sobol evaluates only the first and last studied years
              in the requested studied year set.
        figures: Whether to render figures.
            Default is ``True``.
        figure_options: ASR figure options. Defaults to
            ``{"per_method": True, "multi_method": True, "inter_method": True,``
            ``"polar": {"active": True, "polar_years": None,``
            ``"polar_style": "violin"}}``.

            Nested keys:

            - ``per_method``: Whether to render method specific figures, with
              one separate figure for each allocation method.
            - ``multi_method``: Whether to render cross method comparison
              figures, with multiple allocation methods shown in the same
              figure.
            - ``inter_method``: Whether to render inter method uncertainty
              figures. These figures use the same method specific layout as
              ``per_method``, but represent uncertainty induced by the inter
              method uncertainty setting rather than comparing individual
              allocation methods. This option is ignored when inter method
              uncertainty is inactive.
            - ``polar``: Nested polar figure selector. The block is optional
              and defaults to ``{"active": True, "polar_years": None,
              "polar_style": "violin"}``. Its ``active`` key enables polar
              figures and defaults to ``True``. Its ``polar_years`` key
              selects studied output years evaluated by ASR polar figures;
              when omitted, polar figures use only the first and last studied
              years in the requested studied year set. Its ``polar_style`` key
              accepts ``"violin"``, ``"whisker"``, and ``"both"``.

        figure_format: Figure render settings mapping. Defaults to
            ``{"format": "png", "dpi": 500}``.

            Nested keys:

            - ``format``: Figure file format. Accepted values are ``"png"``,
              ``"pdf"``, and ``"svg"``.
            - ``dpi``: Positive integer figure resolution used for raster
              outputs.
        subfigures: Whether prerequisite aCC and IO-LCA uncertainty calls
            render their own figures when ``figures=True``.
            Default is ``True``.

        refresh: If ``True``, refresh the resolved ASR Monte Carlo outputs and
            every upstream component output scope called by this ASR
            uncertainty request. This can refresh the matching
            ``uncertainty_acc(...)`` output scope and its prerequisites,
            including aSoCC output scopes and, when dynamic AR6 CC is used,
            the matching processed AR6 output scope selected by
            ``process_ar6(...)``, the matching
            ``deterministic_ar6_cc(...)`` output scope, and, when dynamic AR6
            CC uncertainty is active, the matching
            ``uncertainty_ar6_cc(...)`` output scope. When the numerator route
            is IO-LCA with LCIA uncertainty active, the matching
            ``deterministic_io_lca(...)`` and ``uncertainty_io_lca(...)``
            output scopes can also be refreshed. For example, matching ASR
            Monte Carlo run folders are refreshed under
            ``<repo>/demo/C_asr/exiobase_3102_ixi__elec/external_lca__test_v1/monte_carlo/mc_<generated_id>``.
            External LCA staged inputs, processed MRIO inputs, processed
            population and GDP, raw downloads, and deterministic ASR outputs
            are not refreshed. Defaults to ``False``.

    Returns:
        UncertaintyRunReport describing ASR uncertainty table outputs and
        figure outputs when figures are requested.

    Raises:
        ValueError: If the request, source configuration, or persisted input
            files are inconsistent.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

    Example:
        Run static ASR for ``L2.c.b`` producing sector ``Paper`` and
        consuming region ``FR`` with IO-LCA and LCIA uncertainty added, using
        defaults where omitted::

            from pyaesa import uncertainty_asr

            uncertainty_asr(
                project_name="demo",
                source="exiobase_3102_ixi",
                years=range(2020, 2031),
                lcia_method="gwp100_lcia",
                fu_code="L2.c.b",
                s_p=["Paper"],
                r_c=["FR"],
                lca_args={
                    "external_lca": {"active": False, "version_name": None},
                    "io_lca": {"active": True},
                },
                uncertainty_config={
                    "asocc_uncertainty_sources": {
                        "lcia_uncertainty": {
                            "active": True,
                            "sector_cov_mapping": {"Paper": "Paper"},
                        },
                    },
                    "io_lca_uncertainty_sources": {
                        "lcia_uncertainty": {
                            "active": True,
                            "sector_cov_mapping": {"Paper": "Paper"},
                        },
                    },
                },
            )
    """
    subfigures_effective = all((figures, subfigures))
    return run_uncertainty_asr(
        project_name=project_name,
        source=source,
        group_reg=group_reg,
        group_sec=group_sec,
        group_version=group_version,
        years=years,
        fu_code=fu_code,
        r_p=r_p,
        s_p=s_p,
        r_c=r_c,
        r_f=r_f,
        aggreg_indices=aggreg_indices,
        lcia_method=lcia_method,
        base_asocc_args=base_asocc_args,
        external_method=external_method,
        base_cc_args=base_cc_args,
        lca_args=lca_args,
        uncertainty_config=uncertainty_config,
        sobol_parameters=sobol_parameters,
        output_format=output_format,
        figures=figures,
        figure_options=figure_options,
        figure_format=figure_format,
        subfigures=subfigures_effective,
        refresh=refresh,
    )
