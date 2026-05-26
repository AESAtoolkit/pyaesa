"""Public IO-LCA uncertainty entrypoint."""

from typing import Any

from pyaesa.io_lca.uncertainty.runner import run_uncertainty_io_lca
from pyaesa.shared.uncertainty_assessment.run_state.report import UncertaintyRunReport


def uncertainty_io_lca(
    *,
    base_io_lca_args: dict[str, Any] = {
        "project_name": None,
        "source": None,
        "agg_reg": False,
        "agg_sec": False,
        "agg_version": "",
        "years": None,
        "lcia_method": None,
        "fu_code": None,
        "s_p": None,
        "r_p": None,
        "r_c": None,
        "r_f": None,
        "group_indices": False,
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
    },
    output_format: str = "csv_compact",
    figures: bool = True,
    figure_format: dict[str, Any] = {"format": "png", "dpi": 500},
    refresh: bool = False,
) -> UncertaintyRunReport:
    """Run IO-LCA Monte Carlo uncertainty from deterministic IO-LCA outputs.

    The function loads deterministic IO-LCA outputs and samples only the
    uncertainty sources requested in ``uncertainty_config`` (pyaesa currently
    only supports LCIA uncertainty). Deterministic prerequisite outputs are
    created or reused through ``deterministic_io_lca(...)``. It writes run
    values, summary statistics, uncertainty source parameters, and figures
    when requested under the IO-LCA Monte Carlo output folder.
    Omit arguments to use their default.

    Args:
        base_io_lca_args: Deterministic IO-LCA selector envelope.
            Write nested arguments as ``base_io_lca_args={"project_name":
            "...", "source": "...", "lcia_method": "...", "fu_code":
            "..."}``. Required keys are ``project_name``, ``source``,
            ``lcia_method``, and ``fu_code``.

            Nested keys:

            - ``project_name``: Required project name used to build
              ``<repo>/<project_name>``.
            - ``source``: MRIO source key (``"exiobase_396_ixi"``,
              ``"exiobase_396_pxp"``, ``"exiobase_3102_ixi"``,
              ``"exiobase_3102_pxp"``). pyaesa currently only supports
              EXIOBASE for LCIA characterization.
            - ``agg_reg``: If ``True``, reclassify MRIO regions with the
              ``agg_reg_<agg_version>.csv`` MRIO aggregation and disaggregation mapping.
              The mapping can keep native labels, aggregate several native regions
              into one target label, or disaggregate one native region across
              several target labels when a ``weight`` column is provided.
              Default ``False`` keeps native source regions.
            - ``agg_sec``: If ``True``, reclassify MRIO sectors with the
              ``agg_sec_<agg_version>.csv`` MRIO aggregation and disaggregation mapping.
              The mapping can keep native labels, aggregate several native sectors
              into one target label, or disaggregate one native sector across
              several target labels when a ``weight`` column is provided.
              Default ``False`` keeps native source sectors.
            - ``agg_version``: Name token used to resolve the matching
              ``agg_reg_<agg_version>.csv`` and/or
              ``agg_sec_<agg_version>.csv`` MRIO aggregation and disaggregation
              mapping files in ``data_raw/mrio/<source>/aggregation``.
              Required when ``agg_reg`` or ``agg_sec`` is True. Defaults to
              an empty string for native source classification. Use the same
              token in downstream calls that should reuse the processed
              classification. When a mapping file has a ``weight``
              column, weights must sum to ``1`` for each original label.
            - ``years``: Studied years. Accepts a single year, list, or
              range. If omitted, all available MRIO years for the selected
              source and ``agg_version`` are used.
            - ``lcia_method``: Required LCIA method(s) selected for IO-LCA
              results (for example ``"pb_lcia"`` or
              ``["pb_lcia", "gwp100_lcia"]``). The method(s) must have been
              processed for the same MRIO source with ``process_mrio(...)``.
              pyaesa currently supports IO-LCA only for EXIOBASE sources. To
              add a custom LCIA method with which run ``process_mrio(...)``,
              follow
              ``README_add_custom_lcia_characterization_matrices.txt`` in
              ``data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/``
              and pass the custom method file stem here.
            - ``fu_code``: Required functional unit code (for example
              ``"L1.a"``, ``"L2.c.b"``). See
              ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
              for all available functional unit codes and the system
              boundaries each represents.
            - ``s_p``: Producing sector filter(s), single string or list. If
              this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid producing sectors. To
              identify valid sector names, see the first column of the
              relevant
              ``data_raw/mrio/.../aggregation/.../agg_sec_template.csv`` file.
              For EXIOBASE sector definitions, see
              ``data_raw/mrio/exiobase_3/sector_classification.xlsx``;
              EXIOBASE ixi and pxp use different sector lists.
            - ``r_p``: Producing region filter(s), single string or list. If
              this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid producing regions. To
              identify valid region names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/agg_reg_template.csv``
              file.
            - ``r_c``: Consuming region filter(s), single string or list. If
              this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid consuming regions. To
              identify valid region names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/agg_reg_template.csv``
              file.
            - ``r_f``: Final demand region filter(s), single string or list.
              If this is a required axis for ``fu_code`` and the argument is
              omitted, the run expands to all valid final demand regions. To
              identify valid region names, see the first column of the
              relevant ``data_raw/mrio/.../aggregation/agg_reg_template.csv``
              file.
            - ``group_indices``: Whether multiple selected region or sector
              filter values are kept as separate result rows or summed into one
              result row after the function calculation has been performed.
              - ``False`` (default): keep selected values as independent rows.
              - ``True``: sum selected values into one result row.
              The function refuses to run when ``group_indices=True`` is used
              with ``L2.a.b``, ``L2.b.b``, or ``L2.c.b`` because summing output
              rows for CBA total demand boundaries can double count. For these
              functional units, change the upstream MRIO aggregation and disaggregation
              scope with ``agg_reg``, ``agg_sec``, and ``agg_version`` before
              running the study.
        uncertainty_config: Monte Carlo configuration dictionary. It must
            activate ``lcia_uncertainty``, which is the IO-LCA public
            uncertainty source. LCIA uncertainty is inactive by default
            because L2 LCIA rows require ``sector_cov_mapping``: keys are
            output ``s_p`` labels and values are sector CoV codes from
            ``sec_cbca_covs.csv``. Source blocks use an ``active`` boolean. See
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
        output_format: Public uncertainty table format, either
            ``"csv_compact"`` or ``"parquet"``. Defaults to
            ``"csv_compact"``.
        figures: Whether to render figures.
            Default is ``True``.
        figure_format: Figure render settings mapping. Defaults to
            ``{"format": "png", "dpi": 500}``.

            Nested keys:

            - ``format``: Figure file format. Accepted values are ``"png"``,
              ``"pdf"``, and ``"svg"``.
            - ``dpi``: Positive integer figure resolution used for raster
              outputs.
        refresh: If ``True``, refresh both the resolved deterministic IO-LCA
            prerequisite and the resolved IO-LCA Monte Carlo outputs for this
            uncertainty request. The deterministic refresh clears the selected
            ``deterministic_io_lca(...)`` source and version output scope
            under ``<project>/A_lca/io_lca``. The Monte Carlo refresh removes
            matching run folders for the current request under the adjacent
            ``monte_carlo`` root. For example, matching IO-LCA Monte Carlo run
            folders are refreshed under
            ``<repo>/demo/A_lca/io_lca/exiobase_3102_ixi__elec/monte_carlo/mc_<generated_id>``.
            Processed MRIO inputs, processed population and GDP, raw downloads,
            and downstream ASR outputs are not refreshed. Defaults to
            ``False``.

    Returns:
        UncertaintyRunReport describing IO-LCA uncertainty table outputs and
        figure outputs when figures are requested.

    Raises:
        ValueError: If the request, source configuration, or persisted input
            files are inconsistent.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

    Example:
        Run IO-LCA for ``L2.c.b`` producing sector ``Paper`` and consuming
        region ``FR`` with LCIA uncertainty added, using defaults where
        omitted::

            from pyaesa import uncertainty_io_lca

            uncertainty_io_lca(
                base_io_lca_args={
                    "project_name": "demo",
                    "source": "exiobase_3102_ixi",
                    "years": 2019,
                    "lcia_method": "gwp100_lcia",
                    "fu_code": "L2.c.b",
                    "s_p": ["Paper"],
                    "r_c": ["FR"],
                },
                uncertainty_config={
                    "lcia_uncertainty": {
                        "active": True,
                        "sector_cov_mapping": {"Paper": "Paper"},
                    },
                },
            )
    """
    return run_uncertainty_io_lca(
        base_io_lca_args=base_io_lca_args,
        uncertainty_config=uncertainty_config,
        output_format=output_format,
        figures=figures,
        figure_options=None,
        figure_format=figure_format,
        refresh=refresh,
    )
