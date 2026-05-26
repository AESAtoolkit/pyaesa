"""Public entrypoint for allocated shares of carrying capacities (aSoCC) disaggregation."""

from pyaesa.shared.runtime.reporting.composite_phase_index import (
    phase_ready_detail,
    phase_reused_detail,
)
from pyaesa.shared.runtime.reporting.phase import PhasePrinter

from .disaggregation.config import parse_disaggregate_args
from .disaggregation.models import DisaggregationReport
from .disaggregation.pipeline import run_disaggregation
from .io.logging import close_loggers_for_scope
from .runtime.scope.branch_resolution import outputs_project_root


def disaggregate_asocc(
    *,
    disaggregation_config: dict = {
        "target_agg_run": None,
        "ref_agg_run": None,
        "ref_disagg_run": None,
        "disaggregation_specs": None,
        "new_disagg_version_name": None,
    },
    base_asocc_args: dict = {
        "project_name": None,
        "years": None,
        "fu_code": None,
        "r_p": None,
        "r_c": None,
        "r_f": None,
        "group_indices": False,
        "method_plan": "default",
        "l1_methods": None,
        "one_step_methods": None,
        "two_step_methods": None,
        "l1_l2_pairs": None,
        "l1_reg_aggreg": "post",
        "ssp_scenario": ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
        "projection_mode": "regression",
        "reg_window": None,
        "l2_reuse_years": None,
    },
    output_format: str = "csv",
    figures: bool = True,
    figure_format: dict = {"format": "png", "dpi": 500},
    figure_external_method: dict[str, list[str]] | None = None,
    refresh: bool = False,
) -> DisaggregationReport:
    """Disaggregate non LCIA deterministic allocated shares of carrying capacities (aSoCC).

    The function creates a published disaggregated aSoCC source when a target
    source is available at an aggregated sector resolution. It uses a reference
    ixi MRIO available at both that same aggregated sector resolution and the
    requested detailed sector resolution to distribute each target aggregated
    sector across the detailed sectors. For every requested year ``y`` the
    allocated shares are equal to
    ``target_aggregated(y) * ref_disaggregated(y) / ref_aggregated(y)``.

    Only non LCIA aSoCC methods are supported. Do not pass ``lcia_method`` in
    ``base_asocc_args``. Supported sources are ``"oecd_v2025"``,
    ``"exiobase_3102_ixi"``, and ``"exiobase_396_ixi"`` because OECD ICIO is
    an ixi MRIO. It renders figures when requested.
    Omit arguments to use their default.

    Args:
        disaggregation_config: Required disaggregation envelope. Required keys
            are ``target_agg_run``, ``ref_agg_run``,
            ``ref_disagg_run``, ``disaggregation_specs``, and
            ``new_disagg_version_name``.

            Disaggregation configuration fields:

            - ``target_agg_run``: aggregated deterministic aSoCC source to
              disaggregate. Its published rows supply ``target_aggregated`` in
              the formula above. Example: OECD ICIO sector ``D``.
            - ``ref_agg_run``: reference ixi source aggregated to the same
              sector labels as ``target_agg_run``. Its published rows
              supply ``ref_aggregated``. Example: EXIOBASE ixi aggregated to OECD
              ICIO sector ``D``.
            - ``ref_disagg_run``: the same reference source as
              ``ref_agg_run``, but at the detailed disaggregated sector labels
              that should be written in the new source. Its published rows
              supply ``ref_disaggregated``. Example: EXIOBASE ixi electricity sectors.
            - ``disaggregation_specs``: mapping from each aggregated sector label
              to the detailed disaggregated sector label(s) that replace it in the new
              source.
            - ``new_disagg_version_name``: output source label used for
              the published disaggregated aSoCC source created by this
              function.

            Each ``*_run`` selector requires:

            - ``source``: supported MRIO source key for this transformation.
              Accepted values are ``"oecd_v2025"``, ``"exiobase_3102_ixi"``,
              and ``"exiobase_396_ixi"``. Only ixi MRIO layouts are supported.
            - ``s_p``: non empty list of sector labels.
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
            - ``fu_code``: Required functional unit code (for example
              ``"L1.a"``, ``"L2.c.b"``). See
              ``data_raw/methodological_notes/methodological_note__asocc_fus_allocation_methods.pdf``
              for all available functional unit codes and the system
              boundaries each represents. Disaggregation is defined only on
              L2 published outputs.
            - ``years``: Studied years. Accepts a single year, list, or
              range. If omitted, all available MRIO years for the selected
              source and ``agg_version`` are used.
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
        figure_external_method: Optional external deterministic aSoCC selector
            block used only for figure rendering. Use
            ``prepare_external_inputs(...)`` to import the external aSoCC
            README guidance and runnable CSV examples, then follow the
            imported guide for method syntax and data input format. This
            argument is valid only when ``figures=True``. Omit it to render only native
            deterministic aSoCC method rows. Defaults to ``None``.
        refresh: If ``True``, remove and rebuild only the published
            disaggregated aSoCC source created by this call. The cleared scope
            is the ``deterministic`` folder under
            ``<project>/B1_asocc/<new_disagg_version_name>``. For
            example, for ``project_name="demo"`` and
            ``new_disagg_version_name="oecd_electricity"``, the
            refreshed path is
            ``<repo>/demo/B1_asocc/oecd_electricity/deterministic``. The
            deterministic prerequisite scopes named in ``target_agg_run``,
            ``ref_agg_run``, and ``ref_disagg_run`` are not refreshed.
            Processed MRIO inputs, processed population and GDP, raw
            downloads, and downstream aCC or ASR outputs are not refreshed.
            Defaults to ``False``.

    Returns:
        DisaggregationReport describing disaggregated aSoCC table outputs and
        figure outputs when figures are requested.

    Raises:
        ValueError: If the configuration is invalid, one prerequisite
            deterministic scope is unavailable, requested published output
            coverage is missing, or one strict disaggregation failure rule is
            triggered.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.

        - Region labels are matched strictly between the MRIO sources.
          Studied regions requested through ``r_p``/``r_c``/``r_f`` must use
          the same labels in all selected sources. Use region aggregation to
          update region names syntax when they do not already match.
        - Disaggregation may run for any requested year whose prerequisite
          published outputs exist for all three selectors.

    Example:
        Disaggregate OECD ICIO sector D into EXIOBASE electricity sector,
        using defaults where omitted::

            disaggregate_asocc(
                disaggregation_config={
                    "target_agg_run": {
                        "source": "oecd_v2025",
                        "s_p": ["D"],
                    },
                    "ref_agg_run": {
                        "source": "exiobase_3102_ixi",
                        "agg_sec": True,
                        "agg_version": "oecd_d",
                        "s_p": ["D"],
                    },
                    "ref_disagg_run": {
                        "source": "exiobase_3102_ixi",
                        "agg_sec": True,
                        "agg_version": "elec",
                        "s_p": ["Electricity"],
                    },
                    "disaggregation_specs": [
                        {
                            "agg_sector_label": "D",
                            "disagg_sector_label": "Electricity",
                        }
                    ],
                    "new_disagg_version_name": "disagg_oecd_elec",
                },
                base_asocc_args={
                    "project_name": "demo",
                    "fu_code": "L2.c.b",
                    "years": range(2005, 2031),
                    "r_c": ["FR", "US"],
                    "ssp_scenario": ["SSP2", "SSP3"],
                },
            )
    """
    phase = PhasePrinter("disaggregate_asocc")
    project_root = None
    try:
        phase.announce("Phase B.1: aSoCC", "disaggregate_asocc")
        parsed = parse_disaggregate_args(
            disaggregation_config=disaggregation_config,
            base_allocate_args=base_asocc_args,
            output_format=output_format,
            figures=figures,
            figure_options=None,
            figure_format=figure_format,
            figure_external_method=figure_external_method,
            refresh=refresh,
        )
        project_root = outputs_project_root(project_name=parsed.base_allocate_args["project_name"])
        report = run_disaggregation(parsed, phase=phase)
        output_root = report.output_root()
        detail = (
            phase_reused_detail if report.reuse_status() == "reused_exact" else phase_ready_detail
        )
        phase.complete(
            detail(scope_name="disaggregate_asocc", output_root=output_root),
            owner="disaggregate_asocc",
        )
        return report
    finally:
        phase.finish()
        if project_root is not None:
            close_loggers_for_scope(project_root)
