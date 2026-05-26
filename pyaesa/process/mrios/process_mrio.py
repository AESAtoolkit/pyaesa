"""Public entry point for MRIO processing."""

from typing import Optional, Sequence

from pyaesa.download.mrios.utils.year_selection import YearSelection
from pyaesa.process.mrios.utils.pipeline.contracts import ProcessReportMRIO
from pyaesa.process.mrios.utils.pipeline.runner import run_process_mrio


def process_mrio(
    source: str,
    years: YearSelection = None,
    *,
    refresh: bool = False,
    lcia_method: Optional[str | Sequence[str]] = None,
    agg_reg: bool = False,
    agg_sec: bool = False,
    agg_version: str = "",
    keep_intermediate_uncasext: bool = False,
    pymrio_calc_all: bool = False,
) -> ProcessReportMRIO | None:
    """Process MRIO archives into ``data_processed`` folder for selected years.

    Raw MRIO archives produced by ``download_mrio(...)`` must already exist on
    disk in the active workspace. This function reads those archives and does not
    download missing MRIO data. Omit arguments to use their default.

    Args:
        source: MRIO source key (``"exiobase_396_ixi"``,
            ``"exiobase_396_pxp"``, ``"exiobase_3102_ixi"``,
            ``"exiobase_3102_pxp"``, or ``"oecd_v2025"``).
        years: Studied years. Accepts a single year, list, or range. If
            omitted, all available MRIO
            years for the selected source and ``agg_version`` are used.
        refresh: If ``True``, clear and recompute only the requested processed
            MRIO year folders inside the resolved source and classification
            output scope. The output scope is
            ``data_processed/mrio/<source>/<version_tag>``, where
            ``version_tag`` is ``original_classification`` for native source
            classification or ``custom_classification_<agg_version>`` for
            custom MRIO aggregation and disaggregation processing. For each requested
            year, the corresponding processed year folder and metadata year
            entry are removed before recomputation. Raw downloads and project
            outputs are not refreshed. Defaults to ``False``.
        lcia_method: LCIA method(s) used to characterize MRIO
            environmental stressors into the selected method(s) impact
            categories (for example ``"pb_lcia"`` or
            ``["pb_lcia", "gwp100_lcia"]``). ``None`` skips LCIA
            characterization. Defaults to ``None``. pyaesa currently supports
            LCIA characterization only for EXIOBASE sources. To add a custom
            LCIA method, follow
            ``README_add_custom_lcia_characterization_matrices.txt`` in
            ``data_raw/mrio/exiobase_3/lcia/characterization_factors_matrices/``
            and pass the custom method file stem here.
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
        keep_intermediate_uncasext: If ``True``, keep intermediate UNCASExt
            matrices needed only for optional diagnostic upstream supply chain
            decomposition in
            ``deterministic_io_lca(..., upstream_analysis=True)``.
            Written files are the post clip core matrices (``A``, ``G``, ``L``,
            ``Z``, ``unit``), plus characterized LCIA ``extensions/`` payloads.
            Default ``False`` writes only the required processed outputs and
            avoids saving optional intermediate outputs.
        pymrio_calc_all: If ``True``, write the optional PyMRIO function
            ``calc_all`` diagnostic outputs. These outputs are not used by any
            downstream pyaesa public function. The written payload is PyMRIO
            ``calc_all`` on original matrices without clipping negative values,
            stored under ``preclip/`` and ``preclip/extensions/``. Default
            ``False`` skips this diagnostic payload.

    Returns:
        A :class:`ProcessReportMRIO` when something was processed or failed,
        otherwise ``None`` when all requested years were already satisfied.

    Raises:
        ValueError: If source, aggregation, or LCIA arguments are invalid for the
            requested public processing scope.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function. Run ``download_mrio(...)`` for the same source and years
        before processing when the raw archives are not already present.
        The workspace prerequisite folders used by MRIO aggregation and disaggregation
        and EXIOBASE LCIA extension are documented in local README guides under
        ``data_raw``.
        Use ``README_aggregation.txt`` inside the active
        MRIO aggregation folder for custom region or sector
        MRIO aggregation and disaggregation, and use
        ``README_add_custom_lcia_characterization_matrices.txt`` plus
        ``README_add_custom_lcia_responsibility_periods.txt`` inside the
        EXIOBASE LCIA prerequisite folders when adding custom method specific
        files beyond the package shipped methods. If custom regional MRIO
        aggregation and disaggregation outputs will later be used with LCIA
        uncertainty, follow
        ``README_agg_reg_and_group_indices_lcia_covs.txt`` in
        ``data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/``.

    Example:
        Process EXIOBASE 3.10.2 ixi outputs with selected LCIA methods::

            from pyaesa import process_mrio

            process_mrio(
                source="exiobase_3102_ixi",
                lcia_method=["pb_lcia","gwp100_lcia"],
            )
    """
    return run_process_mrio(
        source=source,
        years=years,
        refresh=refresh,
        lcia_method=lcia_method,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
        keep_intermediate_uncasext=keep_intermediate_uncasext,
        pymrio_calc_all=pymrio_calc_all,
    )
