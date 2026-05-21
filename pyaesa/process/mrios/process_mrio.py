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
    group_reg: bool = False,
    group_sec: bool = False,
    group_version: str = "",
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
            years for the selected source/group version are used.
        refresh: If ``True``, clear and recompute only the requested processed
            MRIO year folders inside the resolved source and classification
            output scope. The output scope is
            ``data_processed/mrio/<source>/<version_tag>``, where
            ``version_tag`` is ``original_classification`` for ungrouped
            processing or ``custom_classification_<group_version>`` for
            grouped processing. For each requested year, the corresponding
            processed year folder and metadata year entry are removed before
            recomputation. Raw downloads and project outputs are not refreshed.
            Defaults to ``False``.
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
        ValueError: If source, grouping, or LCIA arguments are invalid for the
            requested public processing scope.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function. Run ``download_mrio(...)`` for the same source and years
        before processing when the raw archives are not already present.
        The workspace prerequisite folders used by grouping and EXIOBASE LCIA
        extension are documented in local README guides under ``data_raw``.
        Use ``README_grouping.txt`` inside the active
        MRIO grouping folder for custom region or sector grouping, and use
        ``README_add_custom_lcia_characterization_matrices.txt`` plus
        ``README_add_custom_lcia_responsibility_periods.txt`` inside the
        EXIOBASE LCIA prerequisite folders when adding custom method specific
        files beyond the package shipped methods. If grouped MRIO outputs will
        later be used with LCIA uncertainty, follow
        ``README_grouped_and_aggregate_lcia_covs.txt`` in
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
        group_reg=group_reg,
        group_sec=group_sec,
        group_version=group_version,
        keep_intermediate_uncasext=keep_intermediate_uncasext,
        pymrio_calc_all=pymrio_calc_all,
    )
