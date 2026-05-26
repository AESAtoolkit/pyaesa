"""Public dynamic AR6 carrying capacity (CC) uncertainty entrypoint."""

from typing import Any

from pyaesa.ar6_cc.uncertainty.runner import run_uncertainty_ar6_cc
from pyaesa.shared.uncertainty_assessment.run_state.report import UncertaintyRunReport


def uncertainty_ar6_cc(
    *,
    base_ar6_cc_args: dict[str, Any] = {
        "years": None,
        "harmonization": True,
        "harmonization_method": "offset",
        "category": ["C1", "C2", "C3", "C4"],
        "ssp_scenario": ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
        "emission_type": "kyoto_gases",
        "include_afolu": False,
        "emissions_mode": "gross_alt",
        "subset_version": None,
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
        "dynamic_ar6_cc_uncertainty": {
            "active": True,
            "sampling_method": "srs",
            "category_uncertainty": False,
        },
    },
    output_format: str = "csv_compact",
    figures: bool = True,
    figure_format: dict[str, Any] = {"format": "png", "dpi": 500},
    refresh: bool = False,
) -> UncertaintyRunReport:
    """Run dynamic AR6 carrying capacity (CC) Monte Carlo uncertainty.

    The function loads the deterministic AR6 CC row universe identified by
    ``base_ar6_cc_args`` and samples only the uncertainty sources requested in
    ``uncertainty_config``. Deterministic prerequisite outputs are created or
    reused through ``deterministic_ar6_cc(...)``. It writes run values,
    summary statistics, uncertainty source parameters, and figures when
    requested under the AR6 CC Monte Carlo output folder.
    Omit arguments to use their default.

    Args:
        base_ar6_cc_args: Deterministic AR6 CC selector envelope written as a
            dictionary. Required key: ``years``. Accepted optional keys are
            ``harmonization``, ``harmonization_method``, ``category``,
            ``ssp_scenario``, ``emission_type``, ``include_afolu``,
            ``emissions_mode``, and ``subset_version``.

            Nested keys:

            - ``years``: Study year selector provided as a consecutive year
              list or ``range(start_year, end_year + 1)``. The resolved years
              must contain at least two consecutive years with no gaps.
            - ``harmonization``: Whether to harmonize retained AR6 pathways
              to the historical baseline. Defaults to ``True``.
            - ``harmonization_method``: Harmonization method applied only when
              ``harmonization=True``. Defaults to ``"offset"``. The only
              supported value is currently ``"offset"``.
              Ignored when ``harmonization=False``.
            - ``category``: AR6 category classification for global warming
              trajectories, as a string or list, such as ``"C3"`` or
              ``["C1", "C2"]``. Valid values are ``"C1"`` through ``"C8"``.
              Defaults to ``["C1", "C2", "C3", "C4"]``, the categories
              aligned with the 2015 Paris Agreement.
            - ``ssp_scenario``: SSP label selector as a string, list, or
              ``None``. Defaults to
              ``["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]``.
            - ``emission_type``: Dynamic AR6 emission type. Accepted values
              are ``"kyoto_gases"`` (default) and ``"co2"``.
              ``emission_type="kyoto_gases"`` uses the GWP100 Kyoto Gases
              aggregate; ``emission_type="co2"`` uses direct CO2 pathways.
            - ``include_afolu``: Whether AFOLU emissions are included inside
              the selected ``emission_type``. Defaults to ``False``.
            - ``emissions_mode``: Dynamic AR6 emissions mode. Accepted values
              are ``"net"``, ``"gross"``, and ``"gross_alt"``. Defaults to
              ``"gross_alt"``. ``"net"`` uses net AR6 emissions pathways
              directly. ``"gross"`` removes all sequestration sources from
              net emissions. ``"gross_alt"`` removes all sequestration
              sources except CCS. CCS is retained because IPCC AR6 treats CCS
              as capture at fossil or industrial point sources rather than
              direct removal of CO2 from the atmosphere, so it is kept
              separate from net negative sequestration. Gross modes write
              positive emissions rows and signed negative sequestration
              companion rows; downstream aCC and ASR consume only the positive
              emissions rows. See
              ``data_raw/methodological_notes/methodological_note__steady_state__dynamic_cc.pdf``
              for the methodological explanation.
            - ``subset_version``: Optional selector for a subset of AR6
              model-scenario pairs. Follow
              ``data_processed/ar6/<processed_scope>/README_model_scenario_subset.txt``
              to create the subset CSV. Defaults to ``None``.
        uncertainty_config: Monte Carlo configuration dictionary. The default
            signature activates dynamic AR6 CC uncertainty. Category
            uncertainty is inactive by default. Source blocks use an
            ``active`` boolean; write ``active=False`` to disable dynamic AR6
            CC uncertainty. See
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
        refresh: If ``True``, refresh the matching processed AR6 output scope
            selected by ``process_ar6(...)``, the resolved deterministic AR6
            CC output scope, and the resolved AR6 CC Monte Carlo outputs for
            this uncertainty request. The Monte Carlo refresh removes
            matching run folders for the current request under the adjacent
            ``monte_carlo`` root of that deterministic output scope. For
            example, matching AR6 CC Monte Carlo run folders are refreshed
            under
            ``<repo>/data_processed/ar6/2019-2060_harmonization_offset/ar6_cc/gross_alt_kyoto_gases_wo_afolu/C1__SSP1/monte_carlo/mc_<generated_id>``.
            Raw downloads and downstream aCC or ASR outputs are not refreshed.
            Defaults to ``False``.

    Returns:
        UncertaintyRunReport describing AR6 CC uncertainty table outputs and
        figure outputs when figures are requested.

    Raises:
        ValueError: If the request, source configuration, or persisted input
            files are inconsistent.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.
        ``download_ar6(...)`` must have run before this function can read raw
        AR6 inputs through its deterministic AR6 CC prerequisite.
        Methodological details on AR6 scenario filtering, harmonization, and
        dynamic carrying capacity construction are provided in
        ``data_raw/methodological_notes/methodological_note__steady_state__dynamic_cc.pdf``.

    Example:
        Run dynamic AR6 CC with category uncertainty added, using defaults
        where omitted::

            from pyaesa import uncertainty_ar6_cc

            uncertainty_ar6_cc(
                base_ar6_cc_args={"years": range(2019, 2061)},
                uncertainty_config={
                    "dynamic_ar6_cc_uncertainty": {
                        "active": True,
                        "category_uncertainty": True,
                    },
                },
            )
    """
    return run_uncertainty_ar6_cc(
        base_ar6_cc_args=base_ar6_cc_args,
        uncertainty_config=uncertainty_config,
        output_format=output_format,
        figures=figures,
        figure_options=None,
        figure_format=figure_format,
        refresh=refresh,
    )
