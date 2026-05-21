"""Public entrypoint for deterministic dynamic AR6 carrying capacity (CC)."""

from pathlib import Path

from pyaesa.shared.figures.request_validation import normalize_figure_format
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_B1_AR6_DYNAMIC_CC,
    phase_ready_detail,
    phase_reused_detail,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter

from .deterministic.runner import run_deterministic_ar6_cc
from .deterministic.runtime.reports import ComputeAR6CCReport


def deterministic_ar6_cc(
    *,
    years: list[int] | range,
    harmonization: bool = True,
    harmonization_method: str = "offset",
    category: str | list[str] = ["C1", "C2", "C3", "C4"],
    ssp_scenario: str | list[str] = [
        "SSP1",
        "SSP2",
        "SSP3",
        "SSP4",
        "SSP5",
    ],
    emission_type: str = "kyoto_gases",
    include_afolu: bool = False,
    emissions_mode: str = "gross_alt",
    subset_version: str | None = None,
    output_format: str = "csv",
    figures: bool = True,
    figure_format: dict[str, object] = {"format": "png", "dpi": 500},
    refresh: bool = False,
    _status: PhasePrinter | NullPhasePrinter | None = None,
) -> ComputeAR6CCReport:
    """Extract AR6 climate change pathways from ``process_ar6(...)`` outputs.

    The function reads processed AR6 pathways and extracts the
    selected emission variable and years. Matching
    processed AR6 outputs are created or reused through
    ``process_ar6(...)``. It writes deterministic dynamic AR6 carrying
    capacity (CC) tables and renders figures when requested.
    Omit arguments to use their default.

    Args:
        years: Study year selector provided as a consecutive year
            list or ``range(start_year, end_year + 1)``. The resolved years
            must contain at least two consecutive years with no gaps.
        harmonization: Whether to harmonize retained AR6 pathways to the
            historical baseline. Defaults to ``True``.
        harmonization_method: Harmonization method applied only when
            ``harmonization=True``. Defaults to ``"offset"``. The only
            supported value is currently ``"offset"``.
            Ignored when ``harmonization=False``.
        category: AR6 category classification filter for global warming
            trajectories. Defaults to ``["C1", "C2", "C3", "C4"]``. Pass a
            string such as ``"C2"`` or a list such as ``["C1", "C2"]`` to
            restrict.
        ssp_scenario: SSP scenario filter. Defaults to
            ``["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]``. Pass a string
            such as ``"SSP2"`` or a list such as ``["SSP1", "SSP2"]`` to
            restrict.
        emission_type: Dynamic AR6 emission type. Accepted values are
            ``"kyoto_gases"`` (default) and ``"co2"``.
            ``emission_type="kyoto_gases"`` uses the GWP100 Kyoto Gases
            aggregate; ``emission_type="co2"`` uses direct CO2 pathways.
        include_afolu: Whether AFOLU is included inside the selected
            ``emission_type``. ``False`` uses the ``WO AFOLU`` pathway
            family. ``True`` uses the AFOLU-inclusive family. Defaults to
            ``False``.
        emissions_mode: Dynamic AR6 emissions mode. Accepted values are
            ``"net"``, ``"gross"``, and ``"gross_alt"``. Defaults to
            ``"gross_alt"``. ``emissions_mode`` selects net, gross, or gross
            alternative emissions. ``"gross"`` removes all sequestration
            sources from net emissions. ``"gross_alt"`` removes all
            sequestration sources except CCS, as it does not directly capture
            CO2 from the atmosphere; IPCC AR6 recommends treating CCS
            separately from net negative sequestration. Gross modes write
            positive emissions denominator rows and signed negative
            sequestration companion rows; downstream allocated carrying
            capacity (aCC) and absolute sustainability ratio (ASR) consume
            only the denominator gross positive rows. See
            ``data_raw/methodological_notes/methodological_note__steady_state__dynamic_cc.pdf``
            for the methodological explanation.
        subset_version: Optional selector for a subset of AR6 model-scenario
            pairs. Follow
            ``data_processed/ar6/<processed_scope>/README_model_scenario_subset.txt``
            to create the subset CSV. Omit this argument to use every retained
            model-scenario pair.
            Defaults to ``None``.
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
            AR6 CC output scope for the requested study period, harmonization
            flag, harmonization method, emission variables, and subset version,
            plus the matching processed AR6 output scope selected by
            ``process_ar6(...)`` for that request. The cleared AR6 CC scope is
            the selector-specific ``ar6_cc`` deterministic output folder beside
            that processed AR6 scope. For example, for years 2019 to 2060,
            default harmonization, default Kyoto gas settings, category
            ``["C1"]``, and SSP ``["SSP1"]``, the refreshed path is
            ``<repo>/data_processed/ar6/2019-2060_harmonization_offset/ar6_cc/gross_alt_kyoto_gases_wo_afolu/C1__SSP1/deterministic``.
            Raw downloads and downstream aCC or ASR outputs are not refreshed.
            Defaults to ``False``.

    Returns:
        ``ComputeAR6CCReport`` describing deterministic AR6 CC table outputs
        and figure outputs when figures are requested. The study year table is
        written as ``results/ar6_cc.<format>``. When the study period ends
        before 2100, the run also writes
        ``results/ar6_cc_post_study_period.<format>`` for figure rendering.

    Raises:
        ValueError: If arguments are invalid or no pathways match the
            requested filters.
        RuntimeError: If processed AR6 prerequisite outputs are unavailable
            after calling ``process_ar6(...)``.
        FileNotFoundError: If the requested subset CSV is unavailable.

    Notes:
        The repository root is taken from the package default configured by
        ``set_workspace()``; call ``set_workspace()`` before invoking this
        function.
        ``download_ar6(...)`` must have run before this function can read raw
        AR6 inputs. Methodological details on AR6 scenario filtering,
        harmonization, and dynamic carrying capacity construction are provided
        in
        ``data_raw/methodological_notes/methodological_note__steady_state__dynamic_cc.pdf``.

    Example:
        Extract dynamic AR6 carrying capacity pathways for one study period
        and a subset of model-scenario pairs::

            from pyaesa import deterministic_ar6_cc

            deterministic_ar6_cc(
                years=range(2019, 2061),
                category=["C1", "C2"],
                ssp_scenario=["SSP2"],
                subset_version="study_subset",
            )
    """
    if _status is None:
        owned_phase = PhasePrinter("deterministic_ar6_cc")
        status: PhasePrinter | NullPhasePrinter = owned_phase
    else:
        owned_phase = None
        status = _status
    try:
        if owned_phase is not None:
            owned_phase.announce(PHASE_B1_AR6_DYNAMIC_CC, "deterministic_ar6_cc")
        report = run_deterministic_ar6_cc(
            years=years,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            category=category,
            ssp_scenario=ssp_scenario,
            emission_type=emission_type,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode,
            subset_version=subset_version,
            output_format=output_format,
            figures=figures,
            figure_format=normalize_figure_format(figure_format),
            refresh=refresh,
            _status=status,
        )
        if owned_phase is not None:
            if report.reuse_status in {"reused_exact", "partially_reused"}:
                _complete_cached_process_ar6(report=report, phase=owned_phase)
            detail = (
                phase_reused_detail if report.reuse_status == "reused_exact" else phase_ready_detail
            )
            output_root = report.cc_dir if report.cc_dir is not None else report.output_file.parent
            owned_phase.complete(
                detail(scope_name="dynamic AR6 CC", output_root=output_root),
                owner="deterministic_ar6_cc",
            )
        elif isinstance(status, PhasePrinter) and report.reuse_status in {
            "reused_exact",
            "partially_reused",
        }:
            _complete_cached_process_ar6(report=report, phase=status)
        return report
    finally:
        if owned_phase is not None:
            owned_phase.finish()


def _complete_cached_process_ar6(*, report: ComputeAR6CCReport, phase: PhasePrinter) -> None:
    """Print the cached process_ar6 prerequisite line inside a visible dynamic phase."""
    payload = report.process_ar6
    detail = (
        phase_reused_detail
        if str(payload["reuse_status"]) == "reused_exact"
        else phase_ready_detail
    )
    phase.complete(
        detail(scope_name="AR6 processed", output_root=Path(str(payload["output_root"]))),
        owner="process_ar6",
    )
