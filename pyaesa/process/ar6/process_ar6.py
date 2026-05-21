"""Public AR6 climate processing entrypoint."""

from pyaesa.download.ar6.utils.config import (
    DEFAULT_CATEGORIES,
    DEFAULT_DATABASE,
    DEFAULT_SSPS,
    DEFAULT_VARIABLES_OUTPUT,
)
from pyaesa.download.ar6.utils.io.paths import (
    get_citation_txt_path,
    get_raw_dir,
)
from pyaesa.shared.figures.request_validation import normalize_figure_format
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    PHASE_B1_AR6_DYNAMIC_CC,
    phase_ready_detail,
    phase_reused_detail,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reporting.status import StatusSink

from .utils.figures.figure_sampling_config import (
    SamplingFigureConfig,
    validate_sampling_figure_config,
)
from .utils.io.reports import ProcessReportAR6
from .utils.pipeline.process_runner import run_process_ar6_workflow
from .utils.pipeline.runtime_helpers import (
    process_signature,
    validate_harmonization_method,
)
from .utils.pipeline.study_period import resolve_study_period


def process_ar6(
    years: list[int] | range,
    figures: bool = True,
    harmonization: bool = True,
    harmonization_method: str = "offset",
    refresh: bool = False,
    figure_format: dict[str, object] = {"format": "png", "dpi": 500},
    figure_convergence_tol: float = 5e-2,
    figure_convergence_max_runs: int = 20000000,
    _status: PhasePrinter | NullPhasePrinter | None = None,
) -> ProcessReportAR6:
    """Process AR6 scenarios for use by downstream workflows with optional harmonization
    of pathways based on an update of historical baselines.

    The function transforms the downloaded AR6 scenario table into processed
    pathway outputs for a user defined study period. Depending on the
    ``harmonization`` argument, it either harmonizes retained scenarios to the
    historical PRIMAP plus Global Carbon Budget baseline or keeps only the
    scenarios that pass the package interpolation and derived variable
    construction rules without harmonization. The retained
    AR6 domain is categories ``C1-C4``, SSP families ``SSP1-SSP5``, and
    pathways whose historical and future AR6 vetting fields both equal ``"Pass"``.
    Processing first keeps model-scenario pairs whose raw CO2 row has the
    requested study start year and year 2100, then applies the CO2
    decomposition reconstruction check, derives net emissions and carbon
    sequestration rows from the downloaded AR6 decomposition inputs, removes
    model-scenario pairs with negative carbon sequestration, and harmonizes
    the retained net emissions variables. It then appends the matching
    sequestration companion rows, computes gross and gross alternative emissions
    from the harmonized net pathways, and writes twelve positive emissions variables
    plus two sequestration variables.
    Later dynamic AR6 carrying capacity (CC) functions can subset the saved
    tables by category, SSP family, and model-scenario pair. Omit arguments
    to use their default.

    Args:
        years: Study year selector provided as a consecutive year
            list or ``range(start_year, end_year + 1)``. The resolved years
            must contain at least two consecutive years with no gaps.
        figures: Whether to render figures.
            Default is ``True``.
        harmonization: Whether to harmonize retained AR6 pathways to the
            historical baseline. Defaults to ``True``. If ``True``, write
            ``harmonized_ar6_public.xlsx`` plus
            the separate harmonization log workbook
            ``harmonized_ar6_public_log.xlsx``. If ``False``, apply the same
            required CO2 coverage and derived variable construction filters,
            write ``filtered_original_ar6_public.xlsx``, and omit the
            harmonization log workbook. When required component inputs are
            missing for a derived retained variable, the package omits that
            derived row and records the omission in the AR6 row issue log. The
            required CO2 coverage, CO2 reconstruction, sequestration, and gross
            emissions filters are shared with the harmonized mode. Figure
            generation is available for harmonized runs.
        harmonization_method: Harmonization method applied only when
            ``harmonization=True``. Defaults to ``"offset"``. The only
            supported value is currently ``"offset"``.
            Ignored when ``harmonization=False``.
        refresh: If ``True``, clear and recompute only the resolved processed
            AR6 output scope for the requested study period, harmonization
            flag, and harmonization method. Raw downloads and downstream AR6
            CC, aCC, or ASR outputs are not refreshed. Defaults to ``False``.
        figure_format: Figure render settings mapping. Defaults to
            ``{"format": "png", "dpi": 500}``.

            Nested keys:

            - ``format``: Figure file format. Accepted values are ``"png"``,
              ``"pdf"``, and ``"svg"``.
            - ``dpi``: Positive integer figure resolution used for raster
              outputs.
        figure_convergence_tol: Relative convergence tolerance used only by the
            SRS/LHS figure sampling diagnostics when ``figures=True``. The default
            is ``5e-2``, i.e. a ``5%`` maximum relative change between successive
            checkpoint summaries for each monitored summary statistic. Figure
            sampling is accepted only after ``3`` consecutive stable checkpoint
            comparisons. Because the figure workflow evaluates those
            comparisons every ``10000`` runs per bucket, the earliest accepted
            convergence checkpoint is ``40000`` completed runs per bucket.
        figure_convergence_max_runs: Maximum per bucket run count allowed for the
            SRS/LHS figure sampling convergence loop when ``figures=True`` before
            figure generation fails. Default: ``20000000``.
    Returns:
        ``ProcessReportAR6`` describing the requested processed AR6 scope. The same
        summary contract is returned for new processing, figure generation, and
        compatible reuse of existing outputs.
    Raises:
        ValueError: If ``years`` has fewer than two consecutive years, has
            gaps, or lies outside AR6 coverage.
        RuntimeError: If required raw AR6 inputs are missing or saved artefacts
            are structurally inconsistent.

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
        Process the AR6 study period used by dynamic carrying capacities::

            from pyaesa import process_ar6

            process_ar6(range(2019, 2061))
    """
    if _status is None:
        owned_phase = PhasePrinter("process_ar6")
        phase_owner: PhasePrinter | NullPhasePrinter = owned_phase
    else:
        owned_phase = None
        phase_owner = _status
    status: StatusSink = phase_owner
    try:
        phase_owner.announce(PHASE_B1_AR6_DYNAMIC_CC, "process_ar6")
        study_period_norm = resolve_study_period(years)
        if figures and not harmonization:
            raise ValueError(
                "figures=True is only supported when harmonization=True because the figure "
                "workflow depends on harmonized-plus-historical outputs."
            )
        harmonization_method = validate_harmonization_method(
            harmonization=harmonization,
            harmonization_method=harmonization_method,
        )
        sampling_config: SamplingFigureConfig = (
            validate_sampling_figure_config(
                figure_convergence_tol=figure_convergence_tol,
                figure_convergence_max_runs=figure_convergence_max_runs,
            )
            if figures
            else SamplingFigureConfig(relative_tolerance=5e-2, max_runs_per_bucket=1000000)
        )
        figure_format_norm = (
            normalize_figure_format(figure_format) if figures else {"format": "png", "dpi": 500}
        )
        report = run_process_ar6_workflow(
            study_period=study_period_norm,
            figures=figures,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            refresh=refresh,
            figure_output_format=str(figure_format_norm["format"]),
            figure_dpi=int(figure_format_norm["dpi"]),
            sampling_config=sampling_config,
            signature=process_signature(study_period_norm, harmonization, harmonization_method),
            categories=list(DEFAULT_CATEGORIES),
            ssps=[int(value) for value in DEFAULT_SSPS],
            variables_output=list(DEFAULT_VARIABLES_OUTPUT),
            database=DEFAULT_DATABASE,
            raw_data_dir=get_raw_dir(),
            citation_txt_path=get_citation_txt_path(),
            status=status,
        )
        detail = (
            phase_reused_detail if report.reuse_status == "reused_exact" else phase_ready_detail
        )
        phase_owner.complete(
            detail(scope_name="AR6 processed", output_root=report.processed_dir),
            owner="process_ar6",
        )
        return report
    finally:
        if owned_phase is not None:
            owned_phase.finish()
