"""Family-local reporting and figure-sync ownership for deterministic aSoCC."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def emit_family_warning(
    *,
    logger,
    message: str,
) -> None:
    """Record one family warning in the deterministic aSoCC logger."""
    logger.warning(message)


def emit_deduplicated_family_warning(
    *,
    context,
    state,
    key: str,
    message: str,
) -> None:
    """Record one deduplicated family warning for the final summary."""
    if key in state.notices_emitted:
        return
    state.notices_emitted.add(key)
    warnings = getattr(state, "summary_warnings", None)
    if not isinstance(warnings, list):
        warnings = []
        setattr(state, "summary_warnings", warnings)
    warnings.append(str(message))
    emit_family_warning(
        logger=context.logger,
        message=message,
    )


@dataclass(frozen=True)
class AsoccFigureSyncResult:
    """Figure-sync result for one deterministic or disaggregated aSoCC branch."""

    figure_paths: list[Path]


def sync_asocc_branch_figures(
    *,
    mode_result: Any,
    figures: bool,
    refresh: bool,
    figure_external_method: dict[str, Any] | None,
    figure_options: dict[str, bool],
    figure_output_format: str,
    figure_dpi: int,
    status_source: str,
    status: Any | None = None,
) -> AsoccFigureSyncResult:
    """Render one deterministic aSoCC figure branch when requested."""
    if not figures:
        return AsoccFigureSyncResult(figure_paths=[])

    from ...figures.render import render_asocc_figures

    written_paths = render_asocc_figures(
        proj_base=mode_result.proj_base,
        source=mode_result.output_source_label,
        fu_code=mode_result.fu_code,
        requested_years=mode_result.requested_years,
        lcia_methods=mode_result.lcia_methods,
        ssp_scenario_options_by_year=mode_result.ssp_scenario_options_by_year,
        compute_signature=mode_result.run_signature,
        output_paths=mode_result.output_paths,
        figure_external_method=figure_external_method,
        figure_options=figure_options,
        dpi=int(figure_dpi),
        output_format=str(figure_output_format),
        refresh=refresh,
        skip_if_exact=mode_result.skipped,
        status_source=status_source,
        status=status,
    )
    return AsoccFigureSyncResult(
        figure_paths=[] if written_paths is None else list(written_paths),
    )
