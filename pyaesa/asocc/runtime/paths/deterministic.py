"""Family-local deterministic aSoCC paths for logs, provenance, and diagnostics."""

from pathlib import Path

from pyaesa.asocc.runtime.paths.published import _asocc_deterministic_scope_root
from pyaesa.shared.runtime.metadata.contracts import (
    FIGURE_MANIFEST_FILENAME,
    SCOPE_MANIFEST_FILENAME,
)
from pyaesa.shared.runtime.reporting.summary_log import SUMMARY_LOG_FILENAME


def _get_allocate_refresh_scope_root(
    *,
    proj_base: Path,
    source: str,
    group_version: str | None,
) -> Path:
    """Return the deterministic aSoCC refresh scope for one source/version branch."""
    return _asocc_deterministic_scope_root(
        proj_base=proj_base,
        source=source,
        group_version=group_version,
    )


def _get_allocate_summary_log_path(
    proj_base: Path,
    *,
    source: str,
    group_version: str | None,
) -> Path:
    """Return deterministic_asocc summary log path inside the aSoCC scope."""
    return (
        _get_allocate_logs_dir(
            proj_base,
            source=source,
            group_version=group_version,
        )
        / SUMMARY_LOG_FILENAME
    )


def _get_allocate_logs_dir(
    proj_base: Path,
    *,
    source: str,
    group_version: str | None,
) -> Path:
    """Return deterministic_asocc logs directory owned by the deterministic aSoCC family."""
    return (
        _asocc_deterministic_scope_root(
            proj_base=proj_base,
            source=source,
            group_version=group_version,
        )
        / "logs"
    )


def _get_allocate_run_metadata_path(
    proj_base: Path,
    *,
    source: str,
    group_version: str | None,
) -> Path:
    """Return the deterministic aSoCC scope manifest path."""
    return (
        _get_allocate_logs_dir(
            proj_base,
            source=source,
            group_version=group_version,
        )
        / SCOPE_MANIFEST_FILENAME
    )


def _get_projection_regression_dir(
    proj_base: Path,
    *,
    source: str,
    group_version: str | None,
) -> Path:
    """Return the deterministic regression diagnostics directory."""
    return (
        _get_allocate_logs_dir(
            proj_base,
            source=source,
            group_version=group_version,
        )
        / "regression_proj"
    )


def suffix_for_output_format(*, output_format: str) -> str:
    """Return deterministic extension for one output format."""
    return {
        "csv": ".csv",
        "pickle": ".pickle",
        "parquet": ".parquet",
    }[str(output_format)]


def _get_allocate_regression_stats_path(
    *,
    proj_base: Path,
    output_format: str,
    source: str,
    group_version: str | None,
) -> Path:
    """Return regression diagnostics output path for deterministic_asocc."""
    return (
        _get_projection_regression_dir(
            proj_base,
            source=source,
            group_version=group_version,
        )
        / f"regression_stats{suffix_for_output_format(output_format=output_format)}"
    )


def _get_allocate_regression_fit_inputs_path(
    *,
    proj_base: Path,
    output_format: str,
    source: str,
    group_version: str | None,
) -> Path:
    """Return regression fit inputs output path for deterministic_asocc."""
    return (
        _get_projection_regression_dir(
            proj_base,
            source=source,
            group_version=group_version,
        )
        / f"regression_fit_inputs{suffix_for_output_format(output_format=output_format)}"
    )


def _get_allocate_ut_gvaa_identity_closure_path(
    *,
    proj_base: Path,
    source: str,
    group_version: str | None,
) -> Path:
    """Return UT(GVAa) identity closure audit path for deterministic_asocc."""
    return (
        _get_allocate_logs_dir(
            proj_base,
            source=source,
            group_version=group_version,
        )
        / "ut_gvaa_identity_closure_audit.csv"
    )


def _get_asocc_figure_metadata_path(
    *,
    proj_base: Path,
    source: str,
    group_version: str | None,
) -> Path:
    """Return deterministic aSoCC figure state metadata path for one source scope."""
    return (
        _get_allocate_logs_dir(
            proj_base,
            source=source,
            group_version=group_version,
        )
        / FIGURE_MANIFEST_FILENAME
    )


def allocate_regression_logs_dir(
    *,
    proj_base: Path,
    source: str,
    group_version: str | None,
) -> Path:
    """Return deterministic regression logs dir."""
    base = _get_allocate_logs_dir(
        proj_base,
        source=source,
        group_version=group_version,
    )
    return base / "regression_proj"


def runtime_regression_logs_dir(
    *,
    state,
) -> Path:
    """Return regression logs dir for a runtime state."""
    proj_base = state.runtime_proj_base
    output_source = state.runtime_output_source
    runtime_group_version = getattr(state, "runtime_group_version", None)
    return allocate_regression_logs_dir(
        proj_base=Path(proj_base),
        source=output_source,
        group_version=runtime_group_version,
    )


def projection_clipping_log_path(
    *,
    state,
) -> Path:
    """Return clipping log path for one regression fit window."""
    return runtime_regression_logs_dir(state=state) / "projection_clipping_log.csv"


def share_fit_window_log_path(
    *,
    state,
) -> Path:
    """Return share fit window diagnostics path for one regression window."""
    return runtime_regression_logs_dir(state=state) / "share_fit_window_log.csv"


def stats_path_for_format(
    *,
    proj_base: Path,
    output_format: str,
    source: str,
    group_version: str | None,
) -> Path:
    """Return deterministic regression stats output path."""
    logs_dir = allocate_regression_logs_dir(
        proj_base=proj_base,
        source=source,
        group_version=group_version,
    )
    return logs_dir / f"regression_stats{suffix_for_output_format(output_format=output_format)}"


def fit_inputs_path_for_format(
    *,
    proj_base: Path,
    output_format: str,
    source: str,
    group_version: str | None,
) -> Path:
    """Return deterministic regression fit inputs output path."""
    logs_dir = allocate_regression_logs_dir(
        proj_base=proj_base,
        source=source,
        group_version=group_version,
    )
    return logs_dir / (
        f"regression_fit_inputs{suffix_for_output_format(output_format=output_format)}"
    )


def columns_defs_path_for_stats(*, stats_path: Path) -> Path:
    """Return the human-readable regression column definitions sidecar path."""
    return stats_path.with_name("regression_stats_columns_defs.txt")
