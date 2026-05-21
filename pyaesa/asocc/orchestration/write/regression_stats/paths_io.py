"""Path/window resolution and table I/O for regression diagnostics."""

from pathlib import Path
import pandas as pd

from ....runtime.paths.deterministic import (
    _get_allocate_logs_dir,
    columns_defs_path_for_stats,
    suffix_for_output_format,
)
from ...projection.config.types import FIT_INPUT_REGRESSION_KEY
from pyaesa.asocc.orchestration.write.regression_stats.columns import (
    REGRESSION_MODELS_COLUMNS,
    render_regression_columns_defs,
)

_FIT_INPUT_KEY_COLUMNS = list(FIT_INPUT_REGRESSION_KEY[1:])
_FIT_INPUT_OUTPUT_COLUMNS = [
    "projection_branch",
    "source",
    "fu_code",
    "l2_method",
    "model_type",
    "target_object",
    "domain_key",
    "fit_start_year",
    "fit_end_year",
    "fit_year",
    "x_object",
    "x_unit",
    "x_value",
    "y_object",
    "y_unit",
    "y_value",
    "y_kind",
    "ratio_value",
    "numerator_object",
    "numerator_value",
    "denominator_object",
    "denominator_value",
]
_UNCERTAINTY_REQUIRED_COLUMNS = [
    "sigma2_hat",
    "df_resid",
    "x_mean",
    "ssx",
    "x_min",
    "x_max",
    "years_used",
    "notes",
]


def existing_scoped_stats_paths(
    *,
    proj_base: Path,
    output_format: str,
    source: str,
    group_version: str | None,
) -> list[Path]:
    """Return the deterministic regression stats path when it exists."""
    suffix = suffix_for_output_format(output_format=output_format)
    logs_dir = _get_allocate_logs_dir(
        proj_base,
        source=source,
        group_version=group_version,
    )
    if not logs_dir.exists():
        return []
    path = logs_dir / "regression_proj" / f"regression_stats{suffix}"
    return [path] if path.exists() else []


def _reorder_fit_inputs_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return fit input frame with strict deterministic public column order."""
    out = frame.copy()
    for column in _FIT_INPUT_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, _FIT_INPUT_OUTPUT_COLUMNS]


def _fit_windows_in_frame(*, frame: pd.DataFrame, label: str) -> set[tuple[int, int]]:
    """Return fit window tuples present in one frame."""
    if frame.empty:
        return set()
    missing = [column for column in ("fit_start_year", "fit_end_year") if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing required regression key columns: {missing}.")
    return {
        (int(fit_start_year), int(fit_end_year))
        for fit_start_year, fit_end_year in frame.loc[
            :, ["fit_start_year", "fit_end_year"]
        ].itertuples(index=False, name=None)
    }


def _resolve_single_fit_window(*, windows: set[tuple[int, int]]) -> tuple[int, int] | None:
    """Return unique fit window or fail fast on mixed windows."""
    if not windows:
        return None
    if len(windows) == 1:
        return next(iter(windows))
    sample = sorted(windows)[:5]
    raise ValueError(
        "Regression diagnostics rows span multiple fit windows in one write call: "
        f"{sample}. Split writes per fit window."
    )


def _resolve_fit_window_for_write(
    *,
    stats_frame: pd.DataFrame,
    uncertainty_frame: pd.DataFrame,
    fit_inputs_frame: pd.DataFrame,
) -> tuple[int, int] | None:
    """Resolve strict single fit window for one write call."""
    windows: set[tuple[int, int]] = set()
    windows |= _fit_windows_in_frame(frame=stats_frame, label="regression_stats_rows")
    windows |= _fit_windows_in_frame(
        frame=uncertainty_frame,
        label="regression_uncertainty_rows",
    )
    windows |= _fit_windows_in_frame(
        frame=fit_inputs_frame,
        label="regression_fit_inputs_rows",
    )
    return _resolve_single_fit_window(windows=windows)


def _write_regression_columns_defs(*, stats_path: Path) -> None:
    """Write `regression_stats_columns_defs.txt` next to regression stats output."""
    defs_path = columns_defs_path_for_stats(stats_path=stats_path)
    defs_path.write_text(
        render_regression_columns_defs(columns=REGRESSION_MODELS_COLUMNS),
        encoding="utf-8",
    )


def _write_table(*, path: Path, output_format: str, frame: pd.DataFrame) -> None:
    """Write one diagnostics table by output format."""
    if output_format == "csv":
        frame.to_csv(path, index=False)
        return
    if output_format == "pickle":
        frame.to_pickle(path)
        return
    frame.to_parquet(path, index=False)
