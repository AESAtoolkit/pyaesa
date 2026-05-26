"""Regression diagnostics writer for projection mode."""

from typing import cast

import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

from ....runtime.paths.deterministic import (
    fit_inputs_path_for_format,
    stats_path_for_format,
)
from ...projection.regression.projection_clipping_log import clip_counts_by_key
from pyaesa.asocc.orchestration.write.regression_stats.models import (
    _merge_regression_models_frame,
)
from pyaesa.asocc.orchestration.write.regression_stats.paths_io import (
    _FIT_INPUT_KEY_COLUMNS,
    _UNCERTAINTY_REQUIRED_COLUMNS,
    _reorder_fit_inputs_columns,
    _write_regression_columns_defs,
    _write_table,
)
from pyaesa.asocc.orchestration.write.writers.progress import tick_write_progress


def write_regression_stats(
    *,
    context,
    state,
):
    """Write regression diagnostics tables for the current deterministic run."""
    output_source = context.output_source
    has_stats = bool(state.regression_stats_rows)
    has_fit_inputs = bool(state.regression_fit_inputs_rows)
    if not has_stats and not has_fit_inputs:
        return None

    stats_frame = pd.DataFrame(state.regression_stats_rows) if has_stats else pd.DataFrame()
    fit_inputs_frame = (
        pd.DataFrame(state.regression_fit_inputs_rows) if has_fit_inputs else pd.DataFrame()
    )
    uncertainty_frame = (
        pd.DataFrame(state.regression_uncertainty_rows) if has_stats else pd.DataFrame()
    )

    fit_inputs_path = fit_inputs_path_for_format(
        proj_base=context.proj_base,
        output_format=context.output_format,
        source=output_source,
        agg_version=context.agg_version,
    )

    if not has_stats:
        fit_inputs_path = ensure_file_parent(fit_inputs_path)
        out_fit_inputs = fit_inputs_frame.copy()
        out_fit_inputs = out_fit_inputs.drop_duplicates(
            subset=_FIT_INPUT_KEY_COLUMNS,
            keep="last",
        )
        out_fit_inputs = out_fit_inputs.sort_values(_FIT_INPUT_KEY_COLUMNS).reset_index(drop=True)
        out_fit_inputs = _reorder_fit_inputs_columns(out_fit_inputs)
        _write_table(
            path=fit_inputs_path,
            output_format=context.output_format,
            frame=out_fit_inputs,
        )
        tick_write_progress(context=context, state=state)
        return None

    fit_window = cast(tuple[int, int], context.projection_context.reg_window)
    fit_start_year, fit_end_year = fit_window
    path = stats_path_for_format(
        proj_base=context.proj_base,
        output_format=context.output_format,
        source=output_source,
        agg_version=context.agg_version,
    )
    path = ensure_file_parent(path)

    clip_counts = clip_counts_by_key(
        proj_base=context.proj_base,
        fit_start_year=int(fit_start_year),
        fit_end_year=int(fit_end_year),
        source=output_source,
        agg_version=context.agg_version,
    )
    merged = _merge_regression_models_frame(
        stats_frame=stats_frame,
        uncertainty_frame=uncertainty_frame,
        clip_counts=clip_counts,
        required_uncertainty_columns=_UNCERTAINTY_REQUIRED_COLUMNS,
    )
    _write_table(path=path, output_format=context.output_format, frame=merged)
    _write_regression_columns_defs(stats_path=path)
    tick_write_progress(context=context, state=state)

    out_fit_inputs = fit_inputs_frame.copy()
    out_fit_inputs = out_fit_inputs.drop_duplicates(
        subset=_FIT_INPUT_KEY_COLUMNS,
        keep="last",
    )
    out_fit_inputs = out_fit_inputs.sort_values(_FIT_INPUT_KEY_COLUMNS).reset_index(drop=True)
    out_fit_inputs = _reorder_fit_inputs_columns(out_fit_inputs)
    _write_table(
        path=fit_inputs_path,
        output_format=context.output_format,
        frame=out_fit_inputs,
    )
    tick_write_progress(context=context, state=state)
    return path
