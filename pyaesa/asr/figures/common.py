"""Common ASR figure contracts."""

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from pyaesa.acc.figures.common import (
    DYNAMIC_SCOPE_COLUMNS as DYNAMIC_SCOPE_COLUMNS,
    VALUE_ARRAY_COLUMN as VALUE_ARRAY_COLUMN,
    attach_common_columns as attach_common_columns,
    format_year_axis as format_year_axis,
    impact_panel_title as impact_panel_title,
    ordered_impacts as ordered_impacts,
    requested_single_year as requested_single_year,
    save_figure as save_figure,
    scope_slices as scope_slices,
    static_asocc_ssp_slices as static_asocc_ssp_slices,
    visible_values,
)
from pyaesa.asr.figures.axis import (
    ASR_LOG_SCALE,
    ASR_NORMAL_SCALE,
    ASRScaleMode,
    apply_asr_log_axis,
    apply_asr_normal_axis,
    apply_positive_log_axis,
    resolve_asr_log_limits,
    resolve_asr_normal_limits,
    resolve_asr_scale_mode,
    resolve_positive_log_limits,
)
from pyaesa.asr.figures.risk_guides import (
    ASR_NON_POLAR_RISK_BACKGROUND_ALPHA_SCALE,
    ASR_RISK_LEGEND_GROUP_TITLE,
    render_asr_threshold_guides,
)
from pyaesa.asr.figures.threshold_contract import has_max_asr_threshold
from pyaesa.shared.figures.dynamic_ar6 import (
    AR6_CATEGORY_SCOPE_COLUMN,
    category_scope_label,
    dynamic_ar6_detail_line,
    model_scenario_pair_token,
)
from pyaesa.shared.figures.lcia_metadata import lcia_title_parts
from pyaesa.shared.figures.layout import TRANSITION_PANEL_TITLE_PAD
from pyaesa.shared.figures.multi_year_transitions import (
    TransitionMarker,
    render_transition_markers,
    transition_title_pad,
)
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter
from pyaesa.shared.runtime.scenario.columns import AR6_CC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.scalars import sanitize_token

MEAN_LINE_NOTE = "ASR pathway lines represent Monte Carlo runs mean values."


def asr_scope_stem(
    label: str,
    frame: pd.DataFrame,
    *,
    include_impact: bool = False,
    studied_year: int | None = None,
    product: str | None = None,
    selector_token: str = "all",
) -> str:
    """Return one ASR figure file stem."""
    parts = [label]
    if product is not None:
        parts.append(product)
    if str(selector_token).strip() and selector_token != "all":
        parts.append(str(selector_token).strip())
    parts.extend(visible_values(frame, "lcia_method")[:1])
    if include_impact:
        parts.extend(visible_values(frame, "impact")[:1])
    parts.extend(_scope_scenario_values(frame)[:1])
    parts.extend(visible_values(frame, "cc_category")[:1])
    parts.extend(value for value in visible_values(frame, "cc_bound")[:1] if value != "both")
    if studied_year is not None:
        parts.append(str(int(studied_year)))
    model_pair = _dynamic_model_scenario_token(frame)
    if model_pair is not None:
        parts.append(model_pair)
    return "__".join(sanitize_token(part) for part in parts if str(part).strip())


def asr_scope_title(
    family_label: str,
    label: str | None,
    frame: pd.DataFrame,
    *,
    include_impact: bool,
    studied_year: int | None = None,
    selector_title: str | None = None,
) -> str:
    """Return one compact ASR figure title."""
    del selector_title
    parts = [family_label]
    if label is not None:
        parts.append(label)
    parts.extend(lcia_title_parts(frame, include_impact=include_impact))
    if studied_year is not None:
        parts.append(str(int(studied_year)))
    scenario = _scope_scenario_values(frame)
    if scenario:
        parts.append(scenario[0])
    dynamic_detail = dynamic_ar6_detail_line(
        categories=visible_values(frame, "cc_category"),
        models=visible_values(frame, "cc_model"),
        scenarios=visible_values(frame, "cc_scenario"),
    )
    if dynamic_detail:
        return f"{' | '.join(parts)}\n{dynamic_detail}"
    category_scope = _ar6_category_scope(frame)
    if category_scope:
        noun = "categories" if _is_multi_category_scope(category_scope) else "category"
        parts.append(f"AR6 {noun}: {category_scope}")
    return " | ".join(parts)


def _apply_log_asr_axis_policy(
    axis: Any,
    *,
    values: np.ndarray,
    frame: pd.DataFrame,
    grouped_legend: bool = False,
    limits: tuple[float, float] | None = None,
    threshold_background_alpha_scale: float = ASR_NON_POLAR_RISK_BACKGROUND_ALPHA_SCALE,
) -> int:
    """Apply ASR log scale and threshold guides."""
    zero_count = apply_asr_log_axis(axis, values=values, context="ASR figure", limits=limits)
    render_asr_threshold_guides(
        axis,
        has_max_threshold=has_max_asr_threshold(frame=frame),
        max_threshold=_max_threshold(frame),
        grouped_title=ASR_RISK_LEGEND_GROUP_TITLE if grouped_legend else None,
        background_alpha_scale=threshold_background_alpha_scale,
    )
    axis.set_ylabel(format_scientific_figure_text("ASR"))
    axis.set_xlabel("")
    axis.grid(alpha=0.25, which="major")
    axis.grid(alpha=0.12, which="minor", axis="y")
    return zero_count


def _apply_normal_asr_axis_policy(
    axis: Any,
    *,
    values: np.ndarray,
    frame: pd.DataFrame,
    grouped_legend: bool = False,
    limits: tuple[float, float] | None = None,
    threshold_background_alpha_scale: float = ASR_NON_POLAR_RISK_BACKGROUND_ALPHA_SCALE,
) -> None:
    """Apply the ASR normal axis and threshold guides."""
    threshold_values = [1.0]
    max_threshold = _max_threshold(frame)
    if max_threshold is not None:
        threshold_values.append(float(max_threshold))
    apply_asr_normal_axis(
        axis,
        values=np.concatenate(
            [np.asarray(values, dtype=np.float64), np.asarray(threshold_values, dtype=np.float64)]
        ),
        limits=limits,
    )
    render_asr_threshold_guides(
        axis,
        has_max_threshold=has_max_asr_threshold(frame=frame),
        max_threshold=max_threshold,
        grouped_title=ASR_RISK_LEGEND_GROUP_TITLE if grouped_legend else None,
        background_alpha_scale=threshold_background_alpha_scale,
    )
    axis.set_ylabel(format_scientific_figure_text("ASR"))
    axis.set_xlabel("")
    axis.grid(alpha=0.25)


def apply_scaled_asr_axis_policy(
    axis: Any,
    *,
    values: np.ndarray,
    frame: pd.DataFrame,
    scale_mode: ASRScaleMode,
    grouped_legend: bool = False,
    limits: tuple[float, float] | None = None,
    threshold_background_alpha_scale: float = ASR_NON_POLAR_RISK_BACKGROUND_ALPHA_SCALE,
) -> None:
    """Apply the requested ASR normal or log axis policy."""
    if scale_mode == ASR_LOG_SCALE:
        _apply_log_asr_axis_policy(
            axis,
            values=values,
            frame=frame,
            grouped_legend=grouped_legend,
            limits=limits,
            threshold_background_alpha_scale=threshold_background_alpha_scale,
        )
        return
    _apply_normal_asr_axis_policy(
        axis,
        values=values,
        frame=frame,
        grouped_legend=grouped_legend,
        limits=limits,
        threshold_background_alpha_scale=threshold_background_alpha_scale,
    )


def asr_scale_mode_for_values(*values: np.ndarray) -> ASRScaleMode:
    """Return one ASR scale mode from all visible ASR value arrays."""
    numeric = np.concatenate([np.asarray(value, dtype=np.float64).ravel() for value in values])
    return resolve_asr_scale_mode(numeric)


def asr_axis_limits(
    *,
    values: np.ndarray,
    frame: pd.DataFrame,
    scale_mode: ASRScaleMode,
) -> tuple[float, float]:
    """Return ASR axis limits for the resolved scale mode."""
    thresholds = pd.Series(pd.to_numeric(frame["__asr_max_threshold"], errors="coerce")).dropna()
    numeric = np.concatenate(
        [
            np.asarray(values, dtype=np.float64).ravel(),
            thresholds.to_numpy(dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
        ]
    )
    if scale_mode == ASR_LOG_SCALE:
        limits, _zero_count = resolve_asr_log_limits(numeric, context="ASR figure")
        return limits
    return resolve_asr_normal_limits(numeric)


def component_axis_limits(
    *,
    values: np.ndarray,
    scale_mode: ASRScaleMode,
) -> tuple[float, float]:
    """Return aCC versus LCA axis limits for the inherited ASR scale mode."""
    numeric = np.asarray(values, dtype=np.float64).ravel()
    if scale_mode == ASR_LOG_SCALE:
        return resolve_positive_log_limits(numeric, context="ASR component figure")
    return data_linear_limits(numeric)


def format_acc_lca_component_axis(
    *,
    axis: Any,
    frame: pd.DataFrame,
    years: list[int],
    show_x_labels: bool,
    title: str,
    limits: tuple[float, float],
    markers: list[TransitionMarker] | None = None,
    title_pad: int | None = None,
    scale_mode: ASRScaleMode = ASR_LOG_SCALE,
    transition_shade_right: float | None = None,
) -> None:
    """Apply the shared dynamic ASR aCC versus LCA axis contract."""
    if scale_mode == ASR_NORMAL_SCALE:
        axis.set_ylim(*limits)
        axis.yaxis.set_major_formatter(FuncFormatter(scientific_tick_formatter))
    else:
        apply_positive_log_axis(axis, limits=limits)
    if years:
        format_year_axis(axis, years=sorted(set(years)), show_labels=show_x_labels)
        render_transition_markers(
            axis,
            markers=markers or [],
            shade_right=transition_shade_right,
        )
    axis.set_ylabel(format_scientific_figure_text(component_unit_label(frame)))
    axis.grid(alpha=0.25, axis="both")
    axis.set_title(
        title,
        fontweight="bold",
        pad=int(title_pad)
        if title_pad is not None
        else transition_title_pad(
            markers or [],
            no_transition=6,
            single_transition=TRANSITION_PANEL_TITLE_PAD,
            component_transition=38,
        ),
    )


def data_linear_limits(*values: np.ndarray) -> tuple[float, float]:
    """Return padded linear limits around visible data without forcing zero."""
    numeric = np.concatenate([np.asarray(value, dtype=np.float64) for value in values])
    valid = numeric[np.isfinite(numeric)]
    if valid.size == 0:
        return (0.0, 1.0)
    lower = float(np.min(valid))
    upper = float(np.max(valid))
    if lower == upper:
        padding = abs(lower) * 0.12 if lower != 0.0 else 1.0
        return (lower - padding, upper + padding)
    span = upper - lower
    return (lower - 0.12 * span, upper + 0.12 * span)


def dynamic_linear_limits(*values: np.ndarray) -> tuple[float, float]:
    """Return padded linear limits, zero based when all visible values are nonnegative."""
    numeric = np.concatenate([np.asarray(value, dtype=np.float64) for value in values])
    valid = numeric[np.isfinite(numeric)]
    if valid.size == 0:
        return (0.0, 1.0)
    lower = float(np.min(valid))
    upper = float(np.max(valid))
    if lower >= 0.0:
        return (0.0, 1.0 if upper == 0.0 else upper * 1.12)
    if lower == upper:
        padding = abs(lower) * 0.12 if lower != 0.0 else 1.0
        return (lower - padding, upper + padding)
    span = upper - lower
    return (lower - 0.12 * span, upper + 0.12 * span)


def component_unit_label(frame: pd.DataFrame) -> str:
    """Return the unit label used by dynamic ASR aCC versus LCA panels."""
    values = visible_values(frame, "impact_unit")
    return values[0] if values else "value"


def _max_threshold(frame: pd.DataFrame) -> float | None:
    values = pd.Series(pd.to_numeric(frame["__asr_max_threshold"], errors="coerce")).dropna()
    positive = [float(value) for value in values.tolist() if float(value) > 1.0]
    return max(positive) if positive else None


def _scope_scenario_values(frame: pd.DataFrame) -> list[str]:
    ar6_values = visible_values(frame, AR6_CC_SSP_SCENARIO_COLUMN)
    if ar6_values:
        return ar6_values
    return visible_values(frame, "asocc_ssp_scenario")


def _ar6_category_scope(frame: pd.DataFrame) -> str:
    categories = visible_values(frame, "cc_category")
    if categories:
        return category_scope_label(categories)
    scopes = visible_values(frame, AR6_CATEGORY_SCOPE_COLUMN)
    return scopes[0] if scopes else ""


def _is_multi_category_scope(scope: str) -> bool:
    text = str(scope).strip()
    return "-" in text or "," in text


def _dynamic_model_scenario_token(frame: pd.DataFrame) -> str | None:
    return model_scenario_pair_token(
        models=visible_values(frame, "cc_model"),
        scenarios=visible_values(frame, "cc_scenario"),
    )
