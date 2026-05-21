"""Common aCC figure contracts shared by deterministic and uncertainty renderers."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from pyaesa.asocc.figures.per_method_renderer import single_requested_year
from pyaesa.shared.figures.layout import format_integer_year_axis
from pyaesa.shared.figures.lcia_metadata import (
    lcia_title_parts,
    ordered_impact_panels,
    resolve_frame_impact_title,
)
from pyaesa.shared.figures.dynamic_ar6 import (
    AR6_CATEGORY_SCOPE_COLUMN,
    category_scope_label,
    dynamic_ar6_detail_line,
    DYNAMIC_AR6_CC_TYPE,
    model_scenario_pair_token,
    MODEL_SCENARIO_PAIR_COUNT_COLUMN,
    MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
)
from pyaesa.shared.figures.nonnegative_axis import require_nonnegative_figure_ylim
from pyaesa.shared.figures.paths import output_file_path
from pyaesa.shared.figures.scenario_scopes import repeat_invariant_rows_into_scenarios
from pyaesa.shared.figures.scientific_ticks import scientific_tick_formatter
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.figures.scope_values import visible_scope_values
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
)
from pyaesa.shared.tabular.scalars import is_display_missing, sanitize_token

METHOD_COLUMNS = ("l1_l2_method", "l1_method", "l2_method")
SELECTOR_COLUMNS = ("r_c", "r_p", "r_f", "s_p")
LCIA_COLUMNS = ("lcia_method", "impact")
DYNAMIC_SCOPE_COLUMNS = (
    AR6_CC_SSP_SCENARIO_COLUMN,
    "cc_category",
    "cc_model",
    "cc_scenario",
)
VALUE_ARRAY_COLUMN = "__values"
BUDGET_VALUES_COLUMN = "__budget_values"
PAIR_COUNT_COLUMN = MODEL_SCENARIO_PAIR_COUNT_COLUMN
MEAN_LINE_NOTE = "The lines represent Monte Carlo runs mean values."
MIN_CC_BOUND = "min_cc"
MAX_CC_BOUND = "max_cc"
DYNAMIC_CC_TYPE = DYNAMIC_AR6_CC_TYPE


def visible_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Return sorted visible values for one aCC figure column."""
    return visible_scope_values(frame, column)


def method_labels(frame: pd.DataFrame) -> pd.Series:
    """Return one visible allocation method label per row."""
    labels = pd.Series(["aCC"] * len(frame), index=frame.index, dtype="object")
    for column in reversed(METHOD_COLUMNS):
        if column not in frame.columns:
            continue
        values = pd.Series(frame[column], copy=False)
        mask = ~values.map(is_display_missing)
        labels.loc[mask] = values.loc[mask].astype(str).str.strip()
    return labels


def attach_common_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach method labels and normalize optional L2 reuse year columns."""
    out = frame.copy()
    out["__method"] = method_labels(out)
    for column in (ASOCC_SSP_SCENARIO_COLUMN, AR6_CC_SSP_SCENARIO_COLUMN):
        if column in out.columns:
            series = pd.Series(out[column], copy=False)
            out[column] = series.where(
                series.map(is_display_missing),
                series.astype(str).str.upper(),
            )
    return out


def scope_slices(frame: pd.DataFrame, columns: tuple[str, ...]) -> list[pd.DataFrame]:
    """Return one figure slice per visible combination of selected columns."""
    group_columns = [column for column in columns if column in frame.columns]
    return [group.copy() for _key, group in frame.groupby(group_columns, dropna=False, sort=True)]


def static_asocc_ssp_slices(
    frame: pd.DataFrame,
    *,
    requested_ssps: tuple[str, ...] = (),
) -> list[pd.DataFrame]:
    """Return static aCC slices with invariant rows repeated into SSP scopes."""
    return repeat_invariant_rows_into_scenarios(
        frame,
        scenario_column=ASOCC_SSP_SCENARIO_COLUMN,
        scope_column="__figure_ssp_scope",
        requested_scenarios=requested_ssps,
        identity_excluded_columns={
            VALUE_ARRAY_COLUMN,
            BUDGET_VALUES_COLUMN,
            PAIR_COUNT_COLUMN,
            MODEL_SCENARIO_SAMPLING_METHOD_COLUMN,
            "acc",
            "mean",
            "std",
            "min",
            "p5",
            "p25",
            "median",
            "p75",
            "p95",
            "max",
        },
    )


def ordered_impacts(frame: pd.DataFrame) -> list[str]:
    """Return impact categories for multi panel aCC figures."""
    impacts = visible_values(frame, "impact")
    if len(impacts) <= 1:
        return impacts
    return ordered_impact_panels(
        lcia_method=visible_values(frame, "lcia_method")[0], impacts=impacts
    )


def impact_panel_title(frame: pd.DataFrame, *, impact: str) -> str:
    """Return the visible panel title for one aCC impact."""
    title = resolve_frame_impact_title(frame)
    return str(title).strip() if title is not None else str(impact).strip()


def panel_unit_label(frame: pd.DataFrame) -> str:
    """Return one y axis label for an aCC impact panel."""
    values = visible_values(frame, "impact_unit")
    return format_scientific_figure_text(f"aCC ({values[0]})")


def cumulative_budget_unit_label(frame: pd.DataFrame) -> str:
    """Return one y axis label for a cumulative dynamic aCC budget panel."""
    values = visible_values(frame, "impact_unit")
    unit = str(values[0]).replace("/yr", "").replace(" yr^-1", "").strip()
    return format_scientific_figure_text(f"aCC ({unit})")


def is_dynamic_scope(frame: pd.DataFrame) -> bool:
    """Return whether one figure frame represents dynamic AR6 aCC rows."""
    if "cc_type" in frame.columns:
        return visible_values(frame, "cc_type") == [DYNAMIC_CC_TYPE]
    return bool(visible_values(frame, AR6_CC_SSP_SCENARIO_COLUMN))


def apply_acc_axis_policy(axis, *, values: np.ndarray, context: str) -> None:
    """Format one aCC y axis using nonnegative scientific notation."""
    axis.set_ylim(*require_nonnegative_figure_ylim(values=values, context=context))
    axis.yaxis.set_major_formatter(FuncFormatter(scientific_tick_formatter))
    axis.set_axisbelow(True)


def format_year_axis(axis, *, years: list[int], show_labels: bool) -> None:
    """Apply automatic integer x ticks for aCC multi year products."""
    format_integer_year_axis(axis, years=years, rotation=45, ha="right")
    axis.set_xlabel("")
    if not show_labels:
        axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


def acc_scope_stem(
    label: str,
    frame: pd.DataFrame,
    *,
    include_impact: bool = False,
    selector_token: str = "all",
    studied_year: int | None = None,
) -> str:
    """Return one aCC figure file stem."""
    parts = [label]
    if str(selector_token).strip() and selector_token != "all":
        parts.append(str(selector_token).strip())
    parts.extend(visible_values(frame, "lcia_method")[:1])
    if include_impact:
        parts.extend(visible_values(frame, "impact")[:1])
    parts.extend(_scope_scenario_values(frame)[:1])
    parts.extend(visible_values(frame, "cc_category")[:1])
    cc_bounds = visible_values(frame, "cc_bound")
    if len(cc_bounds) == 1:
        parts.append(cc_bounds[0])
    if studied_year is not None:
        parts.append(str(int(studied_year)))
    model_pair = dynamic_model_scenario_token(frame)
    if model_pair is not None:
        parts.append(model_pair)
    return "__".join(sanitize_token(part) for part in parts if str(part).strip())


def dynamic_model_scenario_token(frame: pd.DataFrame) -> str | None:
    """Return the final deterministic dynamic model-scenario stem token."""
    return model_scenario_pair_token(
        models=visible_values(frame, "cc_model"),
        scenarios=visible_values(frame, "cc_scenario"),
    )


def scope_title(
    family_label: str,
    label: str | None,
    frame: pd.DataFrame,
    *,
    include_impact: bool,
    selector_title: str | None = None,
    studied_year: int | None = None,
) -> str:
    """Return one compact aCC figure title."""
    parts = [family_label]
    if label is not None:
        parts.append(label)
    if selector_title is not None and str(selector_title).strip():
        parts.append(str(selector_title).strip())
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


def _scope_scenario_values(frame: pd.DataFrame) -> list[str]:
    ar6_values = visible_values(frame, AR6_CC_SSP_SCENARIO_COLUMN)
    return ar6_values if ar6_values else visible_values(frame, ASOCC_SSP_SCENARIO_COLUMN)


def _ar6_category_scope(frame: pd.DataFrame) -> str:
    categories = visible_values(frame, "cc_category")
    if categories:
        return category_scope_label(categories)
    scopes = visible_values(frame, AR6_CATEGORY_SCOPE_COLUMN)
    return scopes[0] if scopes else ""


def _is_multi_category_scope(scope: str) -> bool:
    text = str(scope).strip()
    return "-" in text or "," in text


def requested_single_year(requested_years: list[int]) -> int | None:
    """Return the single requested year if this is a one year product."""
    return single_requested_year(requested_years)


def save_figure(fig: Any, *, output_stem: Path, output_format: str, dpi: int) -> list[Path]:
    """Save one matplotlib figure and return the written path."""
    path = output_file_path(base_path=output_stem, output_format=output_format)
    fig.savefig(path, dpi=int(dpi), bbox_inches="tight", format=output_format)
    return [path]


def has_static_min_max_bounds(frame: pd.DataFrame) -> bool:
    """Return whether one static aCC figure scope contains both static CC bounds."""
    return {str(value).strip() for value in visible_values(frame, "cc_bound")} == {
        MIN_CC_BOUND,
        MAX_CC_BOUND,
    }


def cc_bound_order_key(value: object) -> int:
    """Return the canonical display order for static aCC bounds."""
    text = str(value).strip()
    return {MIN_CC_BOUND: 0, MAX_CC_BOUND: 1}[text]


def cc_bound_layer_key(value: object) -> int:
    """Return the rendering order for overlaid static aCC bounds."""
    text = str(value).strip()
    return {MAX_CC_BOUND: 0, MIN_CC_BOUND: 1}[text]
