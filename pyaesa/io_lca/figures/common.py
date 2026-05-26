"""Common IO-LCA figure helpers for deterministic and uncertainty products."""

from collections.abc import Iterator
import re
from typing import cast

import pandas as pd

from pyaesa.shared.figures.layout import multi_impact_panel_figure_size, single_impact_figure_size
from pyaesa.shared.figures.lcia_metadata import (
    load_lcia_metadata,
    resolve_frame_impact_unit,
)
from pyaesa.shared.figures.contracts import resolved_selector_columns
from pyaesa.shared.figures.multi_year_transitions import TransitionMarker
from pyaesa.shared.runtime.scenario.columns import EXT_LCA_SSP_SCENARIO_COLUMN
from pyaesa.shared.figures.scientific_text import format_scientific_figure_text
from pyaesa.shared.selectors.path_tokens import selector_scope_token_from_frame
from pyaesa.shared.tabular.scalars import display_scalar, sanitize_token

TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")
SELECTOR_COLUMNS = ("r_f", "r_c", "r_p", "s_p")
_TRANSITION_COLOR = "#7d7d7d"


def lcia_method_tag(lcia_method: str) -> str:
    """Return a deterministic file tag for one LCIA method."""
    text = TOKEN_RE.sub("_", str(lcia_method).strip())
    text = text.strip("._-")
    return text or "lcia_method"


def ordered_impacts(*, frame: pd.DataFrame, lcia_method: str) -> tuple[list[str], dict[str, str]]:
    """Return metadata ordered impacts and display labels for one LCIA method."""
    available = sorted({str(value) for value in frame["impact"].dropna().astype(str).tolist()})
    meta = load_lcia_metadata(lcia_method)
    ordered = [impact for impact in meta.impacts if impact in set(available)]
    labels = {impact: meta.labels[impact] for impact in ordered}
    return ordered, labels


def panel_impact_unit(*, frame: pd.DataFrame) -> str:
    """Return one y axis label carrying the impact unit for an impact panel."""
    unit = resolve_frame_impact_unit(frame)
    return "" if unit is None else format_scientific_figure_text(str(unit).strip())


def selector_scope_token(
    *,
    group_frame: pd.DataFrame,
    selector_cols: list[str],
    reference_frame: pd.DataFrame | None = None,
) -> str:
    """Return one deterministic selector scope token from a grouped figure frame."""
    if not selector_cols:
        return "all_selectors"
    return selector_scope_token_from_frame(
        group_frame=group_frame,
        selector_columns=selector_cols,
        reference_frame=reference_frame,
    )


def figure_stem(
    *,
    lcia_method: str,
    selector_scope_token: str,
    year: int | None = None,
    scenario_token: str | None = None,
    stem_prefix: str | None = None,
) -> str:
    """Return a deterministic IO-LCA figure stem."""
    parts = [
        str(stem_prefix).strip() if stem_prefix is not None else "",
        str(lcia_method).strip(),
        str(selector_scope_token).strip(),
    ]
    if scenario_token is not None and str(scenario_token).strip() not in {"", "all"}:
        parts.append(str(scenario_token).strip())
    if year is not None:
        parts.append(str(int(year)))
    stem = "__".join(part for part in parts if part)
    return stem or "figure"


def normalize_plot_years(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Return figure rows with canonical integer year values."""
    raw_years = cast(pd.Series, frame["year"])
    trimmed_years = cast(pd.Series, raw_years.astype("string").str.strip())
    valid_mask = cast(pd.Series, trimmed_years.notna() & trimmed_years.ne(""))
    candidate_years = cast(pd.Series, trimmed_years.loc[valid_mask])
    numeric_years = cast(pd.Series, pd.to_numeric(candidate_years, errors="raise"))
    out = frame.loc[valid_mask].copy()
    out["year"] = numeric_years.astype(int).to_numpy()
    return out


def selector_groups(
    *,
    frame: pd.DataFrame,
    selector_columns: tuple[str, ...] | None,
) -> tuple[list[str], Iterator[tuple[object, pd.DataFrame]]]:
    """Return selector columns and yield grouped figure frames."""
    selector_source = SELECTOR_COLUMNS if selector_columns is None else selector_columns
    selector_cols = list(
        resolved_selector_columns(
            frame,
            selector_columns=selector_source,
            require_non_null=True,
        )
    )

    def grouped_frames() -> Iterator[tuple[object, pd.DataFrame]]:
        """Yield one selector grouped frame at a time."""
        if selector_cols:
            for idx, group in frame.groupby(selector_cols, dropna=False, sort=True):
                yield idx, group.copy()
            return
        yield (None,), frame.copy()

    return selector_cols, grouped_frames()


def impact_panel_layout(
    *,
    impacts_count: int,
    single_year: bool = False,
) -> dict[str, float | int | str]:
    """Resolve one or two column impact panel geometry for LCA figures."""
    count = max(1, int(impacts_count))
    if count == 1:
        width, height = single_impact_figure_size(single_year=single_year)
        return {
            "layout": "single",
            "ncols": 1,
            "nrows": 1,
            "fig_width": width,
            "fig_height": height,
        }
    ncols = 2
    nrows = (count + 1) // ncols
    fig_width, fig_height = multi_impact_panel_figure_size(
        nrows=nrows,
        compact=single_year,
    )
    return {
        "layout": "double",
        "ncols": ncols,
        "nrows": nrows,
        "fig_width": fig_width,
        "fig_height": fig_height,
    }


def lca_prospective_scope_slices(
    frame: pd.DataFrame,
) -> Iterator[tuple[str | None, str | None, pd.DataFrame]]:
    """Yield historical or SSP scoped LCA figure frames.

    External LCA owns prospective scope from its SSP row metadata. Historical
    rows are repeated into each SSP scoped figure so the plotted trajectory is
    continuous through the retrospective period. IO-LCA frames do not carry SSP
    metadata and therefore render once without transition markers.
    """
    scenario = lca_scenario_series(frame)
    values = sorted({value for value in scenario.tolist() if value is not None})
    if not values:
        yield None, None, frame.copy()
        return
    generic = scenario.isna()
    for value in values:
        mask = scenario.eq(value) | generic
        scoped = frame.loc[mask].copy()
        yield sanitize_token(value), f"Prospective: {value}", scoped.reset_index(drop=True)


def lca_transition_markers(frame: pd.DataFrame) -> list[TransitionMarker]:
    """Return the external LCA retrospective to prospective marker for a frame."""
    scenario = lca_scenario_series(frame)
    prospective = frame.loc[scenario.notna()].copy()
    if prospective.empty:
        return []
    years = cast(
        pd.Series,
        pd.to_numeric(pd.Series(prospective["year"], copy=False), errors="raise"),
    ).astype(int)
    return [
        TransitionMarker(
            year=int(years.min()),
            label="retrospective/prospective transition",
            color=_TRANSITION_COLOR,
        )
    ]


def lca_scenario_series(frame: pd.DataFrame) -> pd.Series:
    """Return display normalized external LCA SSP labels for one figure frame."""
    for column in (EXT_LCA_SSP_SCENARIO_COLUMN, "ssp_scenario"):
        if column in frame.columns:
            return pd.Series(
                [display_scalar(value) for value in frame[column].tolist()],
                index=frame.index,
                dtype="object",
            )
    return pd.Series([None] * len(frame), index=frame.index, dtype="object")
