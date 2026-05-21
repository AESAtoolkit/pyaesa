"""ASR multi year retrospective to prospective transition policy."""

from collections.abc import Iterable
from typing import cast

import pandas as pd

from pyaesa.shared.figures.asocc_transition_policy import (
    ASOCC_TRANSITION_SERIES_EXCLUDED_COLUMNS,
    asocc_transition_year,
)
from pyaesa.shared.figures.multi_year_transitions import TransitionMarker
from pyaesa.shared.runtime.scenario.columns import (
    EXT_LCA_SSP_SCENARIO_COLUMN,
    LCA_SSP_START_YEAR_COLUMN,
)
from pyaesa.shared.tabular.scalars import is_display_missing

GENERIC_ASR_TRANSITION_LABEL = "retrospective/prospective transition"
ASR_ASOCC_TRANSITION_LABEL = "aSoCC"
ASR_LCA_TRANSITION_LABEL = "LCA"

ASR_TRANSITION_SERIES_EXCLUDED_COLUMNS = frozenset(
    {
        *ASOCC_TRANSITION_SERIES_EXCLUDED_COLUMNS,
        EXT_LCA_SSP_SCENARIO_COLUMN,
        "asocc_prospective_start_year",
        LCA_SSP_START_YEAR_COLUMN,
    }
)


def asr_transition_markers(group: pd.DataFrame) -> list[TransitionMarker]:
    """Return ASR transition markers from aSoCC route and LCA SSP axes."""
    asocc_year = asocc_transition_year(group)
    lca_year = _lca_transition_year(group)
    return _paired_transition_markers(
        first_year=asocc_year,
        first_label=ASR_ASOCC_TRANSITION_LABEL,
        second_year=lca_year,
        second_label=ASR_LCA_TRANSITION_LABEL,
    )


def merged_asr_transition_markers(groups: Iterable[pd.DataFrame]) -> list[TransitionMarker]:
    """Return one marker set from the visible ASR plotted series."""
    markers: dict[int, TransitionMarker] = {}
    for group in groups:
        for marker in asr_transition_markers(group):
            markers[int(marker.year)] = marker
    return list(markers.values())


def _paired_transition_markers(
    *,
    first_year: int | None,
    first_label: str,
    second_year: int | None,
    second_label: str,
) -> list[TransitionMarker]:
    if first_year is None and second_year is None:
        return []
    if first_year is None:
        return [
            _transition_marker(
                year=int(cast(int, second_year)),
                label=GENERIC_ASR_TRANSITION_LABEL,
            )
        ]
    if second_year is None:
        return [_transition_marker(year=int(first_year), label=GENERIC_ASR_TRANSITION_LABEL)]
    if int(first_year) == int(second_year):
        return [_transition_marker(year=int(first_year), label=GENERIC_ASR_TRANSITION_LABEL)]
    return [
        _transition_marker(year=int(first_year), label=str(first_label)),
        _transition_marker(year=int(second_year), label=str(second_label)),
    ]


def _transition_marker(*, year: int, label: str) -> TransitionMarker:
    return TransitionMarker(year=int(year), label=str(label), color="#7d7d7d")


def _lca_transition_year(group: pd.DataFrame) -> int | None:
    if LCA_SSP_START_YEAR_COLUMN in group.columns:
        return _lca_transition_year_from_start_column(group)
    if EXT_LCA_SSP_SCENARIO_COLUMN not in group.columns:
        return None
    years = pd.Series(pd.to_numeric(group["year"], errors="raise"), copy=False).astype(int)
    prospective_by_year = (
        (~group[EXT_LCA_SSP_SCENARIO_COLUMN].map(is_display_missing))
        .groupby(years, sort=True)
        .any()
    )
    if not bool(prospective_by_year.any()) or bool(prospective_by_year.all()):
        return None
    return int(prospective_by_year.loc[prospective_by_year].index.min())


def _lca_transition_year_from_start_column(group: pd.DataFrame) -> int | None:
    starts = pd.Series(
        pd.to_numeric(group[LCA_SSP_START_YEAR_COLUMN], errors="coerce"),
        copy=False,
    ).dropna()
    if starts.empty:
        return None
    start_year = int(starts.astype(int).min())
    years = pd.Series(pd.to_numeric(group["year"], errors="raise"), copy=False).astype(int)
    if int(years.min()) < start_year <= int(years.max()):
        return start_year
    return None
