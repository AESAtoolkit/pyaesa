"""ASR threshold semantics for figure rendering."""

from dataclasses import dataclass

import pandas as pd

from pyaesa.shared.figures.lcia_metadata import load_lcia_metadata


@dataclass(frozen=True)
class AsrThresholdContract:
    """Visible ASR threshold labels for one figure scope."""

    has_max_threshold: bool
    threshold_group_title: str
    min_line_label: str
    max_line_label: str | None
    lower_zone_label: str
    middle_zone_label: str | None
    upper_zone_label: str
    fnt_label: str


def has_max_asr_threshold(*, frame: pd.DataFrame | None) -> bool:
    """Return whether one ASR figure scope should display the max threshold."""
    if frame is None or "cc_bound" not in frame.columns:
        return False
    values = _observed_cc_bounds(frame)
    return values == {"both"} or values == {"min_cc", "max_cc"}


def _observed_cc_bounds(frame: pd.DataFrame) -> set[str]:
    """Return the visible static CC bound values in one ASR figure scope."""
    return {
        str(value).strip()
        for value in pd.Series(frame["cc_bound"], copy=False).dropna().astype(str).tolist()
        if str(value).strip()
    }


def build_asr_threshold_contract(
    *,
    cc_source: str,
    has_max_threshold: bool,
) -> AsrThresholdContract:
    """Return the visible ASR threshold wording for one figure scope."""
    schema_kind = load_lcia_metadata(cc_source).schema_kind
    is_planetary_boundary = str(schema_kind).strip() == "planetary boundary"
    min_label = "Min SOS" if is_planetary_boundary else "Min CC"
    max_label = "Max SOS" if is_planetary_boundary else "Max CC"
    title_root = "Safe operating space" if is_planetary_boundary else "Carrying capacity"
    threshold_group_title = f"{title_root} threshold{'s' if has_max_threshold else ''}"
    lower_zone = "Safe operating space" if is_planetary_boundary else "Safe zone"
    middle_zone = "Zone of increasing risk" if has_max_threshold else None
    upper_zone = "High risk zone"
    fnt_label = (
        r"$f^{\mathrm{NT}}$ = frequency of no-transgression of Min SOS (ASR <= 1)"
        if is_planetary_boundary
        else r"$f^{\mathrm{NT}}$ = frequency of no-transgression of Min CC (ASR <= 1)"
    )
    return AsrThresholdContract(
        has_max_threshold=has_max_threshold,
        threshold_group_title=threshold_group_title,
        min_line_label=min_label,
        max_line_label=max_label if has_max_threshold else None,
        lower_zone_label=lower_zone,
        middle_zone_label=middle_zone,
        upper_zone_label=upper_zone,
        fnt_label=fnt_label,
    )
