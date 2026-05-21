"""SSP label and transition ownership for deterministic downstream aSoCC shares."""

import pandas as pd

from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.deterministic_companion_stems import (
    parse_deterministic_companion_stem,
)
from pyaesa.shared.tabular.scalars import sanitize_token
from pyaesa.shared.tabular.wide_tables import detect_year_columns
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens

from .inputs import LoadedAsoccShare


def asocc_share_ssp_scenario_labels(
    asocc_share: LoadedAsoccShare,
    *,
    frame_wide: pd.DataFrame | None = None,
) -> set[str]:
    """Return non null SSP labels represented by one deterministic aSoCC share."""
    frame = frame_wide if frame_wide is not None else asocc_share.frame_wide
    if frame is not None and ASOCC_SSP_SCENARIO_COLUMN in frame.columns:
        year_columns = _active_year_columns(frame)
        if not year_columns:
            return set()
        active_rows = frame.loc[
            frame.loc[:, year_columns].notna().any(axis=1),
            ASOCC_SSP_SCENARIO_COLUMN,
        ]
        series = pd.Series(active_rows, copy=False)
        values = [str(value).strip() for value in series.dropna().tolist() if str(value).strip()]
        return set(normalize_ssp_tokens(values))
    return set()


def share_transition_metadata(
    *,
    asocc_shares: list[LoadedAsoccShare],
    scenario_tokens: list[str],
    switch_label: str = "Switch year for SSP-dependent series",
    marker_color: str = "#7d7d7d",
) -> dict[str, dict[str, object]]:
    """Return per share-stem SSP transition metadata for downstream consumers."""
    resolved_asocc_shares: list[tuple[LoadedAsoccShare, str, str | None, list[str], set[int]]] = []
    historical_years_by_base: dict[str, set[int]] = {}
    for asocc_share in asocc_shares:
        ssp_scenario_labels = sorted(
            asocc_share_ssp_scenario_labels(
                asocc_share,
            )
        )
        stem_identity = parse_deterministic_companion_stem(
            asocc_share.file_stem,
            scenario_tokens=[*scenario_tokens, *ssp_scenario_labels],
        )
        base_stem = stem_identity.base_stem
        ssp_scenario = ssp_scenario_labels[0] if ssp_scenario_labels else None
        covered_years = _asocc_share_years(asocc_share)
        if ssp_scenario is None:
            historical_years_by_base.setdefault(base_stem, set()).update(covered_years)
        resolved_asocc_shares.append(
            (
                asocc_share,
                base_stem,
                ssp_scenario,
                sorted(ssp_scenario_labels),
                covered_years,
            )
        )
    out: dict[str, dict[str, object]] = {}
    for asocc_share, base_stem, ssp_scenario, labels, covered_years in resolved_asocc_shares:
        out[asocc_share.file_stem] = {
            "base_stem": base_stem,
            ASOCC_SSP_SCENARIO_COLUMN: ssp_scenario,
            "asocc_ssp_scenario_labels": labels,
            "ssp_start_year": _asocc_share_ssp_start_year(
                ssp_scenario=ssp_scenario,
                ssp_scenario_labels=set(labels),
                covered_years=covered_years,
                historical_years=historical_years_by_base.get(base_stem, set()),
            ),
            "marker_label": switch_label,
            "marker_color": marker_color,
        }
    return out


def _asocc_share_years(asocc_share: LoadedAsoccShare) -> set[int]:
    """Return detected wide year coverage for one deterministic aSoCC share."""
    return {int(column) for column in _active_year_columns(asocc_share.frame_wide)}


def _active_year_columns(frame: pd.DataFrame) -> list[str]:
    """Return year columns with at least one persisted value."""
    year_columns = detect_year_columns(frame)
    return [
        str(column)
        for column in year_columns
        if bool(pd.Series(frame.loc[:, column], copy=False).notna().any())
    ]


def _asocc_share_ssp_start_year(
    *,
    ssp_scenario: str | None,
    ssp_scenario_labels: set[str],
    covered_years: set[int],
    historical_years: set[int],
) -> int | None:
    """Return the first prospective year for one SSP-tagged aSoCC share."""
    if ssp_scenario is None or str(ssp_scenario) not in ssp_scenario_labels or not covered_years:
        return None
    prospective_years = sorted(
        int(year) for year in covered_years if int(year) not in historical_years
    )
    if prospective_years:
        return prospective_years[0]
    return None


def share_transition_payload_for_output_stem(
    *,
    output_stem: str,
    share_transition_meta: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Return the resolved share transition payload for one downstream output stem."""
    stem = str(output_stem).strip()
    if not stem:
        return {}
    if stem in share_transition_meta:
        return dict(share_transition_meta[stem])
    candidates = [
        key
        for key in share_transition_meta
        if str(key).strip()
        and (stem == str(key).strip() or stem.startswith(f"{str(key).strip()}__"))
    ]
    if not candidates:
        normalized_stem = _normalized_transition_stem(stem)
        candidates = [
            key
            for key in share_transition_meta
            if str(key).strip()
            and (
                normalized_stem == _normalized_transition_stem(str(key).strip())
                or normalized_stem.startswith(f"{_normalized_transition_stem(str(key).strip())}__")
            )
        ]
    if not candidates:
        return {}
    selected = max(
        candidates,
        key=lambda key: len(_normalized_transition_stem(key)),
    )
    return dict(share_transition_meta[selected])


def _normalized_transition_stem(stem: str) -> str:
    """Return one sanitized deterministic stem for downstream aSoCC share matching."""
    return "__".join(sanitize_token(piece) for piece in str(stem).split("__") if str(piece).strip())
