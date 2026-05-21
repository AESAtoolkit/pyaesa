"""Shared deterministic downstream aSoCC share compatibility selection."""

from typing import cast

import pandas as pd

from pyaesa.shared.lcia.path_tokens import infer_lcia_method_from_path
from pyaesa.shared.lcia.contracts import dynamic_cc_match

from .inputs import AsoccShare, LoadedAsoccShare, asocc_share_reference_path
from .tabular_io import requested_year_columns

DownstreamAsoccShare = AsoccShare | LoadedAsoccShare


def static_compatible_share_frame(
    *,
    asocc_share: DownstreamAsoccShare,
    share_frame: pd.DataFrame,
    cc_source: str,
) -> pd.DataFrame | None:
    """Return one static-CC-compatible share frame or ``None`` for mismatched shares."""
    declared_method = asocc_share_declared_lcia_method(
        asocc_share=asocc_share,
        share_frame=share_frame,
    )
    if declared_method is None:
        return share_frame.copy()
    if declared_method != str(cc_source).strip():
        return None
    return share_frame.copy()


def dynamic_compatible_share_frame(
    *,
    asocc_share: DownstreamAsoccShare,
    share_frame: pd.DataFrame,
    lcia_method: str | None,
    requested_years: list[int] | None = None,
) -> pd.DataFrame | None:
    """Return one dynamic-CC-compatible share frame or ``None`` for incompatible shares."""
    target_method = str(lcia_method).strip() if lcia_method is not None else ""
    prepared = share_frame.copy()
    if target_method:
        target_match = cast(dict[str, str], dynamic_cc_match(lcia_method=target_method))
        expected_impact = str(target_match["impact"]).strip()

        if asocc_share.impacts:
            share_impacts = {
                str(value).strip() for value in asocc_share.impacts if str(value).strip()
            }
            if expected_impact not in share_impacts:
                return None
        else:
            declared_impacts = _non_empty_column_values(prepared, "impact")
            if declared_impacts:
                if expected_impact not in declared_impacts:
                    return None
                impact_series = pd.Series(prepared["impact"], copy=False).astype(str).str.strip()
                prepared = prepared.loc[impact_series.eq(expected_impact)].copy()

        declared_method = asocc_share_declared_lcia_method(
            asocc_share=asocc_share,
            share_frame=prepared,
        )
        if declared_method is not None:
            if declared_method != target_method:
                return None

    if requested_years is not None and not requested_year_columns(
        prepared,
        requested_years=[int(year) for year in requested_years],
    ):
        return None
    return prepared


def asocc_share_declared_lcia_method(
    *,
    asocc_share: DownstreamAsoccShare,
    share_frame: pd.DataFrame,
) -> str | None:
    """Return the one canonical deterministic LCIA method for one aSoCC share."""
    explicit_methods = _non_empty_column_values(share_frame, "lcia_method")
    if len(explicit_methods) > 1:
        observed_methods = sorted(explicit_methods)
        raise ValueError(
            "Deterministic downstream aSoCC share selection requires exactly one upstream "
            f"'lcia_method'. Source='{asocc_share.display_name}', observed={observed_methods}."
        )
    explicit_method = next(iter(explicit_methods), None)
    inferred_method = _inferred_lcia_method(asocc_share=asocc_share)
    has_conflict = (
        explicit_method is not None
        and inferred_method is not None
        and explicit_method != inferred_method
    )
    if has_conflict:
        raise ValueError(
            "Deterministic downstream aSoCC share selection found conflicting file-stem and table "
            f"'lcia_method' identities for source '{asocc_share.display_name}': "
            f"stem='{inferred_method}', table='{explicit_method}'."
        )
    return inferred_method or explicit_method


def _inferred_lcia_method(*, asocc_share: DownstreamAsoccShare) -> str | None:
    """Return the file-stem-owned LCIA method for one aSoCC share when present."""
    reference_path = (
        asocc_share.reference_path
        if isinstance(asocc_share, LoadedAsoccShare)
        else asocc_share_reference_path(asocc_share)
    )
    inferred = infer_lcia_method_from_path(reference_path)
    if inferred is None:
        return None
    text = str(inferred).strip()
    return text or None


def _non_empty_column_values(frame: pd.DataFrame, column: str) -> set[str]:
    """Return one normalized set of non-empty scalar values from a column."""
    if column not in frame.columns:
        return set()
    return {
        str(value).strip()
        for value in frame[column].dropna().astype(str).tolist()
        if str(value).strip()
    }
