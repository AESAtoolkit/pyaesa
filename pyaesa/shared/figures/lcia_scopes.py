"""Shared LCIA figure scope slicing."""

from collections.abc import Iterator

import pandas as pd


def lcia_impact_slices(frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield LCIA method and impact scoped figure frames."""
    has_lcia_method = _has_visible_values(frame, "lcia_method")
    has_impact = _has_visible_values(frame, "impact")
    if not has_lcia_method or not has_impact:
        yield frame.copy()
        return
    values = sorted(
        {
            (str(lcia_method).strip(), str(impact).strip())
            for lcia_method, impact in frame[["lcia_method", "impact"]].itertuples(
                index=False,
                name=None,
            )
            if not pd.isna(lcia_method)
            and str(lcia_method).strip()
            and not pd.isna(impact)
            and str(impact).strip()
        }
    )
    for lcia_method, impact in values:
        yield frame.loc[
            frame["lcia_method"].astype(str).eq(lcia_method)
            & frame["impact"].astype(str).eq(impact)
        ].copy()


def _has_visible_values(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and any(
        not pd.isna(value) and str(value).strip() for value in frame[column].tolist()
    )
