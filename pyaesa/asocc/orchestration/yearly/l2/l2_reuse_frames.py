"""Historical reuse row index assembly for L2 outputs."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ....methods.equations.ar_result_indexing import _attach_trailing_constant_levels


def _attach_l2_reuse_year_level(
    *,
    frame: pd.DataFrame,
    l2_reuse_year: int | None,
) -> pd.DataFrame:
    """Return one frame with row owned historical reuse identity when needed."""
    if l2_reuse_year is None:
        return frame
    return _attach_trailing_constant_levels(
        result=frame,
        levels=(("l2_reuse_year", int(l2_reuse_year)),),
    )


def _base_index_components(
    index: pd.Index,
    repeat_count: int,
) -> tuple[list[pd.Index], list[np.ndarray], list[object]]:
    """Return tiled base index components for L2 reuse year concatenation."""
    if isinstance(index, pd.MultiIndex):
        levels = [pd.Index(level, copy=False) for level in index.levels]
        codes = [np.tile(np.asarray(code, dtype=np.intp), repeat_count) for code in index.codes]
        names = list(index.names)
        return levels, codes, names
    base_codes, base_levels = pd.factorize(index, sort=False)
    return (
        [pd.Index(base_levels, name=index.name, copy=False)],
        [np.tile(np.asarray(base_codes, dtype=np.intp), repeat_count)],
        [index.name],
    )


def _combine_l2_reuse_year_frames(
    *,
    frames_by_l2_reuse_year: Sequence[tuple[int, pd.DataFrame]],
    reference_year: int | None = None,
) -> pd.DataFrame:
    """Return one frame containing all historical reuse rows for a target year."""
    first_frame = frames_by_l2_reuse_year[0][1]
    l2_reuse_years = [int(l2_reuse_year) for l2_reuse_year, frame in frames_by_l2_reuse_year]
    values = np.concatenate(
        [item[1].to_numpy(dtype=np.float64, copy=False) for item in frames_by_l2_reuse_year],
        axis=0,
    )
    return _l2_reuse_year_frame_from_values(
        base_index=first_frame.index,
        l2_reuse_years=l2_reuse_years,
        values=values,
        columns=first_frame.columns,
        reference_year=reference_year,
    )


def _l2_reuse_year_frame_from_values(
    *,
    base_index: pd.Index,
    l2_reuse_years: Sequence[int],
    values: np.ndarray,
    columns: pd.Index,
    reference_year: int | None = None,
) -> pd.DataFrame:
    """Return one frame from row stacked values and L2 reuse year identity."""
    base_size = len(base_index)
    levels, codes, names = _base_index_components(
        index=base_index,
        repeat_count=len(l2_reuse_years),
    )
    if reference_year is not None:
        levels.append(pd.Index([int(reference_year)], name="reference_year"))
        codes.append(np.zeros(base_size * len(l2_reuse_years), dtype=np.intp))
        names.append("reference_year")
    levels.append(pd.Index(l2_reuse_years, name="l2_reuse_year"))
    codes.append(np.repeat(np.arange(len(l2_reuse_years), dtype=np.intp), base_size))
    names.append("l2_reuse_year")
    index = pd.MultiIndex(
        levels=levels,
        codes=codes,
        names=names,
        verify_integrity=False,
    )
    return pd.DataFrame(values, index=index, columns=columns.copy())
