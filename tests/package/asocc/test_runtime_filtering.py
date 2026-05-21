import pandas as pd
import numpy as np

from pyaesa.asocc.runtime.scope.filtering import (
    normalize_filter_values,
    slice_frame_any_axis,
    slice_series_any_axis,
)


def test_normalize_filter_values() -> None:
    assert normalize_filter_values(None) is None
    assert normalize_filter_values([]) is None
    assert normalize_filter_values(["A", "1", "A"]) == {"A", "1"}


def test_slice_frame_any_axis_on_multiindex_index_and_columns() -> None:
    idx = pd.MultiIndex.from_product([["EU", "US"], ["s1", "s2"]], names=["r_p", "s_p"])
    cols = pd.MultiIndex.from_product([["EU", "US"], ["fd"]], names=["r_f", "flow"])
    frame = pd.DataFrame(
        np.arange(len(idx) * len(cols)).reshape(len(idx), len(cols)),
        index=idx,
        columns=cols,
    )

    out = slice_frame_any_axis(
        frame,
        axis_name="r_p",
        allowed={"EU"},
    )
    assert set(out.index.get_level_values("r_p")) == {"EU"}

    out2 = slice_frame_any_axis(
        frame,
        axis_name="r_f",
        allowed={"US"},
    )
    assert set(out2.columns.get_level_values("r_f")) == {"US"}


def test_slice_frame_any_axis_on_single_index_and_columns() -> None:
    frame = pd.DataFrame(
        {"EU": [1, 2], "US": [3, 4]},
        index=pd.Index(["EU", "US"], name="r_p"),
    )
    frame.columns.name = "r_f"

    out = slice_frame_any_axis(frame, axis_name="r_p", allowed={"EU"})
    assert list(out.index) == ["EU"]
    out = slice_frame_any_axis(frame, axis_name="r_f", allowed={"US"})
    assert list(out.columns) == ["US"]
    assert slice_frame_any_axis(frame, axis_name="x", allowed={"EU"}).equals(frame)
    assert slice_frame_any_axis(frame, axis_name="r_p", allowed=None).equals(frame)


def test_slice_series_any_axis_multi_and_single() -> None:
    multi_index = pd.MultiIndex.from_product([["EU", "US"], ["s1", "s2"]], names=["r_p", "s_p"])
    s_multi = pd.Series([1, 2, 3, 4], index=multi_index)

    out = slice_series_any_axis(s_multi, axis_name="r_p", allowed={"EU"})
    assert set(out.index.get_level_values("r_p")) == {"EU"}
    assert slice_series_any_axis(s_multi, axis_name="unknown", allowed={"EU"}).equals(s_multi)

    s_single = pd.Series([1, 2], index=pd.Index(["EU", "US"], name="r_p"))
    out = slice_series_any_axis(s_single, axis_name="r_p", allowed={"US"})
    assert list(out.index) == ["US"]
    assert slice_series_any_axis(s_single, axis_name="x", allowed={"EU"}).equals(s_single)
    assert slice_series_any_axis(s_single, axis_name="r_p", allowed=None).equals(s_single)
