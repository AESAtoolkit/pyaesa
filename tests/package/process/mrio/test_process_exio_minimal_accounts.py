import pandas as pd
from typing import Any, cast

from pyaesa.process.mrios.utils.parsers.exio_parser import (
    _calc_characterized_extensions_minimal,
)


class _FakeLciaExtension:
    def __init__(self, f: pd.DataFrame, f_y: pd.DataFrame) -> None:
        self.F = f
        self.F_Y = f_y
        self.S = None
        self.M = None
        self.D_pba = None
        self.D_pba_reg = None


class _FakeIO:
    def __init__(self) -> None:
        cols = pd.MultiIndex.from_tuples(
            [("R1", "S1"), ("R2", "S1")],
            names=["region", "sector"],
        )
        self.x = pd.DataFrame({"indout": [10.0, 20.0]}, index=cols)
        self.L = pd.DataFrame(
            [[1.0, 0.0], [0.0, 1.0]],
            index=cols,
            columns=cols,
        )
        f = pd.DataFrame([[1.0, 2.0]], index=["impact_1"], columns=cols)
        f_y_cols = pd.MultiIndex.from_tuples(
            [("R1", "FD"), ("R2", "FD")],
            names=["region", "final_demand"],
        )
        f_y = pd.DataFrame([[0.5, 1.5]], index=["impact_1"], columns=f_y_cols)
        self.pb_lcia = _FakeLciaExtension(f=f, f_y=f_y)


def test_calc_characterized_extensions_minimal_builds_required_accounts() -> None:
    iosys = _FakeIO()
    _calc_characterized_extensions_minimal(
        cast(Any, iosys),
        ["pb_lcia"],
        keep_direct_intensities=True,
    )

    ext = iosys.pb_lcia
    assert ext.S is not None
    assert ext.M is not None
    assert ext.D_pba is not None
    assert ext.D_pba_reg is not None

    expected_m = pd.DataFrame(
        [[0.1, 0.1]],
        index=["impact_1"],
        columns=ext.M.columns,
    )
    pd.testing.assert_frame_equal(ext.M, expected_m)

    expected_d_pba = pd.DataFrame(
        [[1.0, 2.0]],
        index=["impact_1"],
        columns=ext.D_pba.columns,
    )
    pd.testing.assert_frame_equal(ext.D_pba, expected_d_pba)

    expected_d_pba_reg = pd.DataFrame(
        [[1.5, 3.5]],
        index=["impact_1"],
        columns=pd.Index(["R1", "R2"], name="region"),
    )
    pd.testing.assert_frame_equal(ext.D_pba_reg, expected_d_pba_reg)
