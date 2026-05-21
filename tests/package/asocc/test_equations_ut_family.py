import pandas as pd
import pytest

from pyaesa.asocc.methods.equations import ut_fd
from pyaesa.asocc.methods.equations import ut_fda
from pyaesa.asocc.methods.equations import ut_gva
from pyaesa.asocc.methods.equations import ut_gvaa
from pyaesa.asocc.methods.equations import ut_support
from pyaesa.asocc.methods.equations import ut_td


def _base_inputs():
    fd_rf = pd.Series([4.0, 6.0], index=pd.Index(["FR", "US"], name="r_f"))
    fd_rp_sp_rf = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
        columns=pd.Index(["FR", "US"], name="r_f"),
    )
    fd_rp_sp = pd.Series(
        [3.0, 7.0],
        index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
    )
    fd_rf_sp = pd.Series(
        [2.0, 8.0],
        index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_f", "s_p"]),
    )

    x_to_rc = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
        columns=pd.Index(["FR", "US"], name="r_c"),
    )
    kappa = pd.DataFrame(
        [[0.5, 0.5], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]],
        index=pd.MultiIndex.from_tuples(
            [
                ("FR", "A", "FR"),
                ("FR", "A", "US"),
                ("US", "A", "FR"),
                ("US", "A", "US"),
            ],
            names=["r_p", "s_p", "r_c"],
        ),
        columns=pd.Index(["FR", "US"], name="r_f"),
    )

    gva_rp = pd.Series([5.0, 5.0], index=pd.Index(["FR", "US"], name="r_p"))
    gva_rp_sp = pd.Series(
        [2.0, 8.0],
        index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
    )

    omega_reg = pd.DataFrame(
        [[0.6, 0.4], [0.4, 0.6]],
        index=pd.Index(["FR", "US"], name="r_u"),
        columns=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
    )
    return {
        "fd_rf": fd_rf,
        "fd_rp_sp_rf": fd_rp_sp_rf,
        "fd_rp_sp": fd_rp_sp,
        "fd_rf_sp": fd_rf_sp,
        "x_to_rc": x_to_rc,
        "kappa": kappa,
        "gva_rp": gva_rp,
        "gva_rp_sp": gva_rp_sp,
        "omega_reg": omega_reg,
    }


def test_ut_support_safe_divide_get_x_and_stack() -> None:
    numer = pd.Series([2.0, 4.0], index=["a", "b"])
    out_scalar = ut_support._safe_divide_series(numer, 0.0)
    assert out_scalar.isna().all()
    zero_scalar = ut_support._safe_divide_series(pd.Series([0.0, 0.0], index=["a", "b"]), 0.0)
    assert zero_scalar.tolist() == [0.0, 0.0]

    out_series = ut_support._safe_divide_series(
        numer,
        pd.Series([2.0, 0.0], index=["a", "b"]),
    )
    assert float(out_series.loc["a"]) == 1.0
    assert pd.isna(out_series.loc["b"])
    zero_series = ut_support._safe_divide_series(
        pd.Series([0.0, 4.0], index=["a", "b"]),
        pd.Series([0.0, 0.0], index=["a", "b"]),
    )
    assert float(zero_series.loc["a"]) == 0.0
    assert pd.isna(zero_series.loc["b"])

    x_to_rc = _base_inputs()["x_to_rc"]
    x_vec = ut_support._get_x_vec(x_to_rc)
    assert float(x_vec.loc[("FR", "A")]) == 3.0

    stacked = ut_support._stack_to_year(
        pd.DataFrame(
            [[0.2, 0.8]],
            index=pd.Index(["FR"], name="r_p"),
            columns=pd.Index(["FR", "US"], name="r_f"),
        ),
        2020,
        "r_f",
    )
    assert list(stacked.columns) == [2020]
    assert stacked.index.names == ["r_p", "r_f"]


def test_compute_ut_fd_l2_paths_and_error() -> None:
    p = _base_inputs()
    w_rf = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_f"))

    pre_a = ut_fd.compute_ut_fd_l2(
        fu_code="L2.a.a",
        year=2020,
        l1_weights=None,
        fd_rf=p["fd_rf"],
        fd_rp_sp_rf=p["fd_rp_sp_rf"],
        fd_rp_sp=p["fd_rp_sp"],
        fd_rf_sp=p["fd_rf_sp"],
        pre_weighting=True,
    )
    assert pre_a.index.names == ["r_p", "s_p", "r_f"]

    out_a = ut_fd.compute_ut_fd_l2(
        fu_code="L2.a.a",
        year=2020,
        l1_weights=w_rf,
        fd_rf=p["fd_rf"],
        fd_rp_sp_rf=p["fd_rp_sp_rf"],
        fd_rp_sp=p["fd_rp_sp"],
        fd_rf_sp=p["fd_rf_sp"],
        pre_weighting=False,
    )
    assert list(out_a.columns) == [2020]

    out_b = ut_fd.compute_ut_fd_l2(
        fu_code="L2.b.a",
        year=2020,
        l1_weights=None,
        fd_rf=p["fd_rf"],
        fd_rp_sp_rf=p["fd_rp_sp_rf"],
        fd_rp_sp=p["fd_rp_sp"],
        fd_rf_sp=p["fd_rf_sp"],
        pre_weighting=False,
    )
    assert out_b.index.names == ["r_p", "s_p", "r_f"]

    out_c = ut_fd.compute_ut_fd_l2(
        fu_code="L2.c.a",
        year=2020,
        l1_weights=w_rf,
        fd_rf=p["fd_rf"],
        fd_rp_sp_rf=p["fd_rp_sp_rf"],
        fd_rp_sp=p["fd_rp_sp"],
        fd_rf_sp=p["fd_rf_sp"],
        pre_weighting=False,
    )
    assert list(out_c.columns) == [2020]

    out_c_no_l1 = ut_fd.compute_ut_fd_l2(
        fu_code="L2.c.a",
        year=2020,
        l1_weights=None,
        fd_rf=p["fd_rf"],
        fd_rp_sp_rf=p["fd_rp_sp_rf"],
        fd_rp_sp=p["fd_rp_sp"],
        fd_rf_sp=p["fd_rf_sp"],
        pre_weighting=False,
    )
    assert list(out_c_no_l1.columns) == [2020]


def test_ut_fda_adjusters_and_compute_paths() -> None:
    p = _base_inputs()
    w_rf = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_f"))

    with pytest.raises(ValueError):
        ut_fda._adjust_td_to_fd(pd.DataFrame(index=p["x_to_rc"].index), p["kappa"])

    contrib_rc = ut_fda._adjust_td_to_fd_by_rc(p["x_to_rc"], p["kappa"])
    assert isinstance(contrib_rc.index, pd.MultiIndex)
    assert contrib_rc.index.names == ["r_c", "r_p", "s_p"]

    contrib_rc_sp = ut_fda._adjust_td_to_fd_by_rc_sp(p["x_to_rc"], p["kappa"])
    assert contrib_rc_sp.index.names == ["r_c", "s_p"]

    pre = ut_fda.compute_ut_fda_l2(
        fu_code="L2.a.b",
        year=2020,
        l1_weights=None,
        fd_rf=p["fd_rf"],
        x_to_rc=p["x_to_rc"],
        kappa=p["kappa"],
        pre_weighting=True,
    )
    assert pre.index.names == ["r_p", "s_p", "r_f"]

    out = ut_fda.compute_ut_fda_l2(
        fu_code="L2.b.b",
        year=2020,
        l1_weights=w_rf,
        fd_rf=p["fd_rf"],
        x_to_rc=p["x_to_rc"],
        kappa=p["kappa"],
        pre_weighting=False,
    )
    assert list(out.columns) == [2020]


def test_compute_ut_gva_paths() -> None:
    p = _base_inputs()
    w_rp = pd.Series([0.5, 0.5], index=pd.Index(["FR", "US"], name="r_p"))

    pre = ut_gva.compute_ut_gva_l2(
        year=2020,
        l1_weights=None,
        gva_rp=p["gva_rp"],
        gva_rp_sp=p["gva_rp_sp"],
        pre_weighting=True,
    )
    assert list(pre.columns) == [2020]

    out = ut_gva.compute_ut_gva_l2(
        year=2020,
        l1_weights=w_rp,
        gva_rp=p["gva_rp"],
        gva_rp_sp=p["gva_rp_sp"],
        pre_weighting=False,
    )
    assert list(out.columns) == [2020]

    out_no_l1 = ut_gva.compute_ut_gva_l2(
        year=2020,
        l1_weights=None,
        gva_rp=p["gva_rp"],
        gva_rp_sp=p["gva_rp_sp"],
        pre_weighting=False,
    )
    assert list(out_no_l1.columns) == [2020]

    zero_gva = ut_gva.compute_ut_gva_l2(
        year=2020,
        l1_weights=w_rp,
        gva_rp=pd.Series([0.0, 5.0], index=pd.Index(["FR", "US"], name="r_p")),
        gva_rp_sp=pd.Series(
            [0.0, 0.0, 2.0, 3.0],
            index=pd.MultiIndex.from_tuples(
                [("FR", "A"), ("FR", "B"), ("US", "A"), ("US", "B")],
                names=["r_p", "s_p"],
            ),
        ),
        pre_weighting=False,
    )
    assert float(zero_gva.loc[("FR", "A"), 2020]) == 0.0
    assert float(zero_gva.loc[("FR", "B"), 2020]) == 0.0


def test_ut_gvaa_weighted_omega_and_compute_paths() -> None:
    p = _base_inputs()
    w_ru = pd.Series([0.25, 0.75], index=pd.Index(["FR", "US"], name="r_u"))

    weighted = ut_gvaa._weighted_omega_by_rc(omega_reg=p["omega_reg"], x_to_rc=p["x_to_rc"])
    assert weighted.index.names == ["r_c", "r_p", "s_p"]

    pre = ut_gvaa.compute_ut_gvaa_l2(
        fu_code="L2.a.b",
        year=2020,
        l1_weights=None,
        gva_rp=p["gva_rp"],
        x_to_rc=p["x_to_rc"],
        omega_reg=p["omega_reg"],
        pre_weighting=True,
    )
    assert pre.index.names == ["r_p", "s_p", "r_u"]

    out = ut_gvaa.compute_ut_gvaa_l2(
        fu_code="L2.b.b",
        year=2020,
        l1_weights=w_ru,
        gva_rp=p["gva_rp"],
        x_to_rc=p["x_to_rc"],
        omega_reg=p["omega_reg"],
        pre_weighting=False,
    )
    assert list(out.columns) == [2020]


def test_compute_ut_td_paths_and_error() -> None:
    p = _base_inputs()

    out_a = ut_td.compute_ut_td_l2(
        fu_code="L2.a.b",
        year=2020,
        fd_rf=p["fd_rf"],
        x_to_rc=p["x_to_rc"],
    )
    assert list(out_a.columns) == [2020]

    out_b = ut_td.compute_ut_td_l2(
        fu_code="L2.b.b",
        year=2020,
        fd_rf=p["fd_rf"],
        x_to_rc=p["x_to_rc"],
    )
    assert out_b.index.nlevels >= 2

    out_c = ut_td.compute_ut_td_l2(
        fu_code="L2.c.b",
        year=2020,
        fd_rf=p["fd_rf"],
        x_to_rc=p["x_to_rc"],
    )
    assert out_c.index.nlevels >= 2
