from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa.asocc.orchestration.yearly.l2 import l2_slicing as slicing_mod
from pyaesa.asocc.orchestration.yearly.l2 import l2_types as types_mod
from pyaesa.asocc.orchestration.yearly.shared.year_inputs import build_l2_compute_inputs


def _inputs() -> types_mod._L2ComputeInputs:
    idx_rp_sp = pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"])
    idx_rf_sp = pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_f", "s_p"])
    idx_rp_sp_rc = pd.MultiIndex.from_tuples(
        [("FR", "A", "FR"), ("US", "A", "US")],
        names=["r_p", "s_p", "r_c"],
    )
    omega_columns = pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"])
    return types_mod._L2ComputeInputs(
        fd_rf=pd.Series([1.0, 2.0], index=pd.Index(["FR", "US"], name="r_f")),
        gva_rp=pd.Series([3.0, 4.0], index=pd.Index(["FR", "US"], name="r_p")),
        fd_rp_sp_rf=pd.DataFrame({"FR": [1.0, 2.0], "US": [3.0, 4.0]}, index=idx_rp_sp),
        fd_rp_sp=pd.Series([1.0, 2.0], index=idx_rp_sp),
        fd_rf_sp=pd.Series([1.0, 2.0], index=idx_rf_sp),
        gva_rp_sp=pd.Series([1.0, 2.0], index=idx_rp_sp),
        x_to_rc=pd.DataFrame({"FR": [1.0, 2.0], "US": [3.0, 4.0]}, index=idx_rp_sp),
        kappa=pd.DataFrame({"FR": [1.0, 2.0], "US": [3.0, 4.0]}, index=idx_rp_sp_rc),
        omega_reg=pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]],
            index=pd.Index(["U1", "U2"], name="r_u"),
            columns=omega_columns,
        ),
    )


def test_l2_type_family_predicates() -> None:
    assert types_mod._is_ar_l1("AR(E^{CBA_FD})") is True
    assert types_mod._is_ar_l1("EG(Pop)") is False
    assert types_mod._is_ar_l2(l2_method="AR(E^{CBA_TD})", fu_code="L2.a.b") is True
    assert types_mod._is_ar_l2(l2_method="UT(FD)", fu_code="L2.a.a") is False
    assert types_mod._is_ut_l2(l2_method="UT(FD)", fu_code="L2.a.a") is True
    assert types_mod._is_ut_l2(l2_method="AR(E^{CBA_TD})", fu_code="L2.a.b") is False


def test_validate_l2_inputs_and_slice() -> None:
    inputs = _inputs()
    valid = build_l2_compute_inputs(
        enacting_metric_l1={"fd_rf": inputs.fd_rf, "gva_rp": inputs.gva_rp},
        enacting_metric_l2={
            "fd_rp_sp_rf": inputs.fd_rp_sp_rf,
            "fd_rp_sp": inputs.fd_rp_sp,
            "fd_rf_sp": inputs.fd_rf_sp,
            "gva_rp_sp": inputs.gva_rp_sp,
        },
        utility={
            "x_to_rc": inputs.x_to_rc,
            "kappa": inputs.kappa,
            "omega_reg": inputs.omega_reg,
        },
    )
    assert isinstance(valid, types_mod._L2ComputeInputs)

    series = pd.Series([1.0, 2.0], index=pd.Index(["FR", "US"], name="r_p"))
    assert list(slicing_mod._slice_series_index(series, level="r_p", allowed={"FR"}).index) == [
        "FR"
    ]
    frame = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index(["FR", "US"], name="r_p"))
    assert list(slicing_mod._slice_frame_index(frame, level="r_p", allowed={"US"}).index) == ["US"]
    assert list(
        slicing_mod._slice_frame_columns(
            pd.DataFrame({"FR": [1.0], "US": [2.0]}), allowed={"FR"}
        ).columns
    ) == ["FR"]
    columns = pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"])
    frame_multi_col = pd.DataFrame([[1.0, 2.0]], columns=columns)
    assert list(
        slicing_mod._slice_frame_column_level(
            frame_multi_col, level="r_p", allowed={"US"}
        ).columns.get_level_values("r_p")
    ) == ["US"]
    assert slicing_mod._slice_frame_column_level(
        frame_multi_col,
        level="missing",
        allowed={"US"},
    ).equals(frame_multi_col)
    assert slicing_mod._slice_frame_column_level(
        frame_multi_col,
        level="r_p",
        allowed=None,
    ).equals(frame_multi_col)
    simple_named_cols = pd.DataFrame(
        {"FR": [1.0], "US": [2.0]}, columns=pd.Index(["FR", "US"], name="r_p")
    )
    assert list(
        slicing_mod._slice_frame_column_level(
            simple_named_cols, level="r_p", allowed={"FR"}
        ).columns
    ) == ["FR"]
    assert (
        slicing_mod._slice_frame_column_level(
            pd.DataFrame({"FR": [1.0], "US": [2.0]}),
            level="r_p",
            allowed={"FR"},
        ).shape[1]
        == 2
    )
    multi_without_level = pd.Series(
        [1.0, 2.0],
        index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
    )
    assert slicing_mod._slice_series_index(
        multi_without_level,
        level="r_f",
        allowed={"FR"},
    ).equals(multi_without_level)
    frame_without_level = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples([("FR", "A"), ("US", "A")], names=["r_p", "s_p"]),
    )
    assert slicing_mod._slice_frame_index(
        frame_without_level,
        level="r_f",
        allowed={"FR"},
    ).equals(frame_without_level)


def test_slice_l2_inputs_and_lcia_payload_for_compute() -> None:
    inputs = _inputs()
    context = SimpleNamespace(
        fu_code="L2.a.a",
        filters={"r_p": ["FR"], "s_p": ["A"], "r_c": ["FR"], "r_f": ["FR"], "r_u": ["U1"]},
    )
    sliced = slicing_mod._slice_l2_inputs_for_compute(context=context, inputs=inputs)
    assert list(sliced.fd_rf.index) == ["FR"]
    assert list(sliced.gva_rp.index) == ["FR"]
    assert list(sliced.x_to_rc.columns) == ["FR"]
    assert list(sliced.omega_reg.index) == ["U1"]

    context_keep_full = SimpleNamespace(
        fu_code="L2.a.b",
        filters={"r_p": ["FR"], "s_p": ["A"], "r_c": ["FR"], "r_f": ["FR"], "r_u": ["U1"]},
    )
    kept = slicing_mod._slice_l2_inputs_for_compute(context=context_keep_full, inputs=inputs)
    assert set(kept.fd_rf.index.tolist()) == {"FR", "US"}
    assert set(kept.omega_reg.index.tolist()) == {"U1", "U2"}

    payload = {
        "df": pd.DataFrame({"FR": [1.0], "US": [2.0]}, index=pd.Index(["x"], name="r_p")),
        "series": pd.Series([1.0, 2.0], index=pd.Index(["FR", "US"], name="r_p")),
        "other": "keep",
    }
    sliced_payload = slicing_mod._slice_lcia_payload_for_compute(context=context, payload=payload)
    assert list(sliced_payload["df"].columns) == ["FR", "US"]
    assert list(sliced_payload["series"].index) == ["FR"]
    assert sliced_payload["other"] == "keep"

    context_keep_full_axes = SimpleNamespace(
        fu_code="L2.a.b",
        filters={"r_p": None, "s_p": None, "r_c": None, "r_f": ["FR"], "r_u": ["U1"]},
    )
    payload_keep_full = {
        "series_rf": pd.Series([1.0, 2.0], index=pd.Index(["FR", "US"], name="r_f")),
        "series_ru": pd.Series([1.0, 2.0], index=pd.Index(["U1", "U2"], name="r_u")),
    }
    kept_payload = slicing_mod._slice_lcia_payload_for_compute(
        context=context_keep_full_axes,
        payload=payload_keep_full,
    )
    assert list(kept_payload["series_rf"].index) == ["FR", "US"]
    assert list(kept_payload["series_ru"].index) == ["U1", "U2"]


def test_slice_weights_impact_items_and_normalization() -> None:
    run: Any = SimpleNamespace(
        context=SimpleNamespace(
            fu_code="L2.a.a",
            filters={"r_f": ["FR"], "r_u": ["U1"], "r_p": None, "s_p": None, "r_c": None},
        )
    )
    weights = pd.Series([1.0, 2.0], index=pd.Index(["FR", "US"], name="r_f"))
    sliced = slicing_mod._slice_l1_weights_for_compute(run=run, l2_method="UT(FD)", weights=weights)
    assert sliced is not None
    assert list(sliced.index) == ["FR"]

    run_keep: Any = SimpleNamespace(
        context=SimpleNamespace(
            fu_code="L2.a.b",
            filters={"r_f": ["FR"], "r_u": ["U1"], "r_p": None, "s_p": None, "r_c": None},
        )
    )
    kept = slicing_mod._slice_l1_weights_for_compute(
        run=run_keep,
        l2_method="UT(FDa)",
        weights=weights,
    )
    assert kept is not None
    assert set(kept.index.tolist()) == {"FR", "US"}
    assert (
        slicing_mod._slice_l1_weights_for_compute(run=run, l2_method="UT(FD)", weights=None) is None
    )

    run_keep_u: Any = SimpleNamespace(
        context=SimpleNamespace(
            fu_code="L2.a.b",
            filters={"r_f": ["FR"], "r_u": ["U1"], "r_p": None, "s_p": None, "r_c": None},
        )
    )
    weights_u = pd.Series([1.0, 2.0], index=pd.Index(["U1", "U2"], name="r_u"))
    kept_u = slicing_mod._slice_l1_weights_for_compute(
        run=run_keep_u,
        l2_method="UT(GVAa)",
        weights=weights_u,
    )
    assert kept_u is not None
    assert set(kept_u.index.tolist()) == {"U1", "U2"}
    sliced_frame = slicing_mod._slice_l1_weight_frame_for_compute(
        run=run,
        l2_method="UT(FD)",
        weights=weights.to_frame(name="value"),
    )
    assert sliced_frame is not None
    assert sliced_frame.index.tolist() == ["FR"]
    kept_frame = slicing_mod._slice_l1_weight_frame_for_compute(
        run=run_keep,
        l2_method="UT(FDa)",
        weights=weights.to_frame(name="value"),
    )
    assert kept_frame is not None
    assert kept_frame.index.tolist() == ["FR", "US"]
    assert (
        slicing_mod._slice_l1_weight_frame_for_compute(
            run=run,
            l2_method="UT(FD)",
            weights=None,
        )
        is None
    )

    impact_df = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.MultiIndex.from_tuples(
            [("climate", "FR"), ("water", "FR")], names=["impact", "r_p"]
        ),
    )
    impact_items = slicing_mod._impact_weight_items(impact_df)
    assert impact_items[0][0] == "climate"
    assert impact_items[1][0] == "water"
    impact_matrix_df = pd.DataFrame(
        {"value": [0.2, 0.8, 0.4, 0.6]},
        index=pd.MultiIndex.from_tuples(
            [
                ("climate", "FR"),
                ("climate", "US"),
                ("water", "FR"),
                ("water", "US"),
            ],
            names=["impact", "r_p"],
        ),
    )
    impact_matrix = slicing_mod._impact_weight_matrix(impact_matrix_df)
    assert impact_matrix is not None
    matrix_impacts, matrix_index, matrix_values = impact_matrix
    assert matrix_impacts == ("climate", "water")
    assert matrix_index.tolist() == ["FR", "US"]
    assert matrix_values.tolist() == [[0.2, 0.8], [0.4, 0.6]]
    impact_matrix_multi_df = pd.DataFrame(
        {"value": [0.2, 0.8, 0.4, 0.6]},
        index=pd.MultiIndex.from_tuples(
            [
                ("climate", "FR", "A"),
                ("climate", "US", "B"),
                ("water", "FR", "A"),
                ("water", "US", "B"),
            ],
            names=["impact", "r_p", "s_p"],
        ),
    )
    impact_matrix_multi = slicing_mod._impact_weight_matrix(impact_matrix_multi_df)
    assert impact_matrix_multi is not None
    multi_impacts, multi_index, multi_values = impact_matrix_multi
    assert multi_impacts == ("climate", "water")
    assert multi_index.names == ["r_p", "s_p"]
    assert multi_index.tolist() == [("FR", "A"), ("US", "B")]
    assert multi_values.tolist() == [[0.2, 0.8], [0.4, 0.6]]
    assert slicing_mod._impact_weight_matrix(None) is None
    contiguous_repeated_impact_df = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples(
            [("climate", "FR"), ("climate", "US"), ("water", "FR")],
            names=["impact", "r_p"],
        ),
    )
    contiguous_repeated_items = slicing_mod._impact_weight_items(contiguous_repeated_impact_df)
    assert contiguous_repeated_items[0][1] is not None
    assert contiguous_repeated_items[0][1].index.tolist() == ["FR", "US"]
    plain_df = pd.DataFrame({"value": [1.0]}, index=pd.Index(["FR"], name="r_p"))
    assert slicing_mod._impact_weight_items(plain_df)[0][0] is None
    assert slicing_mod._impact_weight_matrix(plain_df) is None
    assert slicing_mod._impact_weight_items(weights.to_frame(name="value"))[0][0] is None
    assert slicing_mod._impact_weight_items(None) == [(None, None)]
    assert slicing_mod._impact_weight_items(cast(Any, weights))[0][1] is weights

    dup = pd.Series([1.0, 2.0], index=pd.Index(["FR", "FR"], name="r_p"))
    with pytest.raises(ValueError):
        slicing_mod._normalize_l1_weights(dup)
    normalized = slicing_mod._normalize_l1_weights(
        pd.Series([1.0, 2.0], index=pd.Index(["FR", "US"], name="r_p"))
    )
    assert list(normalized.index) == ["FR", "US"]
