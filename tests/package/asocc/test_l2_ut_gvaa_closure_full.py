from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from pyaesa.asocc.orchestration.yearly.l2 import (
    l2_ut_gvaa_closure as closure_mod,
)
from pyaesa.asocc.orchestration.yearly.l2 import l2_types as types_mod


def _run() -> types_mod._L2RunContext:
    inputs = types_mod._L2ComputeInputs(
        fd_rf=pd.Series(dtype=float),
        gva_rp=pd.Series([2.0], index=pd.Index(["FR"], name="r_p")),
        fd_rp_sp_rf=pd.DataFrame(),
        fd_rp_sp=pd.Series(dtype=float),
        fd_rf_sp=pd.Series(dtype=float),
        gva_rp_sp=pd.Series(
            [4.0],
            index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
        ),
        x_to_rc=pd.DataFrame(),
        kappa=pd.DataFrame(),
        omega_reg=pd.DataFrame(),
    )
    context: Any = SimpleNamespace(
        fu_code="L2.a.b",
        source="oecd_v2025",
        proj_base=Path("projection_base"),
    )
    state: Any = SimpleNamespace(ut_gvaa_identity_closure_rows=[])
    return types_mod._L2RunContext(
        context=context,
        state=state,
        year=2030,
        ssp_scenario="SSP2",
        lcia_by_method=None,
        l1_results_year={},
        inputs=inputs,
    )


def test_ut_gvaa_closure() -> None:
    frame = pd.DataFrame({"2030": [1.0]}, index=pd.Index(["FR"], name="r_p"))
    series = closure_mod._single_year_series(frame, label="x")
    assert float(series.iloc[0]) == 1.0

    bad = pd.DataFrame({"2030": [1.0], "2031": [2.0]}, index=pd.Index(["FR"], name="r_p"))
    with pytest.raises(ValueError):
        closure_mod._single_year_series(bad, label="x")

    floor = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
    )
    result = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([("A", "FR")], names=["s_p", "r_p"]),
    )
    aligned = closure_mod._align_floor_index_to_result(floor=floor, result=result)
    assert list(aligned.index.names) == ["s_p", "r_p"]

    floor_single = pd.Series([1.0], index=pd.Index(["FR"], name="r_p"))
    result_single = pd.Series([1.0], index=pd.Index(["FR"], name="r_p"))
    assert closure_mod._align_floor_index_to_result(
        floor=floor_single, result=result_single
    ).equals(floor_single)

    floor_other = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
    )
    result_other = pd.Series(
        [1.0],
        index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_x", "s_x"]),
    )
    assert closure_mod._align_floor_index_to_result(floor=floor_other, result=result_other).equals(
        floor_other
    )


def test_apply_ut_gvaa_identity_closure_paths() -> None:
    run = _run()
    result = pd.DataFrame(
        {"2030": [1.0]},
        index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
    )
    slice_spec = types_mod._L2SliceSpec(
        l2_method="UT(GVAa)",
        l1_name="EG(Pop)",
        l1_name_resolved="EG(Pop)",
        lcia_key="IPCC",
        lcia_data={"x": 1},
        ref_year=2000,
        treat_as_one_step=False,
    )
    weights = pd.Series([1.0], index=pd.Index(["FR"], name="r_p"))

    # Early return path when method/fu/weights do not match closure conditions.
    early = closure_mod.apply_ut_gvaa_identity_closure(
        run=run,
        slice_spec=types_mod._L2SliceSpec(
            l2_method="UT(FD)",
            l1_name="EG(Pop)",
            l1_name_resolved="EG(Pop)",
            lcia_key=None,
            lcia_data=None,
            ref_year=None,
            treat_as_one_step=False,
        ),
        weights=weights,
        impact=None,
        result=result,
    )
    assert early.equals(result)

    # No correction path.
    no_fix = closure_mod.apply_ut_gvaa_identity_closure(
        run=run,
        slice_spec=slice_spec,
        weights=weights,
        impact="climate",
        result=pd.DataFrame(
            {"2030": [3.0]},
            index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
        ),
    )
    assert float(no_fix.iloc[0, 0]) == 3.0
    assert run.state.ut_gvaa_identity_closure_rows == []

    # Correction path + audit rows.
    fixed = closure_mod.apply_ut_gvaa_identity_closure(
        run=run,
        slice_spec=slice_spec,
        weights=weights,
        impact="climate",
        result=result,
        l2_reuse_year=2019,
    )
    assert float(fixed.iloc[0, 0]) == 2.0
    assert len(run.state.ut_gvaa_identity_closure_rows) == 1
    row = run.state.ut_gvaa_identity_closure_rows[0]
    assert row["l2_method"] == "UT(GVAa)"
    assert row["comparator_method"] == "UT(GVA)"
    assert int(row["l2_reuse_year"]) == 2019


def test_apply_ut_gvaa_identity_closure_records_single_index_rows() -> None:
    run = _run()._replace(
        inputs=types_mod._L2ComputeInputs(
            fd_rf=pd.Series(dtype=float),
            gva_rp=pd.Series([2.0], index=pd.Index(["FR"], name="r_p")),
            fd_rp_sp_rf=pd.DataFrame(),
            fd_rp_sp=pd.Series(dtype=float),
            fd_rf_sp=pd.Series(dtype=float),
            gva_rp_sp=pd.Series([4.0], index=pd.Index(["FR"], name="r_p")),
            x_to_rc=pd.DataFrame(),
            kappa=pd.DataFrame(),
            omega_reg=pd.DataFrame(),
        )
    )
    result = pd.DataFrame({"2030": [1.0]}, index=pd.Index(["FR"], name="r_p"))
    slice_spec = types_mod._L2SliceSpec(
        l2_method="UT(GVAa)",
        l1_name="EG(Pop)",
        l1_name_resolved="EG(Pop)",
        lcia_key=None,
        lcia_data={"x": 1},
        ref_year=None,
        treat_as_one_step=False,
    )
    weights = pd.Series([1.0], index=pd.Index(["FR"], name="r_p"))
    out = closure_mod.apply_ut_gvaa_identity_closure(
        run=run,
        slice_spec=slice_spec,
        weights=weights,
        impact=None,
        result=result,
    )
    assert float(out.iloc[0, 0]) == 2.0
    assert run.state.ut_gvaa_identity_closure_rows[0]["r_p"] == "FR"


def test_apply_ut_gvaa_identity_closure_preserves_index_like_column_labels() -> None:
    run = _run()
    result = pd.DataFrame(
        [[1.0]],
        index=pd.MultiIndex.from_tuples([("FR", "A")], names=["r_p", "s_p"]),
        columns=pd.Index([pd.Index(["2030"])], dtype=object),
    )
    slice_spec = types_mod._L2SliceSpec(
        l2_method="UT(GVAa)",
        l1_name="EG(Pop)",
        l1_name_resolved="EG(Pop)",
        lcia_key="IPCC",
        lcia_data={"x": 1},
        ref_year=2000,
        treat_as_one_step=False,
    )
    weights = pd.Series([1.0], index=pd.Index(["FR"], name="r_p"))

    out = closure_mod.apply_ut_gvaa_identity_closure(
        run=run,
        slice_spec=slice_spec,
        weights=weights,
        impact="climate",
        result=result,
    )
    assert float(out.iloc[0, 0]) == 2.0
    assert out.columns.tolist() == ["Index(['2030'], dtype='str')"]
