"""aCC uncertainty public row alignment by carrying capacity branch."""

from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.acc.shared.runtime.dynamic_units import dynamic_acc_unit_factors
from pyaesa.acc.uncertainty.runtime.models import ACCBranchPlan
from pyaesa.shared.lcia.contracts import dynamic_cc_match
from pyaesa.shared.lcia.paths import static_cc_csv_path
from pyaesa.shared.lcia.static_cc import read_static_cc, require_static_cc_bounds_available
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.runtime.scenario.scoped_rows import scenario_target_rows


def build_acc_branch_plans(
    *,
    asocc_identity: pd.DataFrame,
    cc_identity: pd.DataFrame | None,
    branches: list[dict[str, Any]],
) -> tuple[ACCBranchPlan, ...]:
    """Build vectorized ACC branch plans from component public identities."""
    return tuple(
        _branch_plan(
            asocc_identity=asocc_identity,
            cc_identity=cc_identity,
            branch=branch,
        )
        for branch in branches
    )


def _branch_plan(
    *,
    asocc_identity: pd.DataFrame,
    cc_identity: pd.DataFrame | None,
    branch: dict[str, Any],
) -> ACCBranchPlan:
    if branch["cc_type"] == "static":
        return _static_branch_plan(asocc_identity=asocc_identity, branch=branch)
    return _dynamic_branch_plan(
        asocc_identity=asocc_identity,
        cc_identity=cast(pd.DataFrame, cc_identity),
        branch=branch,
    )


def _static_branch_plan(
    *,
    asocc_identity: pd.DataFrame,
    branch: dict[str, Any],
) -> ACCBranchPlan:
    cc_source = str(branch["cc_source"])
    cc_rows = _static_cc_rows(cc_source=cc_source, bounds=list(branch["static_cc_bounds"]))
    asocc = _asocc_rows_for_cc_source(asocc_identity=asocc_identity, cc_source=cc_source)
    asocc["_asocc_position"] = np.arange(len(asocc), dtype=np.int64)
    asocc = asocc.drop(columns=["public_row_id"], errors="ignore")
    generic, constrained = _split_by_optional_impact(frame=asocc)
    pieces: list[pd.DataFrame] = []
    if not generic.empty:
        pieces.append(generic.merge(cc_rows, how="cross"))
    if not constrained.empty:
        pieces.append(constrained.merge(cc_rows, on="impact", how="inner", suffixes=("_asocc", "")))
    rows = pd.concat(pieces, ignore_index=True)
    identity = _acc_identity_from_rows(rows=rows, cc_source=cc_source, cc_type="static")
    return ACCBranchPlan(
        identity=identity,
        asocc_positions=rows["_asocc_position"].to_numpy(dtype=np.int64),
        cc_positions=None,
        static_cc_values=rows["_cc_value"].to_numpy(dtype=np.float64),
        dynamic_cc_factors=None,
        cc_type="static",
        cc_source=cc_source,
    )


def _dynamic_branch_plan(
    *,
    asocc_identity: pd.DataFrame,
    cc_identity: pd.DataFrame,
    branch: dict[str, Any],
) -> ACCBranchPlan:
    cc_source = str(branch["cc_source"])
    expected_impact = _dynamic_expected_impact(cc_source=cc_source)
    scoped = _asocc_rows_for_cc_source(asocc_identity=asocc_identity, cc_source=cc_source)
    asocc = _dynamic_asocc_scope(asocc_identity=scoped, expected_impact=expected_impact)
    asocc["_asocc_position"] = np.arange(len(asocc), dtype=np.int64)
    asocc = asocc.drop(columns=["public_row_id"], errors="ignore")
    cc = cc_identity.copy()
    cc["_cc_position"] = (
        pd.Series(pd.to_numeric(cc["public_row_id"], errors="raise"), copy=False).to_numpy(
            dtype=np.int64
        )
        if "public_row_id" in cc.columns
        else np.arange(len(cc), dtype=np.int64)
    )
    cc = cc.drop(columns=["public_row_id"], errors="ignore")
    cc = cc.rename(columns={"ssp_scenario": AR6_CC_SSP_SCENARIO_COLUMN})
    pairs = _dynamic_pairs(asocc=asocc, cc=cc)
    if pairs.empty:
        cc_ssps = (
            sorted(cc[AR6_CC_SSP_SCENARIO_COLUMN].astype(str).str.strip().unique().tolist())
            if AR6_CC_SSP_SCENARIO_COLUMN in cc.columns
            else []
        )
        raise ValueError(
            "Dynamic aCC uncertainty found no overlapping aSoCC and AR6 carrying "
            "capacity year and SSP rows after LCIA and impact scoping. "
            f"cc_source='{cc_source}', expected_impact='{expected_impact}', "
            f"dynamic_ar6_ssp_scope={cc_ssps}. Check that upstream aSoCC rows and "
            "dynamic AR6 CC rows use overlapping years and SSP scenarios."
        )
    target_unit, factors = dynamic_acc_unit_factors(
        source_units=pd.Series(pairs["impact_unit"], copy=False),
        cc_source=cc_source,
        impact=expected_impact,
    )
    pairs["impact"] = expected_impact
    pairs["impact_unit"] = target_unit
    identity = _acc_identity_from_rows(rows=pairs, cc_source=cc_source, cc_type="dynamic_ar6")
    return ACCBranchPlan(
        identity=identity,
        asocc_positions=pairs["_asocc_position"].to_numpy(dtype=np.int64),
        cc_positions=pairs["_cc_position"].to_numpy(dtype=np.int64),
        static_cc_values=None,
        dynamic_cc_factors=factors,
        cc_type="dynamic_ar6",
        cc_source=cc_source,
    )


def _static_cc_rows(*, cc_source: str, bounds: list[str]) -> pd.DataFrame:
    cc = read_static_cc(static_cc_csv_path(lcia_method=cc_source))
    require_static_cc_bounds_available(
        cc_df=cc,
        requested_bounds=bounds,
        context="aCC uncertainty static carrying capacity request",
    )
    records: list[pd.DataFrame] = []
    for bound in bounds:
        values = cc[bound].to_numpy(dtype=np.float64, copy=False)
        part = cc.loc[:, ["impact", "impact_unit"]].copy()
        part["cc_bound"] = str(bound)
        part["_cc_value"] = values
        records.append(part)
    return pd.concat(records, ignore_index=True)


def _split_by_optional_impact(*, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "impact" not in frame.columns:
        return frame.copy(), frame.iloc[0:0].copy()
    impact = frame["impact"].astype("string").fillna("").str.strip()
    generic = frame.loc[impact.eq("")].drop(columns=["impact"], errors="ignore").copy()
    constrained = frame.loc[~impact.eq("")].copy()
    return generic, constrained


def _dynamic_expected_impact(*, cc_source: str) -> str:
    match = cast(dict[str, str], dynamic_cc_match(lcia_method=cc_source))
    return str(match["impact"]).strip()


def _asocc_rows_for_cc_source(
    *,
    asocc_identity: pd.DataFrame,
    cc_source: str,
) -> pd.DataFrame:
    """Return aSoCC rows whose LCIA method matches the carrying capacity source."""
    if "lcia_method" not in asocc_identity.columns:
        return asocc_identity.copy()
    expected = str(cc_source).strip()
    lcia_method = asocc_identity["lcia_method"].astype("string").fillna("").str.strip()
    return asocc_identity.loc[lcia_method.eq("") | lcia_method.eq(expected)].copy()


def _dynamic_asocc_scope(*, asocc_identity: pd.DataFrame, expected_impact: str) -> pd.DataFrame:
    if "impact" not in asocc_identity.columns:
        return asocc_identity.copy()
    impact = asocc_identity["impact"].astype("string").fillna("").str.strip()
    return asocc_identity.loc[impact.eq("") | impact.eq(expected_impact)].copy()


def _dynamic_pairs(*, asocc: pd.DataFrame, cc: pd.DataFrame) -> pd.DataFrame:
    asocc_year = np.asarray(pd.to_numeric(asocc["year"], errors="raise"), dtype=np.int64)
    cc_year = np.asarray(pd.to_numeric(cc["year"], errors="raise"), dtype=np.int64)
    asocc = asocc.assign(year=asocc_year)
    cc = cc.assign(year=cc_year)
    pieces: list[pd.DataFrame] = []
    identity_columns = _dynamic_asocc_identity_columns(asocc)
    for token, cc_rows in cc.groupby(AR6_CC_SSP_SCENARIO_COLUMN, dropna=False, sort=False):
        asocc_rows = scenario_target_rows(
            asocc,
            target={ASOCC_SSP_SCENARIO_COLUMN: token},
            scenario_columns=(ASOCC_SSP_SCENARIO_COLUMN,),
            identity_columns=identity_columns,
        )
        if not asocc_rows.empty:
            pieces.append(asocc_rows.merge(cc_rows, on="year", how="inner"))
    return pd.concat(pieces, ignore_index=True) if pieces else asocc.iloc[0:0].copy()


def _dynamic_asocc_identity_columns(asocc: pd.DataFrame) -> list[str]:
    excluded = {
        "_asocc_position",
        ASOCC_SSP_SCENARIO_COLUMN,
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
    }
    return [column for column in asocc.columns if column not in excluded]


def _acc_identity_from_rows(*, rows: pd.DataFrame, cc_source: str, cc_type: str) -> pd.DataFrame:
    drop = {"_asocc_position", "_cc_position", "_cc_value"}
    out = rows.drop(columns=[column for column in drop if column in rows.columns]).copy()
    out = _materialize_lcia_method(frame=out, cc_source=cc_source)
    out.insert(0, "cc_type", cc_type)
    out = out.loc[:, _ordered_identity_columns(out)].reset_index(drop=True)
    out.insert(0, "public_row_id", np.arange(len(out), dtype=np.int64))
    return out


def _materialize_lcia_method(*, frame: pd.DataFrame, cc_source: str) -> pd.DataFrame:
    """Attach the public carrying capacity LCIA source to ACC row identity."""
    out = frame.copy()
    if "lcia_method" not in out.columns:
        out.insert(0, "lcia_method", str(cc_source).strip())
    else:
        out["lcia_method"] = str(cc_source).strip()
    return out


def _ordered_identity_columns(frame: pd.DataFrame) -> list[str]:
    priority = [
        "cc_type",
        "lcia_method",
        "cc_bound",
        "cc_category",
        AR6_CC_SSP_SCENARIO_COLUMN,
        "cc_flow",
        "cc_variable",
        "year",
        "impact",
        "impact_unit",
        "l1_method",
        "l1_l2_method",
        "l2_method",
        ASOCC_SSP_SCENARIO_COLUMN,
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        "reference_year",
        "r_p",
        "s_p",
        "r_c",
        "r_f",
    ]
    ordered = [column for column in priority if column in frame.columns]
    ordered.extend(column for column in frame.columns if column not in ordered)
    return ordered


def combined_acc_identity(*, branch_plans: tuple[ACCBranchPlan, ...]) -> pd.DataFrame:
    """Return the combined ACC public row identity for all branch plans."""
    frames = [branch.identity.drop(columns=["public_row_id"]) for branch in branch_plans]
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "public_row_id", np.arange(len(out), dtype=np.int64))
    return out
