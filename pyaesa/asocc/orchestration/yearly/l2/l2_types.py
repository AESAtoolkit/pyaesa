"""Type definitions and family predicates for L2 orchestration."""

from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from ....io.metadata import RunContext, RunState
from ....methods.registry.registry import REGISTRY


@dataclass(frozen=True)
class _L2ComputeInputs:
    fd_rf: pd.Series
    gva_rp: pd.Series
    fd_rp_sp_rf: pd.DataFrame
    fd_rp_sp: pd.Series
    fd_rf_sp: pd.Series
    gva_rp_sp: pd.Series
    x_to_rc: pd.DataFrame
    kappa: pd.DataFrame
    omega_reg: pd.DataFrame


class _L2RunContext(NamedTuple):
    context: RunContext
    state: RunState
    year: int
    ssp_scenario: str | None
    lcia_by_method: dict[str, dict] | None
    l1_results_year: dict[str, pd.DataFrame]
    inputs: _L2ComputeInputs


@dataclass(frozen=True)
class _L2SliceSpec:
    l2_method: str
    l1_name: str | None
    l1_name_resolved: str | None
    lcia_key: str | None
    lcia_data: dict | None
    ref_year: int | None
    treat_as_one_step: bool


@dataclass(frozen=True)
class _L2WeightSpec:
    slice_spec: _L2SliceSpec
    impact: str | None
    weights: pd.Series | None


@dataclass(frozen=True)
class _CombinedSliceRequest:
    l2_method: str
    l1_name: str
    lcia_key: str | None
    lcia_data: dict | None
    ref_year: int | None


def _is_ar_l1(l1_method: str) -> bool:
    """Return whether an L1 method belongs to AR families."""
    return REGISTRY.method_family(l1_method, level="L1") in {"AR_E", "AR_ECAP"}


def _is_ar_l2(*, l2_method: str, fu_code: str) -> bool:
    """Return whether an L2 method belongs to AR families."""
    return REGISTRY.method_family(l2_method, level="L2", fu_code=fu_code) in {
        "AR_E",
        "AR_ECAP",
    }


def _is_ut_l2(*, l2_method: str, fu_code: str) -> bool:
    """Return whether an L2 method belongs to UT families."""
    return REGISTRY.method_family(l2_method, level="L2", fu_code=fu_code) in {
        "UT_FD",
        "UT_FDA",
        "UT_GVAA",
        "UT_TD",
        "UT_GVA",
    }
