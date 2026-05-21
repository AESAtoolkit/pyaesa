"""Type definitions for L1 orchestration."""

from dataclasses import dataclass

import pandas as pd

from ....io.metadata import RunContext, RunState


@dataclass(frozen=True)
class _L1RunContext:
    context: RunContext
    state: RunState
    year: int
    ssp_scenario: str | None
    pop_series: pd.Series
    pop_series_original: pd.Series | None
    pr_pop: pd.Series | None
    pr_gdp: pd.Series | None
    pr_to_mrio: pd.Series | None
    l1_results_year: dict[str, pd.DataFrame]


@dataclass(frozen=True)
class _L1StorePayload:
    resolved_name: str
    lcia_method: str | None
    frame: pd.DataFrame
    year_key: str
    value_frame: pd.DataFrame | None = None


@dataclass(frozen=True)
class _LciaMethodInputs:
    lcia_method: str
    lcia_kind: str
    lcia_reg: pd.DataFrame
    lcia_reg_by_year: dict | None
    rps_df: pd.DataFrame | None
    impact_parent_map: pd.Series | None
    resolved_name: str
    impact_year: int | None = None
