"""External LCA scenario metadata helpers."""

import pandas as pd

from pyaesa.shared.runtime.scenario.columns import (
    EXT_LCA_SSP_SCENARIO_COLUMN,
    LCA_SSP_START_YEAR_COLUMN,
)
from pyaesa.shared.tabular.wide_tables import first_non_null_scenario_year


def with_lca_ssp_start_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the first year where external LCA rows become SSP dependent."""
    start_year = first_non_null_scenario_year(
        frame,
        scenario_column=EXT_LCA_SSP_SCENARIO_COLUMN,
        year_column="year",
    )
    if start_year is None:
        return frame
    out = frame.copy()
    out[LCA_SSP_START_YEAR_COLUMN] = int(start_year)
    return out
