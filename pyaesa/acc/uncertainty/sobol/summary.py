"""aCC Sobol source summary shaping."""

import pandas as pd

from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.uncertainty_assessment.sobol.accumulator import SobolIndexEstimate
from pyaesa.shared.uncertainty_assessment.sobol.summary_levels import (
    SobolInvariantAxisExpansion,
    SobolSummaryLevel,
    sobol_source_summary_by_levels,
)


def acc_sobol_source_summary(
    *,
    identity: pd.DataFrame,
    dimension_names: tuple[str, ...],
    estimates: SobolIndexEstimate,
    confidence_level: float,
    requested_ssp_scenarios: tuple[str, ...],
) -> pd.DataFrame:
    """Return aCC selector preserving Sobol source summary rows."""
    selector_columns = tuple(column for column in identity.columns if column != "public_row_id")
    levels = (
        SobolSummaryLevel(summary_level="selector", group_columns=selector_columns),
        SobolSummaryLevel(
            summary_level="lcia_method",
            group_columns=tuple(column for column in selector_columns if column != "impact"),
        ),
    )
    return sobol_source_summary_by_levels(
        identity=identity,
        dimension_names=dimension_names,
        estimates=estimates,
        confidence_level=confidence_level,
        levels=levels,
        selector_columns=selector_columns,
        invariant_axis=SobolInvariantAxisExpansion(
            axis_column=ASOCC_SSP_SCENARIO_COLUMN,
            axis_values=requested_ssp_scenarios,
            contains_column="contains_ssp_invariant_outputs",
            count_column="ssp_invariant_output_count",
        ),
    )
