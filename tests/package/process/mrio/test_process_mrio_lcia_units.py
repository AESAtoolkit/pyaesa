import pandas as pd
import pytest

from pyaesa.process.mrios.utils.pipeline.lcia_tracking import (
    extract_lcia_units_from_char_matrix,
)


def test_extract_lcia_units_from_char_matrix() -> None:
    char_df = pd.DataFrame(
        {
            "extension": ["satellite_accounts", "satellite_accounts"],
            "impact_parent": ["Climate change", "Climate change"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
        }
    )
    units = extract_lcia_units_from_char_matrix(
        lcia_method="gwp100_lcia",
        char_matrix=char_df,
    )
    assert units == {"Climate change": "kg CO2-eq"}


def test_extract_lcia_units_from_char_matrix_rejects_conflicts() -> None:
    char_df = pd.DataFrame(
        {
            "extension": ["satellite_accounts", "satellite_accounts"],
            "impact_parent": ["Climate change", "Climate change"],
            "impact_unit": ["kg CO2-eq", "t CO2-eq"],
        }
    )
    with pytest.raises(ValueError):
        extract_lcia_units_from_char_matrix(
            lcia_method="gwp100_lcia",
            char_matrix=char_df,
        )
