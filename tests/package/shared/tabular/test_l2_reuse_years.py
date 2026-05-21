from pathlib import Path

import pandas as pd
import pytest

from pyaesa.shared.tabular.l2_reuse_years import (
    canonicalize_l2_reuse_year_column,
    frame_l2_reuse_years,
)


def test_l2_reuse_year_contracts_cover_nullable_values_and_invalid_input() -> None:
    normalized = canonicalize_l2_reuse_year_column(
        pd.DataFrame({"l2_reuse_year": [2030, None], "value": [1.0, 2.0]}),
    )
    assert normalized["l2_reuse_year"].tolist() == [2030, pd.NA]
    assert frame_l2_reuse_years(normalized) == (2030,)
    assert frame_l2_reuse_years(pd.DataFrame({"l2_reuse_year": [None, pd.NA]})) == tuple()

    passthrough = canonicalize_l2_reuse_year_column(pd.DataFrame({"value": [1.0]}))
    assert "l2_reuse_year" not in passthrough.columns

    with pytest.raises(ValueError):
        canonicalize_l2_reuse_year_column(
            pd.DataFrame({"l2_reuse_year": ["not-a-year"]}),
            path=Path("bad.csv"),
        )
