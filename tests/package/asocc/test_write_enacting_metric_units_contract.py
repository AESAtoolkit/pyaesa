from types import SimpleNamespace

import pandas as pd

from pyaesa.asocc.io.metadata import EnactingMetricKey
from pyaesa.asocc.orchestration.write.writers.enacting_metric_units import (
    resolve_enacting_metric_unit,
)


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        wb_df=pd.DataFrame(columns=["2019"]),
        wb_df_raw=pd.DataFrame(columns=["variable", "unit"]),
        ssp_df_raw=pd.DataFrame(columns=["variable", "unit", "scenario"]),
    )


def _dummy_frame() -> pd.DataFrame:
    return pd.DataFrame({"value": [1.0]})


def test_resolve_enacting_metric_unit_uses_explicit_mrio_mapping() -> None:
    key = EnactingMetricKey(metric="fd_rf")
    unit = resolve_enacting_metric_unit(
        context=_context(),
        key=key,
        year_map={2019: pd.Series([1.0], index=pd.Index(["R1"], name="r_f"))},
        mrio_default_monetary_unit="M EUR",
        mrio_units={"fd_rf": "M EUR"},
        lcia_units={},
        df=_dummy_frame(),
    )
    assert unit == "M EUR"


def test_resolve_enacting_metric_unit_uses_default_for_mrio_family_metric() -> None:
    key = EnactingMetricKey(metric="fda_rp_sp_rc")
    unit = resolve_enacting_metric_unit(
        context=_context(),
        key=key,
        year_map={2019: pd.Series([1.0], index=pd.Index(["R1"], name="r_f"))},
        mrio_default_monetary_unit="M EUR",
        mrio_units={},
        lcia_units={},
        df=_dummy_frame(),
    )
    assert unit == "M EUR"
