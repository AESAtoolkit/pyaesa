from types import SimpleNamespace

import pandas as pd
import pytest

from pyaesa.asocc.io.metadata import EnactingMetricKey
from pyaesa.asocc.orchestration.write.writers import enacting_metric_units as mod


def _context(*, wb_year_cols=("2019",), wb_raw=None, ssp_raw=None):
    return SimpleNamespace(
        wb_df=pd.DataFrame(columns=list(wb_year_cols)),
        wb_df_raw=(
            wb_raw
            if wb_raw is not None
            else pd.DataFrame(
                {
                    "variable": [mod.POP_WB_INDICATOR, mod.GDP_WB_INDICATOR],
                    "unit": ["people", "USD"],
                }
            )
        ),
        ssp_df_raw=(
            ssp_raw
            if ssp_raw is not None
            else pd.DataFrame(
                {
                    "variable": [mod.POP_SSP_INDICATOR, mod.GDP_SSP_INDICATOR],
                    "unit": ["people", "USD"],
                    "ssp_scenario": ["SSP2", "SSP2"],
                }
            )
        ),
    )


def test_units_for_variable_and_single_unit_validation() -> None:
    with pytest.raises(ValueError):
        mod._units_for_variable(
            df=pd.DataFrame({"variable": ["x"]}),
            variable="x",
            ssp_scenario=None,
        )

    df = pd.DataFrame(
        {
            "variable": ["x", "x", "x"],
            "unit": ["u1", "u2", None],
            "ssp_scenario": ["S1", "S2", "S1"],
        }
    )
    units_s1 = mod._units_for_variable(df=df, variable="x", ssp_scenario="S1")
    assert units_s1 == {"u1"}
    units_no_scope = mod._units_for_variable(df=df, variable="x", ssp_scenario="S3")
    assert units_no_scope == {"u1", "u2"}

    with pytest.raises(ValueError):
        mod._single_unit(set(), metric="m", key=EnactingMetricKey(metric="m"))

    with pytest.raises(ValueError):
        mod._single_unit({"u1", "u2"}, metric="m", key=EnactingMetricKey(metric="m"))


def test_resolve_enacting_metric_unit_lcia_paths() -> None:
    context = _context()
    unit_map = pd.Series(["kgCO2e"], index=pd.Index(["Climate"], name="impact"))
    unit_map.attrs["source_csv"] = "units.csv"

    key_method = EnactingMetricKey(metric="e_cba_fd_reg", lcia_method="gwp100", ssp_scenario=None)
    out_scalar = mod.resolve_enacting_metric_unit(
        context=context,
        key=key_method,
        year_map={2019: pd.Series([1.0], index=pd.Index(["FR"], name="r_f"))},
        mrio_default_monetary_unit="M EUR",
        mrio_units={},
        lcia_units={"gwp100": unit_map},
        df=pd.DataFrame({"value": [1.0]}),
    )
    assert out_scalar == "kgCO2e"

    key_cap = EnactingMetricKey(metric="e_cba_fd_reg_cap", lcia_method="gwp100", ssp_scenario=None)
    out_cap = mod.resolve_enacting_metric_unit(
        context=context,
        key=key_cap,
        year_map={2019: pd.Series([1.0], index=pd.Index(["FR"], name="r_f"))},
        mrio_default_monetary_unit="M EUR",
        mrio_units={},
        lcia_units={"gwp100": unit_map},
        df=pd.DataFrame({"value": [1.0]}),
    )
    assert out_cap == "kgCO2e/cap"

    with pytest.raises(ValueError):
        mod.resolve_enacting_metric_unit(
            context=context,
            key=key_method,
            year_map={2019: pd.Series([1.0], index=pd.Index(["FR"], name="r_f"))},
            mrio_default_monetary_unit="M EUR",
            mrio_units={},
            lcia_units={"gwp100": pd.Series(["u1", "u2"], index=["i1", "i2"])},
            df=pd.DataFrame({"value": [1.0]}),
        )

    with pytest.raises(ValueError):
        mod.resolve_enacting_metric_unit(
            context=context,
            key=key_method,
            year_map={2019: pd.Series([1.0], index=pd.Index(["FR"], name="r_f"))},
            mrio_default_monetary_unit="M EUR",
            mrio_units={},
            lcia_units={"gwp100": unit_map},
            df=pd.DataFrame({"impact": ["Unknown"]}),
        )

    out_series = mod.resolve_enacting_metric_unit(
        context=context,
        key=key_cap,
        year_map={2019: pd.Series([1.0], index=pd.Index(["FR"], name="r_f"))},
        mrio_default_monetary_unit="M EUR",
        mrio_units={},
        lcia_units={"gwp100": unit_map},
        df=pd.DataFrame({"impact": ["Climate"]}),
    )
    assert isinstance(out_series, pd.Series)
    assert out_series.iloc[0] == "kgCO2e/cap"

    out_series_no_cap = mod.resolve_enacting_metric_unit(
        context=context,
        key=key_method,
        year_map={2019: pd.Series([1.0], index=pd.Index(["FR"], name="r_f"))},
        mrio_default_monetary_unit="M EUR",
        mrio_units={},
        lcia_units={"gwp100": unit_map},
        df=pd.DataFrame({"impact": ["Climate"]}),
    )
    assert isinstance(out_series_no_cap, pd.Series)
    assert out_series_no_cap.iloc[0] == "kgCO2e"


def test_resolve_enacting_metric_unit_pop_gdp_and_mrio_paths() -> None:
    context = _context()

    key_pop = EnactingMetricKey(metric="population", ssp_scenario="SSP2")
    pop_unit = mod.resolve_enacting_metric_unit(
        context=context,
        key=key_pop,
        year_map={2019: pd.Series([1.0]), 2025: pd.Series([1.0])},
        mrio_default_monetary_unit="M EUR",
        mrio_units={},
        lcia_units={},
        df=pd.DataFrame({"value": [1.0]}),
    )
    assert pop_unit == "people"

    key_gdp_cap = EnactingMetricKey(metric="gdp_capita", ssp_scenario="SSP2")
    gdp_cap_unit = mod.resolve_enacting_metric_unit(
        context=context,
        key=key_gdp_cap,
        year_map={2019: pd.Series([1.0]), 2025: pd.Series([1.0])},
        mrio_default_monetary_unit="M EUR",
        mrio_units={},
        lcia_units={},
        df=pd.DataFrame({"value": [1.0]}),
    )
    assert gdp_cap_unit == "USD/cap"

    explicit = mod.resolve_enacting_metric_unit(
        context=context,
        key=EnactingMetricKey(metric="fd_rf"),
        year_map={2019: pd.Series([1.0])},
        mrio_default_monetary_unit="M EUR",
        mrio_units={"fd_rf": "M EUR"},
        lcia_units={},
        df=pd.DataFrame({"value": [1.0]}),
    )
    assert explicit == "M EUR"

    default_mrio = mod.resolve_enacting_metric_unit(
        context=context,
        key=EnactingMetricKey(metric="x_rp_sp"),
        year_map={2019: pd.Series([1.0])},
        mrio_default_monetary_unit="M EUR",
        mrio_units={},
        lcia_units={},
        df=pd.DataFrame({"value": [1.0]}),
    )
    assert default_mrio == "M EUR"

    with pytest.raises(ValueError):
        mod.resolve_enacting_metric_unit(
            context=context,
            key=EnactingMetricKey(metric="unknown_metric"),
            year_map={2019: pd.Series([1.0])},
            mrio_default_monetary_unit="M EUR",
            mrio_units={},
            lcia_units={},
            df=pd.DataFrame({"value": [1.0]}),
        )


def test_resolve_enacting_metric_unit_pop_gdp_single_source_branches() -> None:
    context_wb = _context(wb_year_cols=("2019", "2020"))
    pop_key = EnactingMetricKey(metric="population", ssp_scenario="SSP2")
    gdp_key = EnactingMetricKey(metric="gdp_capita", ssp_scenario="SSP2")

    # WB only years: use_wb=True, use_ssp=False
    assert (
        mod.resolve_enacting_metric_unit(
            context=context_wb,
            key=pop_key,
            year_map={2019: pd.Series([1.0])},
            mrio_default_monetary_unit="M EUR",
            mrio_units={},
            lcia_units={},
            df=pd.DataFrame({"value": [1.0]}),
        )
        == "people"
    )
    assert (
        mod.resolve_enacting_metric_unit(
            context=context_wb,
            key=gdp_key,
            year_map={2020: pd.Series([1.0])},
            mrio_default_monetary_unit="M EUR",
            mrio_units={},
            lcia_units={},
            df=pd.DataFrame({"value": [1.0]}),
        )
        == "USD/cap"
    )

    # SSP only years: use_wb=False, use_ssp=True
    context_ssp = _context(wb_year_cols=("2019",))
    assert (
        mod.resolve_enacting_metric_unit(
            context=context_ssp,
            key=pop_key,
            year_map={2035: pd.Series([1.0])},
            mrio_default_monetary_unit="M EUR",
            mrio_units={},
            lcia_units={},
            df=pd.DataFrame({"value": [1.0]}),
        )
        == "people"
    )
    assert (
        mod.resolve_enacting_metric_unit(
            context=context_ssp,
            key=gdp_key,
            year_map={2040: pd.Series([1.0])},
            mrio_default_monetary_unit="M EUR",
            mrio_units={},
            lcia_units={},
            df=pd.DataFrame({"value": [1.0]}),
        )
        == "USD/cap"
    )
