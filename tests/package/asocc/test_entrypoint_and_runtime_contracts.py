import pandas as pd
import pytest

from pyaesa.asocc.entrypoints.argument_contracts import (
    ensure_list_str,
    normalize_allocate_output_format,
    validate_grouped_request,
)
from pyaesa.asocc.runtime.methods.fallback_policy import (
    resolve_latest_available_historical_year,
    resolve_latest_previous_nonzero_series,
)
from pyaesa.asocc.runtime.methods.labels import l1_l2_method_label, parse_raw_asocc_method_label
from pyaesa.shared.selectors.scenarios import partition_token_to_ssp_token
from pyaesa.shared.tabular.contracts import suffix_for_tabular_output


def test_entrypoint_argument_contracts_cover_grouped_request_branches() -> None:
    assert normalize_allocate_output_format(" CSV ") == "csv"
    assert ensure_list_str(None) is None
    assert ensure_list_str("FR") == ["FR"]
    assert ensure_list_str(["FR", "DE"]) == ["FR", "DE"]

    validate_grouped_request(
        fu_norm="L1.a",
        grouped_requested=False,
        r_p=None,
        s_p=None,
        r_c=None,
        r_f=None,
    )
    validate_grouped_request(
        fu_norm="L1.a",
        grouped_requested=True,
        r_p=["FR", "DE"],
        s_p=None,
        r_c=None,
        r_f=None,
    )
    validate_grouped_request(
        fu_norm="L2.a.a",
        grouped_requested=True,
        r_p=None,
        s_p=["A", "B"],
        r_c=None,
        r_f=None,
    )
    validate_grouped_request(
        fu_norm="other",
        grouped_requested=True,
        r_p=None,
        s_p=None,
        r_c=None,
        r_f=None,
    )
    with pytest.raises(ValueError):
        validate_grouped_request(
            fu_norm="L1.a",
            grouped_requested=True,
            r_p=["FR"],
            s_p=["A", "B"],
            r_c=None,
            r_f=None,
        )
    with pytest.raises(ValueError):
        validate_grouped_request(
            fu_norm="L2.a.a",
            grouped_requested=True,
            r_p=["FR"],
            s_p=["A"],
            r_c=None,
            r_f=None,
        )


def test_runtime_method_contracts_cover_fallback_and_label_failures() -> None:
    exact = resolve_latest_available_historical_year(
        requested_year=2005, available_years=[2004, 2005]
    )
    assert exact is not None
    assert not exact.used_fallback
    assert (
        resolve_latest_available_historical_year(requested_year=2000, available_years=[2004])
        is None
    )

    series_by_year = {
        2003: pd.Series([0.0]),
        2004: pd.Series([2.0]),
        2005: pd.Series([0.0]),
    }

    def _load(year: int) -> pd.Series | None:
        return series_by_year.get(year)

    def _is_zero(series: pd.Series) -> bool:
        return bool((series == 0.0).all())

    reused, fallback = resolve_latest_previous_nonzero_series(
        requested_year=2005,
        available_years=[2003, 2004, 2005],
        load_series=_load,
        is_zero_placeholder=_is_zero,
    )
    assert reused is not None
    assert reused.iloc[0] == 2.0
    assert fallback is not None and fallback.used_fallback

    missing, missing_fallback = resolve_latest_previous_nonzero_series(
        requested_year=2006,
        available_years=[2004],
        load_series=_load,
        is_zero_placeholder=_is_zero,
    )
    assert missing is None
    assert missing_fallback is None

    still_zero, no_fallback = resolve_latest_previous_nonzero_series(
        requested_year=2005,
        available_years=[2003, 2005],
        load_series=_load,
        is_zero_placeholder=_is_zero,
    )
    assert still_zero is series_by_year[2005]
    assert no_fallback is None

    assert l1_l2_method_label(l1_method=" PR(GDPcap) ", l2_method=" UT(FD) ") == (
        "PR(GDPcap)_UT(FD)"
    )
    assert parse_raw_asocc_method_label("PR-HR(Ecap,cum)") == ("PR", "HR", "Ecap,cum")
    for label in ("bad", "()", "-(x)"):
        with pytest.raises(ValueError):
            parse_raw_asocc_method_label(label)


def test_shared_selector_and_tabular_contract_edges() -> None:
    assert partition_token_to_ssp_token("ssp2", context="test") == "SSP2"
    with pytest.raises(ValueError):
        partition_token_to_ssp_token("scenario2", context="test")
    assert suffix_for_tabular_output("parquet") == ".parquet"
