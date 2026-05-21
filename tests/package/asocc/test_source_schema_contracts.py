import pytest

from pyaesa.asocc.data.source_schema import (
    default_historical_cutoff_for_source,
    default_regression_window_for_source,
    is_exio_source,
    is_iso3_source,
    max_modeled_year_for_source,
    min_modeled_year_for_source,
    region_code_column_for_source,
)


def test_source_schema_resolves_known_sources() -> None:
    assert region_code_column_for_source(" ISO3 ") == "iso3_code"
    assert region_code_column_for_source("exiobase_396_ixi") == "exio_code"
    assert region_code_column_for_source("exiobase_396_pxp") == "exio_code"
    assert region_code_column_for_source("oecd_v2025") == "oecd_code"
    assert region_code_column_for_source("iso3") == "iso3_code"

    assert is_exio_source("exiobase_396_ixi")
    assert is_exio_source("exiobase_396_pxp")
    assert not is_exio_source("oecd_v2025")
    assert not is_exio_source("iso3")

    assert is_iso3_source("iso3")
    assert not is_iso3_source("oecd_v2025")

    assert max_modeled_year_for_source("iso3") is None
    assert min_modeled_year_for_source("iso3") is None
    assert default_historical_cutoff_for_source("iso3") is None
    assert default_regression_window_for_source("iso3") is None
    assert max_modeled_year_for_source("exiobase_396_ixi") == 2022
    assert min_modeled_year_for_source("exiobase_396_ixi") == 1995
    assert default_historical_cutoff_for_source("exiobase_396_ixi") == 2019
    assert default_regression_window_for_source("exiobase_396_ixi") == (1995, 2019)
    assert default_historical_cutoff_for_source("oecd_v2025") == 2022
    assert default_regression_window_for_source("oecd_v2025") == (1995, 2022)


def test_source_schema_rejects_unsupported_source() -> None:
    with pytest.raises(ValueError):
        region_code_column_for_source("unknown")
