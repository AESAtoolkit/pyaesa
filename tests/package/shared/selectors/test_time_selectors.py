from typing import Any, cast

import pytest

from pyaesa.shared.selectors import time_selectors


def test_optional_year_and_reg_window_selectors() -> None:
    assert time_selectors.normalize_requested_years(2030) == [2030]
    assert time_selectors.normalize_requested_years(range(2030, 2032)) == [2030, 2031]
    assert time_selectors.normalize_requested_years([2031, 2030]) == [2031, 2030]

    assert time_selectors.normalize_optional_year_selector(None, name="years") is None
    assert time_selectors.normalize_optional_year_selector(2030, name="years") == [2030]
    assert time_selectors.normalize_optional_year_selector(range(2030, 2032), name="years") == [
        2030,
        2031,
    ]
    assert time_selectors.normalize_optional_year_selector([2031, 2030, 2030], name="years") == [
        2030,
        2031,
    ]
    with pytest.raises(ValueError):
        time_selectors.normalize_optional_year_selector(cast(Any, (2030, 2031)), name="years")

    assert time_selectors.normalize_optional_reg_window_selector(None) is None
    assert time_selectors.normalize_optional_reg_window_selector(range(2018, 2020)) == [
        2018,
        2019,
    ]
    assert time_selectors.normalize_optional_reg_window_selector([2018, 2019]) == [2018, 2019]
    with pytest.raises(ValueError):
        time_selectors.normalize_optional_reg_window_selector(cast(Any, 2018))


def test_normalize_time_selector_mapping() -> None:
    assert time_selectors.normalize_time_selector_mapping(None) is None
    assert time_selectors.normalize_time_selector_mapping(
        {
            "years": [2031, 2030],
            "reference_years": range(2005, 2007),
            "l2_reuse_years": 2019,
            "reg_window": range(2018, 2020),
            "other": "keep",
        }
    ) == {
        "years": [2030, 2031],
        "reference_years": [2005, 2006],
        "l2_reuse_years": [2019],
        "reg_window": [2018, 2019],
        "other": "keep",
    }
    assert time_selectors.normalize_time_selector_mapping(
        {"years": [2030]},
        year_keys=("years", "missing_year"),
        reg_window_keys=("reg_window", "missing_reg_window"),
    ) == {"years": [2030]}


def test_normalize_reg_window_for_storage_covers_tuple_and_list_contracts() -> None:
    assert time_selectors.normalize_reg_window_for_storage(None) is None
    assert time_selectors.normalize_reg_window_for_storage((2018, 2020)) == [
        2018,
        2019,
        2020,
    ]
    assert time_selectors.normalize_reg_window_for_storage([2018, 2019, 2020]) == [
        2018,
        2019,
        2020,
    ]
    with pytest.raises(ValueError):
        time_selectors.normalize_reg_window_for_storage(cast(Any, (2018,)))
    with pytest.raises(ValueError):
        time_selectors.normalize_reg_window_for_storage(cast(Any, (2020, 2018)))
    with pytest.raises(ValueError):
        time_selectors.normalize_reg_window_for_storage([])
    with pytest.raises(ValueError):
        time_selectors.normalize_reg_window_for_storage([2018, 2020])
