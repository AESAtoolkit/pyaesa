from pyaesa.shared.selectors.aggregate_labels import (
    aggregate_selector_label,
    aggregate_selector_label_or_none,
)


def test_aggregate_selector_labels_cover_sorting_and_optional_contracts() -> None:
    assert aggregate_selector_label(["US", "FR", "US"]) == "FR, US"

    assert aggregate_selector_label_or_none(None) is None
    assert aggregate_selector_label_or_none("FR") is None
    assert aggregate_selector_label_or_none(42) is None
    assert aggregate_selector_label_or_none([2, "1", 2]) == "1, 2"
