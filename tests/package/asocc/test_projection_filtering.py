from pyaesa.asocc.orchestration.projection.regression import filtering as mod


def test_projection_filtering_cover_empty_and_selected_levels() -> None:
    filters = {
        "r_p": ["FR", 7],
        "s_p": [],
        "r_c": None,
    }
    assert mod.selected_values_for_level(filters=filters, level="r_p") == ["FR", "7"]
    assert mod.selected_values_for_level(filters=filters, level="s_p") is None
    assert mod.selected_values_for_level(filters=filters, level="r_f") is None
    assert mod.selected_values_for_levels(filters=filters, levels=["r_p", "s_p", "r_f"]) == {
        "r_p": ["FR", "7"],
        "s_p": None,
        "r_f": None,
    }
