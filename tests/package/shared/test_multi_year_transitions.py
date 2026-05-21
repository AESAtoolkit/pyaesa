import math

import matplotlib.pyplot as plt
import pandas as pd

from pyaesa.shared.figures import multi_year_transitions as transitions_mod


def test_marker_columns_and_requested_year_contracts() -> None:
    assert transitions_mod.marker_year_column() == "__transition_marker_year"
    assert transitions_mod.marker_label_column() == "__transition_marker_label"
    assert transitions_mod.marker_color_column() == "__transition_marker_color"

    assert transitions_mod.is_missing_scalar(None) is True
    assert transitions_mod.is_missing_scalar(pd.NA) is True
    assert transitions_mod.is_missing_scalar(pd.NaT) is True
    assert transitions_mod.is_missing_scalar(float("nan")) is True
    assert transitions_mod.is_missing_scalar(0.0) is False
    assert transitions_mod.is_missing_scalar(1) is False

    assert transitions_mod.normalized_requested_years([2031, 2030, 2030]) == [
        2030,
        2031,
    ]
    assert transitions_mod.normalized_requested_years([]) == []


def test_scenario_plan_and_stem_contracts() -> None:
    scenario_plan = {2030: [None], 2035: ["SSP2", "SSP3"], 2040: ["SSP2"]}

    assert transitions_mod.ssp_tokens_from_plan(
        ssp_scenario_options_by_year=scenario_plan,
        requested_years=[2030, 2035, 2040],
    ) == ["SSP2", "SSP3"]
    assert (
        transitions_mod.ssp_tokens_from_plan(
            ssp_scenario_options_by_year=None,
            requested_years=[2030],
        )
        == []
    )
    assert transitions_mod.stem_ssp_suffix(
        stem="series_SSP2",
        ssp_tokens=["SSP2", "SSP"],
    ) == ("series", "SSP2")
    assert transitions_mod.stem_ssp_suffix(
        stem="series",
        ssp_tokens=["SSP2"],
    ) == ("series", None)
    assert transitions_mod.generic_ssp_suffix("series_SSP2") == ("series", "SSP2")
    assert transitions_mod.generic_ssp_suffix("series") == ("series", None)

    frame = pd.DataFrame({"scenario": [None, "SSP3", "SSP2", "SSP2"]})
    assert transitions_mod.non_null_prospective_values(frame) == ["SSP2", "SSP3"]
    assert transitions_mod.non_null_prospective_values(pd.DataFrame({"other": [1]})) == []


def test_expand_historical_rows_and_marker_collection() -> None:
    missing_columns = transitions_mod.expand_historical_rows_for_prospective_series(
        pd.DataFrame({"year": [2030], "value": [1.0]}),
        grouping_columns=[],
    )
    assert list(missing_columns.columns)[-3:] == [
        "__transition_marker_year",
        "__transition_marker_label",
        "__transition_marker_color",
    ]
    assert bool(missing_columns["__transition_marker_year"].isna().all())

    historical_only = transitions_mod.expand_historical_rows_for_prospective_series(
        pd.DataFrame(
            {
                "group": ["A", "A"],
                "year": [2030, 2031],
                "scenario": [None, None],
                "value": [1.0, 2.0],
            }
        ),
        grouping_columns=["group"],
    )
    assert bool(historical_only["__transition_marker_year"].isna().all())

    scenario_only = transitions_mod.expand_historical_rows_for_prospective_series(
        pd.DataFrame(
            {
                "group": ["A", "A"],
                "year": [2031, 2032],
                "scenario": ["SSP2", "SSP2"],
                "value": [2.0, 3.0],
            }
        ),
        grouping_columns=["group"],
    )
    assert bool(scenario_only["__transition_marker_year"].isna().all())

    mixed = transitions_mod.expand_historical_rows_for_prospective_series(
        pd.DataFrame(
            {
                "group": ["A", "A", "A"],
                "year": [2030, 2031, 2032],
                "scenario": [None, "SSP2", "SSP2"],
                "value": [1.0, 3.0, 4.0],
            }
        ),
        grouping_columns=["group"],
        marker_label="switch",
        marker_color="#112233",
    )
    assert mixed["scenario"].tolist() == ["SSP2", "SSP2", "SSP2"]
    assert set(mixed["__transition_marker_year"].dropna().tolist()) == {2031}
    assert transitions_mod.markers_from_frame(mixed) == [
        transitions_mod.TransitionMarker(year=2031, label="switch", color="#112233")
    ]
    assert (
        transitions_mod.markers_from_frame(
            pd.DataFrame(
                columns=[
                    "__transition_marker_year",
                    "__transition_marker_label",
                    "__transition_marker_color",
                ]
            )
        )
        == []
    )
    assert transitions_mod.markers_from_frame(pd.DataFrame({"year": [2030]})) == []
    assert (
        transitions_mod.markers_from_frame(
            pd.DataFrame(
                {
                    "__transition_marker_year": [2031, 2032],
                    "__transition_marker_label": [None, ""],
                    "__transition_marker_color": [None, "#112233"],
                }
            )
        )
        == []
    )


def test_render_transition_markers_and_boundary_position() -> None:
    fig, ax = plt.subplots()
    markers = [
        transitions_mod.TransitionMarker(year=2035, label="switch", color="#123456"),
        transitions_mod.TransitionMarker(year=2035, label="switch", color="#123456"),
    ]

    transitions_mod.render_transition_markers(axis=ax, markers=markers)
    transitions_mod.render_transition_markers(axis=ax, markers=[])

    assert len(ax.lines) == 2
    assert [text.get_text() for text in ax.texts] == ["switch"]
    assert math.isclose(transitions_mod.transition_boundary_x(2035), 2035.0)

    plt.close(fig)

    fig, ax = plt.subplots()
    close_markers = [
        transitions_mod.TransitionMarker(year=2035, label="LCA detail", color="#123456"),
        transitions_mod.TransitionMarker(year=2036, label="aSoCC detail", color="#123456"),
    ]
    transitions_mod.render_transition_markers(axis=ax, markers=close_markers)

    assert [text.get_text() for text in ax.texts] == [
        "LCA detail",
        "aSoCC detail",
    ]
    text_y_positions = [text.get_transform().transform(text.get_position())[1] for text in ax.texts]
    assert text_y_positions[0] < text_y_positions[1]

    plt.close(fig)

    fig, ax = plt.subplots()
    component_markers = [
        transitions_mod.TransitionMarker(year=2035, label="LCA", color="#123456"),
        transitions_mod.TransitionMarker(year=2036, label="aSoCC", color="#123456"),
    ]
    transitions_mod.render_transition_markers(axis=ax, markers=component_markers)

    component_text_labels = [text.get_text() for text in ax.texts]
    assert len(component_text_labels) == 3
    assert component_text_labels[0]
    assert component_text_labels[1:] == ["LCA", "aSoCC"]
    component_text_y_positions = [
        text.get_transform().transform(text.get_position())[1] for text in ax.texts
    ]
    assert component_text_y_positions[0] > component_text_y_positions[1]
    assert component_text_y_positions[0] - component_text_y_positions[1] < 18.0
    assert (
        transitions_mod.transition_title_pad(
            component_markers,
            no_transition=1,
            single_transition=2,
            component_transition=3,
        )
        == 3
    )
    assert (
        transitions_mod.transition_title_pad(
            close_markers,
            no_transition=1,
            single_transition=2,
            component_transition=3,
        )
        == 2
    )
    assert (
        transitions_mod.transition_title_pad(
            [],
            no_transition=1,
            single_transition=2,
            component_transition=3,
        )
        == 1
    )

    plt.close(fig)
