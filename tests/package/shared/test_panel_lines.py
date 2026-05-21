from pathlib import Path

import pytest

from pyaesa.shared.figures import panel_lines as lines_mod
from pyaesa.shared.figures.multi_year_transitions import TransitionMarker


def _title_parts() -> dict[str, str | None]:
    return {
        "family": "Demo family",
        "selector_scope": "r_p=FR",
        "lcia_method": None,
        "user_facing_override_label": None,
        "prospective_scope": None,
    }


def _panel_series() -> lines_mod.PanelSeries:
    return [
        (
            "Climate",
            [
                ("Series A", [2030, 2031], [10.0, 20.0], "One step methods"),
                ("Series B", [2030, 2031], [12.0, 24.0], "Two step methods"),
            ],
        ),
        (
            "Land",
            [
                ("Series C", [2030, 2031], [5.0, 9.0], "One step methods"),
            ],
        ),
    ]


def _panel_series_without_labels() -> lines_mod.PanelSeries:
    return [
        (
            "Climate",
            [
                ("", [2030, 2031], [10.0, 20.0], "One step methods"),
            ],
        ),
        (
            "Land",
            [
                ("", [2030, 2031], [5.0, 9.0], "One step methods"),
            ],
        ),
    ]


def _many_panel_series(count: int = 7, *, blank_first: bool = False) -> lines_mod.PanelSeries:
    return [
        (
            "" if blank_first and index == 1 else f"Panel {index}",
            [
                (f"Series {index}", [2030, 2031], [float(index), float(index + 1)], "Group"),
            ],
        )
        for index in range(1, count + 1)
    ]


def test_render_panel_series_covers_empty_overlay_split_and_grid(tmp_path: Path) -> None:
    output_base = tmp_path / "panels"
    markers: lines_mod.PanelMarkers = {
        "Climate": [TransitionMarker(year=2031, label="switch", color="#123456")],
        "Land": [TransitionMarker(year=2031, label="switch", color="#654321")],
    }

    assert (
        lines_mod.render_panel_series(
            panel_series=[],
            output_path=output_base,
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
        )
        == []
    )

    overlay_paths = lines_mod.render_panel_series(
        panel_series=_panel_series(),
        output_path=output_base / "overlay",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        y_percent=True,
        panel_markers=markers,
        overlay_panels=True,
        footer_note="Overlay note",
        force_zero_ymin=True,
    )
    assert overlay_paths == [output_base / "overlay.png"]
    assert overlay_paths[0].is_file()

    overlay_plain_paths = lines_mod.render_panel_series(
        panel_series=_panel_series_without_labels(),
        output_path=output_base / "overlay_plain",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        overlay_panels=True,
    )
    assert overlay_plain_paths == [output_base / "overlay_plain.png"]
    assert overlay_plain_paths[0].is_file()

    split_paths = lines_mod.render_panel_series(
        panel_series=_panel_series(),
        output_path=output_base / "split",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        panel_markers=markers,
        split_panels=True,
        footer_note="Split note",
        force_zero_ymin=True,
    )
    assert len(split_paths) == 2
    assert all(path.is_file() for path in split_paths)
    assert {path.name for path in split_paths} == {"split__Climate.png", "split__Land.png"}

    split_percent_paths = lines_mod.render_panel_series(
        panel_series=_panel_series(),
        output_path=output_base / "split_percent",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        split_panels=True,
        y_percent=True,
    )
    assert len(split_percent_paths) == 2
    assert all(path.is_file() for path in split_percent_paths)

    split_plain_paths = lines_mod.render_panel_series(
        panel_series=_panel_series_without_labels(),
        output_path=output_base / "split_plain",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        split_panels=True,
    )
    assert len(split_plain_paths) == 2
    assert all(path.is_file() for path in split_plain_paths)
    assert {path.name for path in split_plain_paths} == {
        "split_plain__Climate.png",
        "split_plain__Land.png",
    }

    grid_paths = lines_mod.render_panel_series(
        panel_series=_panel_series(),
        output_path=output_base / "grid",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        panel_markers=markers,
        footer_note="Grid note",
        force_zero_ymin=True,
    )
    assert grid_paths == [output_base / "grid.png"]
    assert grid_paths[0].is_file()

    grid_percent_paths = lines_mod.render_panel_series(
        panel_series=_panel_series(),
        output_path=output_base / "grid_percent",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        y_percent=True,
    )
    assert grid_percent_paths == [output_base / "grid_percent.png"]
    assert grid_percent_paths[0].is_file()

    many_grid_paths = lines_mod.render_panel_series(
        panel_series=_many_panel_series(),
        output_path=output_base / "grid_many",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
    )
    assert many_grid_paths == [output_base / "grid_many.png"]
    assert many_grid_paths[0].is_file()

    blank_title_grid_paths = lines_mod.render_panel_series(
        panel_series=_many_panel_series(blank_first=True),
        output_path=output_base / "grid_blank",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        y_percent=True,
    )
    assert blank_title_grid_paths == [output_base / "grid_blank.png"]
    assert blank_title_grid_paths[0].is_file()


def test_render_panel_series_single_panel_title_and_missing_labels(tmp_path: Path) -> None:
    single_paths = lines_mod.render_panel_series(
        panel_series=[
            (
                "Climate",
                [
                    ("", [2030, 2031], [1.0, 2.0], "One step methods"),
                ],
            )
        ],
        output_path=tmp_path / "single",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        split_panels=True,
        force_zero_ymin=True,
    )
    assert single_paths == [tmp_path / "single__Climate.png"]
    assert single_paths[0].is_file()

    no_label_grid = lines_mod.render_panel_series(
        panel_series=[
            (
                "Climate",
                [
                    ("", [2030, 2031], [1.0, 2.0], "One step methods"),
                ],
            )
        ],
        output_path=tmp_path / "nolabel",
        title_parts=_title_parts(),
        ylabel="Value",
        dpi=10,
        output_format="png",
        force_zero_ymin=True,
    )
    assert no_label_grid == [tmp_path / "nolabel.png"]
    assert no_label_grid[0].is_file()


def test_render_panel_series_rejects_negative_values_when_zero_baseline_forced(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        lines_mod.render_panel_series(
            panel_series=[
                (
                    "Climate",
                    [("Series A", [2030, 2031], [-1.0, 2.0], "One step methods")],
                )
            ],
            output_path=tmp_path / "bad",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            force_zero_ymin=True,
        )
