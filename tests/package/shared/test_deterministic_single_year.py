from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pyaesa import set_workspace
from pyaesa.shared.figures.deterministic_single_year import render_single_year_panels
from pyaesa.shared.figures.lcia_metadata import load_lcia_metadata
from pyaesa.shared.tabular.scalars import sanitize_token
from pyaesa.workspace_initialisation.workspace import clear_default_repo_root


def _title_parts() -> dict[str, str | None]:
    return {
        "family": "Demo family",
        "selector_scope": "r_p=FR",
        "lcia_method": "pb_lcia",
        "user_facing_override_label": None,
        "prospective_scope": None,
    }


def _impact_frame() -> pd.DataFrame:
    metadata = load_lcia_metadata("pb_lcia")
    assert metadata.n_impacts >= 2
    first_impact, second_impact = metadata.impacts[:2]
    return pd.DataFrame(
        {
            "year": [2030, 2030, 2030, 2030],
            "value": [10.0, 20.0, 30.0, 40.0],
            "impact": [first_impact, first_impact, second_impact, second_impact],
            "lcia_method": ["pb_lcia"] * 4,
            "series_label": ["Alpha", "Beta", "Gamma", "Delta"],
        }
    )


def _overlay_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2031, 2031],
            "value": [5.0, 7.0],
            "series_label": ["Overlay A", "Overlay B"],
        }
    )


def _single_panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2032, 2032],
            "value": [5.0, 7.0],
            "series_label": ["Overlay A", "Overlay B"],
        }
    )


def _single_impact_frame() -> pd.DataFrame:
    metadata = load_lcia_metadata("pb_lcia")
    assert metadata.n_impacts >= 1
    impact = metadata.impacts[0]
    return pd.DataFrame(
        {
            "year": [2032, 2032],
            "value": [5.0, 7.0],
            "impact": [impact, impact],
            "lcia_method": ["pb_lcia", "pb_lcia"],
            "series_label": ["Overlay A", "Overlay B"],
        }
    )


def _recording_axis_styler(call_log: list[int]):
    def styler(axis: Any, panel_rows: pd.DataFrame) -> None:
        call_log.append(len(panel_rows))
        axis.plot([0.0, 1.0], [1.0, 1.0], label="Threshold")

    return styler


def test_render_single_year_panels_covers_empty_overlay_grid_and_split_paths(
    tmp_path: Path,
) -> None:
    clear_default_repo_root()
    set_workspace(tmp_path / "workspace", refresh=True)
    try:
        output_root = tmp_path / "figures"

        assert (
            render_single_year_panels(
                frame=pd.DataFrame(),
                years=[2030],
                output_base=output_root / "empty",
                title_parts=_title_parts(),
                ylabel="Value",
                dpi=10,
                output_format="png",
            )
            == []
        )

        overlay_paths = render_single_year_panels(
            frame=_overlay_frame(),
            years=[2031],
            output_base=output_root / "overlay",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            overlay_panels=True,
            percent_ticks=True,
            force_zero_ymin=True,
            footer_note="Overlay note",
        )
        assert overlay_paths == [output_root / "overlay__2031.png"]
        assert overlay_paths[0].is_file()

        assert render_single_year_panels(
            frame=_overlay_frame(),
            years=[2030, 2031],
            output_base=output_root / "missing_year",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
        ) == [output_root / "missing_year__2031.png"]

        grid_axis_styler_calls: list[int] = []
        grid_paths = render_single_year_panels(
            frame=_impact_frame(),
            years=[2030],
            output_base=output_root / "grid",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            ylabel_resolver=lambda frame: f"Value ({len(frame)})",
            percent_ticks=True,
            force_zero_ymin=True,
            axis_styler=_recording_axis_styler(grid_axis_styler_calls),
        )
        assert grid_paths == [output_root / "grid__2030.png"]
        assert grid_paths[0].is_file()
        assert grid_axis_styler_calls == [2, 2]

        axis_styler_calls: list[int] = []
        styled_paths = render_single_year_panels(
            frame=_single_panel_frame(),
            years=[2032],
            output_base=output_root / "styled_overlay",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            overlay_panels=True,
            axis_styler=_recording_axis_styler(axis_styler_calls),
        )
        assert styled_paths == [output_root / "styled_overlay__2032.png"]
        assert axis_styler_calls == [2]

        split_paths = render_single_year_panels(
            frame=_impact_frame(),
            years=[2030],
            output_base=output_root / "split",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            split_panels=True,
            footer_note="Split note",
            force_zero_ymin=True,
        )
        assert len(split_paths) == 2
        assert all(path.is_file() for path in split_paths)
        split_impact_tokens = {
            sanitize_token(value)
            for value in pd.Series(_impact_frame()["impact"], copy=False).drop_duplicates().tolist()
        }
        assert {path.name for path in split_paths} == {
            f"split__{token}__2030.png" for token in split_impact_tokens
        }

        split_axis_styler_calls: list[int] = []
        split_styled_paths = render_single_year_panels(
            frame=_impact_frame(),
            years=[2030],
            output_base=output_root / "split_styled",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            split_panels=True,
            axis_styler=_recording_axis_styler(split_axis_styler_calls),
        )
        assert len(split_styled_paths) == 2
        assert split_axis_styler_calls == [2, 2]
        assert {path.name for path in split_styled_paths} == {
            f"split_styled__{token}__2030.png" for token in split_impact_tokens
        }

        single_panel_paths = render_single_year_panels(
            frame=_single_panel_frame(),
            years=[2032],
            output_base=output_root / "single_panel",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            overlay_panels=True,
            force_zero_ymin=True,
        )
        assert single_panel_paths == [output_root / "single_panel__2032.png"]
        assert single_panel_paths[0].is_file()

        titled_overlay_paths = render_single_year_panels(
            frame=_single_impact_frame(),
            years=[2032],
            output_base=output_root / "titled_overlay",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            overlay_panels=True,
            force_zero_ymin=True,
        )
        assert titled_overlay_paths == [output_root / "titled_overlay__2032.png"]
        assert titled_overlay_paths[0].is_file()

        multi_overlay_paths = render_single_year_panels(
            frame=_impact_frame(),
            years=[2030],
            output_base=output_root / "multi_overlay",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            overlay_panels=True,
        )
        assert multi_overlay_paths == [output_root / "multi_overlay__2030.png"]
        assert multi_overlay_paths[0].is_file()

        with pytest.raises(ValueError):
            render_single_year_panels(
                frame=pd.DataFrame({"year": [2030], "value": [1.0]}),
                years=[2030],
                output_base=output_root / "missing_label",
                title_parts=_title_parts(),
                ylabel="Value",
                dpi=10,
                output_format="png",
                overlay_panels=True,
            )

        with pytest.raises(ValueError):
            render_single_year_panels(
                frame=pd.DataFrame(
                    {
                        "year": [2031],
                        "value": [2.0],
                        "series_label": [" "],
                    }
                ),
                years=[2031],
                output_base=output_root / "empty_label_overlay",
                title_parts=_title_parts(),
                ylabel="Value",
                dpi=10,
                output_format="png",
                overlay_panels=True,
            )

        with pytest.raises(ValueError):
            render_single_year_panels(
                frame=pd.DataFrame(
                    {
                        "year": [2030],
                        "value": [1.0],
                        "impact": [load_lcia_metadata("pb_lcia").impacts[0]],
                        "lcia_method": ["pb_lcia"],
                    }
                ),
                years=[2030],
                output_base=output_root / "missing_label_grid",
                title_parts=_title_parts(),
                ylabel="Value",
                dpi=10,
                output_format="png",
            )

        split_percent_paths = render_single_year_panels(
            frame=_impact_frame(),
            years=[2030],
            output_base=output_root / "split_percent",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            split_panels=True,
            percent_ticks=True,
            force_zero_ymin=True,
        )
        assert len(split_percent_paths) == 2
        assert all(path.is_file() for path in split_percent_paths)
        assert {path.name for path in split_percent_paths} == {
            f"split_percent__{token}__2030.png" for token in split_impact_tokens
        }

        split_no_zero_paths = render_single_year_panels(
            frame=_impact_frame(),
            years=[2030],
            output_base=output_root / "split_no_zero",
            title_parts=_title_parts(),
            ylabel="Value",
            dpi=10,
            output_format="png",
            split_panels=True,
        )
        assert len(split_no_zero_paths) == 2
        assert all(path.is_file() for path in split_no_zero_paths)
        assert {path.name for path in split_no_zero_paths} == {
            f"split_no_zero__{token}__2030.png" for token in split_impact_tokens
        }

        with pytest.raises(ValueError):
            render_single_year_panels(
                frame=pd.DataFrame({"year": [2030], "value": [1.0], "series_label": [" "]}),
                years=[2030],
                output_base=output_root / "empty_label",
                title_parts=_title_parts(),
                ylabel="Value",
                dpi=10,
                output_format="png",
                split_panels=True,
            )
    finally:
        clear_default_repo_root()


def test_render_single_year_panels_rejects_missing_or_empty_series_labels(
    tmp_path: Path,
) -> None:
    clear_default_repo_root()
    set_workspace(tmp_path / "workspace", refresh=True)
    try:
        base_frame = pd.DataFrame(
            {
                "year": [2030],
                "value": [1.0],
                "series_label": [" "],
            }
        )

        with pytest.raises(ValueError):
            render_single_year_panels(
                frame=base_frame,
                years=[2030],
                output_base=tmp_path / "bad",
                title_parts=_title_parts(),
                ylabel="Value",
                dpi=10,
                output_format="png",
            )

        with pytest.raises(ValueError):
            render_single_year_panels(
                frame=base_frame.drop(columns=["series_label"]),
                years=[2030],
                output_base=tmp_path / "bad_missing",
                title_parts=_title_parts(),
                ylabel="Value",
                dpi=10,
                output_format="png",
                split_panels=True,
            )
    finally:
        clear_default_repo_root()
