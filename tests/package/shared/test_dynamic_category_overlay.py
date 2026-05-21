from pathlib import Path

import pandas as pd

from pyaesa import set_workspace
from pyaesa.shared.figures import dynamic_category_overlay as overlay_mod
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN


def test_render_selector_overlay_writes_grouped_panel_outputs_and_uses_hooks(
    tmp_path: Path,
) -> None:
    prepared = pd.DataFrame(
        {
            "year": [2030, 2035, 2030, 2035, 2030, 2035, 2030, 2035],
            "value": [1.0, 1.2, 0.8, 0.9, 2.0, 2.1, 1.7, 1.8],
            "cc_category": [
                "safe",
                "safe",
                "safe",
                "safe",
                "risk",
                "risk",
                "risk",
                "risk",
            ],
            "impact": ["AAL"] * 8,
            "lcia_method": ["pb_lcia"] * 8,
            "impact_unit": ["ratio"] * 8,
            "region": ["EU", "EU", "EU", "EU", "NA", "NA", "NA", "NA"],
            "variant": [
                "baseline",
                "baseline",
                "stress",
                "stress",
                "baseline",
                "baseline",
                "stress",
                "stress",
            ],
        }
    )
    set_workspace(tmp_path)

    styled_regions: list[str] = []
    ylabel_regions: list[str] = []

    def _marker_resolver(frame: pd.DataFrame) -> list[object]:
        del frame
        return []

    def _axis_styler(axis, panel_frame: pd.DataFrame) -> None:
        styled_regions.append(str(panel_frame["region"].iloc[0]))
        axis.axhline(0.5, color="black", linewidth=0.5)

    def _ylabel_resolver(panel_frame: pd.DataFrame) -> str:
        ylabel_regions.append(str(panel_frame["region"].iloc[0]))
        return f"Value [{panel_frame['impact_unit'].iloc[0]}]"

    paths = overlay_mod._render_selector_overlay(
        prepared=prepared,
        requested_years=[2030, 2035],
        output_base=tmp_path / "overlay",
        title_parts={
            "family": "Dynamic overlay",
            "selector_scope": None,
            "lcia_method": "pb_lcia",
            "user_facing_override_label": None,
            "prospective_scope": None,
        },
        ylabel="Value",
        dpi=10,
        output_format="png",
        meta_columns=set(),
        marker_resolver=_marker_resolver,
        axis_styler=_axis_styler,
        ylabel_resolver=_ylabel_resolver,
    )

    assert len(paths) == 4
    assert all(path.is_file() for path in paths)
    assert all(path.name.startswith("overlay__Atmospheric_aerosol_loading_AAL") for path in paths)
    assert styled_regions == ["EU", "EU", "NA", "NA"]
    assert ylabel_regions == ["EU", "EU", "NA", "NA"]


def test_render_dynamic_category_overlay_routes_lcia_selector_and_prospective_scopes(
    tmp_path: Path,
) -> None:
    set_workspace(tmp_path)
    prepared = pd.DataFrame(
        {
            "year": [2030, 2035, 2030, 2035],
            "value": [1.0, 1.1, 1.2, 1.3],
            "cc_category": ["safe", "safe", "safe", "safe"],
            "impact": ["AAL", "AAL", "AAL", "AAL"],
            "lcia_method": ["pb_lcia", "pb_lcia", "pb_lcia", "pb_lcia"],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", "SSP1", "SSP2", "SSP2"],
            "r_p": ["EU", "EU", "EU", "EU"],
        }
    )

    paths = overlay_mod.render_dynamic_category_overlay(
        prepared=prepared,
        requested_years=[2030, 2035],
        output_base=tmp_path / "dynamic_overlay",
        family="Dynamic overlay",
        user_facing_override_label=None,
        ylabel="Value",
        dpi=10,
        output_format="png",
        meta_columns=set(),
        marker_resolver=lambda frame: [],
    )

    assert len(paths) == 2
    assert all(path.is_file() for path in paths)
    assert any("prospective_SSP1" in str(path) for path in paths)
    assert any("prospective_SSP2" in str(path) for path in paths)
    assert all("__pb_lcia__" in str(path) for path in paths)
    assert all("Atmospheric_aerosol_loading_AAL" in str(path) for path in paths)


def test_render_selector_overlay_uses_shared_variant_compression_for_reference_and_l2_reuse_year(
    tmp_path: Path,
) -> None:
    set_workspace(tmp_path)
    prepared = pd.DataFrame(
        {
            "year": [2030, 2031, 2030, 2031, 2030, 2031],
            "value": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
            "cc_category": ["safe", "safe", "safe", "safe", "safe", "safe"],
            "impact": ["AAL"] * 6,
            "lcia_method": ["pb_lcia"] * 6,
            "impact_unit": ["ratio"] * 6,
            "reference_year": [2020, 2020, 2025, 2025, 2030, 2030],
            "l2_reuse_year": [2030, 2030, 2035, 2035, 2040, 2040],
            "series_label": [
                "demo ref 2020 reuse 2030",
                "demo ref 2020 reuse 2030",
                "demo ref 2025 reuse 2035",
                "demo ref 2025 reuse 2035",
                "demo ref 2030 reuse 2040",
                "demo ref 2030 reuse 2040",
            ],
        }
    )

    paths = overlay_mod._render_selector_overlay(
        prepared=prepared,
        requested_years=[2030, 2031],
        output_base=tmp_path / "overlay",
        title_parts={
            "family": "Dynamic overlay",
            "selector_scope": None,
            "lcia_method": "pb_lcia",
            "user_facing_override_label": None,
            "prospective_scope": None,
        },
        ylabel="Value",
        dpi=10,
        output_format="png",
        meta_columns=set(),
        marker_resolver=lambda frame: [],
    )

    assert len(paths) == 1
    assert paths[0].is_file()


def test_render_selector_overlay_ignores_transition_scope_metadata_in_trajectory_grouping(
    tmp_path: Path,
) -> None:
    set_workspace(tmp_path)
    prepared = pd.DataFrame(
        {
            "year": [2020, 2021, 2022],
            "value": [1.0, 1.1, 1.2],
            "cc_category": ["safe", "safe", "safe"],
            "impact": ["AAL", "AAL", "AAL"],
            "lcia_method": ["pb_lcia", "pb_lcia", "pb_lcia"],
            "impact_unit": ["ratio", "ratio", "ratio"],
            "cc_model": ["Model A", "Model A", "Model A"],
            "cc_scenario": ["Scenario A", "Scenario A", "Scenario A"],
            ASOCC_SSP_SCENARIO_COLUMN: [pd.NA, pd.NA, "SSP2"],
            "asocc_ssp_start_year": [pd.NA, pd.NA, 2022],
            "series_label": ["demo", "demo", "demo"],
        }
    )

    paths = overlay_mod._render_selector_overlay(
        prepared=prepared,
        requested_years=[2020, 2021, 2022],
        output_base=tmp_path / "overlay",
        title_parts={
            "family": "Dynamic overlay",
            "selector_scope": None,
            "lcia_method": "pb_lcia",
            "user_facing_override_label": None,
            "prospective_scope": None,
        },
        ylabel="Value",
        dpi=10,
        output_format="png",
        meta_columns={"cc_model", "cc_scenario"},
        marker_resolver=lambda frame: [],
    )

    assert len(paths) == 1
    assert paths[0].is_file()
