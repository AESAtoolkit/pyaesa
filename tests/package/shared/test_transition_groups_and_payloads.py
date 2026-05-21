from pathlib import Path

import pandas as pd
import pytest

from pyaesa.shared.figures import deterministic_transition_groups as groups_mod
from pyaesa.shared.figures import transition_panel_payloads as payloads_mod
from pyaesa.shared.figures.multi_year_transitions import TransitionMarker
from pyaesa.shared.figures.variant_selection import VariantCompression
from pyaesa.shared.acc_asr_common.deterministic.downstream import scenarios as scenarios_mod
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_HISTORICAL,
    ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
    ASOCC_TIME_ROUTE_REGRESSION,
)
from pyaesa.shared.tabular import wide_tables as wide_tables_mod


def test_group_files_by_base_normalizes_parents_and_reuse_metadata(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    regression_path = (
        root
        / "results"
        / "level_2"
        / "l2_in_l1"
        / "regression_proj"
        / "native_SSP2__C1__ssp2__gwp100_lcia__min_cc.csv"
    )
    reuse_path = root / "historical_reuse" / "family" / "base.csv"
    regression_path.parent.mkdir(parents=True)
    reuse_path.parent.mkdir(parents=True)
    regression_path.write_text("demo", encoding="utf-8")
    reuse_path.write_text("demo", encoding="utf-8")

    grouped = groups_mod.group_files_by_base(
        root=root,
        paths=[regression_path, reuse_path],
        share_transition_meta={
            "native_SSP2": {
                "base_stem": "native",
                ASOCC_SSP_SCENARIO_COLUMN: "SSP2",
                "marker_label": "transition",
                "marker_color": "#abcdef",
            },
            "base": {},
        },
        l1_l2_methods_by_path={
            regression_path: "UT(GVA)",
            reuse_path: "base",
        },
    )

    assert grouped[0][0] == Path("family")
    assert grouped[0][1] == "base"
    assert grouped[0][2][0].marker_label == "prospective transition"
    assert grouped[1][0] == Path("l2_in_l1")
    assert grouped[1][1] == "native"
    assert grouped[1][2][0].marker_color == "#abcdef"


def test_long_and_combined_transition_frames_expand_reuse_history(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    root.mkdir()
    historical_path = root / "historical.csv"
    prospective_path = root / "historical_reuse" / "prospective.csv"
    prospective_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL],
            "2030": [1.0],
            "2031": [2.0],
        }
    ).to_csv(historical_path, index=False)
    pd.DataFrame(
        {
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL_REUSE],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"],
            "l2_reuse_year": [2035],
            "2030": [3.0],
            "2031": [4.0],
        }
    ).to_csv(prospective_path, index=False)

    grouped_files = [
        groups_mod.GroupedTransitionFile(
            path=historical_path,
            l1_l2_method="base",
            base_stem="base",
            marker_label="transition",
            marker_color="#7d7d7d",
        ),
        groups_mod.GroupedTransitionFile(
            path=prospective_path,
            l1_l2_method="base",
            base_stem="base",
            marker_label="transition",
            marker_color="#7d7d7d",
        ),
    ]

    family = groups_mod.long_frame_from_group(
        grouped_files=grouped_files,
        requested_years=[2030, 2031],
    )

    assert sorted(family["year"].tolist()) == ["2030", "2030", "2031", "2031"]
    assert set(family["l2_reuse_year"].tolist()) == {2035}
    assert set(family[ASOCC_SSP_SCENARIO_COLUMN].dropna().tolist()) == {"SSP2"}
    assert set(family[ASOCC_TIME_ROUTE_PUBLIC_COLUMN].dropna()) == {
        ASOCC_TIME_ROUTE_HISTORICAL,
        ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
    }

    combined = groups_mod.combined_long_frame_from_groups(
        groups=[(Path("."), "base", grouped_files)],
        requested_years=[2030, 2031],
        fu_code="L1.demo",
    )
    assert set(combined.columns) >= {"combined_group_label", "fu_code", "year", "value"}
    assert set(combined["combined_group_label"]) == {"base"}
    assert set(combined["fu_code"]) == {"L1.demo"}
    assert (
        groups_mod.has_multiple_grouped_transition_groups([(Path("."), "base", grouped_files)])
        is False
    )
    assert (
        groups_mod.has_multiple_grouped_transition_groups(
            [
                (Path("."), "base", grouped_files),
                (
                    Path("other"),
                    "other",
                    [
                        groups_mod.GroupedTransitionFile(
                            path=prospective_path,
                            l1_l2_method="other",
                            base_stem="other",
                            marker_label="transition",
                            marker_color="#7d7d7d",
                        )
                    ],
                ),
            ]
        )
        is True
    )

    assert (
        groups_mod.title_stem(relative_parent=Path("results/family"), base_stem="stem")
        == "results / family / stem"
    )
    assert (
        groups_mod.title_stem(
            relative_parent=Path("."),
            base_stem="stem_gwp100",
            lcia_method="gwp100",
        )
        == "stem"
    )
    assert (
        groups_mod.title_stem(
            relative_parent=Path("results/family"),
            base_stem="stem",
            lcia_method="gwp100",
        )
        == "results / family / stem"
    )
    assert groups_mod.normalize_companion_relative_parent(
        Path("historical_reuse/2030/results/family")
    ) == Path("2030/family")
    assert groups_mod.normalize_companion_relative_parent(Path("results/family")) == Path("family")
    assert groups_mod.normalize_companion_relative_parent(Path("family")) == Path("family")
    assert groups_mod.normalize_companion_relative_parent(
        Path("results/level_2/l2_in_l1/regression_proj")
    ) == Path("l2_in_l1")
    assert groups_mod.normalize_companion_relative_parent(
        Path("results/level_1/utility_propagation_contrib/family")
    ) == Path("utility_propagation_contrib/family")
    assert (
        groups_mod.origin_share_stem_from_output_stem(
            output_stem="native_SSP2__C1__ssp2__gwp100_lcia__min_cc",
            share_transition_meta={"native_SSP2": {ASOCC_SSP_SCENARIO_COLUMN: "SSP2"}},
        )
        == "native_SSP2"
    )
    assert groups_mod._wide_to_long(  # noqa: SLF001
        frame=pd.DataFrame({ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"], "2030": [1.0]}),
        requested_years=[2030],
    )[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2"]
    assert groups_mod._wide_to_long(  # noqa: SLF001
        frame=pd.DataFrame({ASOCC_SSP_SCENARIO_COLUMN: [None], "2030": [1.0]}),
        requested_years=[2030],
    )[ASOCC_SSP_SCENARIO_COLUMN].tolist() == [None]
    assert groups_mod._wide_to_long(  # noqa: SLF001
        frame=pd.DataFrame({ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"], "2030": [1.0]}),
        requested_years=[2030],
    )[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2"]
    assert (
        groups_mod.origin_share_stem_from_output_stem(
            output_stem=" ",
            share_transition_meta={"base": {}},
        )
        is None
    )
    assert (
        groups_mod.origin_share_stem_from_output_stem(
            output_stem="base__gwp100_lcia__max_cc",
            share_transition_meta={"base": {}},
        )
        == "base"
    )
    assert (
        groups_mod.origin_share_stem_from_output_stem(
            output_stem="EG_Pop__AR_E_PBA__gwp100_lcia__ssp2__min_cc",
            share_transition_meta={"EG(Pop)_AR(E^{PBA})__gwp100_lcia__ssp2": {}},
        )
        == "EG(Pop)_AR(E^{PBA})__gwp100_lcia__ssp2"
    )
    assert scenarios_mod.share_transition_payload_for_output_stem(
        output_stem="EG_Pop__AR_E_PBA__gwp100_lcia__ssp2__min_cc",
        share_transition_meta={
            "EG(Pop)_AR(E^{PBA})__gwp100_lcia__ssp2": {ASOCC_SSP_SCENARIO_COLUMN: "SSP2"}
        },
    ) == {ASOCC_SSP_SCENARIO_COLUMN: "SSP2"}


def test_transition_group_stem_resolution_contracts(
    tmp_path: Path,
) -> None:
    historical_path = tmp_path / "historical_reuse" / "family" / "base.csv"
    historical_path.parent.mkdir(parents=True)
    pd.DataFrame({"2030": [1.0], "2031": [2.0]}).to_csv(historical_path, index=False)

    grouped_files = [
        groups_mod.GroupedTransitionFile(
            path=historical_path,
            l1_l2_method="base",
            base_stem="base",
            marker_label="transition",
            marker_color="#7d7d7d",
        )
    ]

    family = groups_mod.long_frame_from_group(
        grouped_files=grouped_files,
        requested_years=[2030, 2031],
    )
    assert family["value"].tolist() == [1.0, 2.0]
    assert groups_mod._share_transition_payload(  # noqa: SLF001
        output_path=Path("unknown_output.csv"),
        share_transition_meta={"UT(GVA)": {"marker_label": "fallback"}},
        l1_l2_method="UT(GVA)",
    ) == {"marker_label": "fallback"}
    assert (
        groups_mod._candidate_output_share_stem(  # noqa: SLF001
            "base__gwp100_lcia__max_cc"
        )
        == "base__gwp100_lcia"
    )
    assert (
        groups_mod.origin_share_stem_from_output_stem(
            output_stem="base__gwp100_lcia__max_cc",
            share_transition_meta={"base__gwp100_lcia": {}},
        )
        == "base__gwp100_lcia"
    )
    assert (
        groups_mod.origin_share_stem_from_output_stem(
            output_stem="EG_Pop__AR_E_PBA__gwp100_lcia__min_cc",
            share_transition_meta={"EG(Pop)_AR(E^{PBA})__gwp100_lcia": {}},
        )
        == "EG(Pop)_AR(E^{PBA})__gwp100_lcia"
    )
    assert (
        groups_mod._raw_candidate_output_share_stem(  # noqa: SLF001
            "plain_stem"
        )
        == "plain_stem"
    )
    assert (
        groups_mod._raw_candidate_output_share_stem(  # noqa: SLF001
            "base__gwp100_lcia__min_cc__ssp2__max_cc"
        )
        == "base__gwp100_lcia"
    )
    assert (
        groups_mod.grouped_transition_method_identity(
            relative_parent=Path("family"),
            base_stem="base",
            grouped_files=grouped_files,
        )
        == "base"
    )


def test_prepare_transition_frame_and_panel_payload_contracts() -> None:
    frame = pd.DataFrame(
        {
            "year": [2030, 2031, 2032],
            "value": [1.0, 2.0, 3.0],
            ASOCC_SSP_SCENARIO_COLUMN: [None, "SSP2", "SSP2"],
            "impact": ["climate", "climate", "climate"],
            "level": ["level_1", "level_1", "level_1"],
            "fu_code": ["L1.demo", "L1.demo", "L1.demo"],
            "series_label": ["Visible series", "Visible series", "Visible series"],
        }
    )

    prepared = payloads_mod.prepare_transition_frame(
        frame=frame,
        requested_years=[2030, 2031, 2032],
        marker_label="switch",
        marker_color="#445566",
    )
    assert len(prepared) == 3
    assert prepared[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2", "SSP2", "SSP2"]
    assert set(prepared["__transition_marker_year"].dropna().tolist()) == {2031}

    grouped_panels = payloads_mod.panel_groups(prepared, panel_column="impact")
    assert len(grouped_panels) == 1
    assert grouped_panels[0][0] == "climate"
    assert payloads_mod.panel_groups(prepared, panel_column=None)[0][0] == "value"

    payloads = payloads_mod.series_payloads(
        grouped_panels[0][1],
        requested_years=[2030, 2031, 2032],
        panel_column="impact",
        skip_columns={"level", "fu_code"},
    )
    assert payloads == {
        ("__series_1",): (
            "Visible series",
            [2030, 2031, 2032],
            [1.0, 2.0, 3.0],
            "",
        )
    }
    partial_payloads = payloads_mod.series_payloads(
        grouped_panels[0][1].loc[grouped_panels[0][1]["year"].ge(2031)],
        requested_years=[2030, 2031, 2032],
        panel_column="impact",
        skip_columns={"level", "fu_code"},
    )
    assert partial_payloads[("__series_1",)][1:3] == ([2031, 2032], [2.0, 3.0])

    transition_years = payloads_mod.series_transition_years(
        grouped_panels[0][1],
        panel_column="impact",
        skip_columns={"level", "fu_code", "series_label"},
    )
    assert transition_years == {("__series_1",): 2031}
    assert payloads_mod.panel_markers(prepared) == [
        TransitionMarker(year=2031, label="switch", color="#445566")
    ]

    line_specs = payloads_mod.series_line_specs(
        pd.DataFrame(
            {
                "year": [2030, 2030],
                "value": [1.0, 2.0],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP2", "SSP2"],
                "series_label": ["A", "B"],
                "reference_year": [2020, 2030],
            }
        ),
        panel_column=None,
        compressions=(
            VariantCompression(
                column="reference_year",
                kept_values=(2020, 2030),
                filtered=False,
                base_key=(),
            ),
        ),
        skip_columns={"series_label"},
    )
    assert line_specs[("2020",)].line_style == "solid"
    assert line_specs[("2030",)].line_style == "dotted"
    assert line_specs[("2030",)].prospective_only is False

    no_scenario_prepared = payloads_mod.prepare_transition_frame(
        frame=pd.DataFrame(
            {
                "year": [2030, 2031],
                "value": [5.0, 7.0],
                "impact": ["climate", "climate"],
                "series_label": ["Visible series", "Visible series"],
            }
        ),
        requested_years=[2030, 2031],
        marker_label="switch",
        marker_color="#445566",
    )
    assert ASOCC_SSP_SCENARIO_COLUMN not in no_scenario_prepared.columns
    assert AR6_CC_SSP_SCENARIO_COLUMN not in no_scenario_prepared.columns

    route_prepared = payloads_mod.prepare_transition_frame(
        frame=pd.DataFrame(
            {
                "year": [2020, 2030],
                "value": [1.0, 2.0],
                "series_label": ["Visible series", "Visible series"],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [
                    ASOCC_TIME_ROUTE_HISTORICAL,
                    ASOCC_TIME_ROUTE_REGRESSION,
                ],
            }
        ),
        requested_years=[2020, 2030],
        marker_label="switch",
        marker_color="#445566",
    )
    assert set(route_prepared["__transition_marker_year"].dropna().tolist()) == {2030}

    ssp_prepared = payloads_mod.prepare_transition_frame(
        frame=pd.DataFrame(
            {
                "year": [2005, 2006, 2030],
                "value": [1.0, 2.0, 3.0],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None, "SSP2"],
                "impact": ["climate", "climate", "climate"],
                "series_label": ["Visible series", "Visible series", "Visible series"],
            }
        ),
        requested_years=[2005, 2006, 2030],
        marker_label="switch",
        marker_color="#445566",
    )
    assert ssp_prepared[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2", "SSP2", "SSP2"]
    assert set(ssp_prepared["__transition_marker_year"].dropna().tolist()) == {2030}
    assert payloads_mod.series_payloads(
        ssp_prepared,
        requested_years=[2005, 2006, 2030],
        panel_column=None,
        skip_columns={"impact", "series_label"},
    ) == {
        ("__series_1",): (
            "Visible series",
            [2005, 2006, 2030],
            [1.0, 2.0, 3.0],
            "One-step",
        )
    }

    dynamic_model_prepared = payloads_mod.prepare_transition_frame(
        frame=pd.DataFrame(
            {
                "year": [2020, 2021, 2022],
                "value": [1.0, 2.0, 3.0],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None, "SSP2"],
                "cc_model": [None, None, "Model A"],
                "cc_scenario": [None, None, "Scenario A"],
                "series_label": ["Visible series", "Visible series", "Visible series"],
            }
        ),
        requested_years=[2020, 2021, 2022],
        marker_label="switch",
        marker_color="#445566",
        transition_grouping_skip_columns={"cc_model", "cc_scenario"},
    )
    assert dynamic_model_prepared[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2", "SSP2", "SSP2"]
    assert dynamic_model_prepared["cc_model"].tolist() == ["Model A", "Model A", "Model A"]
    assert dynamic_model_prepared["cc_scenario"].tolist() == [
        "Scenario A",
        "Scenario A",
        "Scenario A",
    ]


def test_melt_requested_year_value_rows_drops_nan_companion_year_rows() -> None:
    frame = pd.DataFrame(
        {
            "l1_l2_method": ["demo", "demo"],
            "2025": [1.0, pd.NA],
            "2026": [pd.NA, 2.0],
        }
    )
    melted = wide_tables_mod.melt_requested_year_value_rows(
        frame,
        requested_years=[2025, 2026],
    )
    assert melted["year"].tolist() == ["2025", "2026"]
    assert melted["value"].tolist() == [1.0, 2.0]


def test_transition_group_and_payload_failures() -> None:
    with pytest.raises(ValueError):
        payloads_mod.series_payloads(
            pd.DataFrame(
                {
                    "year": [2030],
                    "value": [1.0],
                    "level": ["level_1"],
                    "fu_code": ["L1.demo"],
                }
            ),
            requested_years=[2030],
            panel_column=None,
        )
