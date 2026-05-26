from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from pyaesa.shared.figures import lcia_metadata as metadata_mod
from pyaesa.shared.lcia import path_tokens as lcia_path_tokens
from pyaesa.shared.figures import lcia_scope as scope_mod
from pyaesa.shared.figures import method_identity as method_mod
from pyaesa.shared.figures import selector_slices as slices_mod
from pyaesa.shared.figures import series_labels as labels_mod
from pyaesa.shared.figures import contracts as figure_contracts
from pyaesa.shared.figures import title_contract as title_mod
from pyaesa.shared.selectors import path_tokens as token_mod
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
)


def test_title_contracts_cover_selector_and_prospective_scope() -> None:
    frame = pd.DataFrame(
        {
            "r_p": ["FR", "FR", "DE"],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", None, "SSP2"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    assert title_mod.single_visible_value(frame=frame.iloc[[0, 1]], column="r_p") == "FR"
    assert title_mod.single_visible_value(frame=frame, column="r_p") is None
    assert title_mod.single_visible_value(frame=frame, column="missing") is None
    assert title_mod.prospective_scenario_values(frame.iloc[[0, 1]]) == ["SSP1"]
    assert title_mod.resolve_prospective_scope(frame.iloc[[0, 1]]) == "Prospective: SSP1"
    assert title_mod.resolve_prospective_scope(pd.DataFrame({"value": [1.0]})) is None
    with pytest.raises(ValueError):
        title_mod.resolve_prospective_scope(frame)

    prospective_slices = list(title_mod.prospective_scope_slices(frame))
    assert [(token, title) for token, title, _ in prospective_slices] == [
        ("prospective_SSP1", "Prospective: SSP1"),
        ("prospective_SSP2", "Prospective: SSP2"),
    ]
    assert prospective_slices[0][2].iloc[0].to_dict() == {
        "r_p": "FR",
        ASOCC_SSP_SCENARIO_COLUMN: "SSP1",
        "value": 1.0,
    }
    assert prospective_slices[0][2].iloc[1]["r_p"] == "FR"
    assert pd.isna(prospective_slices[0][2].iloc[1][ASOCC_SSP_SCENARIO_COLUMN])
    assert float(prospective_slices[0][2].iloc[1]["value"]) == 2.0
    assert list(
        title_mod.prospective_scope_slices(
            pd.DataFrame(
                [
                    {ASOCC_SSP_SCENARIO_COLUMN: "SSP 1", "value": 1.0},
                    {ASOCC_SSP_SCENARIO_COLUMN: "Net Zero", "value": 2.0},
                    {ASOCC_SSP_SCENARIO_COLUMN: None, "value": 3.0},
                ]
            )
        )
    )[0][0:2] == ("prospective_Net_Zero", "Prospective: Net Zero")
    no_scope_slices = list(title_mod.prospective_scope_slices(pd.DataFrame({"value": [1.0]})))
    assert len(no_scope_slices) == 1
    assert no_scope_slices[0][0] == "all"
    assert no_scope_slices[0][1] is None
    assert no_scope_slices[0][2].equals(pd.DataFrame({"value": [1.0]}))
    assert title_mod.prospective_scenario_values(pd.DataFrame({"value": [1.0]})) == []
    assert figure_contracts.deterministic_prospective_values(frame.iloc[[0, 1]]) == ["SSP1"]
    assert figure_contracts.deterministic_prospective_series(
        pd.DataFrame({"scenario": ["SSP1"], AR6_CC_SSP_SCENARIO_COLUMN: [1]})
    ).tolist() == ["SSP1"]
    assert figure_contracts.deterministic_prospective_series(
        pd.DataFrame({"value": [1.0]})
    ).tolist() == [None]
    assert figure_contracts.deterministic_prospective_series(pd.DataFrame()).empty

    with pytest.raises(ValueError):
        figure_contracts.deterministic_prospective_series(
            pd.DataFrame(
                {ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"], AR6_CC_SSP_SCENARIO_COLUMN: ["SSP2"]}
            )
        )

    selector_frame = pd.DataFrame({"r_p": ["FR", "DE"], "s_p": ["A", "A"]})
    assert (
        title_mod.format_selector_axis(
            frame=selector_frame.iloc[[0]],
            column="r_p",
            reference_frame=selector_frame,
        )
        == "r_p=FR"
    )
    assert (
        title_mod.format_selector_axis(
            frame=selector_frame,
            column="r_p",
            reference_frame=selector_frame,
        )
        == "all r_p"
    )
    assert (
        title_mod.format_selector_axis(
            frame=selector_frame,
            column="r_p",
            reference_frame=pd.DataFrame({"r_p": ["FR"]}),
        )
        == "r_p=DE + FR"
    )
    assert title_mod.format_selector_axis(frame=selector_frame, column="missing") is None
    assert title_mod.format_selector_scope(frame=selector_frame.iloc[[0]]) == "r_p=FR | s_p=A"
    assert title_mod.format_selector_scope(frame=selector_frame, selector_columns=()) is None
    fu_selector_frame = pd.DataFrame(
        {
            "fu_code": ["L2.c.b"],
            "r_p": [pd.NA],
            "s_p": ["Electricity"],
            "r_c": ["FR"],
            "r_f": [pd.NA],
            "value": [1.0],
        }
    )
    assert title_mod.format_selector_scope(frame=fu_selector_frame) == "s_p=Electricity | r_c=FR"
    assert figure_contracts.figure_selector_columns(fu_selector_frame) == ("s_p", "r_c")
    assert figure_contracts.resolved_selector_columns(selector_frame) == ("r_p", "s_p")
    assert figure_contracts.resolved_selector_columns(
        pd.DataFrame({"r_p": [None], "s_p": ["A"]}),
    ) == ("s_p",)
    assert figure_contracts.resolved_selector_columns(
        pd.DataFrame({"r_p": [None]}),
        require_non_null=False,
    ) == ("r_p",)


def test_title_contract_builders_and_request_contracts_cover_remaining_branches() -> None:
    scope_request = title_mod.selector_scope_request_from_filters(
        filters={"r_p": ["FR", "FR", "DE"], "s_p": None}
    )
    assert title_mod.selector_scope_request_from_filters(filters={}) is None
    assert title_mod.selector_scope_request_from_filters(
        filters={"r_p": ["FR"], "r_c": []}
    ) == title_mod.SelectorScopeRequest(axes=(("r_p", ("FR",)),))
    assert scope_request == title_mod.SelectorScopeRequest(
        axes=(("r_p", ("DE", "FR")), ("s_p", None))
    )
    selector_request = title_mod.selector_scope_request_from_selector_values(
        selector_values={"r_p": "FR", "r_c": ["C1", "C2", "C1"], "r_f": None}
    )
    assert selector_request == title_mod.SelectorScopeRequest(
        axes=(("r_p", ("FR",)), ("r_c", ("C1", "C2")), ("r_f", None))
    )
    assert title_mod.selector_scope_request_from_base_allocate_args(
        base_allocate_args={"r_p": "FR", "s_p": ["D"], "r_f": None}
    ) == title_mod.SelectorScopeRequest(axes=(("r_p", ("FR",)), ("s_p", ("D",)), ("r_f", None)))
    assert title_mod.selector_scope_request_from_selector_values(
        selector_values={
            "fu_code": "L2.c.b",
            "r_p": None,
            "s_p": "Electricity",
            "r_c": "FR",
            "r_f": None,
        }
    ) == title_mod.SelectorScopeRequest(axes=(("s_p", ("Electricity",)), ("r_c", ("FR",))))
    assert (
        title_mod.format_selector_scope_request(selector_scope_request=selector_request)
        == "r_p=FR | r_c=C1 + C2 | all r_f"
    )
    assert (
        title_mod.selector_scope_request_from_selector_values(
            selector_values={"r_p": " ", "r_c": []}
        )
        is None
    )
    assert title_mod.selector_scope_request_from_selector_values(
        selector_values={"r_p": 1}
    ) == title_mod.SelectorScopeRequest(axes=(("r_p", ("1",)),))
    assert (
        title_mod.selector_scope_request_from_selector_values(selector_values={"r_p": float("nan")})
        is None
    )
    empty_scope = pd.DataFrame({"value": [1.0]})
    assert (
        title_mod.resolve_selector_scope(
            frame=empty_scope,
            selector_scope_request=selector_request,
        )
        == "r_p=FR | r_c=C1 + C2 | all r_f"
    )
    assert title_mod.join_title_blocks(" Family ", None, " 2030 ") == "Family | 2030"
    assert title_mod.clean_panel_title(panel_title=None) is None
    assert title_mod.clean_panel_title(panel_title=" value ") is None
    assert title_mod.clean_panel_title(panel_title="  panel A  ") == "panel A"
    assert (
        title_mod.build_figure_title(
            family="ASR",
            selector_scope="r_p=FR",
            lcia_method="pb_lcia",
            user_facing_override_label="eq",
            prospective_scope="Prospective: SSP1",
            year=2030,
        )
        == "ASR | r_p=FR | pb_lcia | eq | Prospective: SSP1 | 2030"
    )
    assert (
        title_mod.build_resolved_figure_title(
            title_parts={
                "family": "ASR",
                "selector_scope": "r_p=FR",
                "lcia_method": "pb_lcia",
                "user_facing_override_label": "eq",
                "prospective_scope": None,
            },
            year=2030,
            panel_title="Panel A",
            panel_count=1,
        )
        == "ASR | r_p=FR | pb_lcia | eq | 2030 | Panel A"
    )
    assert (
        title_mod.build_resolved_figure_title(
            title_parts={
                "family": "ASR",
                "selector_scope": None,
                "lcia_method": None,
                "user_facing_override_label": None,
                "prospective_scope": None,
            },
            panel_title="Panel A",
            panel_count=2,
        )
        == "ASR"
    )
    assert title_mod.resolve_panel_title(panel_title="Panel A", panel_count=1) is None
    assert title_mod.resolve_panel_title(panel_title=" value ", panel_count=2) is None
    assert title_mod.resolve_panel_title(panel_title="Panel A", panel_count=2) == "Panel A"
    assert (
        title_mod.selector_scope_request_token(
            selector_scope_request=title_mod.SelectorScopeRequest(axes=(("r_p", ()),))
        )
        == "rp_all"
    )
    assert title_mod.selector_scope_request_token(selector_scope_request=None) == "all_selectors"
    assert (
        title_mod.selector_scope_request_token(
            selector_scope_request=title_mod.SelectorScopeRequest(axes=(("", None),))
        )
        == "missing_all"
    )
    assert token_mod.selector_scope_request_axes_token(()) == "all_selectors"
    assert token_mod.selector_axis_values_token(()) == "all"
    assert token_mod.selector_axis_values_token(
        [f"very long selector value {index}" for index in range(12)],
        max_segment_len=20,
    ).startswith("n12_")
    assert (
        token_mod.selector_scope_token_from_values(
            {"r_p": "FR", "s_p": "Electricity"},
            selector_columns=("r_p", "s_p", "r_c"),
        )
        == "rp_FR__sp_Electricity"
    )
    assert (
        token_mod.selector_scope_token_from_frame(
            group_frame=pd.DataFrame({"value": [1.0]}),
            selector_columns=("r_p",),
        )
        == "all_selectors"
    )
    assert (
        token_mod.selector_scope_token_from_frame(
            group_frame=pd.DataFrame({"s_p": ["A", "B"]}),
            selector_columns=("s_p",),
        )
        == "sp_A+B"
    )
    assert (
        token_mod.selector_scope_token_from_frame(
            group_frame=pd.DataFrame({"s_p": ["Manufacture of basic metals"]}),
            selector_columns=("s_p",),
            reference_frame=pd.DataFrame(
                {
                    "s_p": [
                        "Manufacture of basic metals",
                        "Manufacture of basic plastics",
                    ]
                }
            ),
        )
        == "sp_Manufacture_of_b"
    )
    assert (
        token_mod.build_selector_filter_segment(
            key="s_p",
            values=["a very long selector value"],
            max_segment_len=4,
        )
        == "s_p"
    )
    assert (
        title_mod.uncertainty_family_label(value_column="acc_value", transition_policy="x") == "aCC"
    )
    assert (
        title_mod.uncertainty_family_label(value_column="asr_value", transition_policy="x") == "ASR"
    )
    assert (
        title_mod.uncertainty_family_label(value_column="value", transition_policy="asocc")
        == "aSoCC"
    )
    assert (
        title_mod.uncertainty_family_label(value_column="value", transition_policy="other")
        == "IO-LCA"
    )


def test_selector_slices_cover_grouping_and_matching_masks() -> None:
    frame = pd.DataFrame(
        {
            "r_p": ["FR", "FR", pd.NA],
            "s_p": ["A", "B", "A"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    scope_request = title_mod.SelectorScopeRequest(axes=(("r_p", None),))
    slices = list(slices_mod.selector_slices(frame, selector_scope_request=scope_request))
    assert [(token, scope) for token, scope, _ in slices] == [
        ("rp_FR__sp_A", "r_p=FR | s_p=A"),
        ("rp_FR__sp_B", "r_p=FR | s_p=B"),
        ("rp_all__sp_A", "s_p=A"),
    ]
    collision_frame = pd.DataFrame(
        {
            "s_p": ["Manufacture of basic metals", "Manufacture of basic plastics"],
            "value": [1.0, 2.0],
        }
    )
    assert [token for token, _scope, _frame in slices_mod.selector_slices(collision_frame)] == [
        "sp_Manufacture_of_b",
        "sp_Manufacture_of_b_2",
    ]
    fu_frame = pd.DataFrame(
        {
            "fu_code": ["L2.c.b"],
            "r_p": [pd.NA],
            "s_p": ["Electricity"],
            "r_c": ["FR"],
            "r_f": [pd.NA],
            "value": [1.0],
        }
    )
    assert [(token, scope) for token, scope, _frame in slices_mod.selector_slices(fu_frame)] == [
        ("sp_Electricity__rc_FR", "s_p=Electricity | r_c=FR")
    ]
    unsliced = list(slices_mod.selector_slices(pd.DataFrame({"value": [1.0]})))
    assert len(unsliced) == 1
    assert unsliced[0][0] == "all"
    assert unsliced[0][1] == ""
    assert unsliced[0][2].equals(pd.DataFrame({"value": [1.0]}))
    empty_slices = list(
        slices_mod.selector_slices(
            pd.DataFrame(columns=["r_p", "value"]),
            selector_scope_request=scope_request,
        )
    )
    assert len(empty_slices) == 1
    assert empty_slices[0][0] == "all"
    assert empty_slices[0][1] == "all r_p"
    assert empty_slices[0][2].empty

    mask = slices_mod.matching_selector_mask(
        frame=frame,
        reference_row=pd.Series({"r_p": pd.NA, "s_p": "A"}),
    )
    assert mask.tolist() == [False, False, True]


def test_method_identity_and_series_labels_cover_visible_and_failure_paths() -> None:
    frame = pd.DataFrame(
        {
            "l1_l2_method": ["eq|m1", "eq|m1"],
            "l2_method": ["m1", "m1"],
            "l1_method": ["eq", "eq"],
            "reference_year": [2030, 2035],
        }
    )
    assert method_mod.display_pair("reference_year", 2030) == "ref_year=2030"
    assert method_mod.display_pair("l2_method", "m1") == "m1"
    assert method_mod.display_pair("region", None) is None
    assert method_mod.simplified_method_identity_columns(
        frame,
        columns=("l1_l2_method", "l2_method", "l1_method", "reference_year"),
    ) == ["l1_l2_method", "reference_year"]
    assert method_mod.simplified_method_identity_columns(
        pd.DataFrame({"region": ["EU"]}),
        columns=("l2_method", "l1_method", "region"),
    ) == ["l2_method", "l1_method", "region"]
    assert method_mod.visible_method_identity(frame) == "eq|m1"
    assert method_mod.visible_method_identity(pd.DataFrame({"l2_method": [None, ""]})) is None
    unsliced = list(method_mod.method_scope_slices(pd.DataFrame({"value": [1.0, 2.0]})))
    assert len(unsliced) == 1
    assert unsliced[0][0] is None
    assert unsliced[0][1].equals(pd.DataFrame({"value": [1.0, 2.0]}))
    assert method_mod.resolve_figure_display_label(frame=frame) == "eq|m1"
    assert (
        method_mod.resolve_figure_display_label(
            frame=frame,
            user_facing_override_label="Display label",
        )
        == "Display label"
    )
    assert (
        method_mod.resolve_figure_display_label(frame=frame, user_facing_override_label="   ")
        == "eq|m1"
    )

    labeled = labels_mod.with_series_label_column(
        frame,
        label_columns=("l1_l2_method", "reference_year", "l2_method"),
        context="series test",
    )
    assert labeled["series_label"].tolist() == [
        "eq|m1, ref_year=2030",
        "eq|m1, ref_year=2035",
    ]
    assert (
        labels_mod.resolve_series_label(
            pd.Series({"reference_year": 2030, "value": 1.0}),
            label_columns=("reference_year",),
            context="series test",
        )
        == "ref_year=2030"
    )
    assert (
        labels_mod.resolve_series_label(
            pd.Series({"l1_l2_method": "eq|m1", "reference_year": 2030, "l2_reuse_year": 2040}),
            label_columns=("l1_l2_method", "reference_year", "l2_reuse_year"),
            context="series test",
        )
        == "eq|m1, ref_year=2030, l2_reuse_year=2040"
    )
    assert (
        labels_mod.resolve_series_label(
            pd.Series({"l1_l2_method": "eq|m1", "l2_reuse_year": 2040}),
            label_columns=("l1_l2_method", "l2_reuse_year"),
            display_aliases={"l2_reuse_year": "l2_reuse_year"},
            context="series test",
        )
        == "eq|m1, l2_reuse_year=2040"
    )
    scoped = list(
        method_mod.method_scope_slices(
            pd.DataFrame(
                {
                    "l1_l2_method": ["eq|m1", "eq|m2"],
                    "l2_method": ["m1", "m2"],
                    "value": [1.0, 2.0],
                }
            )
        )
    )
    assert [label for label, _frame in scoped] == ["eq|m1", "eq|m2"]
    empty_labeled = labels_mod.with_series_label_column(
        pd.DataFrame(),
        label_columns=("reference_year",),
        context="series test",
    )
    assert list(empty_labeled.columns) == ["series_label"]
    assert empty_labeled.empty
    with pytest.raises(ValueError):
        labels_mod.resolve_series_label(
            pd.Series({"reference_year": 2030}),
            label_columns=tuple(),
            context="series test",
        )
    with pytest.raises(ValueError):
        labels_mod.resolve_series_label(
            pd.Series({"value": 1.0}),
            label_columns=("missing_column",),
            context="series test",
        )
    with pytest.raises(ValueError):
        labels_mod.resolve_series_label(
            pd.Series({"region": None}),
            label_columns=("region",),
            context="series test",
        )
    with pytest.raises(ValueError):
        labels_mod.require_series_label(pd.Series({"value": 1.0}), context="series test")
    with pytest.raises(ValueError):
        labels_mod.require_series_label(pd.Series({"series_label": "  "}), context="series test")


def test_lcia_metadata_and_scopes_cover_real_metadata_and_errors(project_repo: Path) -> None:
    del project_repo

    methods = lcia_path_tokens.known_lcia_methods()
    assert "pb_lcia" in methods
    pb_metadata = metadata_mod.load_lcia_metadata("pb_lcia")
    assert pb_metadata.family == "pb_lcia"
    assert pb_metadata.schema_kind == "planetary boundary"
    assert pb_metadata.n_impacts >= 1
    gwp100_metadata = metadata_mod.load_lcia_metadata("gwp100_lcia")
    assert (
        metadata_mod.resolve_impact_title(
            lcia_method="gwp100_lcia",
            impact="climate_parent",
        )
        == gwp100_metadata.labels["GWP_100"]
    )

    first_impact = pb_metadata.impacts[0]
    second_impact = pb_metadata.impacts[1]
    impact_title = metadata_mod.resolve_impact_title(lcia_method="pb_lcia", impact=first_impact)
    assert impact_title == pb_metadata.labels[first_impact]
    with pytest.raises(ValueError):
        metadata_mod.resolve_impact_title(lcia_method="pb_lcia", impact="not_a_real_impact")
    assert lcia_path_tokens.infer_lcia_method_from_path(Path("demo_pb_lcia.csv")) == "pb_lcia"
    assert lcia_path_tokens.infer_lcia_method_from_path(Path("demo.csv")) is None

    assert (
        metadata_mod.format_impact_label(
            schema_kind="planetary boundary",
            row=SimpleNamespace(
                impact="AAL",
                planetary_boundary="Climate change",
                control_variable="Atmospheric CO2",
            ),
        )
        == "Climate change: Atmospheric CO2 (AAL)"
    )
    with pytest.raises(ValueError):
        metadata_mod.format_impact_label(
            schema_kind="planetary boundary",
            row=SimpleNamespace(
                impact="AAL",
                planetary_boundary=" ",
                control_variable=None,
            ),
        )
    assert (
        metadata_mod.format_impact_label(
            schema_kind="standard",
            row=SimpleNamespace(
                impact="co2",
                impact_full_name_normalized="Carbon dioxide",
            ),
        )
        == "Carbon dioxide (co2)"
    )

    inferred_frame = pd.DataFrame({"impact": [first_impact], "value": [1.0]})
    inferred_frame.attrs["source_path"] = str(Path("demo_pb_lcia.csv"))
    ensured = metadata_mod.ensure_frame_lcia_method_metadata(inferred_frame)
    assert ensured["lcia_method"].tolist() == ["pb_lcia"]
    assert metadata_mod.resolve_frame_impact_title(ensured) == impact_title
    assert (
        metadata_mod.resolve_frame_impact_unit(pd.DataFrame({"impact_unit": ["kg", "kg", None]}))
        == "kg"
    )
    assert (
        metadata_mod.resolve_frame_impact_unit(
            pd.DataFrame({"impact": [first_impact], "lcia_method": ["pb_lcia"]})
        )
        == pb_metadata.units[first_impact]
    )
    assert metadata_mod.resolve_frame_impact_unit(pd.DataFrame({"value": [1.0]})) is None
    assert metadata_mod.resolve_frame_impact_unit(pd.DataFrame()) is None
    assert metadata_mod.resolve_frame_impact_title(pd.DataFrame()) is None
    passthrough = pd.DataFrame({"value": [1.0]})
    assert metadata_mod.ensure_frame_lcia_method_metadata(passthrough).equals(passthrough)
    assert metadata_mod.ensure_frame_lcia_method_metadata(
        pd.DataFrame({"impact": [None], "value": [1.0]})
    ).equals(pd.DataFrame({"impact": [None], "value": [1.0]}))
    with pytest.raises(ValueError):
        metadata_mod.ensure_frame_lcia_method_metadata(
            pd.DataFrame({"impact": [first_impact], "value": [1.0]})
        )
    with pytest.raises(ValueError):
        metadata_mod.resolve_frame_impact_title(
            pd.DataFrame({"impact": [first_impact], "lcia_method": [None]})
        )
    with pytest.raises(ValueError):
        metadata_mod.resolve_frame_impact_unit(pd.DataFrame({"impact_unit": ["kg", "t"]}))
    assert (
        metadata_mod.resolve_frame_impact_unit(
            pd.DataFrame(
                {"impact": [first_impact], "lcia_method": ["pb_lcia"], "impact_unit": [" "]}
            )
        )
        == pb_metadata.units[first_impact]
    )
    with pytest.raises(ValueError):
        metadata_mod.resolve_frame_impact_title(
            pd.DataFrame(
                {
                    "impact": [first_impact, second_impact],
                    "lcia_method": ["pb_lcia", "pb_lcia"],
                }
            )
        )
    assert (
        metadata_mod.resolve_frame_impact_title(
            pd.DataFrame({"impact": [None], "value": [1.0]}),
            lcia_method_column="custom_method",
        )
        is None
    )
    assert (
        metadata_mod.resolve_frame_impact_title(
            pd.DataFrame({"impact": [None], "lcia_method": [None], "value": [1.0]})
        )
        is None
    )
    assert (
        metadata_mod.resolve_frame_impact_unit(
            pd.DataFrame({"impact": [None], "lcia_method": [None], "value": [1.0]})
        )
        is None
    )
    assert (
        metadata_mod.resolve_frame_impact_unit(pd.DataFrame({"impact": [None], "value": [1.0]}))
        is None
    )
    with pytest.raises(ValueError):
        metadata_mod.resolve_frame_impact_unit(
            pd.DataFrame({"impact": [first_impact], "custom_method": [None]}),
            lcia_method_column="custom_method",
        )
    with pytest.raises(ValueError):
        metadata_mod.resolve_frame_impact_unit(
            pd.DataFrame({"impact": ["not_a_real_impact"], "lcia_method": ["pb_lcia"]})
        )
    with pytest.raises(ValueError):
        metadata_mod.resolve_frame_impact_unit(
            pd.DataFrame(
                {
                    "impact": [first_impact, second_impact],
                    "lcia_method": ["pb_lcia", "pb_lcia"],
                }
            )
        )

    scope_frame = pd.DataFrame(
        {
            "lcia_method": ["pb_lcia", None, "gwp100_lcia"],
            "impact": [first_impact, None, "co2"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    method_slices = list(scope_mod.lcia_method_slices(scope_frame))
    assert [(token, title, lcia_method) for token, title, _, lcia_method in method_slices] == [
        ("gwp100_lcia", "gwp100_lcia", "gwp100_lcia"),
        ("pb_lcia", "pb_lcia", "pb_lcia"),
    ]
    method_slices_without_generic_fill = list(
        scope_mod.lcia_method_slices(
            scope_frame,
            fill_generic_method=False,
        )
    )
    first_method_scope = method_slices_without_generic_fill[0][2]
    assert isinstance(first_method_scope, pd.DataFrame)
    assert bool(first_method_scope["lcia_method"].isna().any())
    assert (
        list(scope_mod.lcia_method_slices(pd.DataFrame())).__repr__()
        == "[('all', '', Empty DataFrame\nColumns: []\nIndex: [], None)]"
    )  # noqa: E501
    no_method_column = pd.DataFrame({"value": [1.0]})
    no_method_slices = list(scope_mod.lcia_method_slices(no_method_column))
    assert len(no_method_slices) == 1
    assert no_method_slices[0][0:2] == ("all", "")
    assert no_method_slices[0][2].equals(no_method_column)
    assert no_method_slices[0][3] is None
    missing_only = pd.DataFrame({"lcia_method": [None, pd.NA], "value": [1.0, 2.0]})
    missing_only_slices = list(scope_mod.lcia_method_slices(missing_only))
    assert len(missing_only_slices) == 1
    assert missing_only_slices[0][0:2] == ("all", "")
    assert missing_only_slices[0][2].equals(missing_only)
    assert missing_only_slices[0][3] is None
    generic_only = pd.DataFrame({"impact": [None], "lcia_method": [None], "value": [1.0]})
    generic_lcia_impact_slices = list(scope_mod.combined_lcia_impact_slices(generic_only))
    assert generic_lcia_impact_slices[0][0:2] == ("all", "")
    assert generic_lcia_impact_slices[0][2].equals(generic_only)
    assert generic_lcia_impact_slices[0][3] is None
    assert scope_mod.resolve_unique_lcia_method(pd.DataFrame()) is None
    assert scope_mod.resolve_unique_lcia_method(pd.DataFrame({"value": [1.0]})) is None
    assert (
        scope_mod.resolve_unique_lcia_method(pd.DataFrame({"lcia_method": ["pb_lcia", None]}))
        == "pb_lcia"
    )
    assert (
        scope_mod.resolve_unique_lcia_method(
            pd.DataFrame({"lcia_method": ["pb_lcia", "gwp100_lcia"]})
        )
        is None
    )

    impact_slices = list(
        scope_mod.impact_slices(
            scope_frame,
            impact_column="impact",
            repeat_generic=True,
        )
    )
    assert [token for token, _ in impact_slices] == [first_impact, "co2"]
    assert (
        list(
            scope_mod.impact_slices(
                pd.DataFrame({"impact": [None], "value": [1.0]}),
                impact_column="impact",
                repeat_generic=True,
            )
        )[0][0]
        == "value"
    )
    assert (
        list(
            scope_mod.impact_slices(
                pd.DataFrame({"impact": ["  AAL  "], "value": [1.0]}),
                impact_column="impact",
                repeat_generic=False,
            )
        )[0][0]
        == "value"
    )
    no_impact_slices = list(
        scope_mod.impact_slices(
            pd.DataFrame({"value": [1.0]}),
            impact_column=None,
            repeat_generic=False,
        )
    )
    assert len(no_impact_slices) == 1
    assert no_impact_slices[0][0] == "value"
    assert no_impact_slices[0][1].equals(pd.DataFrame({"value": [1.0]}))

    combined = list(
        scope_mod.combined_impact_slices(
            pd.DataFrame(
                {
                    "impact": [first_impact, None],
                    "lcia_method": ["pb_lcia", None],
                    "value": [1.0, 2.0],
                }
            )
        )
    )
    assert combined[0][0] == first_impact
    assert combined[0][1] == impact_title
    lcia_impact_combined = list(
        scope_mod.combined_lcia_impact_slices(
            pd.DataFrame(
                {
                    "impact": [first_impact, None, "co2"],
                    "lcia_method": ["pb_lcia", None, "gwp100_lcia"],
                    "value": [1.0, 2.0, 3.0],
                }
            )
        )
    )
    assert [(token, method) for token, _title, _frame, method in lcia_impact_combined] == [
        ("co2", "gwp100_lcia"),
        (first_impact, "pb_lcia"),
    ]
    assert all(len(frame) == 2 for _token, _title, frame, _method in lcia_impact_combined)
    assert (
        list(scope_mod.combined_impact_slices(pd.DataFrame())).__repr__()
        == "[('all', '', Empty DataFrame\nColumns: []\nIndex: [])]"
    )  # noqa: E501
    assert (
        list(scope_mod.combined_lcia_impact_slices(pd.DataFrame())).__repr__()
        == "[('all', '', Empty DataFrame\nColumns: []\nIndex: [], None)]"
    )  # noqa: E501
    no_impact_combined = list(scope_mod.combined_impact_slices(pd.DataFrame({"value": [1.0]})))
    assert len(no_impact_combined) == 1
    assert no_impact_combined[0][0:2] == ("all", "")
    assert no_impact_combined[0][2].equals(pd.DataFrame({"value": [1.0]}))
    missing_impact_combined = list(
        scope_mod.combined_impact_slices(
            pd.DataFrame({"impact": [None], "lcia_method": [None], "value": [1.0]})
        )
    )
    assert len(missing_impact_combined) == 1
    assert missing_impact_combined[0][0:2] == ("all", "")
    whitespace_combined = list(
        scope_mod.combined_impact_slices(
            pd.DataFrame({"impact": ["  AAL  "], "lcia_method": ["pb_lcia"], "value": [1.0]})
        )
    )
    assert len(whitespace_combined) == 1
    assert whitespace_combined[0][0:2] == ("all", "")
    whitespace_lcia_combined = list(
        scope_mod.combined_lcia_impact_slices(
            pd.DataFrame({"impact": ["  AAL  "], "lcia_method": ["pb_lcia"], "value": [1.0]})
        )
    )
    assert len(whitespace_lcia_combined) == 1
    assert whitespace_lcia_combined[0][0:2] == ("all", "")
    assert scope_mod.suffix_path(Path("plots/demo.png"), "impact co2") == Path(
        "plots/demo.png__impact_co2"
    )
    with pytest.raises(ValueError):
        list(scope_mod.combined_impact_slices(pd.DataFrame({"impact": [first_impact]})))
    with pytest.raises(ValueError):
        list(scope_mod.combined_lcia_impact_slices(pd.DataFrame({"impact": [first_impact]})))
    with pytest.raises(ValueError):
        list(
            scope_mod.combined_impact_slices(
                pd.DataFrame({"impact": [first_impact], "lcia_method": [None]})
            )
        )
    with pytest.raises(ValueError):
        list(
            scope_mod.combined_lcia_impact_slices(
                pd.DataFrame({"impact": [first_impact], "lcia_method": [None]})
            )
        )
    with pytest.raises(ValueError):
        list(
            scope_mod.impact_slices(
                pd.DataFrame({"impact": [None], "lcia_method": ["pb_lcia"], "value": [1.0]}),
                impact_column="impact",
                repeat_generic=True,
            )
        )
    with pytest.raises(ValueError):
        list(
            scope_mod.combined_impact_slices(
                pd.DataFrame({"impact": [None], "lcia_method": ["pb_lcia"], "value": [1.0]})
            )
        )
