from pathlib import Path
import pandas as pd
import pytest

from pyaesa.download.ar6.utils.config import (
    GROSS_ALT_KYOTO_WO_AFOLU,
    GROSS_KYOTO_WO_AFOLU,
    NET_CO2_WITH_AFOLU,
    NET_CO2_WO_AFOLU,
    NET_KYOTO_WITH_AFOLU,
    NET_KYOTO_WO_AFOLU,
    SEQUESTRATION_SUBTOTAL,
    SEQUESTRATION_TOTAL,
)
from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NET,
    CC_FLOW_NEGATIVE,
    CC_FLOW_POSITIVE,
    cc_denominator_variable,
    cc_positive_flow,
    cc_sequestration_variable,
    cc_variable,
    downstream_cc_rows,
    emission_type_tag,
    emissions_mode_tag,
    normalize_emission_type,
    normalize_emissions_mode,
)
from pyaesa.ar6_cc.deterministic.figures.render import render_cc_pathway_figures
from pyaesa.ar6_cc.deterministic.io.tables import (
    build_cc_table,
    filter_to_denominator_cc_rows,
    filter_pathways,
    read_cc_output,
    write_cc_output,
)
from pyaesa.ar6_cc.deterministic.io.paths import (
    get_cc_metadata_path,
    get_cc_output_path,
    get_cc_scope_dir,
    get_subset_csv_path,
)
from pyaesa.ar6_cc.shared.runtime.figure_titles import ar6_cc_title, category_scope_label
from pyaesa.ar6_cc.shared.runtime.figure_style import ar6_category_color
from pyaesa.ar6_cc.shared.runtime.paths import afolu_tag, cc_selector_dir_name
from pyaesa.ar6_cc.deterministic.runtime.reports import (
    AR6CCPathwayCount,
    ComputeAR6CCReport,
)
from pyaesa.shared.figures.dynamic_ar6 import (
    dynamic_ar6_detail_line,
    model_scenario_pair_label,
    model_scenario_pair_token,
)


def _read_cc_table(path: Path, output_format: str) -> pd.DataFrame:
    return read_cc_output(output_file=path, output_format=output_format)


def _pathway_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            ("M1", "S1", NET_KYOTO_WO_AFOLU),
            ("M2", "S2", NET_KYOTO_WO_AFOLU),
            ("M3", "S3", NET_CO2_WO_AFOLU),
        ],
        names=["model", "scenario", "variable"],
    )
    return pd.DataFrame(
        {
            "Category": ["C1", "C2", "C1"],
            "Ssp_family": [1, 2, 1],
            "unit": ["Gt", "Gt", "Gt"],
            2019: [10.0, 20.0, 30.0],
            2020: [11.0, 21.0, 31.0],
        },
        index=index,
    )


def test_dynamic_cc_contracts_and_report_summary(tmp_path: Path) -> None:
    assert normalize_emission_type(" CO2 ") == "co2"
    assert afolu_tag(include_afolu=False) == "wo_afolu"
    assert afolu_tag(include_afolu=True) == "with_afolu"
    assert emission_type_tag(emission_type="kyoto_gases") == "kyoto_gases"
    assert normalize_emissions_mode(" Gross_Alt ") == "gross_alt"
    assert emissions_mode_tag(emissions_mode=" gross ") == "gross"
    assert cc_sequestration_variable(emissions_mode="net") is None
    assert cc_sequestration_variable(emissions_mode="gross") == SEQUESTRATION_TOTAL
    assert cc_sequestration_variable(emissions_mode="gross_alt") == SEQUESTRATION_SUBTOTAL
    assert cc_positive_flow(emissions_mode="net") == CC_FLOW_NET
    assert cc_positive_flow(emissions_mode="gross") == CC_FLOW_POSITIVE
    assert cc_variable(emission_type="kyoto_gases", include_afolu=False) == (
        GROSS_ALT_KYOTO_WO_AFOLU
    )
    assert (
        cc_denominator_variable(
            emission_type="kyoto_gases",
            include_afolu=False,
            emissions_mode="gross",
        )
        == GROSS_KYOTO_WO_AFOLU
    )
    assert (
        cc_variable(
            emission_type="kyoto_gases",
            include_afolu=False,
            emissions_mode="net",
        )
        == NET_KYOTO_WO_AFOLU
    )
    assert (
        cc_variable(
            emission_type="kyoto_gases",
            include_afolu=True,
            emissions_mode="net",
        )
        == NET_KYOTO_WITH_AFOLU
    )
    assert (
        cc_variable(emission_type="co2", include_afolu=False, emissions_mode="net")
        == NET_CO2_WO_AFOLU
    )
    assert (
        cc_variable(emission_type="co2", include_afolu=True, emissions_mode="net")
        == NET_CO2_WITH_AFOLU
    )

    with pytest.raises(ValueError, match="emission_type must be"):
        normalize_emission_type("methane")
    with pytest.raises(ValueError, match="emissions_mode must be"):
        normalize_emissions_mode("invalid")

    flow_frame = pd.DataFrame(
        {
            "cc_flow": [CC_FLOW_NET, CC_FLOW_POSITIVE, CC_FLOW_NEGATIVE],
            "value": [1.0, 2.0, -3.0],
        }
    )
    assert downstream_cc_rows(flow_frame)["cc_flow"].tolist() == [CC_FLOW_NET, CC_FLOW_POSITIVE]
    assert filter_to_denominator_cc_rows(flow_frame)["value"].tolist() == [1.0, 2.0]

    output_file = tmp_path / "cc" / "ar6_cc.csv"
    figure_path = tmp_path / "cc" / "figures" / "dynamic_cc_pathways.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_bytes(b"png")
    logs_dir = tmp_path / "cc" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    minimal = ComputeAR6CCReport(
        study_period=[2019, 2020],
        harmonization=True,
        harmonization_method="offset",
        emission_type="kyoto_gases",
        include_afolu=False,
        emissions_mode="net",
        variable=NET_KYOTO_WO_AFOLU,
        categories=["C1"],
        ssp_scenarios=["SSP1"],
        subset_version=None,
        total_model_scenario_pairs=1,
        output_file=output_file,
        process_ar6={"variable_coverage": [], "figures_available": 0},
    )
    minimal_text = str(minimal)
    assert minimal_text

    report = ComputeAR6CCReport(
        study_period=[2019, 2021],
        harmonization=True,
        harmonization_method="offset",
        emission_type="kyoto_gases",
        include_afolu=False,
        emissions_mode="net",
        variable=NET_KYOTO_WO_AFOLU,
        categories=["C1", "C2"],
        ssp_scenarios=["SSP1", "SSP2"],
        subset_version="core",
        total_model_scenario_pairs=2,
        output_file=output_file,
        pathway_counts=[
            AR6CCPathwayCount(category="C1", ssp_scenario="SSP1", model_scenario_pairs=2)
        ],
        missing_pathway_combinations=[
            AR6CCPathwayCount(category="C2", ssp_scenario="SSP2", model_scenario_pairs=0)
        ],
        post_study_output_file=tmp_path / "cc" / "post_study.csv",
        figure_paths=[figure_path],
        cc_dir=tmp_path / "cc",
        logs_dir=logs_dir,
        process_ar6={
            "reuse_status": "reused_exact",
            "study_period": None,
            "variable_coverage": [
                "ignored",
                {
                    "variable": "Emissions|CO2",
                    "retained_model_scenario_pairs": 1,
                    "available_model_scenario_pairs": 2,
                },
            ],
            "figures_available": 1,
        },
    )
    report_text = str(report)
    assert repr(report) == report_text
    assert all(len(line) <= 100 for line in report_text.splitlines() if str(tmp_path) not in line)


def test_dynamic_cc_titles_and_deterministic_figures_cover_edge_cases() -> None:
    assert category_scope_label([]) == ""
    assert category_scope_label(["C4", "C2", "C3"]) == "C2-C4"
    assert category_scope_label(["C1", "C3", "C5"]) == "C1, C3, C5"
    assert category_scope_label(["A", "C1"]) == "A, C1"
    assert model_scenario_pair_token(models=["M 1"], scenarios=["S/1"]) == "M_1_S_1"
    assert model_scenario_pair_token(models=["M1", "M2"], scenarios=["S1"]) is None
    assert model_scenario_pair_label(models=["M1"], scenarios=["S1"]) == "M1 / S1"
    assert dynamic_ar6_detail_line(categories=[], models=["M1"], scenarios=["S1"]) == (
        "Model-scenario pair: M1 / S1"
    )
    assert dynamic_ar6_detail_line(categories=["C1", "C2"], models=["M1"], scenarios=["S1"]) == (
        "AR6 categories: C1-C2 | Model-scenario pair: M1 / S1"
    )
    assert dynamic_ar6_detail_line(categories=["C1"], models=["M1", "M2"], scenarios=["S1"]) == ""
    assert (
        ar6_cc_title(
            variable_name="Emissions|CO2",
            ssp_scenario="SSP2",
            categories=[],
        )
        == "AR6 pathways | Emissions|CO2 | SSP2"
    )
    assert (
        ar6_cc_title(
            variable_name="Emissions|CO2",
            ssp_scenario="SSP2",
            categories=["C1", "C2", "C3"],
        )
        == "AR6 pathways | Emissions|CO2 | SSP2 | C1-C3"
    )

    assert ar6_category_color(category="C1") == "#7FBC41"
    assert ar6_category_color(category="C2") == "#2654D2"
    assert ar6_category_color(category="C3") == "#F39B1F"
    assert ar6_category_color(category="C4") == "#5A0418"
    assert ar6_category_color(category="C3") != ar6_category_color(
        category="C4",
    )


def test_dynamic_cc_io_paths_cover_filters_and_formats(tmp_path: Path) -> None:
    pathways = _pathway_frame()

    with pytest.raises(ValueError, match="Variable 'missing' not found"):
        filter_pathways(
            pathways,
            variable="missing",
            category=None,
            ssp_scenario=None,
            subset_version=None,
            processed_dir=tmp_path,
        )

    with pytest.raises(FileNotFoundError, match="Model-scenario subset CSV not found"):
        filter_pathways(
            pathways,
            variable=NET_KYOTO_WO_AFOLU,
            category=None,
            ssp_scenario=None,
            subset_version="core",
            processed_dir=tmp_path,
        )

    subset_path = get_subset_csv_path(tmp_path, "core")
    pd.DataFrame({"model": ["M1"]}).to_csv(subset_path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        filter_pathways(
            pathways,
            variable=NET_KYOTO_WO_AFOLU,
            category=None,
            ssp_scenario=None,
            subset_version="core",
            processed_dir=tmp_path,
        )

    pd.DataFrame({"model": ["M1"], "scenario": ["S1"]}).to_csv(subset_path, index=False)
    filtered = filter_pathways(
        pathways,
        variable=NET_KYOTO_WO_AFOLU,
        category=["C1"],
        ssp_scenario=["SSP1"],
        subset_version="core",
        processed_dir=tmp_path,
    )
    assert list(filtered["model"]) == ["M1"]
    assert list(filtered["scenario"]) == ["S1"]

    with pytest.raises(ValueError, match="No AR6 pathways remain after filtering"):
        filter_pathways(
            pathways,
            variable=NET_KYOTO_WO_AFOLU,
            category=["C9"],
            ssp_scenario=None,
            subset_version=None,
            processed_dir=tmp_path,
        )

    cc_table = build_cc_table(
        filtered,
        [2019, 2020],
        cc_flow=CC_FLOW_NET,
        cc_variable=NET_KYOTO_WO_AFOLU,
    )
    assert list(cc_table.columns) == [
        "cc_model",
        "cc_scenario",
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        2019,
        2020,
    ]
    csv_path = get_cc_output_path(cc_dir=tmp_path / "csv_scope", output_format="csv")
    parquet_path = get_cc_output_path(cc_dir=tmp_path / "parquet_scope", output_format="parquet")
    pickle_path = get_cc_output_path(cc_dir=tmp_path / "pickle_scope", output_format="pickle")
    assert csv_path.parent.name == "results"
    write_cc_output(cc_table, csv_path, "csv")
    write_cc_output(cc_table, parquet_path, "parquet")
    write_cc_output(cc_table, pickle_path, "pickle")
    expected_roundtrip = cc_table.rename(columns=str)
    csv_loaded = _read_cc_table(csv_path, "csv")
    pd.testing.assert_frame_equal(csv_loaded, expected_roundtrip)
    pd.testing.assert_frame_equal(_read_cc_table(parquet_path, "parquet"), expected_roundtrip)
    pd.testing.assert_frame_equal(_read_cc_table(pickle_path, "pickle"), cc_table)

    with pytest.raises(ValueError, match="Unsupported output_format 'xlsx'"):
        write_cc_output(cc_table, tmp_path / "bad.xlsx", "xlsx")


def test_dynamic_cc_figures_cover_year_resolution_and_empty_scopes(
    tmp_path: Path,
) -> None:
    assert (
        render_cc_pathway_figures(
            cc_table=pd.DataFrame(),
            variable_name=NET_KYOTO_WO_AFOLU,
            output_dir=tmp_path / "empty",
            dpi=10,
            output_format="png",
        )
        == []
    )

    rendered = render_cc_pathway_figures(
        cc_table=pd.DataFrame(
            [
                {
                    "cc_model": "M1",
                    "cc_scenario": "S1",
                    "cc_category": "C1",
                    "ssp_scenario": "SSP1",
                    "cc_flow": CC_FLOW_NET,
                    "cc_variable": NET_KYOTO_WO_AFOLU,
                    "impact_unit": "Gt",
                    "2019": 10.0,
                    "2020": 11.0,
                }
            ]
        ),
        variable_name=NET_KYOTO_WO_AFOLU,
        output_dir=tmp_path / "figures",
        dpi=10,
        output_format="png",
        requested_years=[2019, 2020],
    )
    assert len(rendered) == 1
    assert rendered[0].suffix == ".png"
    assert rendered[0].exists()

    dense_positive_rows = [
        {
            "cc_model": f"M{index}",
            "cc_scenario": f"S{index}",
            "cc_category": "C1",
            "ssp_scenario": "SSP1",
            "cc_flow": CC_FLOW_POSITIVE,
            "cc_variable": GROSS_ALT_KYOTO_WO_AFOLU,
            "impact_unit": "Gt",
            2019: 10.0 + index,
            2020: 11.0 + index,
        }
        for index in range(11)
    ]
    mixed_flow_rendered = render_cc_pathway_figures(
        cc_table=pd.DataFrame(
            [
                *dense_positive_rows,
                {
                    "cc_model": "M12",
                    "cc_scenario": "S12",
                    "cc_category": "C2",
                    "ssp_scenario": "SSP1",
                    "cc_flow": CC_FLOW_POSITIVE,
                    "cc_variable": GROSS_ALT_KYOTO_WO_AFOLU,
                    "impact_unit": "Gt",
                    2019: 10.0,
                    2020: 11.0,
                },
                {
                    "cc_model": "M12",
                    "cc_scenario": "S12",
                    "cc_category": "C2",
                    "ssp_scenario": "SSP1",
                    "cc_flow": CC_FLOW_NEGATIVE,
                    "cc_variable": SEQUESTRATION_SUBTOTAL,
                    "impact_unit": "Gt",
                    2019: 0.0,
                    2020: 0.0,
                },
                {
                    "cc_model": "M13",
                    "cc_scenario": "S13",
                    "cc_category": "C2",
                    "ssp_scenario": "SSP1",
                    "cc_flow": CC_FLOW_POSITIVE,
                    "cc_variable": GROSS_ALT_KYOTO_WO_AFOLU,
                    "impact_unit": "Gt",
                    2019: 12.0,
                    2020: 13.0,
                },
                {
                    "cc_model": "M13",
                    "cc_scenario": "S13",
                    "cc_category": "C2",
                    "ssp_scenario": "SSP1",
                    "cc_flow": CC_FLOW_NEGATIVE,
                    "cc_variable": SEQUESTRATION_SUBTOTAL,
                    "impact_unit": "Gt",
                    2019: 0.0,
                    2020: -1.0,
                },
            ]
        ),
        variable_name=GROSS_ALT_KYOTO_WO_AFOLU,
        output_dir=tmp_path / "mixed_flows",
        dpi=10,
        output_format="png",
        requested_years=[2019, 2020],
    )
    assert len(mixed_flow_rendered) == 1

    violin_only_rendered = render_cc_pathway_figures(
        cc_table=pd.DataFrame(
            [
                {
                    "cc_model": f"M{index}",
                    "cc_scenario": f"S{index}",
                    "cc_category": "C1",
                    "ssp_scenario": "SSP1",
                    "cc_flow": CC_FLOW_POSITIVE,
                    "cc_variable": GROSS_ALT_KYOTO_WO_AFOLU,
                    "impact_unit": "Gt",
                    2019: 10.0 + index,
                    2020: 11.0 + index,
                }
                for index in range(3)
            ]
        ),
        variable_name=GROSS_ALT_KYOTO_WO_AFOLU,
        output_dir=tmp_path / "violin_only",
        dpi=10,
        output_format="png",
        requested_years=[2019, 2020],
    )
    assert len(violin_only_rendered) == 1


def test_dynamic_cc_paths_and_metadata_runtime(project_repo: Path) -> None:
    del project_repo

    scope_dir = get_cc_scope_dir(
        [2019, 2020],
        harmonization=True,
        harmonization_method="offset",
        emission_type="kyoto_gases",
        include_afolu=False,
        subset_version="core",
        category=["C1", "C3"],
        ssp_scenario=["SSP1", "SSP5"],
    )
    assert not scope_dir.exists()
    assert scope_dir.parent.name == "C1-C3__SSP1-SSP5"
    assert cc_selector_dir_name(category=["C3", "C1"], ssp_scenario=["SSP5", "SSP1"]) == (
        "C1-C3__SSP1-SSP5"
    )
    output_path = get_cc_output_path(cc_dir=scope_dir, output_format="csv")
    assert output_path.name == "ar6_cc.csv"
    assert output_path.parent.name == "results"
    assert get_cc_metadata_path(cc_dir=scope_dir).name == "scope_manifest.json"
    assert get_subset_csv_path(scope_dir.parents[3], "core").name == (
        "model_scenario_subset__core.csv"
    )
