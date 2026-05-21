import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.lines import Line2D

from pyaesa import deterministic_ar6_cc
from pyaesa.download.ar6.utils.config import (
    NET_CO2_WITH_AFOLU,
    NET_CO2_WO_AFOLU,
    NET_KYOTO_WITH_AFOLU,
    NET_KYOTO_WO_AFOLU,
    SEQUESTRATION_SUBTOTAL,
)
from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NET,
    CC_FLOW_NEGATIVE,
    CC_FLOW_POSITIVE,
    cc_variable,
    normalize_emission_type,
    normalize_emissions_mode,
)
from pyaesa.ar6_cc.deterministic.figures.render import render_cc_pathway_figures
from pyaesa.ar6_cc.deterministic.figures.period_panels import (
    plot_budget_panel,
    render_deterministic_legends_below,
)
from pyaesa.ar6_cc.deterministic.io.tables import (
    build_cc_table,
    filter_pathways,
    merge_cc_tables,
    read_cc_output,
    read_harmonized_pathways,
    write_cc_output,
)
from pyaesa.ar6_cc.deterministic.io.paths import (
    get_cc_metadata_path,
    get_cc_output_path,
    get_cc_post_study_output_path,
    get_cc_summary_log_path,
    get_subset_csv_path,
)
from pyaesa.ar6_cc.deterministic.runtime.metadata import (
    build_run_metadata_payload,
    figure_state_matches,
    set_figure_state,
)
from pyaesa.ar6_cc.deterministic.runtime.reports import AR6CCPathwayCount, ComputeAR6CCReport
from pyaesa.process.ar6.utils.io.metadata import build_process_metadata_payload, write_json
from pyaesa.process.ar6.utils.io.paths import get_processed_scope_dir
from pyaesa.process.ar6.utils.io.paths import get_logs_dir, get_processed_dir


def _read_cc_table(path: Path, output_format: str) -> pd.DataFrame:
    return read_cc_output(output_file=path, output_format=output_format)


def _run_ar6_cc(
    *,
    figures: bool,
    refresh: bool,
    category: list[str] | None = None,
    ssp_scenario: list[str] | None = None,
    emissions_mode: str = "gross_alt",
    figure_format: dict[str, Any] | None = None,
):
    resolved_figure_format: dict[str, object] = (
        {"format": "png", "dpi": 10} if figure_format is None else figure_format
    )
    return deterministic_ar6_cc(
        years=range(2019, 2022),
        category=(["C1"] if category is None else category),
        ssp_scenario=(["SSP1"] if ssp_scenario is None else ssp_scenario),
        emissions_mode=emissions_mode,
        figures=figures,
        figure_format=resolved_figure_format,
        refresh=refresh,
    )


def test_deterministic_ar6_cc_end_to_end_reuse_and_refresh(ar6_dummy_repo) -> None:
    first_report = _run_ar6_cc(figures=False, refresh=True)

    assert first_report is not None
    assert first_report.study_period == [2019, 2021]
    assert first_report.harmonization is True
    assert first_report.harmonization_method == "offset"
    assert first_report.emission_type == "kyoto_gases"
    assert first_report.include_afolu is False
    assert first_report.emissions_mode == "gross_alt"
    assert first_report.categories == ["C1"]
    assert first_report.ssp_scenarios == ["SSP1"]
    assert first_report.total_model_scenario_pairs == 1
    assert first_report.output_file.exists()
    assert first_report.output_file.name == "ar6_cc.csv"
    assert first_report.output_file.parent.name == "results"
    assert first_report.post_study_output_file is not None
    assert first_report.post_study_output_file.exists()
    assert first_report.post_study_output_file.name == "ar6_cc_post_study_period.csv"
    assert first_report.meta_file is not None and first_report.meta_file.exists()
    assert first_report.meta_file.name == "scope_manifest.json"
    assert first_report.logs_dir is not None and first_report.logs_dir.exists()
    assert first_report.cc_dir is not None
    summary_log = get_cc_summary_log_path(cc_dir=first_report.cc_dir)
    assert summary_log.read_text(encoding="utf-8").strip()
    assert first_report.figure_paths == []
    first_summary = str(first_report)
    assert first_summary

    output_frame = pd.read_csv(first_report.output_file)
    post_output_frame = pd.read_csv(first_report.post_study_output_file)
    assert {
        "cc_model",
        "cc_scenario",
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "2019",
        "2020",
        "2021",
    }.issubset(output_frame.columns)
    assert set(output_frame["cc_model"]) == {"M1"}
    assert set(output_frame["cc_category"]) == {"C1"}
    assert set(output_frame["ssp_scenario"]) == {"SSP1"}
    assert set(output_frame["cc_flow"]) == {CC_FLOW_POSITIVE, CC_FLOW_NEGATIVE}
    assert set(output_frame["cc_variable"]) == {first_report.variable, SEQUESTRATION_SUBTOTAL}
    assert output_frame["impact_unit"].drop_duplicates().tolist() == ["MtCO2eq/yr"]
    assert {"2022", "2100"}.issubset(post_output_frame.columns)
    assert {"2019", "2020", "2021"}.isdisjoint(post_output_frame.columns)
    processed_dir = get_processed_dir(
        first_report.study_period,
        harmonization=first_report.harmonization,
        harmonization_method=first_report.harmonization_method,
    )
    processed_pathways = read_harmonized_pathways(
        processed_dir=processed_dir,
        harmonization=first_report.harmonization,
    )
    assert SEQUESTRATION_SUBTOTAL in processed_pathways.index.get_level_values("variable")

    net_report = _run_ar6_cc(figures=False, refresh=False, emissions_mode="net")
    assert net_report is not None
    net_output = _read_cc_table(net_report.output_file, "csv")
    assert set(net_output["cc_flow"]) == {CC_FLOW_NET}

    reused_report = _run_ar6_cc(figures=False, refresh=False)
    assert reused_report is not None
    assert reused_report.reuse_status == "reused_exact"

    processed_sentinel = processed_dir / "refresh_sentinel.txt"
    raw_sentinel = ar6_dummy_repo.raw_dir / "raw_refresh_sentinel.txt"
    processed_sentinel.write_text("processed", encoding="utf-8")
    raw_sentinel.write_text("raw", encoding="utf-8")
    refreshed_report = _run_ar6_cc(figures=False, refresh=True)

    assert refreshed_report is not None
    assert refreshed_report.output_file.exists()
    assert not processed_sentinel.exists()
    assert raw_sentinel.exists()


def test_deterministic_ar6_cc_default_scope_exact_reuse_returns_report(ar6_dummy_repo) -> None:
    del ar6_dummy_repo

    first_report = deterministic_ar6_cc(
        years=range(2019, 2022),
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    )

    assert first_report is not None
    reused_report = deterministic_ar6_cc(
        years=range(2019, 2022),
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    )
    assert reused_report.reuse_status == "reused_exact"


def test_deterministic_ar6_cc_ending_in_2100_has_no_post_study_table(ar6_dummy_repo) -> None:
    del ar6_dummy_repo
    _write_minimal_harmonized_ar6_workbook_for_cc(study_period=[2099, 2100])

    report = deterministic_ar6_cc(
        years=range(2099, 2101),
        category=["C1"],
        ssp_scenario=["SSP1"],
        emissions_mode="net",
        figures=False,
        refresh=True,
    )
    reused = deterministic_ar6_cc(
        years=range(2099, 2101),
        category=["C1"],
        ssp_scenario=["SSP1"],
        emissions_mode="net",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    )

    assert report.post_study_output_file is None
    assert str(report)
    assert reused.reuse_status == "partially_reused"
    assert reused.figure_paths


def test_deterministic_ar6_cc_selector_scopes_are_isolated(
    ar6_dummy_repo,
) -> None:
    del ar6_dummy_repo

    ssp1_report = deterministic_ar6_cc(
        years=range(2019, 2022),
        category=["C1"],
        ssp_scenario=["SSP1"],
        figures=False,
        refresh=True,
    )

    assert ssp1_report is not None
    assert ssp1_report.cc_dir is not None
    assert ssp1_report.cc_dir.parent.name == "C1__SSP1"

    ssp2_report = deterministic_ar6_cc(
        years=range(2019, 2022),
        category=["C2"],
        ssp_scenario=["SSP2"],
        figures=False,
        refresh=False,
    )
    assert ssp2_report.cc_dir is not None
    assert ssp2_report.cc_dir.parent.name == "C2__SSP2"
    assert ssp2_report.cc_dir != ssp1_report.cc_dir

    non_consecutive_report = deterministic_ar6_cc(
        years=range(2019, 2022),
        category=["C1", "C3"],
        ssp_scenario=["SSP1", "SSP3"],
        figures=True,
        figure_format={"format": "svg", "dpi": 1},
        refresh=False,
    )

    assert non_consecutive_report is not None
    assert non_consecutive_report.cc_dir is not None
    assert non_consecutive_report.cc_dir.parent.name == "C1-C3__SSP1-SSP3"
    assert non_consecutive_report.figure_paths

    ssp1_output_path = get_cc_output_path(cc_dir=ssp1_report.cc_dir, output_format="csv")
    ssp1_post_output_path = get_cc_post_study_output_path(
        cc_dir=ssp1_report.cc_dir,
        output_format="csv",
    )
    ssp1_rows = read_cc_output(output_file=ssp1_output_path, output_format="csv")
    ssp1_post_rows = read_cc_output(output_file=ssp1_post_output_path, output_format="csv")
    assert set(ssp1_rows["cc_category"]) == {"C1"}
    assert set(ssp1_rows["ssp_scenario"]) == {"SSP1"}
    assert set(ssp1_post_rows["cc_category"]) == {"C1"}
    assert set(ssp1_post_rows["ssp_scenario"]) == {"SSP1"}
    metadata = json.loads(
        get_cc_metadata_path(cc_dir=ssp1_report.cc_dir).read_text(encoding="utf-8")
    )
    assert metadata["arguments"]["category"] == ["C1"]
    assert metadata["arguments"]["ssp_scenario"] == ["SSP1"]
    assert metadata["provenance"]["cc_categories"] == ["C1"]
    assert metadata["provenance"]["ssp_scenarios"] == ["SSP1"]

    reused_ssp1_report = deterministic_ar6_cc(
        years=range(2019, 2022),
        category=["C1"],
        ssp_scenario=["SSP1"],
        figures=True,
        figure_format={"format": "svg", "dpi": 1},
        refresh=False,
    )
    assert reused_ssp1_report.reuse_status == "partially_reused"
    assert reused_ssp1_report.cc_dir == ssp1_report.cc_dir


def test_deterministic_ar6_cc_figures_false_preserves_existing_figures(
    ar6_dummy_repo,
) -> None:
    del ar6_dummy_repo

    first_report = _run_ar6_cc(figures=True, refresh=True)

    assert first_report is not None
    assert first_report.cc_dir is not None
    assert len(first_report.figure_paths) == 1
    assert all(path.exists() for path in first_report.figure_paths)
    reused_figure_report = _run_ar6_cc(figures=True, refresh=False)
    assert reused_figure_report is not None
    assert reused_figure_report.reuse_status == "reused_exact"

    preserved_metadata = json.loads(
        get_cc_metadata_path(cc_dir=first_report.cc_dir).read_text(encoding="utf-8")
    )
    skipped_report = _run_ar6_cc(figures=False, refresh=False)

    assert skipped_report is not None
    assert skipped_report.reuse_status == "reused_exact"
    assert all(path.exists() for path in first_report.figure_paths)
    assert (
        json.loads(get_cc_metadata_path(cc_dir=first_report.cc_dir).read_text(encoding="utf-8"))[
            "artifacts"
        ]["figure_paths"]
        == preserved_metadata["artifacts"]["figure_paths"]
    )
    reused_without_figures = _run_ar6_cc(figures=False, refresh=False)
    assert reused_without_figures is not None
    assert reused_without_figures.reuse_status == "reused_exact"


def test_deterministic_ar6_cc_reuse_updates_summary_log(
    ar6_dummy_repo,
) -> None:
    del ar6_dummy_repo

    first_report = _run_ar6_cc(figures=True, refresh=True)

    assert first_report is not None
    assert first_report.cc_dir is not None
    assert first_report.logs_dir is not None
    summary_log = get_cc_summary_log_path(cc_dir=first_report.cc_dir)

    skipped_report = _run_ar6_cc(figures=False, refresh=False)

    assert skipped_report is not None
    assert skipped_report.reuse_status == "reused_exact"
    assert summary_log.read_text(encoding="utf-8").strip()


def test_deterministic_ar6_cc_accepts_consecutive_list_years_and_rejects_gaps(
    ar6_dummy_repo,
) -> None:
    del ar6_dummy_repo

    report = deterministic_ar6_cc(
        years=[2019, 2020, 2021],
        category=["C1"],
        ssp_scenario=["SSP1"],
        figures=False,
        refresh=True,
    )
    assert report.study_period == [2019, 2021]

    with pytest.raises(ValueError, match="consecutive years"):
        deterministic_ar6_cc(
            years=cast(Any, [2019, 2021]),
            category=["C1"],
            ssp_scenario=["SSP1"],
            figures=False,
            refresh=True,
        )


def test_deterministic_ar6_cc_fresh_figure_render_and_signature_change_rerender(
    ar6_dummy_repo,
) -> None:
    del ar6_dummy_repo

    first_report = _run_ar6_cc(
        figures=True,
        refresh=False,
        figure_format={"format": "png", "dpi": 10},
    )

    assert first_report is not None
    assert len(first_report.figure_paths) == 1
    assert first_report.figure_paths[0].suffix == ".png"
    assert first_report.figure_paths[0].exists()

    rerendered_report = _run_ar6_cc(
        figures=True,
        refresh=False,
        figure_format={"format": "svg", "dpi": 1},
    )

    assert rerendered_report is not None
    assert len(rerendered_report.figure_paths) == 1
    assert rerendered_report.figure_paths[0].suffix == ".svg"
    assert rerendered_report.figure_paths[0].exists()
    assert not any(
        path.suffix == ".png" for path in rerendered_report.figure_paths[0].parent.glob("*")
    )
    reused_svg_report = _run_ar6_cc(
        figures=True,
        refresh=False,
        figure_format={"format": "svg", "dpi": 1},
    )
    assert reused_svg_report is not None
    assert reused_svg_report.reuse_status == "reused_exact"


def test_deterministic_ar6_cc_rejects_empty_category_and_ssp_lists(ar6_dummy_repo) -> None:
    del ar6_dummy_repo

    with pytest.raises(ValueError, match="non empty category string or list"):
        _run_ar6_cc(
            figures=False,
            refresh=True,
            category=[],
        )
    with pytest.raises(ValueError, match="non empty category string or list"):
        _run_ar6_cc(
            figures=False,
            refresh=True,
            category=[""],
        )

    with pytest.raises(
        ValueError,
        match="'ssp_scenario' must be a non-empty string or list",
    ):
        _run_ar6_cc(
            figures=False,
            refresh=True,
            ssp_scenario=[],
        )


def test_deterministic_ar6_cc_single_pair_budget_legend_policy() -> None:
    def frame(pair_count: int) -> pd.DataFrame:
        rows = []
        for pair_index in range(pair_count):
            rows.extend(
                [
                    {
                        "cc_model": f"M{pair_index}",
                        "cc_scenario": f"S{pair_index}",
                        "cc_category": "C1",
                        "ssp_scenario": "SSP2",
                        "cc_flow": CC_FLOW_POSITIVE,
                        "cc_variable": "positive",
                        "impact_unit": "kg/yr",
                        2020: 1.0 + pair_index,
                        2021: 2.0 + pair_index,
                    },
                    {
                        "cc_model": f"M{pair_index}",
                        "cc_scenario": f"S{pair_index}",
                        "cc_category": "C1",
                        "ssp_scenario": "SSP2",
                        "cc_flow": CC_FLOW_NEGATIVE,
                        "cc_variable": "negative",
                        "impact_unit": "kg/yr",
                        2020: -0.5 - pair_index,
                        2021: -0.5 - pair_index,
                    },
                ]
            )
        return pd.DataFrame(rows)

    fig, (pathway_axis, budget_axis) = plt.subplots(ncols=2)
    try:
        budget_handles = plot_budget_panel(
            axis=budget_axis,
            frame=frame(1),
            study_years=[2020, 2021],
            post_years=[],
            category_colors={"C1": "#123456"},
            negative_sequestration_style="dotted",
        )
        assert budget_handles == []
        assert any(patch.get_hatch() == ".." for patch in budget_axis.patches)

        render_deterministic_legends_below(
            fig,
            pathway_axis=pathway_axis,
            budget_axis=budget_axis,
            pathway_handles=[Line2D([0], [0], color="#123456", label="Category C1")],
            budget_handles=budget_handles,
            pathway_ncol=1,
            budget_ncol=1,
        )
        fig.canvas.draw()
        canvas = cast(Any, fig.canvas)
        legend_bbox = (
            fig.legends[0]
            .get_window_extent(canvas.get_renderer())
            .transformed(fig.transFigure.inverted())
        )
        assert abs((legend_bbox.x0 + legend_bbox.x1) * 0.5 - 0.5) < 0.02
    finally:
        plt.close(fig)

    fig, (pathway_axis, budget_axis) = plt.subplots(ncols=2)
    try:
        mixed_frame = frame(2)
        mixed_frame = mixed_frame.drop(
            mixed_frame.index[
                mixed_frame["cc_flow"].eq(CC_FLOW_NEGATIVE) & mixed_frame["cc_model"].eq("M1")
            ]
        )
        budget_handles = plot_budget_panel(
            axis=budget_axis,
            frame=mixed_frame,
            study_years=[2020, 2021],
            post_years=[],
            category_colors={"C1": "#123456"},
            negative_sequestration_style="dotted",
        )
        assert [handle.__class__.__name__ for handle in budget_handles] == [
            "ViolinSummaryLegendHandle",
            "Patch",
        ]
        render_deterministic_legends_below(
            fig,
            pathway_axis=pathway_axis,
            budget_axis=budget_axis,
            pathway_handles=[Line2D([0], [0], color="#123456", label="Category C1")],
            budget_handles=budget_handles,
            pathway_ncol=1,
            budget_ncol=1,
        )
        assert len(fig.legends) == 2
    finally:
        plt.close(fig)


def test_deterministic_ar6_cc_modules_cover_io_render_and_report_edges(
    ar6_dummy_repo,
    tmp_path: Path,
) -> None:
    report = _run_ar6_cc(figures=False, refresh=True)

    assert report is not None
    assert normalize_emission_type(" CO2 ") == "co2"
    assert normalize_emissions_mode("Net") == "net"
    assert (
        cc_variable(emission_type="co2", include_afolu=False, emissions_mode="net")
        == NET_CO2_WO_AFOLU
    )
    assert (
        cc_variable(emission_type="co2", include_afolu=True, emissions_mode="net")
        == NET_CO2_WITH_AFOLU
    )
    assert (
        cc_variable(
            emission_type="kyoto_gases",
            include_afolu=True,
            emissions_mode="net",
        )
        == NET_KYOTO_WITH_AFOLU
    )
    with pytest.raises(ValueError, match="emission_type must be 'kyoto_gases' or 'co2'"):
        normalize_emission_type("methane")

    processed_scope_dir = get_processed_scope_dir(
        report.study_period,
        harmonization=report.harmonization,
        harmonization_method=report.harmonization_method,
    )
    processed_dir = get_processed_dir(
        report.study_period,
        harmonization=report.harmonization,
        harmonization_method=report.harmonization_method,
    )
    assert processed_dir.parent == processed_scope_dir
    assert processed_dir.name == "process_ar6"

    pathways = read_harmonized_pathways(processed_dir=processed_dir, harmonization=True)
    filtered = filter_pathways(
        pathways,
        variable=report.variable,
        category=["C1"],
        ssp_scenario=["SSP1"],
        subset_version=None,
        processed_dir=processed_dir,
    )
    with pytest.raises(ValueError, match="Variable 'missing' not found"):
        filter_pathways(
            pathways,
            variable="missing",
            category=None,
            ssp_scenario=None,
            subset_version=None,
            processed_dir=processed_dir,
        )
    with pytest.raises(FileNotFoundError, match="Model-scenario subset CSV not found"):
        filter_pathways(
            pathways,
            variable=report.variable,
            category=["C1"],
            ssp_scenario=["SSP1"],
            subset_version="missing",
            processed_dir=processed_dir,
        )
    bad_subset_path = get_subset_csv_path(processed_dir, "broken")
    pd.DataFrame([{"model": "M1"}]).to_csv(bad_subset_path, index=False)
    with pytest.raises(ValueError, match="is missing required columns"):
        filter_pathways(
            pathways,
            variable=report.variable,
            category=["C1"],
            ssp_scenario=["SSP1"],
            subset_version="broken",
            processed_dir=processed_dir,
        )
    cc_table = build_cc_table(
        filtered,
        list(range(report.study_period[0], report.study_period[1] + 1)),
        cc_flow=CC_FLOW_POSITIVE,
        cc_variable=report.variable,
    )
    duplicate_filtered = pd.concat([filtered, filtered.iloc[[0]]])
    with pytest.raises(ValueError, match="requires unique deterministic CC rows per identity"):
        build_cc_table(
            duplicate_filtered,
            list(range(report.study_period[0], report.study_period[1] + 1)),
            cc_flow=CC_FLOW_POSITIVE,
            cc_variable=report.variable,
        )
    with pytest.raises(
        ValueError,
        match="merge incoming input requires unique deterministic CC rows per identity",
    ):
        merge_cc_tables(
            existing=cc_table.iloc[0:0], incoming=pd.concat([cc_table, cc_table.iloc[[0]]])
        )
    parquet_path = tmp_path / "ar6_cc.parquet"
    pickle_path = tmp_path / "ar6_cc.pickle"
    empty_path = tmp_path / "empty_scope" / "ar6_cc_empty.csv"
    write_cc_output(cc_table, parquet_path, "parquet")
    write_cc_output(cc_table, pickle_path, "pickle")
    empty_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=cc_table.columns).to_csv(empty_path, index=False)
    assert list(_read_cc_table(parquet_path, "parquet").columns) == [
        str(column) for column in cc_table.columns
    ]
    assert list(_read_cc_table(pickle_path, "pickle").columns) == list(cc_table.columns)
    assert read_cc_output(output_file=empty_path, output_format="csv").empty
    missing_ssp_path = tmp_path / "missing_ssp.csv"
    cc_table.drop(columns=["ssp_scenario"]).to_csv(missing_ssp_path, index=False)
    with pytest.raises(ValueError, match="missing required columns: \\['ssp_scenario'\\]"):
        read_cc_output(output_file=missing_ssp_path, output_format="csv")
    missing_scenario_path = tmp_path / "missing_scenario.csv"
    cc_table.drop(columns=["cc_scenario"]).to_csv(missing_scenario_path, index=False)
    with pytest.raises(ValueError, match="missing required columns: \\['cc_scenario'\\]"):
        read_cc_output(output_file=missing_scenario_path, output_format="csv")
    empty_unit_path = tmp_path / "empty_unit.csv"
    empty_unit = cc_table.copy()
    empty_unit["impact_unit"] = ""
    empty_unit.to_csv(empty_unit_path, index=False)
    with pytest.raises(ValueError, match="empty values in required column 'impact_unit'"):
        read_cc_output(output_file=empty_unit_path, output_format="csv")
    with pytest.raises(ValueError, match="Unsupported output_format 'txt'"):
        write_cc_output(cc_table, tmp_path / "ar6_cc.txt", "txt")

    assert (
        render_cc_pathway_figures(
            cc_table=pd.DataFrame(),
            variable_name=report.variable,
            output_dir=tmp_path / "empty_figures",
            dpi=10,
            output_format="png",
        )
        == []
    )
    string_year_paths = render_cc_pathway_figures(
        cc_table=pd.DataFrame(
            [
                {
                    "cc_category": "C1",
                    "ssp_scenario": "SSP1",
                    "cc_model": "M1",
                    "cc_scenario": "S1",
                    "cc_flow": CC_FLOW_NET,
                    "cc_variable": report.variable,
                    "impact_unit": "Gt",
                    "2019": 1.0,
                }
            ]
        ),
        variable_name=report.variable,
        output_dir=tmp_path / "string_years",
        dpi=10,
        output_format="png",
        requested_years=[2019],
    )
    assert len(string_year_paths) == 1
    assert string_year_paths[0].exists()
    process_ar6_payload = {
        "reuse_status": "reused_exact",
        "study_period": "2019-2021",
        "categories": ["C1"],
        "ssps": [1],
        "harmonization": True,
        "harmonization_method": "offset",
        "harmonization_year_message": "Warning message.",
        "output_root": str(tmp_path / "process_ar6"),
        "variable_coverage": [],
    }
    payload = build_run_metadata_payload(
        signature={"study_period": report.study_period},
        identity_payload={"study_period": report.study_period},
        coverage={"cc_category": ["C1"], "ssp_scenario": ["SSP1"]},
        write_scope_identity={"study_period": report.study_period},
        emission_type="kyoto_gases",
        include_afolu=False,
        emissions_mode=report.emissions_mode,
        cc_categories=["C1"],
        ssp_scenarios=["SSP1"],
        total_model_scenario_pairs=1,
        pathway_counts=[
            AR6CCPathwayCount(category="C1", ssp_scenario="SSP1", model_scenario_pairs=1)
        ],
        missing_pathway_combinations=[],
        output_file=tmp_path / "ar6_cc.csv",
        process_ar6=process_ar6_payload,
    )
    figure_path = tmp_path / "metadata_figures" / "dynamic_cc_pathways.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_bytes(b"png")
    set_figure_state(
        payload=payload,
        request_signature={"figure_output_format": "png"},
        compute_signature={
            "identity_key": payload["reuse"]["identity_key"],
            "coverage": {"cc_category": ["C1"], "ssp_scenario": ["SSP1"]},
        },
        paths=[figure_path],
    )
    assert payload["artifacts"]["figure_paths"] == [str(figure_path)]
    assert (
        figure_state_matches(
            payload=payload,
            request_signature={"figure_output_format": "png"},
            compute_signature={
                "identity_key": payload["reuse"]["identity_key"],
                "coverage": {"cc_category": ["C1"], "ssp_scenario": ["SSP1"]},
            },
        )
        is True
    )
    assert (
        figure_state_matches(
            payload=payload,
            request_signature={"figure_output_format": "svg"},
            compute_signature={
                "identity_key": payload["reuse"]["identity_key"],
                "coverage": {"cc_category": ["C1"], "ssp_scenario": ["SSP1"]},
            },
        )
        is False
    )

    custom_report = ComputeAR6CCReport(
        study_period=[2019, 2021],
        harmonization=True,
        harmonization_method="offset",
        emission_type="kyoto_gases",
        include_afolu=False,
        emissions_mode=report.emissions_mode,
        variable=report.variable,
        categories=["C1"],
        ssp_scenarios=["SSP1"],
        subset_version="subset_a",
        total_model_scenario_pairs=1,
        output_file=tmp_path / "ar6_cc.csv",
        figure_paths=[figure_path],
        meta_file=tmp_path / "scope_manifest.json",
        cc_dir=tmp_path / "cc_scope",
        logs_dir=tmp_path / "logs",
        process_ar6=process_ar6_payload,
    )
    summary_text = str(custom_report)
    assert summary_text
    minimal_report = ComputeAR6CCReport(
        study_period=[2019, 2021],
        harmonization=True,
        harmonization_method="offset",
        emission_type="kyoto_gases",
        include_afolu=False,
        emissions_mode=report.emissions_mode,
        variable=report.variable,
        categories=["C1"],
        ssp_scenarios=["SSP1"],
        subset_version=None,
        total_model_scenario_pairs=1,
        output_file=tmp_path / "results" / "ar6_cc.csv",
        process_ar6=process_ar6_payload,
    )
    minimal_text = str(minimal_report)
    assert minimal_text

    assert report.cc_dir is not None
    assert get_cc_metadata_path(cc_dir=report.cc_dir).name == "scope_manifest.json"
    assert bad_subset_path.name == "model_scenario_subset__broken.csv"
    assert ar6_dummy_repo.repo_root.exists()


def _write_minimal_harmonized_ar6_workbook_for_cc(*, study_period: list[int]) -> None:
    processed_dir = get_processed_dir(
        study_period,
        harmonization=True,
        harmonization_method="offset",
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "model": ["M2100"],
            "scenario": ["S2100"],
            "variable": [NET_KYOTO_WO_AFOLU],
            "Category": ["C1"],
            "Ssp_family": [1],
            "unit": ["MtCO2eq/yr"],
            2099: [4.0],
            2100: [3.0],
        }
    ).set_index(["model", "scenario", "variable"])
    workbook = processed_dir / "harmonized_ar6_public.xlsx"
    with pd.ExcelWriter(workbook, engine="xlsxwriter") as writer:
        frame.to_excel(writer, sheet_name="HARMONIZED_AR6", merge_cells=False)
    logs_dir = get_logs_dir(
        study_period,
        harmonization=True,
        harmonization_method="offset",
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        logs_dir / "scope_manifest.json",
        build_process_metadata_payload(
            signature={
                "study_period": study_period,
                "harmonization": True,
                "harmonization_method": "offset",
            },
            categories=["C1"],
            ssps=[1],
            harmonization=True,
            harmonization_method="offset",
            latest_historical_year=None,
            requested_harmonization_year=None,
            harmonization_year=None,
            harmonization_message=None,
            processed_dir=processed_dir,
            logs_dir=logs_dir,
            figures_dir=processed_dir / "figures",
            output_file=workbook,
            log_file=None,
            dropped_rows_csv_file=logs_dir / "dropped_model_scenario_variable_rows.csv",
            variable_coverage_summary_counts={
                NET_KYOTO_WO_AFOLU: {
                    "available_model_scenario_pairs": 1,
                    "retained_model_scenario_pairs": 1,
                    "missing_reason_counts": {},
                }
            },
        ),
    )


def test_deterministic_ar6_cc_rejects_stale_complete_scope_with_missing_outputs(
    ar6_dummy_repo,
) -> None:
    del ar6_dummy_repo
    report = _run_ar6_cc(figures=False, refresh=True)

    assert report is not None
    assert report.cc_dir is not None
    figures_dir = report.cc_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    stale_figure = figures_dir / "stale.png"
    stale_figure.write_bytes(b"stale")
    report.output_file.unlink()

    with pytest.raises(ValueError, match="output files are missing"):
        _run_ar6_cc(figures=True, refresh=False)
    assert stale_figure.exists()
