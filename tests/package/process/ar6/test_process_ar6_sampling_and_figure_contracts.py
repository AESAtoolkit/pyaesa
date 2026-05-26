from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tests.package.helpers.ar6_imports import (
    collection_explorer,
    collection_config,
    processing_derived_variables,
    processing_fig_guides,
    processing_fig_outputs,
    processing_fig_overview,
    processing_fig_sampling_config,
    processing_fig_sampling_panels,
    processing_fig_warming,
    processing_generate_figures,
    processing_plot_budgets,
    processing_plot_helpers,
    processing_plot_sampling,
    processing_preprocessing,
    processing_sampling_convergence,
    processing_sampling_payloads,
)
from tests.package.helpers.ar6_dummy_repo import AR6DummyRepo

read_explorer_csv = collection_explorer.read_explorer_csv
NET_CO2_WITH_AFOLU = collection_config.NET_CO2_WITH_AFOLU
NET_CO2_WO_AFOLU = collection_config.NET_CO2_WO_AFOLU
NET_KYOTO_WITH_AFOLU = collection_config.NET_KYOTO_WITH_AFOLU
NET_KYOTO_WO_AFOLU = collection_config.NET_KYOTO_WO_AFOLU
GROSS_CO2_WITH_AFOLU = collection_config.GROSS_CO2_WITH_AFOLU
GROSS_ALT_CO2_WITH_AFOLU = collection_config.GROSS_ALT_CO2_WITH_AFOLU
GROSS_KYOTO_WITH_AFOLU = collection_config.GROSS_KYOTO_WITH_AFOLU
GROSS_ALT_KYOTO_WITH_AFOLU = collection_config.GROSS_ALT_KYOTO_WITH_AFOLU
RAW_CO2_WITH_AFOLU = collection_config.RAW_CO2_WITH_AFOLU
RAW_KYOTO_WITH_AFOLU = collection_config.RAW_KYOTO_WITH_AFOLU
SEQUESTRATION_TOTAL = collection_config.SEQUESTRATION_TOTAL
SEQUESTRATION_SUBTOTAL = collection_config.SEQUESTRATION_SUBTOTAL
RAW_SEQUESTRATION_COMPONENTS = collection_config.RAW_SEQUESTRATION_COMPONENTS
_figure_explanation_block = processing_fig_guides._figure_explanation_block
ensure_figures_guide = processing_fig_guides.ensure_figures_guide
figures_explanation_text = processing_fig_guides.figures_explanation_text
write_figures_guide = processing_fig_guides.write_figures_guide
ensure_figures = processing_fig_outputs.ensure_figures
figure_signature = processing_fig_outputs.figure_signature
load_saved_figure_files = processing_fig_outputs.load_saved_figure_files
write_figure_metadata = processing_fig_outputs.write_figure_metadata
write_delta_tconv_figure = processing_fig_overview.write_delta_tconv_figure
write_harmonization_stats_figure = processing_fig_overview.write_harmonization_stats_figure
write_processed_budgets_figure = processing_fig_overview.write_processed_budgets_figure
write_sequestration_budgets_figure = processing_fig_overview.write_sequestration_budgets_figure
normalize_sampling_figure_config = processing_fig_sampling_config.validate_sampling_figure_config
_generate_sampling_budget_figure = processing_fig_sampling_panels._generate_sampling_budget_figure
_subsample_sampled_index = processing_fig_sampling_panels._subsample_sampled_index
_VIOLIN_MAX_SAMPLES = processing_fig_sampling_panels._VIOLIN_MAX_SAMPLES
_write_sampling_median_ratio_figure = (
    processing_fig_sampling_panels._write_sampling_median_ratio_figure
)
_write_sampling_probability_ratio_figure = (
    processing_fig_sampling_panels._write_sampling_probability_ratio_figure
)
WARMING_METADATA_COLUMN = processing_fig_warming.WARMING_METADATA_COLUMN
write_median_warming_figure = processing_fig_warming.write_median_warming_figure
_figure_status = processing_generate_figures._figure_status
generate_ar6_figures = processing_generate_figures.generate_ar6_figures
plot_budgets_summary = processing_plot_budgets.plot_budgets_summary
plot_pathways = processing_plot_budgets.plot_pathways
_metadata_scalar_for_index = processing_plot_helpers._metadata_scalar_for_index
_require_row_series = processing_plot_helpers._require_row_series
historical_series = processing_plot_helpers.historical_series
max_year = processing_plot_helpers.max_year
numeric_year_columns = processing_plot_helpers.numeric_year_columns
plot_violin = processing_plot_helpers.plot_violin
remaining_budget_end_year = processing_plot_helpers.remaining_budget_end_year
scenario_df_from_harmonized = processing_plot_helpers.scenario_df_from_harmonized
_sample_variable_until_converged = processing_plot_sampling._sample_variable_until_converged
_ssp_family_sort_rank = processing_plot_sampling._ssp_family_sort_rank
build_sampling_runs_until_convergence = (
    processing_plot_sampling.build_sampling_runs_until_convergence
)
build_sampling_probability_df = processing_plot_sampling.build_sampling_probability_df
_grouped_row_sort_key = processing_sampling_convergence._grouped_row_sort_key
distribution_stats_from_counts = processing_sampling_convergence.distribution_stats_from_counts
flatten_sampled_index_from_counts = (
    processing_sampling_convergence.flatten_sampled_index_from_counts
)
sampling_seed = processing_sampling_convergence.sampling_seed
snapshot_to_log_rows = processing_sampling_convergence.snapshot_to_log_rows
snapshots_are_stable = processing_sampling_convergence.snapshots_are_stable
study_rows_to_frame = processing_sampling_convergence.study_rows_to_frame
build_snapshot_from_counts = processing_sampling_payloads.build_snapshot_from_counts
build_variable_payload = processing_sampling_payloads.build_variable_payload
expected_snapshot_key_count = processing_sampling_payloads.expected_snapshot_key_count
filter_and_format_rawdata = processing_preprocessing.filter_and_format_rawdata
build_pre_harmonization_variables = processing_derived_variables.build_pre_harmonization_variables


def test_ar6_sampling_config_and_guide_templates(tmp_path: Path) -> None:
    config = normalize_sampling_figure_config(
        figure_convergence_tol=0.05,
        figure_convergence_max_runs=12345,
    )
    assert config["relative_tolerance"] == 0.05
    assert config["max_runs_per_bucket"] == 12345
    with pytest.raises(ValueError):
        normalize_sampling_figure_config(
            figure_convergence_tol=0.0,
            figure_convergence_max_runs=10,
        )
    with pytest.raises(ValueError):
        normalize_sampling_figure_config(
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=0,
        )
    assert (
        processing_fig_sampling_config.minimum_completed_runs_per_bucket_for_convergence(
            run_batch_size=10,
            stable_checks_required=2,
        )
        == 30
    )

    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "fig-budgets-CO2-demo-remaining-budget-panel-dropped_rows.csv").write_text(
        "header\n",
        encoding="utf-8",
    )
    (
        figures_dir
        / "fig-sequestration-budgets-forCO2-demo-remaining-budget-panel-dropped_rows.csv"
    ).write_text(
        "header\n",
        encoding="utf-8",
    )
    (
        figures_dir / "fig-LHSSRS-budgets-CO2-net-demo-remaining-budget-panel-dropped_rows.csv"
    ).write_text(
        "header\n",
        encoding="utf-8",
    )
    global_drop_csv = tmp_path / "global_drop.csv"
    figure_names = [
        "fig-processed-historical-emissions.png",
        "fig-harmonization-pathways-demo-studyperiod=2019to2060.png",
        "fig-harmonization-stats-delta-tconv-demo-studyperiod=2019to2060.png",
        "fig-harmonization-stats-demo.png",
        "fig-budgets-CO2-demo.png",
        "fig-budgets-GHG-demo.png",
        "fig-sequestration-contributions-demo.png",
        "fig-sequestration-budgets-forCO2-demo.png",
        "fig-sequestration-budgets-forGHG-demo.png",
        "fig-median-warming-demo-studyperiod=2019to2060.png",
        "fig-LHSSRS-ratioproba-CO2-demo.png",
        "fig-LHSSRS-ratioproba-GHG-demo.png",
        "fig-LHSSRS-ratiomedian-CO2-demo.png",
        "fig-LHSSRS-ratiomedian-GHG-demo.png",
        "fig-LHSSRS-budgets-GHG-net-demo.png",
        "fig-LHSSRS-budgets-CO2-net-demo.png",
    ]
    for figure_name in figure_names:
        block = _figure_explanation_block(
            figure_name,
            [2019, 2060],
            global_drop_csv,
            {
                "fig-budgets-CO2-demo": [
                    "fig-budgets-CO2-demo-remaining-budget-panel-dropped_rows.csv"
                ],
                "fig-sequestration-budgets-forCO2-demo": [
                    "fig-sequestration-budgets-forCO2-demo-remaining-budget-panel-dropped_rows.csv"
                ],
                "fig-LHSSRS-budgets-CO2-net-demo": [
                    "fig-LHSSRS-budgets-CO2-net-demo-remaining-budget-panel-dropped_rows.csv"
                ],
            },
        )
        assert any(figure_name in line for line in block)
        assert len(block) > 1

    text = figures_explanation_text(figure_names, [2019, 2060], figures_dir, global_drop_csv)
    assert text.strip()
    assert all(len(line) <= 100 for line in text.splitlines())
    unknown_text = figures_explanation_text(
        ["fig-unknown-template-demo.png"],
        [2019, 2060],
        figures_dir,
        global_drop_csv,
    )
    assert unknown_text.strip()
    assert all(len(line) <= 100 for line in unknown_text.splitlines())
    missing_file, missing_written = ensure_figures_guide(
        figures_dir=figures_dir / "empty",
        figure_files=[],
        study_period=[2019, 2060],
        global_drop_csv_file=global_drop_csv,
        rewrite=False,
    )
    assert missing_file is None
    assert missing_written is False
    guide_file = write_figures_guide(
        figures_dir=figures_dir,
        figure_files=figure_names,
        study_period=[2019, 2060],
        global_drop_csv_file=global_drop_csv,
    )
    assert guide_file.exists()
    assert all(len(line) <= 100 for line in guide_file.read_text(encoding="utf-8").splitlines())
    rewritten_file, rewritten_written = ensure_figures_guide(
        figures_dir=figures_dir,
        figure_files=figure_names,
        study_period=[2019, 2060],
        global_drop_csv_file=global_drop_csv,
        rewrite=True,
    )
    assert rewritten_written is True
    assert rewritten_file == guide_file
    reused_file, reused_written = ensure_figures_guide(
        figures_dir=figures_dir,
        figure_files=figure_names,
        study_period=[2019, 2060],
        global_drop_csv_file=global_drop_csv,
        rewrite=False,
    )
    assert reused_written is False
    assert reused_file == guide_file


def test_ar6_figure_output_reuse_contracts(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
    ar6_processed_pathway_outputs: dict[str, pd.DataFrame],
) -> None:
    inputs = ar6_processed_pathway_outputs
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    metadata_path = tmp_path / "figures_meta.json"
    figure_path = figures_dir / "figure.png"
    figure_path.write_text("png", encoding="utf-8")
    csv_path = logs_dir / "sampling.csv"
    txt_path = logs_dir / "sampling.txt"
    csv_path.write_text("a\n", encoding="utf-8")
    txt_path.write_text("b", encoding="utf-8")
    sig = figure_signature(
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1"],
        variables_output=[NET_CO2_WITH_AFOLU],
        figure_output_format="png",
        figure_dpi=10,
        harmonization_method="reduced_offset",
        figure_convergence_tol=0.1,
        figure_convergence_max_runs=20000,
    )

    write_figure_metadata(
        figures_metadata_file=metadata_path,
        signature=sig,
        figure_files=[str(figure_path)],
        generation_complete=False,
        sampling_log_csv_file=csv_path,
        sampling_log_columns_txt_file=txt_path,
    )
    assert (
        load_saved_figure_files(
            figures_metadata_file=metadata_path,
            figures_dir=figures_dir,
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[NET_CO2_WITH_AFOLU],
            figure_output_format="png",
            figure_dpi=10,
            harmonization_method="reduced_offset",
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )
        is None
    )

    write_figure_metadata(
        figures_metadata_file=metadata_path,
        signature=sig,
        figure_files=[str(figure_path)],
        generation_complete=True,
        sampling_log_csv_file=csv_path,
        sampling_log_columns_txt_file=txt_path,
    )
    assert load_saved_figure_files(
        figures_metadata_file=metadata_path,
        figures_dir=figures_dir,
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1"],
        variables_output=[NET_CO2_WITH_AFOLU],
        figure_output_format="png",
        figure_dpi=10,
        harmonization_method="reduced_offset",
        figure_convergence_tol=0.1,
        figure_convergence_max_runs=20000,
    ) == [str(figure_path)]

    txt_path.unlink()
    assert (
        load_saved_figure_files(
            figures_metadata_file=metadata_path,
            figures_dir=figures_dir,
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[NET_CO2_WITH_AFOLU],
            figure_output_format="png",
            figure_dpi=10,
            harmonization_method="reduced_offset",
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )
        is None
    )
    txt_path.write_text("b", encoding="utf-8")
    csv_path.unlink()
    assert (
        load_saved_figure_files(
            figures_metadata_file=metadata_path,
            figures_dir=figures_dir,
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[NET_CO2_WITH_AFOLU],
            figure_output_format="png",
            figure_dpi=10,
            harmonization_method="reduced_offset",
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )
        is None
    )
    csv_path.write_text("a\n", encoding="utf-8")
    figure_path.unlink()
    assert (
        load_saved_figure_files(
            figures_metadata_file=metadata_path,
            figures_dir=figures_dir,
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[NET_CO2_WITH_AFOLU],
            figure_output_format="png",
            figure_dpi=10,
            harmonization_method="reduced_offset",
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )
        is None
    )
    figure_path.write_text("png", encoding="utf-8")
    assert (
        load_saved_figure_files(
            figures_metadata_file=metadata_path,
            figures_dir=figures_dir,
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[NET_CO2_WITH_AFOLU],
            figure_output_format="png",
            figure_dpi=10,
            harmonization_method="constant_offset",
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )
        is None
    )

    workbook_path = tmp_path / "harmonized.xlsx"
    log_path = tmp_path / "harmonization_log.xlsx"
    harmonized_data = inputs["harmonized_data"].loc[
        (slice(None), slice(None), NET_CO2_WITH_AFOLU), :
    ]
    original_data = inputs["original_data"].loc[harmonized_data.index, :]
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        harmonized_data.to_excel(writer, sheet_name="HARMONIZED_AR6", merge_cells=False)
        original_data.to_excel(writer, sheet_name="ORIGINAL_AR6", merge_cells=False)
        inputs["historical_data"].to_excel(
            writer, sheet_name="HISTORICAL_PRIMAP_GCP", merge_cells=False
        )
    with pd.ExcelWriter(log_path, engine="xlsxwriter") as writer:
        inputs["harmonization_log"].loc[harmonized_data.index, :].to_excel(
            writer,
            sheet_name="HARMONIZATION_LOG",
            merge_cells=False,
        )

    reused_files, reused = ensure_figures(
        out_file=workbook_path,
        log_file=log_path,
        figures_dir=figures_dir,
        figures_metadata_file=metadata_path,
        logs_dir=logs_dir,
        variables_output=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1"],
        figure_output_format="png",
        figure_dpi=10,
        harmonization_method="reduced_offset",
        figure_convergence_tol=0.1,
        figure_convergence_max_runs=20000,
        refresh=False,
        source_metadata=inputs["source_metadata"],
        raw_data_dir=ar6_dummy_repo.raw_dir,
    )
    assert reused is True
    assert reused_files == [str(figure_path)]


def test_ar6_sampling_contract_branches(
    ar6_dummy_repo: AR6DummyRepo,
    ar6_processed_pathway_outputs: dict[str, pd.DataFrame],
) -> None:
    del ar6_dummy_repo
    inputs = ar6_processed_pathway_outputs
    harmonized_data = inputs["harmonized_data"]
    co2_data = harmonized_data.loc[(slice(None), slice(None), NET_CO2_WITH_AFOLU), :].sort_index()
    probabilities = build_sampling_probability_df(
        co2_data,
        [NET_CO2_WITH_AFOLU],
        ["C1", "C2", "C3", "C4"],
    )
    assert probabilities["proba_SRS"].notna().any()

    payload_no_remaining = build_variable_payload(
        harmonized_data=co2_data,
        tmp_proba_df=probabilities,
        var_sel=NET_CO2_WITH_AFOLU,
        categories=["C1", "C2", "C3", "C4"],
        study_period=[2019, 2100],
        remaining_budget_end_year_value=2100,
        sampling_method="SRS",
    )
    assert all(
        not np.isfinite(np.asarray(bucket["remaining_values"], dtype=float)).any()
        for bucket in payload_no_remaining["buckets"]
    )

    payload_bad_end = build_variable_payload(
        harmonized_data=co2_data,
        tmp_proba_df=probabilities,
        var_sel=NET_CO2_WITH_AFOLU,
        categories=["C1", "C2", "C3", "C4"],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2200,
        sampling_method="SRS",
    )
    assert all(
        not np.isfinite(np.asarray(bucket["remaining_values"], dtype=float)).any()
        for bucket in payload_bad_end["buckets"]
    )

    nan_end_data = co2_data.copy()
    nan_end_data.loc[:, 2100] = np.nan
    payload_nan_end = build_variable_payload(
        harmonized_data=nan_end_data,
        tmp_proba_df=probabilities,
        var_sel=NET_CO2_WITH_AFOLU,
        categories=["C1", "C2", "C3", "C4"],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        sampling_method="SRS",
    )
    assert all(
        not np.isfinite(np.asarray(bucket["remaining_values"], dtype=float)).any()
        for bucket in payload_nan_end["buckets"]
    )

    zero_probabilities = probabilities.copy()
    zero_probabilities["proba_SRS"] = 0.0
    payload_zero_prob = build_variable_payload(
        harmonized_data=co2_data,
        tmp_proba_df=zero_probabilities,
        var_sel=NET_CO2_WITH_AFOLU,
        categories=["C1", "C2", "C3", "C4"],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        sampling_method="SRS",
    )
    assert payload_zero_prob["buckets"] == []
    assert expected_snapshot_key_count(payload_no_remaining["buckets"]) > 0

    empty_snapshot, empty_study_rows = build_snapshot_from_counts(
        sampled_counts={
            bucket["key"]: np.zeros(len(bucket["positions"]), dtype=np.int64)
            for bucket in payload_no_remaining["buckets"]
        },
        payload=payload_no_remaining,
        var_sel=NET_CO2_WITH_AFOLU,
    )
    assert empty_snapshot == {}
    assert empty_study_rows == []

    stats_single = distribution_stats_from_counts(np.array([5.0]), np.array([1], dtype=np.int64))
    assert stats_single == {
        "mean": 5.0,
        "median": 5.0,
        "p25": 5.0,
        "p75": 5.0,
        "p5": 5.0,
        "p95": 5.0,
    }
    assert study_rows_to_frame([], ("mean", "median")).empty
    assert (
        snapshots_are_stable({}, {("study", "C1", 1, "mean"): 1.0}, relative_tolerance=0.1) is False
    )
    assert (
        snapshots_are_stable(
            {("study", "C1", 1, "mean"): np.nan},
            {("study", "C1", 1, "mean"): 1.0},
            relative_tolerance=0.1,
        )
        is False
    )
    assert (
        snapshots_are_stable(
            {("study", "C1", 1, "mean"): 0.0},
            {("study", "C1", 1, "mean"): 0.0},
            relative_tolerance=0.1,
        )
        is True
    )

    buckets = [{"key": ("C1", 1), "labels": [("m", "s", "v"), ("m2", "s2", "v2")]}]
    assert (
        flatten_sampled_index_from_counts(
            buckets,
            {("C1", 1): np.array([0, 0], dtype=np.int64)},
        )
        == []
    )
    assert flatten_sampled_index_from_counts(
        buckets,
        {("C1", 1): np.array([1, 0], dtype=np.int64)},
    ) == [("m", "s", "v")]
    assert sampling_seed(NET_CO2_WITH_AFOLU, "SRS") == sampling_seed(NET_CO2_WITH_AFOLU, "SRS")

    snapshot = {
        ("study", "C1", 1, "mean"): 1.0,
        ("study", "C1", "2", "mean"): 2.0,
        ("study", "C1", "all", "mean"): 3.0,
        ("study", "C1", "sspX", "mean"): 4.0,
    }
    log_rows = snapshot_to_log_rows(
        snapshot,
        variable=NET_CO2_WITH_AFOLU,
        sampling_method="SRS",
        rng_seed_value=1,
        final_runs_per_bucket=100,
        run_batch_size=10,
        maximum_runs_per_bucket=200,
        relative_tolerance=0.1,
        stable_checks_required=1,
    )
    assert [row["ssp_family"] for row in log_rows] == [1, "2", "all", "sspX"]
    assert _grouped_row_sort_key(("study", "C1", 2.0))[2] == (0, 2, "")

    assert _ssp_family_sort_rank(None) == float("inf")
    assert _ssp_family_sort_rank(np.nan) == float("inf")
    assert _ssp_family_sort_rank(2.0) == 2.0

    empty_result = build_sampling_runs_until_convergence(
        harmonized_data=co2_data,
        tmp_proba_df=probabilities,
        all_variables_l=[NET_CO2_WITH_AFOLU],
        categories=["C9"],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        relative_tolerance=0.1,
        max_runs_per_bucket=10000,
        run_batch_size=10000,
        stable_checks_required=1,
    )
    assert empty_result[0] == {NET_CO2_WITH_AFOLU: []}
    assert empty_result[1] == {NET_CO2_WITH_AFOLU: []}
    assert empty_result[2].empty
    assert empty_result[3].empty

    with pytest.raises(RuntimeError):
        build_sampling_runs_until_convergence(
            harmonized_data=co2_data,
            tmp_proba_df=probabilities,
            all_variables_l=[NET_CO2_WITH_AFOLU],
            categories=["C1", "C2", "C3", "C4"],
            study_period=[2019, 2060],
            remaining_budget_end_year_value=2100,
            relative_tolerance=1e-12,
            max_runs_per_bucket=10000,
            run_batch_size=10000,
            stable_checks_required=1,
        )

    incomplete_data = co2_data.loc[
        [
            ("M1", "S1", NET_CO2_WITH_AFOLU),
            ("M2", "S2", NET_CO2_WITH_AFOLU),
        ]
    ].copy()
    incomplete_data.loc[("M2", "S2", NET_CO2_WITH_AFOLU), "Category"] = "C1"
    incomplete_data.loc[("M2", "S2", NET_CO2_WITH_AFOLU), "Ssp_family"] = 1
    incomplete_data.loc[("M2", "S2", NET_CO2_WITH_AFOLU), 2100] = np.nan
    incomplete_probabilities = pd.DataFrame(
        {
            "Category": ["C1", "C1"],
            "Ssp_family": [1, 1],
            "proba_SRS": [0.0, 1.0],
            "proba_LHS": [0.0, 1.0],
        },
        index=incomplete_data.index,
    )
    with pytest.raises(RuntimeError):
        _sample_variable_until_converged(
            harmonized_data=incomplete_data,
            tmp_proba_df=incomplete_probabilities,
            var_sel=NET_CO2_WITH_AFOLU,
            categories=["C1"],
            study_period=[2019, 2060],
            remaining_budget_end_year_value=2100,
            sampling_method="SRS",
            relative_tolerance=1.0,
            max_runs_per_bucket=2,
            run_batch_size=1,
            stable_checks_required=1,
        )

    stable_messages: list[str] = []
    variable_result = _sample_variable_until_converged(
        harmonized_data=co2_data,
        tmp_proba_df=probabilities,
        var_sel=NET_CO2_WITH_AFOLU,
        categories=["C1", "C2", "C3", "C4"],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        sampling_method="SRS",
        relative_tolerance=1.0,
        max_runs_per_bucket=40,
        status_callback=stable_messages.append,
        run_batch_size=1,
        stable_checks_required=2,
    )
    assert variable_result["sampled_index"]
    assert any("stable 1/2" in message for message in stable_messages)


def test_ar6_generate_overview_and_warming_branches(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
    ar6_processed_pathway_outputs: dict[str, pd.DataFrame],
) -> None:
    inputs = ar6_processed_pathway_outputs
    harmonized_data = inputs["harmonized_data"]
    source_metadata = inputs["source_metadata"]
    harmonized_scenarios = scenario_df_from_harmonized(harmonized_data)

    write_median_warming_figure(
        figures_dir=tmp_path,
        ext="svg",
        dpi=1,
        out_paths=[],
        scenario_rows_df=harmonized_scenarios,
        source_metadata=source_metadata.assign(
            **{
                WARMING_METADATA_COLUMN: source_metadata[WARMING_METADATA_COLUMN].where(
                    source_metadata["Category"] != "C2"
                )
            }
        ),
        categories=["C1", "C2", "C9"],
        study_period=[2019, 2060],
        database="ar6-public",
        categories_repr="['C1', 'C2', 'C9']",
    )
    assert (
        tmp_path / "fig-median-warming-ar6-public-MOD=ALL-CAT=['C1', 'C2', 'C9']"
        "-studyperiod=2019to2060.svg"
    ).exists()

    zero_delta_log = inputs["harmonization_log"].copy()
    zero_delta_log["horizon-for-harmonization"] = zero_delta_log["model-netzero-year"]
    out_paths: list[str] = []
    write_delta_tconv_figure(
        figures_dir=tmp_path,
        ext="svg",
        dpi=1,
        out_paths=out_paths,
        harmonization_log=zero_delta_log,
        database="ar6-public",
        categories_repr="['C1']",
        study_period=[2019, 2060],
    )
    assert out_paths

    positive_delta_log = inputs["harmonization_log"].copy()
    positive_delta_log.iloc[
        0,
        positive_delta_log.columns.get_loc("model-netzero-year"),
    ] = 2050.0
    positive_delta_log.iloc[
        0,
        positive_delta_log.columns.get_loc("horizon-for-harmonization"),
    ] = 2055.0
    write_delta_tconv_figure(
        figures_dir=tmp_path,
        ext="svg",
        dpi=1,
        out_paths=[],
        harmonization_log=positive_delta_log,
        database="ar6-public",
        categories_repr="['C1']",
        study_period=[2019, 2060],
    )

    nan_stats_log = inputs["harmonization_log"].copy()
    nan_stats_log["pathway-cumulative"] = np.nan
    nan_stats_log["historic-cumulative"] = np.nan
    nan_stats_log["yearly-correction"] = np.nan
    write_harmonization_stats_figure(
        figures_dir=tmp_path,
        ext="svg",
        dpi=1,
        out_paths=[],
        harmonization_log=nan_stats_log,
        all_variables_l=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database="ar6-public",
        categories_repr="['C1']",
    )
    assert (
        tmp_path
        / "fig-harmonization-stats-ar6-public-MOD=ALL-CAT=['C1']-studyperiod=2019to2060.svg"
    ).exists()

    write_processed_budgets_figure(
        figures_dir=tmp_path,
        ext="svg",
        dpi=1,
        out_paths=[],
        harmonized_data=harmonized_data,
        historical_data=inputs["historical_data"],
        all_variables_l=[NET_KYOTO_WO_AFOLU],
        study_period=[2010, 2100],
        remaining_budget_end_year_value=2100,
        database="ar6-public",
        categories_repr="['C1']",
    )
    assert (
        tmp_path / "fig-budgets-GHG-ar6-public-MOD=ALL-CAT=['C1']-studyperiod=2010to2100.svg"
    ).exists()

    write_sequestration_budgets_figure(
        figures_dir=tmp_path,
        ext="svg",
        dpi=1,
        out_paths=[],
        harmonized_data=harmonized_data,
        all_variables_l=[
            NET_CO2_WITH_AFOLU,
            GROSS_ALT_CO2_WITH_AFOLU,
            NET_KYOTO_WITH_AFOLU,
            GROSS_ALT_KYOTO_WITH_AFOLU,
        ],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        database="ar6-public",
        categories_repr="['C1']",
    )
    assert (
        tmp_path / "fig-sequestration-budgets-forCO2-ar6-public-MOD=ALL-CAT=['C1']"
        "-studyperiod=2019to2060.svg"
    ).exists()
    assert (
        tmp_path / "fig-sequestration-budgets-forGHG-ar6-public-MOD=ALL-CAT=['C1']"
        "-studyperiod=2019to2060.svg"
    ).exists()

    harmonized_vars = harmonized_data.index.get_level_values("variable")
    gross_figure_vars = [NET_CO2_WITH_AFOLU, GROSS_CO2_WITH_AFOLU]
    gross_figure_data = harmonized_data.loc[
        harmonized_vars.isin([*gross_figure_vars, SEQUESTRATION_TOTAL, SEQUESTRATION_SUBTOTAL]), :
    ]
    original_vars = inputs["original_data"].index.get_level_values("variable")
    gross_original_vars = [
        NET_CO2_WITH_AFOLU,
        SEQUESTRATION_TOTAL,
        SEQUESTRATION_SUBTOTAL,
        *RAW_SEQUESTRATION_COMPONENTS,
    ]
    gross_original_data = inputs["original_data"].loc[original_vars.isin(gross_original_vars), :]
    gross_harmonization_log = inputs["harmonization_log"].loc[
        (slice(None), slice(None), NET_CO2_WITH_AFOLU), :
    ]

    def fake_sampling_generate_only(**kwargs):
        return pd.DataFrame()

    gross_paths, gross_convergence = generate_ar6_figures(
        output_dir=str(tmp_path / "gross_generated"),
        harmonized_data=gross_figure_data,
        original_data=gross_original_data,
        harmonization_log=gross_harmonization_log,
        historical_data=inputs["historical_data"],
        source_metadata=source_metadata,
        raw_data_dir=str(ar6_dummy_repo.raw_dir),
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1", "C2", "C3", "C4"],
        variables_output=gross_figure_vars,
        figure_output_format="svg",
        dpi=1,
        figure_convergence_tol=0.1,
        figure_convergence_max_runs=20000,
        write_sampling_figures_func=fake_sampling_generate_only,
    )
    assert gross_convergence.empty
    assert any("fig-sequestration-contributions" in path for path in gross_paths)
    assert any("fig-sequestration-budgets-forCO2" in path for path in gross_paths)

    with pytest.raises(RuntimeError):
        generate_ar6_figures(
            output_dir=str(tmp_path / "missing_processed"),
            harmonized_data=harmonized_data,
            original_data=inputs["original_data"],
            harmonization_log=inputs["harmonization_log"],
            historical_data=inputs["historical_data"],
            source_metadata=source_metadata,
            raw_data_dir=str(ar6_dummy_repo.raw_dir),
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=["missing-variable"],
            figure_output_format="svg",
            dpi=1,
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )

    gross_without_sequestration = harmonized_data.loc[
        harmonized_vars.isin(gross_figure_vars),
        :,
    ]
    with pytest.raises(RuntimeError):
        generate_ar6_figures(
            output_dir=str(tmp_path / "missing_sequestration"),
            harmonized_data=gross_without_sequestration,
            original_data=gross_original_data,
            harmonization_log=gross_harmonization_log,
            historical_data=inputs["historical_data"],
            source_metadata=source_metadata,
            raw_data_dir=str(ar6_dummy_repo.raw_dir),
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=gross_figure_vars,
            figure_output_format="svg",
            dpi=1,
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )

    gross_original_without_sequestration = inputs["original_data"].loc[
        original_vars.isin([NET_CO2_WITH_AFOLU]),
        :,
    ]
    with pytest.raises(RuntimeError):
        generate_ar6_figures(
            output_dir=str(tmp_path / "missing_original_sequestration"),
            harmonized_data=gross_figure_data,
            original_data=gross_original_without_sequestration,
            harmonization_log=gross_harmonization_log,
            historical_data=inputs["historical_data"],
            source_metadata=source_metadata,
            raw_data_dir=str(ar6_dummy_repo.raw_dir),
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=gross_figure_vars,
            figure_output_format="svg",
            dpi=1,
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )

    _figure_status(None, [0], "ignored")
    with pytest.raises(RuntimeError):
        generate_ar6_figures(
            output_dir=str(tmp_path / "bad"),
            harmonized_data=pd.DataFrame(
                {"Category": ["C1"], "Ssp_family": [1]},
                index=pd.MultiIndex.from_tuples(
                    [("M1", "S1", NET_CO2_WITH_AFOLU)], names=["model", "scenario", "variable"]
                ),
            ),
            original_data=pd.DataFrame(
                {"Category": ["C1"], "Ssp_family": [1]},
                index=pd.MultiIndex.from_tuples(
                    [("M1", "S1", NET_CO2_WITH_AFOLU)], names=["model", "scenario", "variable"]
                ),
            ),
            harmonization_log=pd.DataFrame(
                index=pd.MultiIndex.from_tuples([], names=["model", "scenario", "variable"])
            ),
            historical_data=inputs["historical_data"],
            source_metadata=source_metadata,
            raw_data_dir=str(ar6_dummy_repo.raw_dir),
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[NET_CO2_WITH_AFOLU],
            figure_output_format="svg",
            dpi=1,
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )

    no_var_harmonization_log = pd.DataFrame(
        index=pd.MultiIndex.from_tuples([], names=["model", "scenario", "variable"])
    )
    with pytest.raises(RuntimeError):
        generate_ar6_figures(
            output_dir=str(tmp_path / "bad2"),
            harmonized_data=harmonized_data,
            original_data=inputs["original_data"],
            harmonization_log=no_var_harmonization_log,
            historical_data=inputs["historical_data"],
            source_metadata=source_metadata,
            raw_data_dir=str(ar6_dummy_repo.raw_dir),
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[NET_CO2_WITH_AFOLU],
            figure_output_format="svg",
            dpi=1,
            figure_convergence_tol=0.1,
            figure_convergence_max_runs=20000,
        )


def test_ar6_plot_contract_and_budget_alt_branches(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
    ar6_processed_pathway_outputs: dict[str, pd.DataFrame],
) -> None:
    inputs = ar6_processed_pathway_outputs
    harmonized_data = inputs["harmonized_data"]
    co2_data = harmonized_data.loc[(slice(None), slice(None), NET_CO2_WITH_AFOLU), :]
    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)

    duplicate_rows = pd.DataFrame([[1.0], [2.0]], index=["dup", "dup"], columns=[2019])
    with pytest.raises(RuntimeError):
        _require_row_series(duplicate_rows, "dup", context="test")

    conflicting_metadata = pd.Series(
        ["C1", "C2"],
        index=pd.MultiIndex.from_tuples(
            [("M1", "S1", NET_CO2_WITH_AFOLU), ("M1", "S1", NET_CO2_WITH_AFOLU)],
            names=["model", "scenario", "variable"],
        ),
        dtype=object,
    )
    with pytest.raises(RuntimeError):
        _metadata_scalar_for_index(
            conflicting_metadata,
            ("M1", "S1", NET_CO2_WITH_AFOLU),
            field_name="Category",
        )
    unique_metadata = pd.Series(
        ["C1"],
        index=pd.MultiIndex.from_tuples(
            [("M1", "S1", NET_CO2_WITH_AFOLU)],
            names=["model", "scenario", "variable"],
        ),
        dtype=object,
    )
    assert (
        _metadata_scalar_for_index(
            unique_metadata,
            ("M1", "S1", NET_CO2_WITH_AFOLU),
            field_name="Category",
        )
        == "C1"
    )
    all_na_metadata = pd.Series(
        [pd.NA, pd.NA],
        index=pd.MultiIndex.from_tuples(
            [("M1", "S1", NET_CO2_WITH_AFOLU), ("M1", "S1", NET_CO2_WITH_AFOLU)],
            names=["model", "scenario", "variable"],
        ),
        dtype=object,
    )
    assert (
        _metadata_scalar_for_index(
            all_na_metadata,
            ("M1", "S1", NET_CO2_WITH_AFOLU),
            field_name="Category",
        )
        is pd.NA
    )
    assert max_year(pd.DataFrame({"x": [1]})) is None
    assert numeric_year_columns(pd.DataFrame(columns=["2019", "note"])) == [2019]
    with pytest.raises(RuntimeError):
        remaining_budget_end_year(pd.DataFrame({"x": [1]}))

    with pytest.raises(RuntimeError):
        historical_series(
            pd.DataFrame(index=[NET_CO2_WITH_AFOLU], columns=["x"]),
            NET_CO2_WITH_AFOLU,
            2000,
        )
    with pytest.raises(RuntimeError):
        historical_series(inputs["historical_data"], NET_CO2_WITH_AFOLU, 3000, 3001)

    fig, ax = plt.subplots()
    empty_parts = ax.violinplot([1.0, 2.0])
    plot_violin(empty_parts, "blue", 0.5)
    plt.close(fig)

    fig, ax = plt.subplots()
    plot_pathways(
        data_df=co2_data,
        data_historic_df=pd.Series(dtype=float),
        var_selected=NET_CO2_WITH_AFOLU,
        timewindow_l=[2019, 2060],
        ax=ax,
    )
    first_pathway_line = ax.lines[0]
    assert np.asarray(first_pathway_line.get_xdata()).tolist() == list(range(2019, 2060 + 1))
    plt.close(fig)

    hist_to_study_start = historical_series(
        inputs["historical_data"], NET_CO2_WITH_AFOLU, 1950
    ).loc[lambda s: s.index <= 2019]
    fig, ax = plt.subplots()
    plot_pathways(
        data_df=co2_data,
        data_historic_df=hist_to_study_start,
        var_selected=NET_CO2_WITH_AFOLU,
        timewindow_l=[2019, 2100],
        ax=ax,
    )
    assert np.asarray(ax.lines[0].get_xdata()).tolist() == list(range(2019, 2100 + 1))
    assert np.asarray(ax.lines[-1].get_xdata()).tolist() == list(range(1950, 2019 + 1))
    plt.close(fig)

    # plot_pathways with a timewindow outside year columns exercises the
    # _plot_pathway_matrix early return when ``not years``.
    fig, ax = plt.subplots()
    plot_pathways(
        data_df=co2_data,
        data_historic_df=pd.Series(dtype=float),
        var_selected=NET_CO2_WITH_AFOLU,
        timewindow_l=[3000, 3001],
        ax=ax,
    )
    assert len(ax.lines) == 0
    plt.close(fig)

    no_year_df = pd.DataFrame(
        {"Category": ["C1"], "Ssp_family": [1]},
        index=pd.MultiIndex.from_tuples(
            [("M1", "S1", NET_CO2_WITH_AFOLU)],
            names=["model", "scenario", "variable"],
        ),
    )
    fig, ax = plt.subplots()
    with pytest.raises(RuntimeError):
        plot_pathways(
            data_df=no_year_df,
            data_historic_df=pd.Series(dtype=float),
            var_selected=NET_CO2_WITH_AFOLU,
            timewindow_l=[2019, 2060],
            ax=ax,
        )
    plt.close(fig)

    extra_category_row = co2_data.iloc[[0]].copy()
    extra_category_row.index = pd.MultiIndex.from_tuples(
        [("M_extra", "S_extra", NET_KYOTO_WITH_AFOLU)],
        names=co2_data.index.names,
    )
    extra_category_row.loc[:, "Category"] = "C_extra"
    sparse_category_df = pd.concat([co2_data.iloc[[0]], extra_category_row])
    fig, axes = plt.subplots(2, 1)
    plot_budgets_summary(
        data_df=sparse_category_df,
        var_selected=NET_CO2_WITH_AFOLU,
        timewindow_l=[2019, 2060],
        ax=axes,
    )
    plt.close(fig)
    fig, ax = plt.subplots()
    plot_pathways(
        data_df=sparse_category_df,
        data_historic_df=pd.Series(dtype=float),
        var_selected=NET_CO2_WITH_AFOLU,
        timewindow_l=[2019, 2060],
        ax=ax,
    )
    plt.close(fig)

    fig, axes = plt.subplots(2, 1)
    with pytest.raises(RuntimeError):
        plot_budgets_summary(
            data_df=no_year_df,
            var_selected=NET_CO2_WITH_AFOLU,
            timewindow_l=[2019, 2060],
            ax=axes,
        )
    plt.close(fig)

    no_remaining = co2_data.drop(columns=[2100])
    fig, axes = plt.subplots(2, 1)
    plot_budgets_summary(
        data_df=no_remaining,
        var_selected=NET_CO2_WITH_AFOLU,
        timewindow_l=[2019, 2060],
        ax=axes,
        remaining_budget_end_year_value=2100,
    )
    plt.close(fig)

    small_index = [("M1", "S1", NET_CO2_WITH_AFOLU)] * 10
    assert _subsample_sampled_index(small_index) is small_index

    large_index = list(co2_data.index) * (_VIOLIN_MAX_SAMPLES + 1)
    subsampled = _subsample_sampled_index(large_index)
    assert len(subsampled) == _VIOLIN_MAX_SAMPLES
    assert all(label in large_index for label in subsampled)

    second_subsampled = _subsample_sampled_index(large_index)
    assert subsampled == second_subsampled

    _generate_sampling_budget_figure(
        figures_dir=tmp_path,
        ext="png",
        dpi=10,
        out_paths=[],
        harmonized_data=harmonized_data,
        all_variables_l=[],
        tmp_proba_df=pd.DataFrame(),
        ratio_lhs_vs_srs=pd.DataFrame(),
        montecarlo_srs_index_d={},
        montecarlo_lhs_index_d={},
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        database="ar6-public",
        categories_repr="[]",
        figure_stub="fig-empty",
    )

    probability_df = build_sampling_probability_df(
        co2_data,
        [NET_CO2_WITH_AFOLU],
        ["C1", "C2", "C3", "C4"],
    )
    empty_probability_df = probability_df.copy()
    empty_probability_df.loc[:, ["proba_SRS", "proba_LHS"]] = np.nan
    ratio_nonempty = pd.DataFrame(
        {"median": [1.1]},
        index=pd.MultiIndex.from_tuples(
            [(NET_CO2_WITH_AFOLU, "C1", 1)],
            names=["variable", "Category", "Ssp_family"],
        ),
    )
    ratio_empty = ratio_nonempty.copy()
    ratio_empty.loc[:, "median"] = np.nan
    out_paths: list[str] = []
    _write_sampling_probability_ratio_figure(
        figures_dir=tmp_path,
        ext="png",
        dpi=10,
        out_paths=out_paths,
        tmp_proba_df=empty_probability_df,
        all_variables_l=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database="ar6-public",
        categories_repr="['C1']",
        variable_group="CO2",
    )
    _write_sampling_probability_ratio_figure(
        figures_dir=tmp_path,
        ext="png",
        dpi=10,
        out_paths=out_paths,
        tmp_proba_df=probability_df,
        all_variables_l=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database="ar6-public",
        categories_repr="['C1']",
        variable_group="CO2",
    )
    _write_sampling_median_ratio_figure(
        figures_dir=tmp_path,
        ext="png",
        dpi=10,
        out_paths=out_paths,
        ratio_lhs_vs_srs=ratio_empty,
        all_variables_l=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database="ar6-public",
        categories_repr="['C1']",
        variable_group="CO2",
    )
    _write_sampling_median_ratio_figure(
        figures_dir=tmp_path,
        ext="png",
        dpi=10,
        out_paths=out_paths,
        ratio_lhs_vs_srs=ratio_nonempty,
        all_variables_l=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database="ar6-public",
        categories_repr="['C1']",
        variable_group="CO2",
    )
    _generate_sampling_budget_figure(
        figures_dir=tmp_path,
        ext="png",
        dpi=10,
        out_paths=out_paths,
        harmonized_data=co2_data,
        all_variables_l=[NET_CO2_WITH_AFOLU],
        tmp_proba_df=probability_df,
        ratio_lhs_vs_srs=ratio_nonempty,
        montecarlo_srs_index_d={NET_CO2_WITH_AFOLU: list(co2_data.index)},
        montecarlo_lhs_index_d={NET_CO2_WITH_AFOLU: list(co2_data.index)},
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        database="ar6-public",
        categories_repr="['C1']",
        figure_stub="fig-top-bars",
    )
    _generate_sampling_budget_figure(
        figures_dir=tmp_path,
        ext="png",
        dpi=10,
        out_paths=out_paths,
        harmonized_data=co2_data,
        all_variables_l=[NET_CO2_WITH_AFOLU],
        tmp_proba_df=empty_probability_df,
        ratio_lhs_vs_srs=ratio_empty,
        montecarlo_srs_index_d={NET_CO2_WITH_AFOLU: list(co2_data.index)},
        montecarlo_lhs_index_d={NET_CO2_WITH_AFOLU: list(co2_data.index)},
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        database="ar6-public",
        categories_repr="['C1']",
        figure_stub="fig-no-top-bars",
    )
    assert out_paths

    no_category_name_explorer = type(explorer)(data=explorer.data.drop(columns=["Category_name"]))
    formatted_no_category_name = filter_and_format_rawdata(
        no_category_name_explorer,
        {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
    )
    assert formatted_no_category_name["Category_name"].isna().all()

    c4_formatted = filter_and_format_rawdata(
        explorer,
        {"category": "C4", "ssp_family": 4, "model": ["ALL", ["M4"]]},
    )
    processed_df, wo_afolu_drop_log = build_pre_harmonization_variables(
        c4_formatted,
        study_start_year=2019,
    )
    assert not processed_df.empty
    assert wo_afolu_drop_log["variable"].tolist() == [NET_CO2_WO_AFOLU]
    assert wo_afolu_drop_log["retained_variable"].tolist() == [RAW_CO2_WITH_AFOLU]
    assert set(wo_afolu_drop_log["ssp_family"]) == {4}
