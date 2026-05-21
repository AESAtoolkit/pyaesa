from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from tests.package.helpers.ar6_imports import (
    collection_config,
    processing_fig_guides,
    processing_fig_io,
    processing_fig_outputs,
    processing_fig_sampling_panels,
    processing_fig_warming,
    processing_generate_figures,
    processing_plot_helpers,
    processing_plot_historical,
    processing_plot_sampling,
    processing_sampling_convergence,
    processing_sampling_payloads,
)
from tests.package.helpers.ar6_dummy_repo import AR6DummyRepo

NET_CO2_WITH_AFOLU = collection_config.NET_CO2_WITH_AFOLU
NET_CO2_WO_AFOLU = collection_config.NET_CO2_WO_AFOLU
NET_KYOTO_WITH_AFOLU = collection_config.NET_KYOTO_WITH_AFOLU
NET_KYOTO_WO_AFOLU = collection_config.NET_KYOTO_WO_AFOLU
RAW_CO2_WITH_AFOLU = collection_config.RAW_CO2_WITH_AFOLU
_figure_explanation_block = processing_fig_guides._figure_explanation_block
_remaining_budget_drop_csv_map = processing_fig_guides._remaining_budget_drop_csv_map
ensure_figures_guide = processing_fig_guides.ensure_figures_guide
figures_explanation_text = processing_fig_guides.figures_explanation_text
save_figure = processing_fig_io.save_figure
ensure_figures = processing_fig_outputs.ensure_figures
figure_signature = processing_fig_outputs.figure_signature
load_saved_figure_files = processing_fig_outputs.load_saved_figure_files
write_figure_metadata = processing_fig_outputs.write_figure_metadata
write_sampling_figures = processing_fig_sampling_panels.write_sampling_figures
write_median_warming_figure = processing_fig_warming.write_median_warming_figure
generate_ar6_figures = processing_generate_figures.generate_ar6_figures
CATEGORY_COLORS = processing_plot_helpers.CATEGORY_COLORS
WARMING_METADATA_COLUMN = processing_plot_helpers.WARMING_METADATA_COLUMN
append_remaining_budget_drop_records = processing_plot_helpers.append_remaining_budget_drop_records
historical_series = processing_plot_helpers.historical_series
max_year = processing_plot_helpers.max_year
numeric_year_columns = processing_plot_helpers.numeric_year_columns
plot_violin = processing_plot_helpers.plot_violin
remaining_budget_end_year = processing_plot_helpers.remaining_budget_end_year
scenario_df_from_harmonized = processing_plot_helpers.scenario_df_from_harmonized
var_df = processing_plot_helpers.var_df
write_drop_csv = processing_plot_helpers.write_drop_csv
year_slice = processing_plot_helpers.year_slice
year_slice_exclusive_end = processing_plot_helpers.year_slice_exclusive_end
_plot_overlay = processing_plot_historical._plot_overlay
_overlay_series = processing_plot_historical._overlay_series
_read_historical_overlay = processing_plot_historical._read_historical_overlay
plot_historical_emissions = processing_plot_historical.plot_historical_emissions
_ssp_family_sort_rank = processing_plot_sampling._ssp_family_sort_rank
build_sampling_runs_until_convergence = (
    processing_plot_sampling.build_sampling_runs_until_convergence
)
build_sampling_probability_df = processing_plot_sampling.build_sampling_probability_df
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


def test_ar6_figure_plot_contracts_and_historical_overlay(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
    ar6_processed_pathway_outputs: dict[str, pd.DataFrame],
) -> None:
    inputs = ar6_processed_pathway_outputs
    harmonized_data = inputs["harmonized_data"]
    historical_data = inputs["historical_data"]

    assert 2019 in numeric_year_columns(harmonized_data)
    assert year_slice(harmonized_data, 2019, 2021) == [2019, 2020, 2021]
    assert year_slice_exclusive_end(harmonized_data, 2019, 2021) == [2019, 2020]
    assert max_year(harmonized_data) == 2100
    with pytest.raises(RuntimeError):
        var_df(harmonized_data, "Missing")
    with pytest.raises(RuntimeError):
        scenario_df_from_harmonized(harmonized_data.iloc[0:0, :])
    assert remaining_budget_end_year(harmonized_data) == 2100
    with pytest.raises(RuntimeError):
        remaining_budget_end_year(pd.DataFrame({"x": [1]}))

    co2_history = historical_series(historical_data, NET_CO2_WITH_AFOLU, 2000, 2022)
    assert co2_history.index.min() == 2000
    assert co2_history.index.max() == 2021
    with pytest.raises(RuntimeError):
        historical_series(historical_data, "Missing", 2000)

    overlay_df = _read_historical_overlay(ar6_dummy_repo.raw_dir)
    assert not overlay_df.empty
    co2_overlay = _overlay_series(overlay_df, RAW_CO2_WITH_AFOLU)
    assert co2_overlay.index.min() == 1970
    fig, axes = plot_historical_emissions(historical_data, ar6_dummy_repo.raw_dir)
    assert len(axes) == 2
    plt.close(fig)

    no_hist_fig, no_hist_axes = plot_historical_emissions(
        historical_data.loc[:, ["units"]],
        ar6_dummy_repo.raw_dir,
    )
    assert len(no_hist_axes) == 2
    plt.close(no_hist_fig)

    bad_overlay = pd.DataFrame(
        [
            {
                "Model": "EDGAR",
                "Scenario": "historical",
                "Region": "World",
                "Variable": RAW_CO2_WITH_AFOLU,
                "Unit": "Gt CO2/yr",
                "1970": 1.0,
            },
            {
                "Model": "EDGAR",
                "Scenario": "historical",
                "Region": "World",
                "Variable": "Emissions|CO2|Lower",
                "Unit": "Gt CO2/yr",
                "1971": 0.8,
            },
            {
                "Model": "EDGAR",
                "Scenario": "historical",
                "Region": "World",
                "Variable": "Emissions|CO2|Upper",
                "Unit": "Gt CO2/yr",
                "1972": 1.2,
            },
        ]
    )
    fig, ax = plt.subplots()
    with pytest.raises(RuntimeError):
        _plot_overlay(
            ax,
            bad_overlay,
            variable=RAW_CO2_WITH_AFOLU,
            lower_variable="Emissions|CO2|Lower",
            upper_variable="Emissions|CO2|Upper",
            linestyle="solid",
        )
    plt.close(fig)

    duplicate_harmonized = pd.concat([harmonized_data.iloc[[0]], harmonized_data.iloc[[0]]])
    drop_records: list[dict] = []
    remaining_mask = append_remaining_budget_drop_records(
        drop_records=drop_records,
        data_all_cats_df=duplicate_harmonized,
        var_data_df=duplicate_harmonized,
        remaining_budget_end_year_value=2105,
        figure_name="figure",
        subset_name="subset",
        study_period=[2019, 2060],
    )
    assert bool((~remaining_mask).all())
    csv_path = write_drop_csv(tmp_path, "drop-test", drop_records)
    assert csv_path is not None and Path(csv_path).exists()
    assert write_drop_csv(tmp_path, "empty-test", []) is None

    fig, ax = plt.subplots()
    parts = ax.violinplot([1.0, 2.0, 3.0])
    plot_violin(parts, CATEGORY_COLORS["C1"], 0.5)
    plt.close(fig)

    ar6_dummy_repo.overlay_path.unlink()
    with pytest.raises(RuntimeError):
        _read_historical_overlay(ar6_dummy_repo.raw_dir)
    ar6_dummy_repo.overlay_path.write_text("bad", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _read_historical_overlay(ar6_dummy_repo.raw_dir)


def test_ar6_figure_metadata_and_guide_contracts(
    tmp_path: Path, ar6_dummy_repo: AR6DummyRepo
) -> None:
    del ar6_dummy_repo
    sig = figure_signature(
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1"],
        variables_output=[NET_CO2_WITH_AFOLU],
        figure_output_format="png",
        figure_dpi=10,
        harmonization_method="reduced_offset",
        figure_convergence_tol=0.05,
        figure_convergence_max_runs=10000,
    )
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    metadata_path = tmp_path / "figures_meta.json"
    figure_path = figures_dir / "figure.png"
    figure_path.write_text("png", encoding="utf-8")
    sampling_log_csv = tmp_path / "sampling.csv"
    sampling_log_txt = tmp_path / "sampling.txt"
    sampling_log_csv.write_text("x\n", encoding="utf-8")
    sampling_log_txt.write_text("columns", encoding="utf-8")
    write_figure_metadata(
        figures_metadata_file=metadata_path,
        signature=sig,
        figure_files=[str(figure_path)],
        generation_complete=True,
        sampling_log_csv_file=sampling_log_csv,
        sampling_log_columns_txt_file=sampling_log_txt,
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
        figure_convergence_tol=0.05,
        figure_convergence_max_runs=10000,
    ) == [str(figure_path)]

    stale_dir = tmp_path / "stale"
    stale_dir.mkdir()
    stale_figure = stale_dir / "stale.png"
    stale_figure.write_text("png", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_saved_figure_files(
            figures_metadata_file=tmp_path / "missing_meta.json",
            figures_dir=stale_dir,
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[NET_CO2_WITH_AFOLU],
            figure_output_format="png",
            figure_dpi=10,
            harmonization_method="reduced_offset",
            figure_convergence_tol=0.05,
            figure_convergence_max_runs=10000,
        )

    guide_drop_csv = figures_dir / "fig-budgets-CO2-test-remaining-budget-panel-dropped_rows.csv"
    guide_drop_csv.write_text("header\n", encoding="utf-8")
    mapping = _remaining_budget_drop_csv_map(figures_dir)
    assert mapping["fig-budgets-CO2-test"] == [guide_drop_csv.name]
    block = _figure_explanation_block(
        "fig-budgets-CO2-test.png",
        [2019, 2060],
        tmp_path / "global.csv",
        mapping,
    )
    assert block
    fallback_block = _figure_explanation_block(
        "unknown.png",
        [2019, 2060],
        tmp_path / "global.csv",
        {},
    )
    assert fallback_block

    explanation_text = figures_explanation_text(
        ["fig-budgets-CO2-test.png"],
        [2019, 2060],
        figures_dir,
        tmp_path / "global.csv",
    )
    assert explanation_text.strip()
    guide_file, written = ensure_figures_guide(
        figures_dir=figures_dir,
        figure_files=["fig-budgets-CO2-test.png"],
        study_period=[2019, 2060],
        global_drop_csv_file=tmp_path / "global.csv",
        rewrite=True,
    )
    assert written is True and guide_file is not None
    reused_guide_file, reused_written = ensure_figures_guide(
        figures_dir=figures_dir,
        figure_files=["fig-budgets-CO2-test.png"],
        study_period=[2019, 2060],
        global_drop_csv_file=tmp_path / "global.csv",
        rewrite=False,
    )
    assert reused_written is False and reused_guide_file == guide_file
    assert ensure_figures_guide(
        figures_dir=tmp_path / "empty",
        figure_files=[],
        study_period=[2019, 2060],
        global_drop_csv_file=tmp_path / "global.csv",
        rewrite=False,
    ) == (None, False)


def test_ar6_figure_sampling_contracts_and_generate_figures(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
    ar6_processed_pathway_outputs: dict[str, pd.DataFrame],
) -> None:
    inputs = ar6_processed_pathway_outputs
    harmonized_data = inputs["harmonized_data"]
    source_metadata = inputs["source_metadata"]
    one_variable = NET_CO2_WITH_AFOLU
    harmonized_min = harmonized_data.loc[(slice(None), slice(None), one_variable), :].sort_index()
    harmonized_min = harmonized_min.iloc[:3, :]
    original_min = inputs["original_data"].loc[harmonized_min.index, :].sort_index()
    harmonization_log_min = inputs["harmonization_log"].loc[harmonized_min.index, :].sort_index()
    source_metadata_min = source_metadata.loc[
        source_metadata.index.intersection(harmonized_min.index.droplevel("variable").unique())
    ]

    probability_df = build_sampling_probability_df(
        harmonized_min, [one_variable], ["C1", "C2", "C3", "C4"]
    )
    assert "proba_LHS" in probability_df.columns
    payload = build_variable_payload(
        harmonized_data=harmonized_min,
        tmp_proba_df=probability_df,
        var_sel=one_variable,
        categories=["C1", "C2", "C3", "C4"],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        sampling_method="SRS",
    )
    assert payload["buckets"]
    sampled_counts = {
        bucket["key"]: np.ones(len(bucket["positions"]), dtype=np.int64)
        for bucket in payload["buckets"]
    }
    snapshot, study_rows = build_snapshot_from_counts(
        sampled_counts=sampled_counts,
        payload=payload,
        var_sel=one_variable,
    )
    assert snapshot
    assert study_rows
    assert expected_snapshot_key_count(payload["buckets"]) >= len(snapshot)

    stats = distribution_stats_from_counts(
        np.array([1.0, 2.0, np.nan]),
        np.array([2, 1, 3], dtype=np.int64),
    )
    assert stats["median"] == pytest.approx(1.0)
    assert distribution_stats_from_counts(np.array([np.nan]), np.array([0], dtype=np.int64)) is None
    assert snapshots_are_stable(snapshot, snapshot, relative_tolerance=0.01) is True
    changed_snapshot = dict(snapshot)
    first_key = next(iter(changed_snapshot))
    changed_snapshot[first_key] = changed_snapshot[first_key] * 2
    assert snapshots_are_stable(snapshot, changed_snapshot, relative_tolerance=0.01) is False
    assert sampling_seed(one_variable, "SRS") == sampling_seed(one_variable, "SRS")
    assert flatten_sampled_index_from_counts(payload["buckets"], sampled_counts)
    log_rows = snapshot_to_log_rows(
        snapshot,
        variable=one_variable,
        sampling_method="SRS",
        rng_seed_value=123,
        final_runs_per_bucket=10000,
        run_batch_size=10000,
        maximum_runs_per_bucket=30000,
        relative_tolerance=0.05,
        stable_checks_required=3,
    )
    assert log_rows[0]["distribution_kind"] in {"study", "remaining"}
    study_frame = study_rows_to_frame(
        study_rows,
        ("mean", "median", "p25", "p75", "p5", "p95"),
    )
    assert not study_frame.empty
    assert _ssp_family_sort_rank("all") == float("inf")
    assert _ssp_family_sort_rank("2") == 2.0

    sampled_srs, sampled_lhs, ratio_df, convergence_log_df = build_sampling_runs_until_convergence(
        harmonized_data=harmonized_min,
        tmp_proba_df=probability_df,
        all_variables_l=[one_variable],
        categories=["C1", "C2", "C3", "C4"],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        relative_tolerance=1.0,
        max_runs_per_bucket=2000,
        run_batch_size=1000,
        stable_checks_required=1,
    )
    assert one_variable in sampled_srs
    assert one_variable in sampled_lhs
    assert not ratio_df.empty
    assert not convergence_log_df.empty

    figures_dir = tmp_path / "sampling_figures"
    figures_dir.mkdir()
    out_paths: list[str] = []
    sampling_status_messages: list[str] = []
    convergence_log_written = write_sampling_figures(
        figures_dir=figures_dir,
        ext="png",
        dpi=10,
        out_paths=out_paths,
        harmonized_data=harmonized_min,
        all_variables_l=[one_variable],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        database="ar6-public",
        categories=["C1", "C2", "C3", "C4"],
        categories_repr="['C1', 'C2', 'C3', 'C4']",
        relative_tolerance=1.0,
        max_runs_per_bucket=2000,
        run_batch_size=1000,
        stable_checks_required=1,
        status_callback=sampling_status_messages.append,
        build_sampling_runs_until_convergence_func=lambda **kwargs: (
            {NET_CO2_WITH_AFOLU: list(harmonized_min.index)},
            {NET_CO2_WITH_AFOLU: list(harmonized_min.index)},
            ratio_df,
            convergence_log_df,
        ),
    )
    assert out_paths
    assert not convergence_log_written.empty
    assert sampling_status_messages == [
        "sampling probability ratio (CO2)",
        "sampling median ratio (CO2)",
        "sampling budget (CO2, net)",
    ]
    no_status_paths: list[str] = []
    no_status_log = write_sampling_figures(
        figures_dir=figures_dir,
        ext="png",
        dpi=10,
        out_paths=no_status_paths,
        harmonized_data=harmonized_min,
        all_variables_l=[one_variable],
        study_period=[2019, 2060],
        remaining_budget_end_year_value=2100,
        database="ar6-public",
        categories=["C1", "C2", "C3", "C4"],
        categories_repr="['C1', 'C2', 'C3', 'C4']",
        relative_tolerance=1.0,
        max_runs_per_bucket=2000,
        run_batch_size=1000,
        stable_checks_required=1,
        status_callback=None,
        build_sampling_runs_until_convergence_func=lambda **kwargs: (
            {NET_CO2_WITH_AFOLU: list(harmonized_min.index)},
            {NET_CO2_WITH_AFOLU: list(harmonized_min.index)},
            ratio_df,
            convergence_log_df,
        ),
    )
    assert no_status_paths
    assert not no_status_log.empty

    generation_status_messages: list[str] = []

    def fake_sampling_figures(**kwargs):
        kwargs["status_callback"]("sampling probability ratio")
        return pd.DataFrame(
            [
                {
                    "variable": one_variable,
                    "method": "SRS",
                    "distribution_kind": "study",
                    "category": "C1",
                    "ssp_family": 1,
                    "rng_seed": 1,
                    "final_runs_per_bucket": 1000,
                    "run_batch_size": 1000,
                    "maximum_runs_per_bucket": 1000,
                    "relative_tolerance": 1.0,
                    "stable_checks_required": 1,
                    "mean": 1.0,
                    "median": 1.0,
                    "p25": 1.0,
                    "p75": 1.0,
                    "p5": 1.0,
                    "p95": 1.0,
                }
            ]
        )

    output_dir = tmp_path / "generated"
    out_paths_generated, convergence_log_generated = generate_ar6_figures(
        output_dir=str(output_dir),
        harmonized_data=harmonized_min,
        original_data=original_min,
        harmonization_log=harmonization_log_min,
        historical_data=inputs["historical_data"],
        source_metadata=source_metadata_min,
        raw_data_dir=str(ar6_dummy_repo.raw_dir),
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1", "C2", "C3", "C4"],
        variables_output=[one_variable],
        figure_output_format="png",
        dpi=10,
        figure_convergence_tol=1.0,
        figure_convergence_max_runs=2000,
        status_callback=generation_status_messages.append,
        sampling_run_batch_size=1000,
        sampling_stable_checks_required=1,
        write_sampling_figures_func=fake_sampling_figures,
    )
    assert out_paths_generated
    assert not convergence_log_generated.empty
    assert any(
        message.startswith("Generating figure 7/18: sampling probability ratio")
        for message in generation_status_messages
    )

    def fake_sampling_generate_only(**kwargs):
        kwargs["status_callback"]("Generating sample figure")
        return pd.DataFrame()

    out_paths_none, convergence_log_none = generate_ar6_figures(
        output_dir=str(tmp_path / "generated_none"),
        harmonized_data=harmonized_min,
        original_data=original_min,
        harmonization_log=harmonization_log_min,
        historical_data=inputs["historical_data"],
        source_metadata=source_metadata_min,
        raw_data_dir=str(ar6_dummy_repo.raw_dir),
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1", "C2", "C3", "C4"],
        variables_output=[one_variable],
        figure_output_format="png",
        dpi=10,
        figure_convergence_tol=1.0,
        figure_convergence_max_runs=2000,
        status_callback=None,
        sampling_run_batch_size=1000,
        sampling_stable_checks_required=1,
        write_sampling_figures_func=fake_sampling_generate_only,
    )
    assert out_paths_none
    assert convergence_log_none.empty
    with pytest.raises(ValueError):
        generate_ar6_figures(
            output_dir=str(tmp_path / "bad"),
            harmonized_data=harmonized_min,
            original_data=original_min,
            harmonization_log=harmonization_log_min,
            historical_data=inputs["historical_data"],
            source_metadata=source_metadata_min,
            raw_data_dir=str(ar6_dummy_repo.raw_dir),
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[one_variable],
            figure_output_format="bad",
            dpi=10,
            figure_convergence_tol=1.0,
            figure_convergence_max_runs=2000,
            sampling_run_batch_size=1000,
            sampling_stable_checks_required=1,
        )


def test_ar6_figure_output_contracts_and_warming_errors(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
    ar6_processed_pathway_outputs: dict[str, pd.DataFrame],
) -> None:
    inputs = ar6_processed_pathway_outputs
    harmonized_data = inputs["harmonized_data"]
    source_metadata = inputs["source_metadata"]
    scenario_df = scenario_df_from_harmonized(harmonized_data)

    with pytest.raises(RuntimeError):
        write_median_warming_figure(
            figures_dir=tmp_path,
            ext="png",
            dpi=10,
            out_paths=[],
            scenario_rows_df=scenario_df,
            source_metadata=source_metadata.iloc[0:0, :],
            categories=["C1"],
            study_period=[2019, 2060],
            database="ar6-public",
            categories_repr="['C1']",
        )
    with pytest.raises(RuntimeError):
        write_median_warming_figure(
            figures_dir=tmp_path,
            ext="png",
            dpi=10,
            out_paths=[],
            scenario_rows_df=scenario_df,
            source_metadata=source_metadata.drop(columns=[WARMING_METADATA_COLUMN]),
            categories=["C1"],
            study_period=[2019, 2060],
            database="ar6-public",
            categories_repr="['C1']",
        )

    fig, _ax = plt.subplots()
    out_paths: list[str] = []
    saved_lists: list[list[str]] = []
    output_path = tmp_path / "simple.png"
    save_figure(
        fig,
        output_path,
        dpi=10,
        out_paths=out_paths,
        metadata_callback=lambda paths: saved_lists.append(list(paths)),
    )
    assert out_paths == [str(output_path)]
    assert saved_lists == [[str(output_path)]]

    workbook_path = tmp_path / "harmonized.xlsx"
    log_path = tmp_path / "harmonization_log.xlsx"
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    process_metadata = figure_signature(
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1"],
        variables_output=[NET_CO2_WITH_AFOLU],
        figure_output_format="png",
        figure_dpi=10,
        harmonization_method="reduced_offset",
        figure_convergence_tol=0.05,
        figure_convergence_max_runs=10000,
    )
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

    fake_convergence_log = pd.DataFrame(
        [
            {
                "variable": NET_CO2_WITH_AFOLU,
                "method": "SRS",
                "distribution_kind": "study",
                "category": "C1",
                "ssp_family": 1,
                "rng_seed": 1,
                "final_runs_per_bucket": 10000,
                "run_batch_size": 10000,
                "maximum_runs_per_bucket": 10000,
                "relative_tolerance": 0.05,
                "stable_checks_required": 3,
                "mean": 1.0,
                "median": 1.0,
                "p25": 1.0,
                "p75": 1.0,
                "p5": 1.0,
                "p95": 1.0,
            }
        ]
    )

    def fake_generate_figures(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / "fake.png"
        figure_path.write_text("figure", encoding="utf-8")
        return [str(figure_path)], fake_convergence_log

    kept_subdir = figures_dir / "keep-dir"
    kept_subdir.mkdir()

    figure_files, reused = ensure_figures(
        out_file=workbook_path,
        log_file=log_path,
        figures_dir=figures_dir,
        figures_metadata_file=tmp_path / "figures_meta.json",
        logs_dir=tmp_path / "logs",
        variables_output=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1"],
        figure_output_format="png",
        figure_dpi=10,
        harmonization_method="reduced_offset",
        figure_convergence_tol=0.05,
        figure_convergence_max_runs=10000,
        refresh=True,
        source_metadata=inputs["source_metadata"],
        raw_data_dir=ar6_dummy_repo.raw_dir,
        generate_figures_func=fake_generate_figures,
    )
    assert reused is False
    assert figure_files == [str(figures_dir / "fake.png")]
    assert kept_subdir.exists()
    assert (
        load_saved_figure_files(
            figures_metadata_file=tmp_path / "figures_meta.json",
            figures_dir=figures_dir,
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            variables_output=[NET_CO2_WITH_AFOLU],
            figure_output_format="png",
            figure_dpi=10,
            harmonization_method="reduced_offset",
            figure_convergence_tol=0.05,
            figure_convergence_max_runs=10000,
        )
        == figure_files
    )

    partial_metadata_path = tmp_path / "partial_meta.json"
    write_figure_metadata(
        figures_metadata_file=partial_metadata_path,
        signature=process_metadata,
        figure_files=[str(figures_dir / "fake.png")],
        generation_complete=False,
    )
    regenerated_files, regenerated_reused = ensure_figures(
        out_file=workbook_path,
        log_file=log_path,
        figures_dir=figures_dir,
        figures_metadata_file=partial_metadata_path,
        logs_dir=tmp_path / "logs-second",
        variables_output=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1"],
        figure_output_format="png",
        figure_dpi=10,
        harmonization_method="reduced_offset",
        figure_convergence_tol=0.05,
        figure_convergence_max_runs=10000,
        refresh=False,
        source_metadata=inputs["source_metadata"],
        raw_data_dir=ar6_dummy_repo.raw_dir,
        generate_figures_func=fake_generate_figures,
    )
    assert regenerated_reused is False
    assert regenerated_files == [str(figures_dir / "fake.png")]

    with pytest.raises(RuntimeError):
        ensure_figures(
            out_file=workbook_path,
            log_file=tmp_path / "missing_log.xlsx",
            figures_dir=figures_dir,
            figures_metadata_file=tmp_path / "other_meta.json",
            logs_dir=tmp_path / "other_logs",
            variables_output=[NET_CO2_WITH_AFOLU],
            study_period=[2019, 2060],
            database="ar6-public",
            categories=["C1"],
            figure_output_format="png",
            figure_dpi=10,
            harmonization_method="reduced_offset",
            figure_convergence_tol=0.05,
            figure_convergence_max_runs=10000,
            refresh=True,
            source_metadata=inputs["source_metadata"],
            raw_data_dir=ar6_dummy_repo.raw_dir,
            generate_figures_func=fake_generate_figures,
        )
