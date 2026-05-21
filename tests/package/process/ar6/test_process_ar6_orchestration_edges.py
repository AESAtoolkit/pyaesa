from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from tests.package.helpers.ar6_imports import (
    collection_config,
    collection_explorer,
    collection_paths,
    processing_fig_outputs,
    processing_generate_figures,
    processing_harmonization,
    processing_historical,
    processing_loaders,
    processing_metadata,
    processing_paths,
    processing_preprocessing,
    processing_process_runner,
    processing_processing_modes,
    processing_reports,
    processing_report_summaries,
    processing_runtime_helpers,
    processing_study_period,
)
from tests.package.helpers.ar6_dummy_repo import AR6DummyRepo

DEFAULT_CATEGORIES = collection_config.DEFAULT_CATEGORIES
DEFAULT_DATABASE = collection_config.DEFAULT_DATABASE
DEFAULT_SSPS = collection_config.DEFAULT_SSPS
DEFAULT_VARIABLES_OUTPUT = collection_config.DEFAULT_VARIABLES_OUTPUT
NET_CO2_WITH_AFOLU = collection_config.NET_CO2_WITH_AFOLU
get_citation_txt_path = collection_paths.get_citation_txt_path
get_raw_dir = collection_paths.get_raw_dir
read_explorer_csv = collection_explorer.read_explorer_csv
ensure_figures = processing_fig_outputs.ensure_figures
generate_ar6_figures = processing_generate_figures.generate_ar6_figures
signature_matches = processing_metadata.signature_matches
write_json = processing_metadata.write_json
get_scope_dirs = processing_paths.get_scope_dirs
ProcessReportAR6 = processing_reports.ProcessReportAR6
summarize_variable_model_scenario_pairs = (
    processing_report_summaries.summarize_variable_model_scenario_pairs
)
harmonize_emissions = processing_harmonization.harmonize_emissions
_set_year_values = processing_historical._set_year_values
process_historical_emissions = processing_historical.process_historical_emissions
scenario_metadata_from_wide = processing_loaders.scenario_metadata_from_wide
_empty_filtered_dataframe = processing_preprocessing._empty_filtered_dataframe
filter_and_format_rawdata = processing_preprocessing.filter_and_format_rawdata
run_process_ar6_workflow = processing_process_runner.run_process_ar6_workflow
build_pathway_outputs = processing_processing_modes.build_pathway_outputs
process_signature = processing_runtime_helpers.process_signature
resolve_study_period = processing_study_period.resolve_study_period


def _runner_outputs(
    ar6_dummy_repo: AR6DummyRepo,
    *,
    study_period: list[int],
    harmonization: bool,
) -> dict[str, object]:
    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)
    return build_pathway_outputs(
        explorer=explorer,
        categories=list(DEFAULT_CATEGORIES),
        ssps=[int(value) for value in DEFAULT_SSPS],
        variables_output=list(DEFAULT_VARIABLES_OUTPUT),
        study_period=study_period,
        database_raw_dir=ar6_dummy_repo.raw_dir,
        models_relevant_all=sorted(set(explorer.data["model"])),
        harmonization=harmonization,
        harmonization_method="reduced_offset",
    )


def _fake_ensure_figures(**kwargs) -> tuple[list[str], bool]:
    figures_dir = Path(kwargs["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / "fake.png"
    figure_path.write_text("figure", encoding="utf-8")
    return [str(figure_path)], False


def _fake_ensure_figures_guide(**kwargs) -> tuple[Path, bool]:
    guide_path = Path(kwargs["figures_dir"]) / "figures_explanation.txt"
    guide_path.write_text("guide", encoding="utf-8")
    return guide_path, bool(kwargs["rewrite"])


class _StatusRecorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def show(self, message: str) -> None:
        self.messages.append(message)

    def finish(self) -> None:
        self.messages.append("finished")


def test_process_ar6_runner_reuse_and_refresh_branches(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    del tmp_path
    study_period = [2019, 2060]
    signature = process_signature(study_period, True, "reduced_offset")
    processed_dir, logs_dir, figures_dir = get_scope_dirs(
        study_period,
        harmonization=True,
        harmonization_method="reduced_offset",
    )
    out_file = processed_dir / "harmonized_ar6_public.xlsx"
    log_file = logs_dir / "harmonized_ar6_public_log.xlsx"
    dropped_rows_csv_file = logs_dir / "dropped_model_scenario_variable_rows.csv"
    process_meta_file = logs_dir / "scope_manifest.json"
    for path in [processed_dir, logs_dir, figures_dir]:
        path.mkdir(parents=True, exist_ok=True)
    out_file.write_text("workbook", encoding="utf-8")
    log_file.write_text("log", encoding="utf-8")
    dropped_rows_csv_file.write_text("model,scenario,variable\n", encoding="utf-8")
    write_json(
        process_meta_file,
        {
            "arguments": signature,
            "provenance": {"variable_coverage_summary_counts": []},
        },
    )

    figure_path = figures_dir / "cached.png"
    figure_path.write_text("png", encoding="utf-8")
    cached_guide = figures_dir / "figures_explanation.txt"

    reused_report = run_process_ar6_workflow(
        study_period=study_period,
        figures=True,
        harmonization=True,
        harmonization_method="reduced_offset",
        refresh=False,
        figure_output_format="png",
        figure_dpi=10,
        sampling_config={"relative_tolerance": 0.1, "max_runs_per_bucket": 10000},
        signature=signature,
        categories=list(DEFAULT_CATEGORIES),
        ssps=[int(value) for value in DEFAULT_SSPS],
        variables_output=list(DEFAULT_VARIABLES_OUTPUT),
        database=DEFAULT_DATABASE,
        raw_data_dir=get_raw_dir(),
        citation_txt_path=get_citation_txt_path(),
        ensure_figures_func=_fake_ensure_figures,
        ensure_figures_guide_func=lambda **kwargs: (cached_guide, False),
        load_saved_figure_files_func=lambda **kwargs: [str(figure_path)],
    )
    assert isinstance(reused_report, ProcessReportAR6)
    assert reused_report.figure_files == [figure_path]
    assert (logs_dir / "summary.log").read_text(encoding="utf-8").strip()

    status = _StatusRecorder()
    regenerated_report = run_process_ar6_workflow(
        study_period=study_period,
        figures=True,
        harmonization=True,
        harmonization_method="reduced_offset",
        refresh=False,
        figure_output_format="png",
        figure_dpi=10,
        sampling_config={"relative_tolerance": 0.1, "max_runs_per_bucket": 10000},
        signature=signature,
        categories=list(DEFAULT_CATEGORIES),
        ssps=[int(value) for value in DEFAULT_SSPS],
        variables_output=list(DEFAULT_VARIABLES_OUTPUT),
        database=DEFAULT_DATABASE,
        raw_data_dir=get_raw_dir(),
        citation_txt_path=get_citation_txt_path(),
        ensure_figures_func=_fake_ensure_figures,
        ensure_figures_guide_func=_fake_ensure_figures_guide,
        load_saved_figure_files_func=lambda **kwargs: None,
        status=status,
    )
    assert isinstance(regenerated_report, ProcessReportAR6)
    assert regenerated_report.figure_files == [figures_dir / "fake.png"]
    assert "[process_ar6] Generating figures" in status.messages
    owned_status_report = run_process_ar6_workflow(
        study_period=study_period,
        figures=True,
        harmonization=True,
        harmonization_method="reduced_offset",
        refresh=False,
        figure_output_format="png",
        figure_dpi=10,
        sampling_config={"relative_tolerance": 0.1, "max_runs_per_bucket": 10000},
        signature=signature,
        categories=list(DEFAULT_CATEGORIES),
        ssps=[int(value) for value in DEFAULT_SSPS],
        variables_output=list(DEFAULT_VARIABLES_OUTPUT),
        database=DEFAULT_DATABASE,
        raw_data_dir=get_raw_dir(),
        citation_txt_path=get_citation_txt_path(),
        ensure_figures_func=_fake_ensure_figures,
        ensure_figures_guide_func=_fake_ensure_figures_guide,
        load_saved_figure_files_func=lambda **kwargs: None,
    )
    assert owned_status_report.figure_files == [figures_dir / "fake.png"]

    stale_file = processed_dir / "stale.txt"
    stale_logs = logs_dir / "stale.txt"
    stale_figure = figures_dir / "stale.txt"
    stale_file.write_text("stale", encoding="utf-8")
    stale_logs.write_text("stale", encoding="utf-8")
    stale_figure.write_text("stale", encoding="utf-8")
    refreshed_report = run_process_ar6_workflow(
        study_period=study_period,
        figures=True,
        harmonization=True,
        harmonization_method="reduced_offset",
        refresh=True,
        figure_output_format="png",
        figure_dpi=10,
        sampling_config={"relative_tolerance": 0.1, "max_runs_per_bucket": 10000},
        signature=signature,
        categories=list(DEFAULT_CATEGORIES),
        ssps=[int(value) for value in DEFAULT_SSPS],
        variables_output=list(DEFAULT_VARIABLES_OUTPUT),
        database=DEFAULT_DATABASE,
        raw_data_dir=get_raw_dir(),
        citation_txt_path=get_citation_txt_path(),
        ensure_figures_func=_fake_ensure_figures,
        ensure_figures_guide_func=_fake_ensure_figures_guide,
        load_saved_figure_files_func=lambda **kwargs: None,
    )
    assert isinstance(refreshed_report, ProcessReportAR6)
    assert not stale_file.exists()
    assert not stale_logs.exists()
    assert not stale_figure.exists()
    assert refreshed_report.figure_files == [figures_dir / "fake.png"]


def test_process_ar6_processing_edge_contracts(
    tmp_path: Path, ar6_dummy_repo: AR6DummyRepo
) -> None:
    with pytest.raises(ValueError):
        resolve_study_period([])
    with pytest.raises(ValueError):
        resolve_study_period(range(2019, 2020))

    assert signature_matches(None, {"demo": True}) is False
    assert summarize_variable_model_scenario_pairs(None) == []

    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)
    renamed_columns = {col: f"Y{col}" for col in explorer.data.columns if str(col).isdigit()}
    no_year_explorer = type(explorer)(
        data=explorer.data.rename(columns=renamed_columns),
    )
    assert filter_and_format_rawdata(
        no_year_explorer,
        {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
    ).equals(_empty_filtered_dataframe())

    na_category_explorer = type(explorer)(
        data=explorer.data.assign(Category_name=pd.NA),
    )
    formatted = filter_and_format_rawdata(
        na_category_explorer,
        {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
    )
    assert formatted["Category_name"].isna().all()

    set_df = pd.DataFrame([[1.0, 2.0]], index=["row"], columns=[2019, 2020])
    _set_year_values(
        set_df,
        row_label="row",
        years=[2019, 2020],
        values=np.array([3.0, 4.0]),
    )
    assert set_df.loc["row", 2019] == 3.0
    assert set_df.loc["row", 2020] == 4.0

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for source_path in [
        ar6_dummy_repo.primap_final_path,
        ar6_dummy_repo.primap_no_rounding_path,
    ]:
        primap_df = pd.read_csv(source_path)
        primap_df = primap_df.drop(columns=["1850"])
        primap_df.to_csv(raw_dir / source_path.name, index=False)
    gcp_rows: list[dict[str, object]] = [{"col0": np.nan, "col1": np.nan} for _ in range(9)]
    gcp_rows.append({"col0": np.nan, "col1": "Bunkers"})
    gcp_rows.append({"col0": 1850, "col1": 0.1})
    for year in range(3000, 3005):
        gcp_rows.append({"col0": year, "col1": 0.1})
    with pd.ExcelWriter(
        raw_dir / ar6_dummy_repo.gcp_path.name,
        engine="xlsxwriter",
    ) as writer:
        pd.DataFrame(gcp_rows).to_excel(
            writer,
            sheet_name="Territorial Emissions",
            index=False,
        )
    with pytest.raises(ValueError):
        process_historical_emissions(f"{raw_dir.as_posix()}/")


def test_process_ar6_build_pathway_and_harmonization_edge_branches(
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)
    shifted_outputs = build_pathway_outputs(
        explorer=explorer,
        categories=list(DEFAULT_CATEGORIES),
        ssps=[int(value) for value in DEFAULT_SSPS],
        variables_output=list(DEFAULT_VARIABLES_OUTPUT),
        study_period=[2025, 2060],
        database_raw_dir=ar6_dummy_repo.raw_dir,
        models_relevant_all=sorted(set(explorer.data["model"])),
        harmonization=True,
        harmonization_method="reduced_offset",
    )
    assert shifted_outputs["harmonization_year"] == 2023
    assert shifted_outputs["harmonization_message"] is not None

    empty_outputs = build_pathway_outputs(
        explorer=explorer,
        categories=["C9"],
        ssps=[1],
        variables_output=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database_raw_dir=ar6_dummy_repo.raw_dir,
        models_relevant_all=sorted(set(explorer.data["model"])),
        harmonization=False,
        harmonization_method="reduced_offset",
    )
    assert empty_outputs["final_all"].empty
    assert empty_outputs["stats_var"].empty

    year_cols = list(range(2000, 2101))
    idx = pd.MultiIndex.from_tuples(
        [("M1", "S1", NET_CO2_WITH_AFOLU), ("M2", "S2", NET_CO2_WITH_AFOLU)],
        names=["model", "scenario", "variable"],
    )
    data_df = pd.DataFrame(
        index=idx,
        columns=["Category", "Category_name", "Ssp_family", "unit", "region"] + year_cols,
        dtype=object,
    )
    data_df.loc[idx[0], ["Category", "Category_name", "Ssp_family", "unit", "region"]] = [
        "C1",
        "C1",
        1,
        "MtCO2/yr",
        "World",
    ]
    data_df.loc[idx[1], ["Category", "Category_name", "Ssp_family", "unit", "region"]] = [
        "C1",
        "C1",
        1,
        "MtCO2/yr",
        "World",
    ]
    for year in year_cols:
        data_df.loc[idx[0], year] = 1.0
        data_df.loc[idx[1], year] = np.nan

    historic_data = pd.DataFrame(
        index=[NET_CO2_WITH_AFOLU], columns=["units"] + year_cols, dtype=object
    )
    historic_data.loc[NET_CO2_WITH_AFOLU, "units"] = "MtCO2/yr"
    for year in year_cols:
        historic_data.loc[NET_CO2_WITH_AFOLU, year] = 10.0

    harmonized_df, harmonization_log = harmonize_emissions(
        data_df=data_df,
        historic_data_df=historic_data,
        study_timeperiod=[2019, 2060],
        requested_harmonization_year=2019,
        harmonization_method="reduced_offset",
    )
    assert harmonization_log.loc[idx[0], "horizon-for-harmonization"] < 2100
    assert "first negative-emissions year" in str(
        harmonization_log.loc[idx[0], "harmonization-method-note"]
    )
    assert harmonized_df.loc[idx[1], list(range(2019, 2101))].isna().all()

    missing_historic_df = historic_data.drop(index=[NET_CO2_WITH_AFOLU])
    _, missing_historic_log = harmonize_emissions(
        data_df=data_df.iloc[[0]].copy(),
        historic_data_df=missing_historic_df,
        study_timeperiod=[2019, 2060],
        requested_harmonization_year=2019,
        harmonization_method="reduced_offset",
    )
    assert pd.isna(missing_historic_log.loc[idx[0], "historic-cumulative"])


def test_process_ar6_generate_figures_and_partial_metadata_branches(
    tmp_path: Path,
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    outputs = _runner_outputs(ar6_dummy_repo, study_period=[2019, 2060], harmonization=True)
    harmonized_data = cast(
        pd.DataFrame,
        outputs["final_all"],
    ).loc[(slice(None), slice(None), NET_CO2_WITH_AFOLU), :]
    original_data = cast(pd.DataFrame, outputs["original_all"]).loc[harmonized_data.index, :]
    harmonization_log = cast(pd.DataFrame, outputs["harmonization_log_all"]).loc[
        harmonized_data.index, :
    ]
    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)
    source_meta = scenario_metadata_from_wide(explorer.data)
    source_meta = source_meta.loc[
        source_meta.index.intersection(harmonized_data.index.droplevel("variable"))
    ]

    messages: list[str] = []

    def fake_sampling_figures(**kwargs):
        kwargs["status_callback"]("Generating sample figure")
        return pd.DataFrame()

    out_paths, convergence_log = generate_ar6_figures(
        output_dir=str(tmp_path / "generated"),
        harmonized_data=harmonized_data,
        original_data=original_data,
        harmonization_log=harmonization_log,
        historical_data=outputs["historical_emissions"],
        source_metadata=source_meta,
        raw_data_dir=str(ar6_dummy_repo.raw_dir),
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1", "C2", "C3", "C4"],
        variables_output=[NET_CO2_WITH_AFOLU],
        figure_output_format="png",
        dpi=10,
        figure_convergence_tol=0.1,
        figure_convergence_max_runs=10000,
        status_callback=messages.append,
        sampling_run_batch_size=10000,
        sampling_stable_checks_required=1,
        write_sampling_figures_func=fake_sampling_figures,
    )
    assert out_paths
    assert convergence_log.empty
    assert any(message.startswith("Generating figure 1/18") for message in messages)
    assert "Generating sample figure" in messages

    workbook_path = tmp_path / "harmonized.xlsx"
    log_path = tmp_path / "harmonization_log.xlsx"
    figures_dir = tmp_path / "figures"
    logs_dir = tmp_path / "logs"
    figures_dir.mkdir()
    logs_dir.mkdir()
    stale_file = figures_dir / "stale.png"
    stale_file.write_text("stale", encoding="utf-8")
    with pd.ExcelWriter(workbook_path, engine="xlsxwriter") as writer:
        harmonized_data.to_excel(writer, sheet_name="HARMONIZED_AR6", merge_cells=False)
        original_data.to_excel(writer, sheet_name="ORIGINAL_AR6", merge_cells=False)
        cast(pd.DataFrame, outputs["historical_emissions"]).to_excel(
            writer, sheet_name="HISTORICAL_PRIMAP_GCP", merge_cells=False
        )
    with pd.ExcelWriter(log_path, engine="xlsxwriter") as writer:
        harmonization_log.to_excel(writer, sheet_name="HARMONIZATION_LOG", merge_cells=False)

    def fake_generate_figures(**kwargs):
        metadata_callback = kwargs["metadata_callback"]
        metadata_callback([str(Path(kwargs["output_dir"]) / "partial.png")])
        final_path = Path(kwargs["output_dir"]) / "final.png"
        final_path.write_text("figure", encoding="utf-8")
        return [str(final_path)], pd.DataFrame()

    figure_files, reused = ensure_figures(
        out_file=workbook_path,
        log_file=log_path,
        figures_dir=figures_dir,
        figures_metadata_file=tmp_path / "figures_meta.json",
        logs_dir=logs_dir,
        variables_output=[NET_CO2_WITH_AFOLU],
        study_period=[2019, 2060],
        database="ar6-public",
        categories=["C1"],
        figure_output_format="png",
        figure_dpi=10,
        harmonization_method="reduced_offset",
        figure_convergence_tol=0.1,
        figure_convergence_max_runs=10000,
        refresh=True,
        source_metadata=source_meta,
        raw_data_dir=ar6_dummy_repo.raw_dir,
        generate_figures_func=fake_generate_figures,
    )
    assert reused is False
    assert figure_files == [str(figures_dir / "final.png")]
    assert not stale_file.exists()
