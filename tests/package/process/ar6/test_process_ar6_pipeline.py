from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.package.helpers.ar6_imports import (
    collection_config,
    collection_explorer,
    processing_contracts,
    processing_entry,
    processing_derived_variables,
    processing_harmonization,
    processing_historical,
    processing_loaders,
    processing_metadata,
    processing_paths,
    processing_preprocessing,
    processing_processing_modes,
    processing_raw_inputs,
    processing_reports,
    processing_report_summaries,
    processing_runtime_helpers,
    processing_study_period,
    processing_text_outputs,
    processing_writers,
)
from tests.package.helpers.ar6_dummy_repo import AR6DummyRepo

DEFAULT_CATEGORIES = collection_config.DEFAULT_CATEGORIES
DEFAULT_DATABASE = collection_config.DEFAULT_DATABASE
DEFAULT_SSPS = collection_config.DEFAULT_SSPS
DEFAULT_VARIABLES_OUTPUT = collection_config.DEFAULT_VARIABLES_OUTPUT
GROSS_CO2_WITH_AFOLU = collection_config.GROSS_CO2_WITH_AFOLU
GROSS_ALT_CO2_WO_AFOLU = collection_config.GROSS_ALT_CO2_WO_AFOLU
GROSS_ALT_KYOTO_WO_AFOLU = collection_config.GROSS_ALT_KYOTO_WO_AFOLU
NET_CO2_WITH_AFOLU = collection_config.NET_CO2_WITH_AFOLU
NET_CO2_WO_AFOLU = collection_config.NET_CO2_WO_AFOLU
NET_KYOTO_WITH_AFOLU = collection_config.NET_KYOTO_WITH_AFOLU
NET_KYOTO_WO_AFOLU = collection_config.NET_KYOTO_WO_AFOLU
RAW_CH4_AFOLU = collection_config.RAW_CH4_AFOLU
RAW_CO2_AFOLU = collection_config.RAW_CO2_AFOLU
RAW_CO2_AFOLU_LAND = collection_config.RAW_CO2_AFOLU_LAND
RAW_CO2_ENERGY = collection_config.RAW_CO2_ENERGY
RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES = collection_config.RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES
RAW_CO2_INDUSTRIAL_PROCESSES = collection_config.RAW_CO2_INDUSTRIAL_PROCESSES
RAW_CO2_OTHER = collection_config.RAW_CO2_OTHER
RAW_CO2_WASTE = collection_config.RAW_CO2_WASTE
RAW_CO2_WITH_AFOLU = collection_config.RAW_CO2_WITH_AFOLU
RAW_KYOTO_WITH_AFOLU = collection_config.RAW_KYOTO_WITH_AFOLU
RAW_N2O_AFOLU = collection_config.RAW_N2O_AFOLU
RAW_SEQUESTRATION_COMPONENTS = collection_config.RAW_SEQUESTRATION_COMPONENTS
SEQUESTRATION_TOTAL = collection_config.SEQUESTRATION_TOTAL
read_explorer_csv = collection_explorer.read_explorer_csv
process_ar6 = processing_entry.process_ar6
budget_stats_sheet_name = processing_contracts.budget_stats_sheet_name
final_pathways_sheet_name = processing_contracts.final_pathways_sheet_name
harmonization_log_workbook_name = processing_contracts.harmonization_log_workbook_name
processed_workbook_name = processing_contracts.processed_workbook_name
read_json = processing_metadata.read_json
signature_matches = processing_metadata.signature_matches
write_json = processing_metadata.write_json
build_process_metadata_payload = processing_metadata.build_process_metadata_payload
get_figures_dir = processing_paths.get_figures_dir
get_logs_dir = processing_paths.get_logs_dir
get_processed_dir = processing_paths.get_processed_dir
get_processed_scope_dir = processing_paths.get_processed_scope_dir
get_scope_dirs = processing_paths.get_scope_dirs
study_period_folder = processing_paths.study_period_folder
ProcessReportAR6 = processing_reports.ProcessReportAR6
build_process_report = processing_reports.build_process_report
deserialize_variable_coverage_summary_counts = (
    processing_report_summaries.deserialize_variable_coverage_summary_counts
)
serialize_variable_coverage_summary_counts = (
    processing_report_summaries.serialize_variable_coverage_summary_counts
)
summarize_variable_model_scenario_pairs = (
    processing_report_summaries.summarize_variable_model_scenario_pairs
)
excel_readme_sheet = processing_text_outputs.excel_readme_sheet
figure_sampling_log_columns_explanation_text = (
    processing_text_outputs.figure_sampling_log_columns_explanation_text
)
log_columns_explanation_text = processing_text_outputs.log_columns_explanation_text
processing_citation_text = processing_text_outputs.processing_citation_text
build_dropped_rows_df = processing_writers.build_dropped_rows_df
write_harmonization_log_workbook = processing_writers.write_harmonization_log_workbook
write_processed_workbook = processing_writers.write_processed_workbook
_template_ssp_scenario = processing_writers._template_ssp_scenario
_initial_offset_horizon_year = processing_harmonization._initial_offset_horizon_year
get_stats_from_series = processing_harmonization.get_stats_from_series
harmonize_emissions = processing_harmonization.harmonize_emissions
stats_from_retained_pathways = processing_harmonization.stats_from_retained_pathways
_add_year_values = processing_historical._add_year_values
_set_year_values = processing_historical._set_year_values
process_historical_emissions = processing_historical.process_historical_emissions
scenario_metadata_from_wide = processing_loaders.scenario_metadata_from_wide
CH4_AR6_GWP100 = processing_preprocessing.CH4_AR6_GWP100
KT_TO_MT = processing_preprocessing.KT_TO_MT
N2O_AR6_GWP100 = processing_preprocessing.N2O_AR6_GWP100
YEAR_COLUMNS = processing_preprocessing.YEAR_COLUMNS
build_final_harmonized_emission_variables = (
    processing_derived_variables.build_final_harmonized_emission_variables
)
build_pre_harmonization_variables = processing_derived_variables.build_pre_harmonization_variables
CO2_RECONSTRUCTION_DROP_REASON = processing_derived_variables.CO2_RECONSTRUCTION_DROP_REASON
CO2_WO_AFOLU_NOT_PRODUCED_REASON = processing_derived_variables.CO2_WO_AFOLU_NOT_PRODUCED_REASON
KYOTO_WO_AFOLU_NOT_PRODUCED_REASON = processing_derived_variables.KYOTO_WO_AFOLU_NOT_PRODUCED_REASON
MISSING_REQUIRED_CO2_COVERAGE_ROW_REASON = (
    processing_derived_variables.MISSING_REQUIRED_CO2_COVERAGE_ROW_REASON
)
MISSING_REQUIRED_END_YEAR_REASON = processing_derived_variables.MISSING_REQUIRED_END_YEAR_REASON
NEGATIVE_GROSS_EMISSIONS_DROP_REASON = (
    processing_derived_variables.NEGATIVE_GROSS_EMISSIONS_DROP_REASON
)
NEGATIVE_SEQUESTRATION_DROP_REASON = processing_derived_variables.NEGATIVE_SEQUESTRATION_DROP_REASON
_empty_filtered_dataframe = processing_preprocessing._empty_filtered_dataframe
filter_and_format_rawdata = processing_preprocessing.filter_and_format_rawdata
interpolate_and_check = processing_preprocessing.interpolate_and_check
build_pathway_outputs = processing_processing_modes.build_pathway_outputs
require_ar6_historical_figure_reference = (
    processing_raw_inputs.require_ar6_historical_figure_reference
)
require_downloaded_ar6_raw_inputs = processing_raw_inputs.require_downloaded_ar6_raw_inputs
DEFAULT_HARMONIZATION_METHOD = processing_runtime_helpers.DEFAULT_HARMONIZATION_METHOD
HARMONIZATION_METHOD_OPTIONS = processing_runtime_helpers.HARMONIZATION_METHOD_OPTIONS
normalize_harmonization_method = processing_runtime_helpers.validate_harmonization_method
process_signature = processing_runtime_helpers.process_signature
show_stage = processing_runtime_helpers.show_stage
resolve_study_period = processing_study_period.resolve_study_period
validate_study_period_in_ar6 = processing_study_period.validate_study_period_in_ar6


class DummyStatus:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def show(self, message: str) -> None:
        self.messages.append(message)


def _minimal_ar6_variable_frame(
    *,
    variables: dict[str, tuple[float, str]],
    model: str = "M",
    scenario: str = "S",
) -> pd.DataFrame:
    rows: list[dict[object, object]] = []
    for variable, (value, unit) in variables.items():
        row: dict[object, object] = {
            "model": model,
            "scenario": scenario,
            "variable": variable,
            "Category": "C1",
            "Category_name": "Category 1",
            "Ssp_family": 1,
            "unit": unit,
            "region": "World",
        }
        for year in YEAR_COLUMNS:
            row[year] = value
        rows.append(row)
    return pd.DataFrame(rows).set_index(["model", "scenario", "variable"])


def test_process_ar6_path_contracts_and_metadata(project_repo: Path) -> None:
    del project_repo
    assert study_period_folder([2019, 2060], harmonization=False) == "2019-2060_no_harmonization"
    assert (
        study_period_folder(
            [2019, 2060],
            harmonization=True,
            harmonization_method="constant_offset",
        )
        == "2019-2060_harmonization_constant_offset"
    )
    processed_scope_dir = get_processed_scope_dir(
        [2019, 2060], harmonization=True, harmonization_method="constant_offset"
    )
    processed_dir = get_processed_dir(
        [2019, 2060], harmonization=True, harmonization_method="constant_offset"
    )
    logs_dir = get_logs_dir([2019, 2060], harmonization=False)
    figures_dir = get_figures_dir(
        [2019, 2060], harmonization=True, harmonization_method="reduced_offset"
    )
    assert not processed_scope_dir.exists()
    assert not processed_dir.exists()
    assert not logs_dir.exists()
    assert not figures_dir.exists()
    assert processed_scope_dir.name.endswith("harmonization_constant_offset")
    assert processed_dir.name == "process_ar6"
    assert processed_dir.parent == processed_scope_dir
    assert logs_dir.name == "logs"
    assert logs_dir.parent.name == "process_ar6"
    assert logs_dir.parent.parent.name.endswith("no_harmonization")
    assert figures_dir.name == "figures"
    assert figures_dir.parent.name == "process_ar6"
    assert figures_dir.parent.parent.name.endswith("harmonization_reduced_offset")
    assert len(get_scope_dirs([2019, 2060], harmonization=False)) == 3

    meta_path = processed_dir / "meta.json"
    payload = {"arguments": {"a": 1}}
    assert write_json(meta_path, payload) == meta_path
    assert read_json(meta_path)["arguments"] == {"a": 1}
    assert signature_matches(read_json(meta_path), {"a": 1}) is True
    assert signature_matches(read_json(meta_path), {"a": 2}) is False
    assert read_json(processed_dir / "missing.json") is None


def test_process_ar6_runtime_study_period_and_raw_input_guards(
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    assert resolve_study_period(range(2019, 2022)) == [2019, 2021]
    assert resolve_study_period([2019, 2020, 2021]) == [2019, 2021]
    assert resolve_study_period(range(2019, 2061)) == [2019, 2060]
    assert resolve_study_period([2009, 2010]) == [2009, 2010]
    with pytest.raises(ValueError):
        resolve_study_period(2019)
    with pytest.raises(ValueError):
        resolve_study_period([2019])
    with pytest.raises(ValueError):
        resolve_study_period(range(2019, 2023, 2))

    validate_study_period_in_ar6([2019, 2060], list(range(2010, 2101)))
    with pytest.raises(ValueError):
        validate_study_period_in_ar6([2019, 2060], [])
    with pytest.raises(ValueError):
        validate_study_period_in_ar6([2000, 2060], list(range(2010, 2101)))
    with pytest.raises(ValueError):
        validate_study_period_in_ar6([2009, 2010], list(range(2010, 2101)))

    assert (
        normalize_harmonization_method(
            harmonization=False,
            harmonization_method="constant_offset",
        )
        == DEFAULT_HARMONIZATION_METHOD
    )
    assert (
        normalize_harmonization_method(
            harmonization=True,
            harmonization_method="offset",
        )
        == "offset"
    )
    assert "offset" in HARMONIZATION_METHOD_OPTIONS
    with pytest.raises(ValueError):
        normalize_harmonization_method(harmonization=True, harmonization_method="bad")

    signature = process_signature([2019, 2060], True, "reduced_offset")
    assert signature["harmonization_method"] == "reduced_offset"
    assert "timeperiod" not in signature
    assert "harmonization_method" not in process_signature([2019, 2060], False, "constant_offset")

    citation_text = ar6_dummy_repo.citation_txt_path.read_text(encoding="utf-8")
    assert "Raw citation block" in citation_text

    status = DummyStatus()
    show_stage(status, "Loading")
    show_stage(None, "Ignored")
    assert status.messages == ["[process_ar6] Loading"]

    require_downloaded_ar6_raw_inputs(
        raw_data_dir=ar6_dummy_repo.raw_dir,
        citation_txt_path=ar6_dummy_repo.citation_txt_path,
        database=DEFAULT_DATABASE,
    )
    require_ar6_historical_figure_reference(raw_data_dir=ar6_dummy_repo.raw_dir)
    ar6_dummy_repo.overlay_path.unlink()
    with pytest.raises(RuntimeError):
        require_ar6_historical_figure_reference(raw_data_dir=ar6_dummy_repo.raw_dir)
    ar6_dummy_repo.overlay_path.write_text("restored", encoding="utf-8")
    ar6_dummy_repo.citation_txt_path.unlink()
    with pytest.raises(RuntimeError):
        require_downloaded_ar6_raw_inputs(
            raw_data_dir=ar6_dummy_repo.raw_dir,
            citation_txt_path=ar6_dummy_repo.citation_txt_path,
            database=DEFAULT_DATABASE,
        )


def test_process_ar6_template_ssp_formatter_covers_numeric_and_string_contracts() -> None:
    assert _template_ssp_scenario(2.0) == "SSP2"
    assert _template_ssp_scenario("SSP3") == "SSP3"


def test_process_ar6_loader_and_preprocessing_contracts(ar6_dummy_repo: AR6DummyRepo) -> None:
    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)
    source_meta = scenario_metadata_from_wide(explorer.data)
    assert source_meta.index.names == ["model", "scenario"]
    assert source_meta.loc[("M1", "S1"), "Category"] == "C1"
    no_meta = scenario_metadata_from_wide(
        explorer.data.loc[:, ["model", "scenario", "variable", "unit", "region"]]
    )
    assert list(no_meta.index.names) == ["model", "scenario"]
    no_category_name_explorer = type(explorer)(
        data=explorer.data.drop(columns=["Category_name"]),
    )
    no_category_name_filtered = filter_and_format_rawdata(
        no_category_name_explorer,
        {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
    )
    assert no_category_name_filtered["Category_name"].isna().all()
    missing_category_name_explorer = type(explorer)(
        data=explorer.data.assign(Category_name=pd.NA),
    )
    missing_category_name_filtered = filter_and_format_rawdata(
        missing_category_name_explorer,
        {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
    )
    assert missing_category_name_filtered["Category_name"].isna().all()
    empty_category_name_explorer = type(explorer)(
        data=explorer.data.assign(Category_name=""),
    )
    empty_category_name_filtered = filter_and_format_rawdata(
        empty_category_name_explorer,
        {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
    )
    assert empty_category_name_filtered["Category_name"].isna().all()
    missing_meta = scenario_metadata_from_wide(
        explorer.data.assign(Category_name=pd.NA).loc[
            :, ["model", "scenario", "variable", "unit", "region", "Category_name"]
        ]
    )
    assert pd.isna(missing_meta.loc[("M1", "S1"), "Category_name"])
    conflicting_meta = pd.concat(
        [
            explorer.data,
            explorer.data.iloc[[0]].assign(
                Category="C9", variable="Synthetic conflicting variable"
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError):
        scenario_metadata_from_wide(conflicting_meta)

    empty_df = _empty_filtered_dataframe()
    assert empty_df.empty
    assert list(empty_df.index.names) == ["model", "scenario", "variable"]

    missing_filtered = filter_and_format_rawdata(
        explorer,
        {"category": "C9", "ssp_family": 9, "model": ["ALL", ["M1"]]},
    )
    assert missing_filtered.empty

    filtered = filter_and_format_rawdata(
        explorer,
        {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1", "M5"]]},
    )
    assert "Category" in filtered.columns
    assert 2019 in filtered.columns
    assert ("M1", "S1", RAW_KYOTO_WITH_AFOLU) in filtered.index
    duplicated_explorer = type(explorer)(
        data=pd.concat([explorer.data, explorer.data.iloc[[0]]], ignore_index=True),
    )
    with pytest.raises(ValueError):
        filter_and_format_rawdata(
            duplicated_explorer,
            {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
        )
    year_column = next(column for column in explorer.data.columns if str(column).isdigit())
    conflicting_row = explorer.data.iloc[[0]].copy()
    conflicting_row.loc[:, year_column] = 999.0
    conflicting_explorer = type(explorer)(
        data=pd.concat([explorer.data, conflicting_row], ignore_index=True),
    )
    with pytest.raises(ValueError):
        filter_and_format_rawdata(
            conflicting_explorer,
            {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
        )
    missing_vetting_explorer = type(explorer)(
        data=explorer.data.drop(columns=["Vetting_future"]),
    )
    with pytest.raises(ValueError):
        filter_and_format_rawdata(
            missing_vetting_explorer,
            {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
        )
    conflicting_category_name = explorer.data.iloc[[0]].copy()
    conflicting_category_name.loc[:, "Category_name"] = "Conflicting category"
    conflicting_category_name.loc[:, "variable"] = "Synthetic category metadata conflict"
    conflicting_category_explorer = type(explorer)(
        data=pd.concat([explorer.data, conflicting_category_name], ignore_index=True),
    )
    with pytest.raises(ValueError):
        filter_and_format_rawdata(
            conflicting_category_explorer,
            {"category": "C1", "ssp_family": 1, "model": ["ALL", ["M1"]]},
        )

    interpolated = interpolate_and_check(filtered)
    retained_tail = interpolated.loc[("M1", "S1", RAW_CO2_WITH_AFOLU), 2100]
    truncated_tail = interpolated.loc[("M5", "S5", RAW_CO2_WITH_AFOLU), 2100]
    assert pd.notna(retained_tail)
    assert pd.isna(truncated_tail)

    processed_variables, derived_drop_log = build_pre_harmonization_variables(
        interpolated,
        study_start_year=2019,
    )
    assert ("M1", "S1", NET_KYOTO_WO_AFOLU) in processed_variables.index
    assert ("M1", "S1", NET_CO2_WO_AFOLU) in processed_variables.index
    assert ("M5", "S5", RAW_CO2_WITH_AFOLU) not in processed_variables.index
    assert ("M1", "S1", GROSS_ALT_KYOTO_WO_AFOLU) not in processed_variables.index
    assert ("M1", "S1", GROSS_ALT_CO2_WO_AFOLU) not in processed_variables.index
    final_variables, gross_drop_log, _before_gross = build_final_harmonized_emission_variables(
        processed_variables
    )
    assert ("M1", "S1", GROSS_ALT_KYOTO_WO_AFOLU) in final_variables.index
    assert ("M1", "S1", GROSS_ALT_CO2_WO_AFOLU) in final_variables.index
    assert derived_drop_log["drop_reason"].tolist() == [MISSING_REQUIRED_END_YEAR_REASON]
    assert derived_drop_log["drop_stage"].tolist() == ["pre_reconstruction_coverage_check"]
    assert gross_drop_log.empty
    kyoto_afolu = processed_variables.loc[("M1", "S1", "Emissions|Kyoto Gases|AFOLU"), 2019]
    expected_kyoto_afolu = (
        interpolated.loc[("M1", "S1", RAW_CO2_AFOLU), 2019]
        + CH4_AR6_GWP100 * interpolated.loc[("M1", "S1", RAW_CH4_AFOLU), 2019]
        + (N2O_AR6_GWP100 * KT_TO_MT) * interpolated.loc[("M1", "S1", RAW_N2O_AFOLU), 2019]
    )
    assert kyoto_afolu == pytest.approx(expected_kyoto_afolu)

    filtered_missing_afolu = filter_and_format_rawdata(
        explorer,
        {"category": "C4", "ssp_family": 4, "model": ["ALL", ["M4"]]},
    )
    processed_missing_afolu, drop_log = build_pre_harmonization_variables(
        filtered_missing_afolu,
        study_start_year=2019,
    )
    assert ("M4", "S4", NET_CO2_WITH_AFOLU) in processed_missing_afolu.index
    assert ("M4", "S4", NET_CO2_WO_AFOLU) not in processed_missing_afolu.index
    assert ("M4", "S4", NET_KYOTO_WITH_AFOLU) in processed_missing_afolu.index
    assert ("M4", "S4", NET_KYOTO_WO_AFOLU) in processed_missing_afolu.index
    assert drop_log["variable"].tolist() == [NET_CO2_WO_AFOLU]
    assert drop_log["retained_variable"].tolist() == [RAW_CO2_WITH_AFOLU]
    assert drop_log["drop_reason"].tolist() == [CO2_WO_AFOLU_NOT_PRODUCED_REASON]

    start_missing = interpolated.loc[("M1", "S1", slice(None)), :].copy()
    start_missing.loc[:, 2019] = np.nan
    _filtered_start_missing, start_missing_log = build_pre_harmonization_variables(
        start_missing,
        study_start_year=2019,
    )
    assert start_missing_log["drop_reason"].tolist() == ["missing_value_at_study_start_year_2019"]


def test_process_ar6_derived_variable_edge_branches() -> None:
    empty_processed, empty_log = build_pre_harmonization_variables(
        pd.DataFrame(),
        study_start_year=2019,
    )
    assert empty_processed.empty
    assert empty_log.empty
    empty_final, empty_gross_log, empty_before_gross = build_final_harmonized_emission_variables(
        pd.DataFrame()
    )
    assert empty_final.empty
    assert empty_gross_log.empty
    assert empty_before_gross.empty

    missing_required_co2 = _minimal_ar6_variable_frame(
        variables={RAW_SEQUESTRATION_COMPONENTS[0]: (2.0, "MtCO2/yr")}
    )
    processed_missing_required_co2, missing_required_co2_log = build_pre_harmonization_variables(
        missing_required_co2,
        study_start_year=2019,
    )
    assert processed_missing_required_co2.empty
    assert missing_required_co2_log["drop_reason"].tolist() == [
        MISSING_REQUIRED_CO2_COVERAGE_ROW_REASON
    ]

    negative_sequestration = _minimal_ar6_variable_frame(
        variables={
            RAW_CO2_WITH_AFOLU: (1.0, "MtCO2/yr"),
            RAW_CO2_AFOLU: (0.0, "MtCO2/yr"),
            RAW_CO2_OTHER: (0.0, "MtCO2/yr"),
            RAW_CO2_WASTE: (0.0, "MtCO2/yr"),
            RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES: (1.0, "MtCO2/yr"),
            RAW_SEQUESTRATION_COMPONENTS[0]: (-1.0, "MtCO2/yr"),
        }
    )
    processed_negative_sequestration, negative_sequestration_log = (
        build_pre_harmonization_variables(negative_sequestration, study_start_year=2019)
    )
    assert processed_negative_sequestration.empty
    assert set(negative_sequestration_log["drop_reason"]) == {NEGATIVE_SEQUESTRATION_DROP_REASON}

    failed_reconstruction = _minimal_ar6_variable_frame(
        variables={
            RAW_CO2_WITH_AFOLU: (100.0, "MtCO2/yr"),
            RAW_CO2_AFOLU_LAND: (0.0, "MtCO2/yr"),
            RAW_CO2_OTHER: (0.0, "MtCO2/yr"),
            RAW_CO2_WASTE: (0.0, "MtCO2/yr"),
            RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES: (50.0, "MtCO2/yr"),
        }
    )
    processed_failed, failed_log = build_pre_harmonization_variables(
        failed_reconstruction,
        study_start_year=2019,
    )
    assert processed_failed.empty
    assert failed_log["drop_stage"].tolist() == ["co2_reconstruction_check"]
    assert failed_log["drop_reason"].tolist() == [CO2_RECONSTRUCTION_DROP_REASON]

    indirect_reconstruction = _minimal_ar6_variable_frame(
        variables={
            RAW_CO2_WITH_AFOLU: (100.0, "MtCO2/yr"),
            RAW_CO2_AFOLU: (5.0, "MtCO2/yr"),
            RAW_CO2_OTHER: (3.0, "MtCO2/yr"),
            RAW_CO2_WASTE: (2.0, "MtCO2/yr"),
            RAW_CO2_ENERGY: (60.0, "MtCO2/yr"),
            RAW_CO2_INDUSTRIAL_PROCESSES: (30.0, "MtCO2/yr"),
        }
    )
    processed_indirect, indirect_log = build_pre_harmonization_variables(
        indirect_reconstruction,
        study_start_year=2019,
    )
    assert ("M", "S", NET_CO2_WITH_AFOLU) in processed_indirect.index
    assert indirect_log.empty

    negative_gross = _minimal_ar6_variable_frame(
        variables={
            RAW_CO2_WITH_AFOLU: (-1.0, "MtCO2/yr"),
            RAW_CO2_AFOLU: (0.0, "MtCO2/yr"),
            RAW_CO2_OTHER: (0.0, "MtCO2/yr"),
            RAW_CO2_WASTE: (0.0, "MtCO2/yr"),
            RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES: (-1.0, "MtCO2/yr"),
            RAW_SEQUESTRATION_COMPONENTS[0]: (0.1, "MtCO2/yr"),
        }
    )
    pre_negative_gross, pre_negative_gross_log = build_pre_harmonization_variables(
        negative_gross,
        study_start_year=2019,
    )
    processed_negative_gross, negative_gross_log, _before_sign_filter = (
        build_final_harmonized_emission_variables(pre_negative_gross)
    )
    assert pre_negative_gross_log.empty
    dropped_gross_variables = set(negative_gross_log["variable"])
    assert ("M", "S", NET_CO2_WITH_AFOLU) in processed_negative_gross.index
    assert ("M", "S", SEQUESTRATION_TOTAL) in processed_negative_gross.index
    assert ("M", "S", GROSS_CO2_WITH_AFOLU) not in processed_negative_gross.index
    assert not dropped_gross_variables.intersection(
        set(processed_negative_gross.index.get_level_values("variable"))
    )
    assert set(negative_gross_log["drop_reason"]) == {NEGATIVE_GROSS_EMISSIONS_DROP_REASON}

    kyoto_without_common_afolu = pd.concat(
        [
            _minimal_ar6_variable_frame(
                variables={
                    RAW_CO2_WITH_AFOLU: (1.0, "MtCO2/yr"),
                    RAW_CO2_OTHER: (0.0, "MtCO2/yr"),
                    RAW_CO2_WASTE: (0.0, "MtCO2/yr"),
                    RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES: (1.0, "MtCO2/yr"),
                    RAW_KYOTO_WITH_AFOLU: (100.0, "MtCO2eq/yr"),
                },
                model="M1",
                scenario="S1",
            ),
            _minimal_ar6_variable_frame(
                variables={
                    RAW_CO2_WITH_AFOLU: (1.0, "MtCO2/yr"),
                    RAW_CO2_OTHER: (0.0, "MtCO2/yr"),
                    RAW_CO2_WASTE: (0.0, "MtCO2/yr"),
                    RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES: (0.0, "MtCO2/yr"),
                    RAW_CO2_AFOLU: (1.0, "MtCO2/yr"),
                    RAW_CH4_AFOLU: (1.0, "MtCH4/yr"),
                    RAW_N2O_AFOLU: (1.0, "ktN2O/yr"),
                },
                model="M2",
                scenario="S2",
            ),
        ]
    )
    processed_kyoto, kyoto_log = build_pre_harmonization_variables(
        kyoto_without_common_afolu,
        study_start_year=2019,
    )
    assert ("M1", "S1", NET_KYOTO_WITH_AFOLU) in processed_kyoto.index
    assert set(kyoto_log["drop_reason"]) == {
        CO2_WO_AFOLU_NOT_PRODUCED_REASON,
        KYOTO_WO_AFOLU_NOT_PRODUCED_REASON,
    }


def test_process_ar6_historical_and_harmonization_contracts(ar6_dummy_repo: AR6DummyRepo) -> None:
    historical_df = process_historical_emissions(f"{ar6_dummy_repo.raw_dir.as_posix()}/")
    assert "Emissions|CO2|Bunkers" in historical_df.index
    assert historical_df.loc[NET_KYOTO_WITH_AFOLU, "units"] == "MtCO2eq/yr"
    assert historical_df.loc[NET_CO2_WITH_AFOLU, "units"] == "MtCO2/yr"

    update_df = pd.DataFrame([[1.0, 2.0]], index=["row"], columns=[2019, 2020])
    _add_year_values(update_df, row_label="row", years=[2019, 2020], values=np.array([0.5, 1.5]))
    assert update_df.loc["row", 2020] == pytest.approx(3.5)
    update_df.loc["row", 2020] = np.nan
    with pytest.raises(RuntimeError):
        _add_year_values(
            update_df, row_label="row", years=[2019, 2020], values=np.array([1.0, 1.0])
        )

    set_df = pd.DataFrame(columns=[2019, 2020], dtype=float)
    _set_year_values(set_df, row_label="new-row", years=[2019, 2020], values=np.array([1.0, 2.0]))
    assert set_df.loc["new-row", 2020] == 2.0

    assert get_stats_from_series(pd.Series([1.0, 2.0, 3.0])) == {
        "median": 2.0,
        "mean": 2.0,
        "min": 1.0,
        "max": 3.0,
    }
    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)
    pathway_outputs = build_pathway_outputs(
        explorer=explorer,
        categories=list(DEFAULT_CATEGORIES),
        ssps=[int(value) for value in DEFAULT_SSPS],
        variables_output=list(DEFAULT_VARIABLES_OUTPUT),
        study_period=[2019, 2060],
        database_raw_dir=ar6_dummy_repo.raw_dir,
        models_relevant_all=sorted(set(explorer.data["model"])),
        harmonization=True,
        harmonization_method="reduced_offset",
    )
    final_all = pathway_outputs["final_all"]
    original_all = pathway_outputs["original_all"]
    harmonization_log_all = pathway_outputs["harmonization_log_all"]
    assert not final_all.empty
    assert harmonization_log_all is not None
    assert "harmonization-method-note" in harmonization_log_all.columns
    assert "offset-variant-used" in harmonization_log_all.columns
    assert SEQUESTRATION_TOTAL in final_all.index.get_level_values("variable")
    assert GROSS_ALT_KYOTO_WO_AFOLU in final_all.index.get_level_values("variable")
    assert GROSS_ALT_KYOTO_WO_AFOLU not in original_all.index.get_level_values("variable")
    assert set(harmonization_log_all.index.get_level_values("variable")).issubset(
        {NET_CO2_WITH_AFOLU, NET_CO2_WO_AFOLU, NET_KYOTO_WITH_AFOLU, NET_KYOTO_WO_AFOLU}
    )

    stats_df = stats_from_retained_pathways(final_all, NET_CO2_WITH_AFOLU, [2019, 2060])
    assert ("C1", "all") in stats_df.index

    harmonized_constant, harmonization_log_constant = harmonize_emissions(
        data_df=original_all,
        historic_data_df=historical_df,
        study_timeperiod=[2019, 2060],
        requested_harmonization_year=2019,
        harmonization_method="constant_offset",
    )
    assert min(col for col in harmonized_constant.columns if isinstance(col, int)) == 2019
    assert set(harmonization_log_constant["harmonization-method"].dropna()) == {"constant_offset"}

    extra_row = original_all.iloc[[0]].copy()
    extra_index = pd.MultiIndex.from_tuples(
        [("MX", "SX", "Emissions|CH4")],
        names=original_all.index.names,
    )
    extra_row.index = extra_index
    harmonized_extra, log_extra = harmonize_emissions(
        data_df=pd.concat([original_all, extra_row]),
        historic_data_df=historical_df,
        study_timeperiod=[2019, 2060],
        requested_harmonization_year=2019,
        harmonization_method="reduced_offset",
    )
    assert ("MX", "SX", "Emissions|CH4") in harmonized_extra.index
    assert ("MX", "SX", "Emissions|CH4") in log_extra.index
    assert (
        _initial_offset_horizon_year(
            row_end_year=2100,
            model_netzero_year=float("nan"),
            offset_variant="constant_offset",
        )
        == 2100
    )
    assert (
        _initial_offset_horizon_year(
            row_end_year=2100,
            model_netzero_year=2050.0,
            offset_variant="reduced_offset",
        )
        == 2050
    )


def test_process_ar6_text_outputs_writers_and_reports(
    tmp_path: Path, ar6_dummy_repo: AR6DummyRepo
) -> None:
    raw_citation = "raw citation text"
    assert "No raw data citation text file was found." in processing_citation_text(
        "", harmonization=False
    )
    assert "PRIMAP-hist" in processing_citation_text(raw_citation, harmonization=True)
    assert "harmonization-method" in log_columns_explanation_text()
    sampling_text = figure_sampling_log_columns_explanation_text()
    assert "stable_checks_required" in sampling_text
    assert "40000 completed runs per bucket" in sampling_text
    generated_texts = (
        processing_citation_text(raw_citation * 30, harmonization=True),
        log_columns_explanation_text(),
        sampling_text,
    )
    assert all(len(line) <= 100 for text in generated_texts for line in text.splitlines())
    assert "HISTORICAL_PRIMAP_GCP" in excel_readme_sheet(True)["sheet"].tolist()
    assert "RETAINED_AR6_FILTERED" in excel_readme_sheet(False)["sheet"].tolist()

    assert processed_workbook_name(harmonization=True) == "harmonized_ar6_public.xlsx"
    assert processed_workbook_name(harmonization=False) == "filtered_original_ar6_public.xlsx"
    assert harmonization_log_workbook_name() == "harmonized_ar6_public_log.xlsx"
    assert final_pathways_sheet_name(harmonization=True) == "HARMONIZED_AR6"
    assert budget_stats_sheet_name(harmonization=False) == "BUDGET_STATS_RETAINED"

    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)
    source_meta = scenario_metadata_from_wide(explorer.data)
    pathway_outputs = build_pathway_outputs(
        explorer=explorer,
        categories=list(DEFAULT_CATEGORIES),
        ssps=[int(value) for value in DEFAULT_SSPS],
        variables_output=list(DEFAULT_VARIABLES_OUTPUT),
        study_period=[2019, 2060],
        database_raw_dir=ar6_dummy_repo.raw_dir,
        models_relevant_all=sorted(set(explorer.data["model"])),
        harmonization=True,
        harmonization_method="reduced_offset",
    )
    output_file = tmp_path / "processed.xlsx"
    write_processed_workbook(
        harmonization=True,
        output_file=output_file,
        readme_df=excel_readme_sheet(harmonization=True),
        citations_text=processing_citation_text(raw_citation, True),
        final_all=pathway_outputs["final_all"],
        original_all=pathway_outputs["original_all"],
        source_meta=source_meta,
        stats_var=pathway_outputs["stats_var"],
        historical_emissions=pathway_outputs["historical_emissions"],
    )
    with pd.ExcelFile(output_file, engine="calamine") as workbook:
        assert "README" in workbook.sheet_names
        assert "HISTORICAL_PRIMAP_GCP" in workbook.sheet_names

    output_file_no_harmo = tmp_path / "processed_no_harmo.xlsx"
    write_processed_workbook(
        harmonization=False,
        output_file=output_file_no_harmo,
        readme_df=excel_readme_sheet(harmonization=False),
        citations_text=processing_citation_text(raw_citation, False),
        final_all=pathway_outputs["final_all"],
        original_all=pathway_outputs["original_all"],
        source_meta=source_meta,
        stats_var=pathway_outputs["stats_var"],
        historical_emissions=None,
    )
    with pd.ExcelFile(output_file_no_harmo, engine="calamine") as workbook:
        assert "HISTORICAL_PRIMAP_GCP" not in workbook.sheet_names

    log_file = tmp_path / "harmonization_log.xlsx"
    write_harmonization_log_workbook(log_file, pathway_outputs["harmonization_log_all"])
    assert log_file.exists()

    dropped_rows_df = build_dropped_rows_df(pathway_outputs["drop_logs"])
    assert list(dropped_rows_df.columns) == [
        "model",
        "scenario",
        "variable",
        "retained_variable",
        "ssp_family",
        "category",
        "drop_stage",
        "drop_reason",
    ]
    assert build_dropped_rows_df([]).empty

    summaries = summarize_variable_model_scenario_pairs(
        pathway_outputs["variable_coverage_summary_counts"],
    )
    assert summaries

    metadata_payload = build_process_metadata_payload(
        signature={"demo": True},
        categories=["C1", "C2", "C3", "C4"],
        ssps=[1, 2, 3, 4, 5],
        harmonization=True,
        harmonization_method="reduced_offset",
        latest_historical_year=2023,
        requested_harmonization_year=2019,
        harmonization_year=2019,
        harmonization_message="message",
        processed_dir=tmp_path / "processed",
        logs_dir=tmp_path / "logs",
        figures_dir=tmp_path / "figures",
        output_file=output_file,
        log_file=log_file,
        dropped_rows_csv_file=tmp_path / "dropped.csv",
        variable_coverage_summary_counts=pathway_outputs["variable_coverage_summary_counts"],
    )
    assert metadata_payload["function"] == "process_ar6"
    assert metadata_payload["arguments"] == {"demo": True}
    assert metadata_payload["provenance"]["harmonization_method"] == "reduced_offset"
    assert metadata_payload["provenance"]["harmonization_year_message"] == "message"
    report = build_process_report(
        study_period=[2019, 2060],
        categories=["C1", "C2", "C3", "C4"],
        ssps=[1, 2, 3, 4, 5],
        harmonization=True,
        harmonization_method="offset",
        processed_dir=tmp_path / "processed",
        logs_dir=tmp_path / "logs",
        figures_dir=tmp_path / "figures",
        output_file=output_file,
        log_file=log_file,
        dropped_rows_csv_file=tmp_path / "dropped.csv",
        process_meta_file=tmp_path / "scope_manifest.json",
        figures_meta_file=tmp_path / "figure.json",
        variable_coverage_summary_counts=pathway_outputs["variable_coverage_summary_counts"],
        latest_historical_year=2023,
        harmonization_year_requested=2025,
        harmonization_year=2023,
        harmonization_year_message="Warning: requested study period starts in 2025.",
        figure_files=["one.png"],
        figure_guide_file=tmp_path / "figures_explanation.txt",
    )
    assert isinstance(report, ProcessReportAR6)
    assert "Warning: requested study period starts in 2025." in str(report)
    report_with_template = build_process_report(
        study_period=[2019, 2060],
        categories=["C1"],
        ssps=[1],
        harmonization=True,
        harmonization_method="offset",
        processed_dir=tmp_path / "processed",
        logs_dir=tmp_path / "logs",
        figures_dir=tmp_path / "figures",
        output_file=output_file,
        log_file=log_file,
        dropped_rows_csv_file=tmp_path / "dropped.csv",
        process_meta_file=tmp_path / "process_meta.json",
        figures_meta_file=tmp_path / "figures_meta.json",
        template_csv_path=tmp_path / "processed" / "model_scenario_subset__template.csv",
    )
    assert str(report_with_template)
    empty_report = ProcessReportAR6(
        study_period=[2019, 2060],
        categories=["C1"],
        ssps=[1],
        harmonization=False,
        harmonization_method=None,
        processed_dir=tmp_path / "processed",
        logs_dir=tmp_path / "logs",
    )
    assert str(empty_report)


def test_process_ar6_entrypoint_without_figures(ar6_dummy_repo: AR6DummyRepo) -> None:
    report = process_ar6(years=list(range(2019, 2061)), figures=False, refresh=True)
    assert report is not None
    assert report.output_file is not None
    assert report.output_file.exists()
    assert report.harmonization_log_file is not None
    assert report.harmonization_log_file.exists()
    assert report.variable_coverage_summaries
    assert report.figure_files == []
    summary_log = (
        get_logs_dir(
            [2019, 2060],
            harmonization=True,
            harmonization_method="offset",
        )
        / "summary.log"
    )
    assert summary_log.read_text(encoding="utf-8").strip()

    reused_report = process_ar6(years=list(range(2019, 2061)), figures=False, refresh=False)
    assert reused_report.output_file == report.output_file
    assert reused_report.figure_files == []
    assert reused_report.variable_coverage_summaries
    assert summary_log.read_text(encoding="utf-8").strip()
    assert report.harmonization_year_message is None

    fallback_report = process_ar6(years=list(range(2025, 2061)), figures=False, refresh=True)
    assert fallback_report.latest_historical_year == 2023
    assert fallback_report.harmonization_year_requested == 2025
    assert fallback_report.harmonization_year == 2023
    assert fallback_report.harmonization_year_message is not None
    assert str(fallback_report)
    fallback_meta = read_json(
        get_logs_dir([2025, 2060], harmonization=True, harmonization_method="offset")
        / "scope_manifest.json"
    )
    assert fallback_meta["provenance"]["harmonization_year_message"] == (
        fallback_report.harmonization_year_message
    )
    reused_fallback_report = process_ar6(
        years=list(range(2025, 2061)),
        figures=False,
        refresh=False,
    )
    assert reused_fallback_report.harmonization_year_message == (
        fallback_report.harmonization_year_message
    )
    assert str(reused_fallback_report)

    non_harmo_report = process_ar6(
        years=range(2019, 2061),
        figures=False,
        harmonization=False,
        refresh=True,
    )
    assert non_harmo_report is not None
    assert non_harmo_report.harmonization is False
    assert non_harmo_report.harmonization_log_file is None

    with pytest.raises(ValueError):
        process_ar6(
            years=range(2019, 2061),
            figures=True,
            harmonization=False,
            figure_format={"format": "png", "dpi": 10},
        )

    with pytest.raises(ValueError):
        process_ar6(
            years=range(2019, 2061),
            figures=True,
            figure_format={"format": "bad", "dpi": 10},
        )

    missing_meta_path = (
        get_logs_dir([2019, 2060], harmonization=True, harmonization_method="offset")
        / "scope_manifest.json"
    )
    output_path = (
        get_processed_dir(
            [2019, 2060],
            harmonization=True,
            harmonization_method="offset",
        )
        / "harmonized_ar6_public.xlsx"
    )
    missing_meta_path.unlink(missing_ok=True)
    output_path.write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError):
        process_ar6(years=range(2019, 2061), figures=False, refresh=False)


def test_process_ar6_variable_coverage_serialization_and_validation() -> None:
    original = {
        NET_CO2_WO_AFOLU: {
            "available_model_scenario_pairs": 10,
            "retained_model_scenario_pairs": 7,
            "missing_reason_counts": {
                CO2_WO_AFOLU_NOT_PRODUCED_REASON: 3,
            },
        },
        NET_KYOTO_WO_AFOLU: {
            "available_model_scenario_pairs": 10,
            "retained_model_scenario_pairs": 8,
            "missing_reason_counts": {
                KYOTO_WO_AFOLU_NOT_PRODUCED_REASON: 2,
            },
        },
    }
    payload = serialize_variable_coverage_summary_counts(original)
    assert isinstance(payload, list) and len(payload) == 2
    recovered = deserialize_variable_coverage_summary_counts(payload)
    assert recovered == original


def test_build_pathway_outputs_summarizes_selected_processed_variables(
    ar6_dummy_repo: AR6DummyRepo,
) -> None:
    explorer = read_explorer_csv(ar6_dummy_repo.explorer_csv_path)
    pathway_outputs = build_pathway_outputs(
        explorer=explorer,
        categories=["C1", "C2", "C3", "C4"],
        ssps=[1, 2, 3, 4, 5],
        variables_output=[
            NET_CO2_WO_AFOLU,
            NET_CO2_WITH_AFOLU,
        ],
        study_period=[2019, 2060],
        database_raw_dir=ar6_dummy_repo.raw_dir,
        models_relevant_all=sorted(set(explorer.data["model"])),
        harmonization=True,
        harmonization_method="reduced_offset",
    )

    summary_counts = pathway_outputs["variable_coverage_summary_counts"]
    assert NET_CO2_WO_AFOLU in summary_counts
    assert NET_KYOTO_WO_AFOLU not in summary_counts
    assert summary_counts[NET_CO2_WO_AFOLU]["retained_model_scenario_pairs"] > 0


def test_process_ar6_variable_coverage_contract_edge_branches() -> None:
    serialized = serialize_variable_coverage_summary_counts(
        {
            NET_CO2_WITH_AFOLU: {
                "available_model_scenario_pairs": 3,
                "retained_model_scenario_pairs": 2,
                "missing_reason_counts": {"zero_reason": 0, "keep_reason": 1},
            },
            "Custom Variable": {
                "available_model_scenario_pairs": 2,
                "retained_model_scenario_pairs": 1,
                "missing_reason_counts": {},
            },
        }
    )
    assert [entry["variable"] for entry in serialized] == [
        NET_CO2_WITH_AFOLU,
        "Custom Variable",
    ]
    assert serialized[0]["missing_reason_counts"] == [{"reason_code": "keep_reason", "count": 1}]

    recovered = deserialize_variable_coverage_summary_counts(
        [
            {
                "variable": "v",
                "available_model_scenario_pairs": 1,
                "retained_model_scenario_pairs": 1,
                "missing_reason_counts": [
                    {"reason_code": "ignored", "count": 0},
                    {"reason_code": "kept", "count": 1},
                ],
            }
        ]
    )
    assert recovered["v"]["missing_reason_counts"] == {"kept": 1}

    assert summarize_variable_model_scenario_pairs(None) == []
    edge_summaries = summarize_variable_model_scenario_pairs(
        {
            "Skipped Variable": {"retained_model_scenario_pairs": -1},
            "Custom Variable": {
                "retained_model_scenario_pairs": 1,
                "missing_reason_counts": {},
            },
        }
    )
    assert [summary.variable for summary in edge_summaries] == ["Custom Variable"]

    report_with_gap_reason = ProcessReportAR6(
        study_period=[2019, 2060],
        categories=["C1"],
        ssps=[1],
        harmonization=True,
        harmonization_method="offset",
        processed_dir=Path("processed"),
        logs_dir=Path("logs"),
        variable_coverage_summaries=[
            processing_report_summaries.VariableCoverageSummaryAR6(
                variable="v",
                retained_model_scenario_pairs=1,
                missing_reason_counts={"only_reason": 1},
            )
        ],
    )
    assert str(report_with_gap_reason)
    report_without_gap_reason = ProcessReportAR6(
        study_period=[2019, 2060],
        categories=["C1"],
        ssps=[1],
        harmonization=True,
        harmonization_method="offset",
        processed_dir=Path("processed"),
        logs_dir=Path("logs"),
        variable_coverage_summaries=[
            processing_report_summaries.VariableCoverageSummaryAR6(
                variable="v",
                retained_model_scenario_pairs=2,
                missing_reason_counts={},
            )
        ],
    )
    assert str(report_without_gap_reason)
