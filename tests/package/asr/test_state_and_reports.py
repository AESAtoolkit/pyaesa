from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pyaesa.asr.deterministic.figures import state as asr_figure_state_mod
from pyaesa.asr.deterministic.runtime.prerequisites import _io_lca_phase_summary_lines
from pyaesa.asr.deterministic.state.branch_state import cached_branch_state
from pyaesa.asr.deterministic.state.branch_state import written_branch_state
from pyaesa.asr.deterministic.state.metadata import identity_matches
from pyaesa.shared.runtime.manifest_contract import manifest_digest
from pyaesa.asr.deterministic.state import reports as report_mod
from pyaesa.io_lca.contracts.runtime_types import IOLCAReport
from pyaesa.shared.acc_asr_common.reporting import DynamicAR6PathwayCount, DynamicAR6Summary


def test_asr_state_covers_method_loading(tmp_path: Path) -> None:
    csv_path = tmp_path / "asr.csv"
    parquet_path = tmp_path / "asr.parquet"
    pickle_path = tmp_path / "asr.pickle"
    one_row = pd.DataFrame({"l1_l2_method": ["UT(FD)"], "value": [1.0]})
    one_row.to_csv(csv_path, index=False)
    one_row.to_parquet(parquet_path, index=False)
    one_row.to_pickle(pickle_path)

    by_path = asr_figure_state_mod.l1_l2_methods_by_path(
        [csv_path, parquet_path, pickle_path],
        family_label="ASR",
    )
    assert by_path == {
        csv_path: "UT(FD)",
        parquet_path: "UT(FD)",
        pickle_path: "UT(FD)",
    }

    missing_column_path = tmp_path / "missing_column.csv"
    pd.DataFrame({"value": [1.0]}).to_csv(missing_column_path, index=False)
    with pytest.raises(ValueError):
        asr_figure_state_mod.l1_l2_methods_by_path(
            [missing_column_path],
            family_label="ASR",
        )

    empty_values_path = tmp_path / "empty_values.csv"
    pd.DataFrame({"l1_l2_method": [pd.NA]}).to_csv(empty_values_path, index=False)
    with pytest.raises(ValueError):
        asr_figure_state_mod.l1_l2_methods_by_path([empty_values_path], family_label="ASR")


def test_asr_report_covers_empty_single_and_multi_branch_strings(tmp_path: Path) -> None:
    branch_state = written_branch_state(
        cc_source="gwp100_lcia",
        cc_type="dynamic_ar6",
        cc_bounds=["dynamic"],
        impacts_used=["GWP_100"],
        figure_paths=[tmp_path / "figures" / "plot.png"],
        output_dirs=[tmp_path / "outputs"],
        meta_path=tmp_path / "meta.json",
    )

    branch_report = report_mod.build_asr_branch_report(
        state=branch_state,
        lca_type="io_lca",
        n_acc_files_matched=2,
        n_asr_files_written=3,
    )
    branch_report = replace(
        branch_report,
        dynamic_ar6_summary=DynamicAR6Summary(
            categories=["C1"],
            ssp_scenarios=["SSP2"],
            subset_version=None,
            pathway_counts=[
                DynamicAR6PathwayCount(
                    category="C1",
                    ssp_scenario="SSP2",
                    model_scenario_pairs=4,
                ),
            ],
            missing_pathway_combinations=[],
        ),
    )
    assert branch_report.cc_source == "gwp100_lcia"
    assert branch_report.cc_type == "dynamic_ar6"
    assert branch_report.cc_bounds == ["dynamic"]
    assert branch_report.lca_type == "io_lca"
    assert branch_report.n_acc_files_matched == 2
    assert branch_report.n_asr_files_written == 3
    assert branch_report.impacts_used == ["GWP_100"]

    common_lines = ["Project: asr_report_test"]
    single_branch_report = report_mod.ComputeASRReport(
        branches=[branch_report],
        output_root=tmp_path,
        common_lines=common_lines,
    )
    single_text = str(single_branch_report)
    assert single_text
    assert repr(single_branch_report) == single_text

    second_branch = report_mod.ASRBranchReport(
        cc_source="pb_lcia",
        cc_type="static",
        cc_bounds=["min_cc"],
        lca_type="external",
        n_acc_files_matched=1,
        n_asr_files_written=1,
        impacts_used=["AAL"],
        phase_index_path=tmp_path / "composite_phase_index.json",
    )
    multi_branch_text = str(
        report_mod.ComputeASRReport(
            branches=[branch_report, second_branch],
            output_root=tmp_path,
            common_lines=common_lines,
        )
    )
    assert multi_branch_text
    long_bounds_branch = report_mod.ASRBranchReport(
        cc_source="dynamic_ar6_with_a_long_source_label",
        cc_type="dynamic_ar6",
        cc_bounds=["category_C1", "category_C2", "category_C3", "category_C4", "category_C5"],
        lca_type="io_lca",
        n_acc_files_matched=1,
        n_asr_files_written=1,
        impacts_used=["GWP_100"],
    )
    long_text = str(
        report_mod.ComputeASRReport(
            branches=[long_bounds_branch],
            output_root=tmp_path,
            common_lines=common_lines,
        )
    )
    assert long_text


def test_asr_io_lca_prerequisite_summary_reports_upstream_figures(tmp_path: Path) -> None:
    io_report = IOLCAReport(
        source="exiobase_396_ixi",
        fu_code="L2.a.a",
        years=[2005],
        lcia_methods=["gwp100_lcia"],
        main_result_paths=[tmp_path / "results" / "main.csv"],
        origin_paths=[],
        stage_paths=[],
        skipped_method_years={},
        metadata_path=tmp_path / "logs" / "metadata.json",
        figure_paths=[tmp_path / "figures" / "plot.svg"],
    )

    lines = _io_lca_phase_summary_lines(
        io_report=io_report,
        base_allocate_args={
            "group_reg": False,
            "group_sec": True,
            "group_version": "oecd",
            "aggreg_indices": False,
        },
    )

    assert any("figure" in line.lower() for line in lines)


def test_cached_branch_state_reads_package_written_manifest(tmp_path: Path) -> None:
    assert identity_matches(existing_metadata=None, identity_payload={}) is False
    assert identity_matches(
        existing_metadata={"reuse": {"identity_key": manifest_digest({"scope": "demo"})}},
        identity_payload={"scope": "demo"},
    )
    state = cached_branch_state(
        existing_metadata={
            "execution": {"status": "complete"},
            "artifacts": {"output_dirs": [str(tmp_path / "outputs")]},
            "provenance": {
                "cc_source": "gwp100_lcia",
                "cc_type": "static",
                "cc_bounds": ["min_cc"],
                "impacts": ["GWP_100"],
            },
        },
        figure_paths=[],
        meta_path=tmp_path / "scope_manifest.json",
    )
    assert state.cc_source == "gwp100_lcia"
    assert state.output_dirs == [tmp_path / "outputs"]
