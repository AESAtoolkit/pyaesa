import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from pyaesa import deterministic_acc
from pyaesa.download.ar6.utils.config import GROSS_ALT_KYOTO_WO_AFOLU
from pyaesa.ar6_cc.deterministic.request.contracts import CC_FLOW_POSITIVE
from pyaesa.acc.deterministic.runtime.paths import (
    acc_output_relative_dir,
    build_acc_path_context,
    build_acc_scope_label,
    get_acc_figure_metadata_path,
    get_acc_meta_path,
    get_acc_output_dir,
)
from pyaesa.acc.deterministic.runtime.dynamic import (
    _emit_dynamic_status,
    _should_emit_dynamic_status,
    _filter_dynamic_share_ssp_scope,
    _filter_dynamic_cc_subset,
    _asocc_share_ssp_start_year,
    _resolved_dynamic_cc_ssp_tokens,
    _validate_dynamic_share_ssp_alignment,
    process_dynamic_acc,
)
from pyaesa.ar6_cc.deterministic.io.tables import read_cc_output, write_cc_output
from pyaesa.acc.deterministic.runtime.static import (
    _impact_row_positions,
    process_static_acc,
)
from pyaesa.asocc.runtime.reporting.deterministic_summary import (
    deterministic_asocc_phase_inventory_lines,
    deterministic_asocc_phase_summary_lines,
    deterministic_asocc_summary_record_messages,
)
from pyaesa.shared.acc_asr_common.deterministic.downstream.selection import (
    dynamic_compatible_share_frame,
    static_compatible_share_frame,
)
from pyaesa.acc.deterministic.runtime.static_cc import _optional_float, match_cc_for_share
from pyaesa.acc.deterministic.runtime.tables import (
    materialize_acc_scope,
    ordered_acc_output_columns,
    resolve_acc_l1_l2_method,
)
from pyaesa.acc.deterministic.figures.product_renderers import (
    _plot_dynamic_scope,
    _render_dynamic_budget_axis as _render_deterministic_dynamic_budget_axis,
    _static_min_max_geometry,
    prepare_plot_rows,
)
from pyaesa.acc.deterministic.figures.render import render_acc_deterministic_figures
from pyaesa.acc.deterministic.state.reports import (
    ACCBranchReport,
    ComputeACCReport,
)
from pyaesa.acc.deterministic.state.metadata import (
    load_recorded_output_files,
    save_run_metadata,
)
from pyaesa.asocc.runtime.scope.branch_resolution import outputs_project_root
from pyaesa.shared.acc_asr_common.scope.composite import (
    build_composite_base_allocate_args,
    normalize_base_asocc_args,
    normalize_shared_lcia_methods,
)
from pyaesa.shared.acc_asr_common.reporting import (
    DynamicAR6PathwayCount,
    DynamicAR6Summary,
    build_downstream_common_scope_lines,
    format_dynamic_ar6_summary_lines,
)
from pyaesa.shared.figures.deterministic_variant_compressor import (
    MAX_ROLE,
    MIN_ROLE,
    ROLE_COLUMN,
)
from pyaesa.shared.acc_asr_common.deterministic.downstream.inputs import (
    AsoccShare,
    LoadedAsoccShare,
    load_asocc_share,
)
from pyaesa.shared.runtime.metadata.contracts import SCOPE_MANIFEST_FILENAME
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    PHASE_B1_AR6_DYNAMIC_CC,
)
from pyaesa.shared.runtime.reporting.status import TransientStatusPrinter
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_HISTORICAL,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
    ASOCC_TIME_ROUTE_REGRESSION,
)
from pyaesa.shared.selectors.time_selectors import normalize_requested_years
from pyaesa.shared.lcia.paths import static_cc_csv_path
from pyaesa.shared.tabular.scalars import sanitize_token
from tests.package.helpers.acc_dummy_repo import (
    prepare_dynamic_acc_repo,
    prepare_dynamic_acc_repo_with_years,
    prepare_exiobase_repo_with_years,
)

_DYNAMIC_TEST_YEARS = range(2020, 2022)
_DYNAMIC_TEST_YEAR_COLUMNS = ("2020", "2021")


def _fast_figure_format() -> dict[str, Any]:
    return {"format": "png", "dpi": 10}


def _loaded_asocc_share(path: Path) -> LoadedAsoccShare:
    return load_asocc_share(
        AsoccShare(
            file_stem=path.stem,
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="native",
            path=path,
        )
    )


def _static_acc_kwargs(*, project_name: str) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "years": [2005],
        "lcia_method": "gwp100_lcia",
        "fu_code": "L2.a.a",
        "source": "oecd_v2025",
        "base_asocc_args": {
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "include_lcia_based_allocation_methods": False,
        },
        "base_cc_args": {"static": {"exclude_max_cc": False}},
        "figures": False,
        "subfigures": False,
    }


def _allocated_cc_root(
    repo_root: Path,
    *,
    project_name: str,
    public_result_root_name: str,
) -> Path:
    allocated_root = repo_root / f"{project_name}" / "B2_acc"
    return next(allocated_root.rglob(public_result_root_name))


def _acc_path_context(
    *,
    project_name: str,
    source_label: str,
    cc_source: str,
    cc_type: str,
    public_result_root_name: str | None = None,
):
    return build_acc_path_context(
        proj_base=outputs_project_root(project_name=project_name),
        source_label=source_label,
        group_version=None,
        cc_source=cc_source,
        cc_type=cc_type,
        public_result_root_name=public_result_root_name,
    )


def _asocc_root(repo_root: Path, *, project_name: str) -> Path:
    return repo_root / f"{project_name}" / "B1_asocc"


def _dynamic_asocc_share_path(repo_root: Path, *, project_name: str) -> Path:
    root = _asocc_root(repo_root, project_name=project_name)
    return next(root.rglob("UT(FD).csv"))


def _dynamic_acc_kwargs(*, project_name: str) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "years": _DYNAMIC_TEST_YEARS,
        "lcia_method": "gwp100_lcia",
        "fu_code": "L2.a.a",
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "one_step",
            "one_step_methods": ["AR(E^{CBA_FD})"],
        },
        "base_cc_args": {
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C1"], "ssp_scenario": ["SSP1"]},
        },
        "figures": False,
        "subfigures": False,
    }


def _dynamic_acc_ut_kwargs(*, project_name: str) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "years": _DYNAMIC_TEST_YEARS,
        "lcia_method": "gwp100_lcia",
        "fu_code": "L2.a.a",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "include_lcia_based_allocation_methods": False,
        },
        "base_cc_args": {
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C1"], "ssp_scenario": ["SSP1"]},
        },
        "figures": False,
        "subfigures": False,
    }


def _prepared_asocc_share(
    *,
    file_stem: str,
    frame: pd.DataFrame,
    impacts: tuple[str, ...] = tuple(),
    source_label: str = "external",
) -> tuple[LoadedAsoccShare, pd.DataFrame]:
    asocc_share = AsoccShare(
        file_stem=file_stem,
        relative_dir=Path("."),
        impacts=impacts,
        source_label=source_label,
        frame_wide=frame,
    )
    loaded = load_asocc_share(asocc_share)
    return loaded, loaded.frame_wide.copy()


def test_deterministic_acc_static_end_to_end_reuse_and_refresh(
    allocation_dummy_repo: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    kwargs = _static_acc_kwargs(project_name="acc_static_reuse")
    kwargs["figures"] = True
    kwargs["figure_format"] = _fast_figure_format()

    first_report = deterministic_acc(refresh=True, **kwargs)
    first_runtime_output = capsys.readouterr().out

    assert first_report is not None
    assert len(first_report.branches) == 1
    branch = first_report.branches[0]
    assert branch.cc_source == "gwp100_lcia"
    assert branch.cc_type == "static"
    assert branch.cc_bounds == ["min_cc", "max_cc"]
    assert branch.n_share_files_processed == 1
    assert branch.n_acc_files_written == 1
    assert branch.meta_file is not None and branch.meta_file.exists()
    assert branch.phase_index_path is not None and branch.phase_index_path.exists()
    assert first_runtime_output.strip()
    phase_index_payload = json.loads(branch.phase_index_path.read_text(encoding="utf-8"))
    assert [phase["phase"] for phase in phase_index_payload] == [
        "Phase B.1: aSoCC",
        "Phase B.2: aCC",
    ]
    assert [phase["function"] for phase in phase_index_payload] == [
        "deterministic_asocc",
        "deterministic_acc",
    ]
    assert str(first_report)
    metadata_payload = json.loads(branch.meta_file.read_text(encoding="utf-8"))
    first_figure_paths = [Path(path) for path in metadata_payload["artifacts"]["figure_paths"]]
    assert first_figure_paths
    figure_metadata = json.loads(
        get_acc_figure_metadata_path(
            context=_acc_path_context(
                project_name="acc_static_reuse",
                source_label="oecd_v2025",
                cc_source="gwp100_lcia",
                cc_type="static",
                public_result_root_name="results_l2_vs_global",
            )
        ).read_text(encoding="utf-8")
    )
    assert figure_metadata["figure_state"]["paths"] == [str(path) for path in first_figure_paths]
    assert (
        render_acc_deterministic_figures(
            metadata_path=branch.meta_file,
            dpi=10,
            output_format="png",
            figure_options={"per_method": True, "multi_method": True},
        )[0]
        == first_figure_paths
    )
    capsys.readouterr()

    output_paths = sorted(
        _allocated_cc_root(
            allocation_dummy_repo.repo_root,
            project_name="acc_static_reuse",
            public_result_root_name="results_l2_vs_global",
        ).rglob("*.csv")
    )
    assert len(output_paths) == 1
    assert output_paths[0].name == "UT(FD)__gwp100_lcia.csv"

    first_output = pd.read_csv(output_paths[0])
    assert {"impact", "impact_unit", "cc_bound", "2005"}.issubset(first_output.columns)
    assert "lcia_method" not in first_output.columns
    assert set(first_output["impact"]) == {"GWP_100"}
    assert set(first_output["impact_unit"]) == {"kg CO2-eq"}
    assert set(first_output["cc_bound"]) == {"min_cc", "max_cc"}
    assert "share_stem" not in first_output.columns
    assert bool(first_output["2005"].gt(0).all())

    reused_report = deterministic_acc(refresh=False, **kwargs)
    assert reused_report.branches[0].reuse_status == "reused_exact"
    assert reused_report.branches[0].figure_paths == first_figure_paths
    reuse_output = capsys.readouterr().out
    assert reuse_output == ""
    min_only_kwargs = {
        **kwargs,
        "base_cc_args": {"static": {"exclude_max_cc": True}},
    }
    min_only_report = deterministic_acc(refresh=False, **min_only_kwargs)
    assert min_only_report.branches[0].reuse_status == "reused_exact"

    restyled_kwargs = {**kwargs, "figure_format": {"format": "svg", "dpi": 2}}
    restyled_report = deterministic_acc(refresh=False, **restyled_kwargs)
    assert restyled_report.branches[0].reuse_status == "partially_reused"
    assert all(path.suffix == ".svg" for path in restyled_report.branches[0].figure_paths)
    capsys.readouterr()
    per_method_only, _ = render_acc_deterministic_figures(
        metadata_path=branch.meta_file,
        dpi=1,
        output_format="svg",
        figure_options={"per_method": True, "multi_method": False},
    )
    multi_method_only, _ = render_acc_deterministic_figures(
        metadata_path=branch.meta_file,
        dpi=1,
        output_format="svg",
        figure_options={"per_method": False, "multi_method": True},
    )
    assert per_method_only
    assert multi_method_only == []
    assert (
        render_acc_deterministic_figures(
            metadata_path=branch.meta_file,
            dpi=1,
            output_format="svg",
            figure_options={"per_method": False, "multi_method": False},
        )[0]
        == []
    )

    output_paths[0].unlink()
    with pytest.raises(ValueError, match="output files are missing"):
        deterministic_acc(refresh=False, **kwargs)

    asocc_sentinel = (
        _asocc_root(allocation_dummy_repo.repo_root, project_name="acc_static_reuse")
        / "oecd_v2025"
        / "deterministic"
        / "refresh_sentinel.txt"
    )
    asocc_sentinel.write_text("upstream", encoding="utf-8")
    refreshed_report = deterministic_acc(refresh=True, **kwargs)

    assert refreshed_report is not None
    assert refreshed_report.branches[0].n_acc_files_written == 1
    assert not asocc_sentinel.exists()


def test_deterministic_acc_table_and_metadata_cover_remaining_contracts() -> None:
    materialized = materialize_acc_scope(
        pd.DataFrame(
            [
                {
                    "l1_method": "CO(S)",
                    "l2_method": "UT(FD)",
                    "impact": "GWP_100",
                    "impact_unit": "kg CO2-eq",
                    "2005": 1.0,
                }
            ]
        ),
        l1_l2_method="CO(S)_UT(FD)",
        impact="GWP_100",
        impact_unit="kg CO2-eq",
    )
    assert materialized["l1_l2_method"].tolist() == ["CO(S)_UT(FD)"]
    assert materialized["l1_method"].tolist() == ["CO(S)"]
    assert materialized["l2_method"].tolist() == ["UT(FD)"]
    assert "asocc_ssp_start_year" not in materialized.columns

    dynamic_materialized = materialize_acc_scope(
        pd.DataFrame(
            [
                {
                    "l1_l2_method": "UT(FD)",
                    "impact": "GWP_100",
                    "impact_unit": "kg CO2-eq",
                    ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ASOCC_TIME_ROUTE_REGRESSION,
                    "2005": 1.0,
                }
            ]
        ),
        l1_l2_method="UT(FD)",
        impact="GWP_100",
        impact_unit="kg CO2-eq",
        asocc_ssp_start_year=2030,
    )
    assert dynamic_materialized[ASOCC_TIME_ROUTE_PUBLIC_COLUMN].tolist() == [
        ASOCC_TIME_ROUTE_REGRESSION
    ]
    assert dynamic_materialized["asocc_ssp_start_year"].tolist() == [2030]
    assert "asocc_ssp_start_year" in ordered_acc_output_columns(dynamic_materialized)
    assert ASOCC_TIME_ROUTE_PUBLIC_COLUMN in ordered_acc_output_columns(dynamic_materialized)


def test_deterministic_acc_recorded_output_files_contract(tmp_path: Path) -> None:
    output_file = tmp_path / "acc.csv"
    output_file.write_text("impact,2005\nGWP_100,1.0\n", encoding="utf-8")
    metadata_path = tmp_path / "scope_manifest.json"

    save_run_metadata(metadata_path, {"artifacts": {"output_files": [str(output_file)]}})
    assert load_recorded_output_files(metadata_path=metadata_path) == [output_file]


def test_deterministic_acc_asocc_prerequisite_summary_lines(tmp_path: Path) -> None:
    output_root = tmp_path / "demo" / "B1_asocc" / "deterministic"
    metadata_path = output_root / "logs" / "scope_manifest.json"
    metadata_path.parent.mkdir(parents=True)
    (output_root / "logs" / "summary.log").write_text("summary\n", encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "arguments": {
                    "source": "oecd_v2025",
                    "lcia_method": ["gwp100_lcia"],
                    "intermediate_outputs": True,
                },
                "provenance": {
                    "selected_methods": {
                        "l1": ["CO(S)"],
                        "l2_vs_global": "UT(FDa)",
                    }
                },
                "artifacts": {
                    "outputs": "results/output.csv",
                    "figure_paths": ["figure.png", " "],
                    "regression_stats_paths": ["regression.csv", " "],
                },
                "summary_records": [
                    "ignored",
                    {"severity": "INFO", "message": "information"},
                    {"severity": "WARNING", "message": "warning"},
                    {"severity": "WARNING", "message": " "},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary_lines = deterministic_asocc_phase_summary_lines(
        metadata_path=metadata_path,
        output_root=output_root,
        source="fallback_source",
    )
    inventory = deterministic_asocc_phase_inventory_lines(
        metadata_path=metadata_path,
        output_root=output_root,
    )

    assert summary_lines
    assert any("gwp100_lcia" in line for line in summary_lines)
    assert deterministic_asocc_summary_record_messages(
        metadata_path=metadata_path,
        severity="INFO",
    ) == ("information",)
    assert deterministic_asocc_summary_record_messages(
        metadata_path=metadata_path,
        severity="WARNING",
    ) == ("warning",)
    assert inventory
    no_log_root = tmp_path / "demo" / "B1_asocc" / "no_log"
    no_log_metadata_path = no_log_root / "logs" / "scope_manifest.json"
    no_log_metadata_path.parent.mkdir(parents=True)
    no_log_metadata_path.write_text(
        json.dumps(
            {
                "arguments": {"intermediate_outputs": True},
                "provenance": {"selected_methods": {"l2_in_l1": ["CO(S)"]}},
            }
        ),
        encoding="utf-8",
    )
    no_log_inventory = deterministic_asocc_phase_inventory_lines(
        metadata_path=no_log_metadata_path,
        output_root=no_log_root,
    )
    assert no_log_inventory


def test_deterministic_acc_report_renders_dynamic_ar6_phase_once() -> None:
    summary = DynamicAR6Summary(
        categories=["C1"],
        ssp_scenarios=["SSP2"],
        subset_version=None,
        pathway_counts=[
            DynamicAR6PathwayCount(
                category="C1",
                ssp_scenario="SSP2",
                model_scenario_pairs=2,
            )
        ],
        missing_pathway_combinations=[],
        process_ar6={
            "reuse_status": "computed",
            "study_period": [2020, 2021],
            "variable_coverage": [
                {
                    "variable": "Emissions|CO2",
                    "retained_model_scenario_pairs": 2,
                    "available_model_scenario_pairs": 3,
                }
            ],
            "figures_available": 0,
            "harmonization_year_message": "Warning: requested study period starts in 2025.",
        },
    )
    branch = ACCBranchReport(
        cc_source="ar6",
        cc_type="dynamic",
        cc_bounds=[],
        n_share_files_processed=1,
        n_acc_files_written=1,
        impacts_used=["GWP_100"],
        phase_entries=(
            CompositePhaseIndexEntry(
                phase=PHASE_B1_AR6_DYNAMIC_CC,
                function="deterministic_ar6_cc",
                status="complete",
                reuse_status="computed",
                output_root=None,
            ),
            CompositePhaseIndexEntry(
                phase=PHASE_B1_AR6_DYNAMIC_CC,
                function="deterministic_asocc",
                status="complete",
                reuse_status="computed",
                output_root=None,
            ),
        ),
        dynamic_ar6_summary=summary,
    )

    text = str(
        ComputeACCReport(
            branches=[branch],
            output_root=Path("demo"),
            common_lines=["Project: demo"],
        )
    )

    assert text.count("process_ar6") == 1
    assert "deterministic_asocc" in text
    empty_coverage_branch = ACCBranchReport(
        cc_source="ar6",
        cc_type="dynamic",
        cc_bounds=[],
        n_share_files_processed=1,
        n_acc_files_written=1,
        impacts_used=["GWP_100"],
        phase_entries=branch.phase_entries,
        dynamic_ar6_summary=DynamicAR6Summary(
            categories=["C1"],
            ssp_scenarios=["SSP2"],
            subset_version=None,
            pathway_counts=[],
            missing_pathway_combinations=[],
            process_ar6={"variable_coverage": [], "figures_available": 0},
        ),
    )
    empty_coverage_text = str(
        ComputeACCReport(
            branches=[empty_coverage_branch],
            output_root=Path("demo"),
            common_lines=["Project: demo"],
        )
    )
    assert empty_coverage_text


@pytest.mark.parametrize("refresh", [False, True])
def test_deterministic_acc_rejects_shared_scope_identity_drift(
    allocation_dummy_repo: Any,
    refresh: bool,
) -> None:
    del allocation_dummy_repo
    kwargs = _static_acc_kwargs(project_name="acc_identity_guard")
    first_report = deterministic_acc(refresh=True, **kwargs)

    assert first_report is not None
    meta_file = cast(Path, first_report.branches[0].meta_file)
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    written_outputs = [Path(path) for path in metadata["artifacts"]["output_files"]]
    assert written_outputs
    with pytest.raises(
        ValueError,
        match="project_name",
    ):
        deterministic_acc(
            refresh=refresh,
            **_static_acc_kwargs(project_name="acc_identity_guard"),
            r_p=["FR"],
        )
    assert all(path.exists() for path in written_outputs)


def test_deterministic_acc_static_min_only_end_to_end(allocation_dummy_repo: Any) -> None:
    kwargs = _static_acc_kwargs(project_name="acc_static_min_only")
    kwargs["base_cc_args"] = {"static": {"exclude_max_cc": True}}

    report = deterministic_acc(refresh=True, **kwargs)

    assert report is not None
    branch = report.branches[0]
    assert branch.cc_bounds == ["min_cc"]
    assert branch.n_share_files_processed == 1
    assert branch.n_acc_files_written == 1

    output_paths = sorted(
        _allocated_cc_root(
            allocation_dummy_repo.repo_root,
            project_name="acc_static_min_only",
            public_result_root_name="results_l2_vs_global",
        ).rglob("*.csv")
    )
    assert len(output_paths) == 1
    assert output_paths[0].name == "UT(FD)__gwp100_lcia.csv"


def test_deterministic_acc_static_figures_are_public_for_single_and_multi_year(
    allocation_dummy_repo_factory,
) -> None:
    allocation_dummy_repo_factory(name="acc_static_public_figures")
    base_kwargs = {
        "project_name": "acc_static_public_figures",
        "lcia_method": "pb_lcia",
        "fu_code": "L2.a.b",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "include_lcia_based_allocation_methods": False,
        },
        "base_cc_args": {"static": {"exclude_max_cc": False}},
        "figures": True,
        "subfigures": False,
        "figure_format": _fast_figure_format(),
    }
    single = deterministic_acc(years=[2005], refresh=True, **base_kwargs)
    single_paths = [path for branch in single.branches for path in branch.figure_paths]

    assert single_paths
    assert all(path.exists() for path in single_paths)
    assert any("multi_method" in path.parts for path in single_paths)
    assert any("per_method" in path.parts for path in single_paths)
    assert any(path.name.endswith("__2005.png") for path in single_paths)


def test_deterministic_acc_static_figures_cover_public_odd_impact_panels(
    allocation_dummy_repo_factory,
) -> None:
    allocation_dummy_repo_factory(name="acc_static_public_odd_impact_figures")
    cc_path = static_cc_csv_path(lcia_method="custom_lcia")
    cc_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "impact_full_name": ["Impact X", "Impact Y", "Impact Z"],
            "impact": ["X1", "Y2", "Z3"],
            "impact_unit": ["kg", "kg", "kg"],
            "min_cc": [100.0, 200.0, 300.0],
            "max_cc": [150.0, 250.0, 350.0],
        }
    ).to_csv(cc_path, index=False)

    report = deterministic_acc(
        project_name="acc_static_public_odd_impact_figures",
        years=[2005],
        lcia_method="custom_lcia",
        fu_code="L2.a.b",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "include_lcia_based_allocation_methods": False,
        },
        base_cc_args={"static": {"exclude_max_cc": False}},
        figures=True,
        subfigures=False,
        figure_format=_fast_figure_format(),
        refresh=True,
    )

    figure_paths = [path for branch in report.branches for path in branch.figure_paths]
    assert figure_paths
    assert all(path.exists() for path in figure_paths)
    assert any("multi_method" in path.parts for path in figure_paths)

    multi_report = deterministic_acc(
        project_name="acc_static_public_odd_impact_figures_multi",
        years=[2005, 2006],
        lcia_method="custom_lcia",
        fu_code="L2.a.b",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "include_lcia_based_allocation_methods": False,
        },
        base_cc_args={"static": {"exclude_max_cc": False}},
        figures=True,
        subfigures=False,
        figure_format=_fast_figure_format(),
        refresh=True,
    )

    multi_paths = [path for branch in multi_report.branches for path in branch.figure_paths]
    assert multi_paths
    assert all(path.exists() for path in multi_paths)


def test_deterministic_acc_static_figures_cover_public_variants_and_transitions(
    allocation_dummy_repo_factory,
) -> None:
    repo = allocation_dummy_repo_factory(name="acc_static_public_figure_variants")
    cc_path = static_cc_csv_path(lcia_method="custom_transition_lcia")
    cc_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "impact_full_name": ["Impact A", "Impact B"],
            "impact": ["IA", "IB"],
            "impact_unit": ["kg", "kg"],
            "min_cc": [100.0, 200.0],
            "max_cc": [150.0, 250.0],
        }
    ).to_csv(cc_path, index=False)
    prepare_exiobase_repo_with_years(
        repo,
        historical_years=[2018, 2019, 2020],
        scenario_years=[2020, 2030],
    )
    repo.set_processed_pop_gdp_years(
        historical_years=[2018, 2019],
        scenario_years=[2020, 2030],
    )
    common_kwargs: dict[str, Any] = {
        "lcia_method": "custom_transition_lcia",
        "fu_code": "L2.a.b",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "include_lcia_based_allocation_methods": False,
            "ssp_scenario": ["SSP2"],
        },
        "base_cc_args": {"static": {"exclude_max_cc": False}},
        "figures": True,
        "subfigures": False,
        "figure_format": _fast_figure_format(),
    }

    transition = deterministic_acc(
        project_name="acc_static_public_figure_transition",
        years=[2018, 2019, 2020],
        refresh=True,
        **common_kwargs,
    )
    variant_kwargs = dict(common_kwargs)
    variant_kwargs["base_asocc_args"] = {
        **common_kwargs["base_asocc_args"],
        "reference_years": [2018, 2019],
        "projection_mode": "historical_reuse",
        "reg_window": [2018, 2019],
        "l2_reuse_years": [2018, 2019],
    }
    variants = deterministic_acc(
        project_name="acc_static_public_figure_variants",
        years=[2030],
        refresh=True,
        **variant_kwargs,
    )

    transition_paths = [path for branch in transition.branches for path in branch.figure_paths]
    variant_paths = [path for branch in variants.branches for path in branch.figure_paths]

    assert any("SSP2" in path.stem for path in transition_paths)
    assert any(path.name.endswith("__2030.png") for path in variant_paths)
    assert all(path.exists() for path in [*transition_paths, *variant_paths])


def test_normalize_requested_years_accepts_scalar_year() -> None:
    assert normalize_requested_years(2019) == [2019]


@pytest.mark.parametrize(
    ("years", "lcia_method", "base_cc_args", "match"),
    [
        (
            [2030, 2032],
            "gwp100_lcia",
            {"static": {"active": False}, "dynamic_ar6": {}},
            "years must represent consecutive years with no gaps",
        ),
        (
            range(2030, 2032),
            "pb_lcia",
            {"static": {"active": False}, "dynamic_ar6": {}},
            "requires at least one LCIA method",
        ),
    ],
)
def test_deterministic_acc_rejects_invalid_dynamic_public_requests(
    allocation_dummy_repo: Any,
    years: list[int] | range,
    lcia_method: str,
    base_cc_args: dict[str, dict[str, object]],
    match: str,
) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError, match=match):
        deterministic_acc(
            project_name="acc_dynamic_validation",
            years=years,
            lcia_method=lcia_method,
            fu_code="L2.a.a",
            source="oecd_v2025",
            base_asocc_args={
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "include_lcia_based_allocation_methods": False,
            },
            base_cc_args=base_cc_args,
        )


def test_deterministic_acc_dynamic_end_to_end_and_reuse(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_acc_repo(allocation_dummy_repo)
    kwargs = _dynamic_acc_ut_kwargs(project_name="acc_dynamic_success")

    first_report = deterministic_acc(refresh=True, **kwargs)

    assert first_report is not None
    assert len(first_report.branches) == 1
    branch = first_report.branches[0]
    assert branch.cc_type == "dynamic_ar6"
    assert branch.cc_source == "gwp100_lcia"
    assert branch.cc_bounds == ["C1"]
    assert branch.n_share_files_processed == 1
    assert branch.n_acc_files_written == 1
    assert branch.impacts_used == [GROSS_ALT_KYOTO_WO_AFOLU]
    assert branch.meta_file is not None and branch.meta_file.exists()

    output_paths = sorted(
        _allocated_cc_root(
            allocation_dummy_repo.repo_root,
            project_name="acc_dynamic_success",
            public_result_root_name="results_l2_vs_global",
        ).rglob("*.csv")
    )
    assert len(output_paths) == 1
    assert all("__gwp100_lcia__dynamic_ar6.csv" in path.name for path in output_paths)

    first_output = pd.read_csv(output_paths[0])
    assert {
        "cc_model",
        "cc_scenario",
        "cc_category",
        "ar6_cc_ssp_scenario",
        "cc_flow",
        "cc_variable",
    }.issubset(first_output.columns)
    assert set(first_output["impact"]) == {"GWP_100"}
    assert set(first_output["impact_unit"]) == {"kg CO2-eq"}
    assert set(first_output["cc_category"]) == {"C1"}
    assert set(first_output["ar6_cc_ssp_scenario"]) == {"SSP1"}
    assert set(first_output["cc_flow"]) == {CC_FLOW_POSITIVE}
    assert set(first_output["cc_variable"]) == {GROSS_ALT_KYOTO_WO_AFOLU}
    assert "share_stem" not in first_output.columns
    assert set(first_output["l1_l2_method"]) == {"UT(FD)"}
    assert set(_DYNAMIC_TEST_YEAR_COLUMNS).issubset(first_output.columns)
    assert first_output.loc[:, _DYNAMIC_TEST_YEAR_COLUMNS].to_numpy(dtype=float).max() > 1e6
    reused_report = deterministic_acc(refresh=False, **kwargs)
    assert reused_report.branches[0].reuse_status == "reused_exact"

    dynamic_summary = branch.dynamic_ar6_summary
    assert dynamic_summary is not None
    assert dynamic_summary.process_ar6 is not None
    process_sentinel = (
        Path(str(dynamic_summary.process_ar6["output_root"])) / "refresh_sentinel.txt"
    )
    process_sentinel.write_text("processed", encoding="utf-8")
    refreshed_report = deterministic_acc(refresh=True, **kwargs)
    assert refreshed_report.branches[0].n_acc_files_written == 1
    assert not process_sentinel.exists()


def test_deterministic_acc_dynamic_generates_public_figures(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_acc_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2020, 2021],
        scenario_years=[2030],
    )

    kwargs = _dynamic_acc_ut_kwargs(project_name="acc_dynamic_public_figures")
    kwargs["base_asocc_args"]["one_step_methods"] = ["UT(FD)", "AR(E^{CBA_FD})"]
    kwargs["figures"] = True
    kwargs["figure_format"] = _fast_figure_format()
    report = deterministic_acc(refresh=True, **kwargs)

    figure_paths = [path for branch in report.branches for path in branch.figure_paths]
    assert figure_paths
    assert all(path.exists() for path in figure_paths)


def test_deterministic_acc_dynamic_figures_cover_public_asocc_transition(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_acc_repo_with_years(
        allocation_dummy_repo,
        historical_years=list(range(1995, 2025)),
        scenario_years=[2025, 2026],
    )

    report = deterministic_acc(
        project_name="acc_dynamic_public_transition_figures",
        years=range(2024, 2027),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["AR(E^{CBA_FD})"],
            "ssp_scenario": ["SSP2"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C1"], "ssp_scenario": ["SSP1"]},
        },
        figures=True,
        subfigures=False,
        figure_format=_fast_figure_format(),
        refresh=True,
    )

    figure_paths = [path for branch in report.branches for path in branch.figure_paths]
    assert figure_paths
    assert all(path.exists() for path in figure_paths)


def test_deterministic_acc_dynamic_rejects_missing_subset(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_acc_repo_with_years(
        allocation_dummy_repo,
        historical_years=[2020, 2021],
        scenario_years=[2030],
    )
    kwargs = _dynamic_acc_ut_kwargs(project_name="acc_dynamic_missing_subset")
    kwargs["base_cc_args"] = {
        "static": {"active": False},
        "dynamic_ar6": {"category": ["C9"], "ssp_scenario": ["SSP1"]},
    }

    with pytest.raises(ValueError, match="No AR6 pathways remain after filtering"):
        deterministic_acc(**kwargs)


def test_deterministic_acc_dynamic_rejects_unconvertible_cc_unit(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_acc_repo(allocation_dummy_repo)
    explorer_path = (
        allocation_dummy_repo.repo_root
        / "data_raw"
        / "carrying_capacities"
        / "dynamic_climate_change_ar6"
        / "ar6_public_explorer.csv"
    )
    explorer = pd.read_csv(explorer_path)
    kyoto_rows = explorer["variable"].astype(str).str.contains("Kyoto Gases", regex=False)
    explorer.loc[kyoto_rows, "unit"] = "unconvertible unit"
    explorer.to_csv(explorer_path, index=False)

    with pytest.raises(ValueError, match="Dynamic aCC cannot convert"):
        deterministic_acc(refresh=True, **_dynamic_acc_ut_kwargs(project_name="acc_bad_cc_unit"))


def test_static_cc_runtime_covers_optional_forced_and_stem_matching() -> None:
    cc_df = pd.DataFrame(
        {
            "impact": ["GWP_100", "AAL"],
            "impact_unit": ["kg CO2-eq", "planetary unit"],
            "min_cc": [10.0, 2.0],
            "max_cc": [20.0, np.nan],
        }
    )

    assert _optional_float(None) is None
    assert _optional_float(pd.NA) is None
    assert _optional_float(pd.NaT) is None
    assert _optional_float(np.nan) is None
    assert _optional_float("3.5") == pytest.approx(3.5)

    forced_matches = match_cc_for_share(
        Path("ignored.csv"),
        cc_df,
        forced_impacts=["AAL", "GWP_100"],
    )
    assert forced_matches == [
        ("AAL", 2.0, None, "planetary unit"),
        ("GWP_100", 10.0, 20.0, "kg CO2-eq"),
    ]

    stem_matches = match_cc_for_share(
        Path("demo_gwp_100.csv"),
        cc_df,
    )
    assert stem_matches == [("GWP_100", 10.0, 20.0, "kg CO2-eq")]

    fallback_matches = match_cc_for_share(
        Path("demo_other.csv"),
        cc_df,
    )
    assert fallback_matches == [
        ("GWP_100", 10.0, 20.0, "kg CO2-eq"),
        ("AAL", 2.0, None, "planetary unit"),
    ]


def test_acc_deterministic_path_owner_covers_validation_and_relative_contracts(
    tmp_path: Path,
) -> None:
    context = build_acc_path_context(
        proj_base=tmp_path,
        source_label="oecd_v2025",
        group_version=None,
        cc_source="gwp100_lcia",
        cc_type="static",
    )

    assert (
        build_acc_scope_label(
            source_label="oecd_v2025",
            group_version=None,
            cc_source="gwp100 lcia",
            cc_type="static",
        )
        == "oecd_v2025__static__gwp100_lcia"
    )
    assert (
        build_acc_scope_label(
            source_label="oecd_v2025",
            group_version=None,
            cc_source="!!!",
            cc_type="dynamic_ar6",
        )
        == "oecd_v2025__dynamic_ar6__item"
    )
    assert get_acc_output_dir(context=context).name == "static__gwp100_lcia"
    assert get_acc_meta_path(context=context).name == "scope_manifest.json"
    assert acc_output_relative_dir(upstream_relative_dir=Path()) == Path(".")
    assert acc_output_relative_dir(upstream_relative_dir=Path("level_2/results/demo")) == Path(
        "demo"
    )
    assert acc_output_relative_dir(
        upstream_relative_dir=Path("level_1/level_2/results/demo")
    ) == Path("demo")
    assert acc_output_relative_dir(upstream_relative_dir=Path("level_1/l2_vs_global")) == Path(".")
    assert acc_output_relative_dir(upstream_relative_dir=Path("results/nested")) == Path("nested")


def test_prepare_static_compatible_share_frame_skips_mismatched_lcia_inputs(
    tmp_path: Path,
) -> None:
    gwp_frame = pd.DataFrame({"lcia_method": ["gwp100_lcia"], "2005": [1.0]})
    pb_frame = pd.DataFrame({"lcia_method": ["pb_lcia"], "2005": [1.0]})
    path_only = tmp_path / "demo__gwp100_lcia.csv"
    path_only.write_text("demo", encoding="utf-8")

    prepared_gwp = static_compatible_share_frame(
        asocc_share=AsoccShare(
            file_stem="gwp__gwp100_lcia",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="external",
        ),
        share_frame=gwp_frame,
        cc_source="gwp100_lcia",
    )
    assert prepared_gwp is not None
    assert prepared_gwp.equals(gwp_frame)
    assert (
        static_compatible_share_frame(
            asocc_share=AsoccShare(
                file_stem="pb__pb_lcia",
                relative_dir=Path("."),
                impacts=tuple(),
                source_label="external",
            ),
            share_frame=pb_frame,
            cc_source="gwp100_lcia",
        )
        is None
    )
    prepared_path_only = static_compatible_share_frame(
        asocc_share=AsoccShare(
            file_stem="demo__gwp100_lcia",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="native",
            path=path_only,
        ),
        share_frame=pd.DataFrame({"2005": [1.0]}),
        cc_source="gwp100_lcia",
    )
    assert prepared_path_only is not None
    assert prepared_path_only.equals(pd.DataFrame({"2005": [1.0]}))


def test_impact_row_positions_filter_multi_impact_static_share() -> None:
    frame = pd.DataFrame(
        {
            "impact": ["SOD", "AAL"],
            "2005": [1.0, 2.0],
        }
    )
    prepared = _impact_row_positions(
        share_frame=frame,
        impact_code="SOD",
    )
    assert prepared is not None
    assert prepared.tolist() == [0]
    assert (
        _impact_row_positions(
            share_frame=frame,
            impact_code="GWP_100",
        )
        is None
    )
    empty_impact_frame = pd.DataFrame({"impact": ["", pd.NA], "2005": [1.0, 2.0]})
    empty_impact_prepared = _impact_row_positions(
        share_frame=empty_impact_frame,
        impact_code="GWP_100",
    )
    assert empty_impact_prepared is not None
    assert empty_impact_prepared.tolist() == [0, 1]


def test_acc_table_covers_dynamic_materialization_and_method_resolution() -> None:
    class _StatusCapture:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def show(self, message: str) -> None:
            self.messages.append(message)

    assert _should_emit_dynamic_status(step=1, total_steps=30) is True
    assert _should_emit_dynamic_status(step=25, total_steps=30) is True
    assert _should_emit_dynamic_status(step=30, total_steps=30) is True
    assert _should_emit_dynamic_status(step=2, total_steps=30) is False
    status_capture = _StatusCapture()
    _emit_dynamic_status(
        status=cast(TransientStatusPrinter, status_capture),
        step=2,
        total_steps=30,
        display_name="demo.csv",
        category="C1",
        ssp="SSP2",
        cc_model="Model",
        cc_label="GWP100_dynamic",
    )
    assert status_capture.messages == []
    _emit_dynamic_status(
        status=cast(TransientStatusPrinter, status_capture),
        step=25,
        total_steps=30,
        display_name="demo.csv",
        category="C1",
        ssp="SSP2",
        cc_model="Model",
        cc_label="GWP100_dynamic",
    )
    assert status_capture.messages == [
        "[deterministic_acc] GWP100_dynamic: 25/30 demo.csv C1/SSP2/Model"
    ]

    materialized = materialize_acc_scope(
        pd.DataFrame(
            {
                "share_stem": ["legacy"],
                "l1_l2_method": [""],
                "l2_method": ["UT(FD)"],
                "lcia_method": [pd.NA],
                "impact": [""],
                "impact_unit": [" "],
                "2019": [1.0],
            }
        ),
        l1_l2_method="UT(FD)",
        impact="GWP_100",
        impact_unit="kg CO2-eq",
    )
    assert "share_stem" not in materialized.columns
    assert materialized.loc[0, "l1_l2_method"] == "UT(FD)"
    assert materialized.loc[0, "l2_method"] == "UT(FD)"
    assert "l1_method" not in materialized.columns
    assert "lcia_method" not in materialized.columns
    assert materialized.loc[0, "impact"] == "GWP_100"
    assert materialized.loc[0, "impact_unit"] == "kg CO2-eq"

    inserted = materialize_acc_scope(
        pd.DataFrame({"l2_method": ["UT(FD)"], "2019": [1.0]}),
        l1_l2_method="UT(FD)",
        impact="GWP_100",
        impact_unit="kg CO2-eq",
    )
    assert {"l1_l2_method", "l2_method", "impact", "impact_unit"}.issubset(inserted.columns)
    assert "lcia_method" not in inserted.columns
    assert "l1_method" not in inserted.columns

    with pytest.raises(ValueError, match="aCC impact unit scope found rows that conflict"):
        materialize_acc_scope(
            pd.DataFrame(
                {
                    "l1_l2_method": ["UT(FD)"],
                    "lcia_method": ["gwp100_lcia"],
                    "impact": ["GWP_100"],
                    "impact_unit": ["other_unit"],
                    "2019": [1.0],
                }
            ),
            l1_l2_method="UT(FD)",
            impact="GWP_100",
            impact_unit="kg CO2-eq",
        )

    assert (
        resolve_acc_l1_l2_method(
            frame=pd.DataFrame({"l1_l2_method": ["UT(FD)", "UT(FD)"]}),
            source_label="demo",
        )
        == "UT(FD)"
    )
    assert (
        resolve_acc_l1_l2_method(
            frame=pd.DataFrame({"l1_method": ["CO(S)"], "l2_method": ["UT(FD)"]}),
            source_label="demo",
        )
        == "CO(S)_UT(FD)"
    )
    assert (
        resolve_acc_l1_l2_method(
            frame=pd.DataFrame({"l2_method": ["UT(FD)"]}),
            source_label="demo",
        )
        == "UT(FD)"
    )
    assert (
        resolve_acc_l1_l2_method(
            frame=pd.DataFrame({"l1_method": ["CO(S)"]}),
            source_label="demo",
        )
        == "CO(S)"
    )


def test_process_static_acc_covers_missing_asocc_shares_and_requested_year_skip(
    allocation_dummy_repo: Any,
) -> None:
    def _base_allocate_args(project_name: str) -> dict[str, Any]:
        return build_composite_base_allocate_args(
            project_name=project_name,
            years=[2005],
            lcia_method=normalize_shared_lcia_methods("gwp100_lcia"),
            fu_code="L2.a.a",
            r_p=None,
            s_p=None,
            r_c=None,
            r_f=None,
            source="oecd_v2025",
            group_reg=False,
            group_sec=False,
            group_version=None,
            aggreg_indices=False,
            base_asocc_args=normalize_base_asocc_args(
                {
                    "method_plan": "one_step",
                    "one_step_methods": ["UT(FD)"],
                    "include_lcia_based_allocation_methods": False,
                },
                fu_code="L2.a.a",
            ),
        )

    project_name = "acc_static_skipped_share_years"
    proj_base = outputs_project_root(project_name=project_name)
    share_path = allocation_dummy_repo.write_l2_table(
        proj_base=proj_base,
        source_label="oecd_v2025",
        bucket="l2_vs_global",
        method_name="UT(FD)",
        frame=pd.DataFrame({"2010": [0.5]}),
    )
    skip_status = TransientStatusPrinter("deterministic_acc")
    try:
        n_share, n_written, impacts, output_dirs, output_files, cc_csv_path = process_static_acc(
            path_context=_acc_path_context(
                project_name=project_name,
                source_label="oecd_v2025",
                cc_source="gwp100_lcia",
                cc_type="static",
            ),
            public_result_root_name="results_l2_vs_global",
            cc_source="gwp100_lcia",
            years=[2005],
            asocc_shares=[_loaded_asocc_share(share_path)],
            fmt="csv",
            static_cc_bounds=["min_cc"],
            status=skip_status,
        )
    finally:
        skip_status.finish()

    assert n_share == 0
    assert n_written == 0
    assert impacts == []
    assert len(output_dirs) == 1
    assert output_files == []
    assert cc_csv_path.exists()

    incompatible_project_name = "acc_static_incompatible_lcia_only"
    share_path = allocation_dummy_repo.write_l2_table(
        proj_base=outputs_project_root(project_name=incompatible_project_name),
        source_label="oecd_v2025",
        bucket="l2_vs_global",
        method_name="UT(FD)__pb_lcia",
        frame=pd.DataFrame({"2005": [0.5]}),
    )
    incompatible_status = TransientStatusPrinter("deterministic_acc")
    try:
        n_share, n_written, impacts, output_dirs, output_files, cc_csv_path = process_static_acc(
            path_context=_acc_path_context(
                project_name=incompatible_project_name,
                source_label="oecd_v2025",
                cc_source="gwp100_lcia",
                cc_type="static",
            ),
            public_result_root_name="results_l2_vs_global",
            cc_source="gwp100_lcia",
            years=[2005],
            asocc_shares=[_loaded_asocc_share(share_path)],
            fmt="csv",
            static_cc_bounds=["min_cc"],
            status=incompatible_status,
        )
    finally:
        incompatible_status.finish()

    assert n_share == 0
    assert n_written == 0
    assert impacts == []
    assert len(output_dirs) == 1
    assert output_files == []
    assert cc_csv_path.exists()

    impact_filtered_project = "acc_static_impact_filtered_share"
    share_path = allocation_dummy_repo.write_l2_table(
        proj_base=outputs_project_root(project_name=impact_filtered_project),
        source_label="oecd_v2025",
        bucket="l2_vs_global",
        method_name="UT(FD)",
        frame=pd.DataFrame({"l1_l2_method": ["UT(FD)"], "impact": ["AAL"], "2005": [0.5]}),
    )
    impact_filtered_status = TransientStatusPrinter("deterministic_acc")
    try:
        n_share, n_written, impacts, output_dirs, output_files, cc_csv_path = process_static_acc(
            path_context=_acc_path_context(
                project_name=impact_filtered_project,
                source_label="oecd_v2025",
                cc_source="gwp100_lcia",
                cc_type="static",
            ),
            public_result_root_name="results_l2_vs_global",
            cc_source="gwp100_lcia",
            years=[2005],
            asocc_shares=[_loaded_asocc_share(share_path)],
            fmt="csv",
            static_cc_bounds=["min_cc"],
            status=impact_filtered_status,
        )
    finally:
        impact_filtered_status.finish()

    assert n_share == 0
    assert n_written == 0
    assert impacts == []
    assert len(output_dirs) == 1
    assert output_files == []
    assert cc_csv_path.exists()


def test_dynamic_runtime_covers_table_loading_subset_and_compatibility(
    tmp_path: Path,
) -> None:
    table = pd.DataFrame(
        {
            "cc_model": ["M1", "M2", "M3"],
            "cc_scenario": ["S1", "S2", "S3"],
            "cc_category": ["C1", "C2", "C2"],
            "ssp_scenario": ["SSP1", "SSP1", "SSP2"],
            "cc_flow": [CC_FLOW_POSITIVE, CC_FLOW_POSITIVE, CC_FLOW_POSITIVE],
            "cc_variable": [GROSS_ALT_KYOTO_WO_AFOLU] * 3,
            "impact_unit": ["kg CO2-eq", "kg CO2-eq", "kg CO2-eq"],
            "2019": [1.0, 2.0, 3.0],
        }
    )
    csv_path = tmp_path / "ar6_cc.csv"
    parquet_path = tmp_path / "ar6_cc.parquet"
    pickle_path = tmp_path / "ar6_cc.pickle"
    for output_path, output_format in [
        (csv_path, "csv"),
        (parquet_path, "parquet"),
        (pickle_path, "pickle"),
    ]:
        write_cc_output(table, output_path, output_format)

    assert (
        read_cc_output(output_file=csv_path, output_format="csv")
        .loc[:, list(table.columns)]
        .equals(table)
    )
    assert (
        read_cc_output(output_file=parquet_path, output_format="parquet")
        .loc[:, list(table.columns)]
        .equals(table)
    )
    assert (
        read_cc_output(output_file=pickle_path, output_format="pickle")
        .loc[:, list(table.columns)]
        .equals(table)
    )
    assert _resolved_dynamic_cc_ssp_tokens(cc_table=table) == ["SSP1", "SSP2"]
    assert sanitize_token("C1__ssp1__IMAGE/MODEL__A:B?C") == "C1__ssp1__IMAGE_MODEL__A_B_C"

    unchanged = _filter_dynamic_cc_subset(
        cc_table=table,
        category=None,
        ssp_scenario=None,
    )
    assert unchanged.equals(table)

    filtered = _filter_dynamic_cc_subset(
        cc_table=table,
        category=["C2"],
        ssp_scenario=["SSP2"],
    )
    assert list(filtered["cc_category"]) == ["C2"]
    assert list(filtered["ssp_scenario"]) == ["SSP2"]

    asocc_share_no_scenario = _prepared_asocc_share(
        file_stem="native_share",
        frame=pd.DataFrame({"2019": [1.0]}),
        source_label="native",
    )
    _validate_dynamic_share_ssp_alignment(
        prepared_asocc_shares=[asocc_share_no_scenario],
        cc_table=table,
        require_selected_ssp_shares=False,
        cc_source="gwp100_lcia",
        resolved_cc_path=csv_path,
        requested_years=[2019],
    )

    asocc_share_mismatch = _prepared_asocc_share(
        file_stem="external_share",
        frame=pd.DataFrame({ASOCC_SSP_SCENARIO_COLUMN: ["SSP3"], "2019": [1.0]}),
    )
    with pytest.raises(
        ValueError,
        match="aSoCC SSPs=\\['SSP3'\\], dynamic AR6 SSPs=\\['SSP1', 'SSP2'\\]",
    ):
        _validate_dynamic_share_ssp_alignment(
            prepared_asocc_shares=[asocc_share_mismatch],
            cc_table=table,
            require_selected_ssp_shares=False,
            cc_source="gwp100_lcia",
            resolved_cc_path=csv_path,
            requested_years=[2019],
        )
    asocc_share_matching = _prepared_asocc_share(
        file_stem="external_share_ssp1",
        frame=pd.DataFrame({ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"], "2019": [1.0]}),
    )
    _validate_dynamic_share_ssp_alignment(
        prepared_asocc_shares=[asocc_share_matching],
        cc_table=table.loc[table["ssp_scenario"].eq("SSP1")].reset_index(drop=True),
        require_selected_ssp_shares=False,
        cc_source="gwp100_lcia",
        resolved_cc_path=csv_path,
        requested_years=[2019],
    )
    with pytest.raises(ValueError, match="requires SSP dependent aSoCC shares"):
        _validate_dynamic_share_ssp_alignment(
            prepared_asocc_shares=[asocc_share_no_scenario],
            cc_table=table.loc[table["ssp_scenario"].eq("SSP1")].reset_index(drop=True),
            require_selected_ssp_shares=True,
            cc_source="gwp100_lcia",
            resolved_cc_path=csv_path,
            requested_years=[2019],
        )

    scenario_stem_share = _prepared_asocc_share(
        file_stem="external_share__ssp1",
        frame=pd.DataFrame({ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"], "2019": [1.0]}),
    )
    scoped_asocc_shares, has_scenario_dependent_shares = _filter_dynamic_share_ssp_scope(
        prepared_asocc_shares=[
            asocc_share_no_scenario,
            asocc_share_matching,
            scenario_stem_share,
        ],
        cc_table=table.loc[table["ssp_scenario"].eq("SSP1")].reset_index(drop=True),
    )
    assert has_scenario_dependent_shares is True
    assert [asocc_share.file_stem for asocc_share, _frame in scoped_asocc_shares] == [
        "native_share",
        "external_share_ssp1",
        "external_share__ssp1",
    ]
    scoped_without_matching_ssp, has_scenario_dependent_shares = _filter_dynamic_share_ssp_scope(
        prepared_asocc_shares=[asocc_share_no_scenario, scenario_stem_share],
        cc_table=table.loc[table["ssp_scenario"].eq("SSP2")].reset_index(drop=True),
    )
    assert [asocc_share.file_stem for asocc_share, _frame in scoped_without_matching_ssp] == [
        "native_share"
    ]
    assert has_scenario_dependent_shares is True

    no_scenario_only, has_scenario_dependent_shares = _filter_dynamic_share_ssp_scope(
        prepared_asocc_shares=[asocc_share_no_scenario],
        cc_table=table.loc[table["ssp_scenario"].eq("SSP1")].reset_index(drop=True),
    )
    assert len(no_scenario_only) == 1
    assert has_scenario_dependent_shares is False

    impact_labeled = dynamic_compatible_share_frame(
        asocc_share=AsoccShare(
            file_stem="impact_labeled",
            relative_dir=Path("."),
            impacts=("GWP_100",),
            source_label="impact-labeled",
        ),
        share_frame=pd.DataFrame({"value": [1.0]}),
        lcia_method="gwp100_lcia",
    )
    assert impact_labeled is not None
    assert impact_labeled.equals(pd.DataFrame({"value": [1.0]}))

    assert (
        dynamic_compatible_share_frame(
            asocc_share=AsoccShare(
                file_stem="impact_labeled",
                relative_dir=Path("."),
                impacts=("AAL",),
                source_label="impact-labeled",
            ),
            share_frame=pd.DataFrame({"value": [1.0]}),
            lcia_method="gwp100_lcia",
        )
        is None
    )

    generic_share = dynamic_compatible_share_frame(
        asocc_share=AsoccShare(
            file_stem="generic_share",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="generic-share",
        ),
        share_frame=pd.DataFrame({"value": [1.0]}),
        lcia_method="gwp100_lcia",
    )
    assert generic_share is not None
    assert generic_share.equals(pd.DataFrame({"value": [1.0]}))
    assert (
        dynamic_compatible_share_frame(
            asocc_share=AsoccShare(
                file_stem="declared_impact_mismatch",
                relative_dir=Path("."),
                impacts=tuple(),
                source_label="declared-impact-mismatch",
            ),
            share_frame=pd.DataFrame({"impact": ["AAL"], "2019": [1.0]}),
            lcia_method="gwp100_lcia",
        )
        is None
    )

    blank_lcia_generic = dynamic_compatible_share_frame(
        asocc_share=AsoccShare(
            file_stem="empty_lcia",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="empty-lcia",
        ),
        share_frame=pd.DataFrame({"lcia_method": [" ", pd.NA], "2019": [1.0, 2.0]}),
        lcia_method=None,
    )
    assert blank_lcia_generic is not None
    assert blank_lcia_generic.equals(
        pd.DataFrame({"lcia_method": [" ", pd.NA], "2019": [1.0, 2.0]})
    )

    with pytest.raises(ValueError, match="requires exactly one upstream 'lcia_method'"):
        dynamic_compatible_share_frame(
            asocc_share=AsoccShare(
                file_stem="multiple_lcia",
                relative_dir=Path("."),
                impacts=tuple(),
                source_label="multiple-lcia",
            ),
            share_frame=pd.DataFrame({"lcia_method": ["gwp100_lcia", "pb_lcia"]}),
            lcia_method="gwp100_lcia",
        )

    matching_lcia = dynamic_compatible_share_frame(
        asocc_share=AsoccShare(
            file_stem="matching_lcia__gwp100_lcia",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="matching-lcia",
        ),
        share_frame=pd.DataFrame({"lcia_method": ["gwp100_lcia"]}),
        lcia_method="gwp100_lcia",
    )
    assert matching_lcia is not None
    assert matching_lcia.equals(pd.DataFrame({"lcia_method": ["gwp100_lcia"]}))

    inferred_lcia = dynamic_compatible_share_frame(
        asocc_share=AsoccShare(
            file_stem="demo_gwp100_lcia",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="inferred-lcia",
            path=Path("demo_gwp100_lcia.csv"),
        ),
        share_frame=pd.DataFrame({"2019": [1.0]}),
        lcia_method="gwp100_lcia",
    )
    assert inferred_lcia is not None
    assert inferred_lcia.equals(pd.DataFrame({"2019": [1.0]}))
    assert (
        dynamic_compatible_share_frame(
            asocc_share=AsoccShare(
                file_stem="demo_gwp100_lcia",
                relative_dir=Path("."),
                impacts=tuple(),
                source_label="inferred-lcia",
                path=Path("demo_gwp100_lcia.csv"),
            ),
            share_frame=pd.DataFrame({"2018": [1.0]}),
            lcia_method="gwp100_lcia",
            requested_years=[2025, 2026],
        )
        is None
    )

    assert (
        dynamic_compatible_share_frame(
            asocc_share=AsoccShare(
                file_stem="other_lcia__pb_lcia",
                relative_dir=Path("."),
                impacts=tuple(),
                source_label="other-lcia",
            ),
            share_frame=pd.DataFrame({"lcia_method": ["pb_lcia"]}),
            lcia_method="gwp100_lcia",
        )
        is None
    )

    assert (
        _asocc_share_ssp_start_year(
            asocc_share=load_asocc_share(
                AsoccShare(
                    file_stem="demo__ssp2",
                    relative_dir=Path("."),
                    impacts=tuple(),
                    source_label="demo",
                    frame_wide=pd.DataFrame({"2030": [1.0]}),
                )
            ),
            share_transition_meta={"demo__ssp2": {"ssp_start_year": "2030"}},
        )
        == 2030
    )
    assert (
        _asocc_share_ssp_start_year(
            asocc_share=load_asocc_share(
                AsoccShare(
                    file_stem="demo",
                    relative_dir=Path("."),
                    impacts=tuple(),
                    source_label="demo",
                    frame_wide=pd.DataFrame({"2030": [1.0]}),
                )
            ),
            share_transition_meta={},
        )
        is None
    )

    matching_static = static_compatible_share_frame(
        asocc_share=AsoccShare(
            file_stem="matching_static__gwp100_lcia",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="matching-static",
        ),
        share_frame=pd.DataFrame({"lcia_method": ["gwp100_lcia"], "2005": [1.0]}),
        cc_source="gwp100_lcia",
    )
    assert matching_static is not None
    assert matching_static["2005"].tolist() == [1.0]
    assert (
        static_compatible_share_frame(
            asocc_share=AsoccShare(
                file_stem="other_static__pb_lcia",
                relative_dir=Path("."),
                impacts=tuple(),
                source_label="other-static",
            ),
            share_frame=pd.DataFrame({"lcia_method": ["pb_lcia"], "2005": [1.0]}),
            cc_source="gwp100_lcia",
        )
        is None
    )
    assert (
        dynamic_compatible_share_frame(
            asocc_share=AsoccShare(
                file_stem="dynamic_other_method",
                relative_dir=Path("."),
                impacts=tuple(),
                source_label="dynamic-other-method",
            ),
            share_frame=pd.DataFrame({"lcia_method": ["pb_lcia"], "2019": [1.0]}),
            lcia_method="gwp100_lcia",
        )
        is None
    )


def test_process_dynamic_acc_covers_row_ssp_matching_and_duplicate_outputs(
    project_repo: Path,
    tmp_path: Path,
) -> None:
    del project_repo
    cc_table = pd.DataFrame(
        {
            "cc_model": ["M1", "M2"],
            "cc_scenario": ["S1", "S2"],
            "cc_category": ["C1", "C1"],
            "ssp_scenario": ["SSP1", "SSP2"],
            "cc_flow": [CC_FLOW_POSITIVE, CC_FLOW_POSITIVE],
            "cc_variable": [GROSS_ALT_KYOTO_WO_AFOLU, GROSS_ALT_KYOTO_WO_AFOLU],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
            "2020": [10.0, 20.0],
            "2021": [11.0, 21.0],
        }
    )
    share_with_row_ssp = load_asocc_share(
        AsoccShare(
            file_stem="row_ssp_share",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="external",
            frame_wide=pd.DataFrame(
                {
                    "l1_l2_method": ["AR(E^{CBA_FD})"] * 3,
                    "lcia_method": ["gwp100_lcia"] * 3,
                    "impact": ["GWP_100"] * 3,
                    ASOCC_SSP_SCENARIO_COLUMN: ["", "SSP1", "SSP2"],
                    "2020": [0.1, 0.2, 0.3],
                    "2021": [0.4, 0.5, 0.6],
                }
            ),
        )
    )
    invariant_share = load_asocc_share(
        AsoccShare(
            file_stem="invariant_share",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="external",
            frame_wide=pd.DataFrame(
                {
                    "l1_l2_method": ["AR(E^{CBA_FD})"],
                    "lcia_method": ["gwp100_lcia"],
                    "impact": ["GWP_100"],
                    "2020": [0.7],
                    "2021": [0.8],
                }
            ),
        )
    )
    stem_ssp_share = load_asocc_share(
        AsoccShare(
            file_stem="stem_share__ssp1",
            relative_dir=Path("."),
            impacts=tuple(),
            source_label="external",
            frame_wide=pd.DataFrame(
                {
                    "l1_l2_method": ["AR(E^{CBA_FD})"],
                    "lcia_method": ["gwp100_lcia"],
                    "impact": ["GWP_100"],
                    ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"],
                    "2020": [0.9],
                    "2021": [1.0],
                }
            ),
        )
    )
    status = TransientStatusPrinter("deterministic_acc")
    try:
        n_share, n_written, impacts, _dirs, output_files, _cc_path = process_dynamic_acc(
            path_context=build_acc_path_context(
                proj_base=tmp_path,
                source_label="exiobase_396_ixi",
                group_version=None,
                cc_source="gwp100_lcia",
                cc_type="dynamic_ar6",
                public_result_root_name="results_l2_vs_global",
            ),
            public_result_root_name="results_l2_vs_global",
            cc_source="gwp100_lcia",
            asocc_shares=[share_with_row_ssp, invariant_share, stem_ssp_share],
            fmt="csv",
            lcia_method="gwp100_lcia",
            years=range(2020, 2022),
            emission_type="kyoto_gases",
            include_afolu=False,
            emissions_mode="gross_alt",
            share_transition_meta={},
            status=status,
            resolved_cc_path=tmp_path / "cc.csv",
            resolved_cc_table=cc_table,
        )
    finally:
        status.finish()

    assert n_share == 3
    assert n_written == 3
    assert impacts == [GROSS_ALT_KYOTO_WO_AFOLU]
    outputs = {path.stem: pd.read_csv(path) for path in output_files}
    assert len(outputs["row_ssp_share__gwp100_lcia__dynamic_ar6"]) == 4
    assert len(outputs["invariant_share__gwp100_lcia__dynamic_ar6"]) == 2
    assert len(outputs["stem_share__ssp1__gwp100_lcia__dynamic_ar6"]) == 1


def test_deterministic_acc_budget_axis_and_static_min_max_legend_branches(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "__method": ["A", "A"],
            "year": [2020, 2021],
            "value": [1.0, 2.0],
            "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
            "impact": ["GWP_100", "GWP_100"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
            "cc_type": ["dynamic_ar6", "dynamic_ar6"],
            "cc_bound": ["min_cc", "min_cc"],
            "__variant_style": ["solid", "solid"],
            "__series_color": ["#1f77b4", "#1f77b4"],
        }
    )
    fig, axis = plt.subplots()

    numeric = _render_deterministic_dynamic_budget_axis(
        axis=axis,
        frame=frame,
        include_method_in_label=True,
    )

    np.testing.assert_allclose(numeric, [3.0])
    assert [label.get_text() for label in axis.get_xticklabels()] == ["A"]
    plt.close(fig)

    min_max_frame = pd.DataFrame(
        {
            "cc_bound": ["min_cc", "max_cc"],
            ROLE_COLUMN: [MIN_ROLE, MAX_ROLE],
        }
    )
    assert _static_min_max_geometry(min_max_frame, single_year=False) == (
        "Plain = min CC lower retained combination; dotted = max CC upper retained combination."
    )

    color_frame = pd.DataFrame(
        {
            "__method": ["A", "A", "B", "B"],
            "year": [2030, 2030, 2030, 2030],
            "value": [1.0, 2.0, 3.0, 4.0],
            "lcia_method": ["pb_lcia"] * 4,
            "impact": ["AAL", "SOD", "AAL", "SOD"],
            "impact_unit": ["kg"] * 4,
            "cc_type": ["static"] * 4,
            "cc_bound": ["min_cc"] * 4,
        }
    )
    prepared = prepare_plot_rows(color_frame)
    method_colors = {
        method: sorted(set(group["__series_color"].astype(str).tolist()))
        for method, group in prepared.groupby("__method", dropna=False, sort=True)
    }
    assert len(method_colors["A"]) == 1
    assert len(method_colors["B"]) == 1
    assert method_colors["A"] != method_colors["B"]

    dynamic_frame = prepare_plot_rows(
        pd.DataFrame(
            {
                "__method": ["A", "A", "B", "B"],
                "year": [2020, 2021, 2020, 2021],
                "value": [1.0, 2.0, 1.5, 2.5],
                "lcia_method": ["gwp100_lcia"] * 4,
                "impact": ["GWP_100"] * 4,
                "impact_unit": ["kg CO2-eq"] * 4,
                "cc_type": ["dynamic_ar6"] * 4,
                "cc_bound": ["C1"] * 4,
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [
                    ASOCC_TIME_ROUTE_HISTORICAL,
                    ASOCC_TIME_ROUTE_REGRESSION,
                    ASOCC_TIME_ROUTE_HISTORICAL,
                    ASOCC_TIME_ROUTE_REGRESSION,
                ],
            }
        )
    )
    dynamic_paths = _plot_dynamic_scope(
        frame=dynamic_frame,
        requested_years=[2020, 2021],
        output_stem=tmp_path / "dynamic_grouped",
        title="Dynamic aCC grouped methods",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_method_in_label=True,
        figure_note=None,
    )
    assert dynamic_paths == [tmp_path / "dynamic_grouped.svg"]
    assert dynamic_paths[0].exists()


def test_compute_acc_report_and_builder_cover_summary_contracts(tmp_path: Path) -> None:
    single_branch = ACCBranchReport(
        cc_source="gwp100_lcia",
        cc_type="static",
        cc_bounds=["min_cc"],
        n_share_files_processed=1,
        n_acc_files_written=2,
        impacts_used=["GWP_100"],
        output_dirs=[],
        meta_file=None,
    )
    common_lines = ["Project: acc_report_test"]
    single_report = ComputeACCReport(
        branches=[single_branch],
        output_root=tmp_path,
        common_lines=common_lines,
    )
    assert str(single_report)
    assert repr(single_report) == str(single_report)

    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_file = tmp_path / "logs" / SCOPE_MANIFEST_FILENAME
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    meta_file.write_text("{}", encoding="utf-8")
    multi_branch = ACCBranchReport(
        cc_source="gwp100_lcia",
        cc_type="dynamic_ar6",
        cc_bounds=["dynamic"],
        n_share_files_processed=1,
        n_acc_files_written=3,
        impacts_used=["GWP_100"],
        output_dirs=[output_dir],
        meta_file=meta_file,
        phase_index_path=tmp_path / "logs" / "composite_phase_index.json",
        dynamic_ar6_summary=DynamicAR6Summary(
            categories=["C1"],
            ssp_scenarios=["SSP2"],
            subset_version="subset_a",
            pathway_counts=[
                DynamicAR6PathwayCount(
                    category="C1",
                    ssp_scenario="SSP2",
                    model_scenario_pairs=2,
                ),
            ],
            missing_pathway_combinations=[
                DynamicAR6PathwayCount(
                    category="C2",
                    ssp_scenario="SSP5",
                    model_scenario_pairs=0,
                ),
            ],
        ),
    )
    multi_report = ComputeACCReport(
        branches=[single_branch, multi_branch],
        output_root=tmp_path,
        common_lines=common_lines,
    )
    rendered = str(multi_report)
    assert rendered
    assert format_dynamic_ar6_summary_lines(
        DynamicAR6Summary(
            categories=["C1"],
            ssp_scenarios=["SSP2"],
            subset_version=None,
            pathway_counts=[],
            missing_pathway_combinations=[],
        )
    )
    assert build_downstream_common_scope_lines(
        project_name="acc_report_test",
        years=[2020, 2021],
        lcia_methods=["gwp100_lcia"],
        fu_code="L2.a.a",
        group_reg=False,
        group_sec=False,
        group_version=None,
        aggreg_indices=False,
        ssp_scenarios=["SSP2"],
        lca_route="external_lca",
    )
    assert (
        build_downstream_common_scope_lines(
            project_name="acc_report_test",
            years=[2020],
            lcia_methods=[],
            fu_code="L2.a.a",
            group_reg=False,
            group_sec=False,
            group_version=None,
            aggreg_indices=False,
            ssp_scenarios=[],
            lca_route="io_lca",
        )[-1]
        == "LCA route: io_lca"
    )

    long_branch = ACCBranchReport(
        cc_source="dynamic_ar6_with_a_long_source_label",
        cc_type="dynamic_ar6",
        cc_bounds=["category_C1", "category_C2", "category_C3", "category_C4", "category_C5"],
        n_share_files_processed=1,
        n_acc_files_written=1,
        impacts_used=["GWP_100"],
    )
    long_report = ComputeACCReport(
        branches=[long_branch],
        output_root=tmp_path,
        common_lines=common_lines,
    )
    assert str(long_report)
