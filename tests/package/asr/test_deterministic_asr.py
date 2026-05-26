import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa import deterministic_asr, prepare_external_inputs
from pyaesa.ar6_cc.deterministic.request.contracts import CC_FLOW_POSITIVE
from pyaesa.acc.deterministic.runtime.paths import build_acc_path_context, get_acc_output_dir
from pyaesa.acc.shared.runtime.paths import public_result_root_name_for_fu_code
from pyaesa.asr.deterministic.runtime.lca_rows import load_lca_rows
from pyaesa.asr.deterministic.runtime.dynamic import process_dynamic_asr
from pyaesa.asr.deterministic.figures.component_diagnostics import (
    component_rows_from_runtime_frame,
)
from pyaesa.asr.deterministic.state.reports import ASRBranchReport, ComputeASRReport
from pyaesa.download.ar6.utils.config import GROSS_ALT_KYOTO_WO_AFOLU
from pyaesa.asr.shared.runtime.paths import (
    build_asr_path_context,
    get_asr_figure_metadata_path,
)
from pyaesa.shared.acc_asr_common.reporting import (
    DynamicAR6PathwayCount,
    DynamicAR6Summary,
)
from pyaesa.shared.acc_asr_common.scope.composite import (
    build_composite_base_allocate_args,
    normalize_base_asocc_args,
    normalize_shared_lcia_methods,
)
from pyaesa.external_inputs.lca.paths import (
    external_lca_deterministic_dir,
    external_lca_deterministic_figures_dir,
)
from pyaesa.io_lca.data.paths import io_metadata_path_for_source, resolve_io_lca_paths
from pyaesa.shared.runtime.reporting.status import TransientStatusPrinter
from pyaesa.shared.runtime.reporting.composite_phase_index import (
    CompositePhaseIndexEntry,
    PHASE_A_LCA,
    PHASE_B1_AR6_DYNAMIC_CC,
)
from pyaesa.shared.runtime.scenario.columns import (
    AR6_CC_SSP_SCENARIO_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
)
from tests.package.helpers.asr_dummy_repo import (
    prepare_dynamic_asr_io_lca_repo,
    prepare_static_asr_pb_lcia_repo,
    prepare_static_asr_external_lca_repo,
    prepare_static_asr_io_lca_repo,
)

_DYNAMIC_TEST_YEARS = range(2020, 2022)
_DYNAMIC_TEST_YEAR_COLUMNS = ("2020", "2021")


def _run_static_asr(
    *, project_name: str, refresh: bool, figures: bool = False, years: list[int] | None = None
):
    return deterministic_asr(
        project_name=project_name,
        years=[2005] if years is None else years,
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "include_lcia_based_allocation_methods": False,
        },
        base_cc_args={"static": {"exclude_max_cc": True}},
        lca_args={
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        figures=figures,
        figure_format={"format": "svg", "dpi": 1},
        subfigures=False,
        refresh=refresh,
    )


def _asr_root(repo_root: Path, *, project_name: str) -> Path:
    return repo_root / f"{project_name}" / "C_asr"


@pytest.mark.parametrize(
    ("project_name", "figure_options"),
    [
        ("asr_bad_polar_year", {"polar": {"polar_years": [2006]}}),
        ("asr_bad_polar_style", {"polar": {"polar_style": "whisker"}}),
    ],
)
def test_deterministic_asr_rejects_invalid_polar_options(
    project_name: str,
    figure_options: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        deterministic_asr(
            project_name=project_name,
            years=[2005],
            lcia_method="gwp100_lcia",
            fu_code="L2.a.a",
            source="exiobase_396_ixi",
            base_asocc_args={
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "include_lcia_based_allocation_methods": False,
            },
            base_cc_args={"static": {"exclude_max_cc": True}},
            lca_args={
                "external_lca": {"active": False, "version_name": None},
                "io_lca": {"active": True},
            },
            figure_options=figure_options,
            figure_format={"format": "svg", "dpi": 1},
            subfigures=False,
            refresh=True,
        )


def test_deterministic_asr_report_renders_dynamic_ar6_and_external_lca_phases() -> None:
    summary = DynamicAR6Summary(
        categories=["C1"],
        ssp_scenarios=["SSP2"],
        subset_version="subset_a",
        pathway_counts=[
            DynamicAR6PathwayCount(
                category="C1",
                ssp_scenario="SSP2",
                model_scenario_pairs=2,
            )
        ],
        missing_pathway_combinations=[
            DynamicAR6PathwayCount(
                category="C2",
                ssp_scenario="SSP3",
                model_scenario_pairs=0,
            )
        ],
        total_model_scenario_pairs=2,
        process_ar6={
            "reuse_status": "computed",
            "study_period": [2020, 2021],
            "variable_coverage": [
                "ignored",
                {
                    "variable": "Emissions|CO2",
                    "retained_model_scenario_pairs": 2,
                    "available_model_scenario_pairs": 3,
                },
            ],
            "figures_available": 0,
        },
    )
    branch = ASRBranchReport(
        cc_source="ar6",
        cc_type="dynamic_ar6",
        cc_bounds=[],
        lca_type="external",
        n_acc_files_matched=1,
        n_asr_files_written=1,
        impacts_used=["GWP_100"],
        phase_entries=(
            CompositePhaseIndexEntry(
                phase=PHASE_A_LCA,
                function="external_lca",
                status="complete",
                reuse_status="computed",
                output_root=None,
            ),
            CompositePhaseIndexEntry(
                phase=PHASE_B1_AR6_DYNAMIC_CC,
                function="deterministic_ar6_cc",
                status="complete",
                reuse_status="computed",
                output_root=None,
            ),
            CompositePhaseIndexEntry(
                phase=PHASE_B1_AR6_DYNAMIC_CC,
                function="deterministic_acc",
                status="complete",
                reuse_status="computed",
                output_root=None,
            ),
        ),
        dynamic_ar6_summary=summary,
        external_lca_summary={
            "source_type": "deterministic",
            "version_name": "supplier",
            "lcia_method": "gwp100_lcia",
            "figures_available": 2,
            "figure_paths": ["external_a.svg", "external_b.svg"],
        },
    )

    text = str(
        ComputeASRReport(
            branches=[branch],
            output_root=Path("demo"),
            common_lines=["Project: demo"],
        )
    )

    assert text.count("process_ar6") == 1
    assert "external_lca" in text
    assert "Subset version: subset_a" in text
    assert "no retained AR6 CC pathway" in text


def _acc_context(*, proj_base: Path, cc_type: str):
    return build_acc_path_context(
        proj_base=proj_base,
        source_label="exiobase_396_ixi",
        agg_version=None,
        cc_source="gwp100_lcia",
        cc_type=cc_type,
    )


def _asr_context(
    *,
    proj_base: Path,
    lca_type: str,
    cc_type: str,
    fu_code: str = "L2.a.a",
    lca_version_name: str | None = None,
):
    return build_asr_path_context(
        proj_base=proj_base,
        source_label="exiobase_396_ixi",
        agg_version=None,
        fu_code=fu_code,
        lca_type=lca_type,
        cc_source="gwp100_lcia",
        cc_type=cc_type,
        lca_version_name=(
            "supplier_v1"
            if lca_type == "external" and lca_version_name is None
            else lca_version_name
        ),
    )


def _write_acc_wide_table(
    path: Path,
    *,
    l1_l2_method: str,
    lcia_method: str,
    impact: str,
    selector: str,
    ar6_cc_ssp_scenario: str | None,
    values_by_year: dict[int, float],
    cc_category: str,
) -> Path:
    row: dict[str, object] = {
        "share_stem": l1_l2_method,
        "l1_l2_method": l1_l2_method,
        "lcia_method": lcia_method,
        "impact": impact,
        "impact_unit": "kg CO2-eq",
        "r_p": selector,
        "s_p": "D",
        "cc_model": "model_a",
        "cc_scenario": "scenario_a",
        "cc_category": cc_category,
        AR6_CC_SSP_SCENARIO_COLUMN: ar6_cc_ssp_scenario,
    }
    for year, value in values_by_year.items():
        row[str(int(year))] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    return path


def test_deterministic_asr_static_io_lca_end_to_end_reuse_and_refresh(
    allocation_dummy_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    first_report = _run_static_asr(project_name="asr_static_io_lca", refresh=True)
    first_runtime_output = capsys.readouterr().out

    assert first_report is not None
    assert len(first_report.branches) == 1
    branch = first_report.branches[0]
    assert branch.cc_source == "gwp100_lcia"
    assert branch.cc_type == "static"
    assert branch.cc_bounds == ["min_cc"]
    assert branch.lca_type == "io_lca"
    assert branch.n_acc_files_matched == 1
    assert branch.n_asr_files_written == 1
    assert branch.impacts_used == ["GWP_100"]
    assert branch.meta_file is not None and branch.meta_file.exists()
    assert branch.phase_index_path is not None and branch.phase_index_path.exists()
    assert first_runtime_output.strip()
    phase_index_payload = json.loads(branch.phase_index_path.read_text(encoding="utf-8"))
    assert [phase["phase"] for phase in phase_index_payload] == [
        "Phase B.1: aSoCC",
        "Phase B.2: aCC",
        "Phase A: LCA",
        "Phase C: ASR",
    ]
    assert [phase["function"] for phase in phase_index_payload] == [
        "deterministic_asocc",
        "deterministic_acc",
        "deterministic_io_lca",
        "deterministic_asr",
    ]
    assert str(first_report)

    output_paths = sorted(
        _asr_root(
            allocation_dummy_repo.repo_root,
            project_name="asr_static_io_lca",
        ).rglob("*.csv")
    )
    assert len(output_paths) == 1
    assert output_paths[0].name == "UT(FD)__gwp100_lcia.csv"
    output_frame = pd.read_csv(output_paths[0])
    assert {
        "l1_l2_method",
        "impact",
        "impact_unit",
        "2005",
    }.issubset(output_frame.columns)
    assert "asocc_ssp_start_year" not in output_frame.columns
    assert "lca_ssp_start_year" not in output_frame.columns
    assert "share_stem" not in output_frame.columns
    assert set(output_frame["l1_l2_method"]) == {"UT(FD)"}
    assert set(output_frame["impact"]) == {"GWP_100"}
    assert set(output_frame["impact_unit"]) == {"kg CO2-eq"}
    assert bool(output_frame["2005"].gt(0).all())
    assert not any(
        _asr_root(
            allocation_dummy_repo.repo_root,
            project_name="asr_static_io_lca",
        ).parent.parent.rglob("*.png")
    )

    reused_report = _run_static_asr(project_name="asr_static_io_lca", refresh=False)
    assert reused_report.branches[0].reuse_status == "reused_exact"
    reuse_output = capsys.readouterr().out
    assert reuse_output == ""

    output_paths[0].unlink()
    with pytest.raises(ValueError):
        _run_static_asr(project_name="asr_static_io_lca", refresh=False)

    proj_base = allocation_dummy_repo.repo_root / "asr_static_io_lca"
    acc_output = next(
        get_acc_output_dir(context=_acc_context(proj_base=proj_base, cc_type="static")).rglob(
            "*.csv"
        )
    )
    io_paths = resolve_io_lca_paths(
        project_name="asr_static_io_lca",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
    )
    io_output = next(
        (io_metadata_path_for_source(paths=io_paths, source="exiobase_396_ixi").parent.parent)
        .joinpath("results")
        .glob("*.csv")
    )
    acc_output.unlink()
    io_output.unlink()
    refreshed_report = _run_static_asr(project_name="asr_static_io_lca", refresh=True)
    capsys.readouterr()

    assert refreshed_report is not None
    assert refreshed_report.branches[0].n_asr_files_written == 1
    assert acc_output.exists()
    assert io_output.exists()
    expanded_report = _run_static_asr(
        project_name="asr_static_io_lca",
        years=[2005, 2006],
        refresh=False,
    )
    assert expanded_report.branches[0].reuse_status == "computed"
    subset_report = _run_static_asr(project_name="asr_static_io_lca", refresh=False)
    assert subset_report.branches[0].reuse_status == "reused_exact"


def test_deterministic_asr_static_lcia_methods_use_separate_branch_scopes(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    prepare_static_asr_pb_lcia_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        years=[2005],
        impacts=["AAL"],
    )
    kwargs = {
        "project_name": "asr_static_multi_lcia",
        "years": [2005],
        "fu_code": "L2.a.a",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "include_lcia_based_allocation_methods": False,
        },
        "base_cc_args": {"static": {"exclude_max_cc": True}},
        "lca_args": {
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        "figures": False,
        "subfigures": False,
    }

    report = deterministic_asr(
        **kwargs,
        lcia_method=["gwp100_lcia", "pb_lcia"],
        refresh=True,
    )

    assert {branch.cc_source for branch in report.branches} == {"gwp100_lcia", "pb_lcia"}
    route_root = (
        allocation_dummy_repo.repo_root
        / "asr_static_multi_lcia"
        / "C_asr"
        / "exiobase_396_ixi"
        / "io_lca"
        / "deterministic"
    )
    assert sorted(path.name for path in route_root.iterdir()) == [
        "static__gwp100_lcia",
        "static__pb_lcia",
    ]
    for branch_name in ("static__gwp100_lcia", "static__pb_lcia"):
        assert (route_root / branch_name / "logs" / "scope_manifest.json").exists()

    for lcia_method in ("gwp100_lcia", "pb_lcia"):
        reused = deterministic_asr(**kwargs, lcia_method=lcia_method, refresh=False)
        assert reused.branches[0].reuse_status == "reused_exact"


def test_deterministic_asr_static_multi_lcia_reuses_shared_asocc_scope(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    prepare_static_asr_pb_lcia_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        years=[2005],
        impacts=["AAL"],
    )

    report = deterministic_asr(
        project_name="asr_static_multi_lcia_shared_asocc",
        years=[2005],
        lcia_method=["gwp100_lcia", "pb_lcia"],
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
        },
        base_cc_args={"static": {"exclude_max_cc": True}},
        lca_args={
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        figures=False,
        subfigures=False,
        refresh=True,
    )

    assert {branch.cc_source for branch in report.branches} == {"gwp100_lcia", "pb_lcia"}


def test_deterministic_asr_static_figure_reuse_lifecycle(allocation_dummy_repo) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )

    initial_report = _run_static_asr(
        project_name="asr_static_figure_reuse",
        refresh=True,
        figures=True,
    )

    assert initial_report is not None
    branch = initial_report.branches[0]
    assert branch.figure_paths
    assert all(path.exists() for path in branch.figure_paths)
    meta_file = branch.meta_file
    assert meta_file is not None
    figure_meta_file = get_asr_figure_metadata_path(
        context=_asr_context(
            proj_base=allocation_dummy_repo.repo_root / "asr_static_figure_reuse",
            lca_type="io_lca",
            cc_type="static",
        )
    )
    initial_metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    initial_figure_metadata = json.loads(figure_meta_file.read_text(encoding="utf-8"))
    assert initial_metadata["artifacts"]["figure_paths"]
    assert (
        initial_figure_metadata["figure_state"]["paths"]
        == initial_metadata["artifacts"]["figure_paths"]
    )

    reused_figure_report = _run_static_asr(
        project_name="asr_static_figure_reuse",
        refresh=False,
        figures=True,
    )
    assert reused_figure_report.branches[0].reuse_status == "reused_exact"

    skipped_report = _run_static_asr(
        project_name="asr_static_figure_reuse",
        refresh=False,
        figures=False,
    )

    assert skipped_report.branches[0].reuse_status == "reused_exact"
    preserved_metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    preserved_figure_metadata = json.loads(figure_meta_file.read_text(encoding="utf-8"))
    assert (
        preserved_metadata["artifacts"]["figure_paths"]
        == initial_metadata["artifacts"]["figure_paths"]
    )
    assert preserved_figure_metadata == initial_figure_metadata
    assert all(path.exists() for path in branch.figure_paths)


def test_deterministic_asr_static_reference_variant_figures_cover_compression(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )

    report = deterministic_asr(
        project_name="asr_static_reference_variant_figures",
        years=[2006],
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["AR(E^{CBA_FD})"],
            "reference_years": [2005, 2006],
        },
        base_cc_args={"static": {"exclude_max_cc": True}},
        lca_args={
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        figures=True,
        subfigures=False,
        figure_format={"format": "svg", "dpi": 1},
        refresh=True,
    )

    assert report.branches[0].figure_paths
    assert all(path.exists() for path in report.branches[0].figure_paths)


def test_deterministic_asr_static_pb_lcia_public_figures_cover_polar_and_multi_method(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_pb_lcia_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        years=[2005, 2006],
        impacts=["AAL", "BI FD"],
    )

    report = deterministic_asr(
        project_name="asr_static_pb_lcia_public_figures",
        years=[2006],
        lcia_method="pb_lcia",
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "default",
            "include_lcia_based_allocation_methods": False,
            "reference_years": [2005, 2006],
        },
        base_cc_args={"static": {"exclude_max_cc": False}},
        lca_args={
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        figures=True,
        subfigures=False,
        figure_format={"format": "svg", "dpi": 1},
        refresh=True,
    )

    figure_paths = report.branches[0].figure_paths
    assert all(path.exists() for path in figure_paths)
    figure_names = {path.name for path in figure_paths}
    assert "multi_method__pb_lcia__rp_FR__sp_D__2006.svg" in figure_names
    assert any(name.startswith("polar_UT_FD__pb_lcia__") for name in figure_names)


def test_deterministic_asr_static_pb_lcia_multi_year_public_figures_cover_threshold_panels(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_pb_lcia_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        years=[2005, 2006],
        impacts=["AAL", "BI FD"],
    )

    kwargs: dict[str, Any] = dict(
        years=[2005, 2006],
        lcia_method="pb_lcia",
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "include_lcia_based_allocation_methods": False,
        },
        base_cc_args={"static": {}},
        lca_args={
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        figures=True,
        figure_options={
            "per_method": True,
            "multi_method": False,
            "polar": {"polar_years": [2006]},
        },
        subfigures=False,
        figure_format={"format": "svg", "dpi": 1},
    )
    report = deterministic_asr(
        **kwargs,
        project_name="asr_static_pb_lcia_multi_year_public_figures",
        refresh=True,
    )

    figure_paths = report.branches[0].figure_paths
    assert all(path.exists() for path in figure_paths)
    figure_names = {path.name for path in figure_paths}
    assert "UT_FD__pb_lcia__rp_FR__sp_D.svg" in figure_names
    assert "polar_UT_FD__pb_lcia__rp_FR__sp_D__2006.svg" in figure_names
    assert "polar_UT_FD__pb_lcia__rp_FR__sp_D__2005.svg" not in figure_names

    no_product = deterministic_asr(
        **{
            **kwargs,
            "figure_options": {
                "per_method": False,
                "multi_method": False,
                "polar": {"active": False},
            },
        },
        project_name="asr_static_pb_lcia_multi_year_public_figures",
        refresh=False,
    )
    assert no_product.branches[0].figure_paths == []


def test_deterministic_asr_static_external_respects_subfigure_request(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_external_lca_repo(
        allocation_dummy_repo,
        project_name="asr_static_external",
        source="oecd_v2025",
        lcia_method="gwp100_lcia",
        impact="GWP_100",
        impact_unit="kg CO2-eq",
        include_deterministic=True,
    )

    external_kwargs = {
        "project_name": "asr_static_external",
        "years": [2005],
        "lcia_method": "gwp100_lcia",
        "fu_code": "L2.a.a",
        "source": "oecd_v2025",
        "base_asocc_args": {
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "include_lcia_based_allocation_methods": False,
        },
        "base_cc_args": {"static": {"exclude_max_cc": True}},
        "lca_args": {
            "external_lca": {"active": True, "version_name": "supplier_v1"},
            "io_lca": {"active": False},
        },
    }
    figure_format = {"format": "svg", "dpi": 1}
    report = deterministic_asr(
        **external_kwargs,
        figures=True,
        figure_options={"per_method": False, "multi_method": False},
        subfigures=True,
        figure_format=figure_format,
        refresh=True,
    )

    assert report is not None
    assert len(report.branches) == 1
    branch = report.branches[0]
    assert branch.cc_type == "static"
    assert branch.cc_bounds == ["min_cc"]
    assert branch.lca_type == "external"
    assert branch.n_acc_files_matched == 1
    assert branch.n_asr_files_written == 1
    assert branch.figure_paths == []
    assert branch.meta_file is not None
    rerendered_metadata = json.loads(branch.meta_file.read_text(encoding="utf-8"))
    assert rerendered_metadata["provenance"]["external_lca_summary"]["figure_paths"]

    output_paths = sorted(
        _asr_root(
            allocation_dummy_repo.repo_root,
            project_name="asr_static_external",
        ).rglob("*.csv")
    )
    assert len(output_paths) == 1
    assert output_paths[0].name == "UT(FD)__gwp100_lcia.csv"

    subfigure_external_figure_dir = external_lca_deterministic_figures_dir(
        project_base=allocation_dummy_repo.repo_root / "asr_static_external",
    )
    generated_subfigures = sorted(subfigure_external_figure_dir.rglob("supplier_v1__*.svg"))
    assert generated_subfigures
    recorded_subfigures = [
        Path(path)
        for path in rerendered_metadata["provenance"]["external_lca_summary"]["figure_paths"]
    ]
    assert recorded_subfigures

    asr_figure_options = {
        "per_method": True,
        "multi_method": False,
        "polar": {"active": False},
    }
    figure_seed = deterministic_asr(
        **external_kwargs,
        figures=True,
        figure_options=asr_figure_options,
        subfigures=True,
        figure_format=figure_format,
        refresh=False,
    )
    seeded_branch = figure_seed.branches[0]
    assert seeded_branch.figure_paths
    seeded_metadata = json.loads(cast(Path, seeded_branch.meta_file).read_text())
    seeded_subfigures = [
        Path(path) for path in seeded_metadata["provenance"]["external_lca_summary"]["figure_paths"]
    ]
    seeded_subfigures[0].unlink()

    subfigure_repair = deterministic_asr(
        **external_kwargs,
        figures=True,
        figure_options=asr_figure_options,
        subfigures=True,
        figure_format=figure_format,
        refresh=False,
    )
    assert subfigure_repair.branches[0].reuse_status == "partially_reused"
    repaired_metadata = json.loads(cast(Path, subfigure_repair.branches[0].meta_file).read_text())
    repaired_paths = [
        Path(path)
        for path in repaired_metadata["provenance"]["external_lca_summary"]["figure_paths"]
    ]
    assert repaired_paths
    assert all(path.exists() for path in repaired_paths)

    repaired_paths[0].unlink()
    figure_and_subfigure_repair = deterministic_asr(
        **external_kwargs,
        figures=True,
        figure_options={
            "per_method": True,
            "multi_method": False,
            "polar": {"polar_years": [2005]},
        },
        subfigures=True,
        figure_format=figure_format,
        refresh=False,
    )
    repaired_with_asr_figures = figure_and_subfigure_repair.branches[0]
    assert repaired_with_asr_figures.figure_paths
    assert all(path.exists() for path in repaired_with_asr_figures.figure_paths)
    refreshed_metadata = json.loads(cast(Path, repaired_with_asr_figures.meta_file).read_text())
    refreshed_paths = [
        Path(path)
        for path in refreshed_metadata["provenance"]["external_lca_summary"]["figure_paths"]
    ]
    assert all(path.exists() for path in refreshed_paths)


@pytest.mark.parametrize("refresh", [False, True])
def test_deterministic_asr_external_rejects_shared_scope_identity_drift(
    allocation_dummy_repo,
    refresh: bool,
) -> None:
    prepare_static_asr_external_lca_repo(
        allocation_dummy_repo,
        project_name="asr_external_identity_guard",
        source="oecd_v2025",
        lcia_method="gwp100_lcia",
        impact="GWP_100",
        impact_unit="kg CO2-eq",
        include_deterministic=True,
    )

    first_report = deterministic_asr(
        project_name="asr_external_identity_guard",
        years=[2005],
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        source="oecd_v2025",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "include_lcia_based_allocation_methods": False,
        },
        base_cc_args={"static": {"exclude_max_cc": True}},
        lca_args={
            "external_lca": {"active": True, "version_name": "supplier_v1"},
            "io_lca": {"active": False},
        },
        figures=False,
        subfigures=False,
        refresh=True,
    )

    assert first_report is not None
    first_branch = first_report.branches[0]
    first_meta_file = cast(Path, first_branch.meta_file)
    first_metadata = json.loads(first_meta_file.read_text(encoding="utf-8"))
    first_output_files = [Path(path) for path in first_metadata["artifacts"]["output_files"]]
    assert all(path.exists() for path in first_output_files)
    with pytest.raises(ValueError):
        deterministic_asr(
            project_name="asr_external_identity_guard",
            years=[2005],
            lcia_method="gwp100_lcia",
            fu_code="L2.b.a",
            source="oecd_v2025",
            base_asocc_args={
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "include_lcia_based_allocation_methods": False,
            },
            base_cc_args={"static": {"exclude_max_cc": True}},
            lca_args={
                "external_lca": {"active": True, "version_name": "supplier_v1"},
                "io_lca": {"active": False},
            },
            figures=False,
            subfigures=False,
            refresh=refresh,
        )
    assert first_meta_file.exists()
    assert all(path.exists() for path in first_output_files)


def test_deterministic_asr_dynamic_io_lca_end_to_end_and_reuse(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
        historical_years=[2020, 2021],
        scenario_years=[2030],
    )

    first_report = deterministic_asr(
        project_name="asr_dynamic_reuse",
        years=_DYNAMIC_TEST_YEARS,
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C1"], "ssp_scenario": ["SSP1"]},
        },
        lca_args={
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        figures=False,
        figure_format={"format": "svg", "dpi": 1},
        subfigures=False,
        refresh=True,
    )

    assert first_report is not None
    assert len(first_report.branches) == 1
    branch = first_report.branches[0]
    assert branch.cc_source == "gwp100_lcia"
    assert branch.cc_type == "dynamic_ar6"
    assert branch.cc_bounds == ["C1"]
    assert branch.lca_type == "io_lca"
    assert branch.n_acc_files_matched == 1
    assert branch.n_asr_files_written == 1
    assert branch.figure_paths == []
    assert branch.impacts_used == ["GWP_100"]
    assert branch.meta_file is not None and branch.meta_file.exists()

    output_paths = sorted(
        _asr_root(
            allocation_dummy_repo.repo_root,
            project_name="asr_dynamic_reuse",
        ).rglob("*.csv")
    )
    result_paths = [path for path in output_paths if path.name != "dynamic_component_rows.csv"]
    component_paths = [path for path in output_paths if path.name == "dynamic_component_rows.csv"]
    assert {path.name for path in output_paths} == {
        "dynamic_component_rows.csv",
        "UT(FD)__gwp100_lcia__dynamic_ar6.csv",
    }
    assert len(component_paths) == 1

    first_output = pd.concat([pd.read_csv(path) for path in result_paths], ignore_index=True)
    assert {
        "l1_l2_method",
        "impact",
        "impact_unit",
        "cc_category",
        "cc_flow",
        "cc_variable",
        "cumulative_asr",
        *_DYNAMIC_TEST_YEAR_COLUMNS,
    }.issubset(first_output.columns)
    assert "asocc_ssp_start_year" not in first_output.columns
    assert "lca_ssp_start_year" not in first_output.columns
    assert "share_stem" not in first_output.columns
    assert set(first_output["l1_l2_method"]) == {"UT(FD)"}
    assert set(first_output["impact"]) == {"GWP_100"}
    assert set(first_output["impact_unit"]) == {"kg CO2-eq"}
    assert set(first_output["cc_category"]) == {"C1"}
    assert set(first_output["cc_flow"]) == {CC_FLOW_POSITIVE}
    assert set(first_output["cc_variable"]) == {GROSS_ALT_KYOTO_WO_AFOLU}
    assert set(_DYNAMIC_TEST_YEAR_COLUMNS).issubset(first_output.columns)
    for column in _DYNAMIC_TEST_YEAR_COLUMNS:
        assert bool(first_output[column].gt(0).all())
        assert bool(first_output[column].lt(1).all())
    assert bool(first_output["cumulative_asr"].gt(0).all())
    assert bool(first_output["cumulative_asr"].lt(1).all())
    assert "cumulative_no_transgression" not in first_output.columns

    reused_dynamic_report = deterministic_asr(
        project_name="asr_dynamic_reuse",
        years=_DYNAMIC_TEST_YEARS,
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C1"], "ssp_scenario": ["SSP1"]},
        },
        lca_args={
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        figures=False,
        subfigures=False,
        refresh=False,
    )
    assert reused_dynamic_report.branches[0].reuse_status == "reused_exact"


def test_deterministic_asr_dynamic_external_lca_records_lca_transition_year(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
        historical_years=[2020, 2030],
        scenario_years=[2030],
    )
    project_name = "asr_dynamic_external_lca_transition"
    years = [2020, 2030]
    external_report = prepare_external_inputs(project_name=project_name)
    external_dir = external_lca_deterministic_dir(project_base=external_report.project_root)
    pd.DataFrame(
        [
            {
                "r_p": r_p,
                "s_p": s_p,
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "2020": value,
            }
            for r_p, s_p, value in (
                ("FR", "D", 1.0),
                ("FR", "X", 2.0),
                ("US", "D", 3.0),
                ("US", "X", 4.0),
            )
        ]
    ).to_csv(external_dir / "supplier_v1__gwp100_lcia.csv", index=False)
    pd.DataFrame(
        [
            {
                "r_p": r_p,
                "s_p": s_p,
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "2030": value + 1.0,
            }
            for r_p, s_p, value in (
                ("FR", "D", 1.0),
                ("FR", "X", 2.0),
                ("US", "D", 3.0),
                ("US", "X", 4.0),
            )
        ]
    ).to_csv(external_dir / "supplier_v1__gwp100_lcia__ssp2.csv", index=False)
    base_allocate_args = build_composite_base_allocate_args(
        project_name=project_name,
        years=years,
        lcia_method=normalize_shared_lcia_methods("gwp100_lcia"),
        fu_code="L2.a.a",
        source="exiobase_396_ixi",
        agg_reg=False,
        agg_sec=False,
        agg_version=None,
        group_indices=False,
        base_asocc_args=normalize_base_asocc_args(
            {
                "method_plan": "one_step",
                "one_step_methods": ["AR(E^{CBA_FD})"],
                "ssp_scenario": ["SSP2"],
            },
            fu_code="L2.a.a",
        ),
        r_p=None,
        s_p=None,
        r_c=None,
        r_f=None,
    )
    acc_dir = get_acc_output_dir(
        context=_acc_context(
            proj_base=external_report.project_root,
            cc_type="dynamic_ar6",
        ),
        public_result_root_name=public_result_root_name_for_fu_code(fu_code="L2.a.a"),
    )
    acc_path = _write_acc_wide_table(
        acc_dir / "AR(E^{CBA_FD})__gwp100_lcia__dynamic_ar6.csv",
        l1_l2_method="AR(E^{CBA_FD})",
        lcia_method="gwp100_lcia",
        impact="GWP_100",
        selector="FR",
        ar6_cc_ssp_scenario="SSP2",
        cc_category="C1",
        values_by_year={2020: 10.0, 2030: 12.0},
    )
    acc_frame = pd.read_csv(acc_path)
    acc_frame["l2_method"] = "AR(E^{CBA_FD})"
    acc_frame[ASOCC_SSP_SCENARIO_COLUMN] = "SSP2"
    acc_frame.to_csv(acc_path, index=False)
    lca_rows = load_lca_rows(
        proj_base=external_report.project_root,
        source_label="exiobase_396_ixi",
        lca_type="external",
        lcia_method="gwp100_lcia",
        lca_version_name="supplier_v1",
        base_allocate_args=base_allocate_args,
        years=list(years),
    )

    status = TransientStatusPrinter("test_deterministic_asr")
    try:
        process_result = process_dynamic_asr(
            proj_base=external_report.project_root,
            fu_code="L2.a.a",
            cc_source="gwp100_lcia",
            source_label="exiobase_396_ixi",
            base_allocate_args=base_allocate_args,
            years=years,
            fmt="csv",
            lca_type="external",
            lca_version_name="supplier_v1",
            acc_output_files=[acc_path],
            allowed_l1_l2_methods={"AR(E^{CBA_FD})"},
            share_transition_meta={},
            lca_rows=lca_rows,
            status=status,
        )
    finally:
        status.finish()

    assert process_result.n_written == 1
    output_frame = pd.read_csv(process_result.output_files[0])
    assert set(output_frame["lca_ssp_start_year"].astype(int)) == {2030}
    assert process_result.dynamic_component_frame is not None
    component_rows = component_rows_from_runtime_frame(
        component_frame=process_result.dynamic_component_frame,
        lca_rows=lca_rows,
        acc_output_files=[acc_path],
    )
    assert component_rows.acc["__component_value"].tolist() == [10.0, 12.0]
    assert component_rows.lca["__component_value"].tolist() == [1.0, 2.0]


def test_deterministic_asr_rejects_invalid_lca_route_block(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        deterministic_asr(
            project_name="asr_bad_lca_type",
            years=[2005],
            lcia_method="gwp100_lcia",
            fu_code="L2.a.a",
            source="exiobase_396_ixi",
            base_asocc_args={
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "include_lcia_based_allocation_methods": False,
            },
            base_cc_args={"static": {"exclude_max_cc": True}},
            lca_args={"bad_route": {"active": True}},
            figures=False,
            subfigures=False,
            refresh=True,
        )


def test_deterministic_asr_rejects_non_consecutive_dynamic_years(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        deterministic_asr(
            project_name="asr_bad_dynamic_years",
            years=[2019, 2021],
            lcia_method="gwp100_lcia",
            fu_code="L2.a.a",
            source="exiobase_396_ixi",
            base_asocc_args={
                "method_plan": "one_step",
                "one_step_methods": ["AR(E^{CBA_FD})"],
            },
            base_cc_args={"static": {"active": False}, "dynamic_ar6": {}},
            lca_args={
                "external_lca": {"active": False, "version_name": None},
                "io_lca": {"active": True},
            },
            figures=False,
            subfigures=False,
            refresh=True,
        )


def test_deterministic_asr_rejects_non_boolean_subfigures(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        deterministic_asr(
            project_name="asr_bad_subfigures",
            years=[2005],
            lcia_method="gwp100_lcia",
            fu_code="L2.a.a",
            source="exiobase_396_ixi",
            base_asocc_args={
                "method_plan": "one_step",
                "one_step_methods": ["UT(FD)"],
                "include_lcia_based_allocation_methods": False,
            },
            base_cc_args={"static": {"exclude_max_cc": True}},
            lca_args={
                "external_lca": {"active": False, "version_name": None},
                "io_lca": {"active": True},
            },
            figures=False,
            subfigures=cast(Any, "bad"),
            refresh=True,
        )
