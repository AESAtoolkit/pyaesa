from pathlib import Path

import pandas as pd
import pytest

from pyaesa import prepare_external_inputs
from pyaesa.acc.deterministic.runtime.paths import (
    build_acc_path_context,
    get_acc_output_dir,
)
from pyaesa.acc.shared.runtime.paths import public_result_root_name_for_fu_code
from pyaesa.asr.deterministic.runtime import common as common_mod
from pyaesa.asr.deterministic.runtime import compute as compute_mod
from pyaesa.asr.deterministic.runtime import dynamic as dynamic_mod
from pyaesa.asr.deterministic.runtime import lca_rows as lca_rows_mod
from pyaesa.shared.acc_asr_common.deterministic.state import scope_guard as scope_guard_mod
from pyaesa.asr.deterministic.runtime import static as static_mod
from pyaesa.shared.tabular.wide_tables import first_non_null_scenario_year
from pyaesa.asr.deterministic.runtime import tables as tables_mod
from pyaesa.asr.shared.runtime import paths as shared_paths_mod
from pyaesa.shared.lcia import units as units_mod
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.external_inputs.lca.paths import external_lca_deterministic_dir
from pyaesa.io_lca.data.contracts import IO_LCA_FAMILY
from pyaesa.io_lca.data.paths import lca_results_dir_for_source, resolve_io_lca_paths
from pyaesa.shared.lcia.contracts import bundled_cc_expected_impact_units
from pyaesa.shared.runtime.metadata.contracts import SCOPE_MANIFEST_FILENAME
from pyaesa.shared.runtime.manifest_contract import manifest_digest
from pyaesa.shared.runtime.reporting.status import TransientStatusPrinter
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN


def _base_allocate_args(*, project_name: str) -> dict[str, object]:
    return normalize_base_allocate_args(
        {
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "group_reg": False,
            "group_sec": False,
            "group_version": None,
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "l1_methods": None,
            "one_step_methods": ["UT(FD)"],
            "two_step_methods": None,
            "l1_l2_pairs": None,
            "lcia_method": "gwp100_lcia",
            "years": [2005, 2030],
            "reference_years": None,
            "ssp_scenario": None,
            "projection_mode": None,
            "reg_window": None,
            "l2_reuse_years": None,
            "l1_reg_aggreg": "post",
            "aggreg_indices": False,
        }
    )


def _historical_external_base_allocate_args(*, project_name: str) -> dict[str, object]:
    args = dict(_base_allocate_args(project_name=project_name))
    args["years"] = [2005]
    args["reg_window"] = [2005, 2006]
    return args


def _write_acc_file(
    path: Path,
    *,
    l1_l2_method: str = "UT(FD)",
    l2_method: str | None = None,
    impact: str = "GWP_100",
    impact_unit: str = "t CO2eq/yr",
    year_values: dict[int, float] | None = None,
    cc_bound: str | None = None,
    extra_columns: dict[str, object] | None = None,
) -> Path:
    row: dict[str, object] = {
        "l1_l2_method": l1_l2_method,
        "l2_method": l1_l2_method if l2_method is None else l2_method,
        "impact": impact,
        "impact_unit": impact_unit,
        "r_p": "FR",
        "s_p": "D",
    }
    if cc_bound is not None:
        row["cc_bound"] = cc_bound
    row.update(extra_columns or {})
    for year, value in (year_values or {2005: 2.0}).items():
        row[str(int(year))] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    return path


def _write_io_lca_results(
    path: Path,
    *,
    lcia_method: str = "gwp100_lcia",
    impact: str = "GWP_100",
    impact_unit: str = "kg CO2eq/yr",
    year_values: dict[int, float] | None = None,
) -> Path:
    rows = [
        {
            "year": int(year),
            "impact": impact,
            "lca_value": value,
            "impact_unit": impact_unit,
            "lcia_method": lcia_method,
            "r_p": "FR",
            "s_p": "D",
        }
        for year, value in (year_values or {2005: 1000.0}).items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_external_lca_file(
    path: Path,
    *,
    impact: str = "GWP_100",
    impact_unit: str = "kg CO2-eq",
    year_values: dict[int, float] | None = None,
) -> Path:
    row: dict[str, object] = {
        "r_p": "FR",
        "s_p": "D",
        "impact": impact,
        "impact_unit": impact_unit,
    }
    for year, value in (year_values or {2005: 1000.0}).items():
        row[str(int(year))] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def _status() -> TransientStatusPrinter:
    return TransientStatusPrinter("deterministic_asr")


def _acc_context(
    *,
    proj_base: Path,
    source_label: str,
    cc_type: str,
):
    return build_acc_path_context(
        proj_base=proj_base,
        source_label=source_label,
        group_version=None,
        cc_source="gwp100_lcia",
        cc_type=cc_type,
    )


def _asr_context(
    *,
    proj_base: Path,
    source_label: str,
    fu_code: str,
    lca_type: str,
    cc_type: str,
    lca_version_name: str | None = None,
):
    return shared_paths_mod.build_asr_path_context(
        proj_base=proj_base,
        source_label=source_label,
        group_version=None,
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


def test_shared_runtime_paths_cover_dynamic_scope_guard_and_results_dir(tmp_path: Path) -> None:
    assert (
        shared_paths_mod.build_asr_scope_label(
            source_label="oecd_v2025",
            group_version=None,
            lca_type="external",
            cc_source="gwp100_lcia",
            cc_type="dynamic_ar6",
            lca_version_name="supplier_v1",
        )
        == "oecd_v2025__external_lca__supplier_v1__dynamic_ar6__gwp100_lcia"
    )

    assert shared_paths_mod.get_asr_root(proj_base=tmp_path) == (tmp_path / "C_asr")
    assert not (tmp_path / "C_asr").exists()
    assert shared_paths_mod.get_asr_route_root(
        proj_base=tmp_path,
        source_label="oecd_v2025",
        group_version=None,
        lca_type="external",
        lca_version_name="supplier_v1",
    ) == (tmp_path / "C_asr" / "oecd_v2025" / "external_lca__supplier_v1")
    assert not (tmp_path / "C_asr" / "oecd_v2025" / "external_lca__supplier_v1").exists()
    branch_root = (
        tmp_path
        / "C_asr"
        / "oecd_v2025"
        / "external_lca__supplier_v1"
        / "deterministic"
        / "static__gwp100_lcia"
    )
    assert shared_paths_mod.get_asr_results_dir(
        context=_asr_context(
            proj_base=tmp_path,
            source_label="oecd_v2025",
            fu_code="L2.a.a",
            lca_type="external",
            cc_type="static",
        ),
    ) == (branch_root / "results_l2_vs_global")
    assert branch_root.is_dir()
    assert shared_paths_mod.get_asr_logs_dir(
        context=_asr_context(
            proj_base=tmp_path,
            source_label="oecd_v2025",
            fu_code="L2.a.a",
            lca_type="external",
            cc_type="static",
        ),
    ) == (branch_root / "logs")
    assert not (branch_root / "logs").exists()
    assert shared_paths_mod.get_asr_meta_path(
        context=_asr_context(
            proj_base=tmp_path,
            source_label="oecd_v2025",
            fu_code="L2.a.a",
            lca_type="external",
            cc_type="static",
        ),
    ) == (branch_root / "logs" / SCOPE_MANIFEST_FILENAME)
    assert not (branch_root / "logs").exists()
    assert shared_paths_mod.get_asr_figures_dir(
        context=_asr_context(
            proj_base=tmp_path,
            source_label="oecd_v2025",
            fu_code="L2.a.a",
            lca_type="external",
            cc_type="static",
        ),
    ) == (branch_root / "figures_l2_vs_global")
    assert not (branch_root / "figures_l2_vs_global").exists()
    identity = {"source": "demo"}
    assert (
        scope_guard_mod.branch_reuse_mode_or_raise(
            existing_metadata=None,
            requested_identity=identity,
            requested_coverage={"years": [2005]},
            scope_label="demo_scope",
            function_name="deterministic_asr",
        )
        == "compute"
    )
    existing_metadata = {
        "reuse": {
            "identity_key": manifest_digest(identity),
            "coverage": {"years": [2005, 2006], "impact": ["GWP_100"]},
        },
    }
    assert (
        scope_guard_mod.branch_reuse_mode_or_raise(
            existing_metadata=existing_metadata,
            requested_identity=identity,
            requested_coverage={"years": [2005], "impact": ["GWP_100"]},
            scope_label="demo_scope",
            function_name="deterministic_asr",
        )
        == "reuse"
    )
    assert (
        scope_guard_mod.branch_reuse_mode_or_raise(
            existing_metadata=existing_metadata,
            requested_identity=identity,
            requested_coverage={"years": [2005, 2006, 2007], "impact": ["GWP_100"]},
            scope_label="demo_scope",
            function_name="deterministic_asr",
        )
        == "append"
    )
    with pytest.raises(ValueError):
        scope_guard_mod.branch_reuse_mode_or_raise(
            existing_metadata=existing_metadata,
            requested_identity={"source": "other"},
            requested_coverage={"years": [2005]},
            scope_label="demo_scope",
            function_name="deterministic_asr",
        )
    with pytest.raises(ValueError):
        scope_guard_mod.branch_reuse_mode_or_raise(
            existing_metadata=existing_metadata,
            requested_identity=identity,
            requested_coverage={"years": [2006, 2007], "impact": ["GWP_100"]},
            scope_label="demo_scope",
            function_name="deterministic_asr",
        )
    with pytest.raises(ValueError):
        scope_guard_mod.branch_reuse_mode_or_raise(
            existing_metadata=existing_metadata,
            requested_identity=identity,
            requested_coverage={"years": [2005], "impact": ["GWP_100", "ODP"]},
            scope_label="demo_scope",
            function_name="deterministic_asr",
        )
    assert scope_guard_mod.merged_coverage(
        existing_metadata=existing_metadata,
        requested_coverage={"years": [2007], "impact": ["ODP"]},
    ) == {"impact": ["GWP_100", "ODP"], "years": [2005, 2006, 2007]}
    assert scope_guard_mod.merged_coverage(
        existing_metadata=None,
        requested_coverage={"years": [2005]},
    ) == {"years": [2005]}
    assert scope_guard_mod.coverage_signature_covers(
        {"identity_key": "demo", "coverage": {"years": [2005, 2006]}},
        {"identity_key": "demo", "coverage": {"years": [2005]}},
    )
    assert not scope_guard_mod.coverage_signature_covers(
        {"identity_key": "demo", "coverage": {"years": [2005, 2006]}},
        {"identity_key": "other", "coverage": {"years": [2005]}},
    )
    assert not scope_guard_mod.coverage_signature_covers(
        {"identity_key": "demo", "coverage": {"years": [2005]}},
        {"identity_key": "demo", "coverage": {"years": [2006]}},
    )


def test_common_runtime_contracts_cover_text_scenarios_and_transition_contracts() -> None:
    lca_rows = pd.DataFrame({"lca_ssp_scenario": [None, "SSP2"], "year": ["2005", "2030"]})
    assert common_mod.build_external_transition(lca_rows, lca_type="other") is None
    assert common_mod.build_external_transition(lca_rows, lca_type="external") == {
        "switch_year": 2030,
        "marker_label": "external LCA SSP-dependent switch",
        "marker_color": "#375a7f",
    }
    assert (
        common_mod.asocc_ssp_transition_start_year(
            output_stem="demo_SSP2",
            share_transition_meta={
                "demo_SSP2": {
                    "base_stem": "demo",
                    ASOCC_SSP_SCENARIO_COLUMN: "SSP2",
                    "asocc_ssp_scenario_labels": ["SSP2"],
                    "ssp_start_year": 2030,
                    "marker_label": "Switch year for SSP-dependent series",
                    "marker_color": "#7d7d7d",
                }
            },
        )
        == 2030
    )
    assert (
        common_mod.asocc_ssp_transition_start_year(
            output_stem="demo_SSP2",
            share_transition_meta={
                "demo_SSP2": {
                    "base_stem": "demo",
                    ASOCC_SSP_SCENARIO_COLUMN: "SSP2",
                    "asocc_ssp_scenario_labels": ["SSP2"],
                    "ssp_start_year": None,
                    "marker_label": "Switch year for SSP-dependent series",
                    "marker_color": "#7d7d7d",
                }
            },
        )
        is None
    )


def test_table_contracts_cover_read_and_write_paths(tmp_path: Path) -> None:
    frame = pd.DataFrame({"impact": ["GWP_100"], "2030": [1.0]})
    output_path = tmp_path / "out" / "asr.csv"
    tables_mod.write_asr_output(frame, output_path, "csv")
    assert output_path.exists()
    assert pd.read_csv(output_path).equals(frame)


def test_lca_row_loading_covers_io_and_external_routes(
    io_lca_dummy_repo,
    allocation_dummy_repo,
) -> None:
    base_allocate_args = _base_allocate_args(project_name="asr_runtime_io")
    paths = resolve_io_lca_paths(
        project_name="asr_runtime_io",
        group_reg=False,
        group_sec=False,
        group_version=None,
    )
    results_dir = lca_results_dir_for_source(paths=paths, source="exiobase_396_ixi")
    io_rows = pd.DataFrame(
        [
            {
                "year": 2030,
                "impact": "GWP_100",
                "lca_value": 1000.0,
                "impact_unit": "kg CO2eq/yr",
                "lcia_method": "gwp100_lcia",
                "r_p": "FR",
                "s_p": "D",
            }
        ]
    )
    good_path = results_dir / "gwp100_lcia.csv"
    good_path.parent.mkdir(parents=True, exist_ok=True)
    io_rows.to_csv(good_path, index=False)

    discovered = lca_rows_mod._discover_io_lca_result_files(  # noqa: SLF001
        source_label="exiobase_396_ixi",
        base_allocate_args=base_allocate_args,
    )
    assert discovered == [good_path]

    loaded_io = lca_rows_mod._load_io_lca_rows(  # noqa: SLF001
        source_label="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        base_allocate_args=base_allocate_args,
    )
    assert loaded_io.loc[0, "year"] == "2030"
    assert loaded_io.loc[0, "lca_value"] == pytest.approx(1000.0)

    assert (
        lca_rows_mod.load_lca_rows(
            proj_base=paths.project_base,
            source_label="exiobase_396_ixi",
            lca_type=IO_LCA_FAMILY,
            lcia_method="gwp100_lcia",
            lca_version_name=None,
            base_allocate_args=base_allocate_args,
            years=[2030],
        ).shape[0]
        == 1
    )

    external_args = _base_allocate_args(project_name="asr_runtime_external")
    external_proj_base = allocation_dummy_repo.repo_root / "asr_runtime_external"
    external_dir = external_lca_deterministic_dir(project_base=external_proj_base)
    external_dir.mkdir(parents=True, exist_ok=True)
    cc_path, expected_pairs = bundled_cc_expected_impact_units(lcia_method="gwp100_lcia")
    del cc_path
    impact, impact_unit = expected_pairs[0]
    pd.DataFrame(
        {
            "r_p": ["FR"],
            "s_p": ["D"],
            "impact": [impact],
            "impact_unit": [impact_unit],
            "2005": [2.5],
        }
    ).to_csv(external_dir / "supplier_v1__gwp100_lcia.csv", index=False)

    loaded_external = lca_rows_mod._load_contextual_external_rows(  # noqa: SLF001
        proj_base=external_proj_base,
        version_name="supplier_v1",
        lcia_method="gwp100_lcia",
        base_allocate_args=external_args,
        years=[2005],
    )
    assert loaded_external["year"].tolist() == ["2005"]
    assert set(loaded_external["impact"]) == {impact}

    assert (
        lca_rows_mod.load_lca_rows(
            proj_base=external_proj_base,
            source_label="exiobase_396_ixi",
            lca_type="external",
            lcia_method="gwp100_lcia",
            lca_version_name="supplier_v1",
            base_allocate_args=external_args,
            years=[2005],
        ).shape[0]
        == 1
    )


def test_unit_contracts_cover_normalization_splitting_and_known_conversions() -> None:
    assert units_mod._normalize_unit_label(" kg ") == "kg"  # noqa: SLF001
    assert units_mod._normalize_unit_label("kg eq / year") == "kg_per_yr"  # noqa: SLF001
    assert units_mod._normalize_unit_label("kt CH4 eq / year") == "kt_ch4_per_yr"  # noqa: SLF001
    assert units_mod._normalize_unit_label("kg methane/year") == "kg_methane_per_yr"  # noqa: SLF001
    assert units_mod._normalize_unit_label("items") == "items"  # noqa: SLF001
    assert units_mod._split_mass_unit("kt CH4 eq / year") == (  # noqa: SLF001
        "kt",
        "_ch4",
        "_per_yr",
    )
    assert units_mod._split_mass_unit("items") is None  # noqa: SLF001
    assert units_mod.try_unit_conversion("kg", "kg") == pytest.approx(1.0)
    assert units_mod.try_unit_conversion(
        "kt CH4 eq / year",
        "t CH4 eq / year",
    ) == pytest.approx(1000.0)
    assert units_mod.try_unit_conversion("kg N eq / year", "t CH4 eq / year") is None
    assert units_mod.try_unit_conversion("items", "kg") is None


def test_lca_row_loading_covers_missing_discovery_and_external_absence(
    allocation_dummy_repo,
) -> None:
    missing_args = _base_allocate_args(project_name="asr_runtime_missing_io")
    assert (
        lca_rows_mod._discover_io_lca_result_files(  # noqa: SLF001
            source_label="exiobase_396_ixi",
            base_allocate_args=missing_args,
        )
        == []
    )
    with pytest.raises(FileNotFoundError):
        lca_rows_mod._load_io_lca_rows(  # noqa: SLF001
            source_label="exiobase_396_ixi",
            lcia_method="gwp100_lcia",
            base_allocate_args=missing_args,
        )

    mismatch_args = _base_allocate_args(project_name="asr_runtime_mismatch_io")
    mismatch_paths = resolve_io_lca_paths(
        project_name="asr_runtime_mismatch_io",
        group_reg=False,
        group_sec=False,
        group_version=None,
    )
    _write_io_lca_results(
        lca_results_dir_for_source(paths=mismatch_paths, source="exiobase_396_ixi")
        / "other_method.csv",
        lcia_method="other_method",
    )
    _write_io_lca_results(
        lca_results_dir_for_source(paths=mismatch_paths, source="exiobase_396_ixi")
        / "other_method_2.csv",
        lcia_method="other_method",
    )
    with pytest.raises(FileNotFoundError):
        lca_rows_mod._load_io_lca_rows(  # noqa: SLF001
            source_label="exiobase_396_ixi",
            lcia_method="gwp100_lcia",
            base_allocate_args=mismatch_args,
        )

    external_args = _historical_external_base_allocate_args(
        project_name="asr_runtime_missing_external"
    )
    external_report = prepare_external_inputs(project_name="asr_runtime_missing_external")
    with pytest.raises(FileNotFoundError):
        lca_rows_mod._load_contextual_external_rows(  # noqa: SLF001
            proj_base=external_report.project_root,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
            base_allocate_args=external_args,
            years=[2005],
        )
    assert (
        first_non_null_scenario_year(
            pd.DataFrame({"scenario": [None, None], "year": ["2005", "2030"]}),
            scenario_column="scenario",
        )
        is None
    )
    del allocation_dummy_repo


def test_static_runtime_contracts_cover_skip_paths_external_alignment_and_max_derivation(
    allocation_dummy_repo,
) -> None:
    project_name = "asr_runtime_static_process"
    base_allocate_args = _base_allocate_args(project_name=project_name)
    paths = resolve_io_lca_paths(
        project_name=project_name,
        group_reg=False,
        group_sec=False,
        group_version=None,
    )
    results_dir = lca_results_dir_for_source(paths=paths, source="exiobase_396_ixi")
    _write_io_lca_results(results_dir / "gwp100_lcia.csv")
    min_acc_dir = get_acc_output_dir(
        context=_acc_context(
            proj_base=paths.project_base,
            source_label="exiobase_396_ixi",
            cc_type="static",
        ),
        public_result_root_name=public_result_root_name_for_fu_code(fu_code="L2.a.a"),
    )
    status = _status()
    try:
        _write_acc_file(
            min_acc_dir / "Skipped__gwp100_lcia.csv",
            l1_l2_method="Skipped",
            cc_bound="min_cc",
        )
        _write_acc_file(
            min_acc_dir / "UT(FD)__gwp100_lcia__missing_year.csv",
            year_values={1990: 2.0},
            cc_bound="min_cc",
        )
        _write_acc_file(min_acc_dir / "UT(FD)__gwp100_lcia.csv", cc_bound="min_cc")
        _write_acc_file(
            min_acc_dir / "UT(FD)__gwp100_lcia.csv",
            year_values={2005: 4.0},
            cc_bound="max_cc",
        )
        result = static_mod.process_static_asr(
            proj_base=paths.project_base,
            fu_code="L2.a.a",
            cc_source="gwp100_lcia",
            source_label="exiobase_396_ixi",
            base_allocate_args=base_allocate_args,
            years=[2005],
            fmt="csv",
            lca_type=IO_LCA_FAMILY,
            lca_version_name=None,
            static_cc_bounds=["min_cc", "max_cc"],
            acc_output_files=sorted(min_acc_dir.glob("*.csv")),
            allowed_l1_l2_methods={"UT(FD)"},
            lca_rows=lca_rows_mod.load_lca_rows(
                proj_base=paths.project_base,
                source_label="exiobase_396_ixi",
                lca_type=IO_LCA_FAMILY,
                lcia_method="gwp100_lcia",
                lca_version_name=None,
                base_allocate_args=base_allocate_args,
                years=[2005],
            ),
            status=status,
        )
    finally:
        status.finish()

    assert result.n_matched == 1
    assert result.n_written == 1
    assert result.impacts == ["GWP_100"]
    assert len(result.output_dirs) == 1
    assert len(result.output_files) == 1
    assert result.external_lca_transition is None
    assert all(path.exists() for path in result.output_files)
    assert {path.stem for path in result.output_files} == {"UT(FD)__gwp100_lcia"}

    external_args = _historical_external_base_allocate_args(
        project_name="asr_runtime_static_external"
    )
    external_report = prepare_external_inputs(project_name="asr_runtime_static_external")
    _, external_expected_pairs = bundled_cc_expected_impact_units(lcia_method="gwp100_lcia")
    _write_external_lca_file(
        external_lca_deterministic_dir(project_base=external_report.project_root)
        / "supplier_v1__gwp100_lcia.csv",
        impact_unit=external_expected_pairs[0][1],
    )
    external_min_acc_dir = get_acc_output_dir(
        context=_acc_context(
            proj_base=external_report.project_root,
            source_label="exiobase_396_ixi",
            cc_type="static",
        ),
        public_result_root_name=public_result_root_name_for_fu_code(fu_code="L2.a.a"),
    )
    _write_acc_file(
        external_min_acc_dir / "UT(FD)__gwp100_lcia.csv",
        cc_bound="min_cc",
    )
    external_status = _status()
    try:
        external_result = static_mod.process_static_asr(
            proj_base=external_report.project_root,
            fu_code="L2.a.a",
            cc_source="gwp100_lcia",
            source_label="exiobase_396_ixi",
            base_allocate_args=external_args,
            years=[2005],
            fmt="csv",
            lca_type="external",
            lca_version_name="supplier_v1",
            static_cc_bounds=["min_cc"],
            acc_output_files=sorted(external_min_acc_dir.glob("*.csv")),
            allowed_l1_l2_methods={"UT(FD)"},
            lca_rows=lca_rows_mod.load_lca_rows(
                proj_base=external_report.project_root,
                source_label="exiobase_396_ixi",
                lca_type="external",
                lcia_method="gwp100_lcia",
                lca_version_name="supplier_v1",
                base_allocate_args=external_args,
                years=[2005],
            ),
            status=external_status,
        )
    finally:
        external_status.finish()
    assert external_result.n_matched == 1
    assert external_result.n_written == 1
    assert external_result.impacts == ["GWP_100"]
    assert external_result.external_lca_transition == {
        "switch_year": None,
        "marker_label": "external LCA SSP-dependent switch",
        "marker_color": "#375a7f",
    }
    assert external_result.output_files[0].stem == "UT(FD)__gwp100_lcia"
    del allocation_dummy_repo


def test_dynamic_runtime_contracts_cover_missing_acc_skip_paths_and_external_alignment(
    allocation_dummy_repo,
) -> None:
    base_allocate_args = _historical_external_base_allocate_args(
        project_name="asr_runtime_dynamic_external"
    )
    external_report = prepare_external_inputs(project_name="asr_runtime_dynamic_external")
    deterministic_dir = external_lca_deterministic_dir(project_base=external_report.project_root)
    _, external_expected_pairs = bundled_cc_expected_impact_units(lcia_method="gwp100_lcia")
    _write_external_lca_file(
        deterministic_dir / "supplier_v1__gwp100_lcia.csv",
        impact_unit=external_expected_pairs[0][1],
    )
    acc_dir = get_acc_output_dir(
        context=_acc_context(
            proj_base=external_report.project_root,
            source_label="exiobase_396_ixi",
            cc_type="dynamic_ar6",
        ),
        public_result_root_name=public_result_root_name_for_fu_code(fu_code="L2.a.a"),
    )
    status = _status()
    try:
        _write_acc_file(
            acc_dir / "Skipped__gwp100_lcia__dynamic_ar6.csv",
            l1_l2_method="Skipped",
        )
        _write_acc_file(
            acc_dir / "UT(FD)__gwp100_lcia__dynamic_ar6__missing_year.csv",
            year_values={1990: 2.0},
        )
        _write_acc_file(acc_dir / "UT(FD)__gwp100_lcia__dynamic_ar6.csv")
        result = dynamic_mod.process_dynamic_asr(
            proj_base=external_report.project_root,
            fu_code="L2.a.a",
            cc_source="gwp100_lcia",
            source_label="exiobase_396_ixi",
            base_allocate_args=base_allocate_args,
            years=[2005],
            fmt="csv",
            lca_type="external",
            lca_version_name="supplier_v1",
            acc_output_files=sorted(acc_dir.glob("*.csv")),
            allowed_l1_l2_methods={"UT(FD)"},
            share_transition_meta={},
            lca_rows=lca_rows_mod.load_lca_rows(
                proj_base=external_report.project_root,
                source_label="exiobase_396_ixi",
                lca_type="external",
                lcia_method="gwp100_lcia",
                lca_version_name="supplier_v1",
                base_allocate_args=base_allocate_args,
                years=[2005],
            ),
            status=status,
        )
    finally:
        status.finish()

    assert result.n_matched == 1
    assert result.n_written == 1
    assert result.impacts == ["GWP_100"]
    assert len(result.output_dirs) == 1
    assert len(result.output_files) == 1
    assert result.output_files[0].exists()
    assert result.external_lca_transition == {
        "switch_year": None,
        "marker_label": "external LCA SSP-dependent switch",
        "marker_color": "#375a7f",
    }

    pd.DataFrame(
        [
            {
                "r_p": "FR",
                "s_p": "D",
                "impact": "GWP_100",
                "impact_unit": external_expected_pairs[0][1],
                "2030": 1000.0,
            }
        ]
    ).to_csv(deterministic_dir / "supplier_v1__gwp100_lcia__ssp2.csv", index=False)
    _write_acc_file(
        acc_dir / "UT(FD)__gwp100_lcia__dynamic_ar6.csv",
        year_values={2030: 4.0},
    )
    status_with_transitions = _status()
    try:
        transition_result = dynamic_mod.process_dynamic_asr(
            proj_base=external_report.project_root,
            fu_code="L2.a.a",
            cc_source="gwp100_lcia",
            source_label="exiobase_396_ixi",
            base_allocate_args=base_allocate_args,
            years=[2005],
            fmt="csv",
            lca_type="external",
            lca_version_name="supplier_v1",
            acc_output_files=sorted(acc_dir.glob("*.csv")),
            allowed_l1_l2_methods={"UT(FD)"},
            share_transition_meta={"UT(FD)__gwp100_lcia__dynamic_ar6": {"ssp_start_year": 2030}},
            lca_rows=lca_rows_mod.load_lca_rows(
                proj_base=external_report.project_root,
                source_label="exiobase_396_ixi",
                lca_type="external",
                lcia_method="gwp100_lcia",
                lca_version_name="supplier_v1",
                base_allocate_args=base_allocate_args,
                years=[2005],
            ),
            status=status_with_transitions,
        )
    finally:
        status_with_transitions.finish()
    transition_frame = pd.read_csv(transition_result.output_files[-1])
    assert transition_frame.loc[0, "asocc_ssp_start_year"] == 2030
    assert "lca_ssp_start_year" not in transition_frame.columns
    del allocation_dummy_repo


def test_dynamic_runtime_repeats_invariant_rows_for_cumulative_identities(
    allocation_dummy_repo,
) -> None:
    base_allocate_args = _base_allocate_args(project_name="asr_runtime_dynamic_cumulative_scope")
    base_allocate_args["ssp_scenario"] = ["SSP2"]
    external_report = prepare_external_inputs(project_name="asr_runtime_dynamic_cumulative_scope")
    deterministic_dir = external_lca_deterministic_dir(project_base=external_report.project_root)
    _, external_expected_pairs = bundled_cc_expected_impact_units(lcia_method="gwp100_lcia")
    impact_unit = external_expected_pairs[0][1]
    _write_external_lca_file(
        deterministic_dir / "supplier_v1__gwp100_lcia.csv",
        impact_unit=impact_unit,
        year_values={2005: 1000.0},
    )
    _write_external_lca_file(
        deterministic_dir / "supplier_v1__gwp100_lcia__ssp2.csv",
        impact_unit=impact_unit,
        year_values={2030: 1200.0},
    )
    acc_dir = get_acc_output_dir(
        context=_acc_context(
            proj_base=external_report.project_root,
            source_label="exiobase_396_ixi",
            cc_type="dynamic_ar6",
        ),
        public_result_root_name=public_result_root_name_for_fu_code(fu_code="L2.a.a"),
    )
    historical_acc = _write_acc_file(
        acc_dir / "UT(FD)__gwp100_lcia__dynamic_ar6.csv",
        year_values={2005: 2.0},
    )
    prospective_acc = _write_acc_file(
        acc_dir / "UT(FD)__gwp100_lcia__dynamic_ar6__SSP2.csv",
        year_values={2030: 4.0},
        extra_columns={
            ASOCC_SSP_SCENARIO_COLUMN: "SSP2",
            "l2_reuse_year": 2024,
        },
    )
    status = _status()
    try:
        result = dynamic_mod.process_dynamic_asr(
            proj_base=external_report.project_root,
            fu_code="L2.a.a",
            cc_source="gwp100_lcia",
            source_label="exiobase_396_ixi",
            base_allocate_args=base_allocate_args,
            years=[2005, 2030],
            fmt="csv",
            lca_type="external",
            lca_version_name="supplier_v1",
            acc_output_files=[historical_acc, prospective_acc],
            allowed_l1_l2_methods={"UT(FD)"},
            share_transition_meta={},
            lca_rows=lca_rows_mod.load_lca_rows(
                proj_base=external_report.project_root,
                source_label="exiobase_396_ixi",
                lca_type="external",
                lcia_method="gwp100_lcia",
                lca_version_name="supplier_v1",
                base_allocate_args=base_allocate_args,
                years=[2005, 2030],
            ),
            status=status,
        )
    finally:
        status.finish()

    assert result.n_matched == 2
    assert result.n_written == 2
    assert result.impacts == ["GWP_100"]
    output_by_stem = {path.stem: pd.read_csv(path) for path in result.output_files}
    historical_output = output_by_stem["UT(FD)__gwp100_lcia__dynamic_ar6"]
    prospective_output = output_by_stem["UT(FD)__gwp100_lcia__dynamic_ar6__SSP2"]
    assert "cumulative_asr" not in historical_output.columns
    assert prospective_output.loc[0, "cumulative_asr"] == pytest.approx(2.2 / 6.0)
    assert bool(prospective_output.loc[0, "cumulative_no_transgression"])
    del allocation_dummy_repo


def test_static_runtime_contracts_skip_acc_files_without_requested_years(
    allocation_dummy_repo,
) -> None:
    base_allocate_args = _historical_external_base_allocate_args(
        project_name="asr_runtime_static_missing_year_only"
    )
    external_report = prepare_external_inputs(project_name="asr_runtime_static_missing_year_only")
    _, external_expected_pairs = bundled_cc_expected_impact_units(lcia_method="gwp100_lcia")
    _write_external_lca_file(
        external_lca_deterministic_dir(project_base=external_report.project_root)
        / "supplier_v1__gwp100_lcia.csv",
        impact_unit=external_expected_pairs[0][1],
    )
    min_acc_dir = get_acc_output_dir(
        context=_acc_context(
            proj_base=external_report.project_root,
            source_label="exiobase_396_ixi",
            cc_type="static",
        ),
        public_result_root_name=public_result_root_name_for_fu_code(fu_code="L2.a.a"),
    )
    _write_acc_file(
        min_acc_dir / "UT(FD)__gwp100_lcia.csv",
        year_values={1990: 2.0},
        cc_bound="min_cc",
    )
    status = _status()
    try:
        result = static_mod.process_static_asr(
            proj_base=external_report.project_root,
            fu_code="L2.a.a",
            cc_source="gwp100_lcia",
            source_label="exiobase_396_ixi",
            base_allocate_args=base_allocate_args,
            years=[2005],
            fmt="csv",
            lca_type="external",
            lca_version_name="supplier_v1",
            static_cc_bounds=["min_cc"],
            acc_output_files=sorted(min_acc_dir.glob("*.csv")),
            allowed_l1_l2_methods={"UT(FD)"},
            lca_rows=lca_rows_mod.load_lca_rows(
                proj_base=external_report.project_root,
                source_label="exiobase_396_ixi",
                lca_type="external",
                lcia_method="gwp100_lcia",
                lca_version_name="supplier_v1",
                base_allocate_args=base_allocate_args,
                years=[2005],
            ),
            status=status,
        )
    finally:
        status.finish()
    assert result.n_matched == 0
    assert result.n_written == 0
    assert result.impacts == []
    assert len(result.output_dirs) == 1
    assert result.output_files == []
    assert result.external_lca_transition == {
        "switch_year": None,
        "marker_label": "external LCA SSP-dependent switch",
        "marker_color": "#375a7f",
    }


def test_compute_contracts_cover_selector_resolution_matching_and_ratios() -> None:
    acc_df = pd.DataFrame(
        {
            "impact": ["GWP_100"],
            "impact_unit": ["t CO2eq/yr"],
            "r_p": ["FR"],
            "s_p": ["D"],
            "2030": [2.0],
        }
    )
    io_lca_rows = pd.DataFrame(
        {
            "impact": ["GWP_100"],
            "year": ["2030"],
            "lca_value": [1000.0],
            "impact_unit": ["kg CO2eq/yr"],
            "r_p": ["FR"],
            "s_p": ["D"],
        }
    )
    assert compute_mod.required_match_selectors(acc_df) == ["r_p", "s_p"]

    assert compute_mod._resolve_unit_factor(  # noqa: SLF001
        lca_impact_unit="kg CO2eq/yr",
        acc_impact_unit="t CO2eq/yr",
    ) == pytest.approx(0.001)
    with pytest.raises(ValueError):
        compute_mod._resolve_unit_factor(  # noqa: SLF001
            lca_impact_unit="items",
            acc_impact_unit="kg",
        )

    external_rows = pd.DataFrame(
        {
            "impact": ["GWP_100"],
            "year": ["2030"],
            "lca_value": [3.0],
            "impact_unit": ["t CO2eq/yr"],
            "r_p": ["FR"],
            "s_p": ["D"],
            "lca_ssp_scenario": [None],
        }
    )
    invariant_result = compute_mod.deterministic_asr_for_acc_file(
        acc_df=acc_df,
        year_cols=["2030"],
        impact_code="GWP_100",
        lca_rows=external_rows,
        lca_type="external",
    )
    assert invariant_result.loc[0, "2030"] == pytest.approx(1.5)

    scenario_result = compute_mod.deterministic_asr_for_acc_file(
        acc_df=acc_df.assign(ar6_cc_ssp_scenario=["SSP2"]),
        year_cols=["2030"],
        impact_code="GWP_100",
        lca_rows=external_rows.assign(lca_value=[7.0], lca_ssp_scenario=["SSP2"]),
        lca_type="external",
    )
    assert scenario_result.loc[0, "2030"] == pytest.approx(3.5)
    scenario_lca_result = compute_mod.deterministic_asr_for_acc_file(
        acc_df=acc_df.assign(**{"2029": [2.0], "2030": [4.0]}),
        year_cols=["2029", "2030"],
        impact_code="GWP_100",
        lca_rows=pd.DataFrame(
            {
                "impact": ["GWP_100", "GWP_100", "GWP_100"],
                "year": ["2029", "2030", "2030"],
                "lca_value": [3.0, 8.0, 12.0],
                "impact_unit": ["t CO2eq/yr", "t CO2eq/yr", "t CO2eq/yr"],
                "r_p": ["FR", "FR", "FR"],
                "s_p": ["D", "D", "D"],
                "lca_ssp_scenario": [None, "SSP1", "SSP2"],
            }
        ),
        lca_type="external",
    )
    assert scenario_lca_result["lca_ssp_scenario"].tolist() == ["SSP1", "SSP2"]
    assert scenario_lca_result["2029"].tolist() == [pytest.approx(1.5), pytest.approx(1.5)]
    assert scenario_lca_result["2030"].tolist() == [pytest.approx(2.0), pytest.approx(3.0)]
    assert "cumulative_asr" not in scenario_lca_result.columns
    assert "cumulative_no_transgression" not in scenario_lca_result.columns

    with pytest.raises(ValueError):
        compute_mod.deterministic_asr_for_acc_file(
            acc_df=acc_df,
            year_cols=["2030"],
            impact_code="GWP_100",
            lca_rows=external_rows.loc[external_rows["year"].eq("2031")],
            lca_type="external",
        )
    with pytest.raises(ValueError):
        compute_mod.deterministic_asr_for_acc_file(
            acc_df=acc_df,
            year_cols=["2030"],
            impact_code="GWP_100",
            lca_rows=pd.concat([io_lca_rows, io_lca_rows], ignore_index=True),
            lca_type=IO_LCA_FAMILY,
        )
    component_frame = compute_mod.build_deterministic_asr_component_frame(
        acc_df=acc_df,
        year_cols=["2030"],
        impact_code="GWP_100",
        lca_rows=io_lca_rows,
        lca_type=IO_LCA_FAMILY,
    )
    assert component_frame.loc[0, "lca_2030"] == pytest.approx(1.0)
    assert component_frame.loc[0, "acc_2030"] == pytest.approx(2.0)
    assert component_frame.loc[0, "2030"] == pytest.approx(0.5)

    zero_acc = compute_mod.build_deterministic_asr_component_frame(
        acc_df=acc_df.assign(**{"2030": [0.0]}),
        year_cols=["2030"],
        impact_code="GWP_100",
        lca_rows=io_lca_rows,
        lca_type=IO_LCA_FAMILY,
    )
    assert pd.isna(zero_acc.loc[0, "2030"])

    compact = compute_mod.deterministic_asr_for_acc_file(
        acc_df=acc_df,
        year_cols=["2030"],
        impact_code="GWP_100",
        lca_rows=io_lca_rows,
        lca_type=IO_LCA_FAMILY,
    )
    assert "lca_2030" not in compact.columns
    assert "acc_2030" not in compact.columns
    row_owned_transition = compute_mod.deterministic_asr_for_acc_file(
        acc_df=pd.DataFrame(
            {
                "impact": ["GWP_100", "GWP_100"],
                "impact_unit": ["t CO2eq/yr", "t CO2eq/yr"],
                "r_p": ["FR", "US"],
                "s_p": ["D", "X"],
                "asocc_ssp_start_year": [pd.NA, 2026],
                "2030": [2.0, 4.0],
            }
        ),
        year_cols=["2030"],
        impact_code="GWP_100",
        lca_rows=pd.DataFrame(
            {
                "impact": ["GWP_100", "GWP_100"],
                "year": ["2030", "2030"],
                "lca_value": [1000.0, 2000.0],
                "impact_unit": ["kg CO2eq/yr", "kg CO2eq/yr"],
                "r_p": ["FR", "US"],
                "s_p": ["D", "X"],
            }
        ),
        lca_type=IO_LCA_FAMILY,
    )
    assert pd.isna(row_owned_transition.loc[0, "asocc_ssp_start_year"])
    assert row_owned_transition.loc[1, "asocc_ssp_start_year"] == 2026

    finalized = lca_rows_mod._finalize_lca_rows(  # noqa: SLF001
        pd.DataFrame(
            {
                "year": [2030],
                "impact": ["GWP_100"],
                "lca_value": ["4.0"],
                "impact_unit": ["kg"],
            }
        )
    )
    assert finalized.loc[0, "year"] == "2030"
    assert finalized.loc[0, "lca_value"] == pytest.approx(4.0)
    assert (
        first_non_null_scenario_year(
            pd.DataFrame({"value": [1.0]}),
            scenario_column="scenario",
        )
        is None
    )
    assert (
        first_non_null_scenario_year(
            pd.DataFrame({"scenario": [None, "SSP2"], "year": ["2005", "2030"]}),
            scenario_column="scenario",
        )
        == 2030
    )


def test_runtime_contracts_cover_path_and_ssp_edge_contracts(tmp_path: Path) -> None:
    assert shared_paths_mod.get_asr_results_dir(
        context=_asr_context(
            proj_base=tmp_path,
            source_label="oecd_v2025",
            fu_code="L1.a",
            lca_type="external",
            cc_type="static",
        ),
    ) == (
        tmp_path
        / "C_asr"
        / "oecd_v2025"
        / "external_lca__supplier_v1"
        / "deterministic"
        / "static__gwp100_lcia"
        / "results"
    )
    assert (
        common_mod.asocc_ssp_transition_start_year(
            output_stem="demo",
            share_transition_meta={"demo": {"ssp_start_year": pd.NA}},
        )
        is None
    )
    assert (
        common_mod.asocc_ssp_transition_start_year(
            output_stem="demo",
            share_transition_meta={"demo": {"ssp_start_year": "2030"}},
        )
        == 2030
    )
