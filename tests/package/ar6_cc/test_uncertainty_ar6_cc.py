import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from pyaesa import uncertainty_ar6_cc
from pyaesa.ar6_cc.uncertainty.figures.product_renderers import (
    _category_groups,
    _flow_color,
    _format_year_axis,
    plot_trajectory_band_scope,
)
from pyaesa.ar6_cc.uncertainty.figures.render import (
    _active_category_jobs,
    _inactive_category_jobs,
)
from pyaesa.ar6_cc.uncertainty.figures.row_reader import (
    FigureTables,
    _budget_identity_columns,
    _prepare_summary,
    _summary_identity_columns,
    categories_by_common_scope,
    common_pair_counts,
    summary_rows_global,
)
from pyaesa.ar6_cc.uncertainty.figures.scope_planner import (
    build_figure_context,
    category_scope_stem,
    common_scope_stem,
    _years_from_args,
)
from pyaesa.ar6_cc.uncertainty.io.artifacts import (
    ar6_cc_run_layout_from_manifest,
    ar6_cc_run_paths_from_manifest,
)
from pyaesa.download.ar6.utils.config import GROSS_ALT_KYOTO_WO_AFOLU, SEQUESTRATION_SUBTOTAL
from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NEGATIVE,
    CC_FLOW_NET,
    CC_FLOW_POSITIVE,
)
from pyaesa.ar6_cc.deterministic.io.tables import write_cc_output
from pyaesa.ar6_cc.deterministic.io.paths import (
    get_cc_metadata_path,
    get_cc_output_path,
    get_cc_post_study_output_path,
    get_cc_scope_dir,
)
from pyaesa.ar6_cc.deterministic.runtime.metadata import build_run_metadata_payload
from pyaesa.ar6_cc.deterministic.runtime.reports import AR6CCPathwayCount
from pyaesa.ar6_cc.shared.runtime.signatures import build_cc_scope_signature
from pyaesa.ar6_cc.uncertainty.runtime.models import (
    AR6CCCategoryPool,
    AR6CCSamplingGroup,
    AR6CCUncertaintyPlan,
    AR6CCUncertaintyRunPaths,
)
from pyaesa.ar6_cc.uncertainty.evaluation.sampling import (
    deterministic_ar6_cc_identity_and_values,
)
from pyaesa.ar6_cc.uncertainty.io.paths import ar6_cc_monte_carlo_root
from pyaesa.ar6_cc.uncertainty.io.run_outputs import write_ar6_cc_study_post_outputs
from pyaesa.ar6_cc.uncertainty.request.normalization import (
    normalize_ar6_cc_uncertainty_request,
)
from pyaesa.ar6_cc.uncertainty.runner import (
    run_uncertainty_ar6_cc,
    run_uncertainty_ar6_cc_component,
)
from pyaesa.ar6_cc.uncertainty.sobol.evaluator import (
    build_ar6_cc_sobol_evaluation_context,
    evaluate_ar6_cc_sobol_units,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import (
    component_inventory_payload,
)
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import monte_carlo_run_progress
from pyaesa.shared.uncertainty_assessment.request.core import normalize_uncertainty_request
from pyaesa.shared.uncertainty_assessment.run_state.manifest import build_manifest
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix,
    iter_sparse_run_rows,
)
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table
from pyaesa.process.ar6.utils.io.contracts import processed_workbook_name
from pyaesa.process.ar6.utils.io.metadata import build_process_metadata_payload
from pyaesa.process.ar6.utils.io.paths import get_logs_dir, get_processed_dir
from pyaesa.process.ar6.utils.pipeline.runtime_helpers import process_signature


def _seed_pathway_counts(
    *,
    rows: pd.DataFrame,
    categories: list[str],
    ssps: list[str],
) -> tuple[list[AR6CCPathwayCount], list[AR6CCPathwayCount]]:
    pairs = rows.loc[
        :, ["cc_category", "ssp_scenario", "cc_model", "cc_scenario"]
    ].drop_duplicates()
    grouped = pairs.groupby(["cc_category", "ssp_scenario"], sort=True).size()
    retained: list[AR6CCPathwayCount] = []
    missing: list[AR6CCPathwayCount] = []
    for category in categories:
        for ssp in ssps:
            count = int(grouped.get((category, ssp), 0))
            item = AR6CCPathwayCount(
                category=category,
                ssp_scenario=ssp,
                model_scenario_pairs=count,
            )
            if count:
                retained.append(item)
            else:
                missing.append(item)
    return retained, missing


def _base_args(
    *,
    years: int | list[int] | range = range(2019, 2022),
    category: list[str] | None = None,
    ssp_scenario: list[str] | None = None,
    subset_version: str | None = None,
    all_selectors: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"years": years}
    if not all_selectors:
        payload["category"] = ["C1"] if category is None else category
        payload["ssp_scenario"] = ["SSP1"] if ssp_scenario is None else ssp_scenario
    if subset_version is not None:
        payload["subset_version"] = subset_version
    return payload


def _manifest_run_root(manifest) -> Path:
    return Path(manifest.artifacts["scope_manifest"]).parents[1]


def _read_compact_run_matrix_frame(*, path: Path, column_count: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_index, matrix in iter_compact_run_matrix(
        path=path,
        output_format="csv_compact",
        column_count=column_count,
    ):
        frame = pd.DataFrame(matrix, columns=[str(index) for index in range(column_count)])
        frame.insert(0, "run_index", run_index)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["run_index", *[str(index) for index in range(column_count)]])
    return pd.concat(frames, ignore_index=True)


def _read_sparse_run_rows_frame(*, path: Path, value_column: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for rows in iter_sparse_run_rows(path=path, output_format="csv_compact"):
        frames.append(
            pd.DataFrame(
                {
                    "run_index": rows.run_index,
                    "public_row_id": rows.public_row_id,
                    value_column: rows.values,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["run_index", "public_row_id", value_column])
    return pd.concat(frames, ignore_index=True)


def test_uncertainty_ar6_cc_figure_context_scalar_year() -> None:
    assert _years_from_args({"years": 2019}) == [2019]


def test_uncertainty_ar6_cc_public_fixed_csv_reuse_and_artifacts(ar6_dummy_repo) -> None:
    del ar6_dummy_repo
    deterministic_scope_dir = _write_seed_deterministic_scope(
        years=range(2019, 2022),
        categories=["C1"],
        ssps=["SSP1"],
        multi_candidate=False,
    )

    base_args = _base_args(years=[2019, 2020, 2021])
    uncertainty_config = {
        "mc_parameters": {
            "fixed": {"active": True, "n_runs": 4},
            "convergence": {"active": False},
        },
        "dynamic_ar6_cc_uncertainty": {"sampling_method": "srs"},
    }
    no_figure_manifest = uncertainty_ar6_cc(
        base_ar6_cc_args=base_args,
        uncertainty_config=uncertainty_config,
        output_format="csv_compact",
        figures=False,
        refresh=False,
    ).manifest
    assert no_figure_manifest.artifacts is not None
    assert "figure_paths" not in no_figure_manifest.artifacts

    manifest = uncertainty_ar6_cc(
        base_ar6_cc_args=base_args,
        uncertainty_config=uncertainty_config,
        output_format="csv_compact",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    ).manifest

    assert manifest.family == "ar6_cc"
    assert manifest.completed_runs == 4
    assert manifest.active_sources == ("dynamic_ar6_cc_uncertainty",)
    assert manifest.compatibility_context == {
        "active_sources": ["dynamic_ar6_cc_uncertainty"],
        "artifact_contract": "ar6_cc_sparse_selected_trajectory_flow_runs_with_post_study_v1",
        "run_role": {"role": "public"},
    }
    assert manifest.artifacts is not None
    assert manifest.artifacts["public_output"] is not None
    assert "figure_paths" in manifest.artifacts
    assert all(Path(path).exists() for path in manifest.artifacts["figure_paths"])
    deterministic_metadata = json.loads(
        get_cc_metadata_path(cc_dir=deterministic_scope_dir).read_text(encoding="utf-8")
    )
    assert all(Path(path).exists() for path in deterministic_metadata["artifacts"]["figure_paths"])
    identity_path = Path(manifest.artifacts["public_row_identity"])
    runs_path = Path(manifest.artifacts["cc_runs"])
    summary_path = Path(manifest.artifacts["summary_stats_runs"])
    post_identity_path = Path(manifest.artifacts["post_study_period_public_row_identity"])
    post_runs_path = Path(manifest.artifacts["post_study_period_cc_runs"])
    post_summary_path = Path(manifest.artifacts["post_study_period_summary_stats_runs"])
    budget_identity_path = Path(
        manifest.artifacts["study_and_post_study_period_budget_row_identity"]
    )
    budget_runs_path = Path(manifest.artifacts["study_and_post_study_period_budget_runs"])
    budget_summary_path = Path(
        manifest.artifacts["study_and_post_study_period_budget_summary_stats"]
    )
    readme_path = Path(manifest.artifacts["results_readme"])
    source_methods_path = Path(manifest.artifacts["source_methods"])
    assert identity_path.exists()
    assert runs_path.exists()
    assert summary_path.exists()
    assert post_identity_path.exists()
    assert post_runs_path.exists()
    assert post_summary_path.exists()
    assert budget_identity_path.exists()
    assert budget_runs_path.exists()
    assert budget_summary_path.exists()
    assert readme_path.exists()
    assert source_methods_path.exists()
    assert not list(identity_path.parents[1].glob(".summary_values_*.dat"))

    identity = pd.read_csv(identity_path)
    runs = _read_sparse_run_rows_frame(path=runs_path, value_column="cc")
    summary = pd.read_csv(summary_path)
    post_identity = pd.read_csv(post_identity_path)
    post_runs = _read_sparse_run_rows_frame(path=post_runs_path, value_column="cc")
    budget_identity = pd.read_csv(budget_identity_path)
    budget_runs = _read_compact_run_matrix_frame(
        path=budget_runs_path,
        column_count=len(budget_identity),
    )
    source_methods = pd.read_csv(source_methods_path)
    assert identity.to_dict(orient="list") == {
        "public_row_id": list(range(6)),
        "cc_category": ["C1"] * 6,
        "ssp_scenario": ["SSP1"] * 6,
        "cc_flow": [CC_FLOW_POSITIVE] * 3 + [CC_FLOW_NEGATIVE] * 3,
        "cc_variable": [GROSS_ALT_KYOTO_WO_AFOLU] * 3 + [SEQUESTRATION_SUBTOTAL] * 3,
        "impact_unit": ["MtCO2eq/yr"] * 6,
        "cc_model": ["M_shared"] * 6,
        "cc_scenario": ["S0_0"] * 6,
        "year": [2019, 2020, 2021] * 2,
    }
    assert list(runs.columns) == ["run_index", "public_row_id", "cc"]
    assert runs["run_index"].drop_duplicates().tolist() == [0, 1, 2, 3]
    assert len(runs) == 4 * 2 * 3
    assert post_identity["year"].min() == 2022
    assert post_identity["year"].max() == 2100
    assert post_runs["run_index"].drop_duplicates().tolist() == [0, 1, 2, 3]
    assert set(budget_identity["period_segment"]) == {"study_period", "post_study_period"}
    assert list(budget_runs.columns) == [
        "run_index",
        *[str(index) for index in range(len(budget_identity))],
    ]
    assert {"cc_model", "cc_scenario", "public_row_id"}.isdisjoint(summary.columns)
    assert len(summary) == 2 * 3
    assert source_methods["cc_model"].drop_duplicates().tolist() == ["M_shared"]
    assert set(source_methods["trajectory_probability"]) == {1.0}
    readme_text = readme_path.read_text(encoding="utf-8")
    assert readme_text
    assert all(len(line) <= 100 for line in readme_text.splitlines())

    reused = uncertainty_ar6_cc(
        base_ar6_cc_args=base_args,
        uncertainty_config=uncertainty_config,
        output_format="csv_compact",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    ).manifest

    assert reused.run_id == manifest.run_id
    assert reused.artifacts is not None
    assert "figure_paths" in reused.artifacts
    stale_run_file = Path(manifest.artifacts["scope_manifest"]).parents[1] / "stale.txt"
    stale_run_file.write_text("stale", encoding="utf-8")
    stale_upstream_file = (
        Path(manifest.deterministic_prerequisites[0]["output_file"]).parents[1] / "stale.txt"
    )
    stale_upstream_file.write_text("stale", encoding="utf-8")
    refreshed = uncertainty_ar6_cc(
        base_ar6_cc_args=base_args,
        uncertainty_config=uncertainty_config,
        output_format="csv_compact",
        figures=False,
        refresh=True,
    ).manifest
    assert refreshed.status == "complete"
    assert not stale_run_file.exists()
    assert not stale_upstream_file.exists()


def test_uncertainty_ar6_cc_selector_scopes_and_source_settings_are_isolated(
    project_repo: Path,
) -> None:
    del project_repo
    years = range(2019, 2021)
    fixed_config = {
        "mc_parameters": {
            "fixed": {"active": True, "n_runs": 1},
            "convergence": {"active": False},
        },
        "dynamic_ar6_cc_uncertainty": {"sampling_method": "srs"},
    }
    _write_seed_deterministic_scope(
        years=years,
        categories=["C1"],
        ssps=["SSP1"],
        multi_candidate=False,
    )
    _write_seed_deterministic_scope(
        years=years,
        categories=["C1"],
        ssps=["SSP2"],
        multi_candidate=False,
    )

    ssp1 = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(years=years, category=["C1"], ssp_scenario=["SSP1"]),
        uncertainty_config=fixed_config,
        output_format="csv_compact",
        figures=False,
        refresh=False,
    ).manifest
    ssp2 = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(years=years, category=["C1"], ssp_scenario=["SSP2"]),
        uncertainty_config=fixed_config,
        output_format="csv_compact",
        figures=False,
        refresh=False,
    ).manifest

    ssp1_root = _manifest_run_root(ssp1)
    ssp2_root = _manifest_run_root(ssp2)
    assert ssp1_root.parent.parent.name == "C1__SSP1"
    assert ssp2_root.parent.parent.name == "C1__SSP2"
    assert ssp1_root != ssp2_root

    categories = ["C1", "C2"]
    ssps = ["SSP1", "SSP2"]
    _write_seed_deterministic_scope(years=years, categories=categories, ssps=ssps)
    selector_args = _base_args(years=years, category=categories, ssp_scenario=ssps)
    category_config = {
        "mc_parameters": {
            "fixed": {"active": True, "n_runs": 1},
            "convergence": {"active": False},
        },
        "dynamic_ar6_cc_uncertainty": {
            "sampling_method": "srs",
            "category_uncertainty": True,
        },
    }
    without_category_uncertainty = uncertainty_ar6_cc(
        base_ar6_cc_args=selector_args,
        uncertainty_config=fixed_config,
        output_format="csv_compact",
        figures=False,
        refresh=False,
    ).manifest
    with_category_uncertainty = uncertainty_ar6_cc(
        base_ar6_cc_args=selector_args,
        uncertainty_config=category_config,
        output_format="csv_compact",
        figures=False,
        refresh=False,
    ).manifest
    reused_without_category_uncertainty = uncertainty_ar6_cc(
        base_ar6_cc_args=selector_args,
        uncertainty_config=fixed_config,
        output_format="csv_compact",
        figures=False,
        refresh=False,
    ).manifest

    false_root = _manifest_run_root(without_category_uncertainty)
    true_root = _manifest_run_root(with_category_uncertainty)
    assert false_root.parent.parent.name == "C1-C2__SSP1-SSP2"
    assert true_root.parent.parent == false_root.parent.parent
    assert with_category_uncertainty.run_id != without_category_uncertainty.run_id
    assert reused_without_category_uncertainty.run_id == without_category_uncertainty.run_id


def test_uncertainty_ar6_cc_accepts_extended_category_domain(project_repo: Path) -> None:
    del project_repo
    years = range(2019, 2021)
    fixed_config = {
        "mc_parameters": {
            "fixed": {"active": True, "n_runs": 1},
            "convergence": {"active": False},
        },
        "dynamic_ar6_cc_uncertainty": {"sampling_method": "srs"},
    }
    _write_seed_deterministic_scope(
        years=years,
        categories=["C8"],
        ssps=["SSP5"],
        multi_candidate=False,
    )

    manifest = uncertainty_ar6_cc(
        base_ar6_cc_args={
            "years": years,
            "category": "c8",
            "ssp_scenario": "ssp5",
        },
        uncertainty_config=fixed_config,
        output_format="csv_compact",
        figures=False,
        refresh=False,
    ).manifest
    request = normalize_ar6_cc_uncertainty_request(
        base_ar6_cc_args={
            "years": years,
            "category": "c8",
            "ssp_scenario": "ssp5",
        },
        source_parameters={"sampling_method": "srs"},
    )

    assert manifest.status == "complete"
    assert request.category == ["C8"]
    assert request.ssp_scenario == ["SSP5"]


def test_uncertainty_ar6_cc_public_parquet_lhs_category_uncertainty(
    project_repo: Path,
) -> None:
    del project_repo
    years = range(2019, 2021)
    categories = ["C1", "C2"]
    ssps = ["SSP1", "SSP2"]
    _write_seed_deterministic_scope(
        years=years,
        categories=categories,
        ssps=ssps,
        subset_version="seeded",
    )

    manifest = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(
            years=years,
            category=categories,
            ssp_scenario=ssps,
            subset_version="seeded",
        ),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {
                "sampling_method": "lhs",
                "category_uncertainty": True,
            },
        },
        output_format="parquet",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    ).manifest

    assert manifest.completed_runs == 1
    assert manifest.convergence is None
    assert manifest.artifacts is not None
    assert "figure_paths" in manifest.artifacts
    assert all(Path(path).exists() for path in manifest.artifacts["figure_paths"])
    runs = read_uncertainty_table(
        path=Path(manifest.artifacts["cc_runs"]),
        output_format="parquet",
    )
    identity = read_uncertainty_table(
        path=Path(manifest.artifacts["public_row_identity"]),
        output_format="parquet",
    )
    source_methods = pd.read_csv(manifest.artifacts["source_methods"])
    summary = read_uncertainty_table(
        path=Path(manifest.artifacts["summary_stats_runs"]),
        output_format="parquet",
    )
    assert len(identity) > len(categories) * len(ssps) * len(years)
    assert "cc_category" in identity.columns
    assert "cc_category" not in summary.columns
    assert {"cc_model", "cc_scenario", "public_row_id"}.isdisjoint(summary.columns)
    assert len(summary) == len(ssps) * len(years) * 2
    assert list(runs.columns) == ["run_index", "public_row_id", "cc"]
    assert not any(runs["cc"].isna().tolist())
    assert source_methods["category_probability"].drop_duplicates().tolist() == [0.5]
    assert set(source_methods["sampling_method"]) == {"lhs"}
    assert bool(source_methods["category_uncertainty"].all())


def test_uncertainty_ar6_cc_ending_in_2100_has_no_post_study_artifacts(
    project_repo: Path,
) -> None:
    del project_repo
    years = range(2099, 2101)
    _write_seed_deterministic_scope(
        years=years,
        categories=["C1"],
        ssps=["SSP1"],
        multi_candidate=False,
    )

    manifest = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(years=years),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {"sampling_method": "srs"},
        },
        output_format="csv_compact",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    ).manifest

    assert manifest.artifacts is not None
    assert "post_study_period_public_row_identity" not in manifest.artifacts
    assert "post_study_period_cc_runs" not in manifest.artifacts
    assert "post_study_period_summary_stats_runs" not in manifest.artifacts
    assert "figure_paths" in manifest.artifacts


def test_uncertainty_ar6_cc_all_selectors_and_lhs_model_stratified(
    project_repo: Path,
) -> None:
    del project_repo
    years = range(2019, 2021)
    _write_seed_deterministic_scope(
        years=years,
        categories=["C1", "C2", "C3", "C4"],
        ssps=["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"],
        use_all_signature=True,
    )

    manifest = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(years=years, all_selectors=True),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {"sampling_method": "lhs"},
        },
        output_format="csv_compact",
        figures=False,
    ).manifest

    assert manifest.artifacts is not None
    readme = Path(manifest.artifacts["results_readme"]).read_text(encoding="utf-8")
    source_methods = pd.read_csv(manifest.artifacts["source_methods"])
    assert all(category in readme for category in ("C1", "C2", "C3", "C4"))
    assert all(ssp in readme for ssp in ("SSP1", "SSP2", "SSP3", "SSP4", "SSP5"))
    assert set(source_methods["trajectory_probability"]) == {0.25, 0.5, 1.0}


def test_uncertainty_ar6_cc_component_inventory_appends_in_place(
    project_repo: Path,
) -> None:
    del project_repo
    _write_seed_deterministic_scope(
        years=range(2019, 2021),
        categories=["C1"],
        ssps=["SSP1"],
        multi_candidate=False,
    )
    base_args = _base_args(years=range(2019, 2021))
    parent = run_uncertainty_ar6_cc(
        base_ar6_cc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {"active": True},
        },
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        component_inventory=component_inventory_payload(
            composite_family="acc",
            component_name="dynamic_cc",
            target_runs=1,
        ),
    ).manifest
    child = run_uncertainty_ar6_cc(
        base_ar6_cc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {"active": True},
        },
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        component_inventory=component_inventory_payload(
            composite_family="acc",
            component_name="dynamic_cc",
            target_runs=2,
        ),
    ).manifest

    assert child.run_id == parent.run_id
    assert child.lineage is not None
    assert "source_inventory" in child.lineage
    assert child.completed_runs == 2
    reused_with_parent_phase = run_uncertainty_ar6_cc(
        base_ar6_cc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {"active": True},
        },
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        component_inventory=component_inventory_payload(
            composite_family="acc",
            component_name="dynamic_cc",
            target_runs=2,
            parent_mode="convergence",
            parent_max_runs=2,
        ),
        phase=NullPhasePrinter(),
    ).manifest
    assert reused_with_parent_phase.run_id == child.run_id

    _write_seed_deterministic_scope(
        years=range(2099, 2101),
        categories=["C1"],
        ssps=["SSP1"],
        multi_candidate=False,
    )
    no_post_args = _base_args(years=range(2099, 2101))
    no_post_parent = run_uncertainty_ar6_cc(
        base_ar6_cc_args=no_post_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {"active": True},
        },
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        component_inventory=component_inventory_payload(
            composite_family="acc",
            component_name="dynamic_cc",
            target_runs=1,
        ),
    ).manifest
    no_post_child = run_uncertainty_ar6_cc(
        base_ar6_cc_args=no_post_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {"active": True},
        },
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        component_inventory=component_inventory_payload(
            composite_family="acc",
            component_name="dynamic_cc",
            target_runs=2,
        ),
    ).manifest
    assert no_post_child.run_id == no_post_parent.run_id
    assert "post_study_period_cc_runs" not in no_post_child.artifacts


def test_uncertainty_ar6_cc_component_session_figures_render_deterministic(
    project_repo: Path,
) -> None:
    del project_repo
    scope_dir = _write_seed_deterministic_scope(
        years=range(2019, 2021),
        categories=["C1"],
        ssps=["SSP1"],
        multi_candidate=False,
    )
    base_args = _base_args(years=range(2019, 2021))
    config = {
        "mc_parameters": {
            "fixed": {"active": True, "n_runs": 1},
            "convergence": {"active": False},
        },
        "dynamic_ar6_cc_uncertainty": {"active": True},
    }
    inventory = component_inventory_payload(
        composite_family="acc",
        component_name="dynamic_cc",
        target_runs=1,
        parent_mode="convergence",
        parent_max_runs=1,
    )
    first = run_uncertainty_ar6_cc_component(
        base_ar6_cc_args=base_args,
        uncertainty_config=config,
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        component_inventory=inventory,
        run_id=None,
        show_progress=False,
        phase=NullPhasePrinter(),
        progress=None,
        finalize_component_inventory=False,
    )
    finalized = run_uncertainty_ar6_cc_component(
        base_ar6_cc_args=base_args,
        uncertainty_config=config,
        output_format="csv_compact",
        figures=True,
        figure_options=None,
        figure_format={"format": "svg", "dpi": 1},
        refresh=False,
        component_inventory=inventory,
        run_id=None,
        show_progress=False,
        phase=NullPhasePrinter(),
        progress=None,
        component_session=first.session,
        finalize_component_inventory=True,
    )
    reused = run_uncertainty_ar6_cc_component(
        base_ar6_cc_args=base_args,
        uncertainty_config=config,
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        component_inventory=inventory,
        run_id=None,
        show_progress=False,
        phase=NullPhasePrinter(),
        progress=None,
    )

    metadata = json.loads(get_cc_metadata_path(cc_dir=scope_dir).read_text(encoding="utf-8"))
    assert all(Path(path).exists() for path in metadata["artifacts"]["figure_paths"])
    assert finalized.report.reuse_status == "computed"
    assert reused.report.reuse_status == "reused_exact"


def test_uncertainty_ar6_cc_convergence_batches_can_cross_checkpoints(
    tmp_path: Path,
) -> None:
    years = [2000, 2001]
    identity = pd.DataFrame(
        [
            {
                "public_row_id": public_row_id,
                "cc_category": "C1",
                "ssp_scenario": "SSP1",
                "cc_flow": CC_FLOW_POSITIVE,
                "cc_variable": GROSS_ALT_KYOTO_WO_AFOLU,
                "impact_unit": "kg CO2-eq",
                "cc_model": model,
                "cc_scenario": scenario,
                "year": year,
            }
            for public_row_id, (model, scenario, year) in enumerate(
                [
                    ("M1", "S1", 2000),
                    ("M1", "S1", 2001),
                    ("M2", "S2", 2000),
                    ("M2", "S2", 2001),
                ]
            )
        ]
    )
    plan = AR6CCUncertaintyPlan(
        identity=identity,
        group_identity=identity,
        trajectory_values=np.array([[10.0, 11.0], [10.0, 11.0]], dtype=np.float64),
        groups=(
            AR6CCSamplingGroup(
                category="C1",
                ssp_scenario="SSP1",
                flow_count=1,
                candidate_positions=np.array([0, 1], dtype=np.int64),
                model_candidate_positions=(
                    np.array([0], dtype=np.int64),
                    np.array([1], dtype=np.int64),
                ),
                output_start=0,
                output_stop=2,
            ),
        ),
        category_pools=(AR6CCCategoryPool(ssp_scenario="SSP1", group_indices=(0,)),),
        source_method_rows=pd.DataFrame(),
        source_parameters={"sampling_method": "srs", "category_uncertainty": False},
        availability_messages=(),
    )
    run_root = tmp_path / "mc_ar6_crossing"
    paths = AR6CCUncertaintyRunPaths(
        run_root=run_root,
        public_row_identity=run_root / "results" / "public_row_identity.csv",
        public_runs=run_root / "results" / "cc_runs.csv",
        summary_stats_runs=run_root / "results" / "summary_stats_runs.csv",
        post_study_public_row_identity=(
            run_root / "results" / "post_study_period_public_row_identity.csv"
        ),
        post_study_public_runs=run_root / "results" / "post_study_period_cc_runs.csv",
        post_study_summary_stats_runs=(
            run_root / "results" / "post_study_period_summary_stats_runs.csv"
        ),
        budget_row_identity=(
            run_root / "results" / "study_and_post_study_period_budget_row_identity.csv"
        ),
        budget_runs=run_root / "results" / "study_and_post_study_period_budget_runs.csv",
        budget_summary_stats_runs=(
            run_root / "results" / "study_and_post_study_period_budget_summary_stats.csv"
        ),
        results_readme=run_root / "results" / "README.txt",
        source_methods=run_root / "logs" / "source_methods.csv",
        scope_manifest=run_root / "logs" / "scope_manifest.json",
    )
    runtime = replace(
        normalize_uncertainty_request(
            family="ar6_cc",
            output_format="csv_compact",
            mc_parameters={
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 3, "stable_runs": 2, "rtol": 0.0},
            },
        ),
        batch_size=1,
    )

    completed, convergence, has_post, states = write_ar6_cc_study_post_outputs(
        paths=paths,
        plan=plan,
        study_years=years,
        post_study_years=[],
        runtime=runtime,
        progress=monte_carlo_run_progress(source="uncertainty_ar6_cc", enabled=False),
    )

    assert completed == 3
    assert convergence is not None
    assert convergence["reached"] is True
    assert has_post is False
    assert states is None


def test_uncertainty_ar6_cc_convergence_reports_unreached(
    project_repo: Path,
) -> None:
    del project_repo
    _write_seed_deterministic_scope(
        years=range(2019, 2022),
        categories=["C1"],
        ssps=["SSP1"],
    )

    manifest = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "rtol": 0.01, "stable_runs": 2},
            },
            "dynamic_ar6_cc_uncertainty": {"active": True},
        },
        output_format="csv_compact",
        refresh=False,
    ).manifest

    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is False
    assert manifest.convergence["completed_runs"] == 2

    reused = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "rtol": 0.01, "stable_runs": 2},
            },
            "dynamic_ar6_cc_uncertainty": {"active": True},
        },
        output_format="csv_compact",
        refresh=False,
    ).manifest

    assert reused.run_id == manifest.run_id


def test_uncertainty_ar6_cc_convergence_can_reach_for_constant_source(
    project_repo: Path,
) -> None:
    del project_repo
    years = range(2019, 2021)
    _write_seed_deterministic_scope(
        years=years,
        categories=["C1"],
        ssps=["SSP1"],
        multi_candidate=False,
    )

    manifest = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(years=years),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 4, "stable_runs": 2},
            },
            "dynamic_ar6_cc_uncertainty": {"active": True},
        },
    ).manifest

    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is True
    assert manifest.convergence["completed_runs"] == 4


def test_uncertainty_ar6_cc_validates_public_configuration(project_repo: Path) -> None:
    _write_seed_deterministic_scope(
        years=range(2019, 2022),
        categories=["C1"],
        ssps=["SSP1"],
    )

    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args={**_base_args(), "output_format": "csv"},
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                }
            },
        )
    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args={"category": ["C1"]},
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                }
            },
        )
    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args=_base_args(years=[2019, 2021]),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                }
            },
        )
    with pytest.raises(ValueError, match="C1, C2, C3, C4, C5, C6, C7, C8"):
        uncertainty_ar6_cc(
            base_ar6_cc_args=_base_args(category=["C9"]),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                }
            },
        )
    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args=_base_args(),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "dynamic_ar6_cc_uncertainty": {"active": False},
            },
        )
    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args=_base_args(),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "dynamic_ar6_cc_uncertainty": {"sampling_method": "bad"},
            },
        )
    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args=_base_args(),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "dynamic_ar6_cc_uncertainty": {"category_uncertainty": "yes"},
            },
        )
    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args={**_base_args(), "subset_version": " "},
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                }
            },
        )
    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args=_base_args(),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "dynamic_ar6_cc_uncertainty": {"extra": True},
            },
        )
    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args=_base_args(),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "dynamic_ar6_cc_uncertainty": {
                    "category_uncertainty": True,
                },
            },
            refresh=False,
        )
    del project_repo


def test_uncertainty_ar6_cc_rejects_invalid_deterministic_scope(project_repo: Path) -> None:
    del project_repo
    years = range(2019, 2021)
    categories = ["C1", "C2"]
    ssps = ["SSP1", "SSP2"]
    scope_dir = _write_seed_deterministic_scope(
        years=years,
        categories=categories,
        ssps=ssps,
    )
    output_path = get_cc_output_path(cc_dir=scope_dir, output_format="csv")
    rows = pd.read_csv(output_path).drop(columns=["cc_model"])
    rows.to_csv(output_path, index=False)
    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args=_base_args(
                years=years,
                category=categories,
                ssp_scenario=ssps,
            ),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "dynamic_ar6_cc_uncertainty": {"active": True},
            },
        )


def test_uncertainty_ar6_cc_rejects_category_source_without_common_categories(
    project_repo: Path,
) -> None:
    del project_repo
    _write_seed_deterministic_scope(
        years=range(2019, 2022),
        categories=["C1", "C2"],
        ssps=["SSP1", "SSP2"],
        omit_pairs={
            ("C2", "SSP1"),
            ("C1", "SSP2"),
            ("C2", "SSP2"),
        },
    )

    with pytest.raises(ValueError):
        uncertainty_ar6_cc(
            base_ar6_cc_args=_base_args(
                category=["C1", "C2"],
                ssp_scenario=["SSP1", "SSP2"],
            ),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "dynamic_ar6_cc_uncertainty": {
                    "category_uncertainty": True,
                },
            },
            refresh=False,
        )


def test_uncertainty_ar6_cc_category_uncertainty_uses_ssp_local_categories(
    project_repo: Path,
) -> None:
    del project_repo
    years = range(2019, 2021)
    categories = ["C1", "C2", "C3"]
    ssps = ["SSP1", "SSP2"]
    _write_seed_deterministic_scope(
        years=years,
        categories=categories,
        ssps=ssps,
        omit_pairs={("C3", "SSP2")},
    )

    manifest = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(
            years=years,
            category=categories,
            ssp_scenario=ssps,
        ),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {
                "category_uncertainty": True,
            },
        },
    ).manifest

    assert manifest.lineage is not None
    source_inventory = manifest.lineage["source_inventory"]
    assert len(source_inventory["scope_availability_messages"]) == 1
    assert source_inventory["category_pools"] == [
        {
            "ssp_scenario": "SSP1",
            "candidate_categories": ["C1", "C2", "C3"],
        },
        {
            "ssp_scenario": "SSP2",
            "candidate_categories": ["C1", "C2"],
        },
    ]
    assert manifest.artifacts is not None
    source_methods = pd.read_csv(manifest.artifacts["source_methods"])
    summary = pd.read_csv(manifest.artifacts["summary_stats_runs"])
    readme = Path(manifest.artifacts["results_readme"]).read_text(encoding="utf-8")
    assert readme
    assert "cc_category" not in summary.columns
    assert set(summary["ssp_scenario"]) == {"SSP1", "SSP2"}
    probabilities = source_methods.groupby("ssp_scenario")["category_probability"].unique()
    assert probabilities["SSP1"].tolist() == pytest.approx([1 / 3])
    assert probabilities["SSP2"].tolist() == pytest.approx([1 / 2])


def test_uncertainty_ar6_cc_reports_fully_unavailable_requested_selectors(
    project_repo: Path,
) -> None:
    del project_repo
    years = range(2019, 2021)
    categories = ["C1", "C3"]
    ssps = ["SSP1", "SSP5"]
    _write_seed_deterministic_scope(
        years=years,
        categories=categories,
        ssps=ssps,
        omit_pairs={
            ("C3", "SSP1"),
            ("C3", "SSP5"),
            ("C1", "SSP5"),
        },
    )

    manifest = uncertainty_ar6_cc(
        base_ar6_cc_args=_base_args(
            years=years,
            category=categories,
            ssp_scenario=ssps,
        ),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "dynamic_ar6_cc_uncertainty": {
                "category_uncertainty": False,
            },
        },
    ).manifest

    assert manifest.lineage is not None
    assert len(manifest.lineage["source_inventory"]["scope_availability_messages"]) == 3


def test_ar6_cc_sobol_source_unit_evaluator_contract() -> None:
    identity = pd.DataFrame(
        {
            "public_row_id": [0, 1, 2, 3],
            "cc_category": ["C1", "C1", "C2", "C2"],
            "ssp_scenario": ["SSP1", "SSP1", "SSP1", "SSP1"],
            "cc_flow": [CC_FLOW_POSITIVE] * 4,
            "cc_variable": [GROSS_ALT_KYOTO_WO_AFOLU] * 4,
            "impact_unit": ["MtCO2eq/yr"] * 4,
            "year": [2019, 2020, 2019, 2020],
        }
    )
    groups = (
        AR6CCSamplingGroup(
            category="C1",
            ssp_scenario="SSP1",
            flow_count=1,
            candidate_positions=np.array([0, 1], dtype=np.int64),
            model_candidate_positions=(
                np.array([0], dtype=np.int64),
                np.array([1], dtype=np.int64),
            ),
            output_start=0,
            output_stop=2,
        ),
        AR6CCSamplingGroup(
            category="C2",
            ssp_scenario="SSP1",
            flow_count=1,
            candidate_positions=np.array([2, 3], dtype=np.int64),
            model_candidate_positions=(
                np.array([2], dtype=np.int64),
                np.array([3], dtype=np.int64),
            ),
            output_start=2,
            output_stop=4,
        ),
    )
    values = np.array(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
            [40.0, 41.0],
        ],
        dtype=np.float64,
    )
    plan = AR6CCUncertaintyPlan(
        identity=identity,
        group_identity=identity,
        trajectory_values=values,
        groups=groups,
        category_pools=(
            AR6CCCategoryPool(
                ssp_scenario="SSP1",
                group_indices=(0, 1),
            ),
        ),
        source_method_rows=pd.DataFrame(),
        source_parameters={"sampling_method": "srs", "category_uncertainty": False},
        availability_messages=(),
    )
    context = build_ar6_cc_sobol_evaluation_context(plan=plan)

    returned_identity, evaluated = evaluate_ar6_cc_sobol_units(
        context=context,
        units=np.array([[0.0], [0.75]], dtype=np.float64),
    )

    assert returned_identity.equals(identity)
    assert evaluated.tolist() == [
        [10.0, 11.0, 30.0, 31.0],
        [20.0, 21.0, 40.0, 41.0],
    ]

    lhs_plan = AR6CCUncertaintyPlan(
        identity=identity,
        group_identity=identity,
        trajectory_values=values,
        groups=groups,
        category_pools=plan.category_pools,
        source_method_rows=pd.DataFrame(),
        source_parameters={"sampling_method": "lhs", "category_uncertainty": True},
        availability_messages=(),
    )
    lhs_context = build_ar6_cc_sobol_evaluation_context(plan=lhs_plan)
    _, category_evaluated = evaluate_ar6_cc_sobol_units(
        context=lhs_context,
        units=np.array([[0.10], [0.60]], dtype=np.float64),
    )
    _, sparse_category_evaluated = evaluate_ar6_cc_sobol_units(
        context=lhs_context,
        units=np.array([[0.10]], dtype=np.float64),
    )

    assert list(category_evaluated.shape) == [2, 2]
    assert category_evaluated[0].tolist() == [10.0, 11.0]
    assert category_evaluated[1].tolist() == [30.0, 31.0]
    assert sparse_category_evaluated[0].tolist() == [10.0, 11.0]


def test_uncertainty_ar6_cc_deterministic_identity_excludes_sequestration() -> None:
    request = normalize_ar6_cc_uncertainty_request(
        base_ar6_cc_args=_base_args(years=range(2019, 2021)),
        source_parameters={"sampling_method": "srs"},
    )
    deterministic_rows = pd.DataFrame(
        [
            {
                "cc_model": "M1",
                "cc_scenario": "S1",
                "cc_category": "C1",
                "ssp_scenario": "SSP1",
                "cc_flow": CC_FLOW_POSITIVE,
                "cc_variable": GROSS_ALT_KYOTO_WO_AFOLU,
                "impact_unit": "MtCO2eq/yr",
                2019: 10.0,
                2020: 11.0,
            },
            {
                "cc_model": "M1",
                "cc_scenario": "S1",
                "cc_category": "C1",
                "ssp_scenario": "SSP1",
                "cc_flow": CC_FLOW_NEGATIVE,
                "cc_variable": SEQUESTRATION_SUBTOTAL,
                "impact_unit": "MtCO2eq/yr",
                2019: -1.0,
                2020: -1.1,
            },
        ]
    )

    identity, values = deterministic_ar6_cc_identity_and_values(
        request=request,
        deterministic_rows=deterministic_rows,
    )

    assert identity.to_dict(orient="list") == {
        "public_row_id": [0, 1],
        "cc_model": ["M1", "M1"],
        "cc_scenario": ["S1", "S1"],
        "cc_category": ["C1", "C1"],
        "ssp_scenario": ["SSP1", "SSP1"],
        "impact_unit": ["MtCO2eq/yr", "MtCO2eq/yr"],
        "year": [2019, 2020],
    }
    assert values.tolist() == [10.0, 11.0]


def test_uncertainty_ar6_cc_figure_helpers_cover_flow_scopes(tmp_path: Path) -> None:
    run_root = tmp_path / "mc_figure"
    paths = AR6CCUncertaintyRunPaths(
        run_root=run_root,
        public_row_identity=run_root / "results" / "public_row_identity.csv",
        public_runs=run_root / "results" / "cc_runs.csv",
        summary_stats_runs=run_root / "results" / "summary_stats_runs.csv",
        post_study_public_row_identity=(
            run_root / "results" / "post_study_period_public_row_identity.csv"
        ),
        post_study_public_runs=run_root / "results" / "post_study_period_cc_runs.csv",
        post_study_summary_stats_runs=(
            run_root / "results" / "post_study_period_summary_stats_runs.csv"
        ),
        budget_row_identity=(
            run_root / "results" / "study_and_post_study_period_budget_row_identity.csv"
        ),
        budget_runs=run_root / "results" / "study_and_post_study_period_budget_runs.csv",
        budget_summary_stats_runs=(
            run_root / "results" / "study_and_post_study_period_budget_summary_stats.csv"
        ),
        results_readme=run_root / "results" / "README.txt",
        source_methods=run_root / "logs" / "source_methods.csv",
        scope_manifest=run_root / "logs" / "scope_manifest.json",
    )
    manifest = build_manifest(
        family="ar6_cc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("dynamic_ar6_cc_uncertainty",),
        status="complete",
        completed_runs=2,
        source_parameters={
            "dynamic_ar6_cc_uncertainty": {
                "sampling_method": "srs",
                "category_uncertainty": True,
            }
        },
        arguments={
            "base_ar6_cc_args": {
                "years": [2019, 2020],
                "ssp_scenario": ["SSP1"],
            }
        },
        deterministic_prerequisites=(
            {
                "variable": GROSS_ALT_KYOTO_WO_AFOLU,
                "categories": ["C1", "C2"],
            },
        ),
        artifacts={
            "scope_manifest": str(paths.scope_manifest),
            "public_row_identity": str(paths.public_row_identity),
            "cc_runs": str(paths.public_runs),
            "summary_stats_runs": str(paths.summary_stats_runs),
            "results_readme": str(paths.results_readme),
            "source_methods": str(paths.source_methods),
            "public_output": {"cc_runs": {"layout": "sparse"}},
        },
    )
    context = build_figure_context(
        manifest=manifest,
        paths=paths,
        figure_options=None,
        figure_format={"format": "png", "dpi": 10},
    )
    assert context.category_uncertainty is True
    assert _summary_identity_columns(context=context) == (
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "year",
    )
    assert _budget_identity_columns(context=context) == (
        "budget_row_id",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "period_segment",
    )
    assert common_scope_stem(ssp_scenario="SSP1") == "ssp1"
    assert (
        category_scope_stem(
            ssp_scenario="SSP1",
            category="C1",
        )
        == "ssp1__cat_C1"
    )
    assert ar6_cc_run_paths_from_manifest(manifest=manifest).run_root == paths.run_root
    assert ar6_cc_run_layout_from_manifest(manifest=manifest) == "sparse"

    summary = pd.DataFrame(
        [
            row
            for row_id, (category, flow, variable, year, value) in enumerate(
                [
                    ("C1", CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, 2019, 10.0),
                    ("C1", CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, 2020, 11.0),
                    ("C1", CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, 2019, -2.0),
                    ("C1", CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, 2020, -2.2),
                    ("C2", CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, 2019, 12.0),
                    ("C2", CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, 2020, 13.0),
                    ("C2", CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, 2019, -2.4),
                    ("C2", CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, 2020, -2.6),
                ]
            )
            for row in [
                {
                    "public_row_id": row_id,
                    "cc_category": category,
                    "ssp_scenario": "SSP1",
                    "cc_flow": flow,
                    "cc_variable": variable,
                    "impact_unit": "MtCO2eq/yr",
                    "year": year,
                    "mean": value,
                    "std": 0.1,
                    "min": value - 0.2,
                    "p5": value - 0.1,
                    "p25": value - 0.05,
                    "median": value,
                    "p75": value + 0.05,
                    "p95": value + 0.1,
                    "max": value + 0.2,
                }
            ]
        ]
    )
    source_methods = pd.DataFrame(
        [
            row
            for category, flow, variable, model, scenario in [
                ("C1", CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, "M1", "S1"),
                ("C1", CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, "M1", "S1"),
                ("C2", CC_FLOW_POSITIVE, GROSS_ALT_KYOTO_WO_AFOLU, "M2", "S2"),
                ("C2", CC_FLOW_NEGATIVE, SEQUESTRATION_SUBTOTAL, "M2", "S2"),
            ]
            for row in [
                {
                    "cc_category": category,
                    "ssp_scenario": "SSP1",
                    "cc_flow": flow,
                    "cc_variable": variable,
                    "impact_unit": "MtCO2eq/yr",
                    "cc_model": model,
                    "cc_scenario": scenario,
                }
            ]
        ]
    )
    budget_rows = pd.DataFrame(
        [
            {
                "budget_row_id": row_id,
                "cc_category": category,
                "ssp_scenario": "SSP1",
                "cc_flow": flow,
                "cc_variable": variable,
                "impact_unit": "MtCO2eq/yr",
                "period_segment": period_segment,
                "__budget_values": np.asarray(values, dtype=np.float64),
            }
            for row_id, (category, flow, variable, period_segment, values) in enumerate(
                [
                    (
                        "C1",
                        CC_FLOW_POSITIVE,
                        GROSS_ALT_KYOTO_WO_AFOLU,
                        "study_period",
                        [21.0, 22.0],
                    ),
                    (
                        "C1",
                        CC_FLOW_NEGATIVE,
                        SEQUESTRATION_SUBTOTAL,
                        "study_period",
                        [-4.2, -4.4],
                    ),
                    (
                        "C2",
                        CC_FLOW_POSITIVE,
                        GROSS_ALT_KYOTO_WO_AFOLU,
                        "study_period",
                        [25.0, 26.0],
                    ),
                    (
                        "C2",
                        CC_FLOW_NEGATIVE,
                        SEQUESTRATION_SUBTOTAL,
                        "study_period",
                        [-5.0, -5.2],
                    ),
                ]
            )
        ]
    )
    tables = FigureTables(
        summary=_prepare_summary(summary),
        post_study_summary=None,
        budget_rows=budget_rows,
        source_methods=source_methods,
    )
    assert "cc_category" not in _prepare_summary(summary.drop(columns=["cc_category"])).columns
    assert summary_rows_global(tables=tables)["year"].tolist() == [
        2019,
        2019,
        2019,
        2019,
        2020,
        2020,
        2020,
        2020,
    ]
    assert common_pair_counts(tables=tables) == {"SSP1": 2}
    assert categories_by_common_scope(tables=tables) == {"SSP1": ["C1", "C2"]}
    assert len(list(_active_category_jobs(context=context, tables=tables))) == 1

    inactive_manifest = build_manifest(
        family="ar6_cc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("dynamic_ar6_cc_uncertainty",),
        status="complete",
        completed_runs=2,
        source_parameters={
            "dynamic_ar6_cc_uncertainty": {
                "sampling_method": "srs",
                "category_uncertainty": False,
            }
        },
        arguments={"base_ar6_cc_args": {"years": [2019, 2020]}},
        deterministic_prerequisites=(
            {
                "variable": GROSS_ALT_KYOTO_WO_AFOLU,
                "categories": ["C1", "C2"],
            },
        ),
    )
    inactive_context = build_figure_context(
        manifest=inactive_manifest,
        paths=paths,
        figure_options=None,
        figure_format={"format": "png", "dpi": 10},
    )
    assert inactive_context.requested_ssps == ()
    assert _summary_identity_columns(context=inactive_context) == (
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "year",
    )
    assert _budget_identity_columns(context=inactive_context) == (
        "budget_row_id",
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        "period_segment",
    )
    assert len(list(_inactive_category_jobs(context=inactive_context, tables=tables))) == 2
    assert _category_groups(summary.drop(columns=["cc_category"]))[0][0] == "AR6 CC"
    assert _flow_color(CC_FLOW_NEGATIVE) == "#E68613"
    figure, axis = plt.subplots()
    _format_year_axis(axis, np.asarray([], dtype=np.int64))
    plt.close(figure)

    plotted = plot_trajectory_band_scope(
        frame=tables.summary.loc[
            tables.summary["impact_unit"].eq("MtCO2eq/yr")
            | tables.summary["impact_unit"].eq("MtCO2/yr")
        ],
        budget_frame=budget_rows,
        output_stem=tmp_path / "figures" / "active_scope",
        title_categories=["C1", "C2"],
        variable_name=GROSS_ALT_KYOTO_WO_AFOLU,
        ssp_scenario="SSP1",
        pair_count=2,
        sampling_method="lhs",
        study_years=[2019, 2020],
        dpi=10,
        output_format="png",
    )
    assert plotted[0].exists()

    net_rows = tables.summary.loc[tables.summary["cc_flow"].eq(CC_FLOW_POSITIVE)].copy()
    net_rows.loc[:, "cc_flow"] = CC_FLOW_NET
    plotted_net = plot_trajectory_band_scope(
        frame=net_rows,
        budget_frame=budget_rows.loc[budget_rows["cc_flow"].eq(CC_FLOW_POSITIVE)],
        output_stem=tmp_path / "figures" / "net_categories",
        title_categories=["C1", "C2"],
        variable_name=GROSS_ALT_KYOTO_WO_AFOLU,
        ssp_scenario="SSP1",
        pair_count=2,
        sampling_method="srs",
        study_years=[2019, 2020],
        dpi=10,
        output_format="png",
    )
    assert plotted_net[0].exists()

    plotted_single_category_net = plot_trajectory_band_scope(
        frame=net_rows.loc[net_rows["cc_category"].eq("C1")],
        budget_frame=budget_rows.loc[
            budget_rows["cc_category"].eq("C1") & budget_rows["cc_flow"].eq(CC_FLOW_POSITIVE)
        ],
        output_stem=tmp_path / "figures" / "net_single_category",
        title_categories=["C1"],
        variable_name=GROSS_ALT_KYOTO_WO_AFOLU,
        ssp_scenario="SSP1",
        pair_count=1,
        sampling_method="srs",
        study_years=[2019, 2020],
        dpi=10,
        output_format="png",
    )
    assert plotted_single_category_net[0].exists()

    plotted_common = plot_trajectory_band_scope(
        frame=tables.summary.drop(columns=["cc_category"]),
        budget_frame=budget_rows.drop(columns=["cc_category"]),
        output_stem=tmp_path / "figures" / "common_scope",
        title_categories=["C1", "C2"],
        variable_name=GROSS_ALT_KYOTO_WO_AFOLU,
        ssp_scenario="SSP1",
        pair_count=2,
        sampling_method="lhs",
        study_years=[2019, 2020],
        dpi=10,
        output_format="png",
    )
    assert plotted_common[0].exists()


def _write_seed_deterministic_scope(
    *,
    years: range,
    categories: list[str],
    ssps: list[str],
    subset_version: str | None = None,
    use_all_signature: bool = False,
    multi_candidate: bool = True,
    omit_pairs: set[tuple[str, str]] | None = None,
) -> Path:
    study_period = [min(years), max(years)]
    scope_dir = get_cc_scope_dir(
        study_period,
        harmonization=True,
        harmonization_method="offset",
        emission_type="kyoto_gases",
        include_afolu=False,
        emissions_mode="gross_alt",
        subset_version=subset_version,
        category=categories,
        ssp_scenario=ssps,
    )
    output_path = get_cc_output_path(cc_dir=scope_dir, output_format="csv")
    post_output_path = (
        None
        if max(years) == 2100
        else get_cc_post_study_output_path(cc_dir=scope_dir, output_format="csv")
    )
    metadata_path = get_cc_metadata_path(cc_dir=scope_dir)
    signature = build_cc_scope_signature(
        study_period=study_period,
        harmonization=True,
        harmonization_method="offset",
        emission_type="kyoto_gases",
        include_afolu=False,
        emissions_mode="gross_alt",
        category=None if use_all_signature else categories,
        ssp_scenario=None if use_all_signature else ssps,
        subset_version=subset_version,
    )
    study_rows = []
    post_rows = []
    model_index = 0
    trajectory_count = 0
    for category in categories:
        for ssp in ssps:
            if omit_pairs and (category, ssp) in omit_pairs:
                continue
            scenario_count = 3 if multi_candidate and category == "C1" and ssp == "SSP1" else 1
            for scenario_number in range(scenario_count):
                model = "M_shared" if scenario_number < 2 else f"M{model_index}"
                base_row: dict[object, object] = {
                    "cc_model": model,
                    "cc_scenario": f"S{model_index}_{scenario_number}",
                    "cc_category": category,
                    "ssp_scenario": ssp,
                    "impact_unit": "MtCO2eq/yr",
                }
                positive_row: dict[object, object] = {
                    **base_row,
                    "cc_flow": CC_FLOW_POSITIVE,
                    "cc_variable": GROSS_ALT_KYOTO_WO_AFOLU,
                }
                negative_row: dict[object, object] = {
                    **base_row,
                    "cc_flow": CC_FLOW_NEGATIVE,
                    "cc_variable": SEQUESTRATION_SUBTOTAL,
                }
                post_positive_row = positive_row.copy()
                post_negative_row = negative_row.copy()
                for year in years:
                    value = 10.0 + model_index + scenario_number + (int(year) - 2019)
                    positive_row[int(year)] = value
                    negative_row[int(year)] = -0.25 * value
                if post_output_path is not None:
                    for year in range(max(years) + 1, 2101):
                        value = 10.0 + model_index + scenario_number + (int(year) - 2019)
                        post_positive_row[int(year)] = value
                        post_negative_row[int(year)] = -0.25 * value
                    post_rows.extend([post_positive_row, post_negative_row])
                study_rows.extend([positive_row, negative_row])
                trajectory_count += 1
            model_index += 1
    study_frame = pd.DataFrame(study_rows)
    pathway_counts, missing_pathway_combinations = _seed_pathway_counts(
        rows=study_frame,
        categories=categories,
        ssps=ssps,
    )
    write_cc_output(study_frame, output_path, "csv")
    if post_output_path is not None:
        write_cc_output(pd.DataFrame(post_rows), post_output_path, "csv")
    process_ar6_payload = _write_seed_process_ar6_metadata(
        study_period=study_period,
        categories=categories,
        ssps=ssps,
        trajectory_count=trajectory_count,
    )
    identity = dict(signature)
    metadata_path.write_text(
        json.dumps(
            build_run_metadata_payload(
                signature=signature,
                identity_payload=identity,
                coverage={"cc_category": categories, "ssp_scenario": ssps},
                write_scope_identity=signature,
                emission_type="kyoto_gases",
                include_afolu=False,
                emissions_mode="gross_alt",
                cc_categories=categories,
                ssp_scenarios=ssps,
                total_model_scenario_pairs=trajectory_count,
                pathway_counts=pathway_counts,
                missing_pathway_combinations=missing_pathway_combinations,
                output_file=output_path,
                process_ar6=process_ar6_payload,
                post_study_output_file=post_output_path,
            )
        ),
        encoding="utf-8",
    )
    assert ar6_cc_monte_carlo_root(deterministic_manifest_path=metadata_path).name == (
        "monte_carlo"
    )
    return scope_dir


def _write_seed_process_ar6_metadata(
    *,
    study_period: list[int],
    categories: list[str],
    ssps: list[str],
    trajectory_count: int,
) -> dict[str, object]:
    processed_dir = get_processed_dir(
        study_period,
        harmonization=True,
        harmonization_method="offset",
        category=categories,
    )
    logs_dir = get_logs_dir(
        study_period,
        harmonization=True,
        harmonization_method="offset",
        category=categories,
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_file = processed_dir / processed_workbook_name(harmonization=True)
    process_meta = build_process_metadata_payload(
        signature=process_signature(study_period, True, "offset", categories),
        categories=categories,
        ssps=[int(str(value).removeprefix("SSP")) for value in ssps],
        harmonization=True,
        harmonization_method="offset",
        latest_historical_year=None,
        requested_harmonization_year=None,
        harmonization_year=None,
        harmonization_message=None,
        processed_dir=processed_dir,
        logs_dir=logs_dir,
        figures_dir=processed_dir / "figures",
        output_file=output_file,
        log_file=None,
        dropped_rows_csv_file=logs_dir / "dropped_model_scenario_variable_rows.csv",
        variable_coverage_summary_counts={
            GROSS_ALT_KYOTO_WO_AFOLU: {
                "available_model_scenario_pairs": trajectory_count,
                "retained_model_scenario_pairs": trajectory_count,
                "missing_reason_counts": {},
            },
            SEQUESTRATION_SUBTOTAL: {
                "available_model_scenario_pairs": trajectory_count,
                "retained_model_scenario_pairs": trajectory_count,
                "missing_reason_counts": {},
            },
        },
    )
    (logs_dir / "scope_manifest.json").write_text(
        json.dumps(process_meta),
        encoding="utf-8",
    )
    variable_coverage = [
        {
            "variable": variable,
            "available_model_scenario_pairs": trajectory_count,
            "retained_model_scenario_pairs": trajectory_count,
        }
        for variable in (GROSS_ALT_KYOTO_WO_AFOLU, SEQUESTRATION_SUBTOTAL)
    ]
    return {
        "reuse_status": "reused_exact",
        "study_period": f"{study_period[0]}-{study_period[1]}",
        "categories": categories,
        "ssps": [int(str(value).removeprefix("SSP")) for value in ssps],
        "harmonization": True,
        "harmonization_method": "offset",
        "output_root": str(processed_dir),
        "output_files_available": 1,
        "figures_available": 0,
        "variable_coverage": variable_coverage,
    }
