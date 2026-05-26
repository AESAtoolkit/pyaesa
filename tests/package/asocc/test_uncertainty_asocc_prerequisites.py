from pathlib import Path
import hashlib

import pandas as pd
import pytest

from pyaesa import deterministic_asocc, uncertainty_asocc
from pyaesa.asocc.io.metadata import _build_run_metadata, _save_run_metadata
from pyaesa.asocc.runtime.scope.branch_resolution import allocate_run_metadata_path
from pyaesa.asocc.runtime.paths.external import get_asocc_external_method_level_dir
from pyaesa.asocc.runtime.request.scope import build_asocc_scope
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.asocc.uncertainty.engine.reuse.prerequisites import (
    prepare_asocc_deterministic_prerequisite,
)
from pyaesa.asocc.uncertainty.engine.runner import run_uncertainty_asocc
from pyaesa.asocc.uncertainty.io.paths import build_asocc_uncertainty_run_paths
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.table_io import write_table
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix,
    iter_sparse_run_rows,
)
from pyaesa.shared.uncertainty_assessment.io.tables import uncertainty_table_columns
from pyaesa.shared.uncertainty_assessment.sobol.plan import normalize_sobol_plan
from pyaesa.shared.uncertainty_assessment.sobol.readme_text import build_sobol_readme_lines


def _base_asocc_args(*, project_name: str) -> dict:
    return {
        "project_name": project_name,
        "source": "oecd_v2025",
        "years": [2005],
        "fu_code": "L2.a.a",
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
    }


def _reference_reuse_base_asocc_args(*, project_name: str) -> dict:
    return {
        **_base_asocc_args(project_name=project_name),
        "reference_years": [2004],
        "projection_mode": "historical_reuse",
        "l2_reuse_years": [2003],
    }


def _uncertainty_config() -> dict:
    return {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 1}, "convergence": {"active": False}},
        **_inactive_default_sources(),
        "reference_year_uncertainty": {},
    }


def _inactive_default_sources() -> dict:
    return {
        "projection_uncertainty": {"active": False},
        "reference_year_uncertainty": {"active": False},
        "inter_method_uncertainty": {"active": False},
    }


def _fixed_no_source_config() -> dict:
    return {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 1}, "convergence": {"active": False}},
        **_inactive_default_sources(),
    }


def test_uncertainty_asocc_rejects_base_asocc_figure_controls() -> None:
    with pytest.raises(ValueError):
        prepare_asocc_deterministic_prerequisite(
            base_asocc_args={
                **_base_asocc_args(project_name="base_figure_controls"),
                "figure_format": {"format": "svg"},
            },
            refresh=False,
        )


def test_uncertainty_asocc_parent_phase_reuses_direct_and_inter_mrio_runs(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    no_source_args = _base_asocc_args(project_name="asocc_uncertainty_parent_phase_no_source")
    no_source = uncertainty_asocc(
        base_asocc_args=no_source_args,
        uncertainty_config=_fixed_no_source_config(),
        sobol_parameters={"active": False},
        output_format="csv_compact",
        figures=False,
        refresh=True,
    ).manifest
    no_source_reused = run_uncertainty_asocc(
        base_asocc_args=no_source_args,
        uncertainty_config=_fixed_no_source_config(),
        sobol_parameters={"active": False},
        external_method=None,
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        phase=NullPhasePrinter(),
    ).manifest
    assert no_source_reused.run_id == no_source.run_id

    inter_mrio_args = {
        **_base_asocc_args(project_name="asocc_uncertainty_parent_phase_inter_mrio"),
        "source": "oecd_v2025",
    }
    inter_mrio_config = {
        **_fixed_no_source_config(),
        "inter_mrio_uncertainty": {"source": "exiobase_396_ixi"},
    }
    inter_mrio = uncertainty_asocc(
        base_asocc_args=inter_mrio_args,
        uncertainty_config=inter_mrio_config,
        sobol_parameters={"active": False},
        output_format="csv_compact",
        figures=False,
        refresh=True,
    ).manifest
    inter_mrio_reused = run_uncertainty_asocc(
        base_asocc_args=inter_mrio_args,
        uncertainty_config=inter_mrio_config,
        sobol_parameters={"active": False},
        external_method=None,
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        phase=NullPhasePrinter(),
    ).manifest

    source_methods = pd.read_csv(inter_mrio.artifacts["source_methods"])
    assert inter_mrio_reused.run_id == inter_mrio.run_id
    assert "inter_mrio_uncertainty" in set(source_methods["source_name"])


def _deterministic_output_paths(repo_root: Path, *, project_name: str) -> list[Path]:
    return sorted((repo_root / f"{project_name}" / "B1_asocc" / "oecd_v2025").rglob("UT(FD).csv"))


def _uncertainty_run_root(
    repo_root: Path,
    *,
    project_name: str,
    run_id: str,
    source: str = "oecd_v2025",
) -> Path:
    return repo_root / f"{project_name}" / "B1_asocc" / source / "monte_carlo" / run_id


def _read_asocc_runs(run_root: Path) -> pd.DataFrame:
    identity = _read_result_table(run_root=run_root, stem="public_row_identity")
    matrix = _read_result_table(run_root=run_root, stem="asocc_runs")
    runs = matrix.melt(id_vars="run_index", var_name="public_row_id", value_name="asocc")
    runs["public_row_id"] = runs["public_row_id"].astype(int)
    return runs.merge(identity, on="public_row_id", how="left").sort_values(
        ["run_index", "public_row_id"],
        ignore_index=True,
    )


def _read_result_table(*, run_root: Path, stem: str) -> pd.DataFrame:
    csv_path = run_root / "results" / f"{stem}.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    if csv_path.is_dir():
        return _read_run_artifact(path=csv_path, output_format="csv_compact")
    parquet_path = run_root / "results" / f"{stem}.parquet"
    if parquet_path.is_dir():
        return _read_run_artifact(path=parquet_path, output_format="parquet")
    return pd.read_parquet(parquet_path)


def _read_run_artifact(*, path: Path, output_format: str) -> pd.DataFrame:
    columns = uncertainty_table_columns(path=path, output_format=output_format)
    if "public_row_id" in columns:
        value_column = next(
            column for column in columns if column not in {"run_index", "public_row_id"}
        )
        pieces = [
            pd.DataFrame(
                {
                    "run_index": rows.run_index,
                    "public_row_id": rows.public_row_id,
                    value_column: rows.values,
                }
            )
            for rows in iter_sparse_run_rows(path=path, output_format=output_format)
        ]
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    column_count = len(columns) - 1
    pieces = []
    for run_index, values in iter_compact_run_matrix(
        path=path,
        output_format=output_format,
        column_count=column_count,
    ):
        frame = pd.DataFrame(values, columns=[str(index) for index in range(column_count)])
        frame.insert(0, "run_index", run_index)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)


def _write_external_deterministic_rows(
    *,
    repo_root: Path,
    project_name: str,
    value: float = 0.42,
) -> Path:
    external_dir = get_asocc_external_method_level_dir(
        proj_base=repo_root / f"{project_name}",
        storage_mode="deterministic",
        level="level_2",
    )
    external_dir.mkdir(parents=True, exist_ok=True)
    path = external_dir / "UT(TD).csv"
    pd.DataFrame(
        {
            "r_p": ["FR"],
            "s_p": ["D"],
            "2005": [value],
        }
    ).to_csv(path, index=False)
    return path


def _write_external_monte_carlo_rows(*, repo_root: Path, project_name: str) -> None:
    external_dir = get_asocc_external_method_level_dir(
        proj_base=repo_root / f"{project_name}",
        storage_mode="monte_carlo",
        level="level_2",
    )
    external_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "run_index": [0, 1],
            "year": [2005, 2005],
            "r_p": ["FR", "FR"],
            "s_p": ["D", "D"],
            ASOCC_SSP_SCENARIO_COLUMN: [None, None],
            "value": [0.72, 0.73],
        }
    ).to_csv(external_dir / "UT(TD).csv", index=False)


def _write_external_compact_monte_carlo_rows(*, repo_root: Path, project_name: str) -> None:
    external_dir = get_asocc_external_method_level_dir(
        proj_base=repo_root / f"{project_name}",
        storage_mode="monte_carlo",
        level="level_2",
    )
    compact_dir = external_dir / "UT(TD)"
    compact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "public_row_id": [0],
            "year": [2005],
            ASOCC_SSP_SCENARIO_COLUMN: [None],
            "r_p": ["FR"],
            "s_p": ["D"],
        }
    ).to_csv(compact_dir / "public_row_identity.csv", index=False)
    pd.DataFrame({"run_index": [0, 1], "0": [0.72, 0.73]}).to_csv(
        compact_dir / "asocc_runs.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "run_index": [0, 1],
            "year": [2005, 2005],
            "r_p": ["FR", "FR"],
            "s_p": ["D", "D"],
            ASOCC_SSP_SCENARIO_COLUMN: [None, None],
            "value": [0.12, 0.13],
        }
    ).to_csv(external_dir / "UT(TD).csv", index=False)


def _write_short_external_monte_carlo_rows(*, repo_root: Path, project_name: str) -> None:
    external_dir = get_asocc_external_method_level_dir(
        proj_base=repo_root / f"{project_name}",
        storage_mode="monte_carlo",
        level="level_2",
    )
    external_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "run_index": [0],
            "year": [2005],
            "r_p": ["FR"],
            "s_p": ["D"],
            ASOCC_SSP_SCENARIO_COLUMN: [None],
            "value": [0.72],
        }
    ).to_csv(external_dir / "UT(TD).csv", index=False)


def _write_prerequisite_scope_metadata(
    *,
    base_asocc_args: dict,
    output_path: Path,
    signature_overrides: dict,
    ssp_scenarios: list[str] | None = None,
) -> None:
    asocc_scope = build_asocc_scope(
        base_allocate_args=normalize_base_allocate_args(base_asocc_args)
    )
    metadata_path = allocate_run_metadata_path(scope=asocc_scope.resolve_path_scope())
    signature = asocc_scope.compute_signature(
        years=[2005],
        output_format="csv",
        intermediate_outputs=True,
        historical_year_cap=None,
    )
    signature.update(signature_overrides)
    scope_payload = _build_run_metadata(
        requested_years=[2005],
        resolved_years=[2005],
        selected_methods=dict(signature.get("selected_methods", {})),
        fu_code=str(signature["fu_code"]),
        studied_indices_tag=str(signature.get("studied_indices_tag", "")),
        skipped_years={},
        outputs=[str(output_path)],
        signature=signature,
    )
    scope_payload["execution"]["completed_years"] = [2005]
    scope_payload["provenance"]["ssp_scenarios"] = ssp_scenarios
    _save_run_metadata(metadata_path, scope_payload)


def _write_disaggregated_scope_manifest(
    *,
    base_asocc_args: dict,
    signature_args: dict | None = None,
    completed_years: list[int] | None = None,
) -> None:
    scope_args = {**base_asocc_args, **(signature_args or {})}
    asocc_scope = build_asocc_scope(base_allocate_args=normalize_base_allocate_args(scope_args))
    years = completed_years or list(base_asocc_args["years"])
    signature = asocc_scope.compute_signature(
        years=years,
        output_format="csv",
        intermediate_outputs=False,
        historical_year_cap=None,
    )
    manifest_path = allocate_run_metadata_path(scope=asocc_scope.resolve_path_scope())
    scope_payload = _build_run_metadata(
        requested_years=years,
        resolved_years=years,
        selected_methods=dict(signature.get("selected_methods", {})),
        fu_code=str(signature["fu_code"]),
        studied_indices_tag=str(signature.get("studied_indices_tag", "")),
        skipped_years={},
        outputs=[f"{scope_args['source']}.csv"],
        signature=signature,
    )
    scope_payload["execution"]["completed_years"] = years
    _save_run_metadata(manifest_path, scope_payload)


def test_uncertainty_asocc_refresh_materializes_deterministic_prerequisite(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_prerequisite"
    base_args = {**_base_asocc_args(project_name=project_name), "years": [2006]}
    deterministic_asocc(
        **base_args,
        figures=False,
        refresh=True,
    )
    deterministic_outputs = _deterministic_output_paths(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
    )
    assert len(deterministic_outputs) == 1
    deterministic_outputs[0].unlink()

    refreshed = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config=_uncertainty_config(),
        sobol_parameters={"active": False},
        output_format="csv_compact",
        figures=False,
        refresh=True,
    ).manifest

    assert refreshed.status == "complete"
    assert refreshed.completed_runs == 1
    assert deterministic_outputs[0].exists()

    reused = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config=_uncertainty_config(),
        sobol_parameters={"active": False},
        output_format="csv_compact",
        figures=False,
        refresh=False,
    ).manifest

    assert reused.status == "complete"
    assert reused.completed_runs == 1


def test_uncertainty_asocc_materializes_missing_deterministic_prerequisite(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_prerequisite_missing"

    manifest = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config=_uncertainty_config(),
        output_format="csv_compact",
        refresh=False,
    ).manifest

    assert manifest.status == "complete"
    assert _deterministic_output_paths(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
    )


def test_prerequisite_accepts_persisted_row_axis_coverage(tmp_path: Path) -> None:
    base_args = _reference_reuse_base_asocc_args(project_name="coverage_match")
    output_path = tmp_path / "UT(FD).csv"
    write_table(
        path=output_path,
        frame=pd.DataFrame(
            {
                "l1_l2_method": ["UT(FD)", "UT(FD)"],
                "reference_year": [2004, 2002],
                "l2_reuse_year": [2003, 2001],
                "2005": [0.5, 0.6],
            }
        ),
    )
    _write_prerequisite_scope_metadata(
        base_asocc_args=base_args,
        output_path=output_path,
        signature_overrides={
            "intermediate_outputs": False,
            "reference_years_input": None,
            "l2_reuse_years": None,
        },
    )

    prerequisite = prepare_asocc_deterministic_prerequisite(
        base_asocc_args=base_args,
        refresh=False,
    )

    assert [(scope.outputs, years) for scope, years in prerequisite.persisted_scope_matches] == [
        ([str(output_path)], [2005])
    ]


def test_uncertainty_asocc_recomputes_persisted_scope_without_requested_row_axes(
    allocation_dummy_repo,
    tmp_path: Path,
) -> None:
    project_name = "coverage_miss_public"
    base_args = {
        **_reference_reuse_base_asocc_args(project_name=project_name),
        "reference_years": [2005],
        "l2_reuse_years": [2005],
    }
    output_path = tmp_path / "UT(FD).csv"
    write_table(
        path=output_path,
        frame=pd.DataFrame(
            {
                "l1_l2_method": ["UT(FD)"],
                "reference_year": [2006],
                "l2_reuse_year": [2006],
                "2005": [0.6],
            }
        ),
    )
    _write_prerequisite_scope_metadata(
        base_asocc_args=base_args,
        output_path=output_path,
        signature_overrides={
            "intermediate_outputs": False,
            "reference_years_input": None,
            "l2_reuse_years": None,
        },
    )

    report = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            }
        },
        sobol_parameters={"active": False},
        output_format="csv_compact",
        refresh=False,
    )

    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=report.manifest.run_id,
    )
    identity = _read_result_table(run_root=run_root, stem="public_row_identity")

    assert report.reuse_status == "computed"
    assert report.manifest.completed_runs == 1
    assert not identity.empty
    assert _deterministic_output_paths(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
    )


def test_uncertainty_asocc_prerequisite_resolves_disaggregated_source(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_disaggregated_prerequisite"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "source": "split_source",
        "years": [2030],
    }
    asocc_scope = build_asocc_scope(
        base_allocate_args=normalize_base_allocate_args(
            {
                **base_args,
                "projection_mode": "regression",
                "reg_window": [2005, 2006],
                "l2_reuse_years": [2005, 2006],
            }
        )
    )
    signature = asocc_scope.compute_signature(
        years=[2030],
        output_format="csv",
        intermediate_outputs=False,
        historical_year_cap=None,
    )
    manifest_path = allocate_run_metadata_path(scope=asocc_scope.resolve_path_scope())
    scope_payload = _build_run_metadata(
        requested_years=[2030],
        resolved_years=[2030],
        selected_methods=dict(signature.get("selected_methods", {})),
        fu_code=str(signature["fu_code"]),
        studied_indices_tag=str(signature.get("studied_indices_tag", "")),
        skipped_years={},
        outputs=["split_source.csv"],
        signature=signature,
    )
    scope_payload["execution"]["completed_years"] = [2030]
    _save_run_metadata(manifest_path, scope_payload)

    prerequisite = prepare_asocc_deterministic_prerequisite(
        base_asocc_args=base_args,
        refresh=False,
    )

    assert prerequisite.path_scope.source_label == "split_source"
    assert prerequisite.base_asocc_args["projection_mode"] == "regression"


def test_prerequisite_resolves_disaggregated_source_with_explicit_projection(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_disaggregated_explicit_projection"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "source": "split_source_direct",
        "years": [2030],
        "projection_mode": "regression",
        "reg_window": [2005, 2006],
        "l2_reuse_years": [2005, 2006],
    }
    _write_disaggregated_scope_manifest(base_asocc_args=base_args)

    prerequisite = prepare_asocc_deterministic_prerequisite(
        base_asocc_args=base_args,
        refresh=False,
    )

    assert prerequisite.base_asocc_args["projection_mode"] == "regression"
    assert [years for _scope, years in prerequisite.persisted_scope_matches] == [[2030]]


def test_prerequisite_keeps_explicit_disaggregated_projection_request(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_disaggregated_explicit_projection_miss"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "source": "split_source_explicit_miss",
        "years": [2030],
        "projection_mode": "historical_reuse",
        "reg_window": [2005, 2006],
        "l2_reuse_years": [2005, 2006],
    }
    _write_disaggregated_scope_manifest(
        base_asocc_args=base_args,
        signature_args={
            "projection_mode": "regression",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005, 2006],
        },
    )

    prerequisite = prepare_asocc_deterministic_prerequisite(
        base_asocc_args=base_args,
        refresh=False,
    )

    assert prerequisite.base_asocc_args["projection_mode"] == "historical_reuse"
    assert prerequisite.persisted_scope_matches == ()


def test_prerequisite_returns_no_disaggregated_match_for_uncovered_years(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_disaggregated_uncovered_year"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "source": "split_source_uncovered",
        "years": [2030],
    }
    _write_disaggregated_scope_manifest(
        base_asocc_args=base_args,
        signature_args={
            "projection_mode": "regression",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005, 2006],
        },
        completed_years=[2029],
    )

    prerequisite = prepare_asocc_deterministic_prerequisite(
        base_asocc_args=base_args,
        refresh=False,
    )

    assert prerequisite.base_asocc_args["projection_mode"] == "regression"
    assert prerequisite.persisted_scope_matches == ()


def test_prerequisite_rejects_future_reference_year_without_uncertainty() -> None:
    with pytest.raises(ValueError):
        prepare_asocc_deterministic_prerequisite(
            base_asocc_args={
                **_base_asocc_args(project_name="future_reference_year_without_uncertainty"),
                "years": [2005],
                "reference_years": [2006],
            },
            refresh=False,
        )


def test_prerequisite_rejects_reference_year_uncertainty_without_candidate() -> None:
    with pytest.raises(ValueError):
        prepare_asocc_deterministic_prerequisite(
            base_asocc_args={
                **_base_asocc_args(project_name="reference_year_uncertainty_without_candidate"),
                "years": [2005],
                "reference_years": [2006],
            },
            refresh=False,
            reference_year_uncertainty_active=True,
        )


def test_uncertainty_asocc_without_active_source_repeats_deterministic_rows(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_stable_rows"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )

    report = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            }
        },
        sobol_parameters={"active": False},
        output_format="csv_compact",
    )
    manifest = report.manifest
    reused = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            }
        },
        sobol_parameters={"active": False},
        output_format="csv_compact",
        figures=False,
    )

    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)

    assert manifest.active_sources == ()
    assert report.reuse_status == "computed"
    assert reused.reuse_status == "reused_exact"
    assert reused.manifest.run_id == manifest.run_id
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1]
    assert runs.groupby("run_index").size().nunique() == 1
    assert (run_root / "logs" / "source_methods.csv").exists()
    readme = (run_root / "results" / "README.txt").read_text(encoding="utf-8")
    assert readme.strip()
    assert "public_row_identity" in readme
    assert "asocc_runs" in readme
    assert all(len(line) <= 100 for line in readme.splitlines())


def test_uncertainty_asocc_skips_sobol_without_two_active_sources(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_sobol_outputs"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )

    manifest = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config=_uncertainty_config(),
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 1},
            "convergence": {"active": False},
        },
        figures=False,
    ).manifest

    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    assert manifest.sobol is not None
    assert manifest.sobol["mode"] == "fixed"
    assert not manifest.sobol["ran"]
    assert manifest.sobol["active_source_count"] == len(manifest.active_sources)
    assert not (run_root / "results" / "sobol" / "sobol_indices.csv").exists()

    sobol_convergence = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config=_uncertainty_config(),
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": False},
            "convergence": {"active": True, "max_base_samples": 4},
        },
    ).manifest

    assert sobol_convergence.run_id == manifest.run_id
    assert sobol_convergence.completed_runs == manifest.completed_runs
    assert sobol_convergence.sobol is not None
    assert sobol_convergence.sobol["mode"] == "fixed"
    assert not sobol_convergence.sobol["ran"]


def test_uncertainty_asocc_writes_inter_method_sobol_outputs(allocation_dummy_repo) -> None:
    project_name = "asocc_uncertainty_inter_method_projection_sobol"
    base_args = {
        "project_name": project_name,
        "source": "exiobase_396_ixi",
        "years": [2030],
        "reference_years": [2005],
        "fu_code": "L2.a.b",
        "method_plan": "pairs",
        "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FDa)", "PR(GDPcap)::UT(GVAa)"],
        "lcia_method": "gwp100_lcia",
        "r_p": ["FR"],
        "s_p": ["D"],
        "r_f": ["FR"],
        "l1_reg_aggreg": "pre",
        "ssp_scenario": ["SSP2"],
        "projection_mode": "historical_reuse",
        "reg_window": [2005, 2006],
        "l2_reuse_years": [2005, 2006],
    }
    deterministic_asocc(
        **base_args,
        figures=False,
        refresh=True,
    )
    no_sobol_manifest = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
            "projection_uncertainty": {},
        },
        output_format="csv_compact",
        figures=False,
    ).manifest

    manifest = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
            "projection_uncertainty": {},
        },
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 1},
            "convergence": {"active": False},
        },
        figures=False,
    ).manifest

    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
        source="exiobase_396_ixi",
    )
    sobol_root = run_root / "results" / "sobol"
    sobol = pd.read_csv(sobol_root / "sobol_indices.csv")
    source_summary = pd.read_csv(sobol_root / "sobol_source_summary.csv")

    assert manifest.sobol is not None
    assert manifest.run_id == no_sobol_manifest.run_id
    assert manifest.sobol["mode"] == "fixed"
    assert manifest.sobol["ran"]
    assert manifest.sobol["selected_output_years"] == [2030]
    assert not (sobol_root / "sobol_convergence.csv").exists()
    assert manifest.artifacts is not None
    assert manifest.artifacts["sobol_source_summary"].endswith("sobol_source_summary.csv")
    assert manifest.artifacts["public_output"] is not None
    public_output = manifest.artifacts["public_output"]
    sobol_columns = public_output["sobol_indices"]["columns"]
    sobol_summary_columns = public_output["sobol_source_summary"]["columns"]
    assert "sobol_output_variance" in sobol_columns
    assert "S1_confidence_half_width" in sobol_columns
    assert "ST_minus_S1" in sobol_columns
    assert "estimator_diagnostic" in sobol_columns
    assert "variance_weighted_ST" in sobol_summary_columns
    assert "variance_weighted_ST_confidence_half_width" in sobol_summary_columns
    assert "diagnostic_output_count" in sobol_summary_columns
    assert "summary_level" in sobol_summary_columns
    assert "contains_ssp_invariant_outputs" in sobol_summary_columns
    assert sobol["source_name"].unique().tolist() == [
        "inter_method_uncertainty",
        "projection_uncertainty",
    ]
    assert not {"l1_l2_method", "l1_method", "l2_method"} & set(sobol.columns)
    assert set(source_summary["source_name"]) == {
        "inter_method_uncertainty",
        "projection_uncertainty",
    }
    assert "year" in source_summary.columns
    assert "sobol_output_variance" in sobol.columns
    assert "ST_confidence_half_width" in sobol.columns
    assert "summary_level" in source_summary.columns
    assert "variance_weighted_ST_minus_S1" in source_summary.columns
    assert "variance_weighted_S1_confidence_half_width" in source_summary.columns
    assert not (sobol_root / "sobol_methods.csv").exists()
    assert "sobol_methods" not in manifest.artifacts
    assert manifest.sobol["method"]["selected_output_years"] == [2030]
    assert manifest.sobol["method"]["confidence_method"] == (
        "deterministic bootstrap over Sobol base rows"
    )
    assert manifest.sobol["parameters"]["convergence_targets"] == ["S1", "ST"]
    assert "targets" not in manifest.sobol["parameters"]
    sobol_readme = (sobol_root / "README_sobol.txt").read_text(encoding="utf-8")
    assert sobol_readme.strip()
    assert "sobol_source_summary" in sobol_readme
    assert "2030" in sobol_readme
    assert "scope_manifest.json" in sobol_readme
    assert "confidence_resamples=100" in sobol_readme
    for column in ("S1", "ST", "ST_minus_S1"):
        assert column in sobol_readme
    assert max(len(line) for line in sobol_readme.splitlines()) <= 88
    assert "Sobol" in sobol_readme
    assert not list(run_root.glob(".summary_values_*.dat"))

    reused_fixed_sobol = run_uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
            "projection_uncertainty": {},
        },
        external_method=None,
        output_format="csv_compact",
        figures=False,
        figure_options=None,
        figure_format=None,
        refresh=False,
        phase=NullPhasePrinter(),
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 1},
            "convergence": {"active": False},
        },
    ).manifest

    assert reused_fixed_sobol.run_id == manifest.run_id

    same_render_convergence = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
            "projection_uncertainty": {},
        },
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": False},
            "convergence": {"active": True, "max_base_samples": 1},
        },
        figures=False,
    ).manifest
    same_render_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=same_render_convergence.run_id,
        source="exiobase_396_ixi",
    )
    assert same_render_convergence.lineage is None
    assert same_render_convergence.sobol is not None
    assert same_render_convergence.sobol["mode"] == "convergence"
    assert same_render_convergence.artifacts is not None
    assert "sobol_convergence" not in same_render_convergence.artifacts
    assert "sobol_source_summary" in same_render_convergence.artifacts
    assert not (same_render_root / "results" / "sobol" / "sobol_convergence.csv").exists()
    assert not list(same_render_root.glob(".summary_values_*.dat"))

    convergence_manifest = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
            "projection_uncertainty": {},
        },
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": False},
            "convergence": {"active": True, "max_base_samples": 1},
        },
        figures=False,
    ).manifest
    convergence_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=convergence_manifest.run_id,
        source="exiobase_396_ixi",
    )

    assert convergence_manifest.sobol is not None
    assert convergence_manifest.sobol["mode"] == "convergence"
    assert (
        convergence_manifest.sobol["convergence_monitor"]
        == "selected_scope_source_confidence_interval"
    )
    assert "confidence_precision_pass" in convergence_manifest.sobol
    assert convergence_manifest.artifacts is not None
    assert "sobol_convergence" not in convergence_manifest.artifacts
    assert not (convergence_root / "results" / "sobol" / "sobol_convergence.csv").exists()
    assert "scope_manifest.json" in (
        convergence_root / "results" / "sobol" / "README_sobol.txt"
    ).read_text(encoding="utf-8")


def test_shared_sobol_readme_builder_handles_empty_notes() -> None:
    lines = build_sobol_readme_lines(
        suffix=".csv",
        family_label="aSoCC",
        source_names=("projection_uncertainty",),
        selected_scope_line="",
        plan=normalize_sobol_plan(sobol_parameters={}),
        source_summary_notes=("- retained selector scope note",),
        indices_notes=("",),
        method_notes=("",),
    )

    rendered = "\n".join(lines)
    assert lines[-1] != ""
    assert "retained" in rendered
    assert max(len(line) for line in lines) <= 88


def test_uncertainty_asocc_convergence_stops_on_stable_public_rows(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_convergence_stable_rows"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )

    manifest = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 5, "stable_runs": 2},
            }
        },
        output_format="csv_compact",
    ).manifest
    reused = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 5, "stable_runs": 2},
            }
        },
        output_format="csv_compact",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
    ).manifest
    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)

    assert manifest.completed_runs == 4
    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is True
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1, 2, 3]
    assert reused.run_id == manifest.run_id
    assert reused.artifacts is not None and reused.artifacts["figure_paths"]
    assert not (run_root / "private").exists()
    assert not list(run_root.glob(".convergence_values_*.dat"))


def test_uncertainty_asocc_convergence_reports_unreached_at_max_runs(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_convergence_external_unreached"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )
    _write_external_compact_monte_carlo_rows(
        repo_root=allocation_dummy_repo.repo_root,
        project_name=project_name,
    )

    manifest = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        external_method={"one_step_methods": ["UT(TD)"]},
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "rtol": 1e-12, "stable_runs": 1},
            },
            **_inactive_default_sources(),
        },
        output_format="csv_compact",
    ).manifest

    assert manifest.completed_runs == 2
    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is False
    assert manifest.convergence["last_check_runs"] == 2
    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    external = _read_asocc_runs(run_root)
    external = external.loc[external["l1_l2_method"].eq("UT(TD)")].reset_index(drop=True)
    assert external["asocc"].tolist() == [0.72, 0.73]
    assert manifest.artifacts is not None
    stale_run_file = Path(str(manifest.artifacts["scope_manifest"])).parents[1] / "stale.txt"
    stale_run_file.write_text("stale", encoding="utf-8")
    stale_upstream_file = (
        Path(str(manifest.deterministic_prerequisites[0]["scope_manifest"])).parents[1]
        / "stale.txt"
    )
    stale_upstream_file.write_text("stale", encoding="utf-8")
    refreshed = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        external_method={"one_step_methods": ["UT(TD)"]},
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "rtol": 1e-12, "stable_runs": 1},
            },
            **_inactive_default_sources(),
        },
        output_format="csv_compact",
        refresh=True,
    ).manifest
    assert refreshed.status == "complete"
    assert not stale_run_file.exists()
    assert not stale_upstream_file.exists()


def test_uncertainty_asocc_computes_larger_fixed_run(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_larger_fixed"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )

    first = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            }
        },
        output_format="csv_compact",
    ).manifest
    extended = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            }
        },
        output_format="csv_compact",
    ).manifest
    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=extended.run_id,
    )
    runs = _read_asocc_runs(run_root)

    assert extended.run_id != first.run_id
    assert extended.lineage is None
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1]

    reused = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            }
        },
        output_format="csv_compact",
    ).manifest
    assert reused.run_id == extended.run_id


def test_uncertainty_asocc_convergence_runs_without_parent_copy(
    allocation_dummy_repo,
) -> None:
    output_format = "parquet"
    project_name = "asocc_uncertainty_convergence_no_parent_copy"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )

    fixed = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            }
        },
        output_format=output_format,
    ).manifest
    convergence = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "stable_runs": 3},
            }
        },
        output_format=output_format,
    ).manifest
    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=convergence.run_id,
    )
    runs = _read_asocc_runs(run_root)

    assert convergence.run_id != fixed.run_id
    assert convergence.lineage is None
    assert convergence.convergence is not None
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1]

    extended_fixed = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            }
        },
        output_format=output_format,
    ).manifest

    assert extended_fixed.lineage is None
    reused_convergence = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "stable_runs": 3},
            }
        },
        output_format=output_format,
    ).manifest

    assert reused_convergence.run_id == convergence.run_id


def test_uncertainty_asocc_external_method_repeats_deterministic_rows(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_external_deterministic"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )
    external_path = _write_external_deterministic_rows(
        repo_root=allocation_dummy_repo.repo_root,
        project_name=project_name,
    )
    base_args = _base_asocc_args(project_name=project_name)
    config = {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 1}, "convergence": {"active": False}},
        **_inactive_default_sources(),
    }

    manifest = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config=config,
        external_method={"one_step_methods": ["UT(TD)"]},
        output_format="csv_compact",
    ).manifest

    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)
    external = runs.loc[runs["l1_l2_method"].eq("UT(TD)")].reset_index(drop=True)

    assert external["run_index"].tolist() == [0]
    assert external["asocc"].tolist() == [0.42]
    assert manifest.external_inputs == (
        {
            "storage_mode": "deterministic",
            "selection": "UT(TD)",
            "files": [
                {
                    "path": str(external_path),
                    "size_bytes": external_path.stat().st_size,
                    "sha256": hashlib.sha256(external_path.read_bytes()).hexdigest(),
                    "lcia_method": None,
                    "requested_years": [2005],
                    ASOCC_SSP_SCENARIO_COLUMN: None,
                }
            ],
        },
    )


def test_uncertainty_asocc_reference_year_keeps_invariant_external_rows(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_external_reference_invariant"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )
    _write_external_deterministic_rows(
        repo_root=allocation_dummy_repo.repo_root,
        project_name=project_name,
    )

    manifest = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config=_uncertainty_config(),
        external_method={"one_step_methods": ["UT(TD)"]},
        output_format="csv_compact",
    ).manifest

    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)

    assert runs.loc[runs["l1_l2_method"].eq("UT(TD)"), "asocc"].tolist() == [0.42]


def test_uncertainty_asocc_external_method_prefers_monte_carlo_rows(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_external_monte_carlo"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )
    _write_external_deterministic_rows(
        repo_root=allocation_dummy_repo.repo_root,
        project_name=project_name,
    )
    _write_external_monte_carlo_rows(
        repo_root=allocation_dummy_repo.repo_root,
        project_name=project_name,
    )

    manifest = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            **_inactive_default_sources(),
        },
        external_method={"one_step_methods": ["UT(TD)"]},
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 4},
            "convergence": {"active": False},
        },
    ).manifest

    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)
    external = runs.loc[runs["l1_l2_method"].eq("UT(TD)")].reset_index(drop=True)

    assert external["run_index"].tolist() == [0, 1]
    assert external["asocc"].tolist() == [0.72, 0.73]
    assert manifest.sobol is not None
    assert not manifest.sobol["ran"]
    assert manifest.sobol["active_source_count"] == 1


def test_uncertainty_asocc_rejects_non_lcia_external_monte_carlo_reference_year(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_external_monte_carlo_reference_year"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "years": [2006],
        "reference_years": [2005, 2006],
    }
    deterministic_asocc(
        **base_args,
        figures=False,
        refresh=True,
    )
    external_dir = get_asocc_external_method_level_dir(
        proj_base=allocation_dummy_repo.repo_root / f"{project_name}",
        storage_mode="monte_carlo",
        level="level_2",
    )
    external_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "run_index": [0, 0],
            "year": [2006, 2006],
            "r_p": ["FR", "FR"],
            "s_p": ["D", "D"],
            "reference_year": [2005, 2006],
            ASOCC_SSP_SCENARIO_COLUMN: [None, None],
            "value": [0.44, 0.45],
        }
    ).to_csv(external_dir / "UT(TD).csv", index=False)

    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args=base_args,
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 1},
                    "convergence": {"active": False},
                },
                "reference_year_uncertainty": {},
            },
            external_method={"one_step_methods": ["UT(TD)"]},
            output_format="csv_compact",
        )


def test_uncertainty_asocc_inter_method_writes_tree_artifacts(allocation_dummy_repo) -> None:
    project_name = "asocc_uncertainty_inter_method_tree"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "source": "exiobase_396_ixi",
        "lcia_method": "gwp100_lcia",
        "reference_years": [2005],
        "r_p": ["FR"],
        "s_p": ["D"],
        "one_step_methods": ["AR(E^{CBA_FD})", "UT(FD)"],
    }
    deterministic_asocc(
        **base_args,
        figures=False,
        refresh=True,
    )

    manifest = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
        },
        output_format="csv_compact",
        figure_format={"format": "svg", "dpi": 1},
    ).manifest

    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
        source="exiobase_396_ixi",
    )
    tree_csv = run_root / "figures" / "inter_method_tree" / "equal_weights.csv"
    tree_figure = run_root / "figures" / "inter_method_tree" / "probability_tree__equal_weights.svg"

    assert tree_csv.exists()
    assert tree_figure.exists()
    assert manifest.artifacts is not None
    assert manifest.artifacts["inter_method_tree_csv"] == str(tree_csv)
    assert manifest.artifacts["inter_method_tree_figure"] == str(tree_figure)
    assert manifest.source_parameters is not None
    assert "probability_tree_key" in manifest.source_parameters["inter_method_uncertainty"]


def test_uncertainty_asocc_inter_method_tree_paths_use_custom_version_name(
    tmp_path: Path,
) -> None:
    paths = build_asocc_uncertainty_run_paths(
        deterministic_manifest_path=tmp_path / "deterministic" / "logs" / "scope_manifest.json",
        run_id="mc_custom",
        output_format="csv_compact",
        inter_method_parameters={"mode": "custom", "version_name": "custom_v1"},
    )

    assert paths.inter_method_tree_csv.name == "weights__custom_v1.csv"
    assert paths.inter_method_tree_figure_base.name == "probability_tree__custom_v1"


def test_uncertainty_asocc_inter_method_larger_fixed_run_writes_sparse_rows(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_inter_method_larger_fixed"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "one_step_methods": ["AR(E^{CBA_FD})", "UT(FD)"],
    }
    deterministic_asocc(
        **base_args,
        figures=False,
        refresh=True,
    )

    first = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
        },
        output_format="csv_compact",
    ).manifest
    extended = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 3},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
        },
        output_format="csv_compact",
    ).manifest
    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=extended.run_id,
    )
    runs = _read_result_table(run_root=run_root, stem="asocc_runs")

    assert extended.run_id != first.run_id
    assert extended.lineage is None
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1, 2]


def test_uncertainty_asocc_inter_method_convergence_runs_without_parent_copy(
    allocation_dummy_repo,
) -> None:
    output_format = "parquet"
    project_name = "asocc_uncertainty_inter_method_convergence_no_parent"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "one_step_methods": ["AR(E^{CBA_FD})", "UT(FD)"],
    }
    deterministic_asocc(
        **base_args,
        figures=False,
        refresh=True,
    )

    fixed = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
        },
        output_format=output_format,
    ).manifest
    convergence = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "stable_runs": 3},
            },
            "inter_method_uncertainty": {},
        },
        output_format=output_format,
    ).manifest
    run_root = _uncertainty_run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=convergence.run_id,
    )
    runs = _read_result_table(run_root=run_root, stem="asocc_runs")

    assert convergence.run_id != fixed.run_id
    assert convergence.lineage is None
    assert convergence.convergence is not None
    assert convergence.completed_runs == 2
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1]

    reused = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "stable_runs": 3},
            },
            "inter_method_uncertainty": {},
        },
        output_format=output_format,
    ).manifest

    assert reused.run_id == convergence.run_id


def test_uncertainty_asocc_inter_method_convergence_without_parent(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_inter_method_convergence_no_parent"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "one_step_methods": ["AR(E^{CBA_FD})", "UT(FD)"],
    }
    deterministic_asocc(
        **base_args,
        figures=False,
        refresh=True,
    )

    manifest = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 3, "rtol": 1e-12, "stable_runs": 1},
            },
            "inter_method_uncertainty": {},
        },
        output_format="csv_compact",
    ).manifest

    assert manifest.convergence is not None
    assert 2 <= manifest.completed_runs <= 3


def test_uncertainty_asocc_external_method_reports_missing_input_files(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_external_missing"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )

    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args=_base_asocc_args(project_name=project_name),
            uncertainty_config=_uncertainty_config(),
            external_method={"one_step_methods": ["UT(TD)"]},
            output_format="csv_compact",
        )


def test_uncertainty_asocc_external_monte_carlo_requires_requested_runs(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_external_short_mc"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )
    _write_short_external_monte_carlo_rows(
        repo_root=allocation_dummy_repo.repo_root,
        project_name=project_name,
    )

    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args=_base_asocc_args(project_name=project_name),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                }
            },
            external_method={"one_step_methods": ["UT(TD)"]},
            output_format="csv_compact",
        )


def test_uncertainty_asocc_convergence_reports_external_render_inventory_exhaustion(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_external_short_mc_convergence"
    deterministic_asocc(
        **_base_asocc_args(project_name=project_name),
        figures=False,
        refresh=True,
    )
    _write_short_external_monte_carlo_rows(
        repo_root=allocation_dummy_repo.repo_root,
        project_name=project_name,
    )

    manifest = uncertainty_asocc(
        base_asocc_args=_base_asocc_args(project_name=project_name),
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "stable_runs": 1},
            },
            "inter_method_uncertainty": {"active": False},
        },
        external_method={"one_step_methods": ["UT(TD)"]},
        output_format="csv_compact",
    ).manifest

    assert manifest.completed_runs == 1
    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is False
    assert manifest.convergence["reason"]


def test_uncertainty_asocc_inter_method_convergence_reports_external_inventory_exhaustion(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_inter_method_external_short_mc_convergence"
    base_args = {
        **_base_asocc_args(project_name=project_name),
        "one_step_methods": ["UT(FD)"],
    }
    deterministic_asocc(
        **base_args,
        figures=False,
        refresh=True,
    )
    _write_short_external_monte_carlo_rows(
        repo_root=allocation_dummy_repo.repo_root,
        project_name=project_name,
    )

    manifest = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 2, "stable_runs": 1},
            },
            "inter_method_uncertainty": {},
        },
        external_method={"one_step_methods": ["UT(TD)"]},
        output_format="csv_compact",
    ).manifest

    assert manifest.completed_runs == 1
    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is False
