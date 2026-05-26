from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pytest

from pyaesa import write_asocc_weight_template, preview_asocc_weight_tree
from pyaesa.asocc.runtime.scope.branch_resolution import AsoccDeterministicPathScope
from pyaesa.asocc.runtime.request.scope import AsoccScope
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    ASOCC_TIME_ROUTE_COLUMN,
    ASOCC_VALUE_COLUMN,
    LoadedAsoccFinalRows,
    read_deterministic_asocc_rows,
)
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_HISTORICAL,
    ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
    ASOCC_TIME_ROUTE_REGRESSION,
)
from pyaesa.shared.runtime.reporting.run_progress import monte_carlo_run_progress
from pyaesa.asocc.uncertainty.sources.inter_method import (
    InterMethodPlan,
    build_inter_method_plan,
    inter_method_row_labels,
    sample_inter_method_labels,
)
from pyaesa.asocc.uncertainty.sources.inter_mrio import (
    InterMrioPlan,
    InterMrioRouteReport,
    _alternate_base_args,
    _interpolation_positions,
    apply_inter_mrio_uncertainty_to_matrix,
    inter_mrio_interpolation_matches,
    inter_mrio_route_report,
    inter_mrio_source_method_row,
)
from pyaesa.asocc.uncertainty.engine.inter_method.identity import (
    public_identity_for_sampling_plan,
)
from pyaesa.asocc.uncertainty.engine.monte_carlo.sampling import (
    compact_batch_inter_mrio_matches,
)
from pyaesa.asocc.uncertainty.sources.lcia import (
    LCIAPlan,
    lcia_sampling_memory_row_counts,
    lcia_uncertainty_has_targets,
    lcia_shared_u_for_plan,
)
from pyaesa.asocc.uncertainty.lcia_support.sampling import lcia_sample_block
from pyaesa.asocc.inter_method_tools.tree import (
    InterMethodCandidate,
    build_inter_method_tree_frame,
    candidates_from_rows,
    candidates_from_scope,
    default_inter_method_tree_probabilities,
    inter_method_tree_version_name,
    load_inter_method_tree_probabilities,
    write_inter_method_tree_csv,
)
from pyaesa.asocc.inter_method_tools.tree_figure import render_inter_method_tree
from pyaesa.asocc.uncertainty.schema.public_rows import (
    ASOCC_PUBLIC_VALUE_COLUMN,
    expand_rows_to_reference_lcia_axis,
)
from pyaesa.asocc.uncertainty.sources.projection import (
    apply_projection_uncertainty_to_matrix,
    build_projection_plan,
    projection_public_row_template,
    projection_value_matrix_for_indices,
    sample_projection_indices,
)
from pyaesa.asocc.uncertainty.sources.reference_year import (
    admissible_reference_year_rows,
    apply_reference_year_uncertainty_to_matrix,
    reference_year_uncertainty_has_targets,
    reference_year_source_method_row,
)
from pyaesa.asocc.uncertainty.engine.inter_method.execution import (
    InterMethodExecutionPlan,
    build_inter_method_execution_plan,
)
from pyaesa.asocc.uncertainty.engine.planning import _asocc_sampling_memory_blocks
from pyaesa.asocc.uncertainty.engine.inter_method.sampling import (
    _assign_branch_summary_values,
    _assign_sparse_branch_rows,
    _sparse_row_offsets,
    external_run_offsets_for_start,
    inter_method_inter_mrio_matches_by_branch,
    sample_inter_method_summary_matrix_batch,
    sample_sparse_inter_method_batch,
)
from pyaesa.asocc.uncertainty.engine.monte_carlo.sampling import (
    _compact_inter_mrio_input_template,
    sample_compact_batch,
)
from pyaesa.asocc.uncertainty.engine.evaluation.source_unit_intervals import (
    SourceUnitIntervalSamples,
)
from pyaesa.asocc.uncertainty.engine.monte_carlo.run_execution import (
    write_monte_carlo_run_outputs,
)
from pyaesa.asocc.uncertainty.engine.reuse.reuse import compatible_complete_sobol_run
from pyaesa.asocc.uncertainty.engine.monte_carlo.batch_sizing import batch_row_count
from pyaesa.asocc.uncertainty.engine.convergence.state import initial_convergence_state
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix,
    iter_sparse_run_rows,
)
from pyaesa.asocc.uncertainty.engine.convergence.convergence import (
    write_convergence_batches,
    write_sparse_inter_method_convergence_batches,
)
from pyaesa.asocc.uncertainty.engine.monte_carlo.fixed_batches import write_fixed_batches
from pyaesa.asocc.uncertainty.inputs.external_rows import (
    EXTERNAL_ASOCC_RUN_SOURCE,
    ExternalAsoccRowsPlan,
    append_external_monte_carlo_matrix,
    external_asocc_has_monte_carlo_rows,
    external_plan_for_years,
    external_method_row_mask,
)
from pyaesa.asocc.uncertainty.io.manifest_payloads import manifest_context
from pyaesa.asocc.uncertainty.sources.activation import deactivate_inter_mrio_without_targets
from pyaesa.asocc.uncertainty.sources.inter_mrio_reporting import inter_mrio_notes
from pyaesa.asocc.uncertainty.sources.names import (
    INTER_METHOD_SOURCE,
    INTER_MRIO_SOURCE,
    REFERENCE_YEAR_SOURCE,
)
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.asocc.uncertainty.io.artifacts import asocc_run_layout_from_manifest
from pyaesa.external_inputs.asocc.schema.contracts import ExternalMethodSelection
from pyaesa.external_inputs.asocc.monte_carlo.files import (
    ExternalMonteCarloRunMatrix,
    ExternalMonteCarloFileSelection,
    ExternalMonteCarloRowsSource,
    MaterializedExternalMonteCarloRowsSource,
    materialize_external_monte_carlo_source,
)
from pyaesa.asocc.uncertainty.engine.evaluation.summary_identity import (
    ASOCC_SUMMARY_SCOPE_COLUMN,
    ASOCC_SUMMARY_SCOPE_INTER_METHOD,
    ASOCC_SUMMARY_SCOPE_PER_METHOD,
    summary_identity_groups,
)
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    collapse_values_to_summary_groups,
)
from pyaesa.asocc.uncertainty.engine.sobol.evaluator import (
    AsoccSobolEvaluationContext,
    build_asocc_sobol_evaluation_context,
    build_asocc_sobol_evaluation_context_from_request,
    evaluate_asocc_sobol_units,
)
from pyaesa.asocc.uncertainty.engine.sobol.scope import (
    inter_mrio_plan_for_sobol_years,
    selected_sobol_years,
)
from pyaesa.asocc.uncertainty.engine.sobol.summary import asocc_sobol_source_summary
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan
from pyaesa.shared.uncertainty_assessment.sobol.plan import sobol_plan_payload
from pyaesa.shared.uncertainty_assessment.sobol.accumulator import SobolIndexEstimate
from pyaesa.shared.uncertainty_assessment.run_state.manifest import build_manifest
from pyaesa.shared.uncertainty_assessment.run_state.runs import CompatibleMonteCarloRun
from pyaesa.shared.uncertainty_assessment.request.sources import ActiveSource, SourceActivationPlan
from pyaesa.shared.uncertainty_assessment.request.core import UncertaintyRuntimeRequest
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.uncertainty_assessment.monte_carlo.random_streams import uniform_by_run_index
from pyaesa.asocc.uncertainty.io.source_methods import (
    SourceMethodRow,
)
from pyaesa.shared.uncertainty_assessment.io.csv_fragments import (
    CSV_COMPACT_RUN_FRAGMENT_COMPRESSION,
    CSV_COMPACT_RUN_FRAGMENT_SUFFIX,
)
from pyaesa.shared.uncertainty_assessment.io.run_writers import (
    SparseRunRows,
    SparseRunRowsWriter,
)


def _materialized_external_source(
    source: ExternalMonteCarloRowsSource,
) -> MaterializedExternalMonteCarloRowsSource:
    years = tuple(
        dict.fromkeys(
            year
            for file_selection in source.file_selections
            for year in file_selection.requested_years
        )
    ) or (2030,)
    return MaterializedExternalMonteCarloRowsSource(
        metadata=source,
        run_matrix=ExternalMonteCarloRunMatrix(
            template=pd.DataFrame({"year": list(years), "value": [0.0] * len(years)}),
            values=np.zeros((len(source.run_indices), len(years)), dtype=np.float64),
        ),
    )


def _loaded(
    *,
    rows: pd.DataFrame,
    base_asocc_args: dict | None = None,
    proj_base: Path | None = None,
):
    return LoadedAsoccFinalRows(
        base_asocc_args=base_asocc_args or {"fu_code": "L2.c.b"},
        asocc_scope=cast(AsoccScope, SimpleNamespace(combined=())),
        path_scope=cast(
            AsoccDeterministicPathScope,
            SimpleNamespace(proj_base=proj_base or Path(".")),
        ),
        persisted_scopes=(),
        deterministic_manifest_path=Path("."),
        requested_years=[2030],
        final_bucket="l2_vs_global",
        rows=rows,
    )


def _source_method_row() -> SourceMethodRow:
    return SourceMethodRow(
        source_component="asocc",
        source_name="test",
        scope="L2.c.b",
        applied_bucket="l2_vs_global",
        year_min=2030,
        year_max=2030,
        distribution="test",
        shared_random_variable="run_index",
        formula="test",
        notes="test",
    )


def _uncertainty_paths(*, run_root: Path) -> AsoccUncertaintyRunPaths:
    return AsoccUncertaintyRunPaths(
        run_root=run_root,
        public_row_identity=run_root / "results" / "public_row_identity.csv",
        public_runs=run_root / "results" / "asocc_runs.csv",
        summary_stats_runs=run_root / "results" / "summary_stats_runs.csv",
        results_readme=run_root / "results" / "README.txt",
        source_methods=run_root / "logs" / "source_methods.csv",
        inter_method_tree_csv=run_root / "figures" / "inter_method_tree.csv",
        inter_method_tree_figure_base=run_root / "figures" / "inter_method_tree",
        sobol_indices=run_root / "results" / "sobol" / "sobol_indices.csv",
        sobol_source_summary=run_root / "results" / "sobol" / "sobol_source_summary.csv",
        sobol_readme=run_root / "results" / "sobol" / "README_sobol.txt",
        scope_manifest=run_root / "logs" / "scope_manifest.json",
    )


def _compact_runs_frame(*, path: Path, column_count: int) -> pd.DataFrame:
    pieces = []
    for run_index, values in iter_compact_run_matrix(
        path=path,
        output_format="csv_compact",
        column_count=column_count,
    ):
        frame = pd.DataFrame(values, columns=[str(index) for index in range(column_count)])
        frame.insert(0, "run_index", run_index)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _sparse_runs_frame(*, path: Path) -> pd.DataFrame:
    pieces = []
    for rows in iter_sparse_run_rows(path=path, output_format="csv_compact"):
        pieces.append(
            pd.DataFrame(
                {
                    "run_index": rows.run_index,
                    "public_row_id": rows.public_row_id,
                    rows.value_column: rows.values,
                }
            )
        )
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def _external_sparse_inter_method_context(
    *,
    tmp_path: Path,
    available_runs: int,
) -> tuple[
    LoadedAsoccFinalRows,
    SourceActivationPlan,
    InterMethodPlan,
    InterMethodExecutionPlan,
    AsoccUncertaintyRunPaths,
]:
    path = tmp_path / "UT(TD).csv"
    pd.DataFrame(
        {
            "run_index": list(range(available_runs)),
            "year": [2030] * available_runs,
            ASOCC_SSP_SCENARIO_COLUMN: [None] * available_runs,
            "r_p": ["FR"] * available_runs,
            "s_p": ["Electricity"] * available_runs,
            "value": [0.7 + index / 10 for index in range(available_runs)],
        }
    ).to_csv(path, index=False)
    selection = ExternalMethodSelection(
        fu_code="L2.a.a",
        l2_method="UT(TD)",
        l1_method=None,
        level="level_2",
    )
    external_plan = ExternalAsoccRowsPlan(
        method_labels=("UT(TD)",),
        monte_carlo_sources=(
            materialize_external_monte_carlo_source(
                source=ExternalMonteCarloRowsSource(
                    selection=selection,
                    file_selections=(
                        ExternalMonteCarloFileSelection(
                            path=path,
                            lcia_method=None,
                            requested_years=(2030,),
                            ssp_scenario_options_by_year=None,
                        ),
                    ),
                    run_indices=tuple(range(available_runs)),
                ),
            ),
        ),
    )
    sources = SourceActivationPlan(
        sources=(ActiveSource(name="inter_method_uncertainty", parameters={}),)
    )
    inter_method_plan = InterMethodPlan(
        candidates=(),
        candidate_labels=("UT(TD)",),
        selection_probabilities=np.array([1.0], dtype=np.float64),
        tree_frame=pd.DataFrame(),
        source_method_row=_source_method_row(),
    )
    loaded = _loaded(
        rows=pd.DataFrame(columns=["l1_l2_method", "l2_method", ASOCC_VALUE_COLUMN]),
        base_asocc_args={"fu_code": "L2.a.a"},
    )
    execution_plan = build_inter_method_execution_plan(
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        sources=sources,
        external_plan=external_plan,
        lcia_plan=None,
        projection_plan=None,
    )
    return (
        loaded,
        sources,
        inter_method_plan,
        execution_plan,
        _uncertainty_paths(run_root=tmp_path / "run"),
    )


def _base_args(*, project_name: str, source: str = "exiobase_396_ixi") -> dict:
    return {
        "project_name": project_name,
        "source": source,
        "fu_code": "L2.c.b",
        "years": [2030],
        "method_plan": "one_step",
        "one_step_methods": ["UT(TD)"],
        "r_c": ["FR"],
        "s_p": ["Electricity"],
        "l1_reg_aggreg": "post",
    }


def _inter_mrio_rows(*, value: float, route: str = ASOCC_TIME_ROUTE_HISTORICAL) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "l1_l2_method": ["PR(GDPcap)_UT(GVAa)", "AR(E^{CBA_TD})"],
            "l1_method": ["PR(GDPcap)", None],
            "l2_method": ["UT(GVAa)", "AR(E^{CBA_TD})"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
            "impact": ["climate_parent", "climate_parent"],
            "year": [2030, 2030],
            ASOCC_TIME_ROUTE_COLUMN: [route, route],
            ASOCC_VALUE_COLUMN: [value, value + 1.0],
        }
    )


def test_inter_mrio_interpolates_only_non_lcia_compact_rows() -> None:
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=5)
    template = pd.concat(
        [
            _inter_mrio_rows(value=0.2),
            pd.DataFrame(
                {
                    "l1_l2_method": ["UT(TD)"],
                    "r_c": ["FR"],
                    "s_p": ["Electricity"],
                    "year": [2030],
                    ASOCC_TIME_ROUTE_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL],
                    ASOCC_VALUE_COLUMN: [0.5],
                }
            ),
        ],
        ignore_index=True,
    )
    values = np.broadcast_to(np.array([0.2, 1.2, 0.5]), (2, 3))
    alternate = pd.concat(
        [
            _inter_mrio_rows(value=0.8),
            pd.DataFrame(
                {
                    "l1_l2_method": ["UT(TD)"],
                    "r_c": ["FR"],
                    "s_p": ["Electricity"],
                    "year": [2030],
                    ASOCC_TIME_ROUTE_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL],
                    ASOCC_VALUE_COLUMN: [1.1],
                }
            ),
        ],
        ignore_index=True,
    )
    plan = InterMrioPlan(
        alternate_source="split_source",
        alternate_loaded=_loaded(rows=alternate),
        alternate_projection_plan=None,
        route_report=InterMrioRouteReport(
            interpolated_years=(2030,),
            skipped_years=(),
            skipped_route_pairs=(),
            skipped_scopes=(),
        ),
        source_method_row=_source_method_row(),
    )
    matches = inter_mrio_interpolation_matches(template=template, plan=plan)

    _template, sampled = apply_inter_mrio_uncertainty_to_matrix(
        template=template,
        values=values,
        plan=plan,
        batch=batch,
        projection_selection=None,
        matches=matches,
    )

    alpha = uniform_by_run_index(
        stream_name="asocc.inter_mrio.alpha",
        run_indices=batch.run_indices(),
    )
    expected = 0.2 + alpha * (0.8 - 0.2)
    np.testing.assert_allclose(sampled[:, 0], expected)
    np.testing.assert_allclose(sampled[:, 1], [1.2, 1.2])
    np.testing.assert_allclose(sampled[:, 2], 0.5 + alpha * (1.1 - 0.5))
    compact_matches = compact_batch_inter_mrio_matches(
        loaded=_loaded(rows=template),
        inter_mrio_plan=plan,
        lcia_plan=None,
        projection_plan=None,
    )
    assert compact_matches is not None
    np.testing.assert_array_equal(compact_matches.main_positions, matches.main_positions)


def test_inter_mrio_private_selectors_preserve_alternate_scope_and_positions() -> None:
    args = _alternate_base_args(
        base_asocc_args={
            "project_name": "demo",
            "source": "main",
            "agg_reg": True,
            "agg_sec": False,
            "agg_version": "eu27",
            "fu_code": "L2.c.b",
        },
        alternate_source="alternate",
    )
    assert args["source"] == "alternate"
    assert "agg_reg" not in args
    assert "agg_sec" not in args
    assert "agg_version" not in args

    main = pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)", "UT(TD)"],
            "r_c": ["FR", "DE"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2030, 2030],
            ASOCC_TIME_ROUTE_COLUMN: [
                ASOCC_TIME_ROUTE_HISTORICAL,
                ASOCC_TIME_ROUTE_HISTORICAL,
            ],
            ASOCC_VALUE_COLUMN: [0.2, 0.3],
        }
    )
    alternate = main.assign(**{ASOCC_VALUE_COLUMN: [0.8, 0.9]})
    main_positions, alternate_positions = _interpolation_positions(
        template=main,
        alternate_template=alternate,
        external_method_labels=(),
    )
    np.testing.assert_array_equal(main_positions, [0, 1])
    np.testing.assert_array_equal(alternate_positions, [0, 1])


def test_inter_mrio_matches_non_lcia_rows_across_lcia_public_axis() -> None:
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=1, rng_seed=5)
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)"],
            "l2_method": ["UT(TD)"],
            "r_c": ["FR"],
            "s_p": ["Electricity"],
            "lcia_method": [None],
            "impact": [None],
            "year": [2030],
            ASOCC_TIME_ROUTE_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL],
            ASOCC_VALUE_COLUMN: [0.2],
        }
    )
    alternate = rows.assign(
        lcia_method="gwp100_lcia",
        impact="GWP_100",
        **{ASOCC_VALUE_COLUMN: 0.8},
    )
    report = inter_mrio_route_report(main_rows=rows, alternate_rows=alternate)
    plan = InterMrioPlan(
        alternate_source="split_source",
        alternate_loaded=_loaded(rows=alternate),
        alternate_projection_plan=None,
        route_report=report,
        source_method_row=_source_method_row(),
    )

    _template, sampled = apply_inter_mrio_uncertainty_to_matrix(
        template=rows,
        values=np.array([[0.2]], dtype=np.float64),
        plan=plan,
        batch=batch,
        projection_selection=None,
        unit_values=np.array([0.5], dtype=np.float64),
    )

    assert report.interpolated_years == (2030,)
    assert report.skipped_years == ()
    np.testing.assert_allclose(sampled, [[0.5]])


def test_external_methods_skip_lcia_projection_and_inter_mrio_sources(
    allocation_dummy_repo,
) -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["AR(E^{CBA_FD})", "CO(S)"],
            "l1_method": [None, None],
            "l2_method": ["AR(E^{CBA_FD})", "CO(S)"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "lcia_method": ["gwp100_lcia", None],
            "impact": ["climate change", None],
            "year": [2030, 2030],
            "l2_reuse_year": [None, 2005],
            ASOCC_TIME_ROUTE_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL] * 2,
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    loaded = _loaded(
        rows=rows,
        base_asocc_args={"fu_code": "L2.c.b"},
        proj_base=allocation_dummy_repo.repo_root,
    )
    external_labels = ("AR(E^{CBA_FD})", "CO(S)")

    assert not lcia_uncertainty_has_targets(
        loaded=loaded,
        external_method_labels=external_labels,
    )
    projection_plan = build_projection_plan(
        loaded=loaded,
        external_method_labels=external_labels,
    )
    report = inter_mrio_route_report(
        main_rows=rows,
        alternate_rows=rows.iloc[0:0].copy(),
        external_method_labels=external_labels,
    )
    external_plan = ExternalAsoccRowsPlan(method_labels=external_labels)
    sources = SourceActivationPlan(
        sources=(
            ActiveSource(name="inter_method_uncertainty", parameters={}),
            ActiveSource(name="inter_mrio_uncertainty", parameters={}),
        )
    )
    inter_method_plan = build_inter_method_plan(
        loaded=loaded,
        parameters={},
        external_plan=external_plan,
    )
    execution_plan = build_inter_method_execution_plan(
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        sources=sources,
        external_plan=external_plan,
        lcia_plan=None,
        projection_plan=None,
    )
    inter_mrio_plan = InterMrioPlan(
        alternate_source="alternate",
        alternate_loaded=loaded,
        alternate_projection_plan=build_projection_plan(loaded=loaded),
        route_report=report,
        source_method_row=_source_method_row(),
        external_method_labels=external_labels,
    )
    assert (
        inter_method_inter_mrio_matches_by_branch(
            execution_plan=execution_plan,
            inter_mrio_plan=inter_mrio_plan,
        )
        == {}
    )

    assert projection_plan.passthrough_rows["l1_l2_method"].tolist() == [
        "AR(E^{CBA_FD})",
        "CO(S)",
    ]
    assert projection_plan.sampled_rows.empty
    assert report.skipped_years == ()
    assert (
        batch_row_count(
            loaded=loaded,
            inter_method_execution_plan=execution_plan,
            inter_mrio_plan=inter_mrio_plan,
            lcia_plan=None,
            projection_plan=None,
            sources=sources,
            external_plan=external_plan,
        )
        == 1
    )


def test_external_method_mask_supports_l1_only_rows() -> None:
    rows = pd.DataFrame({"l1_method": ["CO(S)"], ASOCC_VALUE_COLUMN: [0.2]})

    mask = external_method_row_mask(frame=rows, method_labels=("CO(S)",))

    assert mask.tolist() == [True]


def test_external_method_monte_carlo_discovery_uses_persisted_files(tmp_path: Path) -> None:
    loaded = replace(
        _loaded(
            rows=pd.DataFrame(
                {
                    "l2_method": ["UT(TD)"],
                    "r_c": ["FR"],
                    "s_p": ["Electricity"],
                    "year": [2030],
                    ASOCC_VALUE_COLUMN: [0.2],
                }
            ),
            base_asocc_args={"fu_code": "L2.c.b", "source": "main_source"},
            proj_base=tmp_path,
        ),
        path_scope=cast(
            AsoccDeterministicPathScope,
            SimpleNamespace(proj_base=tmp_path, source_label="demo_source"),
        ),
    )
    external_method = {"one_step_methods": ["UT(TD)"]}

    assert not external_asocc_has_monte_carlo_rows(
        loaded=loaded,
        external_method=external_method,
    )
    for lcia_method in ("gwp100_lcia", ["gwp100_lcia"]):
        assert not external_asocc_has_monte_carlo_rows(
            loaded=replace(
                loaded,
                base_asocc_args={**loaded.base_asocc_args, "lcia_method": lcia_method},
            ),
            external_method=external_method,
        )

    matrix_dir = tmp_path / "B1_asocc" / "external_asocc" / "monte_carlo" / "UT(TD)"
    matrix_dir.mkdir(parents=True)
    pd.DataFrame({"public_row_id": [0], "year": [2030]}).to_csv(
        matrix_dir / "public_row_identity.csv",
        index=False,
    )
    pd.DataFrame({"run_index": [0], "0": [0.2]}).to_csv(
        matrix_dir / "asocc_runs.csv",
        index=False,
    )

    assert external_asocc_has_monte_carlo_rows(
        loaded=loaded,
        external_method=external_method,
    )


def test_deterministic_row_loader_records_regression_route(tmp_path) -> None:
    path = tmp_path / "UT(TD).csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)"],
            "r_c": ["FR"],
            "s_p": ["Electricity"],
            ASOCC_TIME_ROUTE_COLUMN: [ASOCC_TIME_ROUTE_REGRESSION],
            "2030": [0.2],
        }
    ).to_csv(path, index=False)

    rows = read_deterministic_asocc_rows(path=path, requested_years=[2030])

    assert rows[ASOCC_TIME_ROUTE_COLUMN].tolist() == ["regression_proj"]


def test_inter_mrio_uses_compact_projection_selection_for_alternate_endpoint() -> None:
    main_rows = pd.DataFrame(
        {
            "l1_l2_method": ["PR(GDPcap)_UT(GVAa)", "PR(GDPcap)_UT(GVAa)"],
            "l1_method": ["PR(GDPcap)", "PR(GDPcap)"],
            "l2_method": ["UT(GVAa)", "UT(GVAa)"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2030, 2030],
            "l2_reuse_year": [2005, 2006],
            ASOCC_TIME_ROUTE_COLUMN: [
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
            ],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    alternate_rows = main_rows.copy()
    alternate_rows[ASOCC_VALUE_COLUMN] = [0.8, 1.0]
    main_projection = build_projection_plan(loaded=_loaded(rows=main_rows))
    alternate_projection = build_projection_plan(loaded=_loaded(rows=alternate_rows))
    plan = InterMrioPlan(
        alternate_source="split_source",
        alternate_loaded=_loaded(rows=alternate_rows),
        alternate_projection_plan=alternate_projection,
        route_report=InterMrioRouteReport(
            interpolated_years=(2030,),
            skipped_years=(),
            skipped_route_pairs=(),
            skipped_scopes=(),
        ),
        source_method_row=_source_method_row(),
    )
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=9)
    selected = sample_projection_indices(plan=main_projection, batch=batch)
    template = projection_public_row_template(plan=main_projection)
    values = projection_value_matrix_for_indices(
        plan=main_projection,
        batch=batch,
        selected_indices=selected,
    )

    _template, sampled = apply_inter_mrio_uncertainty_to_matrix(
        template=template,
        values=values,
        plan=plan,
        batch=batch,
        projection_selection=selected,
    )

    alternate_values = projection_value_matrix_for_indices(
        plan=alternate_projection,
        batch=batch,
        selected_indices=selected,
    )
    alpha = uniform_by_run_index(
        stream_name="asocc.inter_mrio.alpha",
        run_indices=batch.run_indices(),
    )
    expected = values[:, 0] + alpha * (alternate_values[:, 0] - values[:, 0])
    np.testing.assert_allclose(sampled[:, 0], expected)


def test_inter_mrio_keeps_main_values_when_route_is_not_comparable() -> None:
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=1, rng_seed=5)
    template = pd.concat(
        [
            _inter_mrio_rows(value=0.2, route=ASOCC_TIME_ROUTE_HISTORICAL_REUSE),
            pd.DataFrame(
                {
                    "l1_l2_method": ["UT(TD)"],
                    "r_c": ["FR"],
                    "s_p": ["Electricity"],
                    "year": [2030],
                    ASOCC_TIME_ROUTE_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL],
                    ASOCC_VALUE_COLUMN: [0.3],
                }
            ),
        ],
        ignore_index=True,
    )
    alternate = pd.concat(
        [
            _inter_mrio_rows(value=0.8, route=ASOCC_TIME_ROUTE_HISTORICAL),
            pd.DataFrame(
                {
                    "l1_l2_method": ["UT(TD)"],
                    "r_c": ["FR"],
                    "s_p": ["Electricity"],
                    "year": [2030],
                    ASOCC_TIME_ROUTE_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL],
                    ASOCC_VALUE_COLUMN: [0.9],
                }
            ),
        ],
        ignore_index=True,
    )
    values = np.array([[0.2, 1.2, 0.3]])
    plan = InterMrioPlan(
        alternate_source="split_source",
        alternate_loaded=_loaded(rows=alternate),
        alternate_projection_plan=None,
        route_report=InterMrioRouteReport(
            interpolated_years=(),
            skipped_years=(2030,),
            skipped_route_pairs=(
                "main=historical; alternate=historical",
                "main=historical_reuse; alternate=historical",
            ),
            skipped_scopes=("UT(GVAa) 2030", "UT(TD) 2030"),
        ),
        source_method_row=_source_method_row(),
    )

    _template, sampled = apply_inter_mrio_uncertainty_to_matrix(
        template=template,
        values=values,
        plan=plan,
        batch=batch,
        projection_selection=None,
    )

    np.testing.assert_allclose(sampled, values)
    report = inter_mrio_route_report(main_rows=template, alternate_rows=alternate)
    assert report.interpolated_years == ()
    assert report.skipped_years == (2030,)
    assert report.skipped_route_pairs == (
        "main=historical; alternate=historical",
        "main=historical_reuse; alternate=historical",
    )
    assert report.skipped_scopes == (
        "UT(GVAa) 2030",
        "UT(TD) 2030",
    )


def test_inter_mrio_direct_l2_rows_and_empty_external_template() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)"],
            "year": [2030],
            ASOCC_TIME_ROUTE_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL],
            ASOCC_VALUE_COLUMN: [0.2],
        }
    )
    alternate = rows.assign(**{ASOCC_VALUE_COLUMN: [0.8]})

    report = inter_mrio_route_report(main_rows=rows, alternate_rows=alternate)
    assert report.interpolated_years == (2030,)
    assert report.skipped_years == ()

    plan = InterMrioPlan(
        alternate_source="split_source",
        alternate_loaded=_loaded(rows=alternate),
        alternate_projection_plan=None,
        route_report=report,
        source_method_row=_source_method_row(),
    )
    template, values = apply_inter_mrio_uncertainty_to_matrix(
        template=rows.iloc[0:0].copy(),
        values=np.empty((1, 0), dtype=np.float64),
        plan=plan,
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=1, rng_seed=5),
        projection_selection=None,
    )

    assert template.empty
    assert values.shape == (1, 0)


def test_inter_mrio_report_uses_runtime_l2_reuse_year_alignment() -> None:
    main = pd.DataFrame(
        {
            "l1_l2_method": [
                "PR(GDPcap)_UT(GVAa)",
                "PR(GDPcap)_UT(GVAa)",
                "PR(GDPcap)_UT(GVAa)",
                "UT(TD)",
            ],
            "l1_method": ["PR(GDPcap)", "PR(GDPcap)", "PR(GDPcap)", None],
            "l2_method": ["UT(GVAa)", "UT(GVAa)", "UT(GVAa)", "UT(TD)"],
            "r_c": ["FR", "FR", "FR", "FR"],
            "s_p": ["Electricity", "Electricity", "Electricity", "Electricity"],
            "year": [2023, 2025, 2025, 2025],
            "l2_reuse_year": [np.nan, 2005.0, 2006.0, np.nan],
            ASOCC_TIME_ROUTE_COLUMN: [
                ASOCC_TIME_ROUTE_HISTORICAL,
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_REGRESSION,
            ],
            ASOCC_VALUE_COLUMN: [0.2, 0.3, 0.4, 0.5],
        }
    )
    alternate = pd.DataFrame(
        {
            "l1_l2_method": [
                "PR(GDPcap)_UT(GVAa)",
                "PR(GDPcap)_UT(GVAa)",
                "PR(GDPcap)_UT(GVAa)",
                "PR(GDPcap)_UT(GVAa)",
                "UT(TD)",
            ],
            "l1_method": ["PR(GDPcap)", "PR(GDPcap)", "PR(GDPcap)", "PR(GDPcap)", None],
            "l2_method": ["UT(GVAa)", "UT(GVAa)", "UT(GVAa)", "UT(GVAa)", "UT(TD)"],
            "r_c": ["FR", "FR", "FR", "FR", "FR"],
            "s_p": ["Electricity", "Electricity", "Electricity", "Electricity", "Electricity"],
            "year": [2023, 2023, 2025, 2025, 2025],
            "l2_reuse_year": [2005.0, 2006.0, 2005.0, 2006.0, np.nan],
            ASOCC_TIME_ROUTE_COLUMN: [
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_REGRESSION,
            ],
            ASOCC_VALUE_COLUMN: [0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )

    report = inter_mrio_route_report(main_rows=main, alternate_rows=alternate)

    assert report.interpolated_years == (2025,)
    assert report.skipped_years == (2023,)
    assert report.skipped_route_pairs == ("main=historical; alternate=historical_reuse",)
    assert report.skipped_scopes == ("UT(GVAa) 2023",)

    plan = InterMrioPlan(
        alternate_source="split_source",
        alternate_loaded=_loaded(rows=alternate),
        alternate_projection_plan=None,
        route_report=report,
        source_method_row=_source_method_row(),
    )
    _template, values = apply_inter_mrio_uncertainty_to_matrix(
        template=main,
        values=np.array([[0.2, 0.3, 0.4, 0.5]], dtype=np.float64),
        plan=plan,
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=1, rng_seed=5),
        projection_selection=None,
        unit_values=np.array([0.5], dtype=np.float64),
    )
    np.testing.assert_allclose(values, [[0.2, 0.55, 0.65, 0.75]])


def test_sampling_applies_inter_mrio_compact_owner() -> None:
    rows = _inter_mrio_rows(value=0.2).iloc[:1].reset_index(drop=True)
    alternate = _inter_mrio_rows(value=0.8).iloc[:1].reset_index(drop=True)
    plan = InterMrioPlan(
        alternate_source="split_source",
        alternate_loaded=_loaded(rows=alternate),
        alternate_projection_plan=None,
        route_report=InterMrioRouteReport(
            interpolated_years=(2030,),
            skipped_years=(),
            skipped_route_pairs=(),
            skipped_scopes=(),
        ),
        source_method_row=_source_method_row(),
    )

    identity, run_indices, values = sample_compact_batch(
        loaded=_loaded(rows=rows),
        inter_mrio_plan=plan,
        lcia_plan=None,
        projection_plan=None,
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=1, rng_seed=5),
        sources=SourceActivationPlan(
            sources=(ActiveSource(name="inter_mrio_uncertainty", parameters={}),)
        ),
        external_plan=ExternalAsoccRowsPlan(),
    )

    assert run_indices.tolist() == [0]
    assert identity["l1_l2_method"].tolist() == ["PR(GDPcap)_UT(GVAa)"]
    assert values.shape == (1, 1)
    assert values[0, 0] != 0.2


def test_inter_mrio_source_method_log_records_alternate_source() -> None:
    row = inter_mrio_source_method_row(
        loaded=_loaded(rows=_inter_mrio_rows(value=0.2)),
        alternate_source="split_source",
        route_report=InterMrioRouteReport(
            interpolated_years=(2029, 2031),
            skipped_years=(2030, 2031, 2033),
            skipped_route_pairs=("main=historical_reuse; alternate=historical",),
            skipped_scopes=(
                "UT(GVAa) 2030",
                "UT(GVAa) 2031",
                "UT(GVAa) 2033",
                "UT(TD) 2030",
            ),
        ),
    )

    assert row.source_name == "inter_mrio_uncertainty"
    assert row.notes is not None and "split_source" in row.notes
    for token in ("2029", "2031", "2030", "2033", "historical_reuse", "historical"):
        assert token in row.notes
    for method in ("UT(GVAa)", "UT(TD)"):
        assert method in row.notes


def test_inter_mrio_activation_keeps_sources_with_interpolation_targets() -> None:
    sources = SourceActivationPlan(sources=(ActiveSource(name=INTER_MRIO_SOURCE, parameters={}),))
    plan = InterMrioPlan(
        alternate_source="split_source",
        alternate_loaded=_loaded(rows=_inter_mrio_rows(value=0.8)),
        alternate_projection_plan=None,
        route_report=InterMrioRouteReport(
            interpolated_years=(2030,),
            skipped_years=(),
            skipped_route_pairs=(),
            skipped_scopes=(),
        ),
        source_method_row=_source_method_row(),
    )

    active_sources, active_plan = deactivate_inter_mrio_without_targets(
        sources=sources,
        plan=plan,
    )
    notes = inter_mrio_notes(alternate_source="split_source", route_report=plan.route_report)

    assert active_sources.is_active(INTER_MRIO_SOURCE)
    assert active_plan is plan
    assert "2030" in notes
    inactive_sources, inactive_plan = deactivate_inter_mrio_without_targets(
        sources=sources,
        plan=InterMrioPlan(
            alternate_source="split_source",
            alternate_loaded=_loaded(rows=_inter_mrio_rows(value=0.8)),
            alternate_projection_plan=None,
            route_report=InterMrioRouteReport(
                interpolated_years=(),
                skipped_years=(2030,),
                skipped_route_pairs=(),
                skipped_scopes=(),
            ),
            source_method_row=_source_method_row(),
        ),
    )
    assert not inactive_sources.is_active(INTER_MRIO_SOURCE)
    assert inactive_plan is None


def test_asocc_manifest_context_preserves_summary_records(tmp_path: Path) -> None:
    metadata_path = tmp_path / "scope_manifest.json"
    metadata_path.write_text(
        """
        {
          "summary_records": [
            4,
            {"severity": "INFO", "message": "deterministic info"},
            {"severity": "WARNING", "message": "deterministic warning"},
            {"severity": "DEBUG", "message": "developer note"},
            {"severity": "WARNING", "message": "   "}
          ]
        }
        """,
        encoding="utf-8",
    )
    loaded = replace(
        _loaded(
            rows=_inter_mrio_rows(value=0.2),
            base_asocc_args={"fu_code": "L2.c.b", "source": "main_source"},
            proj_base=tmp_path,
        ),
        path_scope=cast(
            AsoccDeterministicPathScope,
            SimpleNamespace(proj_base=tmp_path, source_label="main_source"),
        ),
        persisted_scopes=(
            SimpleNamespace(
                scope_key="scope_a",
                completed_years=[2030, 2031],
                output_format="csv_compact",
            ),
        ),
        deterministic_manifest_path=metadata_path,
    )
    inter_method_plan = InterMethodPlan(
        candidates=(),
        candidate_labels=("UT(TD)", "PR(GDPcap)_UT(GVAa)"),
        selection_probabilities=np.array([0.4, 0.6], dtype=np.float64),
        tree_frame=pd.DataFrame(),
        source_method_row=_source_method_row(),
    )
    runtime = SimpleNamespace(
        mode="fixed",
        n_runs=2,
        max_runs=2,
        rtol=0.01,
        stable_runs=2,
        convergence_statistics=("median",),
        output_format="csv_compact",
    )

    context = manifest_context(
        base_asocc_args=loaded.base_asocc_args,
        loaded=loaded,
        runtime=runtime,
        sources=SourceActivationPlan(
            sources=(ActiveSource(name=INTER_METHOD_SOURCE, parameters={}),)
        ),
        external_plan=ExternalAsoccRowsPlan(),
        inter_method_plan=inter_method_plan,
        inter_mrio_plan=SimpleNamespace(
            route_report=InterMrioRouteReport(
                interpolated_years=(),
                skipped_years=(2030, 2031),
                skipped_route_pairs=("main=historical; alternate=regression_proj",),
                skipped_scopes=("UT(TD) 2030", "UT(TD) 2031"),
            )
        ),
    )

    assert context["lineage"]["summary_records"][0]["severity"] == "WARNING"
    assert context["deterministic_prerequisites"][0]["summary_records"] == [
        {"severity": "INFO", "message": "deterministic info"},
        {"severity": "WARNING", "message": "deterministic warning"},
    ]
    assert context["source_parameters"][INTER_METHOD_SOURCE]["candidate_count"] == 2
    no_skips = manifest_context(
        base_asocc_args=loaded.base_asocc_args,
        loaded=loaded,
        runtime=runtime,
        sources=SourceActivationPlan(sources=()),
        external_plan=ExternalAsoccRowsPlan(),
        inter_mrio_plan=SimpleNamespace(
            route_report=InterMrioRouteReport(
                interpolated_years=(2030,),
                skipped_years=(),
                skipped_route_pairs=(),
                skipped_scopes=(),
            )
        ),
    )
    assert no_skips["lineage"] is None
    year_only_skip = manifest_context(
        base_asocc_args=loaded.base_asocc_args,
        loaded=loaded,
        runtime=runtime,
        sources=SourceActivationPlan(sources=()),
        external_plan=ExternalAsoccRowsPlan(),
        inter_mrio_plan=SimpleNamespace(
            route_report=InterMrioRouteReport(
                interpolated_years=(),
                skipped_years=(2032,),
                skipped_route_pairs=(),
                skipped_scopes=(),
            )
        ),
    )
    assert year_only_skip["lineage"]["summary_records"][0]["severity"] == "WARNING"


def test_inter_mrio_lcia_axis_expansion_keeps_rows_without_reference_axis() -> None:
    rows = pd.DataFrame({"year": [2030], ASOCC_VALUE_COLUMN: [0.2]})

    no_axis = expand_rows_to_reference_lcia_axis(
        rows=rows,
        reference=pd.DataFrame({"year": [2030]}),
    )
    empty_axis = expand_rows_to_reference_lcia_axis(
        rows=rows,
        reference=pd.DataFrame({"lcia_method": [None], "impact": [None]}),
    )

    assert no_axis.equals(rows)
    assert empty_axis.equals(rows)


def test_inter_method_samples_equal_weight_compact_candidates() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["PR(GDPcap)_UT(GVAa)", "EG(Pop)_UT(GVAa)"],
            "l1_method": ["PR(GDPcap)", "EG(Pop)"],
            "l2_method": ["UT(GVAa)", "UT(GVAa)"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2030, 2030],
            "l2_reuse_year": [None, None],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    plan = build_inter_method_plan(loaded=_loaded(rows=rows), parameters={})
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=3, rng_seed=17)

    selected = sample_inter_method_labels(plan=plan, batch=batch)

    assert plan.candidate_labels == ("EG(Pop)_UT(GVAa)", "PR(GDPcap)_UT(GVAa)")
    assert selected.shape == (3,)
    assert set(selected.tolist()).issubset(plan.candidate_labels)


def test_inter_method_row_labels_support_l1_only_rows() -> None:
    rows = pd.DataFrame(
        {
            "l1_method": ["EG(Pop)", "PR(GDPcap)"],
            "r_f": ["EU27", "US"],
            "year": [2018, 2019],
            ASOCC_VALUE_COLUMN: [0.1, 0.2],
        }
    )

    labels = inter_method_row_labels(rows=rows)
    plan = build_inter_method_plan(
        loaded=_loaded(rows=rows, base_asocc_args={"fu_code": "L1.a"}),
        parameters={},
    )

    assert labels.tolist() == ["EG(Pop)", "PR(GDPcap)"]
    assert plan.candidate_labels == ("EG(Pop)", "PR(GDPcap)")


def test_inter_method_external_render_offsets_count_prior_selected_runs() -> None:
    plan = InterMethodPlan(
        candidates=(),
        candidate_labels=("UT(TD)",),
        selection_probabilities=np.array([1.0], dtype=np.float64),
        tree_frame=pd.DataFrame(),
        source_method_row=_source_method_row(),
    )

    offsets = external_run_offsets_for_start(
        inter_method_plan=plan,
        start_run_index=200_003,
        external_labels=("UT(TD)", "missing_external"),
    )

    assert offsets == {"UT(TD)": 200_003, "missing_external": 0}

    native_plan = InterMethodPlan(
        candidates=(),
        candidate_labels=("UT(TD)", "native"),
        selection_probabilities=np.array([0.0, 1.0], dtype=np.float64),
        tree_frame=pd.DataFrame(),
        source_method_row=_source_method_row(),
    )

    native_offsets = external_run_offsets_for_start(
        inter_method_plan=native_plan,
        start_run_index=5,
        external_labels=("UT(TD)",),
    )

    assert native_offsets == {"UT(TD)": 0}


def test_inter_method_keeps_external_monte_carlo_selection_metadata() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["AR(E^{CBA_FD})_UT(FDa)", "AR(E^{CBA_TD})"],
            "l1_method": ["AR(E^{CBA_FD})", None],
            "l2_method": ["UT(FDa)", "AR(E^{CBA_TD})"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2030, 2030],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    external_plan = ExternalAsoccRowsPlan(
        monte_carlo_sources=(
            _materialized_external_source(
                ExternalMonteCarloRowsSource(
                    selection=ExternalMethodSelection(
                        fu_code="L2.c.b",
                        l2_method="AR(E^{CBA_FD})",
                        l1_method=None,
                        level="level_2",
                    ),
                    file_selections=(),
                    run_indices=(0,),
                )
            ),
        )
    )

    plan = build_inter_method_plan(
        loaded=_loaded(rows=rows),
        parameters={},
        external_plan=external_plan,
    )

    assert plan.candidate_labels == (
        "AR(E^{CBA_FD})",
        "AR(E^{CBA_FD})_UT(FDa)",
        "AR(E^{CBA_TD})",
    )
    assert np.isclose(plan.selection_probabilities.sum(), 1.0)


def test_inter_method_external_monte_carlo_uses_selected_run_index(tmp_path: Path) -> None:
    path = tmp_path / "UT(TD).csv"
    pd.DataFrame(
        {
            "run_index": [0, 1, 2],
            "year": [2030, 2030, 2030],
            ASOCC_SSP_SCENARIO_COLUMN: [None, None, None],
            "r_p": ["FR", "FR", "FR"],
            "s_p": ["Electricity", "Electricity", "Electricity"],
            "value": [0.7, 0.8, 0.9],
        }
    ).to_csv(path, index=False)
    selection = ExternalMethodSelection(
        fu_code="L2.a.a",
        l2_method="UT(TD)",
        l1_method=None,
        level="level_2",
    )
    external_plan = ExternalAsoccRowsPlan(
        method_labels=("UT(TD)",),
        monte_carlo_sources=(
            materialize_external_monte_carlo_source(
                source=ExternalMonteCarloRowsSource(
                    selection=selection,
                    file_selections=(
                        ExternalMonteCarloFileSelection(
                            path=path,
                            lcia_method=None,
                            requested_years=(2030,),
                            ssp_scenario_options_by_year=None,
                        ),
                    ),
                    run_indices=(0, 1, 2),
                ),
            ),
        ),
    )
    inter_method_plan = InterMethodPlan(
        candidates=(),
        candidate_labels=("UT(TD)",),
        selection_probabilities=np.array([1.0], dtype=np.float64),
        tree_frame=pd.DataFrame(),
        source_method_row=_source_method_row(),
    )
    loaded = _loaded(rows=pd.DataFrame(columns=["l1_l2_method", "l2_method", ASOCC_VALUE_COLUMN]))
    execution_plan = build_inter_method_execution_plan(
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        sources=SourceActivationPlan(
            sources=(ActiveSource(name="inter_method_uncertainty", parameters={}),)
        ),
        external_plan=external_plan,
        lcia_plan=None,
        projection_plan=None,
    )

    sampled = sample_sparse_inter_method_batch(
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        execution_plan=execution_plan,
        inter_mrio_plan=None,
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=3, rng_seed=7),
        sources=SourceActivationPlan(
            sources=(ActiveSource(name="inter_method_uncertainty", parameters={}),)
        ),
        identity=None,
    )

    assert sampled.sparse_rows.run_index.tolist() == [0, 1, 2]
    np.testing.assert_allclose(sampled.sparse_rows.values, [0.7, 0.8, 0.9])


def test_sparse_inter_method_fixed_batches_advance_external_render_offsets(tmp_path: Path) -> None:
    loaded, sources, inter_method_plan, execution_plan, paths = (
        _external_sparse_inter_method_context(tmp_path=tmp_path, available_runs=1)
    )
    runtime = SimpleNamespace(n_runs=1, batch_size=1, output_format="csv_compact")

    result = write_fixed_batches(
        paths=paths,
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        inter_method_execution_plan=execution_plan,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=runtime,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        start_run_index=0,
        append_existing=False,
        run_seed=17,
    )

    runs = _sparse_runs_frame(path=paths.public_runs)
    assert result.completed_runs == 1
    assert runs["run_index"].tolist() == [0]
    np.testing.assert_allclose(runs[ASOCC_PUBLIC_VALUE_COLUMN], [0.7])


def test_fixed_batches_append_existing_compact_and_sparse_runs(tmp_path: Path) -> None:
    compact_paths = _uncertainty_paths(run_root=tmp_path / "compact_run")
    compact_loaded = _loaded(
        rows=pd.DataFrame(
            {
                "l2_method": ["UT(TD)"],
                "r_c": ["FR"],
                "s_p": ["Electricity"],
                "year": [2030],
                ASOCC_VALUE_COLUMN: [0.2],
            }
        )
    )
    compact_sources = SourceActivationPlan(sources=())
    first_runtime = SimpleNamespace(n_runs=1, batch_size=1, output_format="csv_compact")
    second_runtime = SimpleNamespace(n_runs=2, batch_size=1, output_format="csv_compact")

    write_fixed_batches(
        paths=compact_paths,
        loaded=compact_loaded,
        inter_method_plan=None,
        inter_method_execution_plan=None,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=first_runtime,
        sources=compact_sources,
        external_plan=ExternalAsoccRowsPlan(),
        start_run_index=0,
        append_existing=False,
        run_seed=11,
        show_progress=False,
    )
    compact_result = write_fixed_batches(
        paths=compact_paths,
        loaded=compact_loaded,
        inter_method_plan=None,
        inter_method_execution_plan=None,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=second_runtime,
        sources=compact_sources,
        external_plan=ExternalAsoccRowsPlan(),
        start_run_index=1,
        append_existing=True,
        run_seed=11,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )

    compact_runs = _compact_runs_frame(path=compact_paths.public_runs, column_count=1)
    assert compact_result.completed_runs == 2
    assert compact_runs["run_index"].tolist() == [0, 1]

    sparse_zero_root = tmp_path / "sparse_zero"
    sparse_zero_root.mkdir()
    zero_loaded, zero_sources, zero_inter_method_plan, zero_execution_plan, zero_paths = (
        _external_sparse_inter_method_context(tmp_path=sparse_zero_root, available_runs=2)
    )
    write_fixed_batches(
        paths=zero_paths,
        loaded=zero_loaded,
        inter_method_plan=zero_inter_method_plan,
        inter_method_execution_plan=zero_execution_plan,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=first_runtime,
        sources=zero_sources,
        external_plan=ExternalAsoccRowsPlan(),
        start_run_index=0,
        append_existing=False,
        run_seed=17,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )
    zero_sparse_result = write_fixed_batches(
        paths=zero_paths,
        loaded=zero_loaded,
        inter_method_plan=zero_inter_method_plan,
        inter_method_execution_plan=zero_execution_plan,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=second_runtime,
        sources=zero_sources,
        external_plan=ExternalAsoccRowsPlan(),
        start_run_index=1,
        append_existing=True,
        run_seed=17,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )
    assert zero_sparse_result.completed_runs == 2

    sparse_root = tmp_path / "sparse"
    sparse_root.mkdir()
    loaded, sources, inter_method_plan, execution_plan, sparse_paths = (
        _external_sparse_inter_method_context(tmp_path=sparse_root, available_runs=3)
    )
    write_fixed_batches(
        paths=sparse_paths,
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        inter_method_execution_plan=execution_plan,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=first_runtime,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        start_run_index=0,
        append_existing=False,
        run_seed=17,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )
    existing_sparse_runs = _sparse_runs_frame(path=sparse_paths.public_runs)
    existing_sparse_runs["run_index"] = 1
    table = pa.Table.from_pandas(existing_sparse_runs, preserve_index=False)
    with pa.CompressedOutputStream(
        str(sparse_paths.public_runs / f"part-00000000{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}"),
        CSV_COMPACT_RUN_FRAGMENT_COMPRESSION,
    ) as handle:
        pacsv.write_csv(table, handle, write_options=pacsv.WriteOptions(include_header=True))
    sparse_runtime = SimpleNamespace(n_runs=3, batch_size=1, output_format="csv_compact")
    sparse_result = write_fixed_batches(
        paths=sparse_paths,
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        inter_method_execution_plan=execution_plan,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=sparse_runtime,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        start_run_index=2,
        append_existing=True,
        run_seed=17,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )

    sparse_runs = _sparse_runs_frame(path=sparse_paths.public_runs)
    assert sparse_result.completed_runs == 3
    assert sparse_runs["run_index"].tolist() == [1, 2]


def test_compact_convergence_append_replays_existing_means(tmp_path: Path) -> None:
    paths = _uncertainty_paths(run_root=tmp_path / "compact_convergence")
    loaded = _loaded(
        rows=pd.DataFrame(
            {
                "l2_method": ["UT(TD)"],
                "r_c": ["FR"],
                "s_p": ["Electricity"],
                "year": [2030],
                ASOCC_VALUE_COLUMN: [0.2],
            }
        )
    )
    sources = SourceActivationPlan(sources=())
    first_runtime = SimpleNamespace(
        mode="convergence",
        n_runs=1,
        batch_size=1,
        output_format="csv_compact",
        rtol=0.0,
        stable_runs=1,
    )
    second_runtime = SimpleNamespace(
        mode="convergence",
        n_runs=2,
        batch_size=1,
        output_format="csv_compact",
        rtol=0.0,
        stable_runs=1,
    )
    write_convergence_batches(
        paths=paths,
        loaded=loaded,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=first_runtime,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        run_seed=11,
        start_run_index=0,
        append_existing=False,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )

    result = write_convergence_batches(
        paths=paths,
        loaded=loaded,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=second_runtime,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        run_seed=11,
        start_run_index=1,
        append_existing=True,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )
    runs = _compact_runs_frame(path=paths.public_runs, column_count=1)

    assert result.completed_runs == 2
    assert result.convergence is not None
    assert result.convergence["reached"] is True
    assert runs["run_index"].tolist() == [0, 1]
    crossing_paths = _uncertainty_paths(run_root=tmp_path / "compact_convergence_crossing")
    crossing_result = write_convergence_batches(
        paths=crossing_paths,
        loaded=loaded,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=SimpleNamespace(
            mode="convergence",
            n_runs=3,
            batch_size=1,
            output_format="csv_compact",
            rtol=0.0,
            stable_runs=2,
        ),
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        run_seed=11,
        start_run_index=0,
        append_existing=False,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )
    assert crossing_result.completed_runs == 3
    assert crossing_result.convergence is not None
    assert crossing_result.convergence["reached"] is False


def test_sparse_inter_method_convergence_stops_when_statistics_stabilize(
    tmp_path: Path,
) -> None:
    loaded, sources, inter_method_plan, execution_plan, paths = (
        _external_sparse_inter_method_context(tmp_path=tmp_path, available_runs=3)
    )
    runtime = SimpleNamespace(
        mode="convergence",
        n_runs=3,
        batch_size=1,
        output_format="csv_compact",
        convergence_statistics=("median",),
        rtol=1e6,
        stable_runs=2,
    )

    result = write_monte_carlo_run_outputs(
        paths=paths,
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        inter_method_execution_plan=execution_plan,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=runtime,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        append_run=None,
        run_seed=17,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
        show_progress=False,
    )

    assert result.completed_runs == 3
    assert result.convergence is not None
    assert result.convergence["reached"] is True
    reused_result = write_sparse_inter_method_convergence_batches(
        paths=paths,
        loaded=loaded,
        inter_method_plan=inter_method_plan,
        inter_method_execution_plan=execution_plan,
        inter_mrio_plan=None,
        runtime=SimpleNamespace(
            mode="convergence",
            n_runs=result.completed_runs,
            batch_size=1,
            output_format="csv_compact",
            convergence_statistics=("median",),
            rtol=1e6,
            stable_runs=2,
        ),
        sources=sources,
        run_seed=17,
        start_run_index=result.completed_runs,
        append_existing=True,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )
    assert reused_result.completed_runs == result.completed_runs
    unstable_root = tmp_path / "unstable"
    unstable_root.mkdir()
    unstable_loaded, unstable_sources, unstable_plan, unstable_execution, unstable_paths = (
        _external_sparse_inter_method_context(tmp_path=unstable_root, available_runs=2)
    )
    unstable_result = write_sparse_inter_method_convergence_batches(
        paths=unstable_paths,
        loaded=unstable_loaded,
        inter_method_plan=unstable_plan,
        inter_method_execution_plan=unstable_execution,
        inter_mrio_plan=None,
        runtime=SimpleNamespace(
            mode="convergence",
            n_runs=2,
            batch_size=1,
            output_format="csv_compact",
            convergence_statistics=("median",),
            rtol=0.0,
            stable_runs=1,
        ),
        sources=unstable_sources,
        run_seed=17,
        start_run_index=0,
        append_existing=False,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )
    assert unstable_result.convergence is not None
    assert unstable_result.convergence["reached"] is False


def test_monte_carlo_run_outputs_reuses_complete_append_run() -> None:
    result = write_monte_carlo_run_outputs(
        paths=None,
        loaded=None,
        inter_method_plan=object(),
        inter_method_execution_plan=None,
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        runtime=SimpleNamespace(mode="fixed", n_runs=2),
        sources=None,
        external_plan=ExternalAsoccRowsPlan(),
        append_run=SimpleNamespace(
            manifest=SimpleNamespace(completed_runs=3, convergence={"reached": True})
        ),
        run_seed=17,
        progress=monte_carlo_run_progress(source="uncertainty_asocc", enabled=False),
    )

    assert result.completed_runs == 3
    assert result.summary_run_count == 3
    assert result.public_runs_sparse is True
    assert result.convergence == {"reached": True}


def test_sparse_inter_method_convergence_initial_state_uses_completed_run_count(
    tmp_path: Path,
) -> None:
    paths = AsoccUncertaintyRunPaths(
        run_root=tmp_path / "run",
        public_row_identity=tmp_path / "run" / "results" / "identity.csv",
        public_runs=tmp_path / "run" / "results" / "runs.csv",
        summary_stats_runs=tmp_path / "run" / "results" / "summary.csv",
        results_readme=tmp_path / "run" / "results" / "README.txt",
        source_methods=tmp_path / "run" / "logs" / "source_methods.csv",
        inter_method_tree_csv=tmp_path / "run" / "logs" / "tree.csv",
        inter_method_tree_figure_base=tmp_path / "run" / "logs" / "tree",
        sobol_indices=tmp_path / "run" / "results" / "sobol" / "indices.csv",
        sobol_source_summary=tmp_path / "run" / "results" / "sobol" / "summary.csv",
        sobol_readme=tmp_path / "run" / "results" / "sobol" / "README.txt",
        scope_manifest=tmp_path / "run" / "logs" / "scope_manifest.json",
    )
    paths.public_runs.parent.mkdir(parents=True, exist_ok=True)
    paths.public_row_identity.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "year": [2030, 2030],
            "l1_l2_method": ["A", "B"],
            "l1_method": ["A", "B"],
            "l2_method": ["A", "B"],
            ASOCC_TIME_ROUTE_COLUMN: [
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
            ],
        }
    ).to_csv(paths.public_row_identity, index=False)
    pd.DataFrame(
        {
            "run_index": [1, 1],
            "public_row_id": [0, 1],
            ASOCC_PUBLIC_VALUE_COLUMN: [2.0, 4.0],
        }
    ).to_csv(paths.public_runs, index=False)

    state = initial_convergence_state(
        paths=paths,
        output_format="csv_compact",
        sources=SourceActivationPlan(
            sources=(ActiveSource(name=INTER_METHOD_SOURCE, parameters={}),)
        ),
        append_existing=True,
        sparse=True,
        completed_runs=2,
    )

    assert state.completed_runs == 2
    assert state.identity_written
    assert state.row_count == 3
    pd.DataFrame(
        {
            "run_index": [0, 1],
            "0": [1.0, 2.0],
            "1": [3.0, 4.0],
        }
    ).to_csv(paths.public_runs, index=False)
    compact_state = initial_convergence_state(
        paths=paths,
        output_format="csv_compact",
        sources=SourceActivationPlan(
            sources=(ActiveSource(name=INTER_METHOD_SOURCE, parameters={}),)
        ),
        append_existing=True,
        sparse=False,
        completed_runs=2,
    )
    assert compact_state.completed_runs == 2
    pd.DataFrame(
        {
            "run_index": [0, 0],
            "public_row_id": [0, 1],
            ASOCC_PUBLIC_VALUE_COLUMN: [2.0, 4.0],
        }
    ).to_csv(paths.public_runs, index=False)
    zero_start_state = initial_convergence_state(
        paths=paths,
        output_format="csv_compact",
        sources=SourceActivationPlan(
            sources=(ActiveSource(name=INTER_METHOD_SOURCE, parameters={}),)
        ),
        append_existing=True,
        sparse=True,
        completed_runs=1,
    )
    assert zero_start_state.completed_runs == 1


def test_asocc_run_layout_contract_reads_manifest_public_output() -> None:
    manifest = build_manifest(
        family="asocc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        artifacts={"public_output": {"asocc_runs": {"layout": "sparse_selected_rows"}}},
    )

    assert asocc_run_layout_from_manifest(manifest=manifest) == "sparse_selected_rows"


def test_sobol_reuse_rejects_stale_external_source_dimensions() -> None:
    sobol_plan = SobolPlan(
        enabled=True,
        mode="fixed",
        n_base_samples=4,
        max_base_samples=4,
        rtol=0.05,
    )
    source = ExternalMonteCarloRowsSource(
        selection=ExternalMethodSelection(
            fu_code="L2.a.a",
            l2_method="UT(TD)",
            l1_method=None,
            level="level_2",
        ),
        file_selections=(
            ExternalMonteCarloFileSelection(
                path=Path("external.csv"),
                lcia_method=None,
                requested_years=(2030,),
                ssp_scenario_options_by_year=None,
            ),
        ),
        run_indices=(0, 1),
    )
    external_plan = ExternalAsoccRowsPlan(
        monte_carlo_sources=(_materialized_external_source(source),)
    )
    sources = SourceActivationPlan(
        sources=(ActiveSource(name="inter_method_uncertainty", parameters={}),)
    )
    runtime = SimpleNamespace(n_runs=2, mode="fixed")
    stale = CompatibleMonteCarloRun(
        run_root=Path("."),
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=sources.names,
            status="complete",
            completed_runs=2,
            requested_runs=2,
            sobol={
                "ran": True,
                "parameters": sobol_plan_payload(plan=sobol_plan),
                "selected_output_years": [2030],
                "method": {
                    "source_dimensions": list(sources.names),
                    "selected_output_years": [2030],
                },
            },
        ),
    )
    stale_scope = CompatibleMonteCarloRun(
        run_root=Path("stale_scope"),
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=sources.names,
            status="complete",
            completed_runs=2,
            requested_runs=2,
            sobol={
                "ran": True,
                "parameters": sobol_plan_payload(plan=sobol_plan),
                "selected_output_years": [2029, 2030],
                "method": {
                    "source_dimensions": [*sources.names, EXTERNAL_ASOCC_RUN_SOURCE],
                    "selected_output_years": [2029, 2030],
                },
            },
        ),
    )
    missing_method = CompatibleMonteCarloRun(
        run_root=Path("missing_method"),
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=sources.names,
            status="complete",
            completed_runs=2,
            requested_runs=2,
            sobol={
                "ran": True,
                "parameters": sobol_plan_payload(plan=sobol_plan),
                "selected_output_years": [2030],
            },
        ),
    )
    current = CompatibleMonteCarloRun(
        run_root=Path("current"),
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=sources.names,
            status="complete",
            completed_runs=2,
            requested_runs=2,
            sobol={
                "ran": True,
                "parameters": sobol_plan_payload(plan=sobol_plan),
                "selected_output_years": [2030],
                "method": {
                    "source_dimensions": [*sources.names, EXTERNAL_ASOCC_RUN_SOURCE],
                    "selected_output_years": [2030],
                },
            },
        ),
    )

    reusable = compatible_complete_sobol_run(
        compatible=(missing_method, stale, stale_scope, current),
        runtime=runtime,
        mc_parameters=None,
        sources=sources,
        external_plan=external_plan,
        sobol_plan=sobol_plan,
        requested_years=(2030,),
    )

    assert reusable == current


def test_sobol_reuse_delegates_to_run_reuse_for_single_source_dimension() -> None:
    sobol_plan = SobolPlan(
        enabled=True,
        mode="fixed",
        n_base_samples=4,
        max_base_samples=4,
        rtol=0.05,
    )
    sources = SourceActivationPlan(sources=(ActiveSource(name="lcia_uncertainty", parameters={}),))
    runtime = SimpleNamespace(n_runs=2, mode="fixed")
    compatible = CompatibleMonteCarloRun(
        run_root=Path("current"),
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=sources.names,
            status="complete",
            completed_runs=2,
            requested_runs=2,
        ),
    )

    reusable = compatible_complete_sobol_run(
        compatible=(compatible,),
        runtime=runtime,
        mc_parameters=None,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        sobol_plan=sobol_plan,
        requested_years=(2030,),
    )

    assert reusable == compatible


def test_inter_method_tree_matches_reference_l2_bucket_weights(tmp_path: Path) -> None:
    candidates = (
        InterMethodCandidate(
            "AR(E^{CBA_FD})_UT(FDa)",
            "level_2",
            "AR(E^{CBA_FD})",
            "UT(FDa)",
        ),
        InterMethodCandidate(
            "AR(E^{PBA})_UT(GVAa)",
            "level_2",
            "AR(E^{PBA})",
            "UT(GVAa)",
        ),
        InterMethodCandidate("AR(E^{CBA_TD})", "level_2", None, "AR(E^{CBA_TD})"),
        InterMethodCandidate("EG(Pop)_UT(FDa)", "level_2", "EG(Pop)", "UT(FDa)"),
        InterMethodCandidate("EG(Pop)_UT(GVAa)", "level_2", "EG(Pop)", "UT(GVAa)"),
        InterMethodCandidate("PR(GDPcap)_UT(FDa)", "level_2", "PR(GDPcap)", "UT(FDa)"),
        InterMethodCandidate("PR(GDPcap)_UT(GVAa)", "level_2", "PR(GDPcap)", "UT(GVAa)"),
        InterMethodCandidate("UT(TD)", "level_2", None, "UT(TD)"),
    )

    frame = build_inter_method_tree_frame(candidates=candidates)
    probabilities = default_inter_method_tree_probabilities(candidates=candidates)
    edge_by_node = {
        str(node_id): float(edge_weight)
        for node_id, edge_weight in frame.loc[:, ["node_id", "edge_weight"]].itertuples(
            index=False,
            name=None,
        )
    }
    probability_by_candidate = {
        candidate.candidate_label: float(probability)
        for candidate, probability in zip(candidates, probabilities, strict=True)
    }
    rendered = render_inter_method_tree(
        frame=frame,
        figure_base_path=tmp_path / "bucket_tree",
        output_format="png",
        dpi=10,
    )

    assert frame.columns.tolist() == [
        "parent_id",
        "node_id",
        "label",
        "node_type",
        "edge_weight",
        "level",
        "candidate_label",
    ]
    assert frame.loc[frame["parent_id"].eq("AR"), "label"].tolist() == ["m_s", "o_s"]
    assert edge_by_node["m_s"] == 0.5
    assert edge_by_node["o_s"] == 0.5
    assert edge_by_node["E_CBA_TD"] == 1.0
    np.testing.assert_allclose(
        [
            probability_by_candidate["AR(E^{CBA_FD})_UT(FDa)"],
            probability_by_candidate["AR(E^{PBA})_UT(GVAa)"],
            probability_by_candidate["AR(E^{CBA_TD})"],
            probability_by_candidate["EG(Pop)_UT(FDa)"],
            probability_by_candidate["EG(Pop)_UT(GVAa)"],
            probability_by_candidate["PR(GDPcap)_UT(FDa)"],
            probability_by_candidate["PR(GDPcap)_UT(GVAa)"],
            probability_by_candidate["UT(TD)"],
        ],
        [0.0625, 0.0625, 0.125, 0.125, 0.125, 0.125, 0.125, 0.25],
    )
    assert rendered == [tmp_path / "bucket_tree.png"]
    assert rendered[0].exists()


def test_inter_method_tree_renders_l1_reference_scope(tmp_path: Path) -> None:
    candidates = (
        InterMethodCandidate("CO-HR(S,cum)", "level_1", "CO-HR(S,cum)", None),
        InterMethodCandidate("EG(Pop)", "level_1", "EG(Pop)", None),
    )

    frame = build_inter_method_tree_frame(candidates=candidates)
    rendered = render_inter_method_tree(
        frame=frame,
        figure_base_path=tmp_path / "l1_tree",
        output_format="png",
        dpi=10,
    )

    assert "m_s" not in frame["label"].tolist()
    assert frame.loc[frame["parent_id"].eq("root"), "label"].tolist() == ["CO", "EG"]
    assert rendered[0].exists()


def test_inter_method_public_weights_feed_custom_probabilities(allocation_dummy_repo) -> None:
    project_name = "asocc_uncertainty_inter_method_weights"
    base_args = {
        **_base_args(project_name=project_name),
        "method_plan": "pairs",
        "one_step_methods": None,
        "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
    }
    report = write_asocc_weight_template(base_asocc_args=base_args)
    custom_path = report.tree_csv_path.parent / "weights__custom_v1.csv"
    guide_text = report.guide_path.read_text(encoding="utf-8")
    candidates = candidates_from_scope(base_asocc_args=base_args)
    custom_probabilities = np.array([1.0, *([0.0] * (len(candidates) - 1))])
    write_inter_method_tree_csv(
        path=custom_path,
        frame=build_inter_method_tree_frame(
            candidates=candidates,
            probabilities=custom_probabilities,
        ),
    )

    preview = preview_asocc_weight_tree(
        base_asocc_args=base_args,
        version_name="custom_v1",
    )
    loaded_probabilities = load_inter_method_tree_probabilities(
        candidates=candidates,
        custom_path=custom_path,
    )
    labels_with_external = tuple(
        candidate.candidate_label
        for candidate in candidates_from_scope(
            base_asocc_args=base_args,
            external_method={"one_step_methods": ["CO(S)"]},
        )
    )
    rows = pd.DataFrame(
        {
            "l1_l2_method": list(report.candidates),
            "r_c": ["FR"] * len(report.candidates),
            "s_p": ["Electricity"] * len(report.candidates),
            "year": [2030] * len(report.candidates),
            ASOCC_VALUE_COLUMN: list(range(len(report.candidates))),
        }
    )
    plan = build_inter_method_plan(
        loaded=_loaded(
            rows=rows,
            base_asocc_args=base_args,
            proj_base=allocation_dummy_repo.repo_root / f"{project_name}",
        ),
        parameters={"mode": "custom", "version_name": "custom_v1"},
    )

    assert preview.probabilities[0] == 1.0
    np.testing.assert_allclose(loaded_probabilities, custom_probabilities)
    assert inter_method_tree_version_name(parameters=None) == "equal_weight_default"
    assert report.probabilities == (0.5, 0.5)
    assert report.tree_csv_path.name == "equal_weights.csv"
    assert report.tree_csv_path.parent.name == "preview_inter_method_weights"
    assert report.guide_path.parent == report.tree_csv_path.parent
    assert preview.tree_csv_path == custom_path
    assert report.figure_paths[0].parent == report.tree_csv_path.parent
    assert report.figure_paths[0].stem == "probability_tree__equal_weights"
    assert preview.figure_paths[0].parent == report.tree_csv_path.parent
    assert preview.figure_paths[0].stem == "probability_tree__custom_v1"
    assert guide_text.strip()
    assert "PR-HR(Ecap,cum^{PBA})" in guide_text
    assert "CO(S)" in guide_text
    assert preview.guide_path.exists()
    assert "CO(S)" in labels_with_external
    assert plan.selection_probabilities.tolist()[0] == 1.0
    invalid_inter_method_parameters = (
        ({"mode": "bad"}, "mode"),
        ({"mode": "equal_weight", "version_name": "custom_v1"}, "Unsupported"),
        ({"mode": "custom"}, "version_name"),
        ({"mode": "custom", "version_name": ""}, "non empty"),
        ({"mode": "custom", "version_name": "equal_weight_default"}, "default token"),
        ({"mode": "custom", "version_name": "bad/path"}, "letters, digits, and underscores"),
        ({"mode": "custom", "version_name": "custom_v1", "extra": 1}, "Unsupported"),
    )
    for parameters, _message in invalid_inter_method_parameters:
        with pytest.raises(ValueError):
            build_inter_method_plan(
                loaded=_loaded(
                    rows=rows,
                    base_asocc_args=base_args,
                    proj_base=allocation_dummy_repo.repo_root / f"{project_name}",
                ),
                parameters=parameters,
            )

    for frame, _message in (
        (
            pd.read_csv(custom_path).iloc[:-1],
            "topology",
        ),
        (
            pd.read_csv(custom_path).assign(
                edge_weight=lambda frame: [-0.1, *frame["edge_weight"].tolist()[1:]]
            ),
            "between 0 and 1",
        ),
        (
            pd.read_csv(custom_path).assign(edge_weight=0.25),
            "sibling edge weights",
        ),
    ):
        frame.to_csv(custom_path, index=False)
        with pytest.raises(ValueError):
            preview_asocc_weight_tree(base_asocc_args=base_args, version_name="custom_v1")


def test_inter_method_tree_supports_l1_scope_and_missing_custom_file(
    allocation_dummy_repo,
) -> None:
    base_args = {
        "project_name": "asocc_uncertainty_l1_method_tree",
        "source": "oecd_v2025",
        "fu_code": "L1.a",
        "years": [2005],
        "method_plan": "default",
        "l1_methods": ["EG(Pop)"],
    }

    report = write_asocc_weight_template(
        base_asocc_args=base_args,
        external_method={"l1_methods": ["CO-HR(S,cum)"]},
        figure_format={"format": "png", "dpi": 10},
    )

    assert report.tree_csv_path.exists()
    assert set(report.candidates) == {"CO-HR(S,cum)", "EG(Pop)"}
    with pytest.raises(ValueError):
        write_asocc_weight_template(
            base_asocc_args=base_args,
            external_method={"l1_methods": ["EG(Pop)"]},
        )
    with pytest.raises(ValueError):
        preview_asocc_weight_tree(
            base_asocc_args=base_args,
            version_name="missing",
            external_method={"l1_methods": ["CO-HR(S,cum)"]},
        )


def test_sampling_uses_inter_method_compact_owner() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["PR(GDPcap)_UT(GVAa)", "EG(Pop)_UT(GVAa)", "UT(TD)"],
            "l1_method": ["PR(GDPcap)", "EG(Pop)", None],
            "l2_method": ["UT(GVAa)", "UT(GVAa)", "UT(TD)"],
            "r_c": ["FR", "FR", "FR"],
            "s_p": ["Electricity", "Electricity", "Electricity"],
            "year": [2030, 2030, 2030],
            "l2_reuse_year": [None, None, None],
            ASOCC_VALUE_COLUMN: [0.2, 0.4, 0.6],
        }
    )
    loaded = _loaded(rows=rows)
    sources = SourceActivationPlan(
        sources=(ActiveSource(name="inter_method_uncertainty", parameters={}),)
    )
    external_plan = ExternalAsoccRowsPlan()
    plan = build_inter_method_plan(loaded=loaded, parameters={})
    execution_plan = build_inter_method_execution_plan(
        loaded=loaded,
        inter_method_plan=plan,
        sources=sources,
        external_plan=external_plan,
        lcia_plan=None,
        projection_plan=None,
    )

    sparse_batch = sample_sparse_inter_method_batch(
        loaded=loaded,
        inter_method_plan=plan,
        execution_plan=execution_plan,
        inter_mrio_plan=None,
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=17),
        sources=sources,
        identity=None,
    )

    assert sparse_batch.run_indices.tolist() == [0, 1]
    assert sparse_batch.identity.columns.tolist() == [
        "public_row_id",
        "l1_l2_method",
        "l1_method",
        "l2_method",
        "r_c",
        "s_p",
        "year",
    ]
    assert sparse_batch.sparse_rows.run_index.tolist() == [0, 1]
    assert set(sparse_batch.sparse_rows.public_row_id.tolist()).issubset({0, 1, 2})
    assert set(sparse_batch.sparse_rows.values.tolist()).issubset({0.2, 0.4, 0.6})


def test_inter_method_reuses_inter_mrio_matches_for_selected_branches() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)"],
            "l2_method": ["UT(TD)"],
            "r_c": ["FR"],
            "s_p": ["Electricity"],
            "year": [2030],
            ASOCC_TIME_ROUTE_COLUMN: [ASOCC_TIME_ROUTE_HISTORICAL],
            ASOCC_VALUE_COLUMN: [0.2],
        }
    )
    alternate = rows.assign(**{ASOCC_VALUE_COLUMN: [0.8]})
    loaded = _loaded(rows=rows)
    sources = SourceActivationPlan(
        sources=(
            ActiveSource(name="inter_method_uncertainty", parameters={}),
            ActiveSource(name="inter_mrio_uncertainty", parameters={}),
        )
    )
    plan = InterMethodPlan(
        candidates=(),
        candidate_labels=("UT(TD)",),
        selection_probabilities=np.array([1.0], dtype=np.float64),
        tree_frame=pd.DataFrame(),
        source_method_row=_source_method_row(),
    )
    execution_plan = build_inter_method_execution_plan(
        loaded=loaded,
        inter_method_plan=plan,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        lcia_plan=None,
        projection_plan=None,
    )
    inter_mrio_plan = InterMrioPlan(
        alternate_source="alternate",
        alternate_loaded=_loaded(rows=alternate),
        alternate_projection_plan=None,
        route_report=inter_mrio_route_report(main_rows=rows, alternate_rows=alternate),
        source_method_row=_source_method_row(),
    )
    matches = inter_method_inter_mrio_matches_by_branch(
        execution_plan=execution_plan,
        inter_mrio_plan=inter_mrio_plan,
    )

    sparse_batch = sample_sparse_inter_method_batch(
        loaded=loaded,
        inter_method_plan=plan,
        execution_plan=execution_plan,
        inter_mrio_plan=inter_mrio_plan,
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=1, rng_seed=17),
        sources=sources,
        identity=None,
        inter_mrio_matches_by_branch=matches,
    )

    assert set(matches) == {"UT(TD)"}
    alpha = uniform_by_run_index(
        stream_name="asocc.inter_mrio.alpha",
        run_indices=np.array([0], dtype=np.int64),
    )
    np.testing.assert_allclose(sparse_batch.sparse_rows.values, 0.2 + alpha * (0.8 - 0.2))


def test_inter_method_sampling_helpers_cover_empty_and_grouped_assignments() -> None:
    empty_offsets, empty_total = _sparse_row_offsets(
        selected=np.asarray([], dtype=object),
        row_counts_by_branch={},
    )
    assert empty_offsets.size == 0
    assert empty_total == 0

    offsets, total = _sparse_row_offsets(
        selected=np.asarray(["a", "b", "a"], dtype=object),
        row_counts_by_branch={"a": 2, "b": 1},
    )
    np.testing.assert_array_equal(offsets, np.asarray([0, 2, 3], dtype=np.int64))
    assert total == 5

    output = np.full((2, 2), np.nan, dtype=np.float64)
    _assign_branch_summary_values(
        output=output,
        run_positions=np.asarray([0, 1], dtype=np.int64),
        public_group_ids=np.asarray([0, 0, 1], dtype=np.int64),
        branch_values=np.asarray([[1.0, 3.0, np.nan], [np.nan, np.nan, 5.0]]),
    )
    np.testing.assert_allclose(output[0], [2.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(output[1], [np.nan, 5.0], equal_nan=True)

    run_index = np.full(total, -1, dtype=np.int64)
    public_row_id = np.full(total, -1, dtype=np.int64)
    values = np.full(total, np.nan, dtype=np.float64)
    _assign_sparse_branch_rows(
        run_index=run_index,
        public_row_id=public_row_id,
        values=values,
        row_offsets=offsets,
        run_positions=np.asarray([], dtype=np.int64),
        branch_run_indices=np.asarray([], dtype=np.int64),
        branch_public_row_id=np.asarray([7], dtype=np.int64),
        branch_values=np.empty((0, 1), dtype=np.float64),
    )
    _assign_sparse_branch_rows(
        run_index=run_index,
        public_row_id=public_row_id,
        values=values,
        row_offsets=offsets,
        run_positions=np.asarray([0, 2], dtype=np.int64),
        branch_run_indices=np.asarray([10, 12], dtype=np.int64),
        branch_public_row_id=np.asarray([7, 8], dtype=np.int64),
        branch_values=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
    )
    np.testing.assert_array_equal(run_index[[0, 1, 3, 4]], [10, 10, 12, 12])
    np.testing.assert_array_equal(public_row_id[[0, 1, 3, 4]], [7, 8, 7, 8])
    np.testing.assert_allclose(values[[0, 1, 3, 4]], [0.1, 0.2, 0.3, 0.4])


def test_sparse_inter_method_memory_blocks_skip_csv_working_buffer_for_parquet() -> None:
    runtime = UncertaintyRuntimeRequest(
        family="asocc",
        mode="fixed",
        output_format="parquet",
        n_runs=4,
        max_runs=0,
        batch_size=4,
        rtol=0.05,
        stable_runs=4,
        convergence_statistics=("mean",),
    )
    blocks = _asocc_sampling_memory_blocks(
        row_count=3,
        sparse_inter_method=True,
        public_row_count=2,
        runtime=runtime,
    )

    assert "asocc_sparse_csv_render_working_bytes" not in {block.name for block in blocks}


def test_inter_method_summary_matrix_batch_collapses_public_groups() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FD)", "UT(GVAa)"],
            "l2_method": ["UT(FD)", "UT(GVAa)"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2030, 2030],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    loaded = _loaded(rows=rows)
    sources = SourceActivationPlan(
        sources=(ActiveSource(name="inter_method_uncertainty", parameters={}),)
    )
    plan = build_inter_method_plan(loaded=loaded, parameters={})
    execution_plan = build_inter_method_execution_plan(
        loaded=loaded,
        inter_method_plan=plan,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        lcia_plan=None,
        projection_plan=None,
    )
    identity, values = sample_inter_method_summary_matrix_batch(
        loaded=loaded,
        inter_method_plan=plan,
        execution_plan=execution_plan,
        inter_mrio_plan=None,
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=17),
        sources=sources,
        source_units=SourceUnitIntervalSamples(values_by_source={}),
    )

    assert not identity.empty
    assert values.shape == (2, len(identity))
    assert np.isfinite(values).any()


def test_inter_method_reuses_run_level_lcia_plan_for_selected_branches() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FDa)", "UT(GVAa)"],
            "l2_method": ["UT(FDa)", "UT(GVAa)"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2030, 2030],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    direct = rows.iloc[[0]].assign(
        _lower_bound=[0.1],
        _upper_bound=[0.3],
        _shared_u_key=["direct_lcia"],
    )
    lcia_plan = LCIAPlan(
        public_columns=tuple(rows.columns),
        passthrough_rows=rows.iloc[[1]].copy(),
        direct_rows=direct,
        direct_block=lcia_sample_block(template=direct),
        combined_routes=(),
        source_method_rows=(),
    )
    loaded = _loaded(rows=rows)
    sources = SourceActivationPlan(
        sources=(
            ActiveSource(name="inter_method_uncertainty", parameters={}),
            ActiveSource(name="lcia_uncertainty", parameters={}),
        )
    )
    plan = build_inter_method_plan(loaded=loaded, parameters={})
    execution_plan = build_inter_method_execution_plan(
        loaded=loaded,
        inter_method_plan=plan,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        lcia_plan=lcia_plan,
        projection_plan=None,
    )

    branch_lcia = {branch.label: branch.lcia_plan is not None for branch in execution_plan.branches}

    assert branch_lcia == {"UT(FDa)": True, "UT(GVAa)": False}
    assert (
        batch_row_count(
            loaded=loaded,
            inter_method_execution_plan=execution_plan,
            inter_mrio_plan=None,
            lcia_plan=lcia_plan,
            projection_plan=None,
            sources=sources,
            external_plan=ExternalAsoccRowsPlan(),
        )
        == 1
    )


def test_inter_method_branch_keeps_inactive_inner_source_axes_stable() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)", "UT(GVAa)"],
            "l2_method": ["UT(TD)", "UT(GVAa)"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2030, 2030],
            "l2_reuse_year": [None, None],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    loaded = _loaded(
        rows=rows,
        base_asocc_args={
            "project_name": "stable_inner_axes",
            "source": "exiobase_396_ixi",
            "agg_reg": False,
            "agg_sec": True,
            "agg_version": None,
            "fu_code": "L2.c.b",
        },
    )
    plan = build_inter_method_plan(loaded=loaded, parameters={})
    projection_plan = build_projection_plan(loaded=loaded)
    sources = SourceActivationPlan(
        sources=(
            ActiveSource(name="inter_method_uncertainty", parameters={}),
            ActiveSource(name="lcia_uncertainty", parameters={}),
            ActiveSource(name="projection_uncertainty", parameters={}),
        )
    )
    execution_plan = build_inter_method_execution_plan(
        loaded=loaded,
        inter_method_plan=plan,
        sources=sources,
        external_plan=ExternalAsoccRowsPlan(),
        lcia_plan=None,
        projection_plan=projection_plan,
    )
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=17)
    assert (
        batch_row_count(
            loaded=loaded,
            inter_method_execution_plan=execution_plan,
            inter_mrio_plan=None,
            lcia_plan=None,
            projection_plan=projection_plan,
            sources=sources,
            external_plan=ExternalAsoccRowsPlan(),
        )
        == 1
    )

    selected_indices = sample_projection_indices(plan=projection_plan, batch=batch)
    dense_values = projection_value_matrix_for_indices(
        plan=projection_plan,
        batch=batch,
        selected_indices=selected_indices,
    )
    projected_template, projected_values = apply_projection_uncertainty_to_matrix(
        template=rows,
        values=np.broadcast_to([0.2, 0.4], (2, 2)),
        plan=projection_plan,
        batch=batch,
        selected_indices=selected_indices,
    )
    sparse_batch = sample_sparse_inter_method_batch(
        loaded=loaded,
        inter_method_plan=plan,
        execution_plan=execution_plan,
        inter_mrio_plan=None,
        batch=batch,
        sources=sources,
        identity=None,
    )

    np.testing.assert_allclose(dense_values, np.broadcast_to([0.2, 0.4], (2, 2)))
    assert "l2_reuse_year" not in projected_template.columns
    np.testing.assert_allclose(projected_values, np.broadcast_to([0.2, 0.4], (2, 2)))
    assert sorted(sparse_batch.sparse_rows.run_index.tolist()) == [0, 1]
    assert set(sparse_batch.sparse_rows.values.tolist()).issubset({0.2, 0.4})


def test_inter_method_public_identity_collapses_reference_year_axis() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["AR(E^{CBA_FD})", "AR(E^{CBA_FD})"],
            "year": [2030, 2030],
            "reference_year": [2019, 2020],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    identity = public_identity_for_sampling_plan(
        loaded=_loaded(rows=rows),
        sources=SourceActivationPlan(
            sources=(ActiveSource(name=REFERENCE_YEAR_SOURCE, parameters={}),)
        ),
        external_plan=ExternalAsoccRowsPlan(),
        lcia_plan=None,
        projection_plan=None,
        reference_axis=None,
    )

    assert "reference_year" not in identity.columns
    assert len(identity) == 1


def test_lcia_target_detection_includes_combined_l2_lcia_owner() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["EG(Pop)_AR(E^{CBA_TD})"],
            "l1_method": ["EG(Pop)"],
            "l2_method": ["AR(E^{CBA_TD})"],
            "r_c": ["FR"],
            "s_p": ["Electricity"],
            "year": [2030],
            ASOCC_VALUE_COLUMN: [0.2],
        }
    )
    loaded = replace(
        _loaded(
            rows=rows,
            base_asocc_args={"fu_code": "L2.c.b"},
        ),
        asocc_scope=cast(
            AsoccScope,
            SimpleNamespace(
                combined=(
                    ("UT(GVAa)", "PR(GDPcap)"),
                    ("AR(E^{CBA_TD})", "EG(Pop)"),
                )
            ),
        ),
    )

    assert lcia_uncertainty_has_targets(loaded=loaded)

    non_lcia_rows = rows.assign(
        l1_l2_method="EG(Pop)_UT(GVAa)",
        l2_method="UT(GVAa)",
    )
    non_lcia_loaded = replace(
        loaded,
        rows=non_lcia_rows,
        asocc_scope=cast(
            AsoccScope,
            SimpleNamespace(combined=(("UT(GVAa)", "EG(Pop)"),)),
        ),
    )
    assert not lcia_uncertainty_has_targets(loaded=non_lcia_loaded)


def test_sampling_applies_lcia_projection_and_reference_sources() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FDa)", "UT(FDa)", "UT(FDa)"],
            "r_c": ["FR", "FR", "FR"],
            "s_p": ["Electricity", "Electricity", "Electricity"],
            "year": [2020, 2030, 2030],
            "l2_reuse_year": [None, 2005, 2006],
            ASOCC_VALUE_COLUMN: [0.1, 0.2, 0.4],
        }
    )
    direct = rows.assign(
        _lower_bound=[0.05, 0.1, 0.3],
        _upper_bound=[0.15, 0.2, 0.5],
        _shared_u_key=["projection_passthrough", "projection_2005", "projection_2006"],
    )
    lcia_plan = LCIAPlan(
        public_columns=tuple(rows.columns),
        passthrough_rows=rows.iloc[:0].copy(),
        direct_rows=direct,
        direct_block=lcia_sample_block(template=direct),
        combined_routes=(),
        source_method_rows=(),
    )
    projection_plan = build_projection_plan(loaded=_loaded(rows=rows))
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=17)
    projection_indices = sample_projection_indices(plan=projection_plan, batch=batch)
    sampled_projection = projection_value_matrix_for_indices(
        plan=projection_plan,
        batch=batch,
        selected_indices=projection_indices,
    )

    lcia_projection_identity, _runs, lcia_projection = sample_compact_batch(
        loaded=_loaded(rows=rows),
        inter_mrio_plan=None,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
        batch=batch,
        sources=SourceActivationPlan(
            sources=(
                ActiveSource(name="lcia_uncertainty", parameters={}),
                ActiveSource(name="projection_uncertainty", parameters={}),
            )
        ),
        external_plan=ExternalAsoccRowsPlan(),
    )
    projection_identity, _runs, projection = sample_compact_batch(
        loaded=_loaded(rows=rows),
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=projection_plan,
        batch=batch,
        sources=SourceActivationPlan(
            sources=(ActiveSource(name="projection_uncertainty", parameters={}),)
        ),
        external_plan=ExternalAsoccRowsPlan(),
    )
    reference_identity, _runs, reference = sample_compact_batch(
        loaded=_loaded(
            rows=pd.DataFrame(
                {
                    "l1_l2_method": ["AR(E^{CBA_FD})", "AR(E^{CBA_FD})"],
                    "year": [2030, 2030],
                    "reference_year": [2019, 2020],
                    ASOCC_VALUE_COLUMN: [0.2, 0.4],
                }
            )
        ),
        inter_mrio_plan=None,
        lcia_plan=None,
        projection_plan=None,
        batch=batch,
        sources=SourceActivationPlan(
            sources=(ActiveSource(name="reference_year_uncertainty", parameters={}),)
        ),
        external_plan=ExternalAsoccRowsPlan(),
    )

    assert "l2_reuse_year" not in lcia_projection_identity.columns
    assert "l2_reuse_year" not in projection_identity.columns
    assert "reference_year" not in reference_identity.columns
    assert sampled_projection.shape == (2, 2)
    assert lcia_projection.shape == (2, 2)
    assert projection.shape == (2, 2)
    assert reference.shape == (2, 1)


def test_compact_inter_mrio_template_uses_lcia_and_projection_axes() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FDa)", "UT(FDa)"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2020, 2030],
            "l2_reuse_year": [None, 2020],
            ASOCC_VALUE_COLUMN: [0.1, 0.2],
        }
    )
    direct = rows.assign(
        _lower_bound=[0.05, 0.1],
        _upper_bound=[0.15, 0.3],
        _shared_u_key=["historical", "projection"],
    )
    lcia_plan = LCIAPlan(
        public_columns=tuple(rows.columns),
        passthrough_rows=rows.iloc[:0].copy(),
        direct_rows=direct,
        direct_block=lcia_sample_block(template=direct),
        combined_routes=(),
        source_method_rows=(),
    )
    loaded = _loaded(rows=rows)
    projection_plan = build_projection_plan(loaded=loaded)

    lcia_template = _compact_inter_mrio_input_template(
        loaded=loaded,
        lcia_plan=lcia_plan,
        projection_plan=None,
    )
    projected_lcia_template = _compact_inter_mrio_input_template(
        loaded=loaded,
        lcia_plan=lcia_plan,
        projection_plan=projection_plan,
    )
    projected_template = _compact_inter_mrio_input_template(
        loaded=loaded,
        lcia_plan=None,
        projection_plan=projection_plan,
    )

    assert len(lcia_template) == len(rows)
    assert len(projected_lcia_template) == len(rows)
    assert len(projected_template) == len(rows)


def test_lcia_sobol_unit_values_replace_shared_random_matrix() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FDa)", "UT(FDa)"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2030, 2030],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    direct = rows.assign(
        _lower_bound=[0.0, 0.2],
        _upper_bound=[1.0, 1.2],
        _shared_u_key=["first_key", "second_key"],
    )
    lcia_plan = LCIAPlan(
        public_columns=tuple(rows.columns),
        passthrough_rows=rows.iloc[:0].copy(),
        direct_rows=direct,
        direct_block=lcia_sample_block(template=direct),
        combined_routes=(),
        source_method_rows=(),
    )
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=17)

    shared_u = lcia_shared_u_for_plan(
        plan=lcia_plan,
        batch=batch,
        unit_values=np.array([0.25, 0.75], dtype=np.float64),
    )

    assert shared_u is not None
    assert set(shared_u.key_positions) == {"first_key", "second_key"}
    np.testing.assert_allclose(shared_u.values, [[0.25, 0.25], [0.75, 0.75]])
    passthrough_plan = LCIAPlan(
        public_columns=tuple(rows.columns),
        passthrough_rows=rows,
        direct_rows=rows.iloc[:0].copy(),
        direct_block=None,
        combined_routes=(),
        source_method_rows=(),
    )
    assert lcia_sampling_memory_row_counts(plan=passthrough_plan) == (0, 0)


def test_sobol_compact_evaluation_uses_source_units_without_method_axis() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FD)", "UT(FD)", "UT(FD)", "UT(FD)"],
            "r_c": ["FR", "FR", "FR", "FR"],
            "s_p": ["Electricity", "Electricity", "Electricity", "Electricity"],
            "year": [2030, 2030, 2030, 2030],
            "l2_reuse_year": [2019, 2020, 2019, 2020],
            "reference_year": [2018, 2018, 2020, 2020],
            ASOCC_VALUE_COLUMN: [1.0, 2.0, 3.0, 4.0],
        }
    )
    loaded = _loaded(rows=rows)
    sources = SourceActivationPlan(
        sources=(
            ActiveSource(name="projection_uncertainty", parameters={}),
            ActiveSource(name="reference_year_uncertainty", parameters={}),
        )
    )

    identity, values = evaluate_asocc_sobol_units(
        context=AsoccSobolEvaluationContext(
            loaded=loaded,
            source_names=("projection_uncertainty", "reference_year_uncertainty"),
            inter_method_plan=None,
            inter_method_execution_plan=None,
            inter_mrio_plan=None,
            lcia_plan=None,
            projection_plan=build_projection_plan(loaded=loaded),
            sources=sources,
            external_plan=ExternalAsoccRowsPlan(),
            selected_years=(2030,),
            requested_ssp_scenarios=(),
        ),
        units=np.array([[0.0, 0.0], [0.99, 0.99]], dtype=np.float64),
    )

    assert "l2_reuse_year" not in identity.columns
    assert "reference_year" not in identity.columns
    assert identity["public_row_id"].tolist() == [0]
    np.testing.assert_allclose(values, [[1.0], [4.0]])


def test_sobol_request_context_uses_reference_and_projection_sources(
    allocation_dummy_repo,
) -> None:
    context = build_asocc_sobol_evaluation_context_from_request(
        base_asocc_args={
            "project_name": "asocc_sobol_request_reference_projection",
            "source": "exiobase_396_ixi",
            "years": [2030],
            "reference_years": [2005, 2006],
            "fu_code": "L2.a.b",
            "method_plan": "pairs",
            "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FDa)"],
            "lcia_method": "gwp100_lcia",
            "r_p": ["FR"],
            "s_p": ["D"],
            "r_f": ["FR"],
            "l1_reg_aggreg": "pre",
            "ssp_scenario": ["SSP2"],
            "projection_mode": "historical_reuse",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005, 2006],
        },
        uncertainty_config={
            "projection_uncertainty": {},
            "reference_year_uncertainty": {},
        },
        external_method=None,
        sobol_plan=SobolPlan(
            enabled=True,
            mode="fixed",
            n_base_samples=4,
            max_base_samples=4,
            rtol=0.05,
        ),
    )

    assert context.source_names == (
        "projection_uncertainty",
        "reference_year_uncertainty",
    )
    assert context.selected_years == (2030,)


def test_sobol_context_drops_inactive_lcia_and_validates_unsampled_axes(
    allocation_dummy_repo,
) -> None:
    no_lcia_context = build_asocc_sobol_evaluation_context(
        loaded=_loaded(
            rows=pd.DataFrame({"year": [2030], ASOCC_VALUE_COLUMN: [1.0]}),
        ),
        inter_mrio_plan=None,
        sources=SourceActivationPlan(
            sources=(ActiveSource(name="lcia_uncertainty", parameters={}),)
        ),
        external_plan=ExternalAsoccRowsPlan(),
        sobol_plan=SobolPlan(
            enabled=True,
            mode="fixed",
            n_base_samples=4,
            max_base_samples=4,
            rtol=0.05,
        ),
        selected_years=(2030,),
    )
    assert no_lcia_context.source_names == ()

    inter_method_context = build_asocc_sobol_evaluation_context_from_request(
        base_asocc_args={
            "project_name": "asocc_sobol_request_inter_method_only",
            "source": "exiobase_396_ixi",
            "years": [2030],
            "fu_code": "L2.a.b",
            "r_p": ["FR"],
            "s_p": ["D"],
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "l1_reg_aggreg": "pre",
            "ssp_scenario": ["SSP2"],
            "projection_mode": "historical_reuse",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005],
        },
        uncertainty_config={"inter_method_uncertainty": {}},
        external_method=None,
        sobol_plan=SobolPlan(
            enabled=True,
            mode="fixed",
            n_base_samples=4,
            max_base_samples=4,
            rtol=0.05,
        ),
    )
    assert inter_method_context.source_names == ("inter_method_uncertainty",)


def test_reference_year_owner_samples_by_run_index() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": [
                "AR(E^{CBA_FD})",
                "AR(E^{CBA_FD})",
                "AR(E^{CBA_FD})",
                "AR(E^{CBA_FD})",
            ],
            "year": [2030, 2020, 2030, 2030],
            "reference_year": [2035, 2019, 2019, 2025],
            ASOCC_VALUE_COLUMN: [0.9, 0.2, 0.3, 0.5],
        }
    )
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=10, rng_seed=17)
    no_axis = pd.DataFrame({"year": [2030], ASOCC_VALUE_COLUMN: [0.2]})
    invariant_rows = pd.DataFrame(
        {
            "year": [2030],
            "reference_year": [None],
            ASOCC_VALUE_COLUMN: [0.2],
        }
    )

    same_template, same_values = apply_reference_year_uncertainty_to_matrix(
        template=no_axis,
        values=np.array([[0.2]]),
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=1, rng_seed=17),
    )
    invariant_template, invariant_values = apply_reference_year_uncertainty_to_matrix(
        template=invariant_rows,
        values=np.array([[0.2]]),
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=1, rng_seed=17),
    )
    template, values = apply_reference_year_uncertainty_to_matrix(
        template=rows,
        values=np.broadcast_to(
            rows[ASOCC_VALUE_COLUMN].to_numpy(dtype=np.float64),
            (batch.n_runs, len(rows)),
        ),
        batch=batch,
    )
    method_row = reference_year_source_method_row(loaded=_loaded(rows=rows))

    assert admissible_reference_year_rows(frame=no_axis).equals(no_axis)
    assert not reference_year_uncertainty_has_targets(rows=no_axis)
    assert not reference_year_uncertainty_has_targets(rows=invariant_rows)
    assert reference_year_uncertainty_has_targets(rows=rows)
    assert same_template.equals(no_axis)
    np.testing.assert_allclose(same_values, [[0.2]])
    assert "reference_year" not in invariant_template.columns
    np.testing.assert_allclose(invariant_values, [[0.2]])
    assert admissible_reference_year_rows(frame=rows)["reference_year"].tolist() == [
        2019,
        2019,
        2025,
    ]
    assert "reference_year" not in template.columns
    assert set(values[:, 0].tolist()) == {0.2}
    assert set(values[:, 1].tolist()).issubset({0.3, 0.5})
    assert method_row.source_name == "reference_year_uncertainty"


def test_summary_identity_groups_source_sampled_axes() -> None:
    identity = pd.DataFrame(
        {
            "public_row_id": [0, 1, 2],
            "l1_l2_method": ["A", "B", "B"],
            "l1_method": ["A1", "B1", "B1"],
            "l2_method": ["A2", "B2", "B2"],
            "reference_year": [2019, 2019, 2020],
            "l2_reuse_year": [2018, 2018, 2018],
            "year": [2030, 2030, 2030],
        }
    )
    summary_identity, groups = summary_identity_groups(
        identity=identity,
        sources=SourceActivationPlan(
            sources=(
                ActiveSource(name="inter_method_uncertainty", parameters={}),
                ActiveSource(name="projection_uncertainty", parameters={}),
                ActiveSource(name="reference_year_uncertainty", parameters={}),
            )
        ),
    )
    values = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, 7.0]])

    collapsed = collapse_values_to_summary_groups(values=values, public_row_groups=groups)

    assert summary_identity.columns.tolist() == [
        "l1_l2_method",
        "l1_method",
        "l2_method",
        "year",
        ASOCC_SUMMARY_SCOPE_COLUMN,
    ]
    assert summary_identity[ASOCC_SUMMARY_SCOPE_COLUMN].tolist() == [
        ASOCC_SUMMARY_SCOPE_PER_METHOD,
        ASOCC_SUMMARY_SCOPE_PER_METHOD,
        ASOCC_SUMMARY_SCOPE_INTER_METHOD,
    ]
    assert groups == (("0",), ("1", "2"), ("0", "1", "2"))
    np.testing.assert_allclose(collapsed, [[1.0, 3.0, 2.0], [np.nan, 6.0, 6.0]])

    unchanged_identity, unchanged_groups = summary_identity_groups(
        identity=identity.drop(columns=["reference_year", "l2_reuse_year"]),
        sources=SourceActivationPlan(
            sources=(
                ActiveSource(name="projection_uncertainty", parameters={}),
                ActiveSource(name="reference_year_uncertainty", parameters={}),
            )
        ),
    )
    assert "public_row_id" in unchanged_identity.columns
    assert unchanged_groups == (("0",), ("1",), ("2",))


def test_asocc_sobol_year_scope_and_selector_summary() -> None:
    default_plan = SobolPlan(
        enabled=True,
        mode="fixed",
        n_base_samples=4,
        max_base_samples=4,
        rtol=0.05,
    )
    explicit_plan = replace(default_plan, sobol_years=(2025, 2019))

    assert selected_sobol_years(
        plan=default_plan,
        requested_years=tuple(range(2019, 2035)),
    ) == (2019, 2034)
    assert selected_sobol_years(
        plan=explicit_plan,
        requested_years=(2019, 2020, 2025),
    ) == (2019, 2025)
    source = ExternalMonteCarloRowsSource(
        selection=ExternalMethodSelection(
            level="level_2",
            fu_code="L2.c.b",
            l1_method="CO(S)",
            l2_method="UT(FD)",
        ),
        file_selections=(
            ExternalMonteCarloFileSelection(
                path=Path("external.csv"),
                lcia_method=None,
                requested_years=(2019, 2024, 2029, 2034),
                ssp_scenario_options_by_year=None,
            ),
        ),
        run_indices=(0, 1),
    )
    scoped_external = external_plan_for_years(
        plan=ExternalAsoccRowsPlan(monte_carlo_sources=(_materialized_external_source(source),)),
        years=(2019, 2034),
    )

    assert external_plan_for_years(plan=ExternalAsoccRowsPlan(), years=(2019,)) == (
        ExternalAsoccRowsPlan()
    )
    assert scoped_external.monte_carlo_sources[0].file_selections[0].requested_years == (
        2019,
        2034,
    )
    assert not external_plan_for_years(
        plan=ExternalAsoccRowsPlan(monte_carlo_sources=(_materialized_external_source(source),)),
        years=(2020,),
    ).monte_carlo_sources
    appended_template, appended_values = append_external_monte_carlo_matrix(
        template=pd.DataFrame({"year": [2019], "value": [0.5]}),
        values=np.array([[0.5], [0.6]], dtype=np.float64),
        plan=ExternalAsoccRowsPlan(monte_carlo_sources=(_materialized_external_source(source),)),
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=9),
        unit_values=np.array([0.0, 0.999999], dtype=np.float64),
    )
    assert len(appended_template) == 5
    assert appended_values.shape == (2, 5)

    identity = pd.DataFrame(
        {
            "public_row_id": [0, 1, 2, 3, 4],
            "r_c": ["FR", "FR", "FR", "FR", "FR"],
            "s_p": ["Electricity", "Electricity", "Electricity", "Electricity", "Electricity"],
            "lcia_method": [
                "gwp100_lcia",
                "gwp100_lcia",
                "gwp100_lcia",
                "gwp100_lcia",
                "gwp100_lcia",
            ],
            "impact": [
                "climate change",
                "climate change",
                "climate change",
                "climate change",
                "climate change",
            ],
            "asocc_ssp_scenario": [None, "SSP1", "SSP5", None, "SSP1"],
            "year": [2030, 2030, 2030, 2020, 2030],
        }
    )
    summary = asocc_sobol_source_summary(
        identity=identity,
        dimension_names=("lcia_uncertainty", "projection_uncertainty"),
        estimates=SobolIndexEstimate(
            s1=np.array([[0.2, 0.3, 0.4, 0.5, 0.35], [0.8, 0.7, 0.6, 0.5, 0.65]]),
            st=np.array([[0.25, 0.35, 0.45, 0.55, 0.4], [0.9, 0.75, 0.65, 0.55, 0.7]]),
            variance=np.array([1.0, 2.0, 3.0, 4.0, 2.5]),
            s1_confidence_half_width=np.full((2, 5), 0.01),
            st_confidence_half_width=np.full((2, 5), 0.02),
            s1_resamples=np.array(
                [
                    [[0.19, 0.29, 0.39, 0.49, 0.34], [0.79, 0.69, 0.59, 0.49, 0.64]],
                    [[0.21, 0.31, 0.41, 0.51, 0.36], [0.81, 0.71, 0.61, 0.51, 0.66]],
                ]
            ),
            st_resamples=np.array(
                [
                    [[0.24, 0.34, 0.44, 0.54, 0.39], [0.89, 0.74, 0.64, 0.54, 0.69]],
                    [[0.26, 0.36, 0.46, 0.56, 0.41], [0.91, 0.76, 0.66, 0.56, 0.71]],
                ]
            ),
        ),
        confidence_level=0.95,
        requested_ssp_scenarios=("SSP1", "SSP5"),
    )
    selector_summary = summary.loc[summary["summary_level"].eq("selector")].reset_index(drop=True)
    lcia_method_summary = summary.loc[summary["summary_level"].eq("lcia_method")].reset_index(
        drop=True
    )

    assert {"r_c", "s_p", "lcia_method", "impact", "asocc_ssp_scenario", "year"}.issubset(
        summary.columns
    )
    assert set(summary["summary_level"]) == {"selector", "lcia_method"}
    mixed_2030 = selector_summary.loc[selector_summary["year"].eq(2030)].reset_index(drop=True)
    historical_2020 = selector_summary.loc[selector_summary["year"].eq(2020)].reset_index(drop=True)

    assert mixed_2030["asocc_ssp_scenario"].tolist() == [
        "SSP1",
        "SSP1",
        "SSP5",
        "SSP5",
    ]
    assert mixed_2030["output_count"].tolist() == [3, 3, 2, 2]
    assert mixed_2030["contains_ssp_invariant_outputs"].tolist() == [True] * 4
    assert mixed_2030["ssp_invariant_output_count"].tolist() == [1, 1, 1, 1]
    assert historical_2020["asocc_ssp_scenario"].isna().all()
    assert historical_2020["contains_ssp_invariant_outputs"].tolist() == [False, False]
    assert lcia_method_summary["impact"].isna().all()
    assert sorted(lcia_method_summary["year"].unique().tolist()) == [2020, 2030]

    plain_summary = asocc_sobol_source_summary(
        identity=pd.DataFrame({"public_row_id": [0], "year": [2030]}),
        dimension_names=("projection_uncertainty", "reference_year_uncertainty"),
        estimates=SobolIndexEstimate(
            s1=np.array([[0.3], [0.7]]),
            st=np.array([[0.35], [0.75]]),
            variance=np.array([2.0]),
            s1_confidence_half_width=np.full((2, 1), 0.01),
            st_confidence_half_width=np.full((2, 1), 0.02),
            s1_resamples=np.array([[[0.29], [0.69]], [[0.31], [0.71]]]),
            st_resamples=np.array([[[0.34], [0.74]], [[0.36], [0.76]]]),
        ),
        confidence_level=0.95,
        requested_ssp_scenarios=(),
    )

    assert plain_summary["summary_level"].tolist() == ["selector", "selector"]
    assert "impact" not in plain_summary.columns


def test_inter_mrio_sobol_plan_filters_alternate_years() -> None:
    rows = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FD)", "UT(FD)", "UT(FD)"],
            "l1_method": [None, None, None],
            "l2_method": ["UT(FD)", "UT(FD)", "UT(FD)"],
            "r_c": ["FR", "FR", "FR"],
            "s_p": ["Electricity", "Electricity", "Electricity"],
            "l2_reuse_year": [2005, 2005, 2006],
            "year": [2030, 2035, 2035],
            ASOCC_TIME_ROUTE_COLUMN: [
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
                ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
            ],
            ASOCC_VALUE_COLUMN: [0.2, 0.4, 0.5],
        }
    )
    alternate_loaded = replace(
        _loaded(
            rows=rows,
            base_asocc_args={"fu_code": "L2.c.b", "years": [2030, 2035]},
        ),
        requested_years=[2030, 2035],
    )
    plan = InterMrioPlan(
        alternate_source="alternate",
        alternate_loaded=alternate_loaded,
        alternate_projection_plan=None,
        route_report=InterMrioRouteReport(
            interpolated_years=(2030, 2035),
            skipped_years=(),
            skipped_route_pairs=(),
            skipped_scopes=(),
        ),
        source_method_row=_source_method_row(),
    )

    filtered = inter_mrio_plan_for_sobol_years(
        plan=plan,
        selected_years=(2035,),
        projection_active=True,
    )

    assert filtered.alternate_loaded.requested_years == [2035]
    assert filtered.alternate_loaded.base_asocc_args["years"] == [2035]
    assert filtered.alternate_loaded.rows["year"].tolist() == [2035, 2035]
    assert filtered.alternate_projection_plan is not None
    assert filtered.alternate_projection_plan.l2_reuse_years == (2005, 2006)


@pytest.mark.parametrize("output_format", ["csv_compact", "parquet"])
def test_sparse_parent_render_reader_streams_multiple_batches(
    tmp_path: Path,
    output_format: str,
) -> None:
    run_index = np.concatenate(
        [
            np.array([0], dtype=np.int64),
            np.ones(1_000_000, dtype=np.int64),
        ]
    )
    path = tmp_path / f"asocc_runs.{('csv' if output_format == 'csv_compact' else 'parquet')}"
    with SparseRunRowsWriter(path=path, output_format=output_format) as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=run_index,
                public_row_id=np.zeros(len(run_index), dtype=np.int64),
                values=np.ones(len(run_index), dtype=np.float64),
                value_column="asocc",
            ),
            batch_index=0,
        )

    chunks = list(iter_sparse_run_rows(path=path, output_format=output_format))

    assert [int(chunk.run_index[0]) for chunk in chunks] == [0, 1]
    assert [len(chunk.run_index) for chunk in chunks] == [1, 1_000_000]


def test_inter_method_supports_level_1_candidate_labels() -> None:
    rows = pd.DataFrame(
        {
            "l1_method": ["PR(GDPcap)", "EG(Pop)"],
            "r_f": ["FR", "FR"],
            "year": [2030, 2030],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )
    loaded = replace(
        _loaded(rows=rows, base_asocc_args={"fu_code": "L1.a"}), final_bucket="level_1"
    )

    plan = build_inter_method_plan(loaded=loaded, parameters={})

    assert plan.candidate_labels == ("EG(Pop)", "PR(GDPcap)")


def test_inter_method_supports_l2_method_candidate_labels() -> None:
    rows = pd.DataFrame(
        {
            "l2_method": ["UT(TD)", "UT(GVAa)"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity", "Electricity"],
            "year": [2030, 2030],
            ASOCC_VALUE_COLUMN: [0.2, 0.4],
        }
    )

    plan = build_inter_method_plan(loaded=_loaded(rows=rows), parameters={})

    assert plan.candidate_labels == ("UT(GVAa)", "UT(TD)")


def test_inter_method_row_labels_support_l2_method_only_rows() -> None:
    rows = pd.DataFrame({"l2_method": ["UT(TD)"]})

    labels = inter_method_row_labels(rows=rows)

    assert labels.tolist() == ["UT(TD)"]


def test_inter_method_candidates_support_l1_l2_label_only_rows() -> None:
    rows = pd.DataFrame({"l1_l2_method": ["UT(FD)"]})

    candidates = candidates_from_rows(rows=rows)

    assert candidates[0].candidate_label == "UT(FD)"
