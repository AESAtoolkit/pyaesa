from pathlib import Path
import json
import re
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import (
    RunBatch,
    FixedRunPlan,
    run_seed_from_run_id,
    fixed_run_plan,
)
from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    run_positions_in_window,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.composite import (
    component_inventory_finalizes,
    component_inventory_payload,
    fixed_inventory_mc_parameters,
    initial_component_inventory_finalizes,
    run_role_payload,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.convergence import (
    convergence_run_checkpoints,
)
from pyaesa.shared.uncertainty_assessment.io.formats import (
    normalize_uncertainty_output_format,
    suffix_for_uncertainty_output,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    RUN_ID_PREFIX,
    allocate_run_id,
    build_compatibility_key,
    build_manifest,
    read_manifest,
    write_manifest,
)
from pyaesa.shared.uncertainty_assessment.run_state.figure_artifacts import (
    manifest_figure_artifacts_current,
    manifest_with_figure_artifacts,
)
from pyaesa.shared.uncertainty_assessment.run_state.report import uncertainty_report
from pyaesa.shared.uncertainty_assessment.request.core import normalize_uncertainty_request
from pyaesa.shared.uncertainty_assessment.run_state.runs import (
    CompatibleMonteCarloRun,
    appendable_completed_run,
    compatible_completed_runs,
    compatible_completed_run_for_id,
    cleanup_monte_carlo_runs_for_refresh,
    complete_run_with_requested_runs,
    complete_run_with_requested_sobol,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.random_streams import uniform_by_run_index
from pyaesa.shared.uncertainty_assessment.io.public_summary import (
    exact_summary_and_frequency_from_public_runs,
    exact_summary_from_public_runs,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import (
    iter_compact_run_matrix,
    iter_sparse_run_row_windows,
    iter_sparse_run_rows,
)
from pyaesa.shared.uncertainty_assessment.io.downstream_run_outputs import (
    DownstreamRunOutputPaths,
    DownstreamRunOutputPlan,
    append_downstream_run_outputs,
    close_downstream_run_output_state,
    new_downstream_run_output_state,
    write_downstream_run_outputs,
)
from pyaesa.shared.uncertainty_assessment.io.summary_kernels import group_block_stop
from pyaesa.shared.uncertainty_assessment.io.csv_fragments import (
    CSV_COMPACT_RUN_FRAGMENT_SUFFIX,
    csv_run_fragment_input,
    max_abs_int,
)
from pyaesa.shared.uncertainty_assessment.io.run_writers import (
    CompactRunMatrixWriter,
    SparseRunRows,
    SparseRunRowsWriter,
)
from pyaesa.shared.uncertainty_assessment.io.run_artifacts import (
    RunIntervalWriterState,
    public_run_artifact_contract,
    public_run_artifact_readme_lines,
    read_run_interval_index,
    run_interval_index_path,
)
from pyaesa.shared.uncertainty_assessment.io.tables import (
    read_uncertainty_table,
    uncertainty_table_columns,
    write_uncertainty_table,
)
from pyaesa.shared.uncertainty_assessment.orchestration import progress_complete
from pyaesa.shared.runtime.reporting.run_progress import RunProgressPrinter


def _read_csv_fragment(path: Path) -> pd.DataFrame:
    with csv_run_fragment_input(path=path) as source:
        return pd.read_csv(source)


def test_uncertainty_formats_reject_pickle() -> None:
    assert normalize_uncertainty_output_format(" CSV_COMPACT ") == "csv_compact"
    assert suffix_for_uncertainty_output("csv_compact") == ".csv"
    assert suffix_for_uncertainty_output("PARQUET") == ".parquet"
    for output_format in ("csv", "pickle"):
        with pytest.raises(ValueError, match="Unsupported uncertainty output_format"):
            normalize_uncertainty_output_format(output_format)


def test_fixed_run_plan_owns_ordered_batches_and_rng() -> None:
    plan = fixed_run_plan(n_runs=5, batch_size=2, seed=123, start_run_index=10)
    batches = plan.batches()
    assert plan.batch_count == 3
    assert plan.stop_run_index == 15
    assert batches[-1].n_runs == 1
    assert [batch.run_indices().tolist() for batch in batches] == [[10, 11], [12, 13], [14]]
    assert (
        batches[0].rng().random(3).tolist()
        == fixed_run_plan(
            n_runs=5,
            batch_size=2,
            seed=123,
            start_run_index=10,
        )
        .batches()[0]
        .rng()
        .random(3)
        .tolist()
    )
    full = fixed_run_plan(n_runs=6, batch_size=2, seed=123).batches()
    continuation = fixed_run_plan(
        n_runs=2,
        batch_size=2,
        seed=123,
        start_run_index=4,
    ).batches()
    assert continuation[0].batch_index == 2
    assert continuation[0].rng().random(3).tolist() == full[2].rng().random(3).tolist()

    automatic_seed_plan = fixed_run_plan(n_runs=1, batch_size=1)
    assert isinstance(automatic_seed_plan.seed, int)
    assert run_seed_from_run_id(run_id="mc_parent") == run_seed_from_run_id(run_id="mc_parent")
    assert run_seed_from_run_id(run_id="mc_parent") != run_seed_from_run_id(run_id="mc_child")

    run_batch_error_cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"batch_index": -1, "start_run_index": 0, "stop_run_index": 1, "rng_seed": 1}, "index"),
        ({"batch_index": 0, "start_run_index": 1, "stop_run_index": 1, "rng_seed": 1}, "range"),
    )
    for kwargs, message in run_batch_error_cases:
        with pytest.raises(ValueError, match=message):
            RunBatch(**kwargs)
    fixed_plan_error_cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"n_runs": 0, "batch_size": 1, "seed": 1}, "n_runs"),
        ({"n_runs": 1, "batch_size": 0, "seed": 1}, "batch_size"),
        ({"n_runs": 1, "batch_size": 1, "seed": 1, "start_run_index": -1}, "start_run_index"),
    )
    for kwargs, message in fixed_plan_error_cases:
        with pytest.raises(ValueError, match=message):
            FixedRunPlan(**kwargs)


def test_run_indexed_random_values_are_reproducible_and_independent() -> None:
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=3, rng_seed=42)

    first = uniform_by_run_index(
        stream_name="asocc.inter_mrio.alpha",
        run_indices=batch.run_indices(),
    )
    repeated = uniform_by_run_index(
        stream_name="asocc.inter_mrio.alpha",
        run_indices=batch.run_indices(),
    )
    other = uniform_by_run_index(
        stream_name="asocc.inter_method.selection",
        run_indices=batch.run_indices(),
    )
    selected = RunBatch(
        batch_index=0,
        start_run_index=0,
        stop_run_index=3,
        rng_seed=42,
        run_index_values=(0, 2),
    )

    assert first.tolist() == repeated.tolist()
    assert first.tolist() != other.tolist()
    assert selected.n_runs == 2
    assert selected.run_indices().tolist() == [0, 2]
    assert (
        uniform_by_run_index(
            stream_name="asocc.inter_mrio.alpha",
            run_indices=selected.run_indices(),
        ).tolist()
        == first[[0, 2]].tolist()
    )


def test_composite_convergence_checkpoints_and_inventory_payloads() -> None:
    fixed_runtime = normalize_uncertainty_request(
        family="acc",
        output_format="csv_compact",
        mc_parameters={"fixed": {"active": True, "n_runs": 7}, "convergence": {"active": False}},
    )
    convergence_runtime = normalize_uncertainty_request(
        family="asr",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 10, "stable_runs": 3},
        },
    )
    short_runtime = normalize_uncertainty_request(
        family="asr",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 2, "stable_runs": 5},
        },
    )
    large_runtime = normalize_uncertainty_request(
        family="asr",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 20_000, "stable_runs": 10_000},
        },
    )
    inventory = component_inventory_payload(
        composite_family="asr",
        component_name="acc",
        target_runs=10,
    )

    assert convergence_run_checkpoints(runtime=fixed_runtime) == (7,)
    assert convergence_run_checkpoints(runtime=convergence_runtime) == (3, 6, 9, 10)
    assert convergence_run_checkpoints(runtime=short_runtime) == (2,)
    assert convergence_run_checkpoints(runtime=large_runtime) == (10_000, 20_000)
    assert fixed_inventory_mc_parameters(target_runs=12) == {
        "fixed": {"active": True, "n_runs": 12},
        "convergence": {
            "active": False,
            "max_runs": 500_000,
            "rtol": 0.05,
            "stable_runs": 10_000,
            "convergence_statistics": ["mean"],
        },
    }
    assert inventory["target_runs"] == 10
    assert "target_runs" not in run_role_payload(component_inventory=inventory)
    assert component_inventory_finalizes(
        component_inventory=None,
        finalize_component_inventory=False,
    )
    assert component_inventory_finalizes(
        component_inventory=inventory,
        finalize_component_inventory=True,
    )
    assert initial_component_inventory_finalizes(checkpoints=(10,))
    assert not initial_component_inventory_finalizes(checkpoints=(3, 6, 9, 10))
    assert not initial_component_inventory_finalizes(
        checkpoints=(10,),
        finalize_outputs=False,
    )


def test_uncertainty_table_io_covers_complete_csv_and_parquet(tmp_path: Path) -> None:
    high_precision = np.float64("0.37491753620123456")
    frame = pd.DataFrame({"run_index": [0, 1], "value": [1.0, high_precision]})
    csv_path = tmp_path / "runs.csv"
    assert (
        write_uncertainty_table(
            path=csv_path,
            frame=frame,
            output_format="csv_compact",
        )
        == csv_path
    )
    read_csv = read_uncertainty_table(path=csv_path, output_format="csv_compact")
    pd.testing.assert_frame_equal(
        read_csv,
        frame,
        check_dtype=False,
    )
    assert format(float(high_precision), ".17g") in csv_path.read_text(encoding="utf-8")
    assert np.array_equal(
        read_csv["value"].to_numpy(dtype=np.float64),
        frame["value"].to_numpy(dtype=np.float64),
    )
    assert uncertainty_table_columns(path=csv_path, output_format="csv_compact") == [
        "run_index",
        "value",
    ]

    parquet_path = tmp_path / "runs.parquet"
    write_uncertainty_table(path=parquet_path, frame=frame, output_format="parquet")
    assert read_uncertainty_table(path=parquet_path, output_format="parquet").equals(frame)
    assert uncertainty_table_columns(path=parquet_path, output_format="parquet") == [
        "run_index",
        "value",
    ]

    year_path = tmp_path / "year_runs.csv"
    year_frame = pd.DataFrame(
        {
            "run_index": [0, 1],
            "year": [2030.0, 2031.0],
            "reference_year": [2005.0, None],
            "l2_reuse_year": [2022.0, None],
            "value": [1.5, 2.5],
        }
    )
    write_uncertainty_table(path=year_path, frame=year_frame, output_format="csv_compact")
    raw_years = year_path.read_text(encoding="utf-8")
    assert "2030.0" not in raw_years
    assert "2005.0" not in raw_years
    assert "2022.0" not in raw_years
    assert "0,2030,2005,2022,1.5" in raw_years
    assert "1,2031,,," in raw_years


def test_run_artifact_edge_contracts_use_interval_indexes(tmp_path: Path) -> None:
    csv_contract = public_run_artifact_contract(
        path=tmp_path / "runs.csv",
        output_format="csv_compact",
    )
    parquet_contract = public_run_artifact_contract(
        path=tmp_path / "runs.parquet",
        output_format="parquet",
    )
    assert csv_contract["artifact_kind"] == "csv_compact_dataset_directory"
    assert csv_contract["fragment_pattern"] == f"part-*{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}"
    assert parquet_contract["artifact_kind"] == "parquet_dataset_directory"
    readme_lines = public_run_artifact_readme_lines(run_name="demo_runs")
    readme_text = "\n".join(readme_lines)
    assert readme_lines[0].startswith("- demo_runs:")
    assert "part-*.csv.zst CSV fragments" in readme_text
    assert "interval index" in readme_text
    assert "read only requested run windows" in readme_text

    empty_sparse_parquet = tmp_path / "empty_sparse.parquet"
    empty_sparse_rows = SparseRunRows(
        run_index=np.empty(0, dtype=np.int64),
        public_row_id=np.empty(0, dtype=np.int64),
        values=np.empty(0, dtype=np.float64),
        value_column="value",
    )
    with SparseRunRowsWriter(path=empty_sparse_parquet, output_format="parquet") as writer:
        writer.write_batch(rows=empty_sparse_rows, batch_index=0)
    assert list(iter_sparse_run_rows(path=empty_sparse_parquet, output_format="parquet")) == []

    sparse_parquet_after_data = tmp_path / "sparse_parquet_after_data.parquet"
    with SparseRunRowsWriter(path=sparse_parquet_after_data, output_format="parquet") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([0], dtype=np.int64),
                public_row_id=np.array([0], dtype=np.int64),
                values=np.array([1.0], dtype=np.float64),
                value_column="value",
            ),
            batch_index=0,
        )
        writer.write_batch(rows=empty_sparse_rows, batch_index=1)
    assert (
        len(read_run_interval_index(path=sparse_parquet_after_data, output_format="parquet")) == 1
    )

    corrupt_sparse_parquet = tmp_path / "corrupt_sparse.parquet"
    corrupt_sparse_parquet.mkdir()
    pd.DataFrame(
        {
            "run_index": pd.Series(dtype="int64"),
            "public_row_id": pd.Series(dtype="int64"),
            "value": pd.Series(dtype="float64"),
        }
    ).to_parquet(corrupt_sparse_parquet / "part-00000000.parquet", index=False)
    pd.DataFrame(
        {
            "batch_index": [0],
            "run_start": [0],
            "run_stop": [1],
            "row_start": [0],
            "row_count": [1],
            "fragment": ["part-00000000.parquet"],
        }
    ).to_parquet(
        run_interval_index_path(path=corrupt_sparse_parquet, output_format="parquet"),
        index=False,
    )
    assert list(iter_sparse_run_rows(path=corrupt_sparse_parquet, output_format="parquet")) == []

    stale_csv = tmp_path / "stale.csv"
    stale_csv.write_text("old", encoding="utf-8")
    run_interval_index_path(path=stale_csv, output_format="csv_compact").write_text(
        "old",
        encoding="utf-8",
    )
    with CompactRunMatrixWriter(path=stale_csv, output_format="csv_compact") as writer:
        writer.write_batch(run_indices=[0], values=[[2.0]], batch_index=0)
    assert stale_csv.is_dir()
    assert not (stale_csv / "old.txt").exists()
    assert _read_csv_fragment(stale_csv / f"part-00000000{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}")[
        "0"
    ].tolist() == [2.0]
    assert read_run_interval_index(path=stale_csv, output_format="csv_compact")[
        "fragment"
    ].tolist() == [f"part-00000000{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}"]

    stale_parquet = tmp_path / "stale.parquet"
    stale_parquet.mkdir()
    (stale_parquet / "old.txt").write_text("old", encoding="utf-8")
    with CompactRunMatrixWriter(path=stale_parquet, output_format="parquet") as writer:
        writer.write_batch(run_indices=[0], values=[[3.0]], batch_index=0)
    assert not (stale_parquet / "old.txt").exists()
    assert pd.read_parquet(stale_parquet)["0"].tolist() == [3.0]

    append_path = tmp_path / "append_missing.parquet"
    append_state = RunIntervalWriterState.create(
        path=append_path,
        output_format="parquet",
        append_existing=True,
    )
    append_state.prepare()
    assert append_path.is_dir()

    assert max_abs_int(values=np.array([], dtype=np.int64)) == 0


def test_compact_run_matrix_writer_and_summary(tmp_path: Path) -> None:
    identity = pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "region": ["FR", "DE"],
        }
    )
    matrix_path = tmp_path / "asocc_runs.csv"
    with CompactRunMatrixWriter(path=matrix_path, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=[0, 1],
            values=[[1.0, 10.0], [3.0, 14.0]],
            batch_index=0,
        )
        writer.write_batch(
            run_indices=[2],
            values=[[5.0, 18.0]],
            batch_index=1,
        )

    matrix = _read_csv_fragment(matrix_path / f"part-00000000{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}")
    assert matrix.columns.tolist() == ["run_index", "0", "1"]
    assert read_run_interval_index(path=matrix_path, output_format="csv_compact")[
        "fragment"
    ].tolist() == [
        f"part-00000000{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}",
        f"part-00000001{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}",
    ]

    summary = exact_summary_from_public_runs(
        identity_frame=identity,
        runs_path=matrix_path,
        output_format="csv_compact",
        run_count=3,
    )
    assert summary["median"].tolist() == [3.0, 14.0]
    grouped_summary = exact_summary_from_public_runs(
        identity_frame=pd.DataFrame({"region": ["FR"]}),
        runs_path=matrix_path,
        output_format="csv_compact",
        run_count=3,
        public_row_groups=(("0", "1"),),
    )
    identity_summary = exact_summary_from_public_runs(
        identity_frame=identity,
        runs_path=matrix_path,
        output_format="csv_compact",
        run_count=3,
        public_row_groups=(("0",), ("1",)),
    )
    assert grouped_summary["median"].tolist() == [8.5]
    assert identity_summary["median"].tolist() == [3.0, 14.0]

    selected_matrix_path = tmp_path / "selected_asocc_runs.csv"
    with CompactRunMatrixWriter(path=selected_matrix_path, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=[0, 1],
            values=[[1.0, 10.0, 100.0], [3.0, 30.0, 300.0]],
            batch_index=0,
        )
    selected_summary = exact_summary_from_public_runs(
        identity_frame=pd.DataFrame({"public_row_id": [0, 2]}),
        runs_path=selected_matrix_path,
        output_format="csv_compact",
        run_count=2,
        public_row_groups=(("0",), ("2",)),
    )
    assert selected_summary["median"].tolist() == [2.0, 200.0]

    parquet_path = tmp_path / "asocc_runs.parquet"
    with CompactRunMatrixWriter(path=parquet_path, output_format="parquet") as writer:
        writer.write_batch(run_indices=[0], values=[[2.0, 4.0]], batch_index=0)
        writer.write_batch(run_indices=[1], values=[[6.0, 8.0]], batch_index=1)
    with CompactRunMatrixWriter(
        path=parquet_path,
        output_format="parquet",
        append_existing=True,
    ) as writer:
        writer.write_batch(run_indices=[2], values=[[10.0, 12.0]], batch_index=2)
    parquet_matrix = pd.read_parquet(parquet_path)
    assert parquet_matrix["0"].tolist() == [2.0, 6.0, 10.0]
    assert parquet_matrix["1"].tolist() == [4.0, 8.0, 12.0]
    empty_parquet_path = tmp_path / "empty.parquet"
    with CompactRunMatrixWriter(path=empty_parquet_path, output_format="parquet"):
        pass
    assert not empty_parquet_path.exists()


@pytest.mark.parametrize(
    "output_format,suffix",
    [("csv_compact", CSV_COMPACT_RUN_FRAGMENT_SUFFIX), ("parquet", ".parquet")],
)
def test_compact_run_matrix_writer_splits_memory_bounded_fragments(
    tmp_path: Path,
    output_format: str,
    suffix: str,
) -> None:
    runs_path = tmp_path / f"split_compact{suffix}"
    values = np.arange(10, dtype=np.float64).reshape(5, 2)
    with CompactRunMatrixWriter(
        path=runs_path,
        output_format=output_format,
        memory_budget_bytes=80,
    ) as writer:
        writer.write_batch(
            run_indices=np.arange(5, dtype=np.int64),
            values=values,
            batch_index=0,
        )

    intervals = read_run_interval_index(path=runs_path, output_format=output_format)
    assert intervals["fragment"].nunique() > 1
    assert sorted(path.name for path in runs_path.glob(f"part-*{suffix}")) == sorted(
        intervals["fragment"].tolist()
    )
    chunks = list(
        iter_compact_run_matrix(path=runs_path, output_format=output_format, column_count=2)
    )
    assert np.concatenate([chunk[0] for chunk in chunks]).tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(np.vstack([chunk[1] for chunk in chunks]), values)


def test_sparse_render_rows_writer_keeps_selected_values_only(tmp_path: Path) -> None:
    rows = SparseRunRows(
        run_index=pd.Series([0, 0, 1]).to_numpy(),
        public_row_id=pd.Series([2, 5, 2]).to_numpy(),
        values=pd.Series([0.4, 0.7, 0.5]).to_numpy(),
        value_column="value",
    )
    csv_path = tmp_path / "run_values.csv"
    with SparseRunRowsWriter(path=csv_path, output_format="csv_compact") as writer:
        writer.write_batch(rows=rows, batch_index=0)
        writer.write_batch(
            rows=SparseRunRows(
                run_index=pd.Series([2]).to_numpy(),
                public_row_id=pd.Series([5]).to_numpy(),
                values=pd.Series([0.8]).to_numpy(),
                value_column="value",
            ),
            batch_index=1,
        )
    csv_rows = _read_csv_fragment(csv_path / f"part-00000000{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}")
    assert csv_rows.columns.tolist() == ["run_index", "public_row_id", "value"]
    assert [
        chunk.public_row_id.tolist()
        for chunk in iter_sparse_run_rows(
            path=csv_path,
            output_format="csv_compact",
        )
    ] == [[2, 5], [2], [5]]

    parquet_path = tmp_path / "run_values.parquet"
    with SparseRunRowsWriter(path=parquet_path, output_format="parquet") as writer:
        writer.write_batch(rows=rows, batch_index=0)
        writer.write_batch(
            rows=SparseRunRows(
                run_index=pd.Series([2]).to_numpy(),
                public_row_id=pd.Series([5]).to_numpy(),
                values=pd.Series([0.8]).to_numpy(),
                value_column="value",
            ),
            batch_index=1,
        )
    parquet_rows = pd.read_parquet(parquet_path)
    assert parquet_rows["value"].tolist() == [0.4, 0.7, 0.5, 0.8]
    assert run_interval_index_path(path=parquet_path, output_format="parquet").exists()


def test_downstream_sparse_writer_counts_runs_without_selected_rows(tmp_path: Path) -> None:
    runtime = normalize_uncertainty_request(
        family="acc",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 4, "rtol": 1.0, "stable_runs": 2},
        },
    )
    runtime = replace(runtime, batch_size=1)
    identity = pd.DataFrame({"public_row_id": [0], "year": [2030]})
    rows = SparseRunRows(
        run_index=np.array([0, 1], dtype=np.int64),
        public_row_id=np.array([0, 0], dtype=np.int64),
        values=np.array([1.0, 2.0], dtype=np.float64),
        value_column="acc",
    )

    def sparse_batches(
        _output_format: str,
        start: int,
        stop: int,
        _batch_size: int,
    ) -> Iterator[tuple[np.ndarray, SparseRunRows]]:
        run_indices = np.arange(start, stop, dtype=np.int64)
        mask = (rows.run_index >= start) & (rows.run_index < stop)
        yield (
            run_indices,
            SparseRunRows(
                run_index=rows.run_index[mask],
                public_row_id=rows.public_row_id[mask],
                values=rows.values[mask],
                value_column=rows.value_column,
            ),
        )

    paths = DownstreamRunOutputPaths(
        run_root=tmp_path,
        public_runs=tmp_path / "runs.csv",
        summary_stats_runs=tmp_path / "summary.csv",
    )
    plan = DownstreamRunOutputPlan(
        run_layout="sparse_selected_rows",
        summary_identity=identity,
        public_row_count=len(identity),
        compact_batches=lambda _output_format, _start, _stop, _batch_size: iter(()),
        sparse_batches=sparse_batches,
        collapse_compact=lambda values: values,
        sparse_public_row_group_membership_index=lambda: np.array([[0, 0]], dtype=np.int64),
        empty_sparse_rows=lambda: SparseRunRows(
            run_index=np.empty(0, dtype=np.int64),
            public_row_id=np.empty(0, dtype=np.int64),
            values=np.empty(0, dtype=np.float64),
            value_column="acc",
        ),
    )
    state = new_downstream_run_output_state(paths=paths)
    try:
        state, convergence = append_downstream_run_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            state=state,
            target_runs=1,
            final_checkpoint=False,
        )
        assert state.completed_runs == 1
        assert convergence is None
        assert not paths.summary_stats_runs.exists()
        state, convergence = append_downstream_run_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            state=state,
            target_runs=4,
            final_checkpoint=True,
            progress=RunProgressPrinter(
                source="uncertainty_acc",
                action="Monte Carlo",
                enabled=False,
            ),
        )
    finally:
        close_downstream_run_output_state(state=state)

    assert state.completed_runs == 4
    assert convergence is not None
    assert convergence["completed_runs"] == 4
    summary = pd.read_csv(paths.summary_stats_runs)
    assert summary["mean"].tolist() == [1.5]

    appended_state = new_downstream_run_output_state(
        paths=paths,
        completed_runs=state.completed_runs,
    )
    try:
        appended_state, appended_convergence = append_downstream_run_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            state=appended_state,
            target_runs=4,
            final_checkpoint=True,
            progress=RunProgressPrinter(
                source="uncertainty_acc",
                action="Monte Carlo",
                enabled=False,
            ),
        )
    finally:
        close_downstream_run_output_state(state=appended_state)

    assert appended_state.completed_runs == 4
    assert appended_convergence is not None

    runtime_fixed = normalize_uncertainty_request(
        family="acc",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": True, "n_runs": 2},
            "convergence": {"active": False},
        },
    )
    runtime_fixed = replace(runtime_fixed, batch_size=1)
    rows_with_initial_gap = SparseRunRows(
        run_index=np.array([1], dtype=np.int64),
        public_row_id=np.array([0], dtype=np.int64),
        values=np.array([3.0], dtype=np.float64),
        value_column="acc",
    )

    def sparse_gap_batches(
        _output_format: str,
        start: int,
        stop: int,
        _batch_size: int,
    ) -> Iterator[tuple[np.ndarray, SparseRunRows]]:
        run_indices = np.arange(start, stop, dtype=np.int64)
        mask = (rows_with_initial_gap.run_index >= start) & (rows_with_initial_gap.run_index < stop)
        yield (
            run_indices,
            SparseRunRows(
                run_index=rows_with_initial_gap.run_index[mask],
                public_row_id=rows_with_initial_gap.public_row_id[mask],
                values=rows_with_initial_gap.values[mask],
                value_column=rows_with_initial_gap.value_column,
            ),
        )

    gap_paths = DownstreamRunOutputPaths(
        run_root=tmp_path / "gap",
        public_runs=tmp_path / "gap" / "runs.csv",
        summary_stats_runs=tmp_path / "gap" / "summary.csv",
    )
    gap_plan = DownstreamRunOutputPlan(
        run_layout="sparse_selected_rows",
        summary_identity=identity,
        public_row_count=len(identity),
        compact_batches=lambda _output_format, _start, _stop, _batch_size: iter(()),
        sparse_batches=sparse_gap_batches,
        collapse_compact=lambda values: values,
        sparse_public_row_group_membership_index=lambda: np.array([[0, 0]], dtype=np.int64),
        empty_sparse_rows=plan.empty_sparse_rows,
    )
    gap_state = new_downstream_run_output_state(paths=gap_paths)
    try:
        gap_state, _gap_convergence = append_downstream_run_outputs(
            paths=gap_paths,
            plan=gap_plan,
            runtime=runtime_fixed,
            state=gap_state,
            target_runs=2,
            final_checkpoint=True,
        )
    finally:
        close_downstream_run_output_state(state=gap_state)
    resumed_gap_state = new_downstream_run_output_state(
        paths=gap_paths,
        completed_runs=gap_state.completed_runs,
    )
    try:
        resumed_gap_state, _resumed_gap_convergence = append_downstream_run_outputs(
            paths=gap_paths,
            plan=gap_plan,
            runtime=runtime_fixed,
            state=resumed_gap_state,
            target_runs=2,
            final_checkpoint=True,
        )
    finally:
        close_downstream_run_output_state(state=resumed_gap_state)
    assert resumed_gap_state.completed_runs == 2


def test_hidden_progress_complete_is_noop() -> None:
    progress_complete(
        progress=RunProgressPrinter(source="uncertainty_acc", action="Monte Carlo", enabled=False),
        completed=0,
        max_runs=2,
        visible=False,
    )


def test_downstream_compact_writer_stops_after_convergence(tmp_path: Path) -> None:
    runtime = normalize_uncertainty_request(
        family="acc",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 4, "stable_runs": 1},
        },
    )
    runtime = replace(runtime, batch_size=1)
    identity = pd.DataFrame({"public_row_id": [0], "year": [2030]})

    def compact_batches(
        _output_format: str,
        start: int,
        stop: int,
        _batch_size: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        run_indices = np.arange(start, stop, dtype=np.int64)
        yield run_indices, np.ones((len(run_indices), 1), dtype=np.float64)

    paths = DownstreamRunOutputPaths(
        run_root=tmp_path,
        public_runs=tmp_path / "compact_runs.csv",
        summary_stats_runs=tmp_path / "compact_summary.csv",
    )
    plan = DownstreamRunOutputPlan(
        run_layout="compact_run_matrix",
        summary_identity=identity,
        public_row_count=len(identity),
        compact_batches=compact_batches,
        collapse_compact=lambda values: values,
    )
    intermediate_paths = DownstreamRunOutputPaths(
        run_root=tmp_path / "intermediate",
        public_runs=tmp_path / "intermediate" / "runs.csv",
        summary_stats_runs=tmp_path / "intermediate" / "summary.csv",
    )
    state = new_downstream_run_output_state(paths=intermediate_paths)
    try:
        state, convergence = append_downstream_run_outputs(
            paths=intermediate_paths,
            plan=plan,
            runtime=runtime,
            state=state,
            target_runs=1,
            final_checkpoint=False,
        )
    finally:
        close_downstream_run_output_state(state=state)
    assert convergence is None
    assert not intermediate_paths.summary_stats_runs.exists()

    delayed_runtime = normalize_uncertainty_request(
        family="acc",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 4, "stable_runs": 2},
        },
    )
    delayed_runtime = replace(delayed_runtime, batch_size=1)
    delayed_paths = DownstreamRunOutputPaths(
        run_root=tmp_path / "delayed",
        public_runs=tmp_path / "delayed" / "runs.csv",
        summary_stats_runs=tmp_path / "delayed" / "summary.csv",
    )
    delayed_state = new_downstream_run_output_state(paths=delayed_paths)
    try:
        delayed_state, delayed_convergence = append_downstream_run_outputs(
            paths=delayed_paths,
            plan=plan,
            runtime=delayed_runtime,
            state=delayed_state,
            target_runs=1,
            final_checkpoint=False,
        )
    finally:
        close_downstream_run_output_state(state=delayed_state)
    assert delayed_convergence is None
    assert delayed_state.completed_runs == 1
    delayed_resumed_state = new_downstream_run_output_state(
        paths=delayed_paths,
        completed_runs=delayed_state.completed_runs,
    )
    try:
        delayed_resumed_state, delayed_resumed_convergence = append_downstream_run_outputs(
            paths=delayed_paths,
            plan=plan,
            runtime=delayed_runtime,
            state=delayed_resumed_state,
            target_runs=delayed_state.completed_runs,
            final_checkpoint=False,
        )
    finally:
        close_downstream_run_output_state(state=delayed_resumed_state)
    assert delayed_resumed_convergence is None
    assert delayed_resumed_state.completed_runs == delayed_state.completed_runs
    assert not delayed_paths.summary_stats_runs.exists()

    completed, convergence = write_downstream_run_outputs(paths=paths, plan=plan, runtime=runtime)

    assert completed == 2
    assert convergence is not None
    assert convergence["reached"] is True
    assert pd.read_csv(paths.summary_stats_runs)["mean"].tolist() == [1.0]

    resumed_state = new_downstream_run_output_state(paths=paths, completed_runs=completed)
    try:
        resumed_state, resumed_convergence = append_downstream_run_outputs(
            paths=paths,
            plan=plan,
            runtime=runtime,
            state=resumed_state,
            target_runs=completed,
            final_checkpoint=True,
            progress=RunProgressPrinter(
                source="uncertainty_acc",
                action="Monte Carlo",
                enabled=False,
            ),
        )
    finally:
        close_downstream_run_output_state(state=resumed_state)
    assert resumed_state.completed_runs == completed
    assert resumed_convergence is not None


def test_run_matrix_readers_cover_csv_and_parquet_chunking(tmp_path: Path) -> None:
    compact_csv = tmp_path / "compact.csv"
    with CompactRunMatrixWriter(path=compact_csv, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=[0],
            values=[[1.0, 3.0]],
            batch_index=0,
        )
        writer.write_batch(
            run_indices=[1],
            values=[[2.0, 4.0]],
            batch_index=1,
        )
    compact_fragments = sorted(
        path.name for path in compact_csv.glob(f"part-*{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}")
    )
    assert compact_fragments == [
        f"part-00000000{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}",
        f"part-00000001{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}",
    ]
    assert read_run_interval_index(path=compact_csv, output_format="csv_compact")[
        "fragment"
    ].tolist() == [
        f"part-00000000{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}",
        f"part-00000001{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}",
    ]
    compact_csv_chunks = list(
        iter_compact_run_matrix(path=compact_csv, output_format="csv_compact", column_count=2)
    )
    assert np.concatenate([chunk[0] for chunk in compact_csv_chunks]).tolist() == [0, 1]
    assert np.vstack([chunk[1] for chunk in compact_csv_chunks]).tolist() == [
        [1.0, 3.0],
        [2.0, 4.0],
    ]
    ranged_compact_csv = list(
        iter_compact_run_matrix(
            path=compact_csv,
            output_format="csv_compact",
            column_count=2,
            start_run_index=1,
            stop_run_index=2,
        )
    )
    assert ranged_compact_csv[0][0].tolist() == [1]
    assert (
        list(
            iter_compact_run_matrix(
                path=compact_csv,
                output_format="csv_compact",
                column_count=2,
                start_run_index=3,
                stop_run_index=4,
            )
        )
        == []
    )
    compact_gap = tmp_path / "compact_gap.csv"
    with CompactRunMatrixWriter(path=compact_gap, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=[0, 2],
            values=[[1.0], [3.0]],
            batch_index=0,
        )
    assert (
        list(
            iter_compact_run_matrix(
                path=compact_gap,
                output_format="csv_compact",
                column_count=1,
                start_run_index=1,
                stop_run_index=2,
            )
        )
        == []
    )

    compact_parquet = tmp_path / "compact.parquet"
    with CompactRunMatrixWriter(path=compact_parquet, output_format="parquet") as writer:
        writer.write_batch(
            run_indices=[0],
            values=[[5.0, 7.0]],
            batch_index=0,
        )
        writer.write_batch(
            run_indices=[1],
            values=[[6.0, 8.0]],
            batch_index=1,
        )
    compact_parquet_chunks = list(
        iter_compact_run_matrix(path=compact_parquet, output_format="parquet", column_count=2)
    )
    assert np.concatenate([chunk[0] for chunk in compact_parquet_chunks]).tolist() == [0, 1]
    assert np.vstack([chunk[1] for chunk in compact_parquet_chunks]).tolist() == [
        [5.0, 7.0],
        [6.0, 8.0],
    ]
    ranged_compact_parquet = list(
        iter_compact_run_matrix(
            path=compact_parquet,
            output_format="parquet",
            column_count=2,
            start_run_index=0,
            stop_run_index=1,
        )
    )
    assert ranged_compact_parquet[0][0].tolist() == [0]
    assert (
        list(
            iter_compact_run_matrix(
                path=compact_parquet,
                output_format="parquet",
                column_count=2,
                start_run_index=2,
                stop_run_index=3,
            )
        )
        == []
    )
    compact_parquet_gap = tmp_path / "compact_gap.parquet"
    with CompactRunMatrixWriter(path=compact_parquet_gap, output_format="parquet") as writer:
        writer.write_batch(
            run_indices=[0, 2],
            values=[[1.0], [3.0]],
            batch_index=0,
        )
    assert (
        list(
            iter_compact_run_matrix(
                path=compact_parquet_gap,
                output_format="parquet",
                column_count=1,
                start_run_index=1,
                stop_run_index=2,
            )
        )
        == []
    )

    sparse_csv = tmp_path / "sparse.csv"
    with SparseRunRowsWriter(path=sparse_csv, output_format="csv_compact") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([0, 0], dtype=np.int64),
                public_row_id=np.array([1, 2], dtype=np.int64),
                values=np.array([0.1, 0.2], dtype=np.float64),
                value_column="value",
            ),
            batch_index=0,
        )
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([1, 1], dtype=np.int64),
                public_row_id=np.array([1, 2], dtype=np.int64),
                values=np.array([0.3, 0.4], dtype=np.float64),
                value_column="value",
            ),
            batch_index=1,
        )
    assert read_run_interval_index(path=sparse_csv, output_format="csv_compact")[
        "fragment"
    ].tolist() == [
        f"part-00000000{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}",
        f"part-00000001{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}",
    ]
    sparse_csv_chunks = list(
        iter_sparse_run_rows(
            path=sparse_csv,
            output_format="csv_compact",
            max_rows_per_chunk=1,
        )
    )
    assert [chunk.run_index.tolist() for chunk in sparse_csv_chunks] == [[0, 0], [1, 1]]
    sparse_csv_first = list(
        iter_sparse_run_rows(
            path=sparse_csv,
            output_format="csv_compact",
            start_run_index=0,
            stop_run_index=1,
        )
    )
    assert [chunk.run_index.tolist() for chunk in sparse_csv_first] == [[0, 0]]
    sparse_csv_range = list(
        iter_sparse_run_rows(
            path=sparse_csv,
            output_format="csv_compact",
            start_run_index=1,
            stop_run_index=2,
        )
    )
    assert [chunk.run_index.tolist() for chunk in sparse_csv_range] == [[1, 1]]
    assert (
        list(
            iter_sparse_run_rows(
                path=sparse_csv,
                output_format="csv_compact",
                start_run_index=3,
                stop_run_index=4,
            )
        )
        == []
    )
    sparse_gap = tmp_path / "sparse_gap.csv"
    with SparseRunRowsWriter(path=sparse_gap, output_format="csv_compact") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([0, 2], dtype=np.int64),
                public_row_id=np.array([1, 1], dtype=np.int64),
                values=np.array([0.1, 0.3], dtype=np.float64),
                value_column="value",
            ),
            batch_index=0,
        )
    assert (
        list(
            iter_sparse_run_rows(
                path=sparse_gap,
                output_format="csv_compact",
                start_run_index=1,
                stop_run_index=2,
            )
        )
        == []
    )

    sparse_parquet = tmp_path / "sparse.parquet"
    with SparseRunRowsWriter(path=sparse_parquet, output_format="parquet") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([0, 0], dtype=np.int64),
                public_row_id=np.array([1, 2], dtype=np.int64),
                values=np.array([0.5, 0.6], dtype=np.float64),
                value_column="value",
            ),
            batch_index=0,
        )
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([1, 1], dtype=np.int64),
                public_row_id=np.array([1, 2], dtype=np.int64),
                values=np.array([0.7, 0.8], dtype=np.float64),
                value_column="value",
            ),
            batch_index=1,
        )
    sparse_parquet_chunks = list(
        iter_sparse_run_rows(
            path=sparse_parquet,
            output_format="parquet",
            max_rows_per_chunk=1,
        )
    )
    assert [chunk.values.tolist() for chunk in sparse_parquet_chunks] == [[0.5, 0.6], [0.7, 0.8]]
    sparse_parquet_first = list(
        iter_sparse_run_rows(
            path=sparse_parquet,
            output_format="parquet",
            start_run_index=0,
            stop_run_index=1,
        )
    )
    assert [chunk.values.tolist() for chunk in sparse_parquet_first] == [[0.5, 0.6]]
    sparse_parquet_range = list(
        iter_sparse_run_rows(
            path=sparse_parquet,
            output_format="parquet",
            start_run_index=1,
            stop_run_index=2,
        )
    )
    assert [chunk.values.tolist() for chunk in sparse_parquet_range] == [[0.7, 0.8]]
    assert (
        list(
            iter_sparse_run_rows(
                path=sparse_parquet,
                output_format="parquet",
                start_run_index=3,
                stop_run_index=4,
            )
        )
        == []
    )

    boundary_rows = 4
    boundary_sparse_parquet = tmp_path / "boundary_sparse.parquet"
    run_index = np.zeros(boundary_rows, dtype=np.int64)
    run_index[-1] = 1
    with SparseRunRowsWriter(path=boundary_sparse_parquet, output_format="parquet") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=run_index,
                public_row_id=np.arange(boundary_rows, dtype=np.int64),
                values=np.ones(boundary_rows, dtype=np.float64),
                value_column="value",
            ),
            batch_index=0,
        )
    boundary_chunks = list(
        iter_sparse_run_rows(
            path=boundary_sparse_parquet,
            output_format="parquet",
            max_rows_per_chunk=3,
        )
    )
    assert [chunk.run_index[0] for chunk in boundary_chunks] == [0, 1]
    assert [len(chunk.run_index) for chunk in boundary_chunks] == [3, 1]


@pytest.mark.parametrize(
    "output_format,suffix",
    [("csv_compact", CSV_COMPACT_RUN_FRAGMENT_SUFFIX), ("parquet", ".parquet")],
)
def test_sparse_run_rows_writer_splits_memory_bounded_fragments(
    tmp_path: Path,
    output_format: str,
    suffix: str,
) -> None:
    runs_path = tmp_path / f"split_sparse{suffix}"
    rows = SparseRunRows(
        run_index=np.array([0, 0, 1, 1, 2], dtype=np.int64),
        public_row_id=np.array([4, 5, 4, 5, 4], dtype=np.int64),
        values=np.array([0.4, 0.5, 1.4, 1.5, 2.4], dtype=np.float64),
        value_column="value",
    )
    with SparseRunRowsWriter(
        path=runs_path,
        output_format=output_format,
        memory_budget_bytes=72,
    ) as writer:
        writer.write_batch(rows=rows, batch_index=0)

    intervals = read_run_interval_index(path=runs_path, output_format=output_format)
    assert intervals["fragment"].nunique() > 1
    assert sorted(path.name for path in runs_path.glob(f"part-*{suffix}")) == sorted(
        intervals["fragment"].tolist()
    )
    chunks = list(iter_sparse_run_rows(path=runs_path, output_format=output_format))
    assert np.concatenate([chunk.run_index for chunk in chunks]).tolist() == rows.run_index.tolist()
    assert (
        np.concatenate([chunk.public_row_id for chunk in chunks]).tolist()
        == rows.public_row_id.tolist()
    )
    np.testing.assert_allclose(np.concatenate([chunk.values for chunk in chunks]), rows.values)


def test_sparse_run_row_windows_use_forward_chunks() -> None:
    empty_rows = SparseRunRows(
        run_index=np.empty(0, dtype=np.int64),
        public_row_id=np.empty(0, dtype=np.int64),
        values=np.empty(0, dtype=np.float64),
        value_column="value",
    )
    chunks = iter(
        [
            SparseRunRows(
                run_index=np.array([0, 0, 2], dtype=np.int64),
                public_row_id=np.array([1, 2, 1], dtype=np.int64),
                values=np.array([0.1, 0.2, 0.3], dtype=np.float64),
                value_column="value",
            ),
            SparseRunRows(
                run_index=np.array([4], dtype=np.int64),
                public_row_id=np.array([2], dtype=np.int64),
                values=np.array([0.4], dtype=np.float64),
                value_column="value",
            ),
        ]
    )

    windows = list(
        iter_sparse_run_row_windows(
            chunks=chunks,
            start_run_index=0,
            stop_run_index=5,
            batch_size=2,
            empty_rows=empty_rows,
        )
    )

    assert [window[0].tolist() for window in windows] == [[0, 1], [2, 3], [4]]
    assert [window[1].run_index.tolist() for window in windows] == [[0, 0], [2], [4]]
    assert run_positions_in_window(
        run_indices=np.array([0, 2, 4], dtype=np.int64),
        row_run_index=np.array([2, 4], dtype=np.int64),
    ).tolist() == [1, 2]


def test_compact_run_matrix_summary_keeps_all_nan_columns_quiet(tmp_path: Path) -> None:
    identity = pd.DataFrame({"public_row_id": [0]})
    matrix_path = tmp_path / "all_nan_runs.csv"
    with CompactRunMatrixWriter(path=matrix_path, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=[0, 1],
            values=[[float("nan")], [float("nan")]],
            batch_index=0,
        )

    summary = exact_summary_from_public_runs(
        identity_frame=identity,
        runs_path=matrix_path,
        output_format="csv_compact",
        run_count=2,
    )

    assert pd.isna(summary.loc[0, "mean"])
    assert pd.isna(summary.loc[0, "median"])


def test_public_grouped_summary_collapses_public_rows_by_run(tmp_path: Path) -> None:
    matrix_path = tmp_path / "wide_runs.csv"
    with CompactRunMatrixWriter(path=matrix_path, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=[0, 1],
            values=[
                [1.0, float("nan"), 2.0, 10.0],
                [3.0, 14.0, 6.0, 12.0],
            ],
            batch_index=0,
        )

    summary = exact_summary_from_public_runs(
        identity_frame=pd.DataFrame({"region": ["FR", "DE", "ES"]}),
        runs_path=matrix_path,
        output_format="csv_compact",
        run_count=2,
        public_row_groups=(("0", "1"), ("2",), ("3",)),
    )

    assert summary["region"].tolist() == ["FR", "DE", "ES"]
    assert summary["mean"].tolist() == [4.75, 4.0, 11.0]
    assert summary["median"].tolist() == [4.75, 4.0, 11.0]
    assert (
        group_block_stop(
            groups=(("0", "1"), ("2", "3")),
            start=0,
            run_count=2,
            max_numeric_cells_per_block=3,
        )
        == 1
    )


@pytest.mark.parametrize("output_format", ["csv_compact", "parquet"])
def test_sparse_public_summary_allows_overlapping_groups(
    tmp_path: Path,
    output_format: str,
) -> None:
    suffix = ".csv" if output_format == "csv_compact" else ".parquet"
    runs_path = tmp_path / f"sparse_runs{suffix}"
    with SparseRunRowsWriter(path=runs_path, output_format=output_format) as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.empty(0, dtype=np.int64),
                public_row_id=np.empty(0, dtype=np.int64),
                values=np.empty(0, dtype=np.float64),
                value_column="value",
            ),
            batch_index=0,
        )
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([0, 0], dtype=np.int64),
                public_row_id=np.array([0, 1], dtype=np.int64),
                values=np.array([0.5, 2.0], dtype=np.float64),
                value_column="value",
            ),
            batch_index=1,
        )
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([1, 1], dtype=np.int64),
                public_row_id=np.array([0, 1], dtype=np.int64),
                values=np.array([0.2, 0.4], dtype=np.float64),
                value_column="value",
            ),
            batch_index=2,
        )

    row_groups = (("0",), ("1",), ("0", "1"), ("2",))
    summary, frequency = exact_summary_and_frequency_from_public_runs(
        identity_frame=pd.DataFrame({"scope": ["per_a", "per_b", "inter", "missing"]}),
        runs_path=runs_path,
        output_format=output_format,
        run_count=2,
        public_row_groups=row_groups,
        sparse=True,
    )
    summary_only = exact_summary_from_public_runs(
        identity_frame=pd.DataFrame({"scope": ["per_a", "per_b", "inter", "missing"]}),
        runs_path=runs_path,
        output_format=output_format,
        run_count=2,
        public_row_groups=row_groups,
        sparse=True,
    )
    identity_summary, identity_frequency = exact_summary_and_frequency_from_public_runs(
        identity_frame=pd.DataFrame({"scope": ["per_a", "per_b"]}),
        runs_path=runs_path,
        output_format=output_format,
        run_count=2,
        sparse=True,
    )
    bounded_summary, bounded_frequency = exact_summary_and_frequency_from_public_runs(
        identity_frame=pd.DataFrame({"scope": ["per_a", "per_b", "inter", "missing"]}),
        runs_path=runs_path,
        output_format=output_format,
        run_count=10**12,
        public_row_groups=row_groups,
        sparse=True,
    )

    assert summary["scope"].tolist() == ["per_a", "per_b", "inter", "missing"]
    assert summary["mean"].iloc[:3].tolist() == [0.35, 1.2, 0.775]
    assert summary["median"].iloc[:3].tolist() == [0.35, 1.2, 0.775]
    assert pd.isna(summary.loc[3, "mean"])
    assert summary_only["median"].iloc[:3].tolist() == [0.35, 1.2, 0.775]
    assert pd.isna(summary_only.loc[3, "median"])
    assert identity_summary["scope"].tolist() == ["per_a", "per_b"]
    assert frequency[:3].tolist() == [1.0, 0.5, 0.5]
    assert pd.isna(frequency[3])
    assert identity_frequency.tolist() == [1.0, 0.5]
    assert bounded_summary["median"].iloc[:3].tolist() == [0.35, 1.2, 0.775]
    assert pd.isna(bounded_summary.loc[3, "median"])
    assert bounded_frequency[:3].tolist() == [1.0, 0.5, 0.5]
    assert pd.isna(bounded_frequency[3])


def test_manifest_allocates_run_ids_and_round_trips(tmp_path: Path) -> None:
    run_id = allocate_run_id()
    assert run_id.startswith(RUN_ID_PREFIX)
    assert re.fullmatch(r"mc_[0-9a-f]{16}", run_id)
    compatibility_key = build_compatibility_key({"scope": "demo", "family": "asocc"})
    assert compatibility_key == build_compatibility_key({"family": "asocc", "scope": "demo"})

    manifest = build_manifest(
        family="asocc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("lcia_uncertainty", "projection_uncertainty"),
        completed_runs=2,
        status="running",
        run_id="mc_manual_test",
        requested_runs=5,
        mc_parameters={"mode": "fixed"},
        source_parameters={"lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}}},
        arguments={"project_name": "demo", "years": range(2005, 2007)},
        deterministic_prerequisites=({"scope_key": "scope-a"},),
        external_inputs=({"path": "external.csv"},),
        artifacts={
            "scope_manifest": tmp_path / "logs" / "scope_manifest.json",
            "asocc_runs": tmp_path / "runs.csv",
            "source_set": {"a", "b"},
            "public_output": {"run_columns": ["run_index", "asocc"]},
        },
        lineage={"source_inventory": {"public_rows": 2}},
        component_inventory={"role": "component_inventory", "component_name": "asocc"},
        convergence={"reached": True, "metrics": ("mean", "median")},
        compatibility_key=compatibility_key,
        compatibility_context={"active_sources": ["lcia_uncertainty"]},
    )
    assert manifest.active_sources == ("lcia_uncertainty", "projection_uncertainty")
    path = tmp_path / "scope_manifest.json"
    write_manifest(path=path, manifest=manifest)
    assert read_manifest(path=path) == manifest
    assert read_manifest(path=path).compatibility_key == compatibility_key
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["function"] == "uncertainty_asocc"
    assert payload["arguments"]["years"] == [2005, 2006]
    assert payload["execution"]["run_id"] == "mc_manual_test"
    assert isinstance(payload["execution"]["created_at"], str)
    assert "created_at_utc" not in payload["execution"]
    assert payload["artifacts"]["asocc_runs"].endswith("runs.csv")
    assert payload["artifacts"]["source_set"] == ["a", "b"]
    assert payload["artifacts"]["public_output"]["run_columns"] == ["run_index", "asocc"]
    assert payload["reuse"]["component_inventory"]["component_name"] == "asocc"
    assert payload["execution"]["convergence"] == {"metrics": ["mean", "median"], "reached": True}
    figure_path = tmp_path / "figure.png"
    figure_path.write_text("figure", encoding="utf-8")
    figure_manifest = manifest_with_figure_artifacts(
        manifest=manifest,
        figure_paths=[figure_path],
        figure_options={"polar": {"polar_years": [2005]}},
        figure_format={"format": "png", "dpi": 10},
        warning_messages=("figure warning", "figure warning"),
    )
    assert figure_manifest.lineage is not None
    latest_warning = figure_manifest.lineage["summary_records"][-1]
    assert latest_warning["severity"] == "WARNING"
    assert latest_warning["source"] == "figures"
    assert "figure warning" in latest_warning["message"]
    figure_report = uncertainty_report(
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=(),
            completed_runs=1,
            status="complete",
            artifacts={"scope_manifest": tmp_path / "report" / "logs" / "scope_manifest.json"},
            lineage=figure_manifest.lineage,
        ),
        reuse_status="computed",
    )
    assert "WARNING:" in str(figure_report)
    assert "figure warning" in str(figure_report)
    assert manifest_figure_artifacts_current(
        manifest=figure_manifest,
        figure_options={"polar": {"polar_years": [2005]}},
        figure_format={"format": "png", "dpi": 10},
    )
    missing_paths_manifest = replace(
        figure_manifest,
        artifacts={
            key: value for key, value in figure_manifest.artifacts.items() if key != "figure_paths"
        },
    )
    assert not manifest_figure_artifacts_current(
        manifest=missing_paths_manifest,
        figure_options={"polar": {"polar_years": [2005]}},
        figure_format={"format": "png", "dpi": 10},
    )

    generated = build_manifest(
        family="asocc",
        mode="convergence",
        output_format="parquet",
        active_sources=("lcia_uncertainty",),
    )
    assert generated.run_id.startswith(RUN_ID_PREFIX)


def test_run_registry_discovers_compatible_completed_runs_and_required_run_id(
    tmp_path: Path,
) -> None:
    compatibility_key = build_compatibility_key({"family": "asocc", "scope": "demo"})
    other_key = build_compatibility_key({"family": "asocc", "scope": "other"})
    first_root = tmp_path / "mc_first"
    second_root = tmp_path / "mc_second"
    incompatible_root = tmp_path / "mc_other"
    running_root = tmp_path / "mc_running"
    running_component_root = tmp_path / "mc_running_component"
    empty_root = tmp_path / "mc_empty"
    for root, completed_runs, run_id, run_signature, status, inventory in (
        (first_root, 2, "mc_first", compatibility_key, "complete", None),
        (second_root, 5, "mc_second", compatibility_key, "complete", None),
        (incompatible_root, 5, "mc_other", other_key, "complete", None),
        (running_root, 10, "mc_running", compatibility_key, "running", None),
        (
            running_component_root,
            3,
            "mc_running_component",
            compatibility_key,
            "running",
            component_inventory_payload(
                composite_family="acc",
                component_name="asocc",
                target_runs=3,
            ),
        ),
        (empty_root, 0, "mc_empty", compatibility_key, "complete", None),
    ):
        write_manifest(
            path=root / "logs" / "scope_manifest.json",
            manifest=build_manifest(
                family="asocc",
                mode="fixed",
                output_format="csv_compact",
                active_sources=("lcia_uncertainty",),
                completed_runs=completed_runs,
                status=status,
                run_id=run_id,
                compatibility_key=run_signature,
                component_inventory=inventory,
            ),
        )
    runs = compatible_completed_runs(
        monte_carlo_root=tmp_path,
        compatibility_key=compatibility_key,
    )
    assert [run.manifest.run_id for run in runs] == ["mc_second", "mc_first"]
    required = compatible_completed_run_for_id(
        monte_carlo_root=tmp_path,
        run_id="mc_first",
    )
    assert required is not None
    assert required.manifest.run_id == "mc_first"
    assert compatible_completed_run_for_id(monte_carlo_root=tmp_path, run_id=None) is None
    assert compatible_completed_run_for_id(monte_carlo_root=tmp_path, run_id="mc_missing") is None
    assert compatible_completed_run_for_id(monte_carlo_root=tmp_path, run_id="mc_running") is None
    assert compatible_completed_run_for_id(monte_carlo_root=tmp_path, run_id="mc_empty") is None
    appendable = compatible_completed_runs(
        monte_carlo_root=tmp_path,
        compatibility_key=compatibility_key,
        include_running_component_inventory=True,
    )
    assert [run.manifest.run_id for run in appendable] == [
        "mc_second",
        "mc_running_component",
        "mc_first",
    ]
    required_appendable = compatible_completed_run_for_id(
        monte_carlo_root=tmp_path,
        run_id="mc_running_component",
        include_running_component_inventory=True,
    )
    assert required_appendable is not None
    assert required_appendable.manifest.run_id == "mc_running_component"
    assert compatible_completed_run_for_id(monte_carlo_root=tmp_path, run_id=None) is None
    assert (
        compatible_completed_run_for_id(
            monte_carlo_root=tmp_path,
            run_id="mc_running",
            include_running_component_inventory=True,
        )
        is None
    )

    delete_root = tmp_path / "mc_delete"
    delete_root.mkdir()
    cleanup_monte_carlo_runs_for_refresh(
        monte_carlo_root=tmp_path,
        compatibility_key=compatibility_key,
        run_id="mc_delete",
    )
    assert not delete_root.exists()
    cleanup_monte_carlo_runs_for_refresh(
        monte_carlo_root=tmp_path,
        compatibility_key=build_compatibility_key({"family": "other"}),
        run_id=None,
    )


def test_run_reuse_policy_selects_completed_and_appendable_runs(tmp_path: Path) -> None:
    fixed = CompatibleMonteCarloRun(
        run_root=tmp_path / "mc_fixed",
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=(),
            status="complete",
            completed_runs=5,
            run_id="mc_fixed",
        ),
    )
    reached_convergence = CompatibleMonteCarloRun(
        run_root=tmp_path / "mc_reached",
        manifest=build_manifest(
            family="asocc",
            mode="convergence",
            output_format="csv_compact",
            active_sources=(),
            status="complete",
            completed_runs=4,
            run_id="mc_reached",
            mc_parameters={
                "mode": "convergence",
                "rtol": 0.01,
                "stable_runs": 10000,
                "convergence_statistics": ["mean", "median", "p25", "p75"],
            },
            convergence={"reached": True},
        ),
    )
    unreached_convergence = CompatibleMonteCarloRun(
        run_root=tmp_path / "mc_unreached",
        manifest=build_manifest(
            family="asocc",
            mode="convergence",
            output_format="csv_compact",
            active_sources=(),
            status="complete",
            completed_runs=3,
            run_id="mc_unreached",
            mc_parameters={
                "mode": "convergence",
                "rtol": 0.01,
                "stable_runs": 10000,
                "convergence_statistics": ["mean", "median", "p25", "p75"],
            },
            convergence={"reached": False},
        ),
    )
    running_component = CompatibleMonteCarloRun(
        run_root=tmp_path / "mc_running_component",
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=(),
            status="running",
            completed_runs=5,
            run_id="mc_running_component",
            component_inventory=component_inventory_payload(
                composite_family="acc",
                component_name="asocc",
                target_runs=5,
            ),
        ),
    )
    old_reached_convergence = CompatibleMonteCarloRun(
        run_root=tmp_path / "mc_old_reached",
        manifest=build_manifest(
            family="asocc",
            mode="convergence",
            output_format="csv_compact",
            active_sources=(),
            status="complete",
            completed_runs=2,
            run_id="mc_old_reached",
            mc_parameters={
                "mode": "convergence",
                "rtol": 0.05,
                "stable_runs": 10000,
                "convergence_statistics": ["mean", "median"],
            },
            convergence={"reached": True},
        ),
    )
    compatible = (fixed, reached_convergence, unreached_convergence, old_reached_convergence)
    convergence_parameters = {
        "mode": "convergence",
        "rtol": 0.01,
        "stable_runs": 10000,
        "convergence_statistics": ["mean", "median", "p25", "p75"],
    }

    assert (
        complete_run_with_requested_runs(
            compatible=compatible,
            requested_runs=2,
            mode="fixed",
        )
        == fixed
    )
    assert (
        complete_run_with_requested_runs(
            compatible=compatible,
            requested_runs=8,
            mode="fixed",
        )
        is None
    )
    assert (
        complete_run_with_requested_runs(
            compatible=compatible,
            requested_runs=5,
            mode="convergence",
            mc_parameters=convergence_parameters,
        )
        == reached_convergence
    )
    assert (
        complete_run_with_requested_runs(
            compatible=(old_reached_convergence,),
            requested_runs=5,
            mode="convergence",
            mc_parameters=convergence_parameters,
        )
        is None
    )
    assert (
        appendable_completed_run(
            compatible=compatible,
            mode="convergence",
            max_completed_runs=4,
        )
        == unreached_convergence
    )
    assert (
        appendable_completed_run(
            compatible=(running_component,),
            mode="fixed",
            max_completed_runs=5,
        )
        == running_component
    )


def test_run_reuse_policy_requires_matching_completed_sobol(tmp_path: Path) -> None:
    fixed_sobol_parameters = {
        "mode": "fixed",
        "n_base_samples": 4,
        "max_base_samples": 4,
        "rtol": 0.05,
        "abs_tol": 0.01,
        "scale_floor": 0.05,
        "convergence_targets": ["S1", "ST"],
        "confidence_level": 0.95,
        "confidence_resamples": 100,
    }
    fixed = CompatibleMonteCarloRun(
        run_root=tmp_path / "mc_fixed_sobol",
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=("source_a", "source_b"),
            status="complete",
            completed_runs=5,
            run_id="mc_fixed_sobol",
            sobol={
                "ran": True,
                "reached": False,
                "parameters": fixed_sobol_parameters,
            },
        ),
    )
    missing_sobol = CompatibleMonteCarloRun(
        run_root=tmp_path / "mc_missing_sobol",
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=("source_a", "source_b"),
            status="complete",
            completed_runs=5,
            run_id="mc_missing_sobol",
        ),
    )
    unreached_convergence = CompatibleMonteCarloRun(
        run_root=tmp_path / "mc_unreached_sobol",
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=("source_a", "source_b"),
            status="complete",
            completed_runs=5,
            run_id="mc_unreached_sobol",
            sobol={
                "ran": True,
                "reached": False,
                "parameters": {
                    **fixed_sobol_parameters,
                    "mode": "convergence",
                },
            },
        ),
    )
    reached_convergence = CompatibleMonteCarloRun(
        run_root=tmp_path / "mc_reached_sobol",
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            active_sources=("source_a", "source_b"),
            status="complete",
            completed_runs=5,
            run_id="mc_reached_sobol",
            sobol={
                "ran": True,
                "reached": True,
                "parameters": {
                    **fixed_sobol_parameters,
                    "mode": "convergence",
                },
            },
        ),
    )

    assert (
        complete_run_with_requested_sobol(
            compatible=(missing_sobol, fixed),
            requested_runs=5,
            mode="fixed",
            mc_parameters={},
            sobol_parameters=fixed_sobol_parameters,
        )
        == fixed
    )
    assert (
        complete_run_with_requested_sobol(
            compatible=(fixed,),
            requested_runs=5,
            mode="fixed",
            mc_parameters={},
            sobol_parameters={**fixed_sobol_parameters, "n_base_samples": 8},
        )
        is None
    )
    assert (
        complete_run_with_requested_sobol(
            compatible=(unreached_convergence, reached_convergence),
            requested_runs=5,
            mode="fixed",
            mc_parameters={},
            sobol_parameters={**fixed_sobol_parameters, "mode": "convergence"},
        )
        == reached_convergence
    )
