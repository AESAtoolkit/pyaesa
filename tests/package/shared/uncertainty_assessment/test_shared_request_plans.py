import pytest

from dataclasses import replace
import numpy as np

from pyaesa.shared.uncertainty_assessment.request.core import (
    BatchMemoryBlock,
    memory_bounded_batch_size,
    normalize_uncertainty_request,
    sparse_selected_run_memory_blocks,
)
from pyaesa.shared.runtime.memory import memory_bounded_rows, runtime_memory_budget
from pyaesa.shared.uncertainty_assessment.sobol.plan import normalize_sobol_plan
from pyaesa.shared.uncertainty_assessment.sobol.plan import selected_sobol_output_years
from pyaesa.shared.uncertainty_assessment.sobol.plan import sobol_plan_payload
from pyaesa.shared.uncertainty_assessment.sobol.plan import studied_output_years
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan
from pyaesa.shared.uncertainty_assessment.sobol.accumulator import (
    SobolIndexEstimate,
)
from pyaesa.shared.uncertainty_assessment.sobol.design import (
    iter_saltelli_chunks,
    saltelli_design,
)
from pyaesa.shared.uncertainty_assessment.sobol.diagnostics import (
    sobol_diagnostic_label,
    sobol_source_summary_estimator_range_pass,
)
from pyaesa.shared.uncertainty_assessment.sobol.runner import (
    EvaluatedSobolChunk,
    run_sobol_analysis,
)
from pyaesa.shared.uncertainty_assessment.sobol.summary import (
    sobol_source_summary_columns,
    sobol_source_summary_by_group,
)
from pyaesa.shared.uncertainty_assessment.request.sources import build_source_activation_plan


def test_request_normalization_rejects_unknown_mc_parameters() -> None:
    omitted_parameters = normalize_uncertainty_request(
        family="asocc",
        output_format="csv_compact",
        mc_parameters=None,
    )
    assert omitted_parameters.mode == "convergence"
    assert omitted_parameters.max_runs == 500_000

    empty_parameters = normalize_uncertainty_request(
        family="asocc",
        output_format="csv_compact",
        mc_parameters={},
    )
    assert empty_parameters.mode == "convergence"
    assert empty_parameters.max_runs == 500_000

    request = normalize_uncertainty_request(
        family="asocc",
        output_format=" CSV_COMPACT ",
        mc_parameters={"fixed": {"active": True, "n_runs": 20}, "convergence": {"active": False}},
    )
    assert request.family == "asocc"
    assert request.output_format == "csv_compact"
    assert request.n_runs == 20
    assert request.batch_size == 20
    default_batch = normalize_uncertainty_request(
        family="asocc",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": True, "n_runs": 50_000},
            "convergence": {"active": False},
        },
    )
    assert default_batch.mode == "fixed"
    assert default_batch.batch_size == 50_000

    convergence = normalize_uncertainty_request(
        family="asocc",
        output_format="parquet",
        mc_parameters={"fixed": {"active": False}, "convergence": {"active": True, "max_runs": 5}},
    )
    assert convergence.mode == "convergence"
    assert convergence.batch_size == 5
    assert convergence.max_runs == 5
    assert convergence.rtol == 0.05
    assert convergence.stable_runs == 10000
    assert convergence.convergence_statistics == ("mean",)

    default_convergence = normalize_uncertainty_request(
        family="asocc",
        output_format="csv_compact",
        mc_parameters={"fixed": {"active": False}, "convergence": {"active": True}},
    )
    assert default_convergence.max_runs == 500_000

    custom_convergence = normalize_uncertainty_request(
        family="asocc",
        output_format="parquet",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {
                "active": True,
                "max_runs": 5,
                "stable_runs": 2,
                "convergence_statistics": ["mean"],
            },
        },
    )
    assert custom_convergence.batch_size == 2
    assert custom_convergence.convergence_statistics == ("mean",)

    single_statistic = normalize_uncertainty_request(
        family="asocc",
        output_format="parquet",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 5, "convergence_statistics": "mean"},
        },
    )
    assert single_statistic.convergence_statistics == ("mean",)
    zero_tolerance = normalize_uncertainty_request(
        family="asocc",
        output_format="parquet",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 5, "rtol": 0.0},
        },
    )
    assert zero_tolerance.rtol == 0.0
    large_fixed = normalize_uncertainty_request(
        family="asocc",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": True, "n_runs": 100_000},
            "convergence": {"active": False},
        },
    )
    test_budget = 3_000_000_000
    assert (
        memory_bounded_batch_size(
            runtime=large_fixed,
            primary_block=BatchMemoryBlock("small_owner_values", row_count=10),
            memory_budget_bytes=test_budget,
        )
        == 100_000
    )
    assert (
        memory_bounded_batch_size(
            runtime=large_fixed,
            primary_block=BatchMemoryBlock("owner_values", row_count=100_000, array_count=24),
            memory_budget_bytes=test_budget,
        )
        == 156
    )
    assert (
        memory_bounded_batch_size(
            runtime=large_fixed,
            primary_block=BatchMemoryBlock("owner_values", row_count=100_000, array_count=24),
            extra_blocks=(BatchMemoryBlock("cumulative", row_count=100_000),),
            memory_budget_bytes=test_budget,
        )
        == 150
    )
    assert (
        memory_bounded_batch_size(
            runtime=replace(large_fixed, mode="convergence", batch_size=2000),
            primary_block=BatchMemoryBlock("owner_values", row_count=100_000, array_count=24),
            memory_budget_bytes=test_budget,
        )
        == 156
    )
    gib = 1024**3
    budget = runtime_memory_budget(
        minimal_working_block_bytes=1024,
        physical_memory_bytes=16 * gib,
        available_memory_bytes=8 * gib,
    )
    assert budget.budget_bytes == 7 * gib
    assert budget.operating_system_reserve_bytes == gib
    assert (
        memory_bounded_rows(
            bytes_per_row=24,
            working_arrays=2,
            memory_budget_bytes=480,
        )
        == 10
    )
    low_memory_budget = runtime_memory_budget(
        minimal_working_block_bytes=2 * gib,
        physical_memory_bytes=4 * gib,
        available_memory_bytes=1 * gib,
    )
    assert low_memory_budget.budget_bytes == 4 * gib

    for params, message in (
        ({"mode": "bad"}, "mode"),
        ({"fixed": {"active": True, "n_runs": 0}, "convergence": {"active": False}}, "n_runs"),
        (
            {"fixed": {"active": True, "n_runs": "bad"}, "convergence": {"active": False}},
            "n_runs",
        ),
        ({"n_runs": 20}, "n_runs"),
        ({"batch_size": 1}, "batch_size"),
        ({"fixed": []}, "fixed"),
        (
            {"fixed": {"active": True, "n_runs": 1, "extra": 1}, "convergence": {"active": False}},
            "Unsupported",
        ),
        (
            {"fixed": {"active": "yes", "n_runs": 1}, "convergence": {"active": False}},
            "active",
        ),
        (
            {"fixed": {"active": True, "n_runs": True}, "convergence": {"active": False}},
            "n_runs",
        ),
        ({"convergence": {"active": True, "rtol": -0.01}}, "rtol"),
        ({"convergence": {"active": True, "stable_runs": 0}}, "stable_runs"),
        (
            {"convergence": {"active": True, "convergence_statistics": ["std"]}},
            "convergence_statistics",
        ),
        (
            {"convergence": {"active": True, "convergence_statistics": 3}},
            "convergence_statistics",
        ),
        ({"seed": 123}, "seed"),
        ({"extra": 1}, "extra"),
        (
            {"fixed": {"active": True}, "convergence": {"active": True}},
            "mode",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            normalize_uncertainty_request(
                family="asocc",
                output_format="csv_compact",
                mc_parameters=params,
            )


def test_sparse_selected_run_memory_blocks_cover_common_sparse_retention() -> None:
    asr_blocks = sparse_selected_run_memory_blocks(
        prefix="asr",
        public_row_count=10,
        summary_row_count=4,
        filters_and_sorts_output=False,
    )
    assert {block.name for block in asr_blocks} == {
        "asr_sparse_source_window_columns",
        "asr_sparse_source_decoded_columns",
        "asr_sparse_source_reader_work_columns",
        "asr_sparse_source_reader_masks",
        "asr_sparse_output_window_columns",
        "asr_sparse_output_concat_columns",
        "asr_sparse_output_render_columns",
        "asr_sparse_summary_source_positions",
        "asr_sparse_summary_group_ids",
        "asr_sparse_summary_values",
    }
    assert sum(block.row_count * block.array_count * block.dtype_bytes for block in asr_blocks) == (
        10 * 8 * 20
    ) + (10 * 1 * 3) + (4 * 8 * 3)

    acc_blocks = sparse_selected_run_memory_blocks(
        prefix="acc",
        public_row_count=10,
        summary_row_count=4,
        filters_and_sorts_output=True,
    )
    acc_names = {block.name for block in acc_blocks}
    assert {
        "acc_sparse_output_finite_mask",
        "acc_sparse_output_sort_order",
        "acc_sparse_output_sorted_columns",
    } < acc_names
    assert sum(block.row_count * block.array_count * block.dtype_bytes for block in acc_blocks) == (
        10 * 8 * 24
    ) + (10 * 1 * 4) + (4 * 8 * 3)


def test_source_activation_plan_keeps_allowed_order() -> None:
    allowed = ("lcia_uncertainty", "projection_uncertainty", "reference_year_uncertainty")
    default_plan = build_source_activation_plan(
        uncertainty_config=None,
        allowed_sources=allowed,
        default_sources=("projection_uncertainty",),
    )
    assert default_plan.names == ("projection_uncertainty",)

    plan = build_source_activation_plan(
        uncertainty_config={
            "reference_year_uncertainty": {"candidate_years": [2019, 2020]},
            "lcia_uncertainty": {"active": True},
            "projection_uncertainty": {"active": False},
        },
        allowed_sources=allowed,
    )
    assert plan.names == ("lcia_uncertainty", "reference_year_uncertainty")
    assert plan.is_active("lcia_uncertainty")
    assert plan.parameters_for("reference_year_uncertainty") == {"candidate_years": [2019, 2020]}
    assert plan.parameters_for("projection_uncertainty") == {}

    partial_with_defaults = build_source_activation_plan(
        uncertainty_config={"lcia_uncertainty": {"active": True}},
        allowed_sources=allowed,
        default_sources=("projection_uncertainty", "reference_year_uncertainty"),
    )
    assert partial_with_defaults.names == (
        "lcia_uncertainty",
        "projection_uncertainty",
        "reference_year_uncertainty",
    )
    disabled_default = build_source_activation_plan(
        uncertainty_config={"projection_uncertainty": {"active": False}},
        allowed_sources=allowed,
        default_sources=("projection_uncertainty", "reference_year_uncertainty"),
    )
    assert disabled_default.names == ("reference_year_uncertainty",)

    alternate_source = build_source_activation_plan(
        uncertainty_config={
            "lcia_uncertainty": {"active": True, "alternate_source": "alt_source"},
        },
        allowed_sources=allowed,
    )
    assert alternate_source.parameters_for("lcia_uncertainty") == {"source": "alt_source"}
    omitted_alternate = build_source_activation_plan(
        uncertainty_config={
            "lcia_uncertainty": {"active": True, "alternate_source": None},
        },
        allowed_sources=allowed,
    )
    assert omitted_alternate.parameters_for("lcia_uncertainty") == {}

    with pytest.raises(ValueError, match="uncertainty_config"):
        build_source_activation_plan(
            uncertainty_config=[],
            allowed_sources=allowed,
        )
    with pytest.raises(ValueError, match="unknown"):
        build_source_activation_plan(
            uncertainty_config={"unknown": True},
            allowed_sources=allowed,
        )
    with pytest.raises(ValueError, match="lcia_uncertainty"):
        build_source_activation_plan(
            uncertainty_config={"lcia_uncertainty": "yes"},
            allowed_sources=allowed,
        )
    with pytest.raises(ValueError, match="lcia_uncertainty"):
        build_source_activation_plan(
            uncertainty_config={"lcia_uncertainty": True},
            allowed_sources=allowed,
        )
    with pytest.raises(ValueError, match="active"):
        build_source_activation_plan(
            uncertainty_config={"lcia_uncertainty": {"active": "yes"}},
            allowed_sources=allowed,
        )


def test_sobol_plan_normalization() -> None:
    assert studied_output_years(2024) == (2024,)
    assert studied_output_years([2025, 2024, 2024]) == (2024, 2025)

    disabled = normalize_sobol_plan(sobol_parameters=None)
    assert not disabled.enabled
    assert disabled.n_base_samples == 0

    default_convergence = normalize_sobol_plan(sobol_parameters={})
    assert default_convergence.n_base_samples == 128
    assert default_convergence.max_base_samples == 1048576

    disabled = normalize_sobol_plan(sobol_parameters={"active": False})
    assert not disabled.enabled
    assert disabled.mode == "convergence"

    enabled = normalize_sobol_plan(
        sobol_parameters={
            "convergence": {"active": True, "max_base_samples": 1024},
            "fixed": {"active": False},
        }
    )
    assert enabled.enabled
    assert enabled.mode == "convergence"
    assert enabled.n_base_samples == 128
    assert enabled.max_base_samples == 1024
    assert sobol_plan_payload(plan=enabled) == {
        "mode": "convergence",
        "n_base_samples": 128,
        "max_base_samples": 1024,
        "rtol": 0.05,
        "abs_tol": 0.01,
        "scale_floor": 0.05,
        "convergence_targets": ["S1", "ST"],
        "confidence_level": 0.95,
        "confidence_resamples": 100,
    }

    selected_years = normalize_sobol_plan(
        sobol_parameters={"sobol_years": [2020, 2020, 2021]},
        available_years=[2020, 2021, 2022],
    )
    assert selected_years.sobol_years == (2020, 2021)
    assert selected_sobol_output_years(
        plan=selected_years,
        available_years=(2020, 2021, 2022),
    ) == (2020, 2021)
    assert sobol_plan_payload(plan=selected_years)["sobol_years"] == [2020, 2021]
    assert selected_sobol_output_years(
        plan=default_convergence,
        available_years=(2020, 2021, 2022),
    ) == (2020, 2022)
    ranged_years = normalize_sobol_plan(sobol_parameters={"sobol_years": range(2020, 2023)})
    assert ranged_years.sobol_years == (2020, 2021, 2022)
    with pytest.raises(ValueError, match="sobol_years.*studied years"):
        normalize_sobol_plan(
            sobol_parameters={"sobol_years": [2020, 2025]},
            available_years=[2020, 2021],
        )
    for value in (-1, True):
        with pytest.raises(ValueError, match="n_base_samples"):
            normalize_sobol_plan(
                sobol_parameters={
                    "fixed": {"active": True, "n_base_samples": value},
                    "convergence": {"active": False},
                }
            )
    with pytest.raises(ValueError, match="power of two"):
        normalize_sobol_plan(
            sobol_parameters={
                "fixed": {"active": True, "n_base_samples": 1000},
                "convergence": {"active": False},
            }
        )
    with pytest.raises(ValueError, match="max_base_samples"):
        normalize_sobol_plan(
            sobol_parameters={
                "fixed": {"active": True, "n_base_samples": 512},
                "convergence": {"active": False, "max_base_samples": 128},
            }
        )
    with pytest.raises(ValueError, match="active"):
        normalize_sobol_plan(sobol_parameters={"active": "yes"})
    with pytest.raises(ValueError, match="seed"):
        normalize_sobol_plan(sobol_parameters={"active": False, "seed": 42})
    with pytest.raises(ValueError, match="mode"):
        normalize_sobol_plan(
            sobol_parameters={
                "fixed": {"active": True, "n_base_samples": 512},
                "convergence": {"active": True},
            }
        )
    with pytest.raises(ValueError, match="fixed"):
        normalize_sobol_plan(sobol_parameters={"fixed": []})
    with pytest.raises(ValueError, match="extra"):
        normalize_sobol_plan(sobol_parameters={"fixed": {"active": True, "extra": 1}})
    with pytest.raises(ValueError, match="active"):
        normalize_sobol_plan(sobol_parameters={"fixed": {"active": "yes"}})
    with pytest.raises(ValueError, match="seed"):
        normalize_sobol_plan(sobol_parameters={"seed": 42})
    with pytest.raises(ValueError, match="rtol"):
        normalize_sobol_plan(sobol_parameters={"convergence": {"rtol": 0}})
    with pytest.raises(ValueError, match="rtol"):
        normalize_sobol_plan(sobol_parameters={"convergence": {"rtol": True}})
    for old_key in (
        "enabled",
        "targets",
        "abs_tol",
        "scale_floor",
        "confidence_level",
        "confidence_resamples",
    ):
        with pytest.raises(ValueError, match=old_key):
            normalize_sobol_plan(sobol_parameters={old_key: 1})
    for value in ("2020", 2020, [], [True], [0]):
        with pytest.raises(ValueError, match="sobol_years"):
            normalize_sobol_plan(sobol_parameters={"sobol_years": value})


def test_sobol_design_and_estimators() -> None:
    design = saltelli_design(n_base_samples=8, dimension_count=2)
    assert design.a.shape == (8, 2)
    assert design.b.shape == (8, 2)
    chunk = next(iter_saltelli_chunks(design=design, chunk_rows=3))
    assert len(chunk.ab) == 2
    assert chunk.ab[0].shape == (3, 2)


def test_sobol_generic_analysis_runner() -> None:
    identity = pytest.importorskip("pandas").DataFrame({"public_row_id": [0]})

    def evaluate(chunk):
        units = np.vstack((chunk.a, chunk.b, *chunk.ab))
        values = units[:, [0]] + units[:, [1]]
        a_stop = chunk.a.shape[0]
        b_stop = a_stop + chunk.b.shape[0]
        start = b_stop
        ab_values = []
        for block in chunk.ab:
            stop = start + block.shape[0]
            ab_values.append(values[start:stop])
            start = stop
        return EvaluatedSobolChunk(
            identity=identity,
            a_values=values[:a_stop],
            b_values=values[a_stop:b_stop],
            ab_values=tuple(ab_values),
        )

    fixed = run_sobol_analysis(
        plan=SobolPlan(
            enabled=True,
            mode="fixed",
            n_base_samples=8,
            max_base_samples=8,
            rtol=0.05,
        ),
        dimension_names=("a", "b"),
        evaluate=evaluate,
    )
    assert fixed.status["mode"] == "fixed"
    assert fixed.status["n_base_samples"] == 8
    assert fixed.indices["source_name"].tolist() == ["a", "b"]
    assert fixed.indices.columns.tolist() == [
        "public_row_id",
        "source_name",
        "sobol_output_variance",
        "S1",
        "S1_confidence_half_width",
        "ST",
        "ST_confidence_half_width",
        "ST_minus_S1",
        "estimator_diagnostic",
    ]
    assert all(fixed.indices["sobol_output_variance"].notna().tolist())
    assert all(fixed.indices["estimator_diagnostic"].notna().tolist())
    assert fixed.source_summary.columns.tolist() == [
        "source_name",
        "output_count",
        "defined_output_count",
        "undefined_output_count",
        "variance_weight_sum",
        "variance_weighted_S1",
        "variance_weighted_S1_confidence_half_width",
        "variance_weighted_ST",
        "variance_weighted_ST_confidence_half_width",
        "variance_weighted_ST_minus_S1",
        "confidence_level",
        "estimator_diagnostics_pass",
        "diagnostic_output_count",
        "negative_S1_count",
        "ST_below_S1_count",
        "above_one_count",
    ]
    assert fixed.source_summary["undefined_output_count"].tolist() == [0, 0]
    assert fixed.source_summary["diagnostic_output_count"].sum() == 0

    single_base = run_sobol_analysis(
        plan=SobolPlan(
            enabled=True,
            mode="fixed",
            n_base_samples=1,
            max_base_samples=1,
            rtol=0.05,
        ),
        dimension_names=("a", "b"),
        evaluate=evaluate,
    )
    assert single_base.status["n_base_samples"] == 1

    chunked = run_sobol_analysis(
        plan=SobolPlan(
            enabled=True,
            mode="fixed",
            n_base_samples=256,
            max_base_samples=256,
            rtol=0.05,
        ),
        dimension_names=("a", "b"),
        evaluate=evaluate,
        max_base_chunk_rows=8,
    )
    assert chunked.status["n_base_samples"] == 256
    assert chunked.indices["source_name"].tolist() == ["a", "b"]
    assert chunked.indices["estimator_diagnostic"].tolist() == ["ok", "ok"]

    convergence = run_sobol_analysis(
        plan=SobolPlan(
            enabled=True,
            mode="convergence",
            n_base_samples=4,
            max_base_samples=16,
            rtol=0.000000001,
        ),
        dimension_names=("a", "b"),
        evaluate=evaluate,
    )
    assert convergence.status["mode"] == "convergence"
    assert not convergence.status["reached"]
    assert convergence.status["convergence_monitor"] == "selected_scope_source_confidence_interval"
    assert "confidence_precision_pass" in convergence.status
    assert "diagnostic_output_count" in convergence.status
    assert convergence.status["n_base_samples"] == 16

    reached = run_sobol_analysis(
        plan=SobolPlan(
            enabled=True,
            mode="convergence",
            n_base_samples=32,
            max_base_samples=128,
            rtol=10.0,
        ),
        dimension_names=("a", "b"),
        evaluate=evaluate,
    )
    assert reached.status["reached"]
    assert reached.status["n_base_samples"] == 64

    def evaluate_constant(chunk):
        units = np.vstack((chunk.a, chunk.b, *chunk.ab))
        values = np.zeros((units.shape[0], 1), dtype=np.float64)
        a_stop = chunk.a.shape[0]
        b_stop = a_stop + chunk.b.shape[0]
        start = b_stop
        ab_values = []
        for block in chunk.ab:
            stop = start + block.shape[0]
            ab_values.append(values[start:stop])
            start = stop
        return EvaluatedSobolChunk(
            identity=identity,
            a_values=values[:a_stop],
            b_values=values[a_stop:b_stop],
            ab_values=tuple(ab_values),
        )

    constant = run_sobol_analysis(
        plan=SobolPlan(
            enabled=True,
            mode="convergence",
            n_base_samples=4,
            max_base_samples=16,
            rtol=0.05,
        ),
        dimension_names=("a", "b"),
        evaluate=evaluate_constant,
    )
    assert constant.status["reached"]
    assert bool(constant.indices[["S1", "ST"]].isna().to_numpy().all())
    assert constant.source_summary["defined_output_count"].tolist() == [0, 0]
    assert bool(
        constant.source_summary[["variance_weighted_S1", "variance_weighted_ST"]]
        .isna()
        .to_numpy()
        .all()
    )

    def evaluate_invalid(chunk):
        row_count = np.vstack((chunk.a, chunk.b, *chunk.ab)).shape[0]
        a_values = np.zeros((chunk.a.shape[0], 1), dtype=np.float64)
        b_values = np.ones((chunk.b.shape[0], 1), dtype=np.float64)
        ab_values = tuple(
            np.full((block.shape[0], 1), -10.0 if index == 0 else 10.0)
            for index, block in enumerate(chunk.ab)
        )
        assert row_count == chunk.a.shape[0] + chunk.b.shape[0] + sum(
            block.shape[0] for block in chunk.ab
        )
        return EvaluatedSobolChunk(
            identity=identity,
            a_values=a_values,
            b_values=b_values,
            ab_values=ab_values,
        )

    invalid = run_sobol_analysis(
        plan=SobolPlan(
            enabled=True,
            mode="convergence",
            n_base_samples=4,
            max_base_samples=8,
            rtol=0.05,
        ),
        dimension_names=("a", "b"),
        evaluate=evaluate_invalid,
    )
    assert not invalid.status["reached"]
    diagnostic_count = invalid.status["diagnostic_output_count"]
    assert isinstance(diagnostic_count, int)
    assert diagnostic_count > 0
    source_summary_range_pass = invalid.status["source_summary_range_pass"]
    assert isinstance(source_summary_range_pass, bool)
    assert not source_summary_range_pass
    assert any(invalid.indices["estimator_diagnostic"].ne("ok").tolist())


def test_sobol_grouped_source_summary_and_diagnostics() -> None:
    pandas = pytest.importorskip("pandas")
    identity = pandas.DataFrame(
        {
            "year": [2030, 2030, 2035],
            "r_c": ["FR", "FR", "FR"],
        }
    )
    s1 = np.array([[0.2, -0.1, 1.2], [0.8, 0.4, 0.3]])
    st = np.array([[0.4, 0.2, 1.1], [0.9, 0.2, 0.5]])
    variance = np.array([2.0, 2.0, 0.0])
    estimates = SobolIndexEstimate(
        s1=s1,
        st=st,
        variance=variance,
        s1_confidence_half_width=np.full_like(s1, 0.01),
        st_confidence_half_width=np.full_like(st, 0.02),
        s1_resamples=np.stack((s1, np.full_like(s1, np.nan))),
        st_resamples=np.stack((st, st + 0.01)),
    )

    summary = sobol_source_summary_by_group(
        identity=identity,
        group_columns=("year", "r_c"),
        dimension_names=("a", "b"),
        estimates=estimates,
        confidence_level=0.95,
    )

    group_2030 = summary.loc[summary["year"].eq(2030)].reset_index(drop=True)
    assert group_2030["source_name"].tolist() == ["a", "b"]
    assert group_2030["diagnostic_output_count"].tolist() == [1, 1]
    assert group_2030["negative_S1_count"].tolist() == [1, 0]
    assert group_2030["ST_below_S1_count"].tolist() == [0, 1]
    assert summary.loc[summary["year"].eq(2035), "defined_output_count"].tolist() == [0, 0]
    global_summary = sobol_source_summary_by_group(
        identity=identity,
        group_columns=(),
        dimension_names=("a", "b"),
        estimates=estimates,
        confidence_level=0.95,
    )
    assert global_summary["source_name"].tolist() == ["a", "b"]
    assert (
        sobol_diagnostic_label(
            s1=0.5,
            st=0.1,
            s1_confidence_half_width=0.0,
            st_confidence_half_width=0.0,
            variance=1.0,
        )
        == "ST_below_S1"
    )
    assert sobol_source_summary_columns(
        selector_columns=("year", "r_c"),
        invariant_columns=("lcia_method",),
    )[:5] == ["summary_level", "year", "r_c", "source_name", "lcia_method"]


def test_sobol_range_diagnostic_allows_undefined_precision_inside_bounds() -> None:
    pandas = pytest.importorskip("pandas")
    plan = normalize_sobol_plan(sobol_parameters={})
    summary = pandas.DataFrame(
        {
            "variance_weight_sum": [1.0, 1.0],
            "variance_weighted_S1": [0.25, -0.25],
            "variance_weighted_S1_confidence_half_width": [np.nan, np.nan],
            "variance_weighted_ST": [0.5, 0.4],
            "variance_weighted_ST_confidence_half_width": [np.nan, np.nan],
        }
    )

    assert sobol_source_summary_estimator_range_pass(
        source_summary=summary.iloc[[0]],
        plan=plan,
    )
    assert not sobol_source_summary_estimator_range_pass(
        source_summary=summary.iloc[[1]],
        plan=plan,
    )


def test_sobol_convergence_streams_checkpoint_designs() -> None:
    identity = pytest.importorskip("pandas").DataFrame({"public_row_id": [0]})
    evaluated_rows: list[int] = []

    def evaluate(chunk):
        evaluated_rows.append(chunk.a.shape[0])
        units = np.vstack((chunk.a, chunk.b, *chunk.ab))
        values = units[:, [0]] + units[:, [1]]
        a_stop = chunk.a.shape[0]
        b_stop = a_stop + chunk.b.shape[0]
        start = b_stop
        ab_values = []
        for block in chunk.ab:
            stop = start + block.shape[0]
            ab_values.append(values[start:stop])
            start = stop
        return EvaluatedSobolChunk(
            identity=identity,
            a_values=values[:a_stop],
            b_values=values[a_stop:b_stop],
            ab_values=tuple(ab_values),
        )

    result = run_sobol_analysis(
        plan=SobolPlan(
            enabled=True,
            mode="convergence",
            n_base_samples=4,
            max_base_samples=16,
            rtol=0.000000001,
        ),
        dimension_names=("a", "b"),
        evaluate=evaluate,
    )

    assert result.status["n_base_samples"] == 16
    assert evaluated_rows == [1, 3, 4, 8]
