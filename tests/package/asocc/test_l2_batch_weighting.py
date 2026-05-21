"""Equivalence tests for batched L2 impact weighting kernels."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pyaesa.asocc.methods.compute_l2 import (
    apply_l1_weights_to_preweighted,
)
from pyaesa.asocc.orchestration.yearly.l2.l2_batch_weighting import (
    batch_weight_preweighted_ar_matrix,
    batch_weight_preweighted_ut_matrix,
    batch_weight_reuse_preweighted_ut_matrix,
)
from pyaesa.asocc.orchestration.yearly.l2.l2_reuse_frames import (
    _combine_l2_reuse_year_frames,
)


def test_batch_weight_preweighted_ut_matches_per_impact_baseline():
    """UT batch weighting must match the prior per impact weighting contract."""
    year = 2019
    pre_index = pd.MultiIndex.from_arrays(
        [
            ["FR", "FR", "US", "US", "FR", "US"],
            ["Electricity", "Steel", "Electricity", "Steel", "Electricity", "Steel"],
            ["FR", "FR", "US", "US", "US", "FR"],
            ["FR", "US", "FR", "US", "DE", "DE"],
        ],
        names=["r_p", "s_p", "r_c", "r_f"],
    )
    pre_weighted = pd.DataFrame(
        {year: [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]},
        index=pre_index,
    )
    impact_weights: list[tuple[str, pd.Series]] = [
        ("A", pd.Series({"FR": 0.2, "US": 0.8, "DE": np.nan}, name="w")),
        ("B", pd.Series({"FR": 0.5, "US": 0.5, "DE": 1.0}, name="w")),
    ]
    required = ("s_p", "r_c", "r_p")

    expected_agg_frames: list[pd.DataFrame] = []
    expected_contrib_frames: list[pd.DataFrame] = []
    expected_impact_names: list[str] = []
    for impact_name, weights in impact_weights:
        expected_impact_names.append(impact_name)
        contribution = pre_weighted.iloc[:, 0].mul(weights, level="r_f").to_frame(year)
        aggregated = (
            contribution.iloc[:, 0]
            .groupby(
                level=list(required),
                sort=False,
            )
            .sum(min_count=1)
            .to_frame(year)
        )
        expected_contrib_frames.append(contribution)
        expected_agg_frames.append(aggregated)
    expected_aggregated = pd.concat(
        expected_agg_frames,
        keys=expected_impact_names,
        names=["impact"],
    )
    expected_contribution = pd.concat(
        expected_contrib_frames,
        keys=expected_impact_names,
        names=["impact"],
    )

    matrix_aggregated, matrix_contribution = batch_weight_preweighted_ut_matrix(
        pre_weighted=pre_weighted,
        impact_names=("A", "B"),
        weight_index=pd.Index(["FR", "US", "DE"], name="r_f"),
        weight_values=np.array([[0.2, 0.8, np.nan], [0.5, 0.5, 1.0]], dtype=np.float64),
        weight_axis="r_f",
        required_indices=required,
        year=year,
        include_contribution=True,
    )
    assert matrix_contribution is not None
    assert_frame_equal(matrix_aggregated, expected_aggregated)
    assert_frame_equal(matrix_contribution, expected_contribution)


def test_batch_weight_reuse_preweighted_ut_matches_per_reuse_baseline():
    """Historical reuse batching must match one weighted frame per L2 reuse year."""
    year = 2030
    pre_index = pd.MultiIndex.from_arrays(
        [
            ["FR", "FR", "US", "US"],
            ["Electricity", "Steel", "Electricity", "Steel"],
            ["FR", "US", "FR", "US"],
        ],
        names=["r_p", "s_p", "r_f"],
    )
    preweights_by_l2_reuse_year = [
        (
            2019,
            pd.DataFrame({year: [1.0, 2.0, np.nan, 4.0]}, index=pre_index),
        ),
        (
            2020,
            pd.DataFrame({year: [3.0, 5.0, 7.0, np.nan]}, index=pre_index),
        ),
    ]
    required = ("s_p", "r_p")
    impact_names = ("A", "B")
    weight_index = pd.Index(["FR", "US"], name="r_f")
    weight_values = np.array([[0.2, 0.8], [0.5, 0.5]], dtype=np.float64)
    baseline_results = []
    baseline_contribs = []
    for l2_reuse_year, preweight in preweights_by_l2_reuse_year:
        aggregated, contribution = batch_weight_preweighted_ut_matrix(
            pre_weighted=preweight,
            impact_names=impact_names,
            weight_index=weight_index,
            weight_values=weight_values,
            weight_axis="r_f",
            required_indices=required,
            year=year,
            include_contribution=True,
        )
        baseline_results.append((l2_reuse_year, aggregated))
        assert contribution is not None
        baseline_contribs.append((l2_reuse_year, contribution))

    actual, actual_contribution = batch_weight_reuse_preweighted_ut_matrix(
        preweights_by_l2_reuse_year=preweights_by_l2_reuse_year,
        impact_names=impact_names,
        weight_index=weight_index,
        weight_values=weight_values,
        weight_axis="r_f",
        required_indices=required,
        year=year,
        include_contribution=True,
        reference_year=2015,
    )
    expected = _combine_l2_reuse_year_frames(
        frames_by_l2_reuse_year=baseline_results,
        reference_year=2015,
    )
    expected_contribution = _combine_l2_reuse_year_frames(
        frames_by_l2_reuse_year=baseline_contribs,
        reference_year=2015,
    )

    assert actual_contribution is not None
    assert_frame_equal(actual, expected)
    assert_frame_equal(actual_contribution, expected_contribution)

    actual_no_contribution, absent_contribution = batch_weight_reuse_preweighted_ut_matrix(
        preweights_by_l2_reuse_year=preweights_by_l2_reuse_year,
        impact_names=impact_names,
        weight_index=weight_index,
        weight_values=weight_values,
        weight_axis="r_f",
        required_indices=required,
        year=year,
        include_contribution=False,
        reference_year=None,
    )
    expected_no_reference = _combine_l2_reuse_year_frames(
        frames_by_l2_reuse_year=baseline_results,
        reference_year=None,
    )
    assert absent_contribution is None
    assert_frame_equal(actual_no_contribution, expected_no_reference)


def test_batch_weight_preweighted_ar_matches_per_impact_baseline():
    """AR batch weighting must match per impact sliced weighting outputs."""
    year = 2019
    pre_index = pd.MultiIndex.from_arrays(
        [
            ["A", "A", "A", "B", "B", "B"],
            ["FR", "FR", "US", "FR", "US", "US"],
            ["Electricity", "Steel", "Electricity", "Steel", "Electricity", "Steel"],
            ["FR", "US", "FR", "FR", "US", "DE"],
        ],
        names=["impact", "r_p", "s_p", "r_f"],
    )
    pre_weighted = pd.DataFrame(
        {year: [1.0, 2.0, 3.0, 1.5, np.nan, 4.0]},
        index=pre_index,
    )
    impact_weights: list[tuple[str, pd.Series]] = [
        ("A", pd.Series({"FR": 0.4, "US": 0.6}, name="w")),
        ("B", pd.Series({"FR": 0.2, "US": 0.3, "DE": 0.5}, name="w")),
    ]
    required = ("impact", "s_p", "r_p")

    expected_frames: list[pd.DataFrame] = []
    for impact, weights in impact_weights:
        pre_for_impact = pre_weighted.loc[pre_weighted.index.get_level_values("impact") == impact]
        expected_frames.append(
            apply_l1_weights_to_preweighted(
                l2_method="AR(E^{CBA_FD})",
                fu_code="L2.a.a",
                year=year,
                pre_weighted=pre_for_impact,
                l1_weights=weights,
                weight_axis="r_f",
                required_indices=required,
            )
        )
    expected = pd.concat(expected_frames)

    matrix_actual = batch_weight_preweighted_ar_matrix(
        pre_weighted=pre_weighted,
        impact_names=("A", "B"),
        weight_index=pd.Index(["FR", "US", "DE"], name="r_f"),
        weight_values=np.array([[0.4, 0.6, np.nan], [0.2, 0.3, 0.5]], dtype=np.float64),
        impact_level="impact",
        weight_axis="r_f",
        required_indices=required,
        year=year,
    )

    assert_frame_equal(matrix_actual, expected)


def test_l2_batch_weighting_error_and_edge_paths() -> None:
    year = 2019
    idx = pd.MultiIndex.from_arrays(
        [["FR", "US"], ["A", "A"], ["FR", "US"], ["FR", "US"]],
        names=["r_p", "s_p", "r_c", "r_f"],
    )
    pre_weighted = pd.DataFrame({year: [1.0, 2.0]}, index=idx)
    aggregated, contribution = batch_weight_preweighted_ut_matrix(
        pre_weighted=pre_weighted,
        impact_names=("A",),
        weight_index=pd.Index(["FR", "US"], name="r_f"),
        weight_values=np.array([[0.2, 0.8]], dtype=np.float64),
        weight_axis="r_f",
        required_indices=("r_p", "s_p"),
        year=year,
        include_contribution=False,
    )
    assert contribution is None
    assert not aggregated.empty

    all_nan_aggregated, all_nan_contribution = batch_weight_preweighted_ut_matrix(
        pre_weighted=pd.DataFrame({year: [np.nan, np.nan]}, index=idx),
        impact_names=("A",),
        weight_index=pd.Index(["FR", "US"], name="r_f"),
        weight_values=np.array([[0.2, 0.8]], dtype=np.float64),
        weight_axis="r_f",
        required_indices=("r_p", "s_p"),
        year=year,
        include_contribution=False,
    )
    assert all_nan_contribution is None
    assert np.isnan(all_nan_aggregated.iloc[:, 0].to_numpy(dtype=np.float64)).all()

    all_nan_ar = batch_weight_preweighted_ar_matrix(
        pre_weighted=pd.DataFrame(
            {year: [np.nan]},
            index=pd.MultiIndex.from_arrays(
                [["A"], ["FR"], ["A"], ["FR"]],
                names=["impact", "r_p", "s_p", "r_f"],
            ),
        ),
        impact_names=("A",),
        weight_index=pd.Index(["FR"], name="r_f"),
        weight_values=np.array([[0.2]], dtype=np.float64),
        impact_level="impact",
        weight_axis="r_f",
        required_indices=("impact", "s_p", "r_p"),
        year=year,
    )
    assert np.isnan(all_nan_ar.iloc[:, 0].to_numpy(dtype=np.float64)).all()

    with pytest.raises(ValueError, match="Unknown impacts"):
        batch_weight_preweighted_ar_matrix(
            pre_weighted=pd.DataFrame(
                {year: [1.0]},
                index=pd.MultiIndex.from_arrays(
                    [["C"], ["FR"], ["A"], ["FR"]],
                    names=["impact", "r_p", "s_p", "r_f"],
                ),
            ),
            impact_names=("A",),
            weight_index=pd.Index(["FR", "US"], name="r_f"),
            weight_values=np.array([[0.2, 0.8]], dtype=np.float64),
            impact_level="impact",
            weight_axis="r_f",
            required_indices=("impact", "s_p", "r_p"),
            year=year,
        )


def test_l2_batch_weighting_cache_and_missing_alignment_branches() -> None:
    idx = pd.MultiIndex.from_arrays(
        [["FR", "US"], ["A", "A"]],
        names=["r_p", "s_p"],
    )
    pre_weighted = pd.DataFrame({2019: [1.0, 2.0]}, index=idx)
    plan_cache = {}
    first_result, first_contribution = batch_weight_preweighted_ut_matrix(
        pre_weighted=pre_weighted,
        impact_names=("A",),
        weight_index=pd.Index(["FR", "US"], name="r_p"),
        weight_values=np.array([[0.2, 0.8]], dtype=np.float64),
        weight_axis="r_p",
        required_indices=("r_p", "s_p"),
        year=2019,
        include_contribution=True,
        plan_cache=plan_cache,
    )
    second_result, second_contribution = batch_weight_preweighted_ut_matrix(
        pre_weighted=pre_weighted,
        impact_names=("A",),
        weight_index=pd.Index(["FR", "US"], name="r_p"),
        weight_values=np.array([[0.2, 0.8]], dtype=np.float64),
        weight_axis="r_p",
        required_indices=("r_p", "s_p"),
        year=2019,
        include_contribution=True,
        plan_cache=plan_cache,
    )
    assert plan_cache
    assert_frame_equal(second_result, first_result)
    assert first_contribution is not None
    assert second_contribution is not None
    assert_frame_equal(second_contribution, first_contribution)

    pre_missing = pd.DataFrame(
        {2019: [1.0, 2.0]},
        index=pd.MultiIndex.from_arrays(
            [["FR", "US"], ["A", "A"]],
            names=["r_p", "s_p"],
        ),
    )
    missing_aggregated, missing_contribution = batch_weight_preweighted_ut_matrix(
        pre_weighted=pre_missing,
        impact_names=("A",),
        weight_index=pd.Index(["FR"], name="r_p"),
        weight_values=np.array([[0.2]], dtype=np.float64),
        weight_axis="r_p",
        required_indices=("r_p",),
        year=2019,
        include_contribution=True,
    )
    assert missing_contribution is not None
    assert missing_aggregated.iloc[:, 0].tolist()[0] == pytest.approx(0.2)
    assert np.isnan(missing_aggregated.iloc[:, 0].tolist()[1])

    year = 2019
    pre_weighted = pd.DataFrame(
        {year: [1.0]},
        index=pd.MultiIndex.from_arrays(
            [["FR"], ["A"], ["FR"], ["FR"]],
            names=["r_p", "s_p", "r_c", "r_f"],
        ),
    )
    aggregated, contribution = batch_weight_preweighted_ut_matrix(
        pre_weighted=pre_weighted,
        impact_names=("A",),
        weight_index=pd.Index(["FR"], name="r_f"),
        weight_values=np.array([[0.2]], dtype=np.float64),
        weight_axis="r_f",
        required_indices=("r_p", "s_p"),
        year=year,
        include_contribution=False,
    )
    assert contribution is None
    assert not aggregated.empty
