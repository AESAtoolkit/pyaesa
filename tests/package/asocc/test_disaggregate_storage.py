from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from pyaesa.asocc.disaggregation.published_storage import (
    PartitionSchema,
    _merge_reference,
    _require_unique_variants,
    _write_table as _write_published_table,
    disaggregate_rows,
    load_partitioned_rows,
    write_partitioned_rows,
)
from pyaesa.asocc.disaggregation.pipeline import _time_route_warning_lines
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)


def _write_table(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    if ASOCC_SSP_SCENARIO_COLUMN not in output.columns:
        scenario = _fixture_ssp_from_stem(path.stem)
        if scenario is not None:
            year_position = next(
                (index for index, column in enumerate(output.columns) if str(column).isdigit()),
                len(output.columns),
            )
            output.insert(year_position, ASOCC_SSP_SCENARIO_COLUMN, scenario)
    output.to_csv(path, index=False)
    return path


def _fixture_ssp_from_stem(stem: str) -> str | None:
    for token in str(stem).split("__"):
        lowered = token.lower()
        if lowered.startswith("ssp") and lowered[3:].isdigit():
            return f"SSP{int(lowered[3:])}"
    return None


def _with_fixture_scenario(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    output = frame.copy()
    year_position = next(
        (index for index, column in enumerate(output.columns) if str(column).isdigit()),
        len(output.columns),
    )
    output.insert(year_position, ASOCC_SSP_SCENARIO_COLUMN, scenario)
    return output


def test_disaggregate_time_route_warning_lines_cover_bridge_summary() -> None:
    assert _time_route_warning_lines(audit_frame=pd.DataFrame()) == []
    assert (
        _time_route_warning_lines(
            audit_frame=pd.DataFrame(
                {
                    "year": [2023],
                    "ref_grouped_time_route_bridge": [False],
                }
            )
        )
        == []
    )
    warnings = _time_route_warning_lines(
        audit_frame=pd.DataFrame(
            {
                "year": [2023, 2024],
                "ref_grouped_time_route_bridge": [True, False],
                "ref_split_time_route_bridge": [False, True],
            }
        )
    )
    warning_text = " ".join(warnings)
    assert all(len(line) <= 100 for line in warnings)
    assert "years 2023-2024" in warning_text
    assert "same studied year" in warning_text
    assert "same l2_reuse_year" in warning_text
    assert "uncertainty_asocc, uncertainty_acc, and uncertainty_asr" in warning_text


def test_published_storage_bridges_mrio_time_route_mismatches() -> None:
    regression_target = pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)"],
            "l2_method": ["UT(TD)"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["regression_proj"],
            "r_c": ["FR"],
            "s_p": ["D"],
            "year": [2023],
            "value": [20.0],
            "l2_reuse_year": [None],
        }
    )
    regression_grouped = pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)"],
            "l2_method": ["UT(TD)"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical"],
            "r_c": ["FR"],
            "s_p": ["D"],
            "year": [2023],
            "value": [10.0],
            "l2_reuse_year": [None],
        }
    )
    regression_split = pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)", "UT(TD)"],
            "l2_method": ["UT(TD)", "UT(TD)"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical", "historical"],
            "r_c": ["FR", "FR"],
            "s_p": ["Electricity_coal", "Electricity_gas"],
            "year": [2023, 2023],
            "value": [4.0, 6.0],
            "l2_reuse_year": [None, None],
        }
    )
    output_rows, audit = disaggregate_rows(
        target_rows=regression_target,
        ref_grouped_rows=regression_grouped,
        ref_split_rows=regression_split,
        grouped_sector_by_split={
            "Electricity_coal": "D",
            "Electricity_gas": "D",
        },
    )
    assert output_rows.sort_values("s_p")["value"].tolist() == [8.0, 12.0]
    assert bool(
        audit[["ref_grouped_time_route_bridge", "ref_split_time_route_bridge"]]
        .to_numpy(dtype=bool)
        .all()
    )

    reuse_target = pd.DataFrame(
        {
            "l1_l2_method": ["EG(Pop)_UT(FDa)"],
            "l1_method": ["EG(Pop)"],
            "l2_method": ["UT(FDa)"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical_reuse"],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"],
            "r_c": ["FR"],
            "s_p": ["D"],
            "l2_reuse_year": [1995],
            "year": [2023],
            "value": [20.0],
        }
    )
    reuse_grouped = pd.DataFrame(
        {
            "l1_l2_method": ["EG(Pop)_UT(FDa)", "EG(Pop)_UT(FDa)"],
            "l1_method": ["EG(Pop)", "EG(Pop)"],
            "l2_method": ["UT(FDa)", "UT(FDa)"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical", "historical_reuse"],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", "SSP1"],
            "r_c": ["FR", "FR"],
            "s_p": ["D", "D"],
            "l2_reuse_year": [None, 1995],
            "year": [2023, 2025],
            "value": [100.0, 10.0],
        }
    )
    reuse_split = pd.DataFrame(
        {
            "l1_l2_method": ["EG(Pop)_UT(FDa)"] * 4,
            "l1_method": ["EG(Pop)"] * 4,
            "l2_method": ["UT(FDa)"] * 4,
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [
                "historical",
                "historical",
                "historical_reuse",
                "historical_reuse",
            ],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"] * 4,
            "r_c": ["FR"] * 4,
            "s_p": [
                "Electricity_coal",
                "Electricity_gas",
                "Electricity_coal",
                "Electricity_gas",
            ],
            "l2_reuse_year": [None, None, 1995, 1995],
            "year": [2023, 2023, 2025, 2025],
            "value": [50.0, 50.0, 4.0, 6.0],
        }
    )
    output_rows, audit = disaggregate_rows(
        target_rows=reuse_target,
        ref_grouped_rows=reuse_grouped,
        ref_split_rows=reuse_split,
        grouped_sector_by_split={
            "Electricity_coal": "D",
            "Electricity_gas": "D",
        },
    )
    assert output_rows.sort_values("s_p")["value"].tolist() == [8.0, 12.0]
    assert bool(
        audit[["ref_grouped_time_route_bridge", "ref_split_time_route_bridge"]]
        .to_numpy(dtype=bool)
        .all()
    )

    output_rows, _audit = disaggregate_rows(
        target_rows=reuse_target.drop(columns=ASOCC_SSP_SCENARIO_COLUMN),
        ref_grouped_rows=reuse_grouped.drop(columns=ASOCC_SSP_SCENARIO_COLUMN),
        ref_split_rows=reuse_split.drop(columns=ASOCC_SSP_SCENARIO_COLUMN),
        grouped_sector_by_split={
            "Electricity_coal": "D",
            "Electricity_gas": "D",
        },
    )
    assert output_rows.sort_values("s_p")["value"].tolist() == [8.0, 12.0]

    same_year_target = reuse_target.assign(l2_reuse_year=1996)
    same_year_grouped = reuse_grouped.iloc[[1]].assign(
        l2_reuse_year=1996,
        year=2023,
        value=10.0,
    )
    same_year_split = reuse_split.iloc[[2, 3]].assign(
        l2_reuse_year=1996,
        year=2023,
        value=[4.0, 6.0],
    )
    output_rows, audit = disaggregate_rows(
        target_rows=same_year_target,
        ref_grouped_rows=same_year_grouped,
        ref_split_rows=same_year_split,
        grouped_sector_by_split={
            "Electricity_coal": "D",
            "Electricity_gas": "D",
        },
    )
    assert output_rows.sort_values("s_p")["value"].tolist() == [8.0, 12.0]
    assert not bool(
        audit[["ref_grouped_time_route_bridge", "ref_split_time_route_bridge"]]
        .to_numpy(dtype=bool)
        .any()
    )


def test_published_storage_disaggregates_transition_rows_and_keeps_target_regression_partition(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "target"
    ref_grouped_root = tmp_path / "ref_grouped"
    ref_split_root = tmp_path / "ref_split"
    output_root = tmp_path / "output"
    target_path = target_root / "regression_proj" / "UT(FD)__ssp2.csv"
    ref_grouped_path = ref_grouped_root / "UT(FD).csv"
    ref_split_path = ref_split_root / "UT(FD).csv"

    _write_table(
        target_path,
        pd.DataFrame(
            {
                "l1_l2_method": ["UT(FD)", "UT(FD)"],
                "l2_method": ["UT(FD)", "UT(FD)"],
                "r_p": ["FR", "US"],
                "s_p": ["D", "D"],
                "2021": [11.0, 13.0],
            }
        ),
    )
    _write_table(
        ref_grouped_path,
        pd.DataFrame(
            {
                "l1_l2_method": ["UT(FD)", "UT(FD)"],
                "l2_method": ["UT(FD)", "UT(FD)"],
                "r_p": ["FR", "US"],
                "s_p": ["D", "D"],
                "2021": [10.0, 20.0],
            }
        ),
    )
    _write_table(
        ref_split_path,
        pd.DataFrame(
            {
                "l1_l2_method": ["UT(FD)"] * 4,
                "l2_method": ["UT(FD)"] * 4,
                "r_p": ["FR", "FR", "US", "US"],
                "s_p": [
                    "Electricity_coal",
                    "Electricity_gas",
                    "Electricity_coal",
                    "Electricity_gas",
                ],
                "2021": [4.0, 6.0, 8.0, 12.0],
            }
        ),
    )

    target_rows, schemas = load_partitioned_rows(
        root=target_root,
        stem_prefix="UT(FD)",
        requested_years=[2021],
        require_requested_coverage=True,
    )
    ref_grouped_rows, _ = load_partitioned_rows(
        root=ref_grouped_root,
        stem_prefix="UT(FD)",
        requested_years=[2021],
        require_requested_coverage=True,
    )
    ref_split_rows, _ = load_partitioned_rows(
        root=ref_split_root,
        stem_prefix="UT(FD)",
        requested_years=[2021],
        require_requested_coverage=True,
    )

    output_rows, audit = disaggregate_rows(
        target_rows=target_rows,
        ref_grouped_rows=ref_grouped_rows,
        ref_split_rows=ref_split_rows,
        grouped_sector_by_split={
            "Electricity_coal": "D",
            "Electricity_gas": "D",
        },
    )
    written = write_partitioned_rows(
        rows=output_rows,
        schemas=schemas,
        output_root=output_root,
        output_format="csv",
    )

    assert audit["year"].tolist() == [2021, 2021, 2021, 2021]
    assert audit[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2"] * 4
    assert bool(audit["l2_reuse_year"].isna().all())
    assert written == [output_root / "regression_proj" / "UT(FD)__ssp2.csv"]
    expected = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FD)"] * 4,
            "l2_method": ["UT(FD)"] * 4,
            "r_p": ["FR", "FR", "US", "US"],
            "s_p": [
                "Electricity_coal",
                "Electricity_gas",
                "Electricity_coal",
                "Electricity_gas",
            ],
            "2021": [4.4, 6.6, 5.2, 7.8],
        }
    )
    actual = (
        pd.read_csv(written[0])
        .sort_values(by=["r_p", "s_p"], kind="mergesort")
        .reset_index(drop=True)
    )
    pdt.assert_frame_equal(actual, _with_fixture_scenario(expected, "SSP2"), check_dtype=False)


def test_published_storage_preserves_target_l2_reuse_year_identity_with_reference_broadcast(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "target"
    ref_grouped_root = tmp_path / "ref_grouped"
    ref_split_root = tmp_path / "ref_split"
    output_root = tmp_path / "output"
    stem = "EG(Pop)_UT(FDa)__ssp2"
    target_path = target_root / "historical_reuse" / f"{stem}.csv"
    ref_grouped_path = ref_grouped_root / "historical_reuse" / f"{stem}.csv"
    ref_split_path = ref_split_root / "historical_reuse" / f"{stem}.csv"

    _write_table(
        target_path,
        pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)", "EG(Pop)_UT(FDa)"],
                "l1_method": ["EG(Pop)", "EG(Pop)"],
                "l2_method": ["UT(FDa)", "UT(FDa)"],
                "r_c": ["FR", "FR"],
                "s_p": ["D", "D"],
                "l2_reuse_year": [2005, 2006],
                "2030": [10.0, 20.0],
            }
        ),
    )
    _write_table(
        ref_grouped_path,
        pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)"],
                "l1_method": ["EG(Pop)"],
                "l2_method": ["UT(FDa)"],
                "r_c": ["FR"],
                "s_p": ["D"],
                "2030": [10.0],
            }
        ),
    )
    _write_table(
        ref_split_path,
        pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)", "EG(Pop)_UT(FDa)"],
                "l1_method": ["EG(Pop)", "EG(Pop)"],
                "l2_method": ["UT(FDa)", "UT(FDa)"],
                "r_c": ["FR", "FR"],
                "s_p": ["Electricity_coal", "Electricity_gas"],
                "2030": [3.0, 7.0],
            }
        ),
    )

    target_rows, schemas = load_partitioned_rows(
        root=target_root,
        stem_prefix="EG(Pop)_UT(FDa)",
        requested_years=[2030],
        require_requested_coverage=True,
    )
    ref_grouped_rows, _ = load_partitioned_rows(
        root=ref_grouped_root,
        stem_prefix="EG(Pop)_UT(FDa)",
        requested_years=[2030],
        require_requested_coverage=True,
    )
    ref_split_rows, _ = load_partitioned_rows(
        root=ref_split_root,
        stem_prefix="EG(Pop)_UT(FDa)",
        requested_years=[2030],
        require_requested_coverage=True,
    )

    output_rows, _audit = disaggregate_rows(
        target_rows=target_rows,
        ref_grouped_rows=ref_grouped_rows,
        ref_split_rows=ref_split_rows,
        grouped_sector_by_split={
            "Electricity_coal": "D",
            "Electricity_gas": "D",
        },
    )
    written = write_partitioned_rows(
        rows=output_rows,
        schemas=schemas,
        output_root=output_root,
        output_format="csv",
    )

    actual = (
        pd.read_csv(written[0])
        .sort_values(
            by=["l2_reuse_year", "s_p"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    expected = pd.DataFrame(
        {
            "l1_l2_method": ["EG(Pop)_UT(FDa)"] * 4,
            "l1_method": ["EG(Pop)"] * 4,
            "l2_method": ["UT(FDa)"] * 4,
            "r_c": ["FR"] * 4,
            "s_p": [
                "Electricity_coal",
                "Electricity_gas",
                "Electricity_coal",
                "Electricity_gas",
            ],
            "l2_reuse_year": [2005, 2005, 2006, 2006],
            "2030": [3.0, 7.0, 6.0, 14.0],
        }
    )
    pdt.assert_frame_equal(actual, _with_fixture_scenario(expected, "SSP2"), check_dtype=False)
    assert written == [output_root / "historical_reuse" / f"{stem}.csv"]


def test_published_storage_matches_future_rows_per_ssp_scenario(tmp_path: Path) -> None:
    target_root = tmp_path / "target"
    ref_grouped_root = tmp_path / "ref_grouped"
    ref_split_root = tmp_path / "ref_split"
    output_root = tmp_path / "output"
    target_paths = [
        target_root / "regression_proj" / "UT(TD)__ssp1.csv",
        target_root / "regression_proj" / "UT(TD)__ssp2.csv",
    ]
    ref_grouped_paths = [
        ref_grouped_root / "regression_proj" / "UT(TD)__ssp1.csv",
        ref_grouped_root / "regression_proj" / "UT(TD)__ssp2.csv",
    ]
    ref_split_paths = [
        ref_split_root / "regression_proj" / "UT(TD)__ssp1.csv",
        ref_split_root / "regression_proj" / "UT(TD)__ssp2.csv",
    ]
    for path, target_value in [
        (target_paths[0], 12.0),
        (target_paths[1], 18.0),
    ]:
        _write_table(
            path,
            pd.DataFrame(
                {
                    "l1_l2_method": ["UT(TD)"],
                    "l2_method": ["UT(TD)"],
                    "r_p": ["FR"],
                    "s_p": ["D"],
                    "2026": [target_value],
                }
            ),
        )
    for path, grouped_value in [
        (ref_grouped_paths[0], 10.0),
        (ref_grouped_paths[1], 12.0),
    ]:
        _write_table(
            path,
            pd.DataFrame(
                {
                    "l1_l2_method": ["UT(TD)"],
                    "l2_method": ["UT(TD)"],
                    "r_p": ["FR"],
                    "s_p": ["D"],
                    "2026": [grouped_value],
                }
            ),
        )
    for path, coal_value, gas_value in [
        (ref_split_paths[0], 4.0, 6.0),
        (ref_split_paths[1], 3.0, 9.0),
    ]:
        _write_table(
            path,
            pd.DataFrame(
                {
                    "l1_l2_method": ["UT(TD)", "UT(TD)"],
                    "l2_method": ["UT(TD)", "UT(TD)"],
                    "r_p": ["FR", "FR"],
                    "s_p": ["Electricity_coal", "Electricity_gas"],
                    "2026": [coal_value, gas_value],
                }
            ),
        )

    target_rows, schemas = load_partitioned_rows(
        root=target_root,
        stem_prefix="UT(TD)",
        requested_years=[2026],
        require_requested_coverage=True,
    )
    ref_grouped_rows, _ = load_partitioned_rows(
        root=ref_grouped_root,
        stem_prefix="UT(TD)",
        requested_years=[2026],
        require_requested_coverage=True,
    )
    ref_split_rows, _ = load_partitioned_rows(
        root=ref_split_root,
        stem_prefix="UT(TD)",
        requested_years=[2026],
        require_requested_coverage=True,
    )

    output_rows, _audit = disaggregate_rows(
        target_rows=target_rows,
        ref_grouped_rows=ref_grouped_rows,
        ref_split_rows=ref_split_rows,
        grouped_sector_by_split={
            "Electricity_coal": "D",
            "Electricity_gas": "D",
        },
    )
    written = write_partitioned_rows(
        rows=output_rows,
        schemas=schemas,
        output_root=output_root,
        output_format="csv",
    )

    assert written == sorted(
        [
            output_root / "regression_proj" / "UT(TD)__ssp1.csv",
            output_root / "regression_proj" / "UT(TD)__ssp2.csv",
        ]
    )
    expected_frames = {
        "UT(TD)__ssp1.csv": pd.DataFrame(
            {
                "l1_l2_method": ["UT(TD)", "UT(TD)"],
                "l2_method": ["UT(TD)", "UT(TD)"],
                "r_p": ["FR", "FR"],
                "s_p": ["Electricity_coal", "Electricity_gas"],
                "2026": [4.8, 7.2],
            }
        ),
        "UT(TD)__ssp2.csv": pd.DataFrame(
            {
                "l1_l2_method": ["UT(TD)", "UT(TD)"],
                "l2_method": ["UT(TD)", "UT(TD)"],
                "r_p": ["FR", "FR"],
                "s_p": ["Electricity_coal", "Electricity_gas"],
                "2026": [4.5, 13.5],
            }
        ),
    }
    for path in written:
        actual = (
            pd.read_csv(path)
            .sort_values(by=["r_p", "s_p"], kind="mergesort")
            .reset_index(drop=True)
        )
        expected = _with_fixture_scenario(
            expected_frames[path.name],
            str(_fixture_ssp_from_stem(path.stem)),
        )
        pdt.assert_frame_equal(actual, expected, check_dtype=False)


def test_published_storage_matches_future_rows_per_ssp_scenario_and_l2_reuse_year(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "target"
    ref_grouped_root = tmp_path / "ref_grouped"
    ref_split_root = tmp_path / "ref_split"
    output_root = tmp_path / "output"
    for scenario, target_values, grouped_values, split_values in [
        ("ssp1", [10.0, 20.0], [10.0, 20.0], [(4.0, 6.0), (5.0, 15.0)]),
        ("ssp2", [30.0, 40.0], [15.0, 20.0], [(3.0, 12.0), (4.0, 16.0)]),
    ]:
        _write_table(
            target_root / "historical_reuse" / f"EG(Pop)_UT(FDa)__{scenario}.csv",
            pd.DataFrame(
                {
                    "l1_l2_method": ["EG(Pop)_UT(FDa)", "EG(Pop)_UT(FDa)"],
                    "l1_method": ["EG(Pop)", "EG(Pop)"],
                    "l2_method": ["UT(FDa)", "UT(FDa)"],
                    "r_c": ["FR", "FR"],
                    "s_p": ["D", "D"],
                    "l2_reuse_year": [2005, 2006],
                    "2030": target_values,
                }
            ),
        )
        _write_table(
            ref_grouped_root / "historical_reuse" / f"EG(Pop)_UT(FDa)__{scenario}.csv",
            pd.DataFrame(
                {
                    "l1_l2_method": ["EG(Pop)_UT(FDa)", "EG(Pop)_UT(FDa)"],
                    "l1_method": ["EG(Pop)", "EG(Pop)"],
                    "l2_method": ["UT(FDa)", "UT(FDa)"],
                    "r_c": ["FR", "FR"],
                    "s_p": ["D", "D"],
                    "l2_reuse_year": [2005, 2006],
                    "2030": grouped_values,
                }
            ),
        )
        _write_table(
            ref_split_root / "historical_reuse" / f"EG(Pop)_UT(FDa)__{scenario}.csv",
            pd.DataFrame(
                {
                    "l1_l2_method": ["EG(Pop)_UT(FDa)"] * 4,
                    "l1_method": ["EG(Pop)"] * 4,
                    "l2_method": ["UT(FDa)"] * 4,
                    "r_c": ["FR"] * 4,
                    "s_p": [
                        "Electricity_coal",
                        "Electricity_gas",
                        "Electricity_coal",
                        "Electricity_gas",
                    ],
                    "l2_reuse_year": [2005, 2005, 2006, 2006],
                    "2030": [
                        split_values[0][0],
                        split_values[0][1],
                        split_values[1][0],
                        split_values[1][1],
                    ],
                }
            ),
        )

    target_rows, schemas = load_partitioned_rows(
        root=target_root,
        stem_prefix="EG(Pop)_UT(FDa)",
        requested_years=[2030],
        require_requested_coverage=True,
    )
    ref_grouped_rows, _ = load_partitioned_rows(
        root=ref_grouped_root,
        stem_prefix="EG(Pop)_UT(FDa)",
        requested_years=[2030],
        require_requested_coverage=True,
    )
    ref_split_rows, _ = load_partitioned_rows(
        root=ref_split_root,
        stem_prefix="EG(Pop)_UT(FDa)",
        requested_years=[2030],
        require_requested_coverage=True,
    )

    output_rows, _audit = disaggregate_rows(
        target_rows=target_rows,
        ref_grouped_rows=ref_grouped_rows,
        ref_split_rows=ref_split_rows,
        grouped_sector_by_split={
            "Electricity_coal": "D",
            "Electricity_gas": "D",
        },
    )
    written = write_partitioned_rows(
        rows=output_rows,
        schemas=schemas,
        output_root=output_root,
        output_format="csv",
    )

    assert written == sorted(
        [
            output_root / "historical_reuse" / "EG(Pop)_UT(FDa)__ssp1.csv",
            output_root / "historical_reuse" / "EG(Pop)_UT(FDa)__ssp2.csv",
        ]
    )
    expected_frames = {
        "EG(Pop)_UT(FDa)__ssp1.csv": pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)"] * 4,
                "l1_method": ["EG(Pop)"] * 4,
                "l2_method": ["UT(FDa)"] * 4,
                "r_c": ["FR"] * 4,
                "s_p": [
                    "Electricity_coal",
                    "Electricity_gas",
                    "Electricity_coal",
                    "Electricity_gas",
                ],
                "l2_reuse_year": [2005, 2005, 2006, 2006],
                "2030": [4.0, 6.0, 5.0, 15.0],
            }
        ),
        "EG(Pop)_UT(FDa)__ssp2.csv": pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)"] * 4,
                "l1_method": ["EG(Pop)"] * 4,
                "l2_method": ["UT(FDa)"] * 4,
                "r_c": ["FR"] * 4,
                "s_p": [
                    "Electricity_coal",
                    "Electricity_gas",
                    "Electricity_coal",
                    "Electricity_gas",
                ],
                "l2_reuse_year": [2005, 2005, 2006, 2006],
                "2030": [6.0, 24.0, 8.0, 32.0],
            }
        ),
    }
    for path in written:
        actual = (
            pd.read_csv(path)
            .sort_values(by=["l2_reuse_year", "s_p"], kind="mergesort")
            .reset_index(drop=True)
        )
        expected = _with_fixture_scenario(
            expected_frames[path.name],
            str(_fixture_ssp_from_stem(path.stem)),
        )
        pdt.assert_frame_equal(actual, expected, check_dtype=False)


def test_published_storage_broadcasts_historical_reference_rows_during_reuse_transition(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "target"
    ref_grouped_root = tmp_path / "ref_grouped"
    ref_split_root = tmp_path / "ref_split"
    output_root = tmp_path / "output"
    for scenario, values_2023, values_2030 in [
        ("ssp1", [10.0, 20.0], [10.0, 20.0]),
        ("ssp2", [30.0, 40.0], [30.0, 40.0]),
    ]:
        _write_table(
            target_root / "historical_reuse" / f"EG(Pop)_UT(FDa)__{scenario}.csv",
            pd.DataFrame(
                {
                    "l1_l2_method": ["EG(Pop)_UT(FDa)", "EG(Pop)_UT(FDa)"],
                    "l1_method": ["EG(Pop)", "EG(Pop)"],
                    "l2_method": ["UT(FDa)", "UT(FDa)"],
                    "r_c": ["FR", "FR"],
                    "s_p": ["D", "D"],
                    "l2_reuse_year": [2005, 2006],
                    "2023": values_2023,
                    "2030": values_2030,
                }
            ),
        )
    _write_table(
        ref_grouped_root / "EG(Pop)_UT(FDa).csv",
        pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)"],
                "l1_method": ["EG(Pop)"],
                "l2_method": ["UT(FDa)"],
                "r_c": ["FR"],
                "s_p": ["D"],
                "2023": [10.0],
            }
        ),
    )
    _write_table(
        ref_split_root / "EG(Pop)_UT(FDa).csv",
        pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)", "EG(Pop)_UT(FDa)"],
                "l1_method": ["EG(Pop)", "EG(Pop)"],
                "l2_method": ["UT(FDa)", "UT(FDa)"],
                "r_c": ["FR", "FR"],
                "s_p": ["Electricity_coal", "Electricity_gas"],
                "2023": [4.0, 6.0],
            }
        ),
    )
    for scenario, grouped_2030, split_2030 in [
        ("ssp1", [10.0, 20.0], [(4.0, 6.0), (5.0, 15.0)]),
        ("ssp2", [15.0, 20.0], [(3.0, 12.0), (4.0, 16.0)]),
    ]:
        _write_table(
            ref_grouped_root / "historical_reuse" / f"EG(Pop)_UT(FDa)__{scenario}.csv",
            pd.DataFrame(
                {
                    "l1_l2_method": ["EG(Pop)_UT(FDa)", "EG(Pop)_UT(FDa)"],
                    "l1_method": ["EG(Pop)", "EG(Pop)"],
                    "l2_method": ["UT(FDa)", "UT(FDa)"],
                    "r_c": ["FR", "FR"],
                    "s_p": ["D", "D"],
                    "l2_reuse_year": [2005, 2006],
                    "2030": grouped_2030,
                }
            ),
        )
        _write_table(
            ref_split_root / "historical_reuse" / f"EG(Pop)_UT(FDa)__{scenario}.csv",
            pd.DataFrame(
                {
                    "l1_l2_method": ["EG(Pop)_UT(FDa)"] * 4,
                    "l1_method": ["EG(Pop)"] * 4,
                    "l2_method": ["UT(FDa)"] * 4,
                    "r_c": ["FR"] * 4,
                    "s_p": [
                        "Electricity_coal",
                        "Electricity_gas",
                        "Electricity_coal",
                        "Electricity_gas",
                    ],
                    "l2_reuse_year": [2005, 2005, 2006, 2006],
                    "2030": [
                        split_2030[0][0],
                        split_2030[0][1],
                        split_2030[1][0],
                        split_2030[1][1],
                    ],
                }
            ),
        )

    target_rows, schemas = load_partitioned_rows(
        root=target_root,
        stem_prefix="EG(Pop)_UT(FDa)",
        requested_years=[2023, 2030],
        require_requested_coverage=True,
    )
    ref_grouped_rows, _ = load_partitioned_rows(
        root=ref_grouped_root,
        stem_prefix="EG(Pop)_UT(FDa)",
        requested_years=[2023, 2030],
        require_requested_coverage=True,
    )
    ref_split_rows, _ = load_partitioned_rows(
        root=ref_split_root,
        stem_prefix="EG(Pop)_UT(FDa)",
        requested_years=[2023, 2030],
        require_requested_coverage=True,
    )

    output_rows, _audit = disaggregate_rows(
        target_rows=target_rows,
        ref_grouped_rows=ref_grouped_rows,
        ref_split_rows=ref_split_rows,
        grouped_sector_by_split={
            "Electricity_coal": "D",
            "Electricity_gas": "D",
        },
    )
    written = write_partitioned_rows(
        rows=output_rows,
        schemas=schemas,
        output_root=output_root,
        output_format="csv",
    )

    assert written == sorted(
        [
            output_root / "historical_reuse" / "EG(Pop)_UT(FDa)__ssp1.csv",
            output_root / "historical_reuse" / "EG(Pop)_UT(FDa)__ssp2.csv",
        ]
    )
    expected_frames = {
        "EG(Pop)_UT(FDa)__ssp1.csv": pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)"] * 4,
                "l1_method": ["EG(Pop)"] * 4,
                "l2_method": ["UT(FDa)"] * 4,
                "r_c": ["FR"] * 4,
                "s_p": [
                    "Electricity_coal",
                    "Electricity_gas",
                    "Electricity_coal",
                    "Electricity_gas",
                ],
                "l2_reuse_year": [2005, 2005, 2006, 2006],
                "2023": [4.0, 6.0, 8.0, 12.0],
                "2030": [4.0, 6.0, 5.0, 15.0],
            }
        ),
        "EG(Pop)_UT(FDa)__ssp2.csv": pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)"] * 4,
                "l1_method": ["EG(Pop)"] * 4,
                "l2_method": ["UT(FDa)"] * 4,
                "r_c": ["FR"] * 4,
                "s_p": [
                    "Electricity_coal",
                    "Electricity_gas",
                    "Electricity_coal",
                    "Electricity_gas",
                ],
                "l2_reuse_year": [2005, 2005, 2006, 2006],
                "2023": [12.0, 18.0, 16.0, 24.0],
                "2030": [6.0, 24.0, 8.0, 32.0],
            }
        ),
    }
    for path in written:
        actual = (
            pd.read_csv(path)
            .sort_values(by=["l2_reuse_year", "s_p"], kind="mergesort")
            .reset_index(drop=True)
        )
        expected = _with_fixture_scenario(
            expected_frames[path.name],
            str(_fixture_ssp_from_stem(path.stem)),
        )
        pdt.assert_frame_equal(actual, expected, check_dtype=False)


def test_published_storage_rejects_incompatible_multi_variant_reference_rows(
    tmp_path: Path,
) -> None:
    target_root = tmp_path / "target"
    ref_grouped_root = tmp_path / "ref_grouped"
    ref_split_root = tmp_path / "ref_split"
    _write_table(
        target_root / "regression_proj" / "UT(FD)__ssp2.csv",
        pd.DataFrame(
            {
                "l1_l2_method": ["UT(FD)"],
                "l2_method": ["UT(FD)"],
                "r_p": ["FR"],
                "s_p": ["D"],
                "2021": [10.0],
            }
        ),
    )
    _write_table(
        ref_grouped_root / "UT(FD).csv",
        pd.DataFrame(
            {
                "l1_l2_method": ["UT(FD)"],
                "l2_method": ["UT(FD)"],
                "r_p": ["FR"],
                "s_p": ["D"],
                "2021": [10.0],
            }
        ),
    )
    _write_table(
        ref_split_root / "historical_reuse" / "UT(FD)__ssp2.csv",
        pd.DataFrame(
            {
                "l1_l2_method": ["UT(FD)", "UT(FD)"],
                "l2_method": ["UT(FD)", "UT(FD)"],
                "r_p": ["FR", "FR"],
                "s_p": ["Electricity_coal", "Electricity_coal"],
                "l2_reuse_year": [2005, 2006],
                "2021": [4.0, 6.0],
            }
        ),
    )

    target_rows, _schemas = load_partitioned_rows(
        root=target_root,
        stem_prefix="UT(FD)",
        requested_years=[2021],
        require_requested_coverage=True,
    )
    ref_grouped_rows, _ = load_partitioned_rows(
        root=ref_grouped_root,
        stem_prefix="UT(FD)",
        requested_years=[2021],
        require_requested_coverage=True,
    )
    ref_split_rows, _ = load_partitioned_rows(
        root=ref_split_root,
        stem_prefix="UT(FD)",
        requested_years=[2021],
        require_requested_coverage=True,
    )

    with pytest.raises(ValueError, match="multiple incompatible published variants"):
        disaggregate_rows(
            target_rows=target_rows,
            ref_grouped_rows=ref_grouped_rows,
            ref_split_rows=ref_split_rows,
            grouped_sector_by_split={"Electricity_coal": "D"},
        )


def test_published_storage_helper_paths_cover_empty_inputs_and_alt_formats(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    rows, schemas = load_partitioned_rows(
        root=missing_root,
        stem_prefix="UT(FD)",
        requested_years=[2030],
        require_requested_coverage=False,
    )
    assert rows.empty
    assert schemas == {}

    with pytest.raises(ValueError, match="Missing required deterministic output files"):
        load_partitioned_rows(
            root=missing_root,
            stem_prefix="UT(FD)",
            requested_years=[2030],
            require_requested_coverage=True,
        )

    year_miss_root = tmp_path / "year_miss"
    _write_table(
        year_miss_root / "UT(FD).csv",
        pd.DataFrame({"r_p": ["FR"], "s_p": ["D"], "2021": [1.0]}),
    )
    rows, schemas = load_partitioned_rows(
        root=year_miss_root,
        stem_prefix="UT(FD)",
        requested_years=[2030],
        require_requested_coverage=False,
    )
    assert rows.empty
    assert schemas == {}

    nan_root = tmp_path / "nan_rows"
    _write_table(
        nan_root / "UT(FD).csv",
        pd.DataFrame({"r_p": ["FR"], "s_p": ["D"], "2030": [float("nan")]}),
    )
    rows, schemas = load_partitioned_rows(
        root=nan_root,
        stem_prefix="UT(FD)",
        requested_years=[2030],
        require_requested_coverage=False,
    )
    assert rows.empty
    assert schemas == {}

    frame = pd.DataFrame({"r_p": ["FR"], "s_p": ["D"], "2030": [1.0]})
    pickle_path = tmp_path / "alt_formats" / "UT(FD).pickle"
    parquet_path = tmp_path / "alt_formats" / "UT(FD)__parquet.parquet"
    _write_published_table(path=pickle_path, frame=frame, output_format="pickle")
    _write_published_table(path=parquet_path, frame=frame, output_format="parquet")
    assert pickle_path.exists()
    assert parquet_path.exists()
    rows, schemas = load_partitioned_rows(
        root=tmp_path / "alt_formats",
        stem_prefix="UT(FD)",
        requested_years=[2030],
        require_requested_coverage=True,
    )
    assert sorted(rows["file_stem"].unique().tolist()) == ["UT(FD)", "UT(FD)__parquet"]
    assert len(schemas) == 2


def test_published_storage_covers_direct_error_and_skip_paths(tmp_path: Path) -> None:
    empty_output, empty_audit = disaggregate_rows(
        target_rows=pd.DataFrame(),
        ref_grouped_rows=pd.DataFrame(),
        ref_split_rows=pd.DataFrame(),
        grouped_sector_by_split={},
    )
    assert empty_output.empty
    assert empty_audit.empty

    target_rows = pd.DataFrame(
        {
            "year": [2030],
            "r_p": ["FR"],
            "s_p": ["D"],
            "value": [10.0],
            "relative_parent": [""],
            "file_stem": ["UT(FD)"],
            ASOCC_SSP_SCENARIO_COLUMN: [None],
            "l2_reuse_year": [None],
        }
    )
    ref_grouped_rows = pd.DataFrame(
        {
            "year": [2030],
            "r_p": ["FR"],
            "s_p": ["D"],
            "value": [10.0],
            "relative_parent": [""],
            "file_stem": ["UT(FD)"],
            ASOCC_SSP_SCENARIO_COLUMN: [None],
            "l2_reuse_year": [None],
        }
    )
    ref_split_missing_mapping = pd.DataFrame(
        {
            "year": [2030],
            "r_p": ["FR"],
            "s_p": ["Electricity_other"],
            "value": [10.0],
            "relative_parent": [""],
            "file_stem": ["UT(FD)"],
            ASOCC_SSP_SCENARIO_COLUMN: [None],
            "l2_reuse_year": [None],
        }
    )
    with pytest.raises(ValueError, match="not declared in disaggregation_specs"):
        disaggregate_rows(
            target_rows=target_rows,
            ref_grouped_rows=ref_grouped_rows,
            ref_split_rows=ref_split_missing_mapping,
            grouped_sector_by_split={"Electricity_coal": "D"},
        )

    ref_grouped_missing = ref_grouped_rows.assign(s_p="X")
    ref_split_rows = ref_split_missing_mapping.assign(
        s_p="Electricity_coal",
        value=4.0,
    )
    with pytest.raises(ValueError, match="Missing reference values"):
        disaggregate_rows(
            target_rows=target_rows,
            ref_grouped_rows=ref_grouped_missing,
            ref_split_rows=ref_split_rows,
            grouped_sector_by_split={"Electricity_coal": "D"},
        )

    with pytest.raises(ValueError, match="grouped reference value is zero"):
        disaggregate_rows(
            target_rows=target_rows,
            ref_grouped_rows=ref_grouped_rows.assign(value=0.0),
            ref_split_rows=ref_split_rows,
            grouped_sector_by_split={"Electricity_coal": "D"},
        )

    with pytest.raises(ValueError, match="both grouped and split reference values are zero"):
        disaggregate_rows(
            target_rows=target_rows,
            ref_grouped_rows=ref_grouped_rows.assign(value=0.0),
            ref_split_rows=ref_split_rows.assign(value=0.0),
            grouped_sector_by_split={"Electricity_coal": "D"},
        )

    zero_output_rows, zero_audit = disaggregate_rows(
        target_rows=target_rows.assign(value=0.0),
        ref_grouped_rows=ref_grouped_rows.assign(value=0.0),
        ref_split_rows=ref_split_rows.assign(value=0.0, s_p="Electricity_coal"),
        grouped_sector_by_split={"Electricity_coal": "D"},
    )
    assert float(zero_output_rows.loc[0, "value"]) == 0.0
    assert float(zero_audit.loc[0, "ratio"]) == 0.0

    schema = PartitionSchema(
        relative_parent=Path("historical_reuse"),
        file_stem="UT(FD)",
        id_columns=["r_p", "s_p", "relative_parent", "file_stem"],
        year_columns=["2030"],
    )
    assert (
        write_partitioned_rows(
            rows=pd.DataFrame(),
            schemas={("historical_reuse", "UT(FD)"): schema},
            output_root=tmp_path / "empty_write",
            output_format="csv",
        )
        == []
    )
    assert (
        write_partitioned_rows(
            rows=pd.DataFrame(
                {
                    "relative_parent": ["other"],
                    "file_stem": ["other"],
                    "r_p": ["FR"],
                    "s_p": ["D"],
                    "year": [2030],
                    "value": [1.0],
                }
            ),
            schemas={("historical_reuse", "UT(FD)"): schema},
            output_root=tmp_path / "skip_write",
            output_format="csv",
        )
        == []
    )

    _require_unique_variants(
        frame=pd.DataFrame(),
        match_keys=["year"],
        label="empty",
    )
    merged = _merge_reference(
        target=pd.DataFrame({"year": [2030], "r_p": ["FR"], "value": [1.0]}),
        reference=pd.DataFrame({"year": [2030], "r_p": ["FR"], "value": [2.0], "s_p": ["D"]}),
        exact_keys=["year", "r_p"],
        label="ref_grouped",
    )
    assert merged.loc[0, "ref_grouped_value"] == 2.0

    schema_with_missing_year = PartitionSchema(
        relative_parent=Path(),
        file_stem="UT(FD)",
        id_columns=["r_p", "s_p", "relative_parent", "file_stem"],
        year_columns=["2029", "2030"],
    )
    written = write_partitioned_rows(
        rows=pd.DataFrame(
            {
                "relative_parent": [""],
                "file_stem": ["UT(FD)"],
                "r_p": ["FR"],
                "s_p": ["D"],
                "year": [2029],
                "value": [1.0],
            }
        ),
        schemas={("", "UT(FD)"): schema_with_missing_year},
        output_root=tmp_path / "missing_year_fill",
        output_format="csv",
    )
    written_frame = pd.read_csv(written[0])
    assert bool(written_frame["2030"].isna().all())


def test_published_storage_optional_filter_and_merge_cover_transition_edges() -> None:
    merged = _merge_reference(
        target=pd.DataFrame(
            {
                "year": [2030],
                "r_p": ["FR"],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP1"],
                "l2_reuse_year": [2006],
                "value": [1.0],
            }
        ),
        reference=pd.DataFrame(
            {
                "year": [2030],
                "r_p": ["FR"],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"],
                "l2_reuse_year": [2005],
                "value": [2.0],
                "s_p": ["D"],
            }
        ),
        exact_keys=["year", "r_p"],
        label="ref_grouped",
    )
    assert bool(merged["ref_grouped_value"].isna().all())
