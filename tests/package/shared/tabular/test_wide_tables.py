from pathlib import Path

import pandas as pd
import pytest

from pyaesa.shared.tabular.wide_tables import (
    detect_year_columns,
    distinct_figure_comparison_method_identity_count,
    distinct_method_identity_count,
    figure_comparison_method_identity_columns,
    first_non_null_scenario_year,
    has_multiple_figure_comparison_method_identities,
    has_multiple_method_identities,
    id_columns,
    melt_requested_year_value_rows,
    method_identity_columns,
    persisted_method_block_columns,
    planned_melt_columns,
    resolved_allocation_method_identities,
    resolve_single_allocation_method_identity,
    requested_year_columns,
    row_label,
    validate_complete_wide_year_values,
)
from pyaesa.shared.tabular.table_io import read_table


def test_read_table_csv_pickle_and_unsupported(tmp_path: Path) -> None:
    csv_path = tmp_path / "table.csv"
    csv_path.write_text("id,2020\nA,1\n", encoding="utf-8")
    csv_frame = read_table(path=csv_path)
    assert csv_frame.attrs["source_path"] == str(csv_path)
    assert list(csv_frame.columns) == ["id", "2020"]

    pickle_path = tmp_path / "table.pickle"
    pd.DataFrame({"id": ["A"], "2020": [1]}).to_pickle(pickle_path)
    pickle_frame = read_table(path=pickle_path)
    assert pickle_frame.attrs["source_path"] == str(pickle_path)
    assert list(pickle_frame.columns) == ["id", "2020"]

    with pytest.raises(ValueError):
        read_table(path=tmp_path / "table.txt")


def test_year_and_identifier_column_contracts() -> None:
    frame = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FD)"],
            "lcia_method": ["gwp100_lcia"],
            "region": ["FR"],
            2020: [1.0],
            "2021": [2.0],
            1800: [3.0],
            "note": ["x"],
        }
    )
    years = detect_year_columns(frame)
    assert years == ["2020", "2021"]
    assert method_identity_columns(frame) == ["l1_l2_method", "lcia_method"]
    assert figure_comparison_method_identity_columns(frame) == ["l1_l2_method"]
    assert distinct_figure_comparison_method_identity_count(frame) == 1
    assert has_multiple_figure_comparison_method_identities(frame) is False
    assert distinct_method_identity_count(frame) == 1
    assert has_multiple_method_identities(frame) is False
    assert persisted_method_block_columns(frame) == ["l1_l2_method", "l2_method"]
    assert resolve_single_allocation_method_identity(frame, where="demo") == "UT(FD)"
    multi_method_frame = pd.concat(
        [
            frame,
            frame.assign(l1_l2_method="AR(E^{CBA_FD})"),
        ],
        ignore_index=True,
    )
    assert distinct_method_identity_count(multi_method_frame) == 2
    assert has_multiple_method_identities(multi_method_frame) is True
    assert has_multiple_figure_comparison_method_identities(multi_method_frame) is True
    assert id_columns(frame, year_columns=years) == [
        "l1_l2_method",
        "lcia_method",
        "region",
        1800,
        "note",
    ]
    assert id_columns(frame, year_columns=years, ignored_columns={"note"}) == [
        "l1_l2_method",
        "lcia_method",
        "region",
        1800,
    ]
    assert requested_year_columns(frame, requested_years=[2021, 2030]) == ["2021"]
    assert planned_melt_columns(
        frame,
        requested_years=[2020],
        ignored_columns={"note"},
    ) == (["2020", "2021"], ["2020"], ["l1_l2_method", "lcia_method", "region", 1800])
    assert persisted_method_block_columns(pd.DataFrame({"l1_method": ["EG(Pop)"]})) == ["l1_method"]
    assert persisted_method_block_columns(
        pd.DataFrame({"l1_l2_method": ["EG(Pop)_UT(FDa)"], "l2_method": ["UT(FDa)"]})
    ) == ["l1_l2_method", "l2_method"]
    assert persisted_method_block_columns(
        pd.DataFrame(
            {
                "l1_l2_method": ["EG(Pop)_UT(FDa)"],
                "l1_method": ["EG(Pop)"],
                "l2_method": ["UT(FDa)"],
            }
        )
    ) == ["l1_l2_method", "l1_method", "l2_method"]
    assert persisted_method_block_columns(pd.DataFrame({"l1_method": [None]})) == []
    assert distinct_figure_comparison_method_identity_count(pd.DataFrame({"region": ["FR"]})) == 0


def test_melt_requested_year_value_rows_and_row_labels() -> None:
    frame = pd.DataFrame(
        {
            "region": ["FR"],
            "detail": [pd.NA],
            "2020": [1.0],
            "2021": [2.0],
        }
    )
    melted = melt_requested_year_value_rows(
        frame,
        requested_years=[2020, 2021],
        ignored_columns={"detail"},
    )
    assert list(melted.columns) == ["region", "year", "value"]
    assert melted["year"].tolist() == ["2020", "2021"]

    with pytest.raises(ValueError):
        resolve_single_allocation_method_identity(
            pd.DataFrame(
                {
                    "l1_method": ["UT"],
                    "l2_method": ["FD"],
                    "l1_l2_method": ["wrong"],
                }
            ),
            where="broken",
        )

    with pytest.raises(ValueError):
        resolve_single_allocation_method_identity(
            pd.DataFrame({"region": ["FR"]}),
            where="empty method rows",
        )

    with pytest.raises(ValueError):
        resolve_single_allocation_method_identity(
            pd.DataFrame({"l1_l2_method": ["UT(FD)", "AR(E^{CBA_TD})"]}),
            where="multi-method rows",
        )

    with pytest.raises(ValueError):
        validate_complete_wide_year_values(
            pd.DataFrame({"region": ["FR"], "2020": [1.0], "2021": [pd.NA]}),
            year_columns=["2020", "2021"],
            where="demo table",
        )
    validate_complete_wide_year_values(frame, year_columns=[], where="demo table")
    validate_complete_wide_year_values(frame, year_columns=["2020"], where="demo table")

    empty = melt_requested_year_value_rows(frame, requested_years=[2030])
    assert list(empty.columns) == ["region", "detail", "year", "value"]
    assert empty.empty

    row = frame.iloc[0]
    assert row_label(row=row, columns=["region", "detail"], default_prefix="row") == "region=FR"
    assert row_label(row=row, columns=[], default_prefix="row") == "row 1"
    assert resolved_allocation_method_identities(pd.DataFrame({"region": ["FR"]})) == []
    assert resolved_allocation_method_identities(pd.DataFrame()) == []


def test_first_non_null_scenario_year() -> None:
    frame = pd.DataFrame(
        {
            "year": [2020, 2021, 2022],
            "asocc_ssp_scenario": [None, "SSP1", "SSP1"],
        }
    )

    assert (
        first_non_null_scenario_year(
            frame,
            scenario_column="asocc_ssp_scenario",
        )
        == 2021
    )
    assert first_non_null_scenario_year(frame, scenario_column="missing") is None
    assert (
        first_non_null_scenario_year(
            frame.assign(asocc_ssp_scenario=None),
            scenario_column="asocc_ssp_scenario",
        )
        is None
    )


def test_resolved_allocation_method_identities_treat_optional_nan_method_columns_as_missing() -> (
    None
):
    frame = pd.DataFrame(
        {
            "l1_l2_method": ["EG(Pop)"],
            "l1_method": ["EG(Pop)"],
            "l2_method": [float("nan")],
        }
    )

    assert (
        resolve_single_allocation_method_identity(frame, where="single step wide rows") == "EG(Pop)"
    )
