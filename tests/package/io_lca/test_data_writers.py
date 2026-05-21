from pathlib import Path

import pandas as pd
import pytest

from pyaesa.io_lca.data import writers


def test_write_and_read_table_cover_supported_formats_and_failures(tmp_path: Path) -> None:
    frame = pd.DataFrame({"year": [2019], "value": [1.0]})

    csv_path = tmp_path / "nested" / "table.csv"
    pickle_path = tmp_path / "table.pickle"
    parquet_path = tmp_path / "table.parquet"
    writers.write_table(path=csv_path, frame=frame, output_format="csv")
    writers.write_table(path=pickle_path, frame=frame, output_format="pickle")
    writers.write_table(path=parquet_path, frame=frame, output_format="parquet")

    assert writers.read_table(csv_path).equals(frame)
    assert writers.read_table(pickle_path).equals(frame)
    assert writers.read_table(parquet_path).equals(frame)


def test_write_table_uses_atomic_replace_without_temp_leaks(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "table.csv"

    writers.write_table(
        path=path,
        frame=pd.DataFrame({"year": [2019], "value": [1.0]}),
        output_format="csv",
    )
    writers.write_table(
        path=path,
        frame=pd.DataFrame({"year": [2020], "value": [2.0]}),
        output_format="csv",
    )

    assert writers.read_table(path).equals(pd.DataFrame({"year": [2020], "value": [2.0]}))
    assert sorted(path.parent.iterdir()) == [path]


def test_merge_with_existing_covers_keyed_and_keyless_paths(
    tmp_path: Path,
) -> None:
    fresh = pd.DataFrame({"id": ["a"], "value": [2.0]})
    path = tmp_path / "merge.csv"

    assert writers.merge_with_existing(path=path, fresh=fresh, key_columns=["id"]).equals(fresh)

    pd.DataFrame({"id": ["a", "b"], "value": [1.0, 3.0]}).to_csv(path, index=False)
    merged = writers.merge_with_existing(path=path, fresh=fresh, key_columns=["id"])
    assert merged.sort_values("id")["value"].tolist() == [2.0, 3.0]

    merged_no_keys = writers.merge_with_existing(path=path, fresh=fresh, key_columns=[])
    assert merged_no_keys["value"].tolist() == [1.0, 3.0, 2.0]


def test_long_to_year_wide_contracts_cover_empty_missing_and_grouped_paths() -> None:
    assert writers.long_to_year_wide(
        frame=pd.DataFrame(),
        id_columns=["id"],
        value_column="value",
    ).columns.tolist() == ["id"]

    wide = writers.long_to_year_wide(
        frame=pd.DataFrame(
            {
                "id": ["b", "a", "a"],
                "year": [2020, 2019, 2019],
                "value": [3.0, 1.0, 2.0],
            }
        ),
        id_columns=["id"],
        value_column="value",
    )
    assert wide.columns.tolist() == ["id", "2019", "2020"]
    assert wide.loc[0, "id"] == "a"
    assert wide.loc[0, "2019"] == pytest.approx(3.0)

    no_id_wide = writers.long_to_year_wide(
        frame=pd.DataFrame({"year": [2020, 2019], "value": [4.0, 1.0]}),
        id_columns=[],
        value_column="value",
    )
    assert no_id_wide.columns.tolist() == ["2019", "2020"]
    assert no_id_wide.loc[0, "2020"] == pytest.approx(4.0)


def test_clear_scope_outputs_handles_existing_and_missing_roots(tmp_path: Path) -> None:
    scope_root = tmp_path / "scope"
    (scope_root / "nested").mkdir(parents=True)
    (scope_root / "nested" / "file.txt").write_text("x", encoding="utf-8")

    writers.clear_scope_outputs(scope_root=scope_root)
    assert not scope_root.exists()

    writers.clear_scope_outputs(scope_root=scope_root)
    assert not scope_root.exists()
