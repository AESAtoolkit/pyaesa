from pathlib import Path

import pytest

from pyaesa.shared.runtime.io.persisted_paths import (
    normalize_persisted_paths,
    scoped_existing_table_paths,
)


def test_normalize_persisted_paths_variants() -> None:
    assert normalize_persisted_paths(raw_paths=None) == []

    normalized = normalize_persisted_paths(
        raw_paths=[
            " a.csv ",
            Path("b.parquet"),
        ],
    )
    assert normalized == [Path("a.csv"), Path("b.parquet")]


def test_scoped_existing_table_paths_success_and_validation(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    csv_path = root / "table.csv"
    csv_path.write_text("x\n1\n", encoding="utf-8")
    pickle_path = root / "table.pickle"
    pickle_path.write_bytes(b"pickle")

    scoped = scoped_existing_table_paths(
        raw_paths=[str(pickle_path), str(csv_path)],
        root=root,
        field_name="paths",
    )
    assert scoped == sorted([pickle_path.resolve(), csv_path.resolve()])

    outside = tmp_path.parent / "outside.csv"
    outside.write_text("x\n1\n", encoding="utf-8")
    bad_suffix = root / "table.txt"
    bad_suffix.write_text("x\n1\n", encoding="utf-8")
    missing = root / "missing.csv"

    with pytest.raises(ValueError):
        scoped_existing_table_paths(
            raw_paths=[str(outside)],
            root=root,
            field_name="paths",
        )

    with pytest.raises(ValueError):
        scoped_existing_table_paths(
            raw_paths=[str(bad_suffix)],
            root=root,
            field_name="paths",
        )

    with pytest.raises(ValueError):
        scoped_existing_table_paths(
            raw_paths=[str(missing)],
            root=root,
            field_name="paths",
        )

    with pytest.raises(ValueError):
        scoped_existing_table_paths(
            raw_paths=[str(csv_path), str(csv_path)],
            root=root,
            field_name="paths",
        )
