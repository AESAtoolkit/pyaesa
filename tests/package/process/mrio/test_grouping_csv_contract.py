import pandas as pd
import pytest

from pyaesa.process.mrios.utils.grouping.grouping import (
    build_agg_vector,
    read_group_map,
)


def test_read_group_map_requires_structured_columns(tmp_path) -> None:
    csv_path = tmp_path / "group_sec_template.csv"
    pd.DataFrame(
        {
            "source_name": ["A"],
            "target_name": ["A"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        read_group_map(csv_path)


def test_build_agg_vector_uses_structured_group_map(tmp_path) -> None:
    csv_path = tmp_path / "group_sec_template.csv"
    pd.DataFrame(
        {
            "original_classification": ["A", "B"],
            "grouped_mrio": ["X", "Y"],
        }
    ).to_csv(csv_path, index=False)

    mapping = read_group_map(csv_path)
    result = build_agg_vector(
        original_order=["A", "B"],
        map_df=mapping,
        label_kind="sector",
        csv_path=csv_path,
    )
    assert result == ["X", "Y"]
