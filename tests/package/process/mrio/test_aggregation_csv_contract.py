import pandas as pd
import pytest

from pyaesa.process.mrios.utils.aggregation.aggregation import (
    agg_map_fingerprint,
    build_agg_vector,
    build_aggregation_spec,
    read_agg_map,
    resolve_agg_map_for_year,
    year_specific_weight_years,
)


def test_read_agg_map_requires_structured_columns(tmp_path) -> None:
    csv_path = tmp_path / "agg_sec_template.csv"
    pd.DataFrame(
        {
            "source_name": ["A"],
            "target_name": ["A"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        read_agg_map(csv_path)


def test_build_agg_vector_uses_structured_agg_map(tmp_path) -> None:
    csv_path = tmp_path / "agg_sec_template.csv"
    pd.DataFrame(
        {
            "original_classification": ["A", "B"],
            "aggregated_mrio": ["X", "Y"],
        }
    ).to_csv(csv_path, index=False)

    mapping = read_agg_map(csv_path)
    result = build_agg_vector(
        original_order=["A", "B"],
        map_df=mapping,
        label_kind="sector",
        csv_path=csv_path,
    )
    assert result == ["X", "Y"]


def test_weighted_agg_map_builds_aggregation_spec(tmp_path) -> None:
    csv_path = tmp_path / "agg_sec_weighted.csv"
    pd.DataFrame(
        {
            "original_classification": ["A", "A", "B"],
            "aggregated_mrio": ["X", "Y", "Y"],
            "weight": [0.85, 0.15, 1.0],
        }
    ).to_csv(csv_path, index=False)

    mapping = read_agg_map(csv_path)
    spec = build_aggregation_spec(
        original_order=["A", "B"],
        map_df=mapping,
        label_kind="sector",
        csv_path=csv_path,
    )

    assert spec.weighted is True
    assert spec.aggregated_labels == ("X", "Y")
    assert spec.rows == ((0, 0, 0.85), (0, 1, 0.15), (1, 1, 1.0))
    assert len(agg_map_fingerprint(mapping)) == 16


def test_year_specific_weights_require_an_exact_year(tmp_path) -> None:
    csv_path = tmp_path / "agg_weighted_by_year.csv"
    pd.DataFrame(
        {
            "original_classification": ["A", "A", "B"],
            "aggregated_mrio": ["X", "Y", "Y"],
            "weight": [0.8, 0.2, 1.0],
            "weight::2021": [0.6, 0.4, 1.0],
            "weight::2020": [0.3, 0.7, 1.0],
        }
    ).to_csv(csv_path, index=False)

    mapping = read_agg_map(csv_path)
    assert year_specific_weight_years(mapping) == (2020, 2021)
    assert resolve_agg_map_for_year(mapping, year=2020)["weight"].tolist() == [0.3, 0.7, 1.0]
    with pytest.raises(ValueError, match="no weights for year 2019"):
        resolve_agg_map_for_year(mapping, year=2019)

    changed = mapping.copy()
    changed["weight::2020"] = [0.4, 0.6, 1.0]
    assert agg_map_fingerprint(changed) != agg_map_fingerprint(mapping)

    unweighted = mapping.loc[:, ["original_classification", "aggregated_mrio"]]
    assert resolve_agg_map_for_year(unweighted, year=2020).equals(unweighted)


@pytest.mark.parametrize("column", ["weight::20200", "weight::year"])
def test_year_specific_weights_require_exact_year_column_names(tmp_path, column: str) -> None:
    csv_path = tmp_path / "agg_invalid_year.csv"
    pd.DataFrame(
        {
            "original_classification": ["A"],
            "aggregated_mrio": ["X"],
            "weight": [1.0],
            column: [1.0],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="invalid year specific weight columns"):
        read_agg_map(csv_path)


def test_year_specific_weights_allow_no_unused_default(tmp_path) -> None:
    csv_path = tmp_path / "agg_missing_default.csv"
    pd.DataFrame(
        {
            "original_classification": ["A"],
            "aggregated_mrio": ["X"],
            "weight::2020": [1.0],
        }
    ).to_csv(csv_path, index=False)

    mapping = read_agg_map(csv_path)
    resolved = resolve_agg_map_for_year(mapping, year=2020)

    assert resolved["weight"].tolist() == [1.0]
    with pytest.raises(ValueError, match="no weights for year 2021"):
        resolve_agg_map_for_year(mapping, year=2021)


def test_weighted_aggregation_spec_ignores_non_source_rows_and_requires_source_labels(
    tmp_path,
) -> None:
    csv_path = tmp_path / "agg_sec_weighted.csv"
    pd.DataFrame(
        {
            "original_classification": ["A", "B", "C"],
            "aggregated_mrio": ["X", "Y", "Ignored"],
            "weight": [1.0, 1.0, 1.0],
        }
    ).to_csv(csv_path, index=False)

    mapping = read_agg_map(csv_path)
    spec = build_aggregation_spec(
        original_order=["A", "B"],
        map_df=mapping,
        label_kind="sector",
        csv_path=csv_path,
    )
    assert spec.aggregated_labels == ("X", "Y", "Ignored")
    assert spec.rows == ((0, 0, 1.0), (1, 1, 1.0))

    with pytest.raises(ValueError, match="Missing 1 sector labels"):
        build_aggregation_spec(
            original_order=["A", "B", "D"],
            map_df=mapping,
            label_kind="sector",
            csv_path=csv_path,
        )


def test_weighted_agg_map_rejects_invalid_weights(tmp_path) -> None:
    invalid_cases = [
        pd.DataFrame(
            {
                "original_classification": ["A"],
                "aggregated_mrio": ["X"],
                "weight": [None],
            }
        ),
        pd.DataFrame(
            {
                "original_classification": ["A"],
                "aggregated_mrio": ["X"],
                "weight": ["bad"],
            }
        ),
        pd.DataFrame(
            {
                "original_classification": ["A"],
                "aggregated_mrio": ["X"],
                "weight": [-0.1],
            }
        ),
        pd.DataFrame(
            {
                "original_classification": ["A", "A"],
                "aggregated_mrio": ["X", "Y"],
                "weight": [0.4, 0.4],
            }
        ),
        pd.DataFrame(
            {
                "original_classification": ["A", "A"],
                "aggregated_mrio": ["X", "X"],
                "weight": [0.4, 0.6],
            }
        ),
        pd.DataFrame(
            {
                "original_classification": ["A", "A"],
                "aggregated_mrio": ["X", "Y"],
                "weight": [0.5, 0.5],
                "weight::2020": [0.4, 0.4],
            }
        ),
    ]
    for index, frame in enumerate(invalid_cases):
        csv_path = tmp_path / f"invalid_{index}.csv"
        frame.to_csv(csv_path, index=False)
        with pytest.raises(ValueError):
            read_agg_map(csv_path)
