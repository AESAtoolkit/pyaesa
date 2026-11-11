from pathlib import Path

import pandas as pd
import pytest

from pyaesa.shared.lcia import prerequisite_tables as tables_mod


def test_lcia_prerequisite_tables_cover_cleaning_and_validation_paths(
    tmp_path: Path,
) -> None:
    source_frame = pd.DataFrame(
        {
            " extension ": ["A"],
            "Unnamed: 0": [99],
            " impact ": ["GWP"],
            " value ": [1.5],
        }
    )

    cleaned = tables_mod._clean_columns(source_frame)  # noqa: SLF001
    assert cleaned.columns.tolist() == ["extension", "impact", "value"]
    assert cleaned.to_dict(orient="records") == [{"extension": "A", "impact": "GWP", "value": 1.5}]

    cc_path = tmp_path / "cc.csv"
    source_frame = pd.DataFrame(
        [
            {
                "extension": "A",
                "stressor": "co2",
                "stressor_unit": "kg",
                "factor": 1.5,
                "impact": "GWP",
                "impact_parent": "GWP",
                "impact_unit": "kg CO2 eq",
            }
        ]
    )
    normalized_cc = tables_mod.clean_characterization_matrix_frame(
        frame=source_frame,
        path=cc_path,
    )
    assert normalized_cc.to_dict(orient="records") == source_frame.to_dict(orient="records")
    for invalid_frame in (
        source_frame.iloc[:0],
        source_frame.assign(**{" extension ": "duplicate"}),
    ):
        with pytest.raises(ValueError, match="rows and unique columns"):
            tables_mod.clean_characterization_matrix_frame(frame=invalid_frame, path=cc_path)

    rp_frame = pd.DataFrame(
        {
            " impact ": ["A"],
            " value ": [2.0],
        }
    )
    rp_path = tmp_path / "rps.csv"
    normalized_rp = tables_mod.clean_responsibility_period_frame(
        frame=rp_frame,
        path=rp_path,
    )
    assert normalized_rp.columns.tolist() == ["impact", "value"]
    assert normalized_rp.to_dict(orient="records") == [{"impact": "A", "value": 2.0}]

    with pytest.raises(ValueError):
        tables_mod.clean_characterization_matrix_frame(
            frame=pd.DataFrame({" impact ": ["GWP"]}),
            path=cc_path,
        )

    with pytest.raises(ValueError):
        tables_mod.clean_responsibility_period_frame(
            frame=pd.DataFrame({" value ": [1.0]}),
            path=rp_path,
        )

    with pytest.raises(ValueError):
        tables_mod.clean_responsibility_period_frame(
            frame=pd.DataFrame(
                {
                    " impact ": ["A", "A"],
                    " responsibility_period_years ": [1, 2],
                }
            ),
            path=rp_path,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"factor": float("nan")},
        {"factor": float("inf")},
        {"factor": "invalid"},
        {"extension": None},
        {"stressor": " "},
        {"impact": "nan"},
        {"stressor_unit": ""},
        {"impact_parent": "null"},
    ],
)
def test_characterization_csv_rejects_missing_or_invalid_values(
    tmp_path: Path, change: dict
) -> None:
    from pyaesa.process.mrios.utils.parsers.exio_parser import _load_characterization_matrix

    row = {
        "extension": "air",
        "stressor": "co2",
        "stressor_unit": "kg",
        "factor": 1.0,
        "impact": "GWP",
        "impact_parent": "GWP",
        "impact_unit": "kg CO2 eq",
    }
    path = tmp_path / "factors.csv"
    pd.DataFrame([{**row, **change}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        _load_characterization_matrix(path)


@pytest.mark.parametrize(
    "second_row",
    [
        {},
        {"impact": "other", "stressor_unit": "t"},
        {"stressor": "methane", "impact_parent": "other"},
        {"stressor": "methane", "impact_unit": "t CO2 eq"},
        {"impact": "other", "impact_unit": "t CO2 eq"},
    ],
)
def test_characterization_csv_rejects_ambiguous_factors(tmp_path: Path, second_row: dict) -> None:
    from pyaesa.process.mrios.utils.parsers.exio_parser import _load_characterization_matrix

    row = {
        "extension": "air",
        "stressor": "co2",
        "stressor_unit": "kg",
        "factor": 1.0,
        "impact": "GWP",
        "impact_parent": "GWP",
        "impact_unit": "kg CO2 eq",
    }
    path = tmp_path / "factors.csv"
    pd.DataFrame([row, {**row, **second_row}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        _load_characterization_matrix(path)
