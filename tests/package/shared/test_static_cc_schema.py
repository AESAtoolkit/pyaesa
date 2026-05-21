from pathlib import Path

import pandas as pd
import pytest

from pyaesa.shared.lcia import static_cc_schema as schema_mod


def test_schema_contracts_cover_csv_text_and_label_building() -> None:
    assert schema_mod._csv_text(None) == ""  # noqa: SLF001
    assert schema_mod._csv_text(float("nan")) == ""  # noqa: SLF001
    assert schema_mod._csv_text("  value  ") == "value"  # noqa: SLF001

    standard_row = pd.Series({"impact_full_name": " ", "impact": "  E1  "})
    assert schema_mod._build_standard_label(row=standard_row) == "E1"  # noqa: SLF001

    standard_named_row = pd.Series({"impact_full_name": "  Full Name ", "impact": "E1"})
    assert schema_mod._build_standard_label(row=standard_named_row) == "Full Name"  # noqa: SLF001

    planetary_row = pd.Series(
        {
            "Planetary boundary": "  PB  ",
            "Control variable": "  Control  ",
        }
    )
    assert schema_mod._build_planetary_boundary_label(row=planetary_row) == "PB: Control"  # noqa: SLF001

    planetary_boundary_only_row = pd.Series(
        {
            "Planetary boundary": "  PB  ",
            "Control variable": " ",
        }
    )
    assert schema_mod._build_planetary_boundary_label(row=planetary_boundary_only_row) == "PB"  # noqa: SLF001


def test_detect_static_cc_schema_covers_all_schema_paths(tmp_path: Path) -> None:
    standard_frame = pd.DataFrame(
        {
            "impact_full_name": ["Full"],
            "impact": ["I1"],
            "impact_unit": ["kg"],
            "min_cc": [1.0],
            "max_cc": [2.0],
        }
    )
    assert (
        schema_mod.detect_static_cc_schema(
            frame=standard_frame,
            path=tmp_path / "standard.csv",
        )
        == "standard"
    )

    planetary_frame = pd.DataFrame(
        {
            "Planetary boundary": ["PB"],
            "Control variable": ["Control"],
            "impact": ["I2"],
            "impact_unit": ["kg"],
            "min_cc": [3.0],
            "max_cc": [4.0],
        }
    )
    assert (
        schema_mod.detect_static_cc_schema(frame=planetary_frame, path=tmp_path / "planetary.csv")
        == "planetary boundary"
    )

    with pytest.raises(ValueError):
        schema_mod.detect_static_cc_schema(
            frame=pd.DataFrame({"impact_full_name": ["Full"], "impact": ["I1"]}),
            path=tmp_path / "missing_standard.csv",
        )

    with pytest.raises(ValueError):
        schema_mod.detect_static_cc_schema(
            frame=pd.DataFrame(
                {
                    "Planetary boundary": ["PB"],
                    "Control variable": ["Control"],
                    "impact": ["I2"],
                }
            ),
            path=tmp_path / "missing_planetary.csv",
        )

    with pytest.raises(ValueError):
        schema_mod.detect_static_cc_schema(
            frame=pd.DataFrame({"impact": ["I3"], "value": [1.0]}),
            path=tmp_path / "unsupported.csv",
        )


def test_standardize_static_cc_rows_covers_standard_and_planetary_norms(tmp_path: Path) -> None:
    standard_frame = pd.DataFrame(
        [
            {
                "impact_full_name": " ",
                "impact": "I1",
                "impact_unit": "kg",
                "min_cc": 1,
                "max_cc": 2,
            },
            {
                "impact_full_name": "Full",
                "impact": "I2",
                "impact_unit": "kg",
                "min_cc": 3.0,
                "max_cc": 4.0,
            },
            {
                "impact_full_name": "ignored",
                "impact": " ",
                "impact_unit": "kg",
                "min_cc": 5.0,
                "max_cc": 6.0,
            },
        ]
    )
    schema_kind, rows = schema_mod.standardize_static_cc_rows(
        frame=standard_frame,
        path=tmp_path / "standard.csv",
    )
    assert schema_kind == "standard"
    assert rows == (
        schema_mod.NormalizedStaticCCRow(
            impact="I1",
            impact_unit="kg",
            min_cc=1.0,
            max_cc=2.0,
            impact_full_name_normalized="I1",
            planetary_boundary=None,
            control_variable=None,
        ),
        schema_mod.NormalizedStaticCCRow(
            impact="I2",
            impact_unit="kg",
            min_cc=3.0,
            max_cc=4.0,
            impact_full_name_normalized="Full",
            planetary_boundary=None,
            control_variable=None,
        ),
    )

    planetary_frame = pd.DataFrame(
        [
            {
                "Planetary boundary": "PB",
                "Control variable": "Control",
                "impact": "I3",
                "impact_unit": "t",
                "min_cc": 7,
                "max_cc": 8,
            },
            {
                "Planetary boundary": "PB2",
                "Control variable": " ",
                "impact": "I4",
                "impact_unit": "t",
                "min_cc": 9.0,
                "max_cc": 10.0,
            },
        ]
    )
    planetary_kind, planetary_rows = schema_mod.standardize_static_cc_rows(
        frame=planetary_frame,
        path=tmp_path / "planetary.csv",
    )
    assert planetary_kind == "planetary boundary"
    assert planetary_rows == (
        schema_mod.NormalizedStaticCCRow(
            impact="I3",
            impact_unit="t",
            min_cc=7.0,
            max_cc=8.0,
            impact_full_name_normalized="PB: Control",
            planetary_boundary="PB",
            control_variable="Control",
        ),
        schema_mod.NormalizedStaticCCRow(
            impact="I4",
            impact_unit="t",
            min_cc=9.0,
            max_cc=10.0,
            impact_full_name_normalized="PB2",
            planetary_boundary="PB2",
            control_variable=None,
        ),
    )


def test_standardize_static_cc_rows_covers_failure_branches(tmp_path: Path) -> None:
    duplicate_frame = pd.DataFrame(
        [
            {
                "impact_full_name": "Full",
                "impact": "I1",
                "impact_unit": "kg",
                "min_cc": 1.0,
                "max_cc": 2.0,
            },
            {
                "impact_full_name": "Full2",
                "impact": "I1",
                "impact_unit": "kg",
                "min_cc": 3.0,
                "max_cc": 4.0,
            },
        ]
    )
    with pytest.raises(ValueError):
        schema_mod.standardize_static_cc_rows(frame=duplicate_frame, path=tmp_path / "dup.csv")

    missing_unit_frame = pd.DataFrame(
        [
            {
                "impact_full_name": "Full",
                "impact": "I2",
                "impact_unit": " ",
                "min_cc": 1.0,
                "max_cc": 2.0,
            }
        ]
    )
    with pytest.raises(ValueError):
        schema_mod.standardize_static_cc_rows(
            frame=missing_unit_frame,
            path=tmp_path / "missing_unit.csv",
        )

    missing_numeric_frame = pd.DataFrame(
        [
            {
                "impact_full_name": "Full",
                "impact": "I3",
                "impact_unit": "kg",
                "min_cc": None,
                "max_cc": 2.0,
            }
        ]
    )
    with pytest.raises(ValueError):
        schema_mod.standardize_static_cc_rows(
            frame=missing_numeric_frame,
            path=tmp_path / "missing_numeric.csv",
        )

    non_numeric_frame = missing_numeric_frame.copy()
    non_numeric_frame.loc[0, "min_cc"] = "bad"
    with pytest.raises(ValueError):
        schema_mod.standardize_static_cc_rows(
            frame=non_numeric_frame,
            path=tmp_path / "non_numeric.csv",
        )

    no_usable_rows_frame = pd.DataFrame(
        [
            {
                "impact_full_name": "Full",
                "impact": " ",
                "impact_unit": "kg",
                "min_cc": 1.0,
                "max_cc": 2.0,
            }
        ]
    )
    with pytest.raises(ValueError):
        schema_mod.standardize_static_cc_rows(
            frame=no_usable_rows_frame,
            path=tmp_path / "no_usable.csv",
        )
