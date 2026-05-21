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
    normalized_cc = tables_mod.clean_characterization_matrix_frame(
        frame=source_frame,
        path=cc_path,
    )
    assert normalized_cc.equals(cleaned)

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
