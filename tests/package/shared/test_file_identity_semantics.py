"""Calculation identities ignore storage and annotation-only changes."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyaesa.shared.runtime.io.file_identity import file_identity_payload, table_sha256
from pyaesa.shared.tabular.table_io import write_table
from pyaesa.shared.uncertainty_assessment.run_state.compatibility import strip_reporting_only_fields


@pytest.mark.parametrize("suffix", [".csv", ".parquet", ".pickle"])
def test_explicit_calculation_columns_ignore_annotations_and_storage(tmp_path: Path, suffix: str):
    original = pd.DataFrame(
        {"region": ["FR", "DE"], "value": [0.1, -0.0], "citation": ["first", "source"]}
    )
    first = tmp_path / f"original{suffix}"
    second = tmp_path / f"revised{suffix}"
    write_table(path=first, frame=original)
    revised = original.assign(citation=["edited citation", "changed source"], comment="new")
    revised["value"] = revised["value"] + 0.0
    revised = revised.iloc[::-1].loc[:, ["comment", "value", "citation", "region"]]
    write_table(path=second, frame=revised)
    kwargs = {"labels": ("region",), "numbers": ("value",)}
    expected = strip_reporting_only_fields(file_identity_payload(path=first, **kwargs))
    assert strip_reporting_only_fields(file_identity_payload(path=second, **kwargs)) == expected
    revised.loc[revised["region"].eq("FR"), "value"] = np.nextafter(0.1, 1.0)
    write_table(path=second, frame=revised)
    assert len({expected["sha256"], file_identity_payload(path=second, **kwargs)["sha256"]}) == 2
    revised = original.assign(region=["US", "DE"])
    write_table(path=second, frame=revised)
    assert len({expected["sha256"], file_identity_payload(path=second, **kwargs)["sha256"]}) == 2


def test_table_identity_preserves_exact_integer_labels():
    frame = pd.DataFrame({"public_row_id": [2**53], "value": [1.0]})
    first = table_sha256(frame, labels=("public_row_id",), numbers=("value",))
    frame["public_row_id"] += 1
    second = table_sha256(frame, labels=("public_row_id",), numbers=("value",))
    assert len({first, second}) == 2
    frame["public_row_id"] = frame["public_row_id"].astype(str) + " "
    whitespace = table_sha256(frame, labels=("public_row_id",), numbers=("value",))
    assert len({second, whitespace}) == 2
