"""Scientific fingerprints track calculation fields rather than CSV annotations."""

from pathlib import Path

import pandas as pd
import pytest

from pyaesa.shared.lcia.identity import (
    carrying_capacity_sha256,
    characterization_sha256,
    responsibility_period_sha256,
)
from pyaesa.shared.lcia.paths import (
    characterization_matrix_path,
    responsibility_periods_csv_path,
    static_cc_csv_path,
)


@pytest.mark.parametrize(
    "kind,column",
    [
        ("cf", "extension"),
        ("cf", "stressor"),
        ("cf", "stressor_unit"),
        ("cf", "impact"),
        ("cf", "impact_parent"),
        ("cf", "impact_unit"),
        ("cf", "factor"),
        ("rps", "impact"),
        ("rps", "impact_parent"),
        ("rps", "responsibility_period_years"),
        ("cc", "impact"),
        ("cc", "impact_unit"),
        ("cc", "min_cc"),
        ("cc", "max_cc"),
    ],
)
def test_scientific_table_fingerprint_ignores_annotations_and_tracks_values(
    project_repo: Path, kind: str, column: str
) -> None:
    """Equivalent serialization preserves identity; every calculation field matters."""
    source, method = "exiobase_3112_ixi", "gwp100_ipcc"
    if kind == "cf":
        path = characterization_matrix_path(source=source, lcia_method=method)

        def identity() -> str:
            return characterization_sha256(source=source, lcia_method=method)

    elif kind == "rps":
        path = responsibility_periods_csv_path(source=source, lcia_method=method)

        def identity() -> str:
            return responsibility_period_sha256(source=source, lcia_method=method)

    else:
        path = static_cc_csv_path(lcia_method=method)

        def identity() -> str:
            return carrying_capacity_sha256(lcia_method=method)

    original = identity()
    frame = pd.read_csv(path, float_precision="round_trip")
    frame["source"] = "Edited citation"
    frame["comment"] = "Edited explanation"
    frame.iloc[::-1, ::-1].to_csv(path, index=False, lineterminator="\r\n", float_format="%.17e")
    assert identity() == original
    if column in {"factor", "responsibility_period_years", "min_cc", "max_cc"}:
        frame[column] = frame[column] + 1
    else:
        frame[column] = frame[column].astype(str) + " changed"
    frame.to_csv(path, index=False, float_format="%.17e")
    assert len({identity(), original}) == 2


def test_optional_capacity_and_period_values_have_consistent_missing_identity(
    project_repo: Path,
) -> None:
    """Absent capacity bounds and alternate empty values have stable identities."""
    method = "gwp100_ipcc"
    cc_path = static_cc_csv_path(lcia_method=method)
    cc = pd.read_csv(cc_path).drop(columns="max_cc")
    cc.to_csv(cc_path, index=False)
    missing_bound = carrying_capacity_sha256(lcia_method=method)
    cc["max_cc"] = float("nan")
    cc.to_csv(cc_path, index=False)
    assert carrying_capacity_sha256(lcia_method=method) == missing_bound
    rps_path = responsibility_periods_csv_path(source="exiobase_3112_ixi", lcia_method=method)
    rps = pd.read_csv(rps_path)
    rps["responsibility_period_years"] = float("nan")
    rps.to_csv(rps_path, index=False)
    missing_period = responsibility_period_sha256(source="exiobase_3112_ixi", lcia_method=method)
    rps["source"] = "Updated reference"
    rps.to_csv(rps_path, index=False, na_rep="NaN")
    assert (
        responsibility_period_sha256(source="exiobase_3112_ixi", lcia_method=method)
        == missing_period
    )
