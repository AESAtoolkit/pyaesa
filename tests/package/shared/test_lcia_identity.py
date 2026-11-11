"""Real file and public workflow checks for scientific result identities."""

import json
from pathlib import Path

import pandas as pd
import pytest

from pyaesa import (
    deterministic_acc,
    deterministic_asr,
    deterministic_asocc,
    deterministic_io_lca,
    process_mrio,
)
from pyaesa.process.mrios.utils.io.metadata import _read_metadata
from pyaesa.process.mrios.utils.io.paths import _get_metadata_path
from pyaesa.shared.lcia.identity import characterization_sha256
from pyaesa.shared.lcia.paths import (
    characterization_matrix_path,
    responsibility_periods_csv_path,
    static_cc_csv_path,
)
from tests.package.helpers.asr_dummy_repo import prepare_static_asr_io_lca_repo
from tests.package.helpers.acc_dummy_repo import prepare_exiobase_repo_with_years
from tests.package.helpers.data_processing_dummy import (
    write_characterization_matrix,
    write_exio_archive_files,
)


def _rewrite_editorial_metadata(path: Path) -> None:
    """Edit citations and CSV layout without changing scientific values."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame["source"] = "Updated citation, without numerical changes"
    frame["comment"] = "Revised explanatory note"
    frame.iloc[::-1, ::-1].to_csv(path, index=False, lineterminator="\r\n")


@pytest.mark.parametrize(
    "change", ["factor", "missing_identity", "missing_year", "append", "editorial"]
)
def test_process_mrio_rejects_stale_or_unverified_factors_without_overwriting(
    project_repo: Path, change: str
) -> None:
    source = "exiobase_396_ixi"
    write_exio_archive_files(project_repo, source=source, years=[2019])
    matrix = write_characterization_matrix(project_repo, source_key=source, method_name="pb_lcia")
    report = process_mrio(source, years=[2019], lcia_method="pb_lcia")
    assert report is not None
    assert report.processed == [2019]
    entry = _read_metadata(source, matrix_version=None)["years"]["2019"]
    assert entry["characterization_inputs"] == {
        "pb_lcia": characterization_sha256(source=source, lcia_method="pb_lcia")
    }
    assert process_mrio(source, years=[2019], lcia_method="pb_lcia") is None
    metadata_path = _get_metadata_path(source)
    if change == "editorial":
        _rewrite_editorial_metadata(matrix)
        assert process_mrio(source, years=[2019], lcia_method="pb_lcia") is None
        assert (
            _read_metadata(source, matrix_version=None)["years"]["2019"]["characterization_inputs"]
            == entry["characterization_inputs"]
        )
        return
    if change in {"missing_identity", "missing_year"}:
        metadata = _read_metadata(source, matrix_version=None)
        if change == "missing_year":
            del metadata["years"]["2019"]
        else:
            del metadata["years"]["2019"]["characterization_inputs"]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    else:
        factors = pd.read_csv(matrix)
        factors["factor"] = factors["factor"] * 2.0
        factors.to_csv(matrix, index=False)
    method = "pb_lcia"
    if change == "append":
        method = "new_method"
        write_characterization_matrix(project_repo, source_key=source, method_name=method)
    before = metadata_path.read_bytes()
    with pytest.raises(ValueError, match="changed or unverified"):
        process_mrio(source, years=[2019], lcia_method=method)
    assert metadata_path.read_bytes() == before


@pytest.mark.parametrize(
    "change",
    ["factor", "processed_identity", "result_identity", "rps", "cf_editorial", "rps_editorial"],
)
def test_io_lca_checks_scientific_inputs_before_completed_reuse(
    io_lca_dummy_repo, change: str
) -> None:
    repo = io_lca_dummy_repo
    arguments = {
        "project_name": "scientific_identity",
        "source": repo.source,
        "years": [2019],
        "lcia_method": repo.lcia_method,
        "fu_code": "L1.a",
        "r_f": "FR",
        "figures": False,
    }
    report = deterministic_io_lca(**arguments)
    assert report is not None
    output = report.main_result_paths[0]
    original_output = output.read_bytes()
    if change.endswith("_editorial"):
        resolver = (
            characterization_matrix_path
            if change == "cf_editorial"
            else responsibility_periods_csv_path
        )
        _rewrite_editorial_metadata(resolver(source=repo.source, lcia_method=repo.lcia_method))
        reused = deterministic_io_lca(**arguments)
        assert reused is not None
        assert reused.main_result_paths == report.main_result_paths
        assert output.read_bytes() == original_output
        return
    if change == "factor":
        path = characterization_matrix_path(source=repo.source, lcia_method=repo.lcia_method)
        frame = pd.read_csv(path)
        frame["factor"] = frame["factor"] * 2.0
        frame.to_csv(path, index=False)
    elif change == "rps":
        path = responsibility_periods_csv_path(source=repo.source, lcia_method=repo.lcia_method)
        frame = pd.read_csv(path)
        frame["responsibility_period_years"] = frame["responsibility_period_years"] * 2
        frame.to_csv(path, index=False)
    else:
        path = (
            _get_metadata_path(repo.source)
            if change == "processed_identity"
            else report.metadata_path
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if change == "processed_identity":
            del payload["years"]["2019"]["characterization_inputs"]
        else:
            del payload["arguments"]["scientific_inputs"]
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="changed or unverified"):
        deterministic_io_lca(**arguments)
    assert output.read_bytes() == original_output


@pytest.mark.parametrize("change", ["factor", "rps", "cf_editorial", "rps_editorial"])
def test_asocc_checks_scientific_inputs_before_completed_reuse(allocation_dummy_repo, change: str):
    repo = allocation_dummy_repo
    prepare_exiobase_repo_with_years(
        repo, historical_years=list(range(1995, 2006)), scenario_years=[2030]
    )
    arguments = {
        "project_name": "scientific_identity",
        "source": "exiobase_396_ixi",
        "years": [2005],
        "lcia_method": "gwp100_lcia",
        "fu_code": "L1.a",
        "method_plan": "default",
        "l1_methods": ["AR(E^{CBA_FD})"],
        "figures": False,
    }
    report = deterministic_asocc(**arguments)
    assert report is not None
    if change.endswith("_editorial"):
        resolver = (
            characterization_matrix_path
            if change == "cf_editorial"
            else responsibility_periods_csv_path
        )
        _rewrite_editorial_metadata(resolver(source="exiobase_396_ixi", lcia_method="gwp100_lcia"))
        reused = deterministic_asocc(**arguments)
        assert reused is not None
        assert reused.reuse_status == "reused_exact"
        return
    resolver = (
        characterization_matrix_path if change == "factor" else responsibility_periods_csv_path
    )
    path = resolver(source="exiobase_396_ixi", lcia_method="gwp100_lcia")
    frame = pd.read_csv(path)
    column = "factor" if change == "factor" else "responsibility_period_years"
    frame[column] = frame[column] * 2
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="changed or unverified"):
        deterministic_asocc(**arguments)


@pytest.mark.parametrize(
    "family,change",
    [
        ("acc", "cc"),
        ("asr", "cc"),
        ("asr", "rps"),
        ("asr", "factor"),
        ("acc", "cc_editorial"),
        ("asr", "cc_editorial"),
        ("asr", "rps_editorial"),
        ("asr", "cf_editorial"),
    ],
)
def test_static_consumers_reject_changed_cc_before_completed_reuse(
    allocation_dummy_repo, family: str, change: str
):
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    arguments = {
        "project_name": "capacity_identity",
        "source": "exiobase_396_ixi",
        "years": [2005],
        "lcia_method": "gwp100_lcia",
        "fu_code": "L2.a.a",
        "base_asocc_args": {
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "include_lcia_based_allocation_methods": False,
        },
        "base_cc_args": {"static": {"exclude_max_cc": True}},
        "figures": False,
        "subfigures": False,
    }
    function = deterministic_acc if family == "acc" else deterministic_asr
    if family == "asr":
        arguments["lca_args"] = {"external_lca": {"active": False}, "io_lca": {"active": True}}
    first = function(**arguments)
    assert first is not None
    second = function(**arguments)
    assert second.branches[0].reuse_status == "reused_exact"
    if change.endswith("_editorial"):
        if change == "cc_editorial":
            path = static_cc_csv_path(lcia_method="gwp100_lcia")
        else:
            resolver = (
                characterization_matrix_path
                if change == "cf_editorial"
                else responsibility_periods_csv_path
            )
            path = resolver(source="exiobase_396_ixi", lcia_method="gwp100_lcia")
        _rewrite_editorial_metadata(path)
        reused = function(**arguments)
        assert reused.branches[0].reuse_status == "reused_exact"
        return
    if change == "cc":
        path = static_cc_csv_path(lcia_method="gwp100_lcia")
        column = "min_cc"
    else:
        resolver = (
            characterization_matrix_path if change == "factor" else responsibility_periods_csv_path
        )
        path = resolver(source="exiobase_396_ixi", lcia_method="gwp100_lcia")
        column = "factor" if change == "factor" else "responsibility_period_years"
    frame = pd.read_csv(path)
    frame[column] = frame[column] * 0.9
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="branch identity changed|changed or unverified"):
        function(**arguments)
