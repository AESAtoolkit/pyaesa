import shutil
from pathlib import Path
from importlib import metadata as importlib_metadata
from types import SimpleNamespace

import pandas as pd
import pytest

from pyaesa.process.mrios.process_mrio import process_mrio
from pyaesa.process.mrios.utils.io.metadata import _read_metadata
from pyaesa.process.mrios.utils.io.paths import (
    _get_group_map_path,
    _get_mrio_raw_corrected_values_log_path,
    _get_year_saved_dir,
)
from pyaesa.process.mrios.utils.pipeline.contracts import ProcessReportMRIO
from pyaesa.process.mrios.utils.pipeline.metadata_payloads import (
    extract_preclip_extension_payload,
    extract_root_extension_payload,
)
from pyaesa.process.mrios.utils.pipeline.runner import run_process_mrio
from pyaesa.process.mrios.utils.pipeline.runtime_environment import runtime_env_versions
from pyaesa.process.mrios.utils.raw_corrections.reporting import record_raw_correction_payload
from pyaesa.process.mrios.utils.raw_corrections.runtime import AppliedCorrectionSummary
from tests.package.helpers.data_processing_dummy import (
    write_characterization_matrix,
    write_exio_archive_files,
    write_oecd_raw_csv_files,
)


def test_process_mrio_wrapper_validates_inputs_and_small_contracts(project_repo: Path) -> None:
    assert extract_root_extension_payload(None) is None
    assert extract_root_extension_payload({"extensions": {"pb_lcia": {}}}) == {"pb_lcia": {}}

    assert extract_preclip_extension_payload(None) is None
    assert extract_preclip_extension_payload({"preclip": None}) is None
    assert extract_preclip_extension_payload({"preclip": {"extensions": {"pb_lcia": []}}}) == {
        "pb_lcia": []
    }

    runtime_env = runtime_env_versions()
    assert "python" in runtime_env
    missing_pkg_env = runtime_env_versions(
        package_names=("missing_pkg",),
        version_resolver=lambda _name: (_ for _ in ()).throw(
            importlib_metadata.PackageNotFoundError("missing")
        ),
    )
    assert missing_pkg_env == {"python": runtime_env["python"]}

    with pytest.raises(ValueError):
        process_mrio("unknown")
    with pytest.raises(ValueError):
        process_mrio("oecd_v2025", lcia_method="pb_lcia")
    with pytest.raises(ValueError):
        process_mrio("oecd_v2025", group_reg=True)
    with pytest.raises(ValueError):
        process_mrio("oecd_v2025", group_version="demo")
    missing_archive_report = process_mrio("oecd_v2025")
    assert missing_archive_report is not None
    assert missing_archive_report.processed == []
    assert 1995 in missing_archive_report.errors
    assert missing_archive_report.errors[1995]


def test_process_report_mrio_groups_raw_corrected_value_years() -> None:
    report = ProcessReportMRIO(source="exiobase_3102_ixi", requested=[2020, 2021, 2022])
    report.processed = [2020, 2021, 2022]
    report.raw_corrected_value_scopes = [
        {
            "year": 2020,
            "region": "LU",
            "extension": "water",
            "stressor_family": "water consumption stressors",
            "correction_method": "ols_level",
            "correction_reason": "LU correction: 2020 extension data are incoherently too small.",
        },
        {
            "year": 2021,
            "region": "CH",
            "extension": "nutrients",
            "stressor_family": "P - agriculture - water",
            "correction_method": "ols_level",
            "correction_reason": (
                "CH correction: 2021 and 2022 extension data are incoherently too small."
            ),
        },
        {
            "year": 2022,
            "region": "CH",
            "extension": "nutrients",
            "stressor_family": "P - agriculture - water",
            "correction_method": "ols_level",
            "correction_reason": (
                "CH correction: 2021 and 2022 extension data are incoherently too small."
            ),
        },
        {
            "year": 2018,
            "region": "MT",
            "extension": "land",
            "stressor_family": "Forest",
            "correction_method": "donor_sector_intensity",
            "correction_reason": (
                "MT correction: missing extension data recorded at 0, and some present "
                "values are incoherently too small."
            ),
        },
        {
            "year": 2016,
            "region": "ZZ",
            "extension": "misc",
            "stressor_family": "Other",
            "correction_method": "custom_method",
            "correction_reason": "Custom correction.",
        },
    ]
    report.raw_corrected_value_log_paths = [Path("C:/tmp/raw_corrected_values.csv")]

    text = str(report)
    compact_text = " ".join(text.split())
    assert compact_text
    assert text.strip()
    assert "raw_corrected_values.csv" in text
    assert all(len(line) <= 100 for line in text.splitlines() if "C:/" not in line)


def test_process_report_mrio_adds_pxp_malta_value_added_note() -> None:
    report = ProcessReportMRIO(source="exiobase_3102_pxp", requested=[2010, 2011, 2013, 2018])
    report.processed = [2010, 2011, 2013, 2018]
    report.raw_corrected_value_scopes = [
        {
            "year": 2010,
            "region": "MT",
            "extension": "land",
            "stressor_family": "Forest",
            "correction_method": "donor_sector_intensity",
            "correction_reason": (
                "MT correction: missing extension data recorded at 0, and some present "
                "values are incoherently too small."
            ),
        },
        {
            "year": 2011,
            "region": "MT",
            "extension": "land",
            "stressor_family": "Forest",
            "correction_method": "donor_sector_intensity",
            "correction_reason": (
                "MT correction: missing extension data recorded at 0, and some present "
                "values are incoherently too small."
            ),
        },
    ]

    text = str(report)
    assert text.strip()
    assert all(len(line) <= 100 for line in text.splitlines())


def test_run_process_mrio_processes_then_skips_cached_oecd_outputs(project_repo: Path) -> None:
    write_oecd_raw_csv_files(project_repo, years=[2020, 2021])
    saved_dir = _get_year_saved_dir("oecd_v2025", 2020)
    if saved_dir.exists():
        shutil.rmtree(saved_dir)
    assert not saved_dir.exists()

    first_report = run_process_mrio(
        source="oecd_v2025",
        years=[2020],
        refresh=False,
    )
    assert first_report is not None
    assert first_report.processed == [2020]
    assert first_report.clipping_unit == "Million USD"
    assert (saved_dir / "utility_propag_uncasext" / "x_to_rc.pickle").exists()
    assert (saved_dir / "enacting_metrics" / "units.json").exists()
    assert not (saved_dir / "preclip").exists()
    assert not (saved_dir / "extensions").exists()

    second_report = run_process_mrio(
        source="oecd_v2025",
        years=[2020],
        refresh=False,
    )
    assert second_report is None
    assert not (saved_dir / "preclip").exists()
    assert not (saved_dir / "extensions").exists()

    calc_all_report = run_process_mrio(
        source="oecd_v2025",
        years=[2021],
        refresh=True,
        pymrio_calc_all=True,
    )
    assert calc_all_report is not None
    calc_all_saved_dir = _get_year_saved_dir("oecd_v2025", 2021)
    assert (calc_all_saved_dir / "preclip" / "A.pickle").exists()


def test_run_process_mrio_records_raw_corrected_values_summary(project_repo: Path) -> None:
    report = ProcessReportMRIO(source="exiobase_3102_ixi", requested=[1995])
    iosys = SimpleNamespace(
        _raw_corrected_values_summary=AppliedCorrectionSummary(
            source="exiobase_3102_ixi",
            year=1995,
            row_count=50,
            log_path=None,
        )
    )
    saved_dir = project_repo / "tmp_saved"

    payload = record_raw_correction_payload(
        iosys=iosys,
        report=report,
        source_key="exiobase_3102_ixi",
        matrix_version=None,
        saved_dir=saved_dir,
        year=1995,
    )
    assert payload is not None
    assert payload["row_count"] == 50
    assert report.raw_corrected_value_row_count == 50
    assert report.raw_corrected_value_log_paths == [
        _get_mrio_raw_corrected_values_log_path("exiobase_3102_ixi", matrix_version=None)
    ]
    report_text = " ".join(str(report).split())
    assert report_text
    assert "summary_lines" in payload
    duplicate_payload = record_raw_correction_payload(
        iosys=iosys,
        report=report,
        source_key="exiobase_3102_ixi",
        matrix_version=None,
        saved_dir=saved_dir,
        year=1995,
    )
    assert duplicate_payload is not None
    assert len(report.raw_corrected_value_log_paths) == 1
    no_log_payload = record_raw_correction_payload(
        iosys=SimpleNamespace(
            _raw_corrected_values_summary=AppliedCorrectionSummary(
                source="exiobase_3102_ixi",
                year=2023,
                row_count=0,
                log_path=None,
            )
        ),
        report=report,
        source_key="exiobase_3102_ixi",
        matrix_version=None,
        saved_dir=saved_dir,
        year=2023,
    )
    assert no_log_payload is not None
    assert no_log_payload["log_path"] == ""
    empty_payload = record_raw_correction_payload(
        iosys=SimpleNamespace(),
        report=report,
        source_key="exiobase_3102_ixi",
        matrix_version=None,
        saved_dir=saved_dir,
        year=1995,
    )
    assert empty_payload is None


def test_run_process_mrio_covers_refresh_grouping_and_recoverable_errors(
    project_repo: Path,
) -> None:
    full_dir = write_oecd_raw_csv_files(project_repo, years=[2019])
    (full_dir / "ICIO2025_2020.csv").write_text("bad", encoding="utf-8")
    reg_path = _get_group_map_path("oecd_v2025", kind="reg", group_version="demo")
    sec_path = _get_group_map_path("oecd_v2025", kind="sec", group_version="demo")
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "original_classification": ["R1", "R2"],
            "grouped_mrio": ["EU", "ROW"],
        }
    ).to_csv(reg_path, index=False)
    pd.DataFrame(
        {
            "original_classification": ["S1", "S2"],
            "grouped_mrio": ["Energy", "Other"],
        }
    ).to_csv(sec_path, index=False)

    grouped_report = run_process_mrio(
        source="oecd_v2025",
        years=[2019, 2020],
        refresh=True,
        group_reg=True,
        group_sec=True,
        group_version="demo",
    )
    assert grouped_report is not None
    assert grouped_report.processed == [2019]
    assert 2020 in grouped_report.errors

    stale_saved_dir = _get_year_saved_dir("oecd_v2025", 2019, matrix_version="demo")
    stale_marker = stale_saved_dir / "stale.txt"
    stale_marker.write_text("old", encoding="utf-8")
    run_process_mrio(
        source="oecd_v2025",
        years=[2019],
        refresh=True,
        group_reg=True,
        group_sec=True,
        group_version="demo",
    )
    assert not stale_marker.exists()


def test_run_process_mrio_covers_exio_lcia_persistence_and_missing_methods(
    project_repo: Path,
) -> None:
    write_exio_archive_files(project_repo, source="exiobase_396_ixi", years=[2019, 2020])
    write_characterization_matrix(
        project_repo,
        source_key="exiobase_396_ixi",
        method_name="pb_lcia",
        extension_name="satellite",
    )

    exio_report = run_process_mrio(
        source="exiobase_396_ixi",
        years=[2019],
        refresh=True,
        lcia_method=["pb_lcia"],
        keep_intermediate_uncasext=True,
        pymrio_calc_all=True,
    )
    assert exio_report is not None
    assert exio_report.processed == [2019]
    exio_saved_dir = _get_year_saved_dir("exiobase_396_ixi", 2019)
    assert (exio_saved_dir / "preclip" / "A.pickle").exists()
    assert (exio_saved_dir / "preclip" / "extensions" / "pb_lcia" / "F.pickle").exists()
    assert (exio_saved_dir / "extensions" / "pb_lcia" / "S.pickle").exists()
    assert (exio_saved_dir / "enacting_metrics" / "level_1" / "pb_lcia" / "F_Y.pickle").exists()

    metadata = _read_metadata("exiobase_396_ixi", matrix_version=None)
    year_entry = metadata["years"]["2019"]
    assert year_entry["preclip"]["pymrio_calc_all"] is True
    assert year_entry["enacting_metrics"]["lcia_methods"] == ["pb_lcia"]
    assert year_entry["lcia_status"] == {"pb_lcia": {"available": True}}

    assert (
        run_process_mrio(
            source="exiobase_396_ixi",
            years=[2019],
            refresh=False,
            lcia_method=["pb_lcia"],
            keep_intermediate_uncasext=True,
            pymrio_calc_all=True,
        )
        is None
    )

    (exio_saved_dir / "A.pickle").unlink()
    rerun_report = run_process_mrio(
        source="exiobase_396_ixi",
        years=[2019],
        refresh=False,
        lcia_method=["pb_lcia"],
        keep_intermediate_uncasext=True,
        pymrio_calc_all=True,
    )
    assert rerun_report is not None
    assert rerun_report.processed == [2019]
    write_characterization_matrix(
        project_repo,
        source_key="exiobase_396_ixi",
        method_name="pb_lcia",
        extension_name="missing_extension",
    )

    missing_report = run_process_mrio(
        source="exiobase_396_ixi",
        years=[2020],
        refresh=True,
        lcia_method=["pb_lcia"],
    )
    assert missing_report is not None
    assert missing_report.lcia_missing_by_year == {2020: {"pb_lcia": ["missing_extension"]}}
    missing_meta = _read_metadata("exiobase_396_ixi", matrix_version=None)
    assert missing_meta["years"]["2020"]["lcia_status"] == {
        "pb_lcia": {"available": False, "missing": ["missing_extension"]}
    }

    missing_saved_dir = _get_year_saved_dir("exiobase_396_ixi", 2020)
    (missing_saved_dir / "utility_propag_uncasext" / "x_to_rc.pickle").unlink()
    rerun_missing = run_process_mrio(
        source="exiobase_396_ixi",
        years=[2020],
        refresh=False,
        lcia_method=["pb_lcia"],
    )
    assert rerun_missing is not None
    assert rerun_missing.processed == [2020]
    assert rerun_missing.lcia_missing_by_year == {}


def test_run_process_mrio_fails_fast_on_label_drift(project_repo: Path) -> None:
    write_oecd_raw_csv_files(project_repo, years=[2019])
    write_oecd_raw_csv_files(project_repo, years=[2020], regions=("EU",), sectors=("S1", "S2"))

    with pytest.raises(ValueError):
        run_process_mrio(
            source="oecd_v2025",
            years=[2019, 2020],
            refresh=True,
        )
