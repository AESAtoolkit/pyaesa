from pathlib import Path

import pandas as pd
import pytest

from pyaesa.process.mrios.utils.aggregation.aggregation import (
    agg_map_fingerprint,
    build_agg_vector,
    read_agg_map,
    unique_in_order,
)
from pyaesa.process.mrios.utils.io.metadata import (
    _get_year_entry,
    _metadata_satisfies,
    _read_metadata,
    _remove_year_entry,
    read_processed_mrio_regions,
    _set_year_entry,
    _write_metadata,
)
from pyaesa.process.mrios.utils.io.paths import (
    _get_characterization_matrix_path,
    _get_agg_map_path,
    _get_aggregation_dir,
    _get_metadata_path,
    _get_mrio_calc_log_path,
    _get_mrio_calc_logs_dir,
    _get_mrio_raw_corrected_values_log_path,
    _get_saved_dir,
    _get_year_saved_dir,
    _resolve_version_tag,
)
from pyaesa.process.mrios.utils.pipeline.contracts import (
    ProcessReportMRIO,
    build_source_configs,
)
from pyaesa.download.mrios.utils.paths import (
    _get_exio_archive_path,
    _get_oecd_csv_path,
)
from pyaesa.download.mrios.utils.source_registry import get_mrio_entry, is_exio_mrio_source
from pyaesa.process.mrios.utils.pipeline.aggregation_validation import (
    validate_metadata_aggregation,
)


def test_processed_mrio_paths_cover_supported_layouts(project_repo: Path) -> None:
    assert get_mrio_entry("oecd_v2025").source_key == "oecd_v2025"
    with pytest.raises(ValueError):
        get_mrio_entry("unknown")

    assert is_exio_mrio_source("exiobase_396_ixi") is True
    assert is_exio_mrio_source("EXIOBASE_396_PXP") is True
    assert is_exio_mrio_source("oecd_v2025") is False

    assert _resolve_version_tag(None) == "original_classification"
    assert _resolve_version_tag(" ") == "original_classification"
    assert _resolve_version_tag("demo") == "custom_classification_demo"

    assert (
        _get_saved_dir("oecd_v2025")
        == project_repo / "data_processed" / "mrio" / "oecd_v2025" / "original_classification"
    )
    assert _get_year_saved_dir("oecd_v2025", 2019).name == "ICIO2025_2019_calc"
    assert _get_year_saved_dir("exiobase_396_ixi", 2018).name == "IOT_2018_ixi_calc"
    assert (
        _get_year_saved_dir("exiobase_396_pxp", 2018, matrix_version="demo").parent.name
        == "custom_classification_demo"
    )

    assert _get_metadata_path("oecd_v2025").name == "processed_metadata.json"
    assert _get_mrio_calc_logs_dir(source_key="oecd_v2025") == (
        project_repo / "data_processed" / "mrio" / "oecd_v2025" / "original_classification" / "logs"
    )
    assert _get_mrio_calc_log_path("x.log", source_key="oecd_v2025") == (
        _get_mrio_calc_logs_dir(source_key="oecd_v2025") / "x.log"
    )
    assert (
        _get_mrio_raw_corrected_values_log_path(
            "oecd_v2025",
            matrix_version="custom version",
        ).name
        == "oecd_v2025_custom_version_raw_corrected_values_log.csv"
    )

    assert get_mrio_entry("exiobase_396_ixi").system == "ixi"
    assert get_mrio_entry("exiobase_396_pxp").system == "pxp"

    assert (
        _get_characterization_matrix_path(
            source_key="exiobase_396_ixi",
            lcia_method="pb_lcia",
        ).name
        == "pb_lcia.csv"
    )

    raw_dir = project_repo / "data_raw" / "mrio" / "oecd_v2025" / "full"
    assert _get_oecd_csv_path(raw_dir, 2019) == raw_dir / "ICIO2025_2019.csv"
    exio_dir = project_repo / "data_raw" / "mrio" / "exiobase_396" / "full_ixi"
    assert (
        _get_exio_archive_path(
            exio_dir,
            2018,
            system="ixi",
        )
        == exio_dir / "IOT_2018_ixi.zip"
    )

    assert _get_aggregation_dir("exiobase_396_ixi", kind="reg") == (
        project_repo / "data_raw" / "mrio" / "exiobase_3" / "aggregation"
    )
    assert _get_aggregation_dir("exiobase_396_pxp", kind="sec") == (
        project_repo / "data_raw" / "mrio" / "exiobase_3" / "aggregation" / "pxp"
    )
    assert _get_aggregation_dir("oecd_v2025", kind="sec") == (
        project_repo / "data_raw" / "mrio" / "oecd_v2025" / "aggregation"
    )
    assert (
        _get_agg_map_path(
            "oecd_v2025",
            kind="reg",
            agg_version="demo",
        ).name
        == "agg_reg_demo.csv"
    )
    assert (
        _get_agg_map_path(
            "oecd_v2025",
            kind="sec",
            agg_version="demo",
        ).name
        == "agg_sec_demo.csv"
    )


def test_read_agg_map_and_build_agg_vector_cover_valid_and_invalid_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        read_agg_map(tmp_path / "missing.csv")

    missing_cols = tmp_path / "missing_cols.csv"
    pd.DataFrame({"source": ["A"], "target": ["B"]}).to_csv(missing_cols, index=False)
    with pytest.raises(ValueError):
        read_agg_map(missing_cols)

    utf16_map = tmp_path / "utf16_aggregate.csv"
    pd.DataFrame(
        {
            "original_classification": ["R1", "R2"],
            "aggregated_mrio": ["G1", "G2"],
        }
    ).to_csv(utf16_map, index=False, encoding="utf-16")
    mapping = read_agg_map(utf16_map)
    assert mapping.to_dict(orient="list") == {
        "original_classification": ["R1", "R2"],
        "aggregated_mrio": ["G1", "G2"],
    }

    nan_original = tmp_path / "nan_original.csv"
    pd.DataFrame(
        {
            "original_classification": ["R1", None],
            "aggregated_mrio": ["G1", "G2"],
        }
    ).to_csv(nan_original, index=False)
    with pytest.raises(ValueError):
        read_agg_map(nan_original)

    blank_original = tmp_path / "blank_original.csv"
    pd.DataFrame(
        {
            "original_classification": ["R1", " "],
            "aggregated_mrio": ["G1", "G2"],
        }
    ).to_csv(blank_original, index=False)
    with pytest.raises(ValueError):
        read_agg_map(blank_original)

    nan_aggregated = tmp_path / "nan_aggregated.csv"
    pd.DataFrame(
        {
            "original_classification": ["R1", "R2"],
            "aggregated_mrio": ["G1", None],
        }
    ).to_csv(nan_aggregated, index=False)
    with pytest.raises(ValueError):
        read_agg_map(nan_aggregated)

    blank_aggregated = tmp_path / "blank_aggregated.csv"
    pd.DataFrame(
        {
            "original_classification": ["R1", "R2"],
            "aggregated_mrio": ["G1", " "],
        }
    ).to_csv(blank_aggregated, index=False)
    with pytest.raises(ValueError):
        read_agg_map(blank_aggregated)

    assert build_agg_vector(
        ["R1", "R2"],
        mapping,
        label_kind="region",
        csv_path=utf16_map,
    ) == ["G1", "G2"]
    with pytest.raises(ValueError):
        build_agg_vector(
            ["R1", "R2"],
            pd.DataFrame(
                {
                    "original_classification": ["R1", "R1"],
                    "aggregated_mrio": ["G1", "G2"],
                }
            ),
            label_kind="region",
            csv_path=utf16_map,
        )
    with pytest.raises(ValueError):
        build_agg_vector(
            ["R1", "R3"],
            mapping,
            label_kind="region",
            csv_path=utf16_map,
        )

    assert unique_in_order(["A", "B", "A", "C", "B"]) == ["A", "B", "C"]


def test_processed_mrio_metadata_contracts_fail_fast_on_invalid_shapes(project_repo: Path) -> None:
    assert _read_metadata("oecd_v2025", matrix_version=None) == {
        "source": "oecd_v2025",
        "version_tag": None,
        "aggregation": {},
        "labels": {},
        "years": {},
    }

    payload = {
        "version_tag": "original_classification",
        "aggregation": {"agg_reg": False},
        "labels": {"regions_used": ["R1"], "sectors_used": ["S1"]},
        "years": {},
    }
    _set_year_entry(
        payload,
        2019,
        {
            "core": ["A"],
            "extensions": {"pb_lcia": {"available": False}},
        },
    )
    _write_metadata("oecd_v2025", payload, matrix_version=None)
    roundtrip = _read_metadata("oecd_v2025", matrix_version=None)
    assert roundtrip["source"] == "oecd_v2025"
    assert "timestamp" in roundtrip
    assert _get_year_entry(roundtrip, 2018) is None
    assert read_processed_mrio_regions("oecd_v2025", matrix_version=None) == ["R1"]
    assert _get_year_entry(roundtrip, 2019) == {
        "core": ["A"],
        "extensions": {"pb_lcia": {"available": False}},
    }
    remove_payload = {"years": dict(roundtrip["years"])}
    _remove_year_entry(remove_payload, 2019)
    assert _get_year_entry(remove_payload, 2019) is None

    assert (
        _metadata_satisfies(
            roundtrip["years"]["2019"],
            saved_exists=True,
            required_core=["A"],
            required_extensions=["pb_lcia"],
            required_lcia_method="pb_lcia",
        )
        is True
    )
    assert (
        _metadata_satisfies(
            roundtrip["years"]["2019"],
            saved_exists=True,
            required_core=["A"],
            required_extensions=["pb_lcia"],
            required_lcia_method=None,
            required_lcia_methods=[],
        )
        is True
    )
    assert (
        _metadata_satisfies(
            roundtrip["years"]["2019"],
            saved_exists=False,
            required_core=["A"],
            required_extensions=[],
            required_lcia_method=None,
        )
        is False
    )
    assert (
        _metadata_satisfies(
            roundtrip["years"]["2019"],
            saved_exists=True,
            required_core=["G"],
            required_extensions=[],
            required_lcia_method=None,
        )
        is False
    )
    assert (
        _metadata_satisfies(
            roundtrip["years"]["2019"],
            saved_exists=True,
            required_core=["A"],
            required_extensions=[],
            required_lcia_method=None,
            required_lcia_methods=["pb_lcia", "gwp100_lcia"],
        )
        is False
    )
    assert (
        _metadata_satisfies(
            roundtrip["years"]["2019"],
            saved_exists=True,
            required_core=["A"],
            required_extensions=["gwp100_lcia"],
            required_lcia_method=None,
        )
        is False
    )
    assert (
        _metadata_satisfies(
            roundtrip["years"]["2019"],
            saved_exists=True,
            required_core=["A"],
            required_extensions=[],
            required_lcia_method="gwp100_lcia",
        )
        is False
    )


def test_validate_metadata_aggregation_covers_matching_and_mismatch_cases(tmp_path: Path) -> None:
    reg_path = tmp_path / "agg_reg_demo.csv"
    sec_path = tmp_path / "agg_sec_demo.csv"
    pd.DataFrame(
        {
            "original_classification": ["R1", "R2"],
            "aggregated_mrio": ["EU", "ROW"],
        }
    ).to_csv(reg_path, index=False)
    pd.DataFrame(
        {
            "original_classification": ["S1", "S2"],
            "aggregated_mrio": ["Energy", "Other"],
        }
    ).to_csv(sec_path, index=False)
    reg_df = read_agg_map(reg_path)
    sec_df = read_agg_map(sec_path)

    metadata = {
        "version_tag": "custom_classification_demo",
        "aggregation": {
            "agg_reg": True,
            "agg_sec": True,
            "agg_version": "demo",
            "agg_reg_file": str(reg_path),
            "agg_sec_file": str(sec_path),
            "agg_reg_weighted": False,
            "agg_sec_weighted": False,
            "agg_reg_fingerprint": None,
            "agg_sec_fingerprint": None,
        },
        "labels": {
            "regions_original": ["R1", "R2"],
            "regions_used": ["EU", "ROW"],
            "sectors_original": ["S1", "S2"],
            "sectors_used": ["Energy", "Other"],
        },
        "years": {"2019": {"core": ["A"], "extensions": {}}},
    }
    aggregation_payload = dict(metadata["aggregation"])
    aggregation_payload["agg_reg_fingerprint"] = agg_map_fingerprint(reg_df)
    aggregation_payload["agg_sec_fingerprint"] = agg_map_fingerprint(sec_df)
    metadata["aggregation"] = dict(aggregation_payload)

    validate_metadata_aggregation(
        metadata=metadata,
        version_tag="custom_classification_demo",
        aggregation_payload=aggregation_payload,
        agg_reg=True,
        agg_sec=True,
        agg_reg_df=reg_df,
        agg_sec_df=sec_df,
        agg_reg_path=reg_path,
        agg_sec_path=sec_path,
    )

    with pytest.raises(ValueError):
        validate_metadata_aggregation(
            metadata=metadata,
            version_tag="custom_classification_other",
            aggregation_payload=aggregation_payload,
            agg_reg=True,
            agg_sec=True,
            agg_reg_df=reg_df,
            agg_sec_df=sec_df,
            agg_reg_path=reg_path,
            agg_sec_path=sec_path,
        )
    with pytest.raises(ValueError):
        validate_metadata_aggregation(
            metadata=metadata,
            version_tag="custom_classification_demo",
            aggregation_payload={**aggregation_payload, "agg_version": "other"},
            agg_reg=True,
            agg_sec=True,
            agg_reg_df=reg_df,
            agg_sec_df=sec_df,
            agg_reg_path=reg_path,
            agg_sec_path=sec_path,
        )
    with pytest.raises(ValueError):
        validate_metadata_aggregation(
            metadata=metadata,
            version_tag="custom_classification_demo",
            aggregation_payload={**aggregation_payload, "agg_sec_fingerprint": "changed"},
            agg_reg=True,
            agg_sec=True,
            agg_reg_df=reg_df,
            agg_sec_df=sec_df,
            agg_reg_path=reg_path,
            agg_sec_path=sec_path,
        )
    with pytest.raises(ValueError):
        validate_metadata_aggregation(
            metadata={"years": {"2019": {"core": ["A"], "extensions": {}}}, "labels": {}},
            version_tag="custom_classification_demo",
            aggregation_payload=aggregation_payload,
            agg_reg=True,
            agg_sec=False,
            agg_reg_df=reg_df,
            agg_sec_df=None,
            agg_reg_path=reg_path,
            agg_sec_path=None,
        )
    with pytest.raises(ValueError):
        validate_metadata_aggregation(
            metadata={
                **metadata,
                "labels": {**metadata["labels"], "regions_used": ["EU"]},
            },
            version_tag="custom_classification_demo",
            aggregation_payload=aggregation_payload,
            agg_reg=True,
            agg_sec=False,
            agg_reg_df=reg_df,
            agg_sec_df=None,
            agg_reg_path=reg_path,
            agg_sec_path=None,
        )
    with pytest.raises(ValueError):
        validate_metadata_aggregation(
            metadata={
                **metadata,
                "labels": {**metadata["labels"], "sectors_used": ["Energy"]},
            },
            version_tag="custom_classification_demo",
            aggregation_payload=aggregation_payload,
            agg_reg=False,
            agg_sec=True,
            agg_reg_df=None,
            agg_sec_df=sec_df,
            agg_reg_path=None,
            agg_sec_path=sec_path,
        )


def test_process_report_summary_and_source_configs_cover_user_visible_paths() -> None:
    configs = build_source_configs()
    assert configs["exiobase_396_ixi"].requires_characterization is True
    assert configs["oecd_v2025"].requires_characterization is False

    report = ProcessReportMRIO(source="exiobase_396_ixi", requested=[2018, 2019, 2020])
    report.processed.extend([2018, 2019])
    report.skipped_already_saved.append(2020)
    report.saved_root = Path("saved_root")
    report.clipping_log_path = Path("clip.csv")
    report.errors[2021] = "boom"
    report.lcia_missing_by_year[2016] = {"empty_only": []}
    report.lcia_missing_by_year[2017] = {}
    report.lcia_missing_by_year[2018] = {
        "empty_method": [],
        "mid_method": ["a", "b"],
        "pb_lcia": ["satellite_accounts"],
        "gwp100_lcia": ["a", "b", "c", "d"],
    }

    assert report.missing() == []
    assert report._format_year_ranges([]) == "[]"
    assert report._format_year_ranges([2018, 2020, 2019]) == "2018-2020"
    assert report._format_year_ranges([2018, 2020, 2022]) == "2018, 2020, 2022"
    assert report._format_year_ranges_with_count([2018, 2019]) == "2018-2019 (2 year(s))"

    text = str(report)
    compact_text = " ".join(text.split())
    source_label = get_mrio_entry("exiobase_396_ixi").display_label
    assert text.strip()
    assert "exiobase_396_ixi" in text
    assert "saved_root" in text
    assert source_label in compact_text
    assert "clip.csv" in text
    assert "2021" in text

    minimal_report = ProcessReportMRIO(source="oecd_v2025", requested=[2019])
    minimal_text = str(minimal_report)
    assert minimal_text.strip()
