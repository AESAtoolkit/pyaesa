from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa.process.mrios.utils.raw_corrections.runtime import AppliedCorrectionSummary
from pyaesa.process.mrios.utils.parsers.exio_parser import ExioCharacterizationOptions
from pyaesa.process.mrios.utils.pipeline.contracts import SourceConfig
from pyaesa.process.mrios.utils.pipeline.year_orchestrator import parse_and_calc_year
from tests.package.helpers.data_processing_dummy import (
    build_dummy_iosystem,
    write_characterization_matrix,
)


def test_parse_and_calc_year_validates_region_sector_inputs() -> None:
    cfg = SourceConfig(
        requires_characterization=False, required_core=("A",), required_extensions=()
    )

    class MissingRegions:
        def get_regions(self):
            return None

        def get_sectors(self):
            return ["S1"]

    with pytest.raises(ValueError):
        parse_and_calc_year(
            source="oecd_v2025",
            cfg=cfg,
            full_dir=Path("."),
            year=2019,
            char_jobs=None,
            group_reg=False,
            group_sec=False,
            group_reg_df=None,
            group_sec_df=None,
            group_reg_path=None,
            group_sec_path=None,
            parse_oecd_func=lambda full_dir, year: MissingRegions(),
        )

    class NonIterableRegions:
        def get_regions(self):
            return 1

        def get_sectors(self):
            return ["S1"]

    with pytest.raises(ValueError):
        parse_and_calc_year(
            source="oecd_v2025",
            cfg=cfg,
            full_dir=Path("."),
            year=2019,
            char_jobs=None,
            group_reg=False,
            group_sec=False,
            group_reg_df=None,
            group_sec_df=None,
            group_reg_path=None,
            group_sec_path=None,
            parse_oecd_func=lambda full_dir, year: NonIterableRegions(),
        )

    class NoneSector:
        def get_regions(self):
            return ["R1"]

        def get_sectors(self):
            return ["S1", None]

    with pytest.raises(ValueError):
        parse_and_calc_year(
            source="oecd_v2025",
            cfg=cfg,
            full_dir=Path("."),
            year=2019,
            char_jobs=None,
            group_reg=False,
            group_sec=False,
            group_reg_df=None,
            group_sec_df=None,
            group_reg_path=None,
            group_sec_path=None,
            parse_oecd_func=lambda full_dir, year: NoneSector(),
        )


def test_parse_and_calc_year_covers_grouping_cache_and_oecd_calc_all() -> None:
    cfg = SourceConfig(
        requires_characterization=False, required_core=("A",), required_extensions=()
    )
    reg_df = pd.DataFrame({"original_classification": ["R1", "R2"], "grouped_mrio": ["EU", "ROW"]})
    sec_df = pd.DataFrame(
        {"original_classification": ["S1", "S2"], "grouped_mrio": ["Energy", "Other"]}
    )
    reg_cache = {("R1", "R2"): ["EU", "ROW"]}
    sec_cache = {("S1", "S2"): ["Energy", "Other"]}

    iosys, applied, missing, regions_original, sectors_original, regions_used, sectors_used = (
        parse_and_calc_year(
            source="oecd_v2025",
            cfg=cfg,
            full_dir=Path("."),
            year=2019,
            char_jobs=None,
            group_reg=True,
            group_sec=True,
            group_reg_df=reg_df,
            group_sec_df=sec_df,
            group_reg_path=Path("group_reg.csv"),
            group_sec_path=Path("group_sec.csv"),
            reg_vec_cache=reg_cache,
            sec_vec_cache=sec_cache,
            pymrio_calc_all=True,
            parse_oecd_func=lambda full_dir, year: build_dummy_iosystem(),
        )
    )
    assert applied is None
    assert missing is None
    assert regions_original == ["R1", "R2"]
    assert sectors_original == ["S1", "S2"]
    assert regions_used == ["EU", "ROW"]
    assert sectors_used == ["Energy", "Other"]
    assert iosys.A is not None
    assert iosys.L is not None
    assert iosys.G is not None


def test_parse_and_calc_year_attaches_raw_corrected_values_summary() -> None:
    cfg = SourceConfig(
        requires_characterization=True,
        required_core=("A",),
        required_extensions=(),
    )
    iosys, *_ = parse_and_calc_year(
        source="exiobase_3102_ixi",
        cfg=cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=None,
        group_reg=False,
        group_sec=False,
        group_reg_df=None,
        group_sec_df=None,
        group_reg_path=None,
        group_sec_path=None,
        parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
        apply_raw_corrections_func=lambda **kwargs: AppliedCorrectionSummary(
            source="exiobase_3102_ixi",
            year=2019,
            row_count=4,
            log_path=None,
        ),
    )
    summary = getattr(iosys, "_raw_corrected_values_summary")
    assert isinstance(summary, AppliedCorrectionSummary)
    assert summary.row_count == 4


def test_parse_and_calc_year_populates_caches_and_covers_exio_non_lcia_paths() -> None:
    oecd_cfg = SourceConfig(
        requires_characterization=False,
        required_core=("A",),
        required_extensions=(),
    )
    reg_df = pd.DataFrame({"original_classification": ["R1", "R2"], "grouped_mrio": ["EU", "ROW"]})
    sec_df = pd.DataFrame(
        {"original_classification": ["S1", "S2"], "grouped_mrio": ["Energy", "Other"]}
    )
    reg_cache: dict[tuple[str, ...], list[str]] = {}
    sec_cache: dict[tuple[str, ...], list[str]] = {}

    parse_and_calc_year(
        source="oecd_v2025",
        cfg=oecd_cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=None,
        group_reg=True,
        group_sec=True,
        group_reg_df=reg_df,
        group_sec_df=sec_df,
        group_reg_path=Path("group_reg.csv"),
        group_sec_path=Path("group_sec.csv"),
        reg_vec_cache=reg_cache,
        sec_vec_cache=sec_cache,
        parse_oecd_func=lambda full_dir, year: build_dummy_iosystem(),
    )
    assert reg_cache == {("R1", "R2"): ["EU", "ROW"]}
    assert sec_cache == {("S1", "S2"): ["Energy", "Other"]}

    exio_cfg = SourceConfig(
        requires_characterization=True, required_core=("A",), required_extensions=()
    )
    exio_calc_all, applied_calc_all, missing_calc_all, *_ = parse_and_calc_year(
        source="exiobase_396_ixi",
        cfg=exio_cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=None,
        group_reg=False,
        group_sec=False,
        group_reg_df=None,
        group_sec_df=None,
        group_reg_path=None,
        group_sec_path=None,
        pymrio_calc_all=True,
        parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
    )
    assert applied_calc_all is None
    assert missing_calc_all is None
    assert hasattr(exio_calc_all, "factor_inputs")

    exio_minimal, applied_minimal, missing_minimal, *_ = parse_and_calc_year(
        source="exiobase_396_ixi",
        cfg=exio_cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=None,
        group_reg=False,
        group_sec=False,
        group_reg_df=None,
        group_sec_df=None,
        group_reg_path=None,
        group_sec_path=None,
        pymrio_calc_all=False,
        parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
    )
    assert applied_minimal is None
    assert missing_minimal is None
    assert exio_minimal.G is not None

    exio_grouped_calc_all, applied_grouped, missing_grouped, *_ = parse_and_calc_year(
        source="exiobase_396_ixi",
        cfg=exio_cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=None,
        group_reg=True,
        group_sec=True,
        group_reg_df=reg_df,
        group_sec_df=sec_df,
        group_reg_path=Path("group_reg.csv"),
        group_sec_path=Path("group_sec.csv"),
        pymrio_calc_all=True,
        parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
    )
    assert applied_grouped is None
    assert missing_grouped is None
    assert exio_grouped_calc_all.G is not None
    assert cast(Any, exio_grouped_calc_all).factor_inputs.S is not None

    parse_and_calc_year(
        source="oecd_v2025",
        cfg=oecd_cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=None,
        group_reg=True,
        group_sec=True,
        group_reg_df=reg_df,
        group_sec_df=sec_df,
        group_reg_path=Path("group_reg.csv"),
        group_sec_path=Path("group_sec.csv"),
        reg_vec_cache=None,
        sec_vec_cache=None,
        parse_oecd_func=lambda full_dir, year: build_dummy_iosystem(),
    )


def test_parse_and_calc_year_covers_exio_characterization_paths(project_repo: Path) -> None:
    cfg = SourceConfig(requires_characterization=True, required_core=("A",), required_extensions=())
    write_characterization_matrix(
        project_repo,
        source_key="exiobase_396_ixi",
        method_name="pb_lcia",
    )

    missing_job = {
        "pb_lcia": ExioCharacterizationOptions(
            lcia_method="pb_lcia",
            matrix_path=Path("pb_lcia.csv"),
            retain_instances=("factor_inputs",),
            requested_extensions=["missing"],
            char_matrix=pd.DataFrame(
                {
                    "extension": ["missing"],
                    "stressor": ["co2"],
                    "factor": [2.0],
                    "impact": ["climate_child"],
                    "impact_unit": ["kg CO2-eq"],
                }
            ),
        )
    }
    iosys, applied, missing, *_ = parse_and_calc_year(
        source="exiobase_396_ixi",
        cfg=cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=missing_job,
        group_reg=False,
        group_sec=False,
        group_reg_df=None,
        group_sec_df=None,
        group_reg_path=None,
        group_sec_path=None,
        parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
    )
    assert applied == []
    assert missing == {"pb_lcia": ["missing"]}
    assert cast(Any, iosys).factor_inputs is not None

    valid_job = {
        "pb_lcia": ExioCharacterizationOptions(
            lcia_method="pb_lcia",
            matrix_path=Path("pb_lcia.csv"),
            retain_instances=("factor_inputs",),
            requested_extensions=["satellite_accounts"],
            char_matrix=pd.DataFrame(
                {
                    "extension": ["satellite_accounts"],
                    "stressor": ["co2"],
                    "factor": [2.0],
                    "impact": ["climate_child"],
                    "impact_unit": ["kg CO2-eq"],
                }
            ),
        )
    }
    grouped_iosys, grouped_applied, grouped_missing, *_ = parse_and_calc_year(
        source="exiobase_396_ixi",
        cfg=cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=valid_job,
        group_reg=True,
        group_sec=True,
        group_reg_df=pd.DataFrame(
            {
                "original_classification": ["R1", "R2"],
                "grouped_mrio": ["EU", "ROW"],
            }
        ),
        group_sec_df=pd.DataFrame(
            {
                "original_classification": ["S1", "S2"],
                "grouped_mrio": ["Energy", "Other"],
            }
        ),
        group_reg_path=Path("group_reg.csv"),
        group_sec_path=Path("group_sec.csv"),
        pymrio_calc_all=True,
        parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
    )
    assert grouped_applied == ["pb_lcia"]
    assert grouped_missing == {}
    assert grouped_iosys.G is not None
    assert cast(Any, grouped_iosys).pb_lcia.S is not None

    char_matrix = pd.read_csv(
        project_repo
        / "data_raw"
        / "mrio"
        / "exiobase_3"
        / "lcia"
        / "characterization_factors_matrices"
        / "pb_lcia.csv"
    )
    char_jobs = {
        "pb_lcia": ExioCharacterizationOptions(
            lcia_method="pb_lcia",
            matrix_path=Path("pb_lcia.csv"),
            retain_instances=("factor_inputs",),
            requested_extensions=["satellite_accounts"],
            char_matrix=char_matrix,
        )
    }
    iosys_minimal, applied_minimal, missing_minimal, *_ = parse_and_calc_year(
        source="exiobase_396_ixi",
        cfg=cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=char_jobs,
        group_reg=False,
        group_sec=False,
        group_reg_df=None,
        group_sec_df=None,
        group_reg_path=None,
        group_sec_path=None,
        pymrio_calc_all=False,
        parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
    )
    assert applied_minimal == ["pb_lcia"]
    assert missing_minimal == {}
    assert cast(Any, iosys_minimal).pb_lcia.D_pba_reg is not None
    assert not hasattr(iosys_minimal, "satellite_accounts")

    iosys_calc_all, applied_calc_all, _, *_ = parse_and_calc_year(
        source="exiobase_396_ixi",
        cfg=cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=char_jobs,
        group_reg=False,
        group_sec=False,
        group_reg_df=None,
        group_sec_df=None,
        group_reg_path=None,
        group_sec_path=None,
        pymrio_calc_all=True,
        parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
    )
    assert applied_calc_all == ["pb_lcia"]
    assert hasattr(iosys_calc_all, "factor_inputs")
    assert hasattr(iosys_calc_all, "pb_lcia")

    duplicate_retain_jobs = {
        "pb_lcia": ExioCharacterizationOptions(
            lcia_method="pb_lcia",
            matrix_path=Path("pb_lcia.csv"),
            retain_instances=("factor_inputs", "pb_lcia"),
            requested_extensions=["satellite_accounts"],
            char_matrix=char_matrix,
        ),
        "gwp100_lcia": ExioCharacterizationOptions(
            lcia_method="gwp100_lcia",
            matrix_path=Path("gwp100_lcia.csv"),
            retain_instances=("factor_inputs", "missing_ext"),
            requested_extensions=["missing"],
            char_matrix=pd.DataFrame(
                {
                    "extension": ["missing"],
                    "stressor": ["co2"],
                    "factor": [1.0],
                    "impact": ["impact"],
                    "impact_unit": ["kg"],
                }
            ),
        ),
    }
    iosys_duplicate, applied_duplicate, missing_duplicate, *_ = parse_and_calc_year(
        source="exiobase_396_ixi",
        cfg=cfg,
        full_dir=Path("."),
        year=2019,
        char_jobs=duplicate_retain_jobs,
        group_reg=False,
        group_sec=False,
        group_reg_df=None,
        group_sec_df=None,
        group_reg_path=None,
        group_sec_path=None,
        pymrio_calc_all=True,
        parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
    )
    assert applied_duplicate == ["pb_lcia"]
    assert missing_duplicate == {"gwp100_lcia": ["missing"]}
    assert hasattr(iosys_duplicate, "pb_lcia")

    no_retention_job = {
        "pb_lcia": ExioCharacterizationOptions(
            lcia_method="pb_lcia",
            matrix_path=Path("pb_lcia.csv"),
            retain_instances=(),
            requested_extensions=["missing"],
            char_matrix=pd.DataFrame(
                {
                    "extension": ["missing"],
                    "stressor": ["co2"],
                    "factor": [2.0],
                    "impact": ["climate_child"],
                    "impact_unit": ["kg CO2-eq"],
                }
            ),
        )
    }
    iosys_missing_calc_all, applied_missing_calc_all, missing_missing_calc_all, *_ = (
        parse_and_calc_year(
            source="exiobase_396_ixi",
            cfg=cfg,
            full_dir=Path("."),
            year=2019,
            char_jobs=no_retention_job,
            group_reg=False,
            group_sec=False,
            group_reg_df=None,
            group_sec_df=None,
            group_reg_path=None,
            group_sec_path=None,
            pymrio_calc_all=True,
            parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
        )
    )
    assert applied_missing_calc_all == []
    assert missing_missing_calc_all == {"pb_lcia": ["missing"]}
    assert iosys_missing_calc_all.G is not None

    iosys_missing_minimal, applied_missing_minimal, missing_missing_minimal, *_ = (
        parse_and_calc_year(
            source="exiobase_396_ixi",
            cfg=cfg,
            full_dir=Path("."),
            year=2019,
            char_jobs=no_retention_job,
            group_reg=False,
            group_sec=False,
            group_reg_df=None,
            group_sec_df=None,
            group_reg_path=None,
            group_sec_path=None,
            pymrio_calc_all=False,
            parse_exio_func=lambda full_dir, year, system: build_dummy_iosystem(),
        )
    )
    assert applied_missing_minimal == []
    assert missing_missing_minimal == {"pb_lcia": ["missing"]}
    assert iosys_missing_minimal.G is not None
