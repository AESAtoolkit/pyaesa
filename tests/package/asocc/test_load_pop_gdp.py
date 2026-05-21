from pathlib import Path

import pandas as pd
import pytest

from pyaesa.asocc.data import paths as paths_mod
from pyaesa.asocc.data import load_pop_gdp as mod
from pyaesa.process.mrios.utils.io.paths import _get_group_map_path, _get_year_saved_path


def test_processed_table_and_scalar_series_cover_basic_contracts(tmp_path: Path) -> None:
    csv_path = tmp_path / "processed_pop_gdp.csv"
    pd.DataFrame(
        {
            "variable": ["pop", "gdp"],
            "oecd_code": ["OECD_A", "OECD_A"],
            "2030": [1.5, 2.5],
        }
    ).to_csv(csv_path, index=False)

    loaded = mod._load_processed_table(csv_path)
    assert loaded.attrs["source_csv"] == str(csv_path)
    assert mod._source_csv_hint(loaded) == f" CSV: {csv_path}"
    assert mod._source_csv_hint(pd.DataFrame()) == ""

    filtered = mod._select_variable(loaded, "pop")
    assert filtered["variable"].tolist() == ["pop"]
    with pytest.raises(FileNotFoundError):
        mod._load_processed_table(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="variable"):
        mod._select_variable(pd.DataFrame({"2030": [1]}), "pop")

    numeric = mod._numeric_indexed_series(
        frame=pd.DataFrame({"code": ["A", "B"], "value": ["1.5", "2"]}),
        index_cols="code",
        value_col="value",
    )
    assert numeric.to_dict() == {"A": 1.5, "B": 2.0}

    indexed = mod._indexed_series(
        frame=pd.DataFrame(
            {
                "ssp_scenario": ["SSP1", "SSP2"],
                "code": ["A", "B"],
                "value": ["x", "y"],
            }
        ),
        index_cols=["ssp_scenario", "code"],
        value_col="value",
    )
    assert indexed.loc[("SSP1", "A")] == "x"

    duplicate_sample = mod._duplicated_label_sample(
        pd.Series([1, 2, 3, 4], index=["A", "A", "B", "B"])
    )
    assert duplicate_sample == ["A", "B"]


def test_apply_grouping_and_get_series_for_year_cover_grouping_and_validation(
    project_repo: Path,
) -> None:
    group_path = _get_group_map_path("oecd_v2025", kind="reg", group_version="demo")
    group_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "original_classification": ["OECD_A", "OECD_B", "OECD_C"],
            "grouped_mrio": ["EU", "EU", "ROW"],
        }
    ).to_csv(group_path, index=False)

    original = pd.Series([1.0, 2.0, 3.0], index=pd.Index(["OECD_A", "OECD_B", "OECD_C"]))
    unchanged = mod._apply_grouping_to_series(
        original,
        source_key="oecd_v2025",
        group_version=None,
    )
    assert unchanged.equals(original)

    grouped = mod._apply_grouping_to_series(
        original,
        source_key="oecd_v2025",
        group_version="demo",
    )
    assert grouped.to_dict() == {"EU": 3.0, "ROW": 3.0}

    unique_group_path = _get_group_map_path("oecd_v2025", kind="reg", group_version="unique")
    pd.DataFrame(
        {
            "original_classification": ["OECD_A", "OECD_B", "OECD_C"],
            "grouped_mrio": ["EU", "NAM", "ROW"],
        }
    ).to_csv(unique_group_path, index=False)
    uniquely_grouped = mod._apply_grouping_to_series(
        original,
        source_key="oecd_v2025",
        group_version="unique",
    )
    assert uniquely_grouped.to_dict() == {"EU": 1.0, "NAM": 2.0, "ROW": 3.0}

    with pytest.raises(ValueError):
        mod._apply_grouping_to_series(
            pd.Series([4.0], index=pd.Index(["OECD_Z"])),
            source_key="oecd_v2025",
            group_version="demo",
        )

    ssp_frame = pd.DataFrame(
        {
            "variable": ["pop", "pop", "pop", "pop"],
            "ssp_scenario": ["SSP1", "SSP1", "SSP2", "SSP2"],
            "oecd_code": ["OECD_A", "OECD_B", "OECD_A", "OECD_B"],
            "2030": [1.0, 2.0, 3.0, None],
        }
    )
    ssp_frame.attrs["source_csv"] = str(project_repo / "ssp.csv")

    ssp_scenario_series = mod._get_series_for_year(
        df=ssp_frame,
        variable="pop",
        year=2030,
        source_key="oecd_v2025",
        group_version="demo",
        ssp_scenario="SSP1",
    )
    assert ssp_scenario_series.to_dict() == {"EU": 3.0}

    multiindex_series = mod._get_series_for_year(
        df=ssp_frame,
        variable="pop",
        year=2030,
        source_key="oecd_v2025",
        group_version=None,
        ssp_scenario=None,
    )
    assert list(multiindex_series.index) == [
        ("SSP1", "OECD_A"),
        ("SSP1", "OECD_B"),
        ("SSP2", "OECD_A"),
    ]

    with pytest.raises(ValueError, match="2040"):
        mod._get_series_for_year(
            df=ssp_frame,
            variable="pop",
            year=2040,
            source_key="oecd_v2025",
            group_version=None,
            ssp_scenario="SSP1",
        )

    with pytest.raises(ValueError, match="missing_region"):
        mod._get_series_for_year(
            df=ssp_frame,
            variable="pop",
            year=2030,
            source_key="oecd_v2025",
            group_version=None,
            ssp_scenario="SSP1",
            region_col_override="missing_region",
        )

    duplicate_ssp_scenario_frame = pd.DataFrame(
        {
            "variable": ["pop", "pop"],
            "ssp_scenario": ["SSP1", "SSP1"],
            "oecd_code": ["OECD_A", "OECD_A"],
            "2030": [1.0, 2.0],
        }
    )
    duplicate_ssp_scenario_frame.attrs["source_csv"] = str(project_repo / "duplicate_scenario.csv")
    with pytest.raises(ValueError):
        mod._get_series_for_year(
            df=duplicate_ssp_scenario_frame,
            variable="pop",
            year=2030,
            source_key="oecd_v2025",
            group_version=None,
            ssp_scenario="SSP1",
        )

    duplicate_region_frame = pd.DataFrame(
        {
            "variable": ["pop", "pop"],
            "oecd_code": ["OECD_A", "OECD_A"],
            "2030": [1.0, 2.0],
        }
    )
    duplicate_region_frame.attrs["source_csv"] = str(project_repo / "duplicate_region.csv")
    with pytest.raises(ValueError):
        mod._get_series_for_year(
            df=duplicate_region_frame,
            variable="pop",
            year=2030,
            source_key="oecd_v2025",
            group_version=None,
        )


def test_get_pr_iso3_inputs_cover_success_identity_and_failure_branches(
    project_repo: Path,
) -> None:
    base = pd.DataFrame(
        {
            "variable": ["gdp", "gdp", "pop", "pop"],
            "ssp_scenario": ["SSP1", "SSP1", "SSP1", "SSP1"],
            "iso3_code": ["AAA", "BBB", "AAA", "BBB"],
            "oecd_code": ["OECD_A", "OECD_B", "OECD_A", "OECD_B"],
            "2030": [100.0, 200.0, 10.0, 20.0],
        }
    )
    base.attrs["source_csv"] = str(project_repo / "pr_inputs.csv")

    pop_iso, gdp_iso, iso_to_mrio = mod._get_pr_iso3_inputs(
        df=base,
        year=2030,
        source_key="oecd_v2025",
        gdp_variable="gdp",
        pop_variable="pop",
        ssp_scenario="SSP1",
    )
    assert pop_iso.to_dict() == {"AAA": 10.0, "BBB": 20.0}
    assert gdp_iso.to_dict() == {"AAA": 100.0, "BBB": 200.0}
    assert iso_to_mrio.to_dict() == {"AAA": "OECD_A", "BBB": "OECD_B"}

    _, _, identity_map = mod._get_pr_iso3_inputs(
        df=base,
        year=2030,
        source_key="iso3",
        gdp_variable="gdp",
        pop_variable="pop",
        ssp_scenario="SSP1",
    )
    assert identity_map.to_dict() == {"AAA": "AAA", "BBB": "BBB"}

    pop_iso_no_filter, gdp_iso_no_filter, iso_to_mrio_no_filter = mod._get_pr_iso3_inputs(
        df=base,
        year=2030,
        source_key="oecd_v2025",
        gdp_variable="gdp",
        pop_variable="pop",
        ssp_scenario=None,
    )
    assert pop_iso_no_filter.to_dict() == {"AAA": 10.0, "BBB": 20.0}
    assert gdp_iso_no_filter.to_dict() == {"AAA": 100.0, "BBB": 200.0}
    assert iso_to_mrio_no_filter.to_dict() == {"AAA": "OECD_A", "BBB": "OECD_B"}

    with pytest.raises(ValueError, match="iso3_code"):
        mod._get_pr_iso3_inputs(
            df=base.drop(columns=["iso3_code"]),
            year=2030,
            source_key="oecd_v2025",
            gdp_variable="gdp",
            pop_variable="pop",
            ssp_scenario="SSP1",
        )

    with pytest.raises(ValueError, match="missing_region"):
        mod._get_pr_iso3_inputs(
            df=base,
            year=2030,
            source_key="oecd_v2025",
            gdp_variable="gdp",
            pop_variable="pop",
            ssp_scenario="SSP1",
            region_col_override="missing_region",
        )

    with pytest.raises(ValueError, match="2040"):
        mod._get_pr_iso3_inputs(
            df=base,
            year=2040,
            source_key="oecd_v2025",
            gdp_variable="gdp",
            pop_variable="pop",
            ssp_scenario="SSP1",
        )

    duplicate_gdp = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "variable": ["gdp"],
                    "ssp_scenario": ["SSP1"],
                    "iso3_code": ["AAA"],
                    "oecd_code": ["OECD_A"],
                    "2030": [150.0],
                }
            ),
        ],
        ignore_index=True,
    )
    duplicate_gdp.attrs["source_csv"] = str(project_repo / "duplicate_gdp.csv")
    with pytest.raises(ValueError):
        mod._get_pr_iso3_inputs(
            df=duplicate_gdp,
            year=2030,
            source_key="oecd_v2025",
            gdp_variable="gdp",
            pop_variable="pop",
            ssp_scenario="SSP1",
        )

    duplicate_pop = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "variable": ["pop"],
                    "ssp_scenario": ["SSP1"],
                    "iso3_code": ["BBB"],
                    "oecd_code": ["OECD_B"],
                    "2030": [25.0],
                }
            ),
        ],
        ignore_index=True,
    )
    duplicate_pop.attrs["source_csv"] = str(project_repo / "duplicate_pop.csv")
    with pytest.raises(ValueError):
        mod._get_pr_iso3_inputs(
            df=duplicate_pop,
            year=2030,
            source_key="oecd_v2025",
            gdp_variable="gdp",
            pop_variable="pop",
            ssp_scenario="SSP1",
        )

    no_overlap = pd.DataFrame(
        {
            "variable": ["gdp", "pop"],
            "ssp_scenario": ["SSP1", "SSP1"],
            "iso3_code": ["AAA", "BBB"],
            "oecd_code": ["OECD_A", "OECD_B"],
            "2030": [100.0, 20.0],
        }
    )
    no_overlap.attrs["source_csv"] = str(project_repo / "no_overlap.csv")
    with pytest.raises(ValueError):
        mod._get_pr_iso3_inputs(
            df=no_overlap,
            year=2030,
            source_key="oecd_v2025",
            gdp_variable="gdp",
            pop_variable="pop",
            ssp_scenario="SSP1",
        )

    duplicate_mapping = pd.DataFrame(
        {
            "variable": ["gdp", "gdp", "pop"],
            "ssp_scenario": ["SSP1", "SSP1", "SSP1"],
            "iso3_code": ["AAA", "AAA", "AAA"],
            "oecd_code": ["OECD_A", "OECD_B", "OECD_A"],
            "2030": [100.0, None, 10.0],
        }
    )
    duplicate_mapping.attrs["source_csv"] = str(project_repo / "duplicate_mapping.csv")
    with pytest.raises(ValueError):
        mod._get_pr_iso3_inputs(
            df=duplicate_mapping,
            year=2030,
            source_key="oecd_v2025",
            gdp_variable="gdp",
            pop_variable="pop",
            ssp_scenario="SSP1",
        )

    missing_mapping = pd.DataFrame(
        {
            "variable": ["gdp", "pop"],
            "ssp_scenario": ["SSP1", "SSP1"],
            "iso3_code": ["AAA", "AAA"],
            "oecd_code": [pd.NA, pd.NA],
            "2030": [100.0, 10.0],
        }
    )
    missing_mapping.attrs["source_csv"] = str(project_repo / "missing_mapping.csv")
    with pytest.raises(ValueError):
        mod._get_pr_iso3_inputs(
            df=missing_mapping,
            year=2030,
            source_key="oecd_v2025",
            gdp_variable="gdp",
            pop_variable="pop",
            ssp_scenario="SSP1",
        )


def test_resolve_scenarios_covers_all_branches(project_repo: Path) -> None:
    wb_df = pd.DataFrame(columns=["2010", "2015"])
    ssp_df = pd.DataFrame({"ssp_scenario": ["SSP2", None, "SSP1"]})
    ssp_df.attrs["source_csv"] = str(project_repo / "ssp.csv")

    assert mod._resolve_ssp_scenarios(
        resolved_years=[2010],
        wb_df=wb_df,
        ssp_df=ssp_df,
        ssp_scenario="SSP2",
    ) == ["SSP2"]
    assert mod._resolve_ssp_scenarios(
        resolved_years=[2010],
        wb_df=wb_df,
        ssp_df=ssp_df,
        ssp_scenario=[" SSP2 ", "SSP1"],
    ) == ["SSP2", "SSP1"]
    assert mod._resolve_ssp_scenarios(
        resolved_years=[2030],
        wb_df=wb_df,
        ssp_df=ssp_df,
        ssp_scenario="SSP2",
    ) == ["SSP2"]

    with pytest.raises(ValueError):
        mod._resolve_ssp_scenarios(
            resolved_years=[2010],
            wb_df=wb_df,
            ssp_df=ssp_df,
            ssp_scenario=["SSP1", " SSP1 "],
        )

    assert mod._resolve_ssp_scenarios(
        resolved_years=[2010, 2030],
        wb_df=wb_df,
        ssp_df=ssp_df,
        ssp_scenario=None,
    ) == ["SSP1", "SSP2"]

    with pytest.raises(ValueError, match="ssp_scenario"):
        mod._resolve_ssp_scenarios(
            resolved_years=[2030],
            wb_df=wb_df,
            ssp_df=pd.DataFrame({"2030": [1.0]}),
            ssp_scenario=None,
        )

    assert mod._resolve_ssp_scenarios(
        resolved_years=[2010, 2015],
        wb_df=wb_df,
        ssp_df=ssp_df,
        ssp_scenario=None,
    ) == [None]

    with pytest.raises(ValueError, match="SSP9"):
        mod._resolve_ssp_scenarios(
            resolved_years=[2030],
            wb_df=wb_df,
            ssp_df=ssp_df,
            ssp_scenario="SSP9",
        )


def test_processed_data_paths_cover_dataset_normalization_and_mrio_passthrough(
    project_repo: Path,
) -> None:
    wb_path = paths_mod._get_processed_pop_gdp_table_path(dataset=" WB ")
    ssp_path = paths_mod._get_processed_pop_gdp_table_path(dataset="ssp")
    assert wb_path == project_repo / "data_processed" / "pop_gdp" / "wb_processed.csv"
    assert ssp_path == project_repo / "data_processed" / "pop_gdp" / "ssp_processed.csv"

    assert paths_mod._get_mrio_year_dir(
        source="oecd_v2025",
        year=2019,
        group_version="demo",
    ) == _get_year_saved_path("oecd_v2025", 2019, matrix_version="demo")
