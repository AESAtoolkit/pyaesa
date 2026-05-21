"""Reusable dummy repository helpers for AR6 collection and processing tests."""

from dataclasses import dataclass
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from pyaesa import set_workspace
from pyaesa.workspace_initialisation.workspace import clear_default_repo_root, get_default_repo_root
from tests.package.helpers.ar6_imports import (
    collection_config,
    collection_explorer,
    collection_historical,
    collection_paths,
)

DEFAULT_DATABASE = collection_config.DEFAULT_DATABASE
get_citation_txt_path = collection_paths.get_citation_txt_path
get_explorer_csv_path = collection_paths.get_explorer_csv_path
get_logs_dir = collection_paths.get_logs_dir
get_metadata_path = collection_paths.get_metadata_path
get_raw_dir = collection_paths.get_raw_dir
historical_sources = collection_historical
drop_non_persisted_columns = collection_explorer.drop_non_persisted_columns
write_explorer_csv = collection_explorer.write_explorer_csv

YEAR_START = 2010
YEAR_END = 2100
HISTORICAL_YEAR_START = 1850
HISTORICAL_YEAR_END = 2023
OVERLAY_YEAR_START = 1970
OVERLAY_YEAR_END = 2023


@dataclass
class AR6DummyRepo:
    """One deterministic AR6 raw data repository scaffold for package tests."""

    repo_root: Path
    raw_dir: Path
    raw_logs_dir: Path
    explorer_csv_path: Path
    citation_txt_path: Path
    metadata_path: Path
    primap_final_path: Path
    primap_no_rounding_path: Path
    gcp_path: Path
    overlay_path: Path
    explorer_long_df: pd.DataFrame
    explorer_meta_df: pd.DataFrame

    def historical_sources_result(self) -> dict[str, object]:
        """Return one deterministic historical sources report payload."""
        citation_path = self.citation_txt_path
        citation_path.write_text("historical citations", encoding="utf-8")
        return {
            "citation_txt_file": str(citation_path),
            "primap": {
                "final_file": {
                    "path": str(self.primap_final_path),
                    "filename": self.primap_final_path.name,
                },
                "final_no_rounding_file": {
                    "path": str(self.primap_no_rounding_path),
                    "filename": self.primap_no_rounding_path.name,
                },
            },
            "gcp": {
                "file": {
                    "path": str(self.gcp_path),
                    "filename": self.gcp_path.name,
                }
            },
            "ar6_historical_figure_reference": {
                "file": {
                    "path": str(self.overlay_path),
                    "filename": self.overlay_path.name,
                }
            },
            "used_local_primap": False,
            "used_local_gcp": False,
            "used_local_ar6_historical_figure_reference": False,
        }


def _year_values(
    *,
    start_year: int,
    end_year: int,
    base_value: float,
    yearly_change: float,
    tail_end_year: int = YEAR_END,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for year in range(start_year, end_year + 1):
        values[year] = base_value + yearly_change * (year - start_year)
    for year in range(end_year + 1, tail_end_year + 1):
        values[year] = np.nan
    return values


def _full_year_values(*, base_value: float, yearly_change: float) -> dict[int, float]:
    return _year_values(
        start_year=YEAR_START,
        end_year=YEAR_END,
        base_value=base_value,
        yearly_change=yearly_change,
        tail_end_year=YEAR_END,
    )


def _base_meta_row(
    *,
    model: str,
    scenario: str,
    category: str,
    ssp_family: int,
    warming_2100: float,
) -> dict[str, object]:
    return {
        "model": model,
        "scenario": scenario,
        "Ssp_family": ssp_family,
        "Vetting_historical": "Pass",
        "Vetting_future": "Pass",
        "Time horizon": "long",
        "Category": category,
        "Category_name": f"{category} scenario group",
        "Category_subset": f"{category}-subset",
        "Category_definition": f"{category} definition",
        "Cumulative net CO2 (2020-2100, Gt CO2) (Harm-Infilled)": 100.0,
        "Cumulative net CO2 (2020 to netzero, Gt CO2) (Harm-Infilled)": 60.0,
        "Cumulative net-negative CO2 (post net-zero, Gt CO2) (Harm-Infilled)": 5.0,
        "Peak Emissions|CO2": 40.0,
        "Peak Emissions|GHGs": 55.0,
        "Exceedance Probability 1.5C (FaIRv1.6.2)": 0.2,
        "Exceedance Probability 1.5C (MAGICCv7.5.3)": 0.25,
        "Exceedance Probability 2.0C (FaIRv1.6.2)": 0.1,
        "Exceedance Probability 2.0C (MAGICCv7.5.3)": 0.15,
        "CO2 emissions 2030 Gt CO2/yr": 20.0,
        "CO2 emissions 2050 Gt CO2/yr": 10.0,
        "CO2 emissions 2100 Gt CO2/yr": -5.0,
        "GHG emissions 2030 Gt CO2-equiv/yr (Harmonized-Infilled)": 45.0,
        "GHG emissions 2050 Gt CO2-equiv/yr (Harmonized-Infilled)": 25.0,
        "GHG emissions 2100 Gt CO2-equiv/yr (Harmonized-Infilled)": 5.0,
        "Median warming in 2100 (FaIRv1.6.2)": warming_2100 + 0.1,
        "Median warming in 2100 (MAGICCv7.5.3)": warming_2100,
        "Policy_category": "P1",
        "Policy_category_name": "Policy 1",
        "Literature Reference (if applicable)": "Reference text",
    }


def _variable_values(
    *,
    variable: str,
    negative_pathway: bool,
    truncated_end_year: int | None,
) -> dict[int, float]:
    end_year = YEAR_END if truncated_end_year is None else truncated_end_year
    co2_yearly_change = -0.55 if negative_pathway else -0.40
    if truncated_end_year is not None:
        co2_yearly_change = -0.50 if not negative_pathway else -0.80
    if variable == collection_config.RAW_KYOTO_WITH_AFOLU:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=60.0,
            yearly_change=-0.55 if negative_pathway else -0.35,
        )
    if variable == collection_config.RAW_CO2_WITH_AFOLU:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=45.0,
            yearly_change=co2_yearly_change,
        )
    if variable == collection_config.RAW_CO2_AFOLU:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=5.0,
            yearly_change=-0.02,
        )
    if variable == collection_config.RAW_CH4_AFOLU:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=0.10,
            yearly_change=0.0,
        )
    if variable == collection_config.RAW_N2O_AFOLU:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=10.0,
            yearly_change=0.0,
        )
    if variable == collection_config.RAW_CO2_AFOLU_LAND:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=5.0,
            yearly_change=-0.02,
        )
    if variable == collection_config.RAW_CO2_OTHER:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=1.0,
            yearly_change=0.0,
        )
    if variable == collection_config.RAW_CO2_WASTE:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=2.0,
            yearly_change=0.0,
        )
    if variable == collection_config.RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=37.0,
            yearly_change=co2_yearly_change + 0.02,
        )
    if variable == collection_config.RAW_CO2_ENERGY:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=30.0,
            yearly_change=co2_yearly_change + 0.01,
        )
    if variable == collection_config.RAW_CO2_INDUSTRIAL_PROCESSES:
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=7.0,
            yearly_change=0.01,
        )
    if variable in collection_config.RAW_SEQUESTRATION_COMPONENTS:
        sequestration_values = {
            "Carbon Sequestration|CCS": (1.0, 0.02),
            "Carbon Sequestration|Direct Air Capture": (0.4, 0.01),
            "Carbon Sequestration|Enhanced Weathering": (0.2, 0.005),
            "Carbon Sequestration|Feedstocks": (0.1, 0.002),
            "Carbon Sequestration|Land Use": (0.8, 0.0),
            "Carbon Sequestration|Other": (0.3, 0.003),
        }
        base_value, yearly_change = sequestration_values[variable]
        return _year_values(
            start_year=YEAR_START,
            end_year=end_year,
            base_value=base_value,
            yearly_change=yearly_change,
        )
    raise ValueError(f"Unsupported AR6 test variable '{variable}'.")


def _append_scenario_rows(
    *,
    long_rows: list[dict[str, object]],
    meta_rows: list[dict[str, object]],
    model: str,
    scenario: str,
    category: str,
    ssp_family: int,
    warming_2100: float,
    negative_pathway: bool = False,
    missing_co2_afolu: bool = False,
    negative_sequestration: bool = False,
    truncated_end_year: int | None = None,
) -> None:
    meta_rows.append(
        _base_meta_row(
            model=model,
            scenario=scenario,
            category=category,
            ssp_family=ssp_family,
            warming_2100=warming_2100,
        )
    )
    variables = list(collection_config.RAW_VARIABLES_RELEVANT)
    if missing_co2_afolu:
        variables.remove(collection_config.RAW_CO2_AFOLU)
    for variable in variables:
        if variable == collection_config.RAW_KYOTO_WITH_AFOLU:
            unit = "MtCO2eq/yr"
        elif variable == collection_config.RAW_CH4_AFOLU:
            unit = "MtCH4/yr"
        elif variable == collection_config.RAW_N2O_AFOLU:
            unit = "ktN2O/yr"
        else:
            unit = "MtCO2/yr"
        values = _variable_values(
            variable=variable,
            negative_pathway=negative_pathway,
            truncated_end_year=truncated_end_year,
        )
        if negative_sequestration and variable == collection_config.RAW_SEQUESTRATION_COMPONENTS[0]:
            values = {year: -0.1 for year in values}
        if missing_co2_afolu and variable == collection_config.RAW_CO2_WITH_AFOLU:
            component_values = [
                _variable_values(
                    variable=component_variable,
                    negative_pathway=negative_pathway,
                    truncated_end_year=truncated_end_year,
                )
                for component_variable in [
                    collection_config.RAW_CO2_OTHER,
                    collection_config.RAW_CO2_WASTE,
                    collection_config.RAW_CO2_ENERGY_AND_INDUSTRIAL_PROCESSES,
                ]
            ]
            values = {
                year: sum(component[year] for component in component_values)
                for year in component_values[0]
            }
        for year, value in values.items():
            long_rows.append(
                {
                    "model": model,
                    "scenario": scenario,
                    "variable": variable,
                    "unit": unit,
                    "region": "World",
                    "year": year,
                    "value": value,
                }
            )


def build_ar6_explorer_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one deterministic AR6 explorer long table plus metadata table."""
    long_rows: list[dict[str, object]] = []
    meta_rows: list[dict[str, object]] = []
    _append_scenario_rows(
        long_rows=long_rows,
        meta_rows=meta_rows,
        model="M1",
        scenario="S1",
        category="C1",
        ssp_family=1,
        warming_2100=1.50,
    )
    _append_scenario_rows(
        long_rows=long_rows,
        meta_rows=meta_rows,
        model="M2",
        scenario="S2",
        category="C2",
        ssp_family=2,
        warming_2100=1.70,
    )
    _append_scenario_rows(
        long_rows=long_rows,
        meta_rows=meta_rows,
        model="M3",
        scenario="S3",
        category="C3",
        ssp_family=3,
        warming_2100=1.90,
        truncated_end_year=2080,
    )
    _append_scenario_rows(
        long_rows=long_rows,
        meta_rows=meta_rows,
        model="M4",
        scenario="S4",
        category="C4",
        ssp_family=4,
        warming_2100=2.10,
        missing_co2_afolu=True,
    )
    _append_scenario_rows(
        long_rows=long_rows,
        meta_rows=meta_rows,
        model="M5",
        scenario="S5",
        category="C1",
        ssp_family=1,
        warming_2100=1.60,
        truncated_end_year=2050,
    )
    _append_scenario_rows(
        long_rows=long_rows,
        meta_rows=meta_rows,
        model="M6",
        scenario="S6",
        category="C2",
        ssp_family=2,
        warming_2100=1.80,
        negative_sequestration=True,
    )
    long_df = pd.DataFrame(long_rows)
    meta_df = (
        pd.DataFrame(meta_rows)
        .drop_duplicates(subset=["model", "scenario"], keep="first")
        .set_index(["model", "scenario"])
    )
    return long_df, drop_non_persisted_columns(meta_df)


def _write_dummy_primap_csv(csv_path: Path, *, no_rounding: bool) -> None:
    records: list[dict[str, object]] = []
    years = range(HISTORICAL_YEAR_START, HISTORICAL_YEAR_END + 1)
    for entity in ["KYOTOGHG (AR6GWP100)", "CO2"]:
        for category, base_value in [("M.0.EL", 5000.0), ("M.LULUCF", 400.0), ("M.AG", 200.0)]:
            row: dict[str, object] = {
                "source": "source",
                "scenario (PRIMAP-hist)": "HISTTP",
                "provenance": "provenance",
                "model": "model",
                "area (ISO3)": "EARTH",
                "category (IPCC2006_PRIMAP)": category,
                "entity": entity,
            }
            for year in years:
                value = base_value + (year - HISTORICAL_YEAR_START) * (
                    1.2 if entity == "CO2" else 1.5
                )
                if no_rounding and category == "M.LULUCF":
                    value += 0.25
                row[str(year)] = value
            records.append(row)
    pd.DataFrame(records).to_csv(csv_path, index=False)


def _write_dummy_gcp_workbook(xlsx_path: Path) -> None:
    rows: list[dict[str, object]] = [{"col0": np.nan, "col1": np.nan} for _ in range(9)]
    rows.append({"col0": np.nan, "col1": "Bunkers"})
    for year in range(HISTORICAL_YEAR_START, HISTORICAL_YEAR_END + 1):
        rows.append(
            {
                "col0": year,
                "col1": 0.05 + 0.001 * (year - HISTORICAL_YEAR_START),
            }
        )
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        pd.DataFrame(rows).to_excel(
            writer,
            sheet_name="Territorial Emissions",
            index=False,
        )


def _write_dummy_overlay_csv(csv_path: Path) -> None:
    rows: list[dict[str, object]] = []
    years = [str(year) for year in range(OVERLAY_YEAR_START, OVERLAY_YEAR_END + 1)]
    edgar_specs: list[tuple[str, str, float]] = [
        ("Emissions|Kyoto Gases (AR6-GWP100)", "Gt CO2-equiv/yr", 25.0),
        ("Emissions|Kyoto Gases (AR6-GWP100)|Lower", "Gt CO2-equiv/yr", 20.0),
        ("Emissions|Kyoto Gases (AR6-GWP100)|Upper", "Gt CO2-equiv/yr", 30.0),
        ("Emissions|CO2", "Gt CO2/yr", 18.0),
        ("Emissions|CO2|Lower", "Gt CO2/yr", 15.0),
        ("Emissions|CO2|Upper", "Gt CO2/yr", 21.0),
    ]
    rcmip_specs: list[tuple[str, str, float]] = [
        ("Emissions|CO2", "Gt CO2/yr", 17.0),
    ]
    for model, scenario, region, specs in [
        ("EDGAR", "historical", "World", edgar_specs),
        ("RCMIP", "historical", "World", rcmip_specs),
    ]:
        for variable, unit, base_value in specs:
            row: dict[str, object] = {
                "Model": model,
                "Scenario": scenario,
                "Region": region,
                "Variable": variable,
                "Unit": unit,
            }
            for index, year in enumerate(years):
                row[year] = base_value + 0.2 * index
            rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def build_ar6_dummy_repo(repo_root: Path) -> AR6DummyRepo:
    """Write one deterministic AR6 raw data repository beneath ``repo_root``."""
    raw_dir = get_raw_dir()
    raw_logs_dir = get_logs_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_logs_dir.mkdir(parents=True, exist_ok=True)
    explorer_csv_path = get_explorer_csv_path(DEFAULT_DATABASE)
    citation_txt_path = get_citation_txt_path()
    metadata_path = get_metadata_path()
    primap_final_path = raw_dir / historical_sources.PRIMAP_FINAL_LOCAL_NAME
    primap_no_rounding_path = raw_dir / historical_sources.PRIMAP_FINAL_NO_ROUNDING_LOCAL_NAME
    gcp_path = raw_dir / historical_sources.GCP_NATIONAL_FOSSIL_LOCAL_NAME
    overlay_path = raw_dir / historical_sources.AR6_HISTORICAL_FIGURE_REFERENCE_LOCAL_NAME

    explorer_long_df, explorer_meta_df = build_ar6_explorer_frames()
    write_explorer_csv(
        csv_file=explorer_csv_path,
        data_df=explorer_long_df,
        meta_df=explorer_meta_df,
    )
    citation_txt_path.write_text("Raw citation block for AR6 test data.", encoding="utf-8")
    _write_dummy_primap_csv(primap_final_path, no_rounding=False)
    _write_dummy_primap_csv(primap_no_rounding_path, no_rounding=True)
    _write_dummy_gcp_workbook(gcp_path)
    _write_dummy_overlay_csv(overlay_path)
    return AR6DummyRepo(
        repo_root=repo_root,
        raw_dir=raw_dir,
        raw_logs_dir=raw_logs_dir,
        explorer_csv_path=explorer_csv_path,
        citation_txt_path=citation_txt_path,
        metadata_path=metadata_path,
        primap_final_path=primap_final_path,
        primap_no_rounding_path=primap_no_rounding_path,
        gcp_path=gcp_path,
        overlay_path=overlay_path,
        explorer_long_df=explorer_long_df,
        explorer_meta_df=explorer_meta_df,
    )


def clone_ar6_dummy_repo(
    template_repo: AR6DummyRepo,
    *,
    top_path: Path,
) -> AR6DummyRepo:
    """Clone one prepared AR6 dummy repo into a fresh active workspace."""
    target_top_path = Path(top_path)
    target_repo_root = target_top_path / "pyaesa"
    if target_repo_root.exists():
        shutil.rmtree(target_repo_root)
    shutil.copytree(template_repo.repo_root, target_repo_root)
    clear_default_repo_root()
    set_workspace(target_top_path, refresh=False)
    return AR6DummyRepo(
        repo_root=get_default_repo_root(),
        raw_dir=get_raw_dir(),
        raw_logs_dir=get_logs_dir(),
        explorer_csv_path=get_explorer_csv_path(DEFAULT_DATABASE),
        citation_txt_path=get_citation_txt_path(),
        metadata_path=get_metadata_path(),
        primap_final_path=get_raw_dir() / historical_sources.PRIMAP_FINAL_LOCAL_NAME,
        primap_no_rounding_path=get_raw_dir()
        / historical_sources.PRIMAP_FINAL_NO_ROUNDING_LOCAL_NAME,
        gcp_path=get_raw_dir() / historical_sources.GCP_NATIONAL_FOSSIL_LOCAL_NAME,
        overlay_path=get_raw_dir() / historical_sources.AR6_HISTORICAL_FIGURE_REFERENCE_LOCAL_NAME,
        explorer_long_df=template_repo.explorer_long_df.copy(),
        explorer_meta_df=template_repo.explorer_meta_df.copy(),
    )
