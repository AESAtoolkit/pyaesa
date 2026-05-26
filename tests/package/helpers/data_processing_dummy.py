"""Reusable real-pymrio IO/repository helpers for MRIO package tests."""

from pathlib import Path
from typing import Any
import zipfile

import pandas as pd
import pymrio

from pyaesa.download.mrios.utils.paths import (
    _get_exio_archive_path,
    _get_full_dir,
    _get_oecd_csv_path,
)
from pyaesa.download.pop_gdp.contracts import GDP_SSP_INDICATOR, POP_SSP_INDICATOR
from pyaesa.download.pop_gdp.download_wb import GDP_WB_INDICATOR, POP_WB_INDICATOR
from pyaesa.download.pop_gdp.raw_paths import _get_output_path
from pyaesa.process.mrios.utils.io.paths import _get_characterization_matrix_path
from pyaesa.process.pop_gdp.io.paths import _get_ssp_matching_path, _get_wb_matching_path


def build_product_index(
    *,
    named_levels: bool = True,
    regions: tuple[str, ...] = ("R1", "R2"),
    sectors: tuple[str, ...] = ("S1", "S2"),
) -> pd.MultiIndex:
    """Return a small product index used by dummy MRIO fixtures."""
    tuples = [(region, sector) for region in regions for sector in sectors]
    names = ["region", "sector"] if named_levels else [None, None]
    return pd.MultiIndex.from_tuples(tuples, names=names)


def build_fd_columns(
    *,
    named_levels: bool = True,
    regions: tuple[str, ...] = ("R1", "R2"),
) -> pd.MultiIndex:
    """Return a final demand column index aligned with dummy regions."""
    tuples = [(region, "FD") for region in regions]
    names = ["region", "final_demand"] if named_levels else [None, None]
    return pd.MultiIndex.from_tuples(tuples, names=names)


class DummyExtension(pymrio.Extension):
    """Thin wrapper around a real ``pymrio.Extension`` for tests."""


class DummyIOSystem(pymrio.IOSystem):
    """Thin wrapper around a real ``pymrio.IOSystem`` with test-friendly init."""

    factor_inputs: DummyExtension
    satellite_accounts: DummyExtension
    pb_lcia: DummyExtension

    def __init__(
        self,
        *,
        Z: Any,
        Y: Any,
        unit: Any,
        extensions: dict[str, Any],
        x: Any = None,
        A: Any = None,
        L: Any = None,
        G: Any = None,
    ) -> None:
        ext_kwargs = {
            name: {
                "name": extension.name,
                "F": extension.F,
                "F_Y": extension.F_Y,
                "unit": extension.unit,
                "S": extension.S,
                "M": extension.M,
                "D_cba": extension.D_cba,
                "D_pba": extension.D_pba,
                "D_pba_reg": extension.D_pba_reg,
            }
            if isinstance(extension, DummyExtension)
            else extension
            for name, extension in extensions.items()
        }
        super().__init__(Z=Z, Y=Y, unit=unit, x=x, A=A, L=L, G=G, **ext_kwargs)


def build_dummy_iosystem(
    *,
    named_levels: bool = True,
    include_satellite_accounts: bool = True,
    include_factor_inputs: bool = True,
    extra_extensions: dict[str, Any] | None = None,
    negative_y: bool = False,
    negative_factor_inputs: bool = False,
) -> DummyIOSystem:
    """Return a small IOSystem like object with deterministic matrices."""
    products = build_product_index(named_levels=named_levels)
    fd_columns = build_fd_columns(named_levels=named_levels)
    z = pd.DataFrame(
        [
            [1.0, 0.2, 0.1, 0.0],
            [0.1, 1.2, 0.0, 0.2],
            [0.2, 0.0, 1.1, 0.1],
            [0.0, 0.1, 0.2, 1.3],
        ],
        index=products,
        columns=products,
    )
    y_values = [[2.0, 1.0], [1.5, 0.5], [1.0, 2.0], [0.5, 1.5]]
    if negative_y:
        y_values[0][0] = -2.0
        y_values[3][1] = -1.5
    y = pd.DataFrame(y_values, index=products, columns=fd_columns)
    unit = pd.DataFrame({"unit": ["M EUR"] * len(products)}, index=products)

    extensions: dict[str, Any] = {}
    if include_factor_inputs:
        factor_values = [[4.0, 3.0, 2.0, 1.0]]
        if negative_factor_inputs:
            factor_values = [[-4.0, 3.0, -2.0, 1.0]]
        extensions["factor_inputs"] = DummyExtension(
            name="factor_inputs",
            F=pd.DataFrame(
                factor_values,
                index=pd.Index(["gva"], name="stressor"),
                columns=products,
            ),
            unit=pd.DataFrame({"unit": ["M EUR"]}, index=pd.Index(["gva"], name="stressor")),
        )
    if include_satellite_accounts:
        extensions["satellite_accounts"] = DummyExtension(
            name="satellite_accounts",
            F=pd.DataFrame(
                [[0.5, 1.0, 1.5, 2.0]],
                index=pd.Index(["co2"], name="stressor"),
                columns=products,
            ),
            F_Y=pd.DataFrame(
                [[0.2, 0.4]],
                index=pd.Index(["co2"], name="stressor"),
                columns=fd_columns,
            ),
            unit=pd.DataFrame({"unit": ["kg"]}, index=pd.Index(["co2"], name="stressor")),
        )
    for name, value in (extra_extensions or {}).items():
        extensions[name] = value

    iosys = DummyIOSystem(Z=z, Y=y, unit=unit, extensions=extensions)
    iosys.calc_system(include_ghosh=True)
    return iosys


def write_mrio_placeholders(repo_root: Path, *, source: str, years: list[int]) -> Path:
    """Create placeholder raw MRIO files for ``source`` and ``years``."""
    del repo_root
    full_dir = _get_full_dir(source)
    full_dir.mkdir(parents=True, exist_ok=True)
    source_key = str(source).strip().lower()
    for year in years:
        if source_key.startswith("exio"):
            system = "ixi" if source_key.endswith("_ixi") else "pxp"
            (full_dir / f"IOT_{int(year)}_{system}.zip").write_bytes(b"placeholder")
        else:
            (full_dir / f"ICIO2025_{int(year)}.csv").write_text("placeholder", encoding="utf-8")
    return full_dir


def write_oecd_raw_csv_files(
    repo_root: Path,
    *,
    years: list[int],
    regions: tuple[str, ...] = ("R1", "R2"),
    sectors: tuple[str, ...] = ("S1", "S2"),
) -> Path:
    """Write small OECD ICIO CSVs that are parseable by pymrio."""
    del repo_root
    full_dir = _get_full_dir("oecd_v2025")
    full_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{region}_{sector}" for region in regions for sector in sectors]
    final_demand = [f"{region}_HFCE" for region in regions]
    columns = [*labels, *final_demand]
    for year in years:
        rows: list[dict[str, object]] = []
        year_offset = (int(year) - min(years)) * 0.01 if years else 0.0
        for row_index, row_label in enumerate(labels):
            row: dict[str, object] = {"": row_label}
            for col_index, column in enumerate(columns):
                row[column] = 1.0 + year_offset if row_index == col_index else 0.1
            rows.append(row)
        factor_row: dict[str, object] = {"": "VA"}
        for column in columns:
            factor_row[column] = 0.5
        rows.append(factor_row)
        pd.DataFrame(rows).to_csv(_get_oecd_csv_path(full_dir, int(year)), index=False)
    return full_dir


def write_exio_archive_files(repo_root: Path, *, source: str, years: list[int]) -> Path:
    """Write small EXIOBASE archives that are parseable by pymrio."""
    del repo_root
    source_key = str(source).strip().lower()
    system = "ixi" if source_key.endswith("_ixi") else "pxp"
    full_dir = _get_full_dir(source_key)
    full_dir.mkdir(parents=True, exist_ok=True)
    base = build_dummy_iosystem()
    iosys = pymrio.IOSystem(
        Z=base.Z,
        Y=base.Y,
        unit=base.unit,
        factor_inputs={
            "name": "factor_inputs",
            "F": base.factor_inputs.F,
            "F_Y": None,
            "unit": base.factor_inputs.unit,
        },
        satellite={
            "name": "satellite",
            "F": base.satellite_accounts.F,
            "F_Y": base.satellite_accounts.F_Y,
            "unit": base.satellite_accounts.unit,
        },
    )
    iosys.calc_system(include_ghosh=True)
    for year in years:
        staging = full_dir / f".dummy_exio_{int(year)}"
        if staging.exists():
            for child in sorted(staging.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            staging.rmdir()
        iosys.save_all(staging)
        archive_path = _get_exio_archive_path(full_dir, int(year), system=system)
        with zipfile.ZipFile(archive_path, "w") as archive:
            for path in staging.rglob("*"):
                archive.write(path, path.relative_to(staging).as_posix())
        for child in sorted(staging.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        staging.rmdir()
    return full_dir


def write_characterization_matrix(
    repo_root: Path,
    *,
    source_key: str,
    method_name: str,
    extension_name: str = "satellite_accounts",
) -> Path:
    """Write a minimal EXIO characterization matrix for one method."""
    del repo_root
    path = _get_characterization_matrix_path(source_key=source_key, lcia_method=method_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "extension": extension_name,
        "stressor": "co2",
        "stressor_unit": "kg",
        "factor": 2.0,
        "impact": "climate_child",
        "impact_parent": "climate_parent",
        "impact_unit": "kg CO2-eq",
    }
    if method_name == "pb_lcia":
        row["Planetary boundary"] = "Climate change"
        row["Control variable"] = "Greenhouse gas emissions"
        frame = pd.DataFrame([row])
    else:
        row["impact_full_name"] = "climate_parent"
        frame = pd.DataFrame([row])
    frame.to_csv(path, index=False)
    return path


def build_pop_gdp_matching_frame() -> pd.DataFrame:
    """Return one small MRIO matching frame shared by pop_gdp tests."""
    return pd.DataFrame(
        [
            {
                "iso3_code": "FRA",
                "exio_code": "FR",
                "oecd_code": "FR",
                "agg_parent": "YES",
                "parent_iso3_code": "EUR",
            },
            {
                "iso3_code": "DEU",
                "exio_code": "DE",
                "oecd_code": "DE",
                "agg_parent": "YES",
                "parent_iso3_code": "EUR",
            },
            {
                "iso3_code": "EUR",
                "exio_code": "EU",
                "oecd_code": "EU",
                "agg_parent": "NO",
                "parent_iso3_code": "",
            },
            {
                "iso3_code": "USA",
                "exio_code": "US",
                "oecd_code": "US",
                "agg_parent": "NO",
                "parent_iso3_code": "",
            },
            {
                "iso3_code": "CHN",
                "exio_code": "CN",
                "oecd_code": "CN",
                "agg_parent": "NO",
                "parent_iso3_code": "",
            },
            {
                "iso3_code": "TWN",
                "exio_code": "TW",
                "oecd_code": "TW",
                "agg_parent": "NO",
                "parent_iso3_code": "",
            },
        ]
    )


def build_wb_raw_frames(*, years: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return small WB and IMF Taiwan raw frames covering core processing paths."""
    year_cols = [str(year) for year in years]
    wb = pd.DataFrame(
        [
            {
                "wb_full_name": "France",
                "iso3_code": "FRA",
                "variable": POP_WB_INDICATOR,
                "unit": "Persons",
                year_cols[0]: 60.0,
                year_cols[-1]: 62.0,
            },
            {
                "wb_full_name": "France",
                "iso3_code": "FRA",
                "variable": GDP_WB_INDICATOR,
                "unit": "USD_2021/yr",
                year_cols[0]: 300.0,
                year_cols[-1]: 340.0,
            },
            {
                "wb_full_name": "Germany",
                "iso3_code": "DEU",
                "variable": POP_WB_INDICATOR,
                "unit": "Persons",
                year_cols[0]: 80.0,
                year_cols[-1]: 81.0,
            },
            {
                "wb_full_name": "Germany",
                "iso3_code": "DEU",
                "variable": GDP_WB_INDICATOR,
                "unit": "USD_2021/yr",
                year_cols[0]: 500.0,
                year_cols[-1]: 520.0,
            },
            {
                "wb_full_name": "China",
                "iso3_code": "CHN",
                "variable": POP_WB_INDICATOR,
                "unit": "Persons",
                year_cols[0]: 1400.0,
                year_cols[-1]: 1410.0,
            },
            {
                "wb_full_name": "China",
                "iso3_code": "CHN",
                "variable": GDP_WB_INDICATOR,
                "unit": "USD_2021/yr",
                year_cols[0]: 14000.0,
                year_cols[-1]: 14500.0,
            },
            {
                "wb_full_name": "United States",
                "iso3_code": "USA",
                "variable": POP_WB_INDICATOR,
                "unit": "Persons",
                year_cols[0]: 320.0,
                year_cols[-1]: 325.0,
            },
            {
                "wb_full_name": "United States",
                "iso3_code": "USA",
                "variable": GDP_WB_INDICATOR,
                "unit": "USD_2021/yr",
                year_cols[0]: 18000.0,
                year_cols[-1]: 19000.0,
            },
        ]
    )
    imf = pd.DataFrame(
        [
            {
                "wb_full_name": "Taiwan",
                "iso3_code": "TWN",
                "variable": POP_WB_INDICATOR,
                "unit": "Persons",
                year_cols[0]: 23.0,
                year_cols[-1]: 24.0,
            },
            {
                "wb_full_name": "Taiwan",
                "iso3_code": "TWN",
                "variable": GDP_WB_INDICATOR,
                "unit": "USD_2021/yr",
                year_cols[0]: 700.0,
                year_cols[-1]: 750.0,
            },
        ]
    )
    return wb, imf


def build_ssp_raw_frame(*, years: list[int]) -> pd.DataFrame:
    """Return a small SSP raw frame covering interpolation and unit conversion."""
    year_cols = [str(year) for year in years]
    return pd.DataFrame(
        [
            {
                "model": "IIASA-WiC POP",
                "ssp_scenario": "SSP2",
                "ssp_full_name": "France",
                "variable": POP_SSP_INDICATOR,
                "unit": "Millions",
                year_cols[0]: 65.0,
                year_cols[-1]: 67.0,
            },
            {
                "model": "IIASA-WiC POP",
                "ssp_scenario": "SSP2",
                "ssp_full_name": "France",
                "variable": GDP_SSP_INDICATOR,
                "unit": "Billion USD",
                year_cols[0]: 400.0,
                year_cols[-1]: 440.0,
            },
            {
                "model": "IIASA-WiC POP",
                "ssp_scenario": "SSP2",
                "ssp_full_name": "Germany",
                "variable": POP_SSP_INDICATOR,
                "unit": "Persons",
                year_cols[0]: 82.0,
                year_cols[-1]: 83.0,
            },
            {
                "model": "IIASA-WiC POP",
                "ssp_scenario": "SSP2",
                "ssp_full_name": "Germany",
                "variable": GDP_SSP_INDICATOR,
                "unit": "USD_2017/yr",
                year_cols[0]: 600.0,
                year_cols[-1]: 630.0,
            },
            {
                "model": "OECD ENV-Growth 2023",
                "ssp_scenario": "SSP2",
                "ssp_full_name": "France",
                "variable": POP_SSP_INDICATOR,
                "unit": "Persons",
                year_cols[0]: 999.0,
                year_cols[-1]: 999.0,
            },
            {
                "model": "IIASA-WiC POP",
                "ssp_scenario": "SSP2",
                "ssp_full_name": "Not a Country",
                "variable": POP_SSP_INDICATOR,
                "unit": "Persons",
                year_cols[0]: 1.0,
                year_cols[-1]: 2.0,
            },
        ]
    )


def write_pop_gdp_raw_files(
    repo_root: Path,
    *,
    wb_years: list[int],
    ssp_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write reusable raw pop_gdp CSVs and return the frames used."""
    wb_raw, imf_raw = build_wb_raw_frames(years=wb_years)
    ssp_raw = build_ssp_raw_frame(years=ssp_years)
    matching = build_pop_gdp_matching_frame()
    for key, frame in {"wb": wb_raw, "imf_twn": imf_raw, "ssp": ssp_raw}.items():
        path = _get_output_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    del repo_root
    return wb_raw, imf_raw, ssp_raw, matching


def write_pop_gdp_matching_files(repo_root: Path, *, matching: pd.DataFrame) -> None:
    """Write matching CSVs for WB and SSP processing wrappers."""
    del repo_root
    for path in (
        _get_wb_matching_path("exiobase_396_ixi"),
        _get_wb_matching_path("oecd_v2025"),
        _get_ssp_matching_path("exiobase_396_ixi"),
        _get_ssp_matching_path("oecd_v2025"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        matching.to_csv(path, index=False)
