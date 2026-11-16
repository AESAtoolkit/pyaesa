"""Shared source-version-aware raw archives for correction generation tests."""

from pathlib import Path
import zipfile

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _eu27_regions() -> list[str]:
    frame = pd.read_csv(
        _repo_root()
        / "pyaesa"
        / "workspace_initialisation"
        / "prerequisites"
        / "mrio"
        / "exiobase_3"
        / "aggregation"
        / "agg_reg_eu27.csv",
        encoding="latin1",
    )
    return sorted(
        {
            str(value).strip()
            for value in frame.loc[frame["aggregated_mrio"].eq("EU27"), "original_classification"]
            if str(value).strip() and str(value).strip() != "MT"
        }
    )


def _fwu_stressors() -> list[str]:
    frame = pd.read_csv(
        _repo_root()
        / "pyaesa"
        / "workspace_initialisation"
        / "prerequisites"
        / "mrio"
        / "exiobase_3"
        / "lcia"
        / "characterization_factors_matrices"
        / "exiobase_3102"
        / "pb_lcia.csv",
        encoding="latin1",
    )
    mask = frame["extension"].astype(str).str.strip().eq("water") & frame["impact_parent"].astype(
        str
    ).str.strip().eq("FWU")
    return sorted(
        {str(value).strip() for value in frame.loc[mask, "stressor"] if str(value).strip()}
    )


def _multiindex_columns(regions: list[str], sectors: list[str]) -> pd.MultiIndex:
    tuples = [(region, sector) for region in regions for sector in sectors]
    return pd.MultiIndex.from_tuples(tuples, names=["region", "sector"]).sort_values()


def _build_workspace(tmp_path: Path, system: str, *, version: str = "3102") -> Path:
    workspace_root = tmp_path / "workspace"
    raw_dir = (
        workspace_root
        / "data_raw"
        / "mrio"
        / "exiobase_3"
        / f"exiobase_{version}"
        / f"full_{system}"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    eu27_regions = _eu27_regions()
    regions = list(dict.fromkeys(["CN", "TW", "MT", "CH", "LU", *eu27_regions]))
    sectors = ["S_donor", "S_skip", "S_ch", "S_lu"]
    columns = _multiindex_columns(regions, sectors)
    fwu_stressors = _fwu_stressors()
    donor_eu27_regions = [region for region in eu27_regions if region not in {"CH", "LU"}]

    for year in range(1995, 2025):
        land = pd.DataFrame(0.0, index=pd.Index(["Forest"], name="stressor"), columns=columns)
        nutrients = pd.DataFrame(
            0.0,
            index=pd.Index(["P - agriculture - water", "P - waste - water"], name="stressor"),
            columns=columns,
        )
        water = pd.DataFrame(
            0.0,
            index=pd.Index(fwu_stressors, name="stressor"),
            columns=columns,
        )
        factor_inputs = pd.DataFrame(
            0.0,
            index=pd.Index(["fi_pos", "fi_neg"], name="stressor"),
            columns=columns,
        )

        land.loc["Forest", ("CN", "S_donor")] = 10.0
        land.loc["Forest", ("TW", "S_donor")] = 0.0
        for region in donor_eu27_regions:
            land.loc["Forest", (region, "S_donor")] = 8.0
            land.loc["Forest", (region, "S_skip")] = 5.0
        land.loc["Forest", ("MT", "S_donor")] = 1.0 if year in (2010, 2011) else 0.0

        factor_inputs.loc["fi_pos", ("CN", "S_donor")] = 2.0
        factor_inputs.loc["fi_pos", ("TW", "S_donor")] = 6.0
        factor_inputs.loc["fi_pos", ("MT", "S_donor")] = 4.0
        factor_inputs.loc["fi_neg", ("MT", "S_skip")] = -3.0
        for region in donor_eu27_regions:
            factor_inputs.loc["fi_pos", (region, "S_donor")] = 4.0
            factor_inputs.loc["fi_pos", (region, "S_skip")] = 3.0

        ch_x = float(year - 1990)
        factor_inputs.loc["fi_pos", ("CH", "S_ch")] = ch_x
        nutrients.loc["P - agriculture - water", ("CH", "S_ch")] = 2.0 * ch_x
        nutrients.loc["P - waste - water", ("CH", "S_ch")] = 3.0 * ch_x
        if version == "3102" and year in (2021, 2022):
            nutrients.loc["P - agriculture - water", ("CH", "S_ch")] = 0.0
            nutrients.loc["P - waste - water", ("CH", "S_ch")] = 0.0

        lu_x = float(year - 1990)
        factor_inputs.loc["fi_pos", ("LU", "S_lu")] = lu_x
        for stressor in fwu_stressors:
            water.loc[stressor, ("LU", "S_lu")] = 4.0 * lu_x
        if version == "3102" and year == 2020:
            for stressor in fwu_stressors:
                water.loc[stressor, ("LU", "S_lu")] = 0.0

        if version == "3112":
            land = pd.concat(
                [
                    land.rename(
                        index={"Forest": "Naturally generating forest - Non primary forest"}
                    )
                    * 0.6,
                    land.rename(index={"Forest": "Planted forest"}) * 0.4,
                ]
            )
        zip_path = raw_dir / f"IOT_{year}_{system}.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for extension_name, frame in {
                "land": land,
                "nutrients": nutrients,
                "water": water,
                "factor_inputs": factor_inputs,
            }.items():
                if version == "3112" and year == 2024 and extension_name == "land":
                    continue
                archive.writestr(f"{extension_name}/F.txt", frame.to_csv(sep="\t"))
    return workspace_root


def _zip_path(*, workspace_root: Path, system: str, year: int) -> Path:
    return (
        workspace_root
        / "data_raw"
        / "mrio"
        / "exiobase_3"
        / "exiobase_3102"
        / f"full_{system}"
        / f"IOT_{year}_{system}.zip"
    )


def _read_extension_frame(
    *, workspace_root: Path, system: str, year: int, extension: str
) -> pd.DataFrame:
    with zipfile.ZipFile(
        _zip_path(workspace_root=workspace_root, system=system, year=year)
    ) as archive:
        with archive.open(f"{extension}/F.txt") as raw:
            return pd.read_csv(
                raw,
                sep="\t",
                index_col=0,
                header=[0, 1],
                encoding="latin1",
            )


def _rewrite_extension_frame(
    *,
    workspace_root: Path,
    system: str,
    year: int,
    extension: str,
    frame: pd.DataFrame,
) -> None:
    zip_path = _zip_path(workspace_root=workspace_root, system=system, year=year)
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(zip_path, "r") as archive:
        for name in archive.namelist():
            if name != f"{extension}/F.txt":
                entries[name] = archive.read(name)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
        archive.writestr(f"{extension}/F.txt", frame.to_csv(sep="\t"))
