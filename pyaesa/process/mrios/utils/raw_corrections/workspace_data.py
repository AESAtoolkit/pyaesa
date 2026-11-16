"""Raw archive and prerequisite access for correction precomputation."""

import zipfile
from pathlib import Path

import pandas as pd

from pyaesa.download.mrios.utils.paths import _resolve_source_layout
from pyaesa.download.mrios.utils.source_registry import get_mrio_entry

from .basis import positive_factor_inputs_basis_from_frame


class _WorkspaceData:
    """Raw EXIO data and repository prerequisite access for one source."""

    def __init__(self, *, workspace_root: Path, source: str) -> None:
        self.source = source
        entry = get_mrio_entry(source)
        self.version_token = entry.version_token
        self.system = str(entry.system)
        source_root, full_dir_name = _resolve_source_layout(source)
        self._raw_zip_dir = workspace_root / "data_raw" / "mrio" / source_root / full_dir_name
        prereq = Path(__file__).resolve().parents[4] / "workspace_initialisation/prerequisites/mrio"
        self._agg_reg_eu27_path = prereq / "exiobase_3/aggregation/agg_reg_eu27.csv"
        self._pb_lcia_path = (
            prereq
            / "exiobase_3/lcia/characterization_factors_matrices"
            / entry.raw_root
            / "pb_lcia.csv"
        )
        self.forest_stressors = {
            "3102": ("Forest",),
            "3112": ("Naturally generating forest - Non primary forest", "Planted forest"),
        }[self.version_token]
        # EXIOBASE 3.11.2 has no land extension for its final modeled year, 2024.
        last_land_year = (
            entry.default_historical_cutoff
            if self.version_token == "3112"
            else entry.modeled_year_max
        )
        self.land_years = tuple(range(entry.modeled_year_min, last_land_year + 1))
        self._frame_cache: dict[tuple[int, str], pd.DataFrame] = {}
        self._basis_cache: dict[int, pd.Series] = {}
        self._basis_history_cache: dict[tuple[str, tuple[int, ...]], pd.DataFrame] = {}
        self._eu27_regions = self._load_eu27_regions()
        self._fwu_stressors = self._load_fwu_stressors()

    @property
    def eu27_regions(self) -> tuple[str, ...]:
        return self._eu27_regions

    @property
    def fwu_stressors(self) -> tuple[str, ...]:
        return self._fwu_stressors

    def _load_eu27_regions(self) -> tuple[str, ...]:
        frame = pd.read_csv(self._agg_reg_eu27_path, encoding="latin1")
        eu27_rows = frame["aggregated_mrio"].astype(str).str.strip().eq("EU27")
        regions = sorted(
            {
                str(value).strip()
                for value in frame.loc[eu27_rows, "original_classification"]
                if str(value).strip() and str(value).strip() != "MT"
            }
        )
        return tuple(regions)

    def _load_fwu_stressors(self) -> tuple[str, ...]:
        frame = pd.read_csv(self._pb_lcia_path, encoding="latin1")
        mask = frame["extension"].astype(str).str.strip().eq("water") & frame[
            "impact_parent"
        ].astype(str).str.strip().eq("FWU")
        stressors = sorted({str(value).strip() for value in frame.loc[mask, "stressor"]})
        return tuple(value for value in stressors if value)

    def load_extension_frame(self, *, year: int, extension: str) -> pd.DataFrame:
        key = (int(year), str(extension))
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached
        zip_path = self._raw_zip_dir / f"IOT_{int(year)}_{self.system}.zip"
        with zipfile.ZipFile(zip_path) as handle:
            with handle.open(f"{extension}/F.txt") as raw:
                frame = pd.read_csv(
                    raw,
                    sep="\t",
                    index_col=0,
                    header=[0, 1],
                    encoding="latin1",
                )
        frame.columns = pd.MultiIndex.from_arrays(
            [
                frame.columns.get_level_values(0).map(str),
                frame.columns.get_level_values(1).map(str),
            ],
            names=["region", "sector"],
        )
        frame.index = frame.index.map(str)
        self._frame_cache[key] = frame
        return frame

    def load_basis(self, *, year: int) -> pd.Series:
        cached = self._basis_cache.get(int(year))
        if cached is not None:
            return cached
        basis = positive_factor_inputs_basis_from_frame(
            self.load_extension_frame(year=int(year), extension="factor_inputs")
        )
        self._basis_cache[int(year)] = basis
        return basis

    def load_basis_history(self, *, region: str, years: tuple[int, ...]) -> pd.DataFrame:
        """Return cached positive factor input history for one region and year set."""
        key = (str(region), tuple(int(year) for year in years))
        cached = self._basis_history_cache.get(key)
        if cached is not None:
            return cached
        history = {
            int(year): self.load_basis(year=year).xs(str(region), level="r_p").astype(float)
            for year in key[1]
        }
        frame = pd.DataFrame.from_dict(history, orient="index").sort_index()
        self._basis_history_cache[key] = frame
        return frame
