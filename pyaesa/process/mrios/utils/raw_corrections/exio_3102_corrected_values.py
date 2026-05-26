"""Maintainer precomputation for EXIOBASE 3.10.2 raw corrections."""

from collections.abc import Callable
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pyaesa.asocc.orchestration.projection.regression.regression_core_utils import (
    coerce_numeric_pairs,
    fit_simple_ols,
    validate_min_ols_uncertainty_observations,
)

from .basis import corrected_values_dir, positive_factor_inputs_basis_from_frame

FULL_YEAR_RANGE = tuple(range(1995, 2025))
CH_FIT_YEARS, LU_FIT_YEARS = (
    tuple(range(1995, 2021)),
    tuple(year for year in range(1995, 2023) if year != 2020),
)


@dataclass(frozen=True)
class BuildOutputs:
    """Raw corrected values table produced for one source."""

    corrected_values_path: Path


def _report(progress: Callable[[str], None], message: str) -> None:
    progress(message)


class _WorkspaceData:
    """Raw EXIO data and repository prerequisite access for one source."""

    def __init__(self, *, workspace_root: Path, source: str) -> None:
        self.source = source
        self.system = "ixi" if source.endswith("_ixi") else "pxp"
        self._workspace_root = workspace_root
        self._repo_root = Path(__file__).resolve().parents[5]
        self._raw_zip_dir = (
            workspace_root
            / "pyaesa"
            / "data_raw"
            / "mrio"
            / "exiobase_3"
            / "exiobase_3102"
            / f"full_{self.system}"
        )
        prereq = self._repo_root / "pyaesa" / "workspace_initialisation" / "prerequisites" / "mrio"
        self._agg_reg_eu27_path = prereq / "exiobase_3" / "aggregation" / "agg_reg_eu27.csv"
        self._pb_lcia_path = (
            prereq / "exiobase_3" / "lcia" / "characterization_factors_matrices" / "pb_lcia.csv"
        )
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


def _series_for_region(frame: pd.DataFrame, *, stressor: str, region: str) -> pd.Series:
    row = pd.Series(
        pd.to_numeric(frame.loc[str(stressor)], errors="coerce"),
        copy=False,
    ).astype(float)
    result = row.xs(str(region), level="region")
    result.index = result.index.map(str)
    return result.sort_index()


def _predict_ols(
    *,
    x_hist: list[float],
    y_hist: list[float],
    x_target: float,
) -> tuple[float, str, bool]:
    x_fit, y_fit = coerce_numeric_pairs(x_values=x_hist, y_values=y_hist)
    n_obs = int(x_fit.size)
    validate_min_ols_uncertainty_observations(
        n_obs=n_obs,
        context="EXIO raw correction OLS levels",
        detail="Requires at least three finite observations.",
    )
    intercept, slope, _r_squared, _p_value = fit_simple_ols(x=x_fit, y=y_fit)
    raw = float(intercept + slope * float(x_target))
    clipped = raw < 0.0
    detail = f"n_obs={n_obs}, intercept={intercept:.17g}, slope={slope:.17g}"
    return (0.0 if clipped else raw), detail, clipped


def _base_row(**kwargs: object) -> dict[str, object]:
    row = dict(kwargs)
    row["basis_label"] = "positive_factor_inputs_sum_by_region_sector"
    return row


def _float_at(series: pd.Series, key: str) -> float:
    value = series.get(key, np.nan)
    return float(np.nan if value is None else value)


def _basis_history(cache: _WorkspaceData, *, region: str, years: tuple[int, ...]) -> pd.DataFrame:
    return cache.load_basis_history(region=region, years=years)


def _stressor_history(
    cache: _WorkspaceData,
    *,
    extension: str,
    stressor: str,
    region: str,
    years: tuple[int, ...],
) -> pd.DataFrame:
    history = {
        int(year): _series_for_region(
            cache.load_extension_frame(year=year, extension=extension),
            stressor=stressor,
            region=region,
        ).astype(float)
        for year in years
    }
    return pd.DataFrame.from_dict(history, orient="index").sort_index()


def _build_tw_rows(
    cache: _WorkspaceData, *, source: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for year in FULL_YEAR_RANGE:
        frame = cache.load_extension_frame(year=year, extension="land")
        donor_env = _series_for_region(frame, stressor="Forest", region="CN")
        target_env = _series_for_region(frame, stressor="Forest", region="TW")
        donor_basis = cache.load_basis(year=year).xs("CN", level="r_p").sort_index()
        target_basis = cache.load_basis(year=year).xs("TW", level="r_p").sort_index()
        common = donor_env.index.intersection(donor_basis.index).intersection(target_basis.index)
        donor_positive = donor_env.reindex(common).astype(float) > 0.0
        eligible = donor_positive & target_basis.reindex(common).astype(float).gt(0.0)
        skipped = donor_positive & ~eligible
        donor_slice = donor_env.reindex(common).loc[eligible]
        donor_basis_slice = donor_basis.reindex(common).loc[eligible]
        if bool((donor_basis_slice <= 0.0).any()):
            raise ValueError(
                "TW CN donor basis must be positive on donor positive eligible sectors."
            )
        corrected = donor_slice.div(donor_basis_slice).mul(
            target_basis.reindex(common).loc[eligible]
        )
        for sector, corrected_value in corrected.items():
            original_value = _float_at(target_env, str(sector))
            rows.append(
                _base_row(
                    source=source,
                    extension="land",
                    stressor="Forest",
                    region="TW",
                    sector=str(sector),
                    year=int(year),
                    original_value=original_value,
                    corrected_value=float(corrected_value),
                    correction_method="donor_sector_intensity",
                    fit_window="",
                    correction_reason=(
                        "TW correction: missing extension data recorded at 0. "
                        "CN donor sectors with positive Forest values are used."
                    ),
                    replaced_nonzero_source=bool(not np.isclose(original_value, 0.0)),
                    prediction_clipped_to_zero=False,
                )
            )
        for sector in skipped[skipped].index:
            diagnostics.append(
                _base_row(
                    source=source,
                    extension="land",
                    stressor="Forest",
                    region="TW",
                    sector=str(sector),
                    year=int(year),
                    original_value=_float_at(target_env, str(sector)),
                    corrected_value=np.nan,
                    correction_method="skip_missing_target_predictor",
                    fit_window="",
                    correction_reason=(
                        "TW correction skipped: CN donor Forest is positive but "
                        "the TW predictor is missing."
                    ),
                    replaced_nonzero_source=False,
                    prediction_clipped_to_zero=False,
                    diagnostic_type="skipped_target_predictor",
                )
            )
    return rows, diagnostics


def _build_mt_rows(
    cache: _WorkspaceData, *, source: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for year in FULL_YEAR_RANGE:
        frame = cache.load_extension_frame(year=year, extension="land")
        target_env = _series_for_region(frame, stressor="Forest", region="MT")
        target_basis = cache.load_basis(year=year).xs("MT", level="r_p").sort_index()
        donor_env = pd.concat(
            [
                _series_for_region(frame, stressor="Forest", region=region)
                for region in cache.eu27_regions
            ],
            axis=1,
        ).sum(axis=1, min_count=1)
        donor_basis = pd.concat(
            [
                cache.load_basis(year=year).xs(region, level="r_p").sort_index()
                for region in cache.eu27_regions
            ],
            axis=1,
        ).sum(axis=1, min_count=1)
        common = donor_env.index.intersection(donor_basis.index).intersection(target_basis.index)
        donor_positive = donor_env.reindex(common).astype(float) > 0.0
        eligible = donor_positive & target_basis.reindex(common).astype(float).gt(0.0)
        corrected = (
            donor_env.reindex(common)
            .loc[eligible]
            .div(donor_basis.reindex(common).loc[eligible])
            .mul(target_basis.reindex(common).loc[eligible])
        )
        for sector, corrected_value in corrected.items():
            original_value = _float_at(target_env, str(sector))
            rows.append(
                _base_row(
                    source=source,
                    extension="land",
                    stressor="Forest",
                    region="MT",
                    sector=str(sector),
                    year=int(year),
                    original_value=original_value,
                    corrected_value=float(corrected_value),
                    correction_method="donor_sector_intensity",
                    fit_window="",
                    correction_reason=(
                        "MT correction: missing extension data recorded at 0, "
                        "and some present values are incoherently too small. "
                        "Pooled EU27 donor sectors with a MT predictor are used."
                    ),
                    replaced_nonzero_source=bool(not np.isclose(original_value, 0.0)),
                    prediction_clipped_to_zero=False,
                )
            )
        skipped = donor_positive & ~eligible
        for sector in skipped[skipped].index:
            diagnostics.append(
                _base_row(
                    source=source,
                    extension="land",
                    stressor="Forest",
                    region="MT",
                    sector=str(sector),
                    year=int(year),
                    original_value=_float_at(target_env, str(sector)),
                    corrected_value=np.nan,
                    correction_method="skip_missing_target_predictor",
                    fit_window="",
                    correction_reason=(
                        "MT correction skipped: pooled EU27 donor Forest is "
                        "positive but the MT predictor is missing."
                    ),
                    replaced_nonzero_source=False,
                    prediction_clipped_to_zero=False,
                    diagnostic_type="skipped_target_predictor",
                )
            )
    return rows, diagnostics


def _build_regression_rows(
    cache: _WorkspaceData,
    *,
    source: str,
    extension: str,
    stressor: str,
    region: str,
    fit_years: tuple[int, ...],
    predict_years: tuple[int, ...],
    correction_reason: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    fit_basis = _basis_history(cache, region=region, years=fit_years)
    target_basis_history = _basis_history(cache, region=region, years=predict_years)
    fit_env = _stressor_history(
        cache,
        extension=extension,
        stressor=stressor,
        region=region,
        years=fit_years,
    )
    target_env_history = _stressor_history(
        cache,
        extension=extension,
        stressor=stressor,
        region=region,
        years=predict_years,
    )
    sectors = (
        fit_env.columns.intersection(fit_basis.columns)
        .intersection(target_basis_history.columns)
        .intersection(target_env_history.columns)
        .tolist()
    )
    fit_basis = fit_basis.reindex(columns=sectors).astype(float)
    target_basis_history = target_basis_history.reindex(columns=sectors).astype(float)
    fit_env = fit_env.reindex(columns=sectors).astype(float)
    target_env_history = target_env_history.reindex(columns=sectors).astype(float)
    for year in predict_years:
        target_basis = target_basis_history.loc[int(year)]
        target_env = target_env_history.loc[int(year)]
        eligible = target_basis > 0.0
        for sector in eligible[eligible].index:
            x_hist = [float(value) for value in fit_basis.loc[:, str(sector)].tolist()]
            y_hist = [float(value) for value in fit_env.loc[:, str(sector)].tolist()]
            corrected_value, detail, clipped = _predict_ols(
                x_hist=x_hist,
                y_hist=y_hist,
                x_target=_float_at(target_basis, str(sector)),
            )
            original_value = _float_at(target_env, str(sector))
            row = _base_row(
                source=source,
                extension=extension,
                stressor=stressor,
                region=region,
                sector=str(sector),
                year=int(year),
                original_value=original_value,
                corrected_value=corrected_value,
                correction_method="ols_level",
                fit_window=f"{min(fit_years)}-{max(fit_years)}",
                correction_reason=correction_reason,
                replaced_nonzero_source=bool(not np.isclose(original_value, 0.0)),
                prediction_clipped_to_zero=bool(clipped),
            )
            rows.append(row)
            diagnostics.append(
                {
                    **row,
                    "diagnostic_type": "ols_level_fit",
                    "detail": detail,
                }
            )
        skipped = target_basis <= 0.0
        for sector in skipped[skipped].index:
            diagnostics.append(
                _base_row(
                    source=source,
                    extension=extension,
                    stressor=stressor,
                    region=region,
                    sector=str(sector),
                    year=int(year),
                    original_value=_float_at(target_env, str(sector)),
                    corrected_value=np.nan,
                    correction_method="skip_missing_target_predictor",
                    fit_window=f"{min(fit_years)}-{max(fit_years)}",
                    correction_reason=(
                        "Target sector skipped because the positive factor "
                        "input predictor is missing."
                    ),
                    replaced_nonzero_source=False,
                    prediction_clipped_to_zero=False,
                    diagnostic_type="skipped_target_predictor",
                )
            )
    return rows, diagnostics


def build_corrected_values_outputs_for_source(
    *,
    workspace_root: Path,
    source: str,
    progress: Callable[[str], None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build raw corrected values review and diagnostics tables for one source."""
    cache = _WorkspaceData(workspace_root=workspace_root, source=source)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []

    _report(progress, f"{source}: building TW and MT land corrections")
    tw_rows, tw_diag = _build_tw_rows(cache, source=source)
    mt_rows, mt_diag = _build_mt_rows(cache, source=source)
    rows.extend(tw_rows)
    rows.extend(mt_rows)
    diagnostics.extend(tw_diag)
    diagnostics.extend(mt_diag)

    _report(progress, f"{source}: building CH nutrient regressions")
    for stressor in ("P - agriculture - water", "P - waste - water"):
        _report(progress, f"{source}: CH regression for {stressor}")
        ch_rows, ch_diag = _build_regression_rows(
            cache,
            source=source,
            extension="nutrients",
            stressor=stressor,
            region="CH",
            fit_years=CH_FIT_YEARS,
            predict_years=(2021, 2022),
            correction_reason=(
                "CH correction: 2021 and 2022 extension data are incoherently too small."
            ),
        )
        rows.extend(ch_rows)
        diagnostics.extend(ch_diag)

    _report(progress, f"{source}: building LU water regressions")
    for idx, stressor in enumerate(cache.fwu_stressors, start=1):
        if idx == 1 or idx % 10 == 0 or idx == len(cache.fwu_stressors):
            _report(progress, f"{source}: LU regression stressor {idx}/{len(cache.fwu_stressors)}")
        lu_rows, lu_diag = _build_regression_rows(
            cache,
            source=source,
            extension="water",
            stressor=stressor,
            region="LU",
            fit_years=LU_FIT_YEARS,
            predict_years=(2020,),
            correction_reason=("LU correction: 2020 extension data are incoherently too small."),
        )
        rows.extend(lu_rows)
        diagnostics.extend(lu_diag)

    corrections = pd.DataFrame(rows).sort_values(
        ["extension", "stressor", "region", "year", "sector"], ignore_index=True
    )
    review = corrections.copy()
    review["delta"] = review["corrected_value"] - review["original_value"]
    review["abs_delta"] = review["delta"].abs()
    review["abs_pct_change"] = np.where(
        review["original_value"].abs() > 0.0,
        review["abs_delta"] / review["original_value"].abs(),
        np.nan,
    )
    diagnostics_frame = pd.DataFrame(diagnostics).sort_values(
        ["extension", "stressor", "region", "year", "sector"], ignore_index=True
    )
    return review, diagnostics_frame


def write_corrected_values_outputs(
    *,
    workspace_root: Path,
    source: str,
    out_dir: Path | None = None,
    progress: Callable[[str], None],
) -> BuildOutputs:
    """Persist one raw corrected values table for one source."""
    review, _diagnostics = build_corrected_values_outputs_for_source(
        workspace_root=workspace_root,
        source=source,
        progress=progress,
    )
    corrected_values = review.copy()
    out_dir = out_dir or corrected_values_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{source}_raw_corrected_values"
    corrected_values_path = out_dir / f"{stem}.csv"
    _report(progress, f"{source}: writing corrected values table")
    corrected_values.to_csv(corrected_values_path, index=False)
    return BuildOutputs(corrected_values_path=corrected_values_path)
