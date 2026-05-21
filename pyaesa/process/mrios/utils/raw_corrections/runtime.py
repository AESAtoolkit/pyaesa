"""Runtime application of EXIOBASE 3.10.2 raw corrections."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pyaesa.shared.runtime.io.filesystem import ensure_dir
from pyaesa.process.mrios.utils.io.paths import _get_mrio_raw_corrected_values_log_path

from .basis import corrected_values_dir

SUPPORTED_SOURCES = ("exiobase_3102_ixi", "exiobase_3102_pxp")


@dataclass(frozen=True)
class AppliedCorrectionSummary:
    """Summary of raw corrected values applied to one parsed year."""

    source: str
    year: int
    row_count: int
    log_path: Path | None


def _stressor_family_label(*, extension: str, region: str, stressor: str) -> str:
    """Return the user facing stressor family label for one correction row."""
    if str(extension) == "water" and str(region) == "LU":
        return "water consumption"
    if str(extension) == "nutrients" and str(region) == "CH":
        return "P"
    return str(stressor)


def _country_label(region: str) -> str:
    """Return the user facing country label for one correction scope."""
    labels = {
        "CH": "Switzerland",
        "LU": "Luxembourg",
        "MT": "Malta",
        "TW": "Taiwan",
    }
    return labels.get(region, region)


def _scope_label(*, region: str, extension: str, stressor_family: str) -> str:
    """Return the user facing scope label for one correction scope."""
    if region == "CH" and extension == "nutrients":
        return "nutrients/P"
    if region == "LU" and extension == "water":
        return "water consumption"
    return f"{extension}/{stressor_family}"


def _issue_text(*, region: str, extension: str) -> str:
    """Return the user facing issue description for one correction scope."""
    if region == "CH" and extension == "nutrients":
        return "EXIOBASE values were too small"
    if region == "LU" and extension == "water":
        return "EXIOBASE values were too small"
    if region == "MT" and extension == "land":
        return "EXIOBASE values were missing at 0 or too small"
    if region == "TW" and extension == "land":
        return "EXIOBASE values were missing at 0"
    return "EXIOBASE values were corrected"


def _method_detail(*, region: str, correction_method: str) -> str:
    """Return the user facing method detail for one correction scope."""
    if correction_method == "donor_sector_intensity" and region == "TW":
        return "sector-wise China intensity per euro of value added"
    if correction_method == "donor_sector_intensity" and region == "MT":
        return "sector-wise EU average intensity per euro of value added"
    if correction_method == "ols_level":
        return "sector-wise OLS levels regression from other years, with value added as predictor"
    return str(correction_method)


def _format_year_ranges(years: Iterable[int]) -> str:
    """Format years as compact ranges."""
    values = sorted({int(year) for year in years})
    if not values:
        return "[]"
    ranges: list[str] = []
    start = values[0]
    prev = values[0]
    for year in values[1:]:
        if year == prev + 1:
            prev = year
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = year
        prev = year
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def build_scope_summary(
    *,
    source: str,
    region: str,
    extension: str,
    stressor_family: str,
    correction_method: str,
    years: Iterable[int],
) -> str:
    """Return one user facing correction summary sentence."""
    unique_years = sorted({int(year) for year in years})
    year_label = _format_year_ranges(unique_years)
    year_text = f"years {year_label}" if len(unique_years) > 1 else f"year {year_label}"
    summary = (
        f"{_country_label(region)}, {year_text}, "
        f"{_scope_label(region=region, extension=extension, stressor_family=stressor_family)}: "
        f"{_issue_text(region=region, extension=extension)}; corrected using "
        f"{_method_detail(region=region, correction_method=correction_method)}."
    )
    if region == "MT" and source.endswith("_pxp"):
        return (
            f"{summary} Other years were not corrected because the relevant sector "
            "had no value added."
        )
    return summary


def summarize_correction_rows(*, year_rows: pd.DataFrame) -> list[str]:
    """Return grouped user facing summary lines for one processed year."""
    if year_rows.empty:
        return []
    frame = year_rows.copy()
    frame["stressor_family"] = frame.apply(
        lambda row: _stressor_family_label(
            extension=str(row["extension"]),
            region=str(row["region"]),
            stressor=str(row["stressor"]),
        ),
        axis=1,
    )
    summaries: list[str] = []
    group_cols = [
        "year",
        "region",
        "extension",
        "stressor_family",
        "correction_method",
        "correction_reason",
    ]
    grouped_records = sorted(
        frame[group_cols].drop_duplicates().itertuples(index=False, name=None),
        key=lambda row: (
            int(row[0]),
            str(row[1]),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            str(row[5]),
        ),
    )
    source = str(year_rows["source"].iloc[0]) if "source" in year_rows.columns else ""
    for row in grouped_records:
        summaries.append(
            build_scope_summary(
                source=source,
                region=str(row[1]),
                extension=str(row[2]),
                stressor_family=str(row[3]),
                correction_method=str(row[4]),
                years=[int(row[0])],
            )
        )
    return summaries


def summarize_correction_scopes(*, year_rows: pd.DataFrame) -> list[dict[str, object]]:
    """Return grouped correction scopes for report level aggregation."""
    if year_rows.empty:
        return []
    frame = year_rows.copy()
    frame["stressor_family"] = frame.apply(
        lambda row: _stressor_family_label(
            extension=str(row["extension"]),
            region=str(row["region"]),
            stressor=str(row["stressor"]),
        ),
        axis=1,
    )
    group_cols = [
        "year",
        "region",
        "extension",
        "stressor_family",
        "correction_method",
        "correction_reason",
    ]
    scopes: list[dict[str, object]] = []
    for row in frame[group_cols].drop_duplicates().itertuples(index=False, name=None):
        scopes.append(
            {
                "year": int(row[0]),
                "region": str(row[1]),
                "extension": str(row[2]),
                "stressor_family": str(row[3]),
                "correction_method": str(row[4]),
                "correction_reason": str(row[5]),
            }
        )
    return scopes


def _format_log_rows(*, year_rows: pd.DataFrame) -> pd.DataFrame:
    """Return a clear per row correction log table for one processed year."""
    if year_rows.empty:
        return year_rows.copy()
    frame = year_rows.copy()
    frame["method_detail"] = frame.apply(
        lambda row: _method_detail(
            region=str(row["region"]),
            correction_method=str(row["correction_method"]),
        ),
        axis=1,
    )
    ordered_columns = [
        "year",
        "region",
        "extension",
        "stressor",
        "sector",
        "correction_method",
        "method_detail",
        "correction_reason",
        "original_value",
        "corrected_value",
        "delta",
        "abs_delta",
        "abs_pct_change",
        "fit_window",
    ]
    return frame.loc[:, [col for col in ordered_columns if col in frame.columns]].copy()


def _corrected_values_path_for_source(
    source: str, *, corrected_values_root: Path | None = None
) -> Path:
    """Return raw corrected values path for one supported source."""
    return (corrected_values_root or corrected_values_dir()) / f"{source}_raw_corrected_values.csv"


def load_raw_corrected_value_rows(
    *,
    source: str,
    year: int,
    corrected_values_root: Path | None = None,
) -> pd.DataFrame:
    """Return raw corrected values rows for one source and year."""
    if source not in SUPPORTED_SOURCES:
        return pd.DataFrame()
    path = _corrected_values_path_for_source(source, corrected_values_root=corrected_values_root)
    if not path.exists():
        raise FileNotFoundError(f"Raw corrected values file is missing at {path}.")
    frame = pd.read_csv(path, encoding="utf-8")
    if frame.empty:
        return frame
    filtered = frame.loc[frame["year"].astype(int).eq(int(year))].copy()
    return filtered.reset_index(drop=True)


def _require_extension_frame(iosys: Any, extension_name: str) -> pd.DataFrame:
    """Return one required raw extension ``F`` table from ``iosys``."""
    extension_obj = getattr(iosys, extension_name, None)
    if extension_obj is None:
        raise ValueError(f"Parsed IOSystem is missing extension '{extension_name}'.")
    frame = getattr(extension_obj, "F", None)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"Parsed IOSystem extension '{extension_name}' is missing a DataFrame F.")
    return frame


def _apply_correction_rows_to_iosys(*, iosys: Any, rows: pd.DataFrame) -> None:
    """Apply raw corrected values rows to parsed raw extension tables."""
    if rows.empty:
        return
    for extension_name, ext_rows in rows.groupby("extension", sort=False):
        extension_obj = getattr(iosys, str(extension_name), None)
        frame = _require_extension_frame(iosys, str(extension_name))
        if not all(pd.api.types.is_float_dtype(dtype) for dtype in frame.dtypes):
            frame = frame.astype(float)
            setattr(extension_obj, "F", frame)
        for row in ext_rows.to_dict(orient="records"):
            stressor = str(row["stressor"])
            column_key = (str(row["region"]), str(row["sector"]))
            if stressor not in frame.index:
                raise ValueError(
                    "Parsed IOSystem extension "
                    f"'{extension_name}' is missing stressor '{stressor}'."
                )
            if column_key not in frame.columns:
                raise ValueError(
                    "Parsed IOSystem extension is missing product column "
                    f"{column_key!r} for extension '{extension_name}'."
                )
            frame.loc[stressor, column_key] = float(row["corrected_value"])


def write_applied_correction_log(
    *,
    source_key: str,
    matrix_version: str | None,
    saved_dir: Path,
    year_rows: pd.DataFrame,
) -> Path | None:
    """Persist a detailed correction log for one processed year."""
    if year_rows.empty:
        return None
    del saved_dir
    log_path = _get_mrio_raw_corrected_values_log_path(
        source_key=source_key,
        matrix_version=matrix_version,
    )
    ensure_dir(log_path.parent)
    payload = _format_log_rows(year_rows=year_rows)
    mode = "a" if log_path.exists() else "w"
    payload.to_csv(log_path, mode=mode, index=False, header=mode == "w")
    return log_path


def apply_raw_corrected_values(
    *,
    iosys: Any,
    source: str,
    year: int,
    corrected_values_root: Path | None = None,
) -> AppliedCorrectionSummary | None:
    """Apply EXIOBASE 3.10.2 raw corrected values to one parsed year."""
    rows = load_raw_corrected_value_rows(
        source=source,
        year=year,
        corrected_values_root=corrected_values_root,
    )
    if rows.empty:
        return None
    _apply_correction_rows_to_iosys(iosys=iosys, rows=rows)
    return AppliedCorrectionSummary(
        source=source,
        year=int(year),
        row_count=int(len(rows)),
        log_path=None,
    )
