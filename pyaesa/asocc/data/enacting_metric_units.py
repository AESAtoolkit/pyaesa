"""Load enacting metric unit mappings from processed MRIO metadata."""

from typing import Any, Mapping, cast

import pandas as pd

from pyaesa.process.mrios.utils.io.metadata import (
    _get_year_entry,
    _read_metadata,
)


def _parse_unit_map(
    raw_map: Any,
    *,
    label: str,
) -> dict[str, str]:
    """Normalize one package-owned ``name -> unit`` mapping."""
    del label
    return {str(raw_key).strip(): str(raw_unit).strip() for raw_key, raw_unit in raw_map.items()}


def _units_payload_from_year_entry(
    *,
    year_entry: Mapping[str, Any],
    year: int,
) -> Mapping[str, Any]:
    """Return units payload for one package-owned metadata year entry."""
    del year
    return cast(Mapping[str, Any], year_entry["enacting_metrics"]["units"])


def parse_enacting_metric_units_from_year_entry(
    *,
    year_entry: Mapping[str, Any],
    year: int,
) -> tuple[str, dict[str, str], dict[str, pd.Series]]:
    """Parse MRIO and LCIA unit mappings from one metadata year entry."""
    payload = _units_payload_from_year_entry(year_entry=year_entry, year=year)
    default_unit = str(payload["mrio_default_monetary"]).strip()
    mrio_by_metric = _parse_unit_map(
        payload["mrio_by_metric"],
        label=f"enacting_metrics.units.mrio_by_metric (year={year})",
    )
    raw_lcia = payload.get("lcia_by_method", {})
    lcia_by_method: dict[str, pd.Series] = {}
    for raw_lcia_method, raw_map in raw_lcia.items():
        lcia_method = str(raw_lcia_method).strip()
        method_map = _parse_unit_map(
            raw_map,
            label=(f"enacting_metrics.units.lcia_by_method['{lcia_method}'] (year={year})"),
        )
        series = pd.Series(method_map, dtype=str).sort_index()
        series.index = series.index.astype(str)
        series.name = "unit"
        lcia_by_method[lcia_method] = series
    return default_unit, mrio_by_metric, lcia_by_method


def lcia_unit_series_for_method(
    *,
    year_entry: Mapping[str, Any],
    year: int,
    lcia_method: str,
) -> pd.Series:
    """Return LCIA ``impact_parent -> unit`` mapping for one LCIA method/year."""
    lcia_method = str(lcia_method).strip()
    _, _, lcia_by_method = parse_enacting_metric_units_from_year_entry(
        year_entry=year_entry,
        year=year,
    )
    series = lcia_by_method.get(lcia_method)
    if series is None:
        raise ValueError(
            f"Processed MRIO enacting metric units do not include LCIA method '{lcia_method}' "
            f"for year {year}."
        )
    return series


def load_enacting_metric_units_from_metadata(
    *,
    source: str,
    matrix_version: str | None,
    years: list[int],
) -> tuple[str, dict[str, str], dict[str, pd.Series]]:
    """Load merged enacting metric unit mappings for a source/domain/year set."""
    if not years:
        raise ValueError("Cannot load MRIO enacting metric units: years is empty.")
    metadata = _read_metadata(source, matrix_version=matrix_version)
    merged_default: str | None = None
    merged_mrio: dict[str, str] = {}
    merged_lcia: dict[str, pd.Series] = {}
    for year in sorted({int(y) for y in years}):
        year_entry = _get_year_entry(metadata, year)
        if year_entry is None:
            raise ValueError(
                f"Processed MRIO prerequisite is missing year {year}. "
                "Re-run process_mrio for this domain."
            )
        default_unit, mrio_by_metric, lcia_by_method = parse_enacting_metric_units_from_year_entry(
            year_entry=year_entry,
            year=year,
        )
        if merged_default is None:
            merged_default = default_unit
        elif merged_default != default_unit:
            raise ValueError(
                "Inconsistent MRIO monetary unit across years: "
                f"'{merged_default}' vs '{default_unit}' (year={year})."
            )
        for metric, unit in mrio_by_metric.items():
            existing = merged_mrio.get(metric)
            if existing is not None and existing != unit:
                raise ValueError(
                    "Inconsistent MRIO unit mapping across years for "
                    f"metric '{metric}': '{existing}' vs '{unit}' (year={year})."
                )
            merged_mrio[metric] = unit
        for lcia_method, series in lcia_by_method.items():
            existing_series = merged_lcia.get(lcia_method)
            if existing_series is not None and not existing_series.equals(series):
                raise ValueError(
                    "Inconsistent LCIA unit mapping across years for method "
                    f"'{lcia_method}' (year={year})."
                )
            merged_lcia[lcia_method] = series
    return cast(str, merged_default), merged_mrio, merged_lcia
