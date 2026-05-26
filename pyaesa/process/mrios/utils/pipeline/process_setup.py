"""Helper operations used by ``process_mrio`` orchestration."""

from pathlib import Path
from typing import Optional, Sequence, cast

import numpy as np
import pandas as pd

from pyaesa.process.mrios.utils.aggregation.aggregation import (
    WEIGHT_COLUMN,
    agg_map_fingerprint,
    read_agg_map,
)
from pyaesa.process.mrios.utils.io.paths import _get_agg_map_path
from pyaesa.process.mrios.utils.parsers.exio_parser import (
    ExioCharacterizationOptions,
    _build_characterization_jobs,
)
from pyaesa.process.mrios.utils.uncasext_metrics.common import (
    _get_prepared_uncasext_inputs,
)
from .lcia_tracking import extract_lcia_units_from_jobs
from pyaesa.process.mrios.utils.pipeline.contracts import ProcessReportMRIO


def _normalize_lcia_methods(
    lcia_method: Optional[str | Sequence[str]],
) -> Optional[list[str]]:
    """Return normalized LCIA method list or ``None``."""
    if lcia_method is None:
        return None
    if isinstance(lcia_method, str):
        values = [lcia_method]
    else:
        values = list(lcia_method)
        if len(values) == 0:
            return None
    cleaned = [str(method_spec).strip() for method_spec in values]
    cleaned = [method_spec for method_spec in cleaned if method_spec]
    return cleaned or None


def _resolve_aggregation_inputs(
    *,
    source: str,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: Optional[str],
) -> tuple[Optional[Path], Optional[Path], Optional[pd.DataFrame], Optional[pd.DataFrame], dict]:
    """Resolve aggregation files, loaded maps, and metadata payload."""
    agg_reg_path: Optional[Path] = None
    agg_sec_path: Optional[Path] = None
    agg_reg_df: Optional[pd.DataFrame] = None
    agg_sec_df: Optional[pd.DataFrame] = None
    agg_version_required = cast(str, agg_version)

    if agg_reg:
        agg_reg_path = _get_agg_map_path(source, kind="reg", agg_version=agg_version_required)
        if not agg_reg_path.exists():
            raise FileNotFoundError(
                f"Region MRIO aggregation and disaggregation file not found: {agg_reg_path}"
            )
        agg_reg_df = read_agg_map(agg_reg_path)

    if agg_sec:
        agg_sec_path = _get_agg_map_path(source, kind="sec", agg_version=agg_version_required)
        if not agg_sec_path.exists():
            raise FileNotFoundError(
                f"Sector MRIO aggregation and disaggregation file not found: {agg_sec_path}"
            )
        agg_sec_df = read_agg_map(agg_sec_path)

    aggregation_payload = {
        "agg_reg": bool(agg_reg),
        "agg_sec": bool(agg_sec),
        "agg_version": agg_version,
        "agg_reg_file": str(agg_reg_path) if agg_reg else None,
        "agg_sec_file": str(agg_sec_path) if agg_sec else None,
        "agg_reg_weighted": bool(agg_reg_df is not None and WEIGHT_COLUMN in agg_reg_df),
        "agg_sec_weighted": bool(agg_sec_df is not None and WEIGHT_COLUMN in agg_sec_df),
        "agg_reg_fingerprint": (
            agg_map_fingerprint(agg_reg_df) if agg_reg_df is not None else None
        ),
        "agg_sec_fingerprint": (
            agg_map_fingerprint(agg_sec_df) if agg_sec_df is not None else None
        ),
    }
    return agg_reg_path, agg_sec_path, agg_reg_df, agg_sec_df, aggregation_payload


def _resolve_year_characterization_jobs(
    *,
    source: str,
    year_lcia_methods: Sequence[str],
    char_jobs_cache: dict[str, ExioCharacterizationOptions],
) -> tuple[dict[str, ExioCharacterizationOptions], dict[str, dict[str, str]]]:
    """Resolve per year characterization jobs from cache, loading missing methods."""
    missing_lcia_methods = [
        lcia_method for lcia_method in year_lcia_methods if lcia_method not in char_jobs_cache
    ]
    if missing_lcia_methods:
        loaded_jobs = _build_characterization_jobs(
            source_key=source,
            lcia_methods=missing_lcia_methods,
        )
        char_jobs_cache.update(loaded_jobs)

    year_char_jobs = {
        lcia_method: char_jobs_cache[lcia_method]
        for lcia_method in year_lcia_methods
        if lcia_method in char_jobs_cache
    }
    year_lcia_units_by_method = extract_lcia_units_from_jobs(year_char_jobs)
    return year_char_jobs, year_lcia_units_by_method


def _update_report_clipping_stats(report: ProcessReportMRIO, iosys) -> None:
    """Update clipping counters/sums from prepared UNCASExt inputs."""
    prepared = _get_prepared_uncasext_inputs(iosys)
    y_values = prepared.y_fd_raw.to_numpy(dtype=float)
    y_mask = y_values < 0.0
    if bool(y_mask.any()):
        y_abs = np.abs(y_values[y_mask])
        report.y_clip_count += int(y_abs.size)
        report.y_clip_abs_sum += float(y_abs.sum())
        report.y_clip_abs_max = max(report.y_clip_abs_max, float(y_abs.max()))

    f_mask = prepared.gva_by_prod_raw < 0
    f_neg: pd.Series = prepared.gva_by_prod_raw.loc[f_mask]
    if not f_neg.empty:
        f_abs = (-f_neg).astype(float)
        report.f_clip_count += int(f_abs.shape[0])
        report.f_clip_abs_sum += float(f_abs.sum())
        report.f_clip_abs_max = max(report.f_clip_abs_max, float(f_abs.max()))
