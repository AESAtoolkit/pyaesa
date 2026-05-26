"""Processed MRIO contract loaders used by IO-LCA and figure generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.asocc.data.enacting_metric_units import lcia_unit_series_for_method
from pyaesa.asocc.data.lcia_status import resolve_lcia_status
from pyaesa.asocc.data.load_mrio import (
    _load_enacting_metric_l2_metric,
    _load_lcia_l1_metric,
    _load_lcia_l2_metric,
    _load_utility_metric,
)
from pyaesa.asocc.io.pickle_io import read_pickle
from pyaesa.asocc.methods.lcia_inputs import (
    aggregate_frame_to_parent,
    load_impact_parent_mapping,
)
from pyaesa.asocc.orchestration.setup.formatting.formatting import _process_mrio_hint
from pyaesa.process.mrios.utils.io.metadata import _get_year_entry, _read_metadata
from pyaesa.process.mrios.utils.io.paths import (
    _get_metadata_path,
    _get_year_saved_path,
)

from pyaesa.io_lca.contracts.fu_mapping import IOLCAFUSpec
from pyaesa.io_lca.data.writers import read_table


@dataclass(frozen=True)
class YearMethodMainPayload:
    """Main IO-LCA payload loaded for one ``(year, lcia_method)`` pair."""

    year: int
    lcia_method: str
    saved_dir: Path
    year_entry: dict[str, Any]
    metric: pd.DataFrame
    unit_by_impact: pd.Series


@dataclass(frozen=True)
class UpstreamPayload:
    """Structural payload required for one upstream decomposition."""

    a_matrix: pd.DataFrame
    l_matrix: pd.DataFrame
    s_matrix: pd.DataFrame
    driver_matrix: pd.DataFrame
    fy_matrix: pd.DataFrame | None


def impact_unit_text_map(*, unit_by_impact: pd.Series) -> dict[str, str]:
    """Return deterministic string keyed impact unit mapping."""
    return {str(impact): str(unit) for impact, unit in unit_by_impact.items()}


def _align_matrix_axis_names_to_driver(
    *,
    frame: pd.DataFrame,
    driver_index: pd.Index,
) -> pd.DataFrame:
    """Align matrix axis names to the active upstream driver product index.

    Upstream algebra is index value based. For consistent downstream labeling,
    align A/L/S axis level names to the driver product axis names when shapes
    are compatible.
    """
    out = frame.copy()
    if isinstance(driver_index, pd.MultiIndex):
        driver_names = [str(name) for name in driver_index.names]
        if isinstance(out.index, pd.MultiIndex) and out.index.nlevels == driver_index.nlevels:
            out.index = out.index.set_names(driver_names)
        if isinstance(out.columns, pd.MultiIndex) and out.columns.nlevels == driver_index.nlevels:
            out.columns = out.columns.set_names(driver_names)
    return out


def load_domain_metadata(
    *,
    source: str,
    agg_version: str | None,
) -> tuple[dict[str, Any], Path]:
    """Load processed MRIO metadata payload and physical path."""
    meta = _read_metadata(source, matrix_version=agg_version)
    meta_path = _get_metadata_path(source, matrix_version=agg_version)
    return meta, meta_path


def _missing_outputs_error(
    *,
    source: str,
    years: list[int],
    agg_version: str | None,
    agg_reg: bool,
    agg_sec: bool,
    lcia_methods: list[str],
    detail: str,
) -> ValueError:
    """Build process_mrio rerun error with deterministic hint text."""
    hint = _process_mrio_hint(
        source=source,
        years=sorted({int(year) for year in years}),
        agg_version=agg_version,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        lcia_methods=lcia_methods,
        keep_intermediate_uncasext=True,
    )
    return ValueError(
        f"{detail} Re-run process_mrio with the required intermediate outputs. "
        f"Suggested call: {hint}"
    )


def _require_year_entry(
    *,
    metadata: dict[str, Any],
    metadata_path: Path,
    year: int,
) -> dict[str, Any]:
    """Return strict metadata year entry."""
    year_entry = _get_year_entry(metadata, year)
    if not isinstance(year_entry, dict):
        raise ValueError(
            "Processed MRIO prerequisite does not contain a valid metadata entry "
            f"for requested year {int(year)} in {metadata_path}."
        )
    return year_entry


def _load_lcia_metric(
    *,
    saved_dir: Path,
    lcia_method: str,
    matrix_key: str,
) -> pd.DataFrame:
    """Load one LCIA metric by key."""
    if matrix_key in {"e_cba_fd_reg", "e_pba_reg"}:
        return _load_lcia_l1_metric(saved_dir, lcia_method, matrix_key)
    return _load_lcia_l2_metric(saved_dir, lcia_method, matrix_key)


def load_main_payload(
    *,
    source: str,
    agg_version: str | None,
    agg_reg: bool,
    agg_sec: bool,
    metadata: dict[str, Any],
    metadata_path: Path,
    year: int,
    lcia_method: str,
    fu_spec: IOLCAFUSpec,
) -> tuple[YearMethodMainPayload | None, str | None]:
    """Load main per year payload and apply LCIA availability gate.

    Returns:
        ``(payload, None)`` when available.
        ``(None, reason)`` when the LCIA method is explicitly unavailable and should be
        skipped for this year.
    """
    year_entry = _require_year_entry(
        metadata=metadata,
        metadata_path=metadata_path,
        year=year,
    )
    available, reason = resolve_lcia_status(year_entry, lcia_method)
    if not available:
        return None, (reason or "LCIA unavailable")

    saved_dir = _get_year_saved_path(source, year, matrix_version=agg_version)
    if not saved_dir.exists():
        raise _missing_outputs_error(
            source=source,
            years=[year],
            agg_version=agg_version,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
            lcia_methods=[lcia_method],
            detail=f"Processed MRIO year directory is missing: {saved_dir}.",
        )
    try:
        metric = _load_lcia_metric(
            saved_dir=saved_dir,
            lcia_method=lcia_method,
            matrix_key=fu_spec.lcia_matrix_key,
        )
    except FileNotFoundError as exc:
        raise _missing_outputs_error(
            source=source,
            years=[year],
            agg_version=agg_version,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
            lcia_methods=[lcia_method],
            detail=f"Processed LCIA output is missing on disk for year {year}.",
        ) from exc
    impact_parent_map = load_impact_parent_mapping(source=source, lcia_method=lcia_method)
    metric = aggregate_frame_to_parent(metric, impact_parent_map)
    unit_by_impact = lcia_unit_series_for_method(
        year_entry=year_entry,
        year=year,
        lcia_method=lcia_method,
    )
    return YearMethodMainPayload(
        year=int(year),
        lcia_method=str(lcia_method),
        saved_dir=saved_dir,
        year_entry=year_entry,
        metric=metric,
        unit_by_impact=unit_by_impact,
    ), None


def _read_processed_frame(path: Path) -> pd.DataFrame:
    """Load one package generated processed MRIO frame."""
    return cast(pd.DataFrame, read_pickle(path))


def _load_fy_matrix(
    *,
    saved_dir: Path,
    lcia_method: str,
    selected_axis_name: str | None = None,
) -> pd.DataFrame:
    """Load and normalize ``F_Y`` matrix to impact by region DataFrame."""
    fy_path = saved_dir / "enacting_metrics" / "level_1" / lcia_method / "F_Y.pickle"
    if not fy_path.exists():
        raise FileNotFoundError(
            "Processed IO-LCA upstream analysis requires the level-1 F_Y pickle "
            f"for lcia_method='{lcia_method}'. Missing file: {fy_path}. "
            "Re-run process_mrio with keep_intermediate_uncasext=True for this source, "
            "year, aggregation scope, and LCIA method."
        )
    fy_raw = _read_processed_frame(fy_path)
    if isinstance(fy_raw.columns, pd.MultiIndex):
        col_names = [str(name) for name in fy_raw.columns.names]
        candidates = []
        if selected_axis_name:
            candidates.append(str(selected_axis_name))
        candidates.extend(["region", "r_f", "r_p", "r_c", "r_u", "r_y"])
        selected_level_name = next((name for name in candidates if name in col_names), None)
        if selected_level_name is None:
            if len(col_names) == 1:
                level = 0
            else:
                raise ValueError(
                    "Processed output contract violation: F_Y columns must expose a resolvable "
                    f"region axis. Got {col_names} at {fy_path}."
                )
        else:
            level = col_names.index(selected_level_name)
        # pandas>=3 removed DataFrame.groupby(..., axis=1), so aggregate on
        # transposed columns and transpose back.
        fy_grouped = cast(pd.DataFrame, fy_raw.T.groupby(level=level).sum(min_count=1))
        fy_region = cast(pd.DataFrame, fy_grouped.T)
    else:
        fy_region = fy_raw
    fy_region.columns = fy_region.columns.astype(str)
    fy_region.columns.name = "region"
    return fy_region


def _load_upstream_driver_matrix(
    *,
    saved_dir: Path,
    fu_spec: IOLCAFUSpec,
) -> pd.DataFrame:
    """Load the canonical upstream driver matrix for one FU contract."""
    if fu_spec.upstream_driver == "y_fd":
        return cast(pd.DataFrame, _load_enacting_metric_l2_metric(saved_dir, "fd_rp_sp_rf"))
    return _load_utility_metric(saved_dir, "x_to_rc")


def load_upstream_payload(
    *,
    source: str,
    saved_dir: Path,
    lcia_method: str,
    fu_spec: IOLCAFUSpec,
) -> UpstreamPayload:
    """Load structural matrices required for upstream analysis."""
    a_matrix = _read_processed_frame(saved_dir / "A.pickle")
    l_matrix = _read_processed_frame(saved_dir / "L.pickle")
    s_matrix = _read_processed_frame(saved_dir / "extensions" / lcia_method / "S.pickle")
    driver_matrix = _load_upstream_driver_matrix(saved_dir=saved_dir, fu_spec=fu_spec)
    a_matrix = _align_matrix_axis_names_to_driver(frame=a_matrix, driver_index=driver_matrix.index)
    l_matrix = _align_matrix_axis_names_to_driver(frame=l_matrix, driver_index=driver_matrix.index)
    s_matrix = _align_matrix_axis_names_to_driver(frame=s_matrix, driver_index=driver_matrix.index)
    impact_parent_map = load_impact_parent_mapping(source=source, lcia_method=lcia_method)
    s_matrix = aggregate_frame_to_parent(s_matrix, impact_parent_map)
    selected_fy_axis = fu_spec.selector_axes[0] if fu_spec.selector_axes else None
    fy_matrix = (
        _load_fy_matrix(
            saved_dir=saved_dir,
            lcia_method=lcia_method,
            selected_axis_name=selected_fy_axis,
        )
        if fu_spec.fy_relevant
        else None
    )
    if fy_matrix is not None:
        fy_matrix = aggregate_frame_to_parent(fy_matrix, impact_parent_map)
    return UpstreamPayload(
        a_matrix=a_matrix,
        l_matrix=l_matrix,
        s_matrix=s_matrix,
        driver_matrix=driver_matrix,
        fy_matrix=fy_matrix,
    )


def load_io_lca_method_table(
    *,
    path: Path,
) -> pd.DataFrame:
    """Load one IO-LCA LCIA method result table based on file extension."""
    return read_table(path)
