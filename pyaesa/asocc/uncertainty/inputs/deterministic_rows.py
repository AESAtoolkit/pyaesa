"""Load deterministic aSoCC final public rows for uncertainty owners."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pyaesa.asocc.runtime.scope.branch_resolution import (
    AsoccDeterministicPathScope,
    asocc_l1_dir,
    asocc_l2_dir,
)
from pyaesa.asocc.runtime.scope.persisted_scope import AsoccPersistedRunScope
from pyaesa.asocc.runtime.request.scope import AsoccScope
from pyaesa.asocc.uncertainty.engine.reuse.prerequisites import AsoccDeterministicPrerequisite
from pyaesa.asocc.uncertainty.schema.public_rows import normalize_asocc_public_row_identity
from pyaesa.shared.figures.lcia_metadata import ensure_frame_lcia_method_metadata
from pyaesa.shared.runtime.io.persisted_paths import normalize_persisted_paths
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens
from pyaesa.shared.tabular.table_io import read_table
from pyaesa.shared.tabular.wide_tables import (
    melt_requested_year_value_rows,
)

ASOCC_VALUE_COLUMN = "allocated_share"
ASOCC_TIME_ROUTE_COLUMN = ASOCC_TIME_ROUTE_PUBLIC_COLUMN


@dataclass(frozen=True)
class AsoccFinalRowScope:
    """Deterministic aSoCC scope metadata used by uncertainty source owners."""

    base_asocc_args: dict[str, Any]
    asocc_scope: AsoccScope
    path_scope: AsoccDeterministicPathScope
    persisted_scopes: tuple[AsoccPersistedRunScope, ...]
    deterministic_manifest_path: Path
    requested_years: list[int]
    final_bucket: str
    reuse_status: str = field(default="computed", kw_only=True)


@dataclass(frozen=True)
class LoadedAsoccFinalRows(AsoccFinalRowScope):
    """Deterministic aSoCC final rows used by uncertainty source owners."""

    rows: pd.DataFrame


def resolve_final_deterministic_asocc_row_scope(
    *,
    prerequisite: AsoccDeterministicPrerequisite,
) -> AsoccFinalRowScope:
    """Resolve deterministic aSoCC row scope metadata without reading result tables."""
    normalized = prerequisite.base_asocc_args
    asocc_scope = prerequisite.asocc_scope
    path_scope = prerequisite.path_scope
    metadata_path = prerequisite.deterministic_manifest_path
    scope_matches = prerequisite.persisted_scope_matches
    requested_years = _requested_years_from_matches(scope_matches=scope_matches)
    _final_root, final_bucket = _final_results_root(
        path_scope=path_scope,
        fu_code=str(normalized["fu_code"]),
    )
    return AsoccFinalRowScope(
        base_asocc_args=normalized,
        asocc_scope=asocc_scope,
        path_scope=path_scope,
        persisted_scopes=tuple(scope for scope, _years in scope_matches),
        deterministic_manifest_path=metadata_path,
        reuse_status=prerequisite.reuse_status,
        requested_years=requested_years,
        final_bucket=final_bucket,
    )


def load_final_deterministic_asocc_rows(
    *,
    prerequisite: AsoccDeterministicPrerequisite,
    row_scope: AsoccFinalRowScope | None = None,
) -> LoadedAsoccFinalRows:
    """Load final deterministic aSoCC rows for one public uncertainty request."""
    scope = row_scope or resolve_final_deterministic_asocc_row_scope(prerequisite=prerequisite)
    final_root, final_bucket = _final_results_root(
        path_scope=scope.path_scope,
        fu_code=str(scope.base_asocc_args["fu_code"]),
    )
    paths = _table_paths_for_scopes(scopes=scope.persisted_scopes, root=final_root)
    frames = [
        read_deterministic_asocc_rows(path=path, requested_years=scope.requested_years)
        for path in paths
    ]
    rows = normalize_asocc_public_row_identity(
        frame=_filter_requested_rows(
            rows=pd.concat(frames, ignore_index=True),
            base_asocc_args=scope.base_asocc_args,
            asocc_scope=scope.asocc_scope,
        )
    ).drop_duplicates(ignore_index=True)
    return LoadedAsoccFinalRows(
        base_asocc_args=scope.base_asocc_args,
        asocc_scope=scope.asocc_scope,
        path_scope=scope.path_scope,
        persisted_scopes=scope.persisted_scopes,
        deterministic_manifest_path=scope.deterministic_manifest_path,
        reuse_status=scope.reuse_status,
        requested_years=scope.requested_years,
        final_bucket=final_bucket,
        rows=rows,
    )


def validate_single_l2_reuse_year_per_identity(
    *,
    rows: pd.DataFrame,
    sampled_identity_columns: tuple[str, ...] = (),
) -> None:
    """Validate inactive workspaceion uncertainty has one L2 reuse year per identity."""
    validate_single_public_axis_per_identity(
        rows=rows,
        axis_column="l2_reuse_year",
        error_message=(
            "Projection uncertainty is required because deterministic aSoCC outputs contain "
            "multiple l2_reuse_year values for the same represented final public row. "
            "Multiple l2_reuse_year values are allowed only when projection uncertainty is "
            "active and samples the L2 reuse year axis."
        ),
        sampled_identity_columns=sampled_identity_columns,
    )


def validate_single_reference_year_per_identity(
    *,
    rows: pd.DataFrame,
    sampled_identity_columns: tuple[str, ...] = (),
) -> None:
    """Validate inactive reference year uncertainty has one reference year per identity."""
    validate_single_public_axis_per_identity(
        rows=rows,
        axis_column="reference_year",
        error_message=(
            "Reference year uncertainty is required because inter-method uncertainty is active "
            "and deterministic aSoCC outputs contain multiple reference_year values for the "
            "same represented final public row after method columns are removed. Multiple "
            "reference_year values are allowed for non inter-method runs, or when reference "
            "year uncertainty is active and samples the reference year axis."
        ),
        sampled_identity_columns=sampled_identity_columns,
    )


def validate_single_public_axis_per_identity(
    *,
    rows: pd.DataFrame,
    axis_column: str,
    error_message: str,
    sampled_identity_columns: tuple[str, ...] = (),
) -> None:
    """Validate an inactive sampled public axis has one value per represented identity."""
    if axis_column not in rows.columns:
        return
    axis = pd.Series(rows.loc[:, axis_column], copy=False)
    identity_columns = [
        column
        for column in rows.columns
        if column not in {ASOCC_VALUE_COLUMN, axis_column, *sampled_identity_columns}
    ]
    work = rows.loc[axis.notna(), [*identity_columns, axis_column]].copy()
    numeric_axis = pd.Series(pd.to_numeric(work.loc[:, axis_column], errors="raise"))
    work[axis_column] = numeric_axis.astype("int64")
    unique = work.drop_duplicates([*identity_columns, axis_column], ignore_index=True)
    if bool(unique.duplicated(identity_columns, keep=False).any()):
        raise ValueError(error_message)


def _final_results_root(
    *,
    path_scope: AsoccDeterministicPathScope,
    fu_code: str,
) -> tuple[Path, str]:
    if str(fu_code).startswith("L2."):
        return (
            asocc_l2_dir(scope=path_scope, bucket="l2_vs_global", lcia_sub=None),
            "l2_vs_global",
        )
    return asocc_l1_dir(scope=path_scope, lcia_sub=None, fu_code=fu_code), "level_1"


def read_deterministic_asocc_rows(*, path: Path, requested_years: list[int]) -> pd.DataFrame:
    """Read one deterministic aSoCC wide table as requested long value rows."""
    frame = ensure_frame_lcia_method_metadata(read_table(path=path))
    melted = melt_requested_year_value_rows(
        frame,
        requested_years=requested_years,
        value_name=ASOCC_VALUE_COLUMN,
    )
    melted["year"] = melted["year"].astype(int)
    return melted


def table_paths_under_deterministic_root(
    *,
    raw_paths: object,
    root: Path,
) -> list[Path]:
    """Return persisted deterministic table paths beneath one requested root."""
    return _table_paths_under_root(raw_paths=raw_paths, root=root)


def _requested_years_from_matches(
    *,
    scope_matches: tuple[tuple[AsoccPersistedRunScope, list[int]], ...],
) -> list[int]:
    return sorted({int(year) for _scope, years in scope_matches for year in years})


def _table_paths_for_scopes(
    *,
    scopes: tuple[AsoccPersistedRunScope, ...],
    root: Path,
) -> list[Path]:
    paths: list[Path] = []
    for scope in scopes:
        paths.extend(
            _table_paths_under_root(
                raw_paths=scope.outputs,
                root=root,
            )
        )
    return sorted(set(paths))


def _filter_requested_rows(
    *,
    rows: pd.DataFrame,
    base_asocc_args: dict[str, Any],
    asocc_scope: AsoccScope,
) -> pd.DataFrame:
    out = rows.copy()
    out = _filter_optional_text(
        frame=out,
        column=ASOCC_SSP_SCENARIO_COLUMN,
        values=normalize_ssp_tokens(base_asocc_args.get("ssp_scenario")),
    )
    out = _filter_optional_text(
        frame=out,
        column="lcia_method",
        values=base_asocc_args.get("lcia_method"),
    )
    out = _filter_optional_text(
        frame=out,
        column="l1_l2_method",
        values=asocc_scope.target_selector_payload.get("methods"),
    )
    out = _filter_optional_int(
        frame=out,
        column="reference_year",
        values=base_asocc_args.get("reference_years"),
    )
    return _filter_optional_int(
        frame=out,
        column="l2_reuse_year",
        values=base_asocc_args.get("l2_reuse_years"),
    )


def _filter_optional_text(
    *,
    frame: pd.DataFrame,
    column: str,
    values: list[str] | None,
) -> pd.DataFrame:
    if not values or column not in frame.columns:
        return frame
    allowed = {str(value) for value in values}
    series = pd.Series(frame.loc[:, column], copy=False)
    mask = series.isna() | series.astype(str).isin(allowed)
    return frame.loc[mask].reset_index(drop=True)


def _filter_optional_int(
    *,
    frame: pd.DataFrame,
    column: str,
    values: list[int] | None,
) -> pd.DataFrame:
    if values is None or column not in frame.columns:
        return frame
    allowed = {int(value) for value in values}
    series = pd.Series(frame.loc[:, column], copy=False)
    numeric = pd.Series(pd.to_numeric(series, errors="raise"), index=frame.index)
    mask = series.isna() | numeric.isin(sorted(allowed))
    return frame.loc[mask].reset_index(drop=True)


def _table_paths_under_root(*, raw_paths: object, root: Path) -> list[Path]:
    root_path = Path(root).resolve()
    paths: list[Path] = []
    for path in normalize_persisted_paths(raw_paths=raw_paths):
        resolved = path.resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError:
            continue
        paths.append(resolved)
    return sorted(paths)
