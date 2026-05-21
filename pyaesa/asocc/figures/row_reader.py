"""Per scope deterministic aSoCC figure row reading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pyaesa.external_inputs.asocc.schema.contracts import iter_external_method_selections
from pyaesa.external_inputs.asocc.deterministic.files import (
    describe_expected_external_deterministic_stems,
    expected_external_deterministic_stems,
    load_external_deterministic_rows,
)
from pyaesa.shared.lcia.path_tokens import infer_lcia_method_from_path
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
)
from pyaesa.shared.figures.scenario_scopes import repeat_invariant_rows_into_scenarios
from pyaesa.shared.tabular.l2_reuse_years import canonicalize_l2_reuse_year_column
from pyaesa.shared.tabular.scalars import is_display_missing
from pyaesa.shared.tabular.table_io import read_table
from pyaesa.shared.tabular.wide_tables import melt_requested_year_value_rows

from .scope_planner import RunScope, scoped_output_paths


@dataclass(frozen=True)
class FigureRows:
    """Loaded deterministic figure rows for one persisted table."""

    frame: pd.DataFrame


def load_figure_rows(
    *,
    scope: RunScope,
    fu_code: str,
    requested_years: list[int],
    lcia_methods: list[str] | None,
    ssp_scenarios: list[str | None],
    compute_signature: dict[str, Any],
    output_paths: list[str],
    figure_external_method: dict[str, Any] | None,
) -> pd.DataFrame:
    """Load only rows needed for one final deterministic figure request."""
    paths = [
        path
        for path in scoped_output_paths(scope=scope, fu_code=fu_code, output_paths=output_paths)
        if _path_matches_requested_lcia(path=path, lcia_methods=lcia_methods)
    ]
    frames = [
        loaded.frame
        for path in paths
        for loaded in [
            read_native_rows(
                path=path,
                fu_code=fu_code,
                requested_years=requested_years,
            )
        ]
        if not loaded.frame.empty
    ]
    frames.extend(
        load_external_rows(
            scope=scope,
            fu_code=fu_code,
            requested_years=requested_years,
            lcia_methods=lcia_methods,
            figure_external_method=figure_external_method,
        )
    )
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    scoped = normalize_ssp_rows(frame, ssp_scenarios=ssp_scenarios)
    return filter_requested_methods(
        frame=scoped,
        compute_signature=compute_signature,
        fu_code=fu_code,
    )


def read_native_rows(
    *,
    path: Path,
    fu_code: str,
    requested_years: list[int],
) -> FigureRows:
    """Read one native output table into long deterministic figure rows."""
    raw = canonicalize_l2_reuse_year_column(read_table(path=path), path=path)
    lcia_method = infer_lcia_method_from_path(path)
    method = method_label_from_frame(raw)
    if lcia_method is not None:
        raw["lcia_method"] = lcia_method
    raw["fu_code"] = str(fu_code).strip()
    raw["__method"] = method
    raw["__external_method"] = False
    raw["__source_path"] = str(path)
    long_frame = melt_requested_year_value_rows(raw, requested_years=requested_years)
    return FigureRows(long_frame)


def _path_matches_requested_lcia(*, path: Path, lcia_methods: list[str] | None) -> bool:
    if not lcia_methods:
        return True
    lcia_method = infer_lcia_method_from_path(path)
    if lcia_method is None:
        return True
    requested = {str(method).strip() for method in lcia_methods if str(method).strip()}
    return str(lcia_method).strip() in requested


def load_external_rows(
    *,
    scope: RunScope,
    fu_code: str,
    requested_years: list[int],
    lcia_methods: list[str] | None,
    figure_external_method: dict[str, Any] | None,
) -> list[pd.DataFrame]:
    """Load requested external deterministic aSoCC rows for figure rendering."""
    if figure_external_method is None:
        return []
    frames: list[pd.DataFrame] = []
    for selection in iter_external_method_selections(
        external_method=figure_external_method,
        fu_code=fu_code,
    ):
        raw = load_external_deterministic_rows(
            proj_base=scope.proj_base,
            selection=selection,
            years=requested_years,
            lcia_methods=lcia_methods,
            ssp_scenario_options_by_year=None,
        )
        if raw is None:
            stems = expected_external_deterministic_stems(
                selection=selection,
                lcia_methods=lcia_methods,
                years=requested_years,
                ssp_scenario_options_by_year=None,
            )
            message = describe_expected_external_deterministic_stems(
                proj_base=scope.proj_base,
                selection=selection,
                stems=stems,
            )
            raise ValueError(
                f"Missing deterministic external aSoCC file for '{selection.user_label}'. "
                f"{message}."
            )
        raw["__method"] = str(selection.asocc_method_label)
        raw["__external_method"] = True
        raw["fu_code"] = str(fu_code).strip()
        raw[ASOCC_SSP_SCENARIO_COLUMN] = raw.get(ASOCC_SSP_SCENARIO_COLUMN, pd.NA)
        frames.append(raw)
    return frames


def filter_requested_methods(
    *,
    frame: pd.DataFrame,
    compute_signature: dict[str, Any],
    fu_code: str,
) -> pd.DataFrame:
    """Keep requested native methods and all explicitly requested external methods."""
    selected = compute_signature["selected_methods"]
    if str(fu_code).startswith("L1."):
        requested = {str(value) for value in selected.get("l1", [])}
    else:
        requested = {
            *(str(value) for value in selected.get("l2_vs_global", [])),
            *(str(value) for value in selected.get("l2_in_l1", [])),
        }
    external = frame["__external_method"].astype(bool)
    requested_method = frame["__method"].astype(str).isin(sorted(requested))
    return frame.loc[external | requested_method].copy()


def normalize_ssp_rows(frame: pd.DataFrame, *, ssp_scenarios: list[str | None]) -> pd.DataFrame:
    """Normalize row owned SSP labels and plan final SSP figure scopes."""
    requested = tuple(str(ssp).upper() for ssp in ssp_scenarios if ssp is not None)
    if not requested:
        return frame
    normalized = frame.copy()
    scenario_series = pd.Series(normalized.loc[:, ASOCC_SSP_SCENARIO_COLUMN], copy=False)
    scenario_mask = ~scenario_series.map(is_display_missing)
    if bool(scenario_mask.any()):
        normalized.loc[scenario_mask, ASOCC_SSP_SCENARIO_COLUMN] = (
            scenario_series.loc[scenario_mask].astype(str).str.upper()
        )
    scoped = repeat_invariant_rows_into_scenarios(
        normalized,
        scenario_column=ASOCC_SSP_SCENARIO_COLUMN,
        scope_column="__figure_ssp_scope",
        requested_scenarios=requested,
        identity_excluded_columns={"asocc"},
    )
    return pd.concat(scoped, ignore_index=True) if scoped else normalized.iloc[0:0].copy()


def method_label_from_frame(frame: pd.DataFrame) -> str:
    """Return the visible aSoCC method label for one persisted frame."""
    column = next(
        column for column in ("l1_l2_method", "l1_method", "l2_method") if column in frame
    )
    return visible_values(frame, column)[0]


def visible_values(frame: pd.DataFrame, column: str) -> list[str]:
    """Return nonmissing display values from one column."""
    return sorted(
        {
            str(value).strip()
            for value in frame[column].tolist()
            if not is_display_missing(value) and str(value).strip()
        }
    )
