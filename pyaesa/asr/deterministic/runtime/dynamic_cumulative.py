"""Dynamic deterministic ASR cumulative output assembly."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.scenario.columns import ASOCC_TIME_ROUTE_PUBLIC_COLUMN
from pyaesa.shared.tabular.l2_reuse_years import (
    canonicalize_l2_reuse_year_column,
    frame_l2_reuse_years,
)

from .tables import write_asr_output


@dataclass
class PendingDynamicAsrOutput:
    """One dynamic ASR output waiting for full-period cumulative metrics."""

    path: Path
    relative_parent: Path
    base_stem: str
    frame: pd.DataFrame
    year_cols: list[str]


def write_dynamic_asr_outputs(
    *,
    outputs: list[PendingDynamicAsrOutput],
    fmt: str,
) -> pd.DataFrame | None:
    """Write dynamic ASR outputs with full period cumulative metrics."""
    component_frames: list[pd.DataFrame] = []
    for group in _dynamic_output_groups(outputs):
        components = _attach_group_cumulative_metrics(group)
        component_frames.append(components)
    for output in outputs:
        write_asr_output(_public_dynamic_frame(output.frame), output.path, fmt)
    return pd.concat(component_frames, ignore_index=True) if component_frames else None


def dynamic_group_parent(relative_output_dir: Path) -> Path:
    """Return the grouping parent shared by dynamic branch companion outputs."""
    parts = [
        part
        for part in relative_output_dir.parts
        if part not in {"", ".", "historical_reuse", "regression_proj"}
    ]
    return Path(*parts) if parts else Path(".")


def _dynamic_output_groups(
    outputs: list[PendingDynamicAsrOutput],
) -> list[list[PendingDynamicAsrOutput]]:
    grouped: dict[tuple[Path, str], list[PendingDynamicAsrOutput]] = {}
    for output in outputs:
        grouped.setdefault((output.relative_parent, output.base_stem), []).append(output)
    return [grouped[key] for key in sorted(grouped)]


def _attach_group_cumulative_metrics(
    outputs: list[PendingDynamicAsrOutput],
) -> pd.DataFrame:
    normalized = [
        (output, canonicalize_l2_reuse_year_column(output.frame, path=output.path))
        for output in outputs
    ]
    l2_reuse_years = _dynamic_l2_reuse_years(normalized)
    scenario_values = _dynamic_scenario_values([frame for _output, frame in normalized])
    components = pd.concat(
        _dynamic_component_rows_for_group(
            normalized,
            l2_reuse_years=l2_reuse_years,
            scenario_values=scenario_values,
        ),
        ignore_index=True,
    )
    cumulative = _dynamic_cumulative_metrics(components)
    for output, frame in normalized:
        _attach_cumulative_to_output(
            output=output,
            normalized_frame=frame,
            cumulative=cumulative,
            l2_reuse_years=l2_reuse_years,
            scenario_values=scenario_values,
        )
    return components


def _dynamic_component_rows_for_group(
    normalized: list[tuple[PendingDynamicAsrOutput, pd.DataFrame]],
    *,
    l2_reuse_years: list[int],
    scenario_values: dict[str, list[str]],
) -> list[pd.DataFrame]:
    return [
        _melt_dynamic_components(
            frame=_repeat_invariant_dynamic_rows(
                frame=frame,
                l2_reuse_years=l2_reuse_years,
                scenario_values=scenario_values,
            ),
            year_cols=output.year_cols,
        )
        for output, frame in normalized
    ]


def _dynamic_l2_reuse_years(
    normalized: list[tuple[PendingDynamicAsrOutput, pd.DataFrame]],
) -> list[int]:
    """Return L2 reuse years represented by companion dynamic outputs."""
    return sorted(
        {
            int(l2_reuse_year)
            for _output, frame in normalized
            for l2_reuse_year in frame_l2_reuse_years(frame)
        }
    )


def _repeat_invariant_dynamic_rows(
    *,
    frame: pd.DataFrame,
    l2_reuse_years: list[int],
    scenario_values: dict[str, list[str]],
) -> pd.DataFrame:
    """Repeat scenario-invariant historical rows into L2 reuse year identities."""
    repeated = [frame]
    if l2_reuse_years and "l2_reuse_year" not in frame.columns:
        repeated = [
            _with_column_value(copy, column="l2_reuse_year", value=value)
            for value in l2_reuse_years
            for copy in repeated
        ]
    for column, values in scenario_values.items():
        next_repeated: list[pd.DataFrame] = []
        for copy in repeated:
            if _frame_has_visible_values(copy, column=column):
                next_repeated.append(copy)
                continue
            source = _matching_dynamic_scenario_series(copy, values=values)
            if source is not None:
                next_repeated.append(_with_column_series(copy, column=column, values=source))
                continue
            next_repeated.extend(
                _with_column_value(copy, column=column, value=value) for value in values
            )
        repeated = next_repeated
    return pd.concat(repeated, ignore_index=True) if len(repeated) > 1 else repeated[0]


def _matching_dynamic_scenario_series(
    frame: pd.DataFrame,
    *,
    values: list[str],
) -> pd.Series | None:
    """Return an existing SSP series that can define one missing SSP axis."""
    allowed = {str(value).strip() for value in values if str(value).strip()}
    for column in ("ar6_cc_ssp_scenario", "asocc_ssp_scenario", "lca_ssp_scenario"):
        if column not in frame.columns:
            continue
        series = pd.Series(frame[column], copy=False).astype("string").str.strip()
        present = series.notna() & series.ne("")
        if not bool(present.all()):
            continue
        unique = {str(value) for value in series.loc[present].tolist()}
        if unique and unique.issubset(allowed):
            return series.astype(object)
    return None


def _dynamic_scenario_values(frames: list[pd.DataFrame]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for column in ("ar6_cc_ssp_scenario", "asocc_ssp_scenario", "lca_ssp_scenario"):
        observed = [
            str(value).strip()
            for frame in frames
            if column in frame.columns
            for value in frame[column].dropna().astype(str).tolist()
            if str(value).strip()
        ]
        unique = list(dict.fromkeys(observed))
        if unique:
            values[column] = unique
    return values


def _frame_has_visible_values(frame: pd.DataFrame, *, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = [str(value).strip() for value in frame[column].dropna().astype(str).tolist()]
    return any(values)


def _with_column_value(frame: pd.DataFrame, *, column: str, value: object) -> pd.DataFrame:
    out = frame.copy()
    out[column] = value
    return out


def _with_column_series(
    frame: pd.DataFrame,
    *,
    column: str,
    values: pd.Series,
) -> pd.DataFrame:
    out = frame.copy()
    out[column] = values.to_numpy(dtype=object, copy=True)
    return out


def _melt_dynamic_components(*, frame: pd.DataFrame, year_cols: list[str]) -> pd.DataFrame:
    lca_columns = [f"lca_{year}" for year in year_cols]
    acc_columns = [f"acc_{year}" for year in year_cols]
    helper_columns = {*lca_columns, *acc_columns}
    metadata = frame.drop(columns=[*year_cols, *helper_columns], errors="ignore")
    row_positions = np.tile(np.arange(len(frame), dtype=np.int64), len(year_cols))
    year_positions = np.repeat(np.arange(len(year_cols), dtype=np.int64), len(frame))
    out = metadata.iloc[row_positions].reset_index(drop=True)
    out["year"] = np.asarray(year_cols, dtype=np.int64)[year_positions]
    out["__lca_component"] = frame.loc[:, lca_columns].to_numpy(dtype=np.float64).T.reshape(-1)
    out["__acc_component"] = frame.loc[:, acc_columns].to_numpy(dtype=np.float64).T.reshape(-1)
    return out


def _dynamic_cumulative_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    present = (
        pd.Series(frame["__lca_component"], copy=False).notna()
        & pd.Series(frame["__acc_component"], copy=False).notna()
    )
    work = frame.loc[present].copy()
    identity_columns = _dynamic_cumulative_identity_columns(work)
    summed = cast(
        pd.DataFrame,
        work.groupby(identity_columns, dropna=False)[["__lca_component", "__acc_component"]].sum(),
    )
    totals = summed.reset_index()
    acc_values = totals["__acc_component"].to_numpy(dtype=np.float64)
    lca_values = totals["__lca_component"].to_numpy(dtype=np.float64)
    totals["cumulative_asr"] = np.divide(
        lca_values,
        acc_values,
        out=np.full(len(totals), np.nan, dtype=np.float64),
        where=acc_values != 0.0,
    )
    return totals.drop(columns=["__lca_component", "__acc_component"])


def _dynamic_cumulative_identity_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "year",
        "__lca_component",
        "__acc_component",
        "asocc_ssp_start_year",
        "lca_ssp_start_year",
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        "cumulative_asr",
    }
    return [column for column in frame.columns if column not in excluded]


def _attach_cumulative_to_output(
    *,
    output: PendingDynamicAsrOutput,
    normalized_frame: pd.DataFrame,
    cumulative: pd.DataFrame,
    l2_reuse_years: list[int],
    scenario_values: dict[str, list[str]],
) -> None:
    identity_columns = _dynamic_cumulative_identity_columns(cumulative)
    materialized = _repeat_invariant_dynamic_rows(
        frame=normalized_frame,
        l2_reuse_years=l2_reuse_years,
        scenario_values=scenario_values,
    )
    if any(column not in materialized.columns for column in identity_columns):
        return
    merged = materialized.loc[:, identity_columns].merge(
        cumulative,
        how="left",
        on=identity_columns,
        validate="many_to_one",
    )
    values = pd.Series(merged["cumulative_asr"], copy=False)
    output.frame = materialized.reset_index(drop=True)
    output.frame["cumulative_asr"] = values.to_numpy(dtype=np.float64)


def _public_dynamic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    helper_columns = [
        column
        for column in frame.columns
        if (
            str(column).startswith("lca_")
            and str(column)[4:].isdigit()
            or str(column).startswith("acc_")
            and str(column)[4:].isdigit()
        )
    ]
    out = frame.drop(columns=helper_columns)
    year_columns = [column for column in out.columns if str(column).isdigit()]
    cumulative_columns = [column for column in ("cumulative_asr",) if column in out.columns]
    leading = [
        column for column in out.columns if column not in {*year_columns, *cumulative_columns}
    ]
    return out.loc[:, [*leading, *cumulative_columns, *year_columns]]
