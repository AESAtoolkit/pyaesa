"""Global AR6 CC figure rows used by dynamic ASR figures."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.ar6_cc.deterministic.figures.period_panels import (
    combine_study_and_post_tables,
    figure_year_columns,
)
from pyaesa.ar6_cc.uncertainty.request.normalization import AR6_DYNAMIC_CC_SOURCE
from pyaesa.ar6_cc.uncertainty.io.artifacts import (
    ar6_cc_run_paths_from_manifest,
)
from pyaesa.asr.figures.common import visible_values
from pyaesa.shared.figures.uncertainty_run_values import (
    collect_selected_compact_run_values,
)
from pyaesa.shared.lcia.units import try_unit_conversion
from pyaesa.shared.runtime.metadata.json import read_json_dict
from pyaesa.shared.runtime.scenario.columns import AR6_CC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.table_io import read_table
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
    read_manifest,
)

_GLOBAL_AR6_PERIOD_PANEL_TITLE_PAD = 24
_GLOBAL_AR6_NO_PERIOD_PANEL_TITLE_PAD = 8


@dataclass(frozen=True)
class DeterministicGlobalAR6Rows:
    """Prepared deterministic global AR6 CC rows for one ASR figure scope."""

    frame: pd.DataFrame
    study_years: list[int]
    post_years: list[int]


@dataclass(frozen=True)
class DeterministicGlobalAR6Source:
    """Deterministic global AR6 CC source tables shared by ASR figure scopes."""

    study: pd.DataFrame
    post: pd.DataFrame | None


@dataclass(frozen=True)
class UncertaintyGlobalAR6Rows:
    """Prepared uncertainty global AR6 CC rows for one ASR figure scope."""

    summary: pd.DataFrame
    budget: pd.DataFrame
    deterministic: DeterministicGlobalAR6Rows | None
    study_years: list[int]
    post_years: list[int]
    pair_count: int


@dataclass(frozen=True)
class UncertaintyGlobalAR6Source:
    """Uncertainty global AR6 CC source tables shared by ASR figure scopes."""

    deterministic: DeterministicGlobalAR6Source | None
    summary: pd.DataFrame
    post_summary: pd.DataFrame
    budget_identity: pd.DataFrame
    budget_values_by_id: dict[int, np.ndarray]
    source_methods: pd.DataFrame
    category_uncertainty: bool


def global_ar6_panel_title_pad(post_years: list[int]) -> int:
    """Return panel title padding for the ASR Global AR6 CC diagnostic row."""
    return (
        _GLOBAL_AR6_PERIOD_PANEL_TITLE_PAD
        if list(post_years)
        else _GLOBAL_AR6_NO_PERIOD_PANEL_TITLE_PAD
    )


def deterministic_global_ar6_source(
    *,
    acc_output_files: list[Path],
) -> DeterministicGlobalAR6Source:
    """Read deterministic AR6 CC source tables through the prerequisite aCC manifest."""
    manifest = _deterministic_acc_manifest(acc_output_files=acc_output_files)
    return _deterministic_global_ar6_source_from_paths(
        cc_input_path=Path(str(cast(dict[str, object], manifest["provenance"])["cc_input_path"])),
        post_study_path=None,
    )


def deterministic_global_ar6_rows_from_source(
    *,
    source: DeterministicGlobalAR6Source,
    asr_frame: pd.DataFrame,
    requested_years: list[int],
    target_unit: str,
) -> DeterministicGlobalAR6Rows:
    """Prepare deterministic global AR6 CC rows for one ASR figure scope."""
    return _deterministic_global_ar6_rows_from_source(
        source=source,
        asr_frame=asr_frame,
        requested_years=requested_years,
        target_unit=target_unit,
    )


def uncertainty_global_ar6_source(
    *,
    manifest: UncertaintyManifest,
) -> UncertaintyGlobalAR6Source:
    """Read AR6 CC uncertainty source artifacts once for ASR figure scopes."""
    acc_manifest = _prerequisite_manifest(manifest=manifest, source="uncertainty_acc")
    ar6_manifest = _prerequisite_manifest_or_none(
        manifest=acc_manifest,
        source="uncertainty_ar6_cc",
    )
    if ar6_manifest is None:
        deterministic = _deterministic_global_ar6_source_from_manifest(
            manifest_path=_deterministic_ar6_manifest_path(manifest=acc_manifest),
        )
        return UncertaintyGlobalAR6Source(
            deterministic=deterministic,
            summary=pd.DataFrame(),
            post_summary=pd.DataFrame(),
            budget_identity=pd.DataFrame(),
            budget_values_by_id={},
            source_methods=pd.DataFrame(),
            category_uncertainty=False,
        )
    paths = ar6_cc_run_paths_from_manifest(manifest=ar6_manifest)
    budget_identity = read_uncertainty_table(
        path=paths.budget_row_identity,
        output_format=str(ar6_manifest.output_format),
    )
    return UncertaintyGlobalAR6Source(
        deterministic=None,
        summary=read_uncertainty_table(
            path=paths.summary_stats_runs,
            output_format=str(ar6_manifest.output_format),
        ),
        post_summary=read_uncertainty_table(
            path=paths.post_study_summary_stats_runs,
            output_format=str(ar6_manifest.output_format),
        ),
        budget_identity=budget_identity,
        budget_values_by_id=collect_selected_compact_run_values(
            path=paths.budget_runs,
            output_format=str(ar6_manifest.output_format),
            public_row_ids=budget_identity["budget_row_id"].astype(int).tolist(),
            stop_run_index=int(manifest.completed_runs),
        ),
        source_methods=pd.read_csv(paths.source_methods),
        category_uncertainty=_category_uncertainty(manifest=ar6_manifest),
    )


def uncertainty_global_ar6_rows_from_source(
    *,
    source: UncertaintyGlobalAR6Source,
    asr_frame: pd.DataFrame,
    requested_years: list[int],
    target_unit: str,
) -> UncertaintyGlobalAR6Rows:
    """Prepare AR6 CC uncertainty rows for one ASR figure scope."""
    if source.deterministic is not None:
        deterministic = _deterministic_global_ar6_rows_from_source(
            source=source.deterministic,
            asr_frame=asr_frame,
            requested_years=requested_years,
            target_unit=target_unit,
        )
        return UncertaintyGlobalAR6Rows(
            summary=pd.DataFrame(),
            budget=pd.DataFrame(),
            deterministic=deterministic,
            study_years=deterministic.study_years,
            post_years=deterministic.post_years,
            pair_count=_deterministic_pair_count(deterministic.frame),
        )
    summary = _scoped_uncertainty_summary(
        summary=source.summary,
        asr_frame=asr_frame,
        target_unit=target_unit,
        category_uncertainty=source.category_uncertainty,
    )
    post_summary = _scoped_uncertainty_summary(
        summary=source.post_summary,
        asr_frame=asr_frame,
        target_unit=target_unit,
        category_uncertainty=source.category_uncertainty,
    )
    summary = pd.concat([summary, post_summary], ignore_index=True)
    budget = _budget_rows(
        source=source,
        asr_frame=asr_frame,
        target_unit=target_unit,
    )
    study_years = sorted(int(year) for year in requested_years)
    visible_years = sorted({int(year) for year in summary["year"].tolist()})
    post_years = [year for year in visible_years if year > max(study_years)]
    return UncertaintyGlobalAR6Rows(
        summary=summary,
        budget=budget,
        deterministic=None,
        study_years=study_years,
        post_years=post_years,
        pair_count=_pair_count(source=source, asr_frame=asr_frame),
    )


def _deterministic_acc_manifest(*, acc_output_files: list[Path]) -> dict[str, object]:
    root = Path(os.path.commonpath([str(path.parent) for path in acc_output_files]))
    return read_json_dict(root.parent / "logs" / "scope_manifest.json")


def _deterministic_global_ar6_source_from_manifest(
    *,
    manifest_path: Path,
) -> DeterministicGlobalAR6Source:
    manifest = read_json_dict(manifest_path)
    artifacts = cast(dict[str, object], manifest["artifacts"])
    post_value = artifacts.get("post_study_output_file")
    return _deterministic_global_ar6_source_from_paths(
        cc_input_path=Path(str(artifacts["output_file"])),
        post_study_path=None if post_value is None else Path(str(post_value)),
    )


def _deterministic_global_ar6_source_from_paths(
    *,
    cc_input_path: Path,
    post_study_path: Path | None,
) -> DeterministicGlobalAR6Source:
    if post_study_path is None:
        post_study_path = cc_input_path.with_name(f"ar6_cc_post_study_period{cc_input_path.suffix}")
    return DeterministicGlobalAR6Source(
        study=read_table(path=cc_input_path),
        post=read_table(path=post_study_path) if post_study_path.exists() else None,
    )


def _deterministic_global_ar6_rows_from_source(
    *,
    source: DeterministicGlobalAR6Source,
    asr_frame: pd.DataFrame,
    requested_years: list[int],
    target_unit: str,
) -> DeterministicGlobalAR6Rows:
    study = _convert_wide_units(source.study, target_unit=target_unit)
    post = (
        None if source.post is None else _convert_wide_units(source.post, target_unit=target_unit)
    )
    combined = combine_study_and_post_tables(
        cc_table=_scope_deterministic_rows(study, asr_frame=asr_frame),
        post_study_cc_table=(
            None if post is None else _scope_deterministic_rows(post, asr_frame=asr_frame)
        ),
    )
    years = sorted(figure_year_columns(combined).values())
    study_years = sorted(int(year) for year in requested_years)
    post_years = [year for year in years if year > max(study_years)]
    return DeterministicGlobalAR6Rows(
        frame=combined,
        study_years=study_years,
        post_years=post_years,
    )


def _scope_deterministic_rows(frame: pd.DataFrame, *, asr_frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    filters = {
        "ssp_scenario": visible_values(asr_frame, AR6_CC_SSP_SCENARIO_COLUMN),
        "cc_category": visible_values(asr_frame, "cc_category"),
        "cc_model": visible_values(asr_frame, "cc_model"),
        "cc_scenario": visible_values(asr_frame, "cc_scenario"),
    }
    for column, values in filters.items():
        accepted = {str(value).strip() for value in values}
        out = out.loc[out[column].astype(str).str.strip().isin(accepted)].copy()
    return out


def _convert_wide_units(frame: pd.DataFrame, *, target_unit: str) -> pd.DataFrame:
    out = frame.copy()
    source_units = sorted({str(value).strip() for value in out["impact_unit"].astype(str)})
    source_unit = source_units[0]
    factor = _unit_factor(source_unit=source_unit, target_unit=target_unit)
    year_columns = list(figure_year_columns(out))
    out.loc[:, year_columns] = out.loc[:, year_columns].apply(pd.to_numeric, errors="raise")
    out.loc[:, year_columns] = out.loc[:, year_columns].to_numpy(dtype=np.float64) * factor
    out["impact_unit"] = target_unit
    return out


def _prerequisite_manifest(*, manifest: UncertaintyManifest, source: str) -> UncertaintyManifest:
    return cast(
        UncertaintyManifest,
        _prerequisite_manifest_or_none(manifest=manifest, source=source),
    )


def _prerequisite_manifest_or_none(
    *,
    manifest: UncertaintyManifest,
    source: str,
) -> UncertaintyManifest | None:
    for item in manifest.deterministic_prerequisites:
        if str(item.get("base_function_source")) == source:
            return read_manifest(path=Path(str(item["scope_manifest"])))
    return None


def _deterministic_ar6_manifest_path(*, manifest: UncertaintyManifest) -> Path:
    return next(
        Path(str(item["scope_manifest"]))
        for item in manifest.deterministic_prerequisites
        if str(item.get("base_function_source")) == "deterministic_ar6_cc"
    )


def _category_uncertainty(*, manifest: UncertaintyManifest) -> bool:
    source_parameters = cast(dict[str, object], manifest.source_parameters or {})
    ar6_parameters = cast(
        dict[str, object],
        source_parameters.get(AR6_DYNAMIC_CC_SOURCE, {}),
    )
    return bool(ar6_parameters.get("category_uncertainty", False))


def _scoped_uncertainty_identity(
    identity: pd.DataFrame,
    *,
    asr_frame: pd.DataFrame,
    category_uncertainty: bool,
) -> pd.DataFrame:
    out = identity.copy()
    filters = {
        "ssp_scenario": visible_values(asr_frame, AR6_CC_SSP_SCENARIO_COLUMN),
    }
    if not category_uncertainty:
        filters["cc_category"] = visible_values(asr_frame, "cc_category")
    for column, values in filters.items():
        accepted = {str(value).strip() for value in values}
        out = out.loc[out[column].astype(str).str.strip().isin(accepted)].copy()
    return out.reset_index(drop=True)


def _scoped_uncertainty_summary(
    *,
    summary: pd.DataFrame,
    asr_frame: pd.DataFrame,
    target_unit: str,
    category_uncertainty: bool,
) -> pd.DataFrame:
    out = _scoped_uncertainty_identity(
        summary,
        asr_frame=asr_frame,
        category_uncertainty=category_uncertainty,
    )
    return _convert_summary_units(summary=out, target_unit=target_unit)


def _convert_summary_units(*, summary: pd.DataFrame, target_unit: str) -> pd.DataFrame:
    out = summary.copy()
    numeric = [column for column in ("mean", "median", "p25", "p75", "p5", "p95") if column in out]
    for source_unit, positions in out.groupby("impact_unit", dropna=False).groups.items():
        factor = _unit_factor(source_unit=str(source_unit), target_unit=target_unit)
        out.loc[positions, numeric] = (
            out.loc[positions, numeric].to_numpy(dtype=np.float64) * factor
        )
    out["impact_unit"] = target_unit
    return out


def _budget_rows(
    *,
    source: UncertaintyGlobalAR6Source,
    asr_frame: pd.DataFrame,
    target_unit: str,
) -> pd.DataFrame:
    identity = _scoped_uncertainty_identity(
        source.budget_identity,
        asr_frame=asr_frame,
        category_uncertainty=source.category_uncertainty,
    )
    rows = []
    for record in identity.to_dict(orient="records"):
        source_unit = str(record["impact_unit"])
        factor = _unit_factor(source_unit=source_unit, target_unit=target_unit)
        values = source.budget_values_by_id[int(record["budget_row_id"])]
        payload = dict(record)
        payload["impact_unit"] = target_unit
        payload["__budget_values"] = values * factor
        rows.append(payload)
    return pd.DataFrame.from_records(rows)


def _pair_count(*, source: UncertaintyGlobalAR6Source, asr_frame: pd.DataFrame) -> int:
    scoped = _scoped_uncertainty_identity(
        source.source_methods,
        asr_frame=asr_frame,
        category_uncertainty=False,
    )
    pairs = scoped.loc[:, ["cc_category", "cc_model", "cc_scenario"]].drop_duplicates()
    return int(len(pairs))


def _deterministic_pair_count(frame: pd.DataFrame) -> int:
    pairs = frame.loc[:, ["cc_category", "cc_model", "cc_scenario"]].drop_duplicates()
    return int(len(pairs))


def _unit_factor(*, source_unit: str, target_unit: str) -> float:
    return float(cast(float, try_unit_conversion(source_unit, target_unit)))
