"""Figure rendering for external LCA subfigures triggered by ASR."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.external_inputs.lca.monte_carlo import (
    ExternalLCAMonteCarloSource,
    external_lca_values_for_runs,
)
from pyaesa.io_lca.plot.figure_writers import (
    write_lcia_method_checkpoint_figures,
    write_lcia_method_figures,
)
from pyaesa.io_lca.uncertainty.figures.product_renderers import (
    write_lca_uncertainty_band_figures,
    write_lca_uncertainty_violin_figures,
)
from pyaesa.io_lca.uncertainty.figures.scope_planner import VALUE_ARRAY_COLUMN
from pyaesa.shared.figures.trajectory_bands import SUMMARY_COLUMNS
from pyaesa.shared.runtime.scenario.columns import EXT_LCA_SSP_SCENARIO_COLUMN
from pyaesa.shared.figures.contracts import SELECTOR_COLUMNS
from pyaesa.shared.runtime.reporting.status import StatusSink

from pyaesa.external_inputs.lca.naming import normalize_external_lca_version_name
from pyaesa.external_inputs.lca.paths import (
    external_lca_deterministic_figures_dir,
    external_lca_monte_carlo_figures_dir,
)


def _normalize_for_figures(
    *,
    frame: pd.DataFrame,
    lcia_method: str,
    value_column: str,
) -> pd.DataFrame:
    out = _normalize_identity_for_figures(frame=frame, lcia_method=lcia_method)
    out["value"] = pd.Series(pd.to_numeric(out[value_column], errors="raise"), copy=False)
    out["lca_value"] = pd.Series(out["value"], copy=False)
    return _figure_columns(out)


def _normalize_identity_for_figures(*, frame: pd.DataFrame, lcia_method: str) -> pd.DataFrame:
    out = frame.copy()
    if "ssp_scenario" not in out.columns and EXT_LCA_SSP_SCENARIO_COLUMN in out.columns:
        out["ssp_scenario"] = pd.Series(out[EXT_LCA_SSP_SCENARIO_COLUMN], copy=False)
    out["lcia_method"] = str(lcia_method)
    out["year"] = pd.Series(pd.to_numeric(out["year"], errors="raise"), copy=False).astype(int)
    out["impact"] = pd.Series(out["impact"], copy=False).astype(str)
    out["impact_unit"] = pd.Series(out["impact_unit"], copy=False).astype(str)
    return out


def _figure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    keep = [
        *[column for column in SELECTOR_COLUMNS if column in out.columns],
        *[
            column
            for column in (
                "l1_l2_method",
                "l2_method",
                "l1_method",
                "reference_year",
                "l2_reuse_year",
                "run_index",
                "year",
                "impact",
                "value",
                "lca_value",
                "impact_unit",
                "ssp_scenario",
                EXT_LCA_SSP_SCENARIO_COLUMN,
                "lcia_method",
                *SUMMARY_COLUMNS,
                VALUE_ARRAY_COLUMN,
            )
            if column in out.columns
        ],
    ]
    return out.loc[:, keep].reset_index(drop=True)


def render_external_lca_deterministic_figures_from_rows(
    *,
    proj_base: Path,
    version_name: str,
    lcia_method: str,
    rows: pd.DataFrame,
    value_column: str = "lca_value",
    output_format: str,
    dpi: int,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render deterministic external LCA figures from already loaded LCA rows."""
    if status is not None:
        status.show("[external_lca] Generating figures: external LCA")
    figure_frame = _normalize_for_figures(
        frame=rows,
        lcia_method=lcia_method,
        value_column=value_column,
    )
    scope_dir = external_lca_deterministic_figures_dir(project_base=proj_base)
    file_stem_prefix = normalize_external_lca_version_name(
        version_name,
        argument_name="external LCA version_name",
    )
    unique_years = sorted({int(year) for year in cast(pd.Series, figure_frame["year"]).tolist()})
    if len(unique_years) == 1:
        paths = write_lcia_method_checkpoint_figures(
            lcia_method_frame=figure_frame,
            reference_frame=figure_frame,
            figures_dir=scope_dir,
            lcia_method=lcia_method,
            checkpoint_years=unique_years,
            dpi=dpi,
            output_format=output_format,
            family_label="LCA",
            selector_columns=SELECTOR_COLUMNS,
            file_stem_prefix=file_stem_prefix,
        )
    else:
        paths = write_lcia_method_figures(
            lcia_method_frame=figure_frame,
            reference_frame=figure_frame,
            figures_dir=scope_dir,
            lcia_method=lcia_method,
            dpi=dpi,
            output_format=output_format,
            family_label="LCA",
            selector_columns=SELECTOR_COLUMNS,
            file_stem_prefix=file_stem_prefix,
        )
    return sorted({Path(path) for path in paths})


def render_external_lca_uncertainty_figures_from_source(
    *,
    proj_base: Path,
    source: ExternalLCAMonteCarloSource,
    output_format: str,
    dpi: int,
    completed_runs: int | None = None,
    status: StatusSink | None = None,
) -> list[Path]:
    """Render external LCA Monte Carlo figures from a materialized source matrix."""
    if status is not None:
        status.show("[external_lca] Generating figures: external LCA")
    values = _source_values(source=source, completed_runs=completed_runs)
    years = sorted({int(year) for year in source.identity["year"].tolist()})
    figures_dir = external_lca_monte_carlo_figures_dir(project_base=proj_base)
    file_stem_prefix = normalize_external_lca_version_name(
        source.version_name,
        argument_name="external LCA version_name",
    )
    if len(years) == 1:
        return write_lca_uncertainty_violin_figures(
            lcia_method_frame=_uncertainty_violin_frame(source=source, values=values),
            reference_frame=source.identity,
            figures_dir=figures_dir,
            lcia_method=source.lcia_method,
            checkpoint_years=years,
            dpi=dpi,
            output_format=output_format,
            family_label="LCA uncertainty",
            selector_columns=SELECTOR_COLUMNS,
            file_stem_prefix=file_stem_prefix,
        )
    return write_lca_uncertainty_band_figures(
        lcia_method_frame=_uncertainty_summary_frame(source=source, values=values),
        reference_frame=source.identity,
        figures_dir=figures_dir,
        lcia_method=source.lcia_method,
        dpi=dpi,
        output_format=output_format,
        family_label="LCA uncertainty",
        selector_columns=SELECTOR_COLUMNS,
        file_stem_prefix=file_stem_prefix,
    )


def _source_values(
    *,
    source: ExternalLCAMonteCarloSource,
    completed_runs: int | None,
) -> np.ndarray:
    run_count = len(source.run_indices) if completed_runs is None else int(completed_runs)
    run_indices = np.arange(run_count, dtype=np.int64)
    return external_lca_values_for_runs(source=source, run_indices=run_indices)


def _uncertainty_summary_frame(
    *,
    source: ExternalLCAMonteCarloSource,
    values: np.ndarray,
) -> pd.DataFrame:
    out = _normalize_identity_for_figures(frame=source.identity, lcia_method=source.lcia_method)
    out["mean"] = np.mean(values, axis=0)
    out["median"] = np.median(values, axis=0)
    out["p25"] = np.percentile(values, 25.0, axis=0)
    out["p75"] = np.percentile(values, 75.0, axis=0)
    out["p5"] = np.percentile(values, 5.0, axis=0)
    out["p95"] = np.percentile(values, 95.0, axis=0)
    return _figure_columns(out)


def _uncertainty_violin_frame(
    *,
    source: ExternalLCAMonteCarloSource,
    values: np.ndarray,
) -> pd.DataFrame:
    out = _normalize_identity_for_figures(frame=source.identity, lcia_method=source.lcia_method)
    out[VALUE_ARRAY_COLUMN] = list(values.T)
    return _figure_columns(out)
