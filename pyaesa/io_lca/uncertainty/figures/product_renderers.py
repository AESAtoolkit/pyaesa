"""Rendering policy for IO-LCA uncertainty figure products."""

from pathlib import Path

import pandas as pd

from pyaesa.io_lca.figures.common import (
    lca_prospective_scope_slices,
    ordered_impacts,
    selector_groups,
)
from pyaesa.io_lca.uncertainty.figures.scope_renderers import (
    write_band_scope,
    write_violin_scope,
)
from pyaesa.shared.figures.request_validation import validate_consecutive_multi_year_figure_request


def write_lca_uncertainty_band_figures(
    *,
    lcia_method_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    figures_dir: Path,
    lcia_method: str,
    dpi: int,
    output_format: str,
    family_label: str = "IO-LCA uncertainty",
    selector_columns: tuple[str, ...] | None = None,
    file_stem_prefix: str | None = None,
) -> list[Path]:
    """Write multi-year IO-LCA uncertainty interval figures."""
    selector_cols, groups = selector_groups(
        frame=lcia_method_frame,
        selector_columns=selector_columns,
    )
    impact_order, impact_labels = ordered_impacts(frame=lcia_method_frame, lcia_method=lcia_method)
    all_paths: list[Path] = []
    for _group_key, base_group_df in groups:
        base_group_df = _normalized_year_frame(base_group_df)
        for scenario_token, scenario_title, group_df in lca_prospective_scope_slices(base_group_df):
            years = sorted({int(year) for year in group_df["year"].tolist()})
            validate_consecutive_multi_year_figure_request(
                requested_years=years,
                family_label=family_label,
            )
            impacts = [impact for impact in impact_order if impact in set(group_df["impact"])]
            all_paths.extend(
                write_band_scope(
                    group_df=group_df,
                    reference_frame=reference_frame,
                    figures_dir=figures_dir,
                    lcia_method=lcia_method,
                    family_label=family_label,
                    selector_cols=selector_cols,
                    impact_labels=impact_labels,
                    impacts=impacts,
                    years=years,
                    scenario_token=scenario_token,
                    scenario_title=scenario_title,
                    dpi=dpi,
                    output_format=output_format,
                    file_stem_prefix=file_stem_prefix,
                )
            )
    return all_paths


def write_lca_uncertainty_violin_figures(
    *,
    lcia_method_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    figures_dir: Path,
    lcia_method: str,
    checkpoint_years: list[int],
    dpi: int,
    output_format: str,
    family_label: str = "IO-LCA uncertainty",
    selector_columns: tuple[str, ...] | None = None,
    file_stem_prefix: str | None = None,
) -> list[Path]:
    """Write single year IO-LCA uncertainty violin figures."""
    selector_cols, groups = selector_groups(
        frame=lcia_method_frame,
        selector_columns=selector_columns,
    )
    impact_order, impact_labels = ordered_impacts(frame=lcia_method_frame, lcia_method=lcia_method)
    all_paths: list[Path] = []
    for _group_key, group_df in groups:
        group_df = _normalized_year_frame(group_df)
        for scenario_token, scenario_title, scoped_group in lca_prospective_scope_slices(group_df):
            for checkpoint_year in [int(year) for year in checkpoint_years]:
                year_df = scoped_group.loc[
                    scoped_group["year"].astype(int).eq(int(checkpoint_year))
                ].copy()
                impacts = [impact for impact in impact_order if impact in set(year_df["impact"])]
                all_paths.extend(
                    write_violin_scope(
                        year_df=year_df,
                        reference_frame=reference_frame,
                        figures_dir=figures_dir,
                        lcia_method=lcia_method,
                        family_label=family_label,
                        selector_cols=selector_cols,
                        impact_labels=impact_labels,
                        impacts=impacts,
                        scenario_token=scenario_token,
                        scenario_title=scenario_title,
                        checkpoint_year=checkpoint_year,
                        dpi=dpi,
                        output_format=output_format,
                        file_stem_prefix=file_stem_prefix,
                    )
                )
    return all_paths


def _normalized_year_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["year"] = pd.Series(pd.to_numeric(out["year"], errors="raise"), copy=False).astype(int)
    return out
