"""Writers for processed AR6 climate artefacts."""

from pathlib import Path

import pandas as pd
from pyaesa.shared.selectors.scenarios import normalize_ssp_token

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

from pyaesa.process.ar6.utils.io.contracts import (
    budget_stats_sheet_name,
    final_pathways_sheet_name,
)


def write_processed_workbook(
    *,
    harmonization: bool,
    output_file: Path,
    readme_df: pd.DataFrame,
    citations_text: str,
    final_all: pd.DataFrame,
    original_all: pd.DataFrame,
    source_meta: pd.DataFrame,
    stats_var: pd.DataFrame,
    historical_emissions: pd.DataFrame | None,
) -> None:
    """Write the main processed workbook."""
    output_file = ensure_file_parent(output_file)
    citations_df = pd.DataFrame({"citation_and_usage_notes": citations_text.splitlines()})
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        readme_df.to_excel(writer, sheet_name="README", index=False, merge_cells=False)
        citations_df.to_excel(writer, sheet_name="CITATIONS", index=False, merge_cells=False)
        final_all.to_excel(
            writer,
            sheet_name=final_pathways_sheet_name(harmonization=harmonization),
            merge_cells=False,
        )
        original_all.to_excel(writer, sheet_name="ORIGINAL_AR6", merge_cells=False)
        source_meta.to_excel(writer, sheet_name="SOURCE_METADATA", merge_cells=False)
        stats_var.to_excel(
            writer,
            sheet_name=budget_stats_sheet_name(harmonization=harmonization),
            merge_cells=False,
        )
        if historical_emissions is not None and not historical_emissions.empty:
            historical_emissions.to_excel(
                writer, sheet_name="HISTORICAL_PRIMAP_GCP", merge_cells=False
            )


def write_harmonization_log_workbook(log_file: Path, harmonization_log_all: pd.DataFrame) -> None:
    """Write the harmonization log workbook."""
    log_file = ensure_file_parent(log_file)
    with pd.ExcelWriter(log_file, engine="xlsxwriter") as writer:
        harmonization_log_all.to_excel(writer, sheet_name="HARMONIZATION_LOG", merge_cells=False)


_TEMPLATE_HEADER_COMMENT = (
    "# Model-scenario template for deterministic_ar6_cc subset selection.\n"
    "# To use a subset: copy this file, remove unwanted rows, and rename it to\n"
    "# model_scenario_subset__{your_version_name}.csv (keep the double-underscore prefix).\n"
    "# The version name is then passed as subset_version='your_version_name'.\n"
)
_TEMPLATE_README_FILE = "README_model_scenario_subset.txt"


def _read_template_guide_text() -> str:
    """Read the packaged model-scenario subset guide text."""
    guide_path = Path(__file__).with_name(_TEMPLATE_README_FILE)
    return guide_path.read_text(encoding="utf-8")


def _template_ssp_scenario(value: object) -> str:
    """Return the canonical public SSP token for one retained AR6 SSP family value."""
    if isinstance(value, int | float):
        return f"SSP{int(value)}"
    return normalize_ssp_token(value, context="Processed AR6 template SSP family")


def write_model_scenario_template(
    *,
    final_all: pd.DataFrame,
    processed_dir: Path,
) -> Path:
    """Write a CSV template listing all retained model-scenario pairs.

    Args:
        final_all: Harmonized AR6 pathway DataFrame with MultiIndex
            ``(model, scenario, variable)`` and metadata columns
            ``Category`` and ``Ssp_family``.
        processed_dir: Output directory for the template file.

    Returns:
        Path to the written template CSV.
    """
    index_frame = final_all.index.to_frame(index=False)
    template = pd.DataFrame(
        {
            "model": index_frame["model"].to_numpy(),
            "scenario": index_frame["scenario"].to_numpy(),
            "category": final_all["Category"].to_numpy(),
            "ssp_scenario": [
                _template_ssp_scenario(value) for value in final_all["Ssp_family"].tolist()
            ],
        }
    ).drop_duplicates()
    template = template.sort_values(["category", "ssp_scenario", "model", "scenario"]).reset_index(
        drop=True
    )
    template_path = processed_dir / "model_scenario_subset__template.csv"
    template_path = ensure_file_parent(template_path)
    with open(template_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(_TEMPLATE_HEADER_COMMENT)
        template.to_csv(fh, index=False)
    ensure_file_parent(processed_dir / _TEMPLATE_README_FILE).write_text(
        _read_template_guide_text(),
        encoding="utf-8",
    )
    return template_path


def build_dropped_rows_df(drop_logs: list[pd.DataFrame]) -> pd.DataFrame:
    """Return the concatenated AR6 row issue log dataframe."""
    columns = [
        "model",
        "scenario",
        "variable",
        "retained_variable",
        "ssp_family",
        "category",
        "drop_stage",
        "drop_reason",
    ]
    if drop_logs:
        normalized_logs = [frame.loc[:, columns] for frame in drop_logs]
        dropped_rows_df = pd.concat(normalized_logs, ignore_index=True).sort_values(
            by=["drop_stage", "drop_reason", "model", "scenario", "variable"],
            kind="stable",
        )
        return dropped_rows_df
    return pd.DataFrame(columns=columns)
