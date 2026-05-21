"""Text and tabular companion outputs for AR6 processing."""

import pandas as pd

from ..figures.figure_sampling_config import (
    RUN_BATCH_SIZE,
    STABLE_CHECKS_REQUIRED,
    minimum_completed_runs_per_bucket_for_convergence,
)
from pyaesa.process.ar6.utils.io.contracts import (
    budget_stats_sheet_name,
    final_pathways_sheet_name,
)
from pyaesa.shared.runtime.text import join_user_text_lines, wrap_user_text_lines


def processing_citation_text(raw_citation_text: str, harmonization: bool) -> str:
    """Return the processed output citation/source usage TXT content."""
    lines = [
        "Recommended citations and source usage for processed AR6 climate outputs",
        "",
    ]
    if raw_citation_text.strip():
        lines += [
            "Raw source citation block copied from the raw data folder:",
            "",
            *wrap_user_text_lines(raw_citation_text.strip().splitlines()),
            "",
        ]
    else:
        lines += ["No raw data citation text file was found.", ""]
    lines += [
        "Processed output usage notes:",
        (
            "- AR6 public explorer: scenario pathways and metadata used for "
            "filtering and processed outputs."
        ),
    ]
    if harmonization:
        lines += [
            "- PRIMAP-hist: historical Kyoto Gases and CO2 baseline inputs.",
            (
                "- Global Carbon Budget national fossil: bunker CO2 additions "
                "used in historical preprocessing."
            ),
            (
                "- AR6 historical comparison file from the AR6 Scenario Explorer "
                "public download 'AR6_historical_emissions.csv': used only as the red "
                "overlay in the historical-emissions figure and not in the "
                "harmonization baseline."
            ),
        ]
    return join_user_text_lines(lines)


def log_columns_explanation_text() -> str:
    """Return the harmonization log column explanation TXT content."""
    return join_user_text_lines(
        [
            "Harmonization log columns explanation",
            "",
            "Category: AR6 scenario category.",
            "Ssp_family: SSP family index.",
            "model-base-year: first year with raw pathway data.",
            "pathway-last-year: last year with raw pathway data.",
            "harmonization-year-requested: requested study start year.",
            "harmonization-year: historical target value used for alignment year.",
            ("harmonization-method: harmonization method family applied to the row ('offset')."),
            (
                "offset-variant-used: offset variant selected for the row "
                "('constant_offset' or 'reduced_offset')."
            ),
            (
                "harmonization-method-note: optional note describing row-specific variant "
                "handling, for example when the reduced offset variant uses "
                "the last available pathway year or when pyaesa reduces the effective "
                "harmonization horizon to preserve the first negative-emissions year."
            ),
            "model-netzero-year: model-implied net-zero timing.",
            "harmonization-netzero-year: adjusted net-zero timing after harmonization.",
            (
                "horizon-for-harmonization: row-specific last year or "
                "net-zero year used for the cumulative correction."
            ),
            "pathway-cumulative: pathway cumulative emissions over the harmonization window.",
            "historic-cumulative: historical cumulative emissions benchmark.",
            "delta-cumulative: difference between pathway and benchmark cumulative emissions.",
            "yearly-correction: per-year correction applied during harmonization.",
        ],
    )


def figure_sampling_log_columns_explanation_text() -> str:
    """Return the figure sampling log column explanation TXT content."""
    minimum_runs = minimum_completed_runs_per_bucket_for_convergence()
    return join_user_text_lines(
        [
            "AR6 figure sampling convergence log columns explanation",
            "",
            "variable: Output variable used in the sampling figure family.",
            "method: Sampling method used for the figure subset (SRS or LHS).",
            "distribution_kind: Budget distribution checked for convergence (study or remaining).",
            "category: AR6 scenario category of the retained sampled rows.",
            "ssp_family: SSP family of the retained sampled rows, or 'all' for category totals.",
            "rng_seed: Deterministic random seed used for the variable and method.",
            (
                "final_runs_per_bucket: Final number of runs generated inside each "
                "category-SSP bucket for the variable and method."
            ),
            (
                "run_batch_size: Number of additional runs added per convergence "
                "iteration and therefore the spacing between checkpoint comparisons."
            ),
            (
                "maximum_runs_per_bucket: Maximum per-bucket run count allowed before "
                "the figure generation fails. Controlled by "
                "process_ar6(..., figure_convergence_max_runs=...)."
            ),
            (
                "relative_tolerance: Maximum allowed relative change between two "
                "convergence checks. Controlled by "
                "process_ar6(..., figure_convergence_tol=...)."
            ),
            (
                "stable_checks_required: Number of consecutive stable checkpoint "
                "comparisons required before the variable-method sample is accepted. "
                f"With these AR6 sampling settings ({RUN_BATCH_SIZE} runs per batch and "
                f"{STABLE_CHECKS_REQUIRED} stable comparisons), convergence cannot be "
                f"accepted before {minimum_runs} completed runs per bucket."
            ),
            "mean: Arithmetic mean of the sampled budget distribution in Gt units.",
            "median: 50th percentile of the sampled budget distribution in Gt units.",
            "p25: 25th percentile of the sampled budget distribution in Gt units.",
            "p75: 75th percentile of the sampled budget distribution in Gt units.",
            "p5: 5th percentile of the sampled budget distribution in Gt units.",
            "p95: 95th percentile of the sampled budget distribution in Gt units.",
        ],
    )


def excel_readme_sheet(harmonization: bool) -> pd.DataFrame:
    """Return the README worksheet content for the processed workbook."""
    rows = [
        {
            "sheet": "ORIGINAL_AR6",
            "description": (
                "Original AR6 scenario pathways after interpolation and derived "
                "variable construction."
            ),
            "data_sources_used": "AR6 public explorer",
        },
        {
            "sheet": "SOURCE_METADATA",
            "description": (
                "Scenario metadata used for metadata-based plots such as median warming in 2100."
            ),
            "data_sources_used": "AR6 public explorer",
        },
        {
            "sheet": final_pathways_sheet_name(harmonization=harmonization),
            "description": (
                "Harmonized net pathways, sequestration companion rows, and gross "
                "pathways for the study period."
                if harmonization
                else (
                    "Retained pathways after interpolation, derived variables, and "
                    "study-year filtering."
                )
            ),
            "data_sources_used": (
                "AR6 + PRIMAP-hist + Global Carbon Budget national fossil"
                if harmonization
                else "AR6 public explorer"
            ),
        },
        {
            "sheet": budget_stats_sheet_name(harmonization=harmonization),
            "description": (
                "Aggregated statistics computed from harmonized net pathways."
                if harmonization
                else "Aggregated statistics computed from retained non-harmonized net pathways."
            ),
            "data_sources_used": (
                "Derived from harmonized outputs"
                if harmonization
                else "Derived from retained AR6 pathways"
            ),
        },
    ]
    if harmonization:
        rows.append(
            {
                "sheet": "HISTORICAL_PRIMAP_GCP",
                "description": "Historical emissions baseline used for harmonization.",
                "data_sources_used": "PRIMAP-hist + Global Carbon Budget national fossil",
            }
        )
    return pd.DataFrame(rows)
