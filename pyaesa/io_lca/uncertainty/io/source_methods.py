"""IO-LCA uncertainty source method and README writers."""

from pathlib import Path

import pandas as pd

from pyaesa.io_lca.uncertainty.runtime.models import (
    IOLCAUncertaintyRequest,
    IOLCAUncertaintyRunPaths,
)
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent, write_via_atomic_temp
from pyaesa.shared.runtime.text import join_user_text_lines
from pyaesa.shared.uncertainty_assessment.io.tables import public_run_artifact_readme_lines


def write_io_lca_source_methods(*, path: Path, rows: pd.DataFrame) -> None:
    """Write the IO-LCA source method log."""
    ordered = rows.sort_values(
        ["source_name", "lcia_method", "primary_cov_kind", "primary_cov_key"],
        kind="mergesort",
    ).reset_index(drop=True)
    write_via_atomic_temp(
        ensure_file_parent(path),
        writer=lambda tmp_path: ordered.to_csv(tmp_path, index=False),
    )


def write_io_lca_results_readme(
    *,
    paths: IOLCAUncertaintyRunPaths,
    request: IOLCAUncertaintyRequest,
) -> None:
    """Write the public IO-LCA Monte Carlo result guide."""
    text = _readme_text(request=request)

    def _write_text(tmp_path: Path) -> None:
        tmp_path.write_text(text, encoding="utf-8")

    write_via_atomic_temp(
        ensure_file_parent(paths.results_readme),
        writer=_write_text,
    )


def _readme_text(*, request: IOLCAUncertaintyRequest) -> str:
    selector_text = ", ".join(
        f"{key}={value}" for key, value in request.filters.items() if value is not None
    )
    lines = [
        "IO-LCA Uncertainty Results",
        "",
        "This run evaluates IO-LCA LCIA value uncertainty.",
        "",
        "Artifacts",
        "- public_row_identity: public IO-LCA rows, one row per matrix column.",
        *public_run_artifact_readme_lines(run_name="lca_runs"),
        "  Layout: compact run by public row numeric matrix.",
        "- summary_stats_runs: exact summary statistics computed from all runs.",
        "- source_methods.csv: LCIA coefficient of variation sources used by the run.",
        "- scope_manifest.json: request, prerequisite, output, reuse metadata,",
        "  and canonical public table schemas for this result scope.",
        "",
        "Sampling Method",
        "For each deterministic component row, the package applies:",
        "lower = value * (1 - cov_value)",
        "upper = value * (1 + cov_value)",
        "sampled_value = lower + u_shared * (upper - lower)",
        "",
        "The shared random variable key is defined by project, source, grouping",
        "scope, driver kind, and driver key. It does not include LCIA method,",
        "impact category, studied year, or public row id, so the same LCIA",
        "uncertainty driver is linked across those outputs within a run.",
        "",
        "LCIA CoV Mapping",
        "L1 country owner rows use packaged country coefficient of variation",
        "values from reg_cbca_covs.csv, or from",
        "reg_cbca_covs_group_<group_version>.csv when group_reg=True.",
        "Aggregated region axes use reg_cbca_covs_aggreg_indices.csv, or",
        "reg_cbca_covs_group_<group_version>_aggreg_indices.csv when group_reg=True.",
        "Sector owner rows use lcia_uncertainty.sector_cov_mapping to map the",
        "output s_p sector labels to packaged sector coefficient of",
        "variation codes. Example:",
        '{"lcia_uncertainty": {"sector_cov_mapping": {"Electricity": "Electricity"}}}',
        "The available country and sector codes are in the project local files:",
        "data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/reg_cbca_covs.csv",
        "data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/"
        "reg_cbca_covs_group_<group_version>.csv",
        "data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/",
        "  reg_cbca_covs_aggreg_indices.csv",
        "data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/",
        "  reg_cbca_covs_group_<group_version>_aggreg_indices.csv",
        "data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/sec_cbca_covs.csv",
        "",
        "Aggregation",
        "When aggreg_indices=True is used,",
        "the run samples deterministic component rows and sums sampled component",
        "values into the public IO-LCA row identity. The public identity therefore",
        "matches deterministic_io_lca(...) output rows, while the source_methods",
        "table records the component uncertainty drivers used to build them.",
        "",
        "Scope",
        f"- fu_code: {request.fu_spec.fu_code}",
        f"- years: {min(request.years)} to {max(request.years)}",
        f"- lcia_method: {', '.join(request.lcia_methods)}",
        f"- selectors: {selector_text or 'all deterministic selectors for the FU'}",
        f"- aggreg_indices: {request.aggreg_indices}",
        "",
    ]
    return join_user_text_lines(lines)
