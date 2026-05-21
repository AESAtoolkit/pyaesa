"""User guidance for compact aSoCC Monte Carlo result tables."""

from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.shared.runtime.text import join_user_text_lines
from pyaesa.shared.uncertainty_assessment.io.tables import public_run_artifact_readme_lines


def write_results_readme(
    *,
    paths: AsoccUncertaintyRunPaths,
) -> None:
    """Write the compact aSoCC Monte Carlo result reading guide."""
    paths.results_readme.parent.mkdir(parents=True, exist_ok=True)
    paths.results_readme.write_text(
        join_user_text_lines(
            [
                "aSoCC Monte Carlo results",
                "",
                "Files",
                "",
                "public_row_identity: one row per public output row.",
                "The public_row_id column is the stable numeric key used by the run matrix.",
                *public_run_artifact_readme_lines(run_name="asocc_runs"),
                "Compact layout: sources that emit every public row.",
                "The first column is run_index. Other columns are public_row_id values.",
                "Sparse layout for inter method uncertainty: selected rows with",
                "run_index, public_row_id, and asocc. Join public_row_id to",
                "public_row_identity to read selected method leaf rows.",
                "summary_stats_runs: exact summary statistics after active source grouping.",
                "Sampled axes are omitted when the matching uncertainty source is active.",
                "Examples include l2_reuse_year, reference_year, and allocation method columns.",
                "source_methods.csv: compact scientific source log.",
                "scope_manifest.json: canonical run metadata, output paths, and public table",
                "schemas for result files written by this uncertainty run.",
                "LCIA level_1 rows use country coefficient of variation values from",
                "reg_cbca_covs.csv, or from reg_cbca_covs_group_<group_version>.csv",
                "when group_reg=True. Aggregated region axes use",
                "reg_cbca_covs_aggreg_indices.csv or",
                "reg_cbca_covs_group_<group_version>_aggreg_indices.csv.",
                "L2 rows use sector coefficient of variation values from sec_cbca_covs.csv.",
                "The files are available in",
                "data_raw/mrio/exiobase_3/lcia/carbon_accounts_covs/.",
                "uncertainty_config sector_cov_mapping maps selected s_p labels to",
                "sec_cbca_covs.csv sector codes.",
                "inter_mrio_uncertainty source is a published disaggregated aSoCC source",
                "label created by disaggregate_asocc(...), for example 'oecd_electricity'.",
                "Eligible sampled rows move between the main deterministic value and the",
                "alternate source value.",
                "sobol/: optional Sobol variance decomposition folder.",
                "It is written only when Sobol is requested and at least two sources are active.",
                "",
                "How to read the compact CSV outputs in pandas",
                "",
                "identity = pandas.read_csv('public_row_identity.csv')",
                "runs = pandas.read_csv('asocc_runs.csv')",
                "long = runs.melt(id_vars='run_index', var_name='public_row_id', "
                "value_name='asocc')",
                "long['public_row_id'] = long['public_row_id'].astype(int)",
                "public_render_rows = long.merge(identity, on='public_row_id', how='left')",
                "",
                "For sparse selected row outputs",
                "",
                "runs = pandas.read_csv('asocc_runs.csv')",
                "public_render_rows = runs.merge(identity, on='public_row_id', how='left')",
                "",
                "For Parquet outputs, use pandas.read_parquet with the same merge logic.",
                "Both run layouts avoid repeating identity text for every run.",
            ],
            trailing_newline=True,
        ),
        encoding="utf-8",
    )
