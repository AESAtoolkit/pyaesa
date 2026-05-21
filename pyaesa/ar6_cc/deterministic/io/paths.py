"""Deterministic path ownership for dynamic AR6 CC outputs."""

from pathlib import Path
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.reporting.summary_log import SUMMARY_LOG_FILENAME

from pyaesa.ar6_cc.shared.runtime.paths import (
    cc_selector_dir_name,
    get_cc_family_dir,
)


def _cc_output_filename(*, output_format: str) -> str:
    """Return the deterministic CC table file name inside ``results/``."""
    return f"ar6_cc.{output_format}"


def _cc_post_study_output_filename(*, output_format: str) -> str:
    """Return the deterministic post study CC table file name."""
    return f"ar6_cc_post_study_period.{output_format}"


def _cc_metadata_filename() -> str:
    """Return the deterministic dynamic AR6 CC scope manifest file name."""
    return "scope_manifest.json"


def get_cc_scope_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
    emission_type: str = "kyoto_gases",
    include_afolu: bool = False,
    emissions_mode: str = "gross_alt",
    subset_version: str | None = None,
    category: str | list[str] | None = None,
    ssp_scenario: str | list[str] | None = None,
) -> Path:
    """Return the deterministic carrying capacity directory for one CC scope.

    The folder identity is owned by the processed AR6 scope, the AR6 variable,
    optional subset version, and requested category and SSP selectors.
    """
    return (
        get_cc_family_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            emission_type=emission_type,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode,
            subset_version=subset_version,
        )
        / cc_selector_dir_name(category=category, ssp_scenario=ssp_scenario)
        / "deterministic"
    )


def get_cc_logs_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
    emission_type: str = "kyoto_gases",
    include_afolu: bool = False,
    emissions_mode: str = "gross_alt",
    subset_version: str | None = None,
    category: str | list[str] | None = None,
    ssp_scenario: str | list[str] | None = None,
) -> Path:
    """Return the deterministic carrying capacity logs directory for one CC scope."""
    return (
        get_cc_scope_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            emission_type=emission_type,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode,
            subset_version=subset_version,
            category=category,
            ssp_scenario=ssp_scenario,
        )
        / "logs"
    )


def get_cc_figures_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
    emission_type: str = "kyoto_gases",
    include_afolu: bool = False,
    emissions_mode: str = "gross_alt",
    subset_version: str | None = None,
    category: str | list[str] | None = None,
    ssp_scenario: str | list[str] | None = None,
) -> Path:
    """Return the deterministic carrying capacity figures directory for one CC scope."""
    return (
        get_cc_scope_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            emission_type=emission_type,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode,
            subset_version=subset_version,
            category=category,
            ssp_scenario=ssp_scenario,
        )
        / "figures"
    )


def get_cc_output_path(
    *,
    cc_dir: Path,
    output_format: str,
) -> Path:
    """Return the deterministic carrying capacity output path for one scope."""
    results_dir = Path(cc_dir) / "results"
    return ensure_file_parent(results_dir / _cc_output_filename(output_format=output_format))


def get_cc_post_study_output_path(
    *,
    cc_dir: Path,
    output_format: str,
) -> Path:
    """Return the deterministic post study carrying capacity output path."""
    results_dir = Path(cc_dir) / "results"
    return ensure_file_parent(
        results_dir / _cc_post_study_output_filename(output_format=output_format)
    )


def get_cc_summary_log_path(
    *,
    cc_dir: Path,
) -> Path:
    """Return the deterministic carrying capacity summary log path."""
    return ensure_file_parent(Path(cc_dir) / "logs" / SUMMARY_LOG_FILENAME)


def get_cc_metadata_path(
    *,
    cc_dir: Path,
) -> Path:
    """Return the deterministic carrying capacity metadata path for one scope."""
    return ensure_file_parent(Path(cc_dir) / "logs" / _cc_metadata_filename())


def get_subset_csv_path(
    processed_dir: Path,
    subset_version: str,
) -> Path:
    """Return the expected model-scenario subset CSV path."""
    return ensure_file_parent(processed_dir / f"model_scenario_subset__{subset_version}.csv")
