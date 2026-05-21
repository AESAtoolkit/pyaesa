"""Path ownership for processed AR6 climate outputs."""

from pathlib import Path

from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root as _get_repo_root,
)


def study_period_folder(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
) -> str:
    """Return the canonical study period folder name."""
    folder = f"{int(study_period[0])}-{int(study_period[1])}"
    if not harmonization:
        return f"{folder}_no_harmonization"
    normalized_harmonization_method = str(harmonization_method).strip()
    return f"{folder}_harmonization_{normalized_harmonization_method}"


def get_processed_scope_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
) -> Path:
    """Return the shared processed AR6 scope directory for ``study_period``."""
    return _get_repo_root().joinpath(
        "data_processed",
        "ar6",
        study_period_folder(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
        ),
    )


def get_processed_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
) -> Path:
    """Return the process_ar6-owned output directory for ``study_period``.

    The shared processed AR6 scope root is returned by
    :func:`get_processed_scope_dir`. This function resolves the owned
    ``process_ar6`` subtree beneath that shared root.
    """
    return get_processed_scope_dir(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
    ).joinpath("process_ar6")


def get_logs_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
) -> Path:
    """Return the log directory for ``study_period``."""
    return get_processed_dir(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
    ).joinpath("logs")


def get_figures_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
) -> Path:
    """Return the figures directory path for ``study_period``.

    The path is resolved without creating the directory. ``process_ar6(...)``
    owns this optional subtree and materializes it only when figure files are
    actually written.
    """
    return get_processed_dir(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
    ).joinpath("figures")


def get_scope_dirs(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
) -> tuple[Path, Path, Path]:
    """Return process, log, and figure directories for one AR6 output scope."""
    return (
        get_processed_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
        ),
        get_logs_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
        ),
        get_figures_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
        ),
    )
