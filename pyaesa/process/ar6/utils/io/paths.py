"""Path ownership for processed AR6 climate outputs."""

from pathlib import Path

from pyaesa.workspace_initialisation.workspace import (
    get_default_repo_root as _get_repo_root,
)
from pyaesa.download.ar6.utils.config import normalize_ar6_categories


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


def category_scope_folder(category: str | list[str] | None = None) -> str:
    """Return the canonical category folder token for one processed AR6 scope."""
    categories = normalize_ar6_categories(category)
    numbers = [int(item[1:]) for item in categories if item.startswith("C") and item[1:].isdigit()]
    if len(categories) == 1:
        return categories[0]
    if (
        categories
        and len(numbers) == len(categories)
        and numbers == list(range(numbers[0], numbers[-1] + 1))
    ):
        return f"C{numbers[0]}-C{numbers[-1]}"
    return "-".join(categories)


def get_processed_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
    category: str | list[str] | None = None,
) -> Path:
    """Return the process_ar6-owned output directory for ``study_period``.

    The shared processed AR6 scope root is returned by
    :func:`get_processed_scope_dir`. This function resolves the owned
    category-scoped ``process_ar6`` subtree beneath that shared root.
    """
    return get_processed_scope_dir(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
    ).joinpath("process_ar6", category_scope_folder(category))


def get_logs_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
    category: str | list[str] | None = None,
) -> Path:
    """Return the log directory for ``study_period``."""
    return get_processed_dir(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        category=category,
    ).joinpath("logs")


def get_figures_dir(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
    category: str | list[str] | None = None,
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
        category=category,
    ).joinpath("figures")


def get_scope_dirs(
    study_period: list[int],
    *,
    harmonization: bool = True,
    harmonization_method: str | None = None,
    category: str | list[str] | None = None,
) -> tuple[Path, Path, Path]:
    """Return process, log, and figure directories for one AR6 output scope."""
    return (
        get_processed_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            category=category,
        ),
        get_logs_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            category=category,
        ),
        get_figures_dir(
            study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            category=category,
        ),
    )
