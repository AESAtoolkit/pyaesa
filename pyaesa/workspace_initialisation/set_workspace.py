"""Workspace setup public coordinator."""

from pathlib import Path

from pyaesa.shared.runtime.text import wrap_user_text_lines
from pyaesa.workspace_initialisation.packaged_prerequisites import (
    import_prerequisites,
)
from pyaesa.workspace_initialisation.workspace import (
    resolve_repo_root,
    set_default_repo_root,
)


def _summary_log_path(*, repo_root: Path) -> Path:
    """Return the workspace setup summary log path."""
    return repo_root / "data_raw" / "summary.log"


def _path_lines(*, repo_root: Path) -> set[str]:
    """Return setup guidance lines that are paths and should not be wrapped."""
    data_raw = repo_root / "data_raw"
    exiobase_root = data_raw / "mrio" / "exiobase_3"
    return {
        str(repo_root),
        str(_summary_log_path(repo_root=repo_root)),
        str(data_raw / "carrying_capacities"),
        str(data_raw / "methodological_notes"),
        str(data_raw / "mrio" / "<source>" / "grouping"),
        str(exiobase_root),
        str(exiobase_root / "lcia" / "characterization_factors_matrices"),
        str(exiobase_root / "lcia" / "responsibility_periods"),
        str(exiobase_root / "lcia" / "carbon_accounts_covs"),
    }


def _wrapped_user_lines(*, lines: tuple[str, ...], repo_root: Path) -> tuple[str, ...]:
    """Return user text lines wrapped through the shared formatter, preserving paths."""
    paths = _path_lines(repo_root=repo_root)
    wrapped: list[str] = []
    for line in lines:
        if line in paths:
            wrapped.append(line)
        else:
            wrapped.extend(wrap_user_text_lines([line]))
    return tuple(wrapped)


def _formatted_guidance_lines(*, repo_root: Path) -> tuple[str, ...]:
    """Return user facing setup guidance lines for one workspace."""
    data_raw = repo_root / "data_raw"
    exiobase_root = data_raw / "mrio" / "exiobase_3"
    return (
        "Imported pyaesa prerequisites:",
        "",
        "Methodological notes, quick argument guides, and citation guidance:",
        str(data_raw / "methodological_notes"),
        "",
        "Files:",
        "Quick guide for selecting pyaesa function arguments for functional units and "
        "allocation methods:",
        "  -> 1_functional_units_and_allocation_methods.md",
        "Allocation paths figure used by the quick guide:",
        "  -> fig-asocc-paths.svg",
        "Detailed methodological guide with scientific references for functional units "
        "and allocation methods:",
        "  -> methodological_note__asocc_fus_allocation_methods.pdf",
        "Prospective allocation:",
        "  -> methodological_note__acc_prospective.pdf",
        "Uncertainty sources:",
        "  -> methodological_note__acc_uncertainty_sources.pdf",
        "Definition of steady state and dynamic carrying capacities:",
        "  -> methodological_note__steady_state__dynamic_cc.pdf",
        "Recommended citations:",
        "  -> recommended_citations.txt",
        "",
        "Carrying capacities:",
        str(data_raw / "carrying_capacities"),
        "",
        "Files:",
        "Carrying capacities available by default:",
        "  -> ef_3.1_cc_steady_state.csv",
        "  -> gwp100_lcia_cc_steady_state.csv",
        "  -> pb_lcia_cc_steady_state.csv",
        "Use only if you want to add custom carrying capacities not among the defaults:",
        "  -> README_add_custom_carrying_capacities.txt",
        "  -> name_lcia_cc_steady_state_template.csv",
        "  -> name_lcia_cc_steady_state_planetary_boundary_template.csv",
        "",
        "MRIOs:",
        "",
        "EXIOBASE sector definitions only:",
        str(exiobase_root),
        "",
        "Files:",
        "  -> sector_classification.xlsx",
        "",
        "Region and sector grouping:",
        str(data_raw / "mrio" / "<source>" / "grouping"),
        "",
        "Files:",
        "Region and sector grouping guide:",
        "  -> README_grouping.txt",
        "Region grouping template:",
        "  -> group_reg_template.csv",
        "Sector grouping template under each MRIO source folder:",
        "  -> group_sec_template.csv",
        "Region grouping tables available by default:",
        "  -> group_reg_eu27.csv",
        "  -> group_reg_world.csv",
        "EXIOBASE ixi sector grouping, electricity sectors grouped together:",
        "  -> group_sec_elec.csv",
        "EXIOBASE ixi sector grouping, electricity, gas, and water grouped to match OECD "
        "ICIO sector D resolution:",
        "  -> group_sec_oecd_d.csv",
        "",
        "EXIOBASE LCIA characterization matrices:",
        str(exiobase_root / "lcia" / "characterization_factors_matrices"),
        "",
        "Files:",
        "LCIA characterization matrices available by default:",
        "  -> pb_lcia.csv",
        "  -> gwp100_lcia.csv",
        "Use only if you want to add custom LCIA matrices not among the defaults:",
        "  -> README_add_custom_lcia_characterization_matrices.txt",
        "  -> name_lcia_template.csv",
        "  -> name_lcia_planetary_boundary_template.csv",
        "",
        "EXIOBASE historical responsibility periods used only by country level (L1) historical "
        "responsibility allocation method:",
        str(exiobase_root / "lcia" / "responsibility_periods"),
        "",
        "Files:",
        "Responsibility periods available by default:",
        "  -> pb_lcia_rps.csv",
        "  -> gwp100_lcia_rps.csv",
        "Use only if you want to add custom responsibility periods not among the defaults:",
        "  -> README_add_custom_lcia_responsibility_periods.txt",
        "  -> name_lcia_rps_template.csv",
        "  -> name_lcia_rps_planetary_boundary_template.csv",
        "",
        "EXIOBASE LCIA uncertainty (Rodrigues et al., 2018; Puig-Samper et al., 2025):",
        str(exiobase_root / "lcia" / "carbon_accounts_covs"),
        "",
        "Files:",
        "CoV CSVs for LCIA based allocation methods and IO-LCA LCIA result uncertainty:",
        "  -> reg_cbca_covs.csv",
        "  -> reg_cbca_covs_group_eu27.csv",
        "  -> reg_cbca_covs_group_world.csv",
        "  -> sec_cbca_covs.csv",
        "Use only if you want LCIA uncertainty with grouped or aggregate CoV labels:",
        "  -> README_grouped_and_aggregate_lcia_covs.txt",
        "",
        "Jupyter notebook tutorials are available on the pyaesa GitHub repository.",
    )


def _summary_log_lines(*, repo_root: Path) -> tuple[str, ...]:
    """Return the persisted setup summary log content."""
    return _wrapped_user_lines(
        repo_root=repo_root,
        lines=(
            "Workspace setup guidance information.",
            "Repository location:",
            str(repo_root),
            "",
            *_formatted_guidance_lines(repo_root=repo_root),
        ),
    )


def _write_summary_log(*, repo_root: Path) -> bool:
    """Write the setup summary log and return whether the file changed."""
    summary_path = _summary_log_path(repo_root=repo_root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(_summary_log_lines(repo_root=repo_root)) + "\n"
    if summary_path.exists() and summary_path.read_text(encoding="utf-8") == content:
        return False
    summary_path.write_text(content, encoding="utf-8", newline="\n")
    return True


def _setup_message_lines(*, repo_root: Path, imported: bool) -> tuple[str, ...]:
    """Return user facing lines printed when setup imports or records files."""
    setup_state_line = (
        "Imported package prerequisites."
        if imported
        else "Recorded workspace setup guidance information."
    )
    return _wrapped_user_lines(
        repo_root=repo_root,
        lines=(
            "Repository location:",
            str(repo_root),
            setup_state_line,
            "",
            *_formatted_guidance_lines(repo_root=repo_root),
            "",
            "Workspace setup guidance information recorded in:",
            str(_summary_log_path(repo_root=repo_root)),
        ),
    )


def _reuse_message_lines(*, repo_root: Path) -> tuple[str, ...]:
    """Return user facing lines printed when setup reuses all files unchanged."""
    return _wrapped_user_lines(
        repo_root=repo_root,
        lines=(
            "Workspace setup guidance information is available in:",
            str(_summary_log_path(repo_root=repo_root)),
        ),
    )


def set_workspace(
    top_path: str | Path,
    *,
    refresh: bool = False,
) -> None:
    """Set the active workspace repository and ensure prerequisites exist.

    The function resolves ``<top_path>/pyaesa``, creates that workspace
    repository root when missing, imports prerequisite files according to
    ``refresh``, and records the resolved path as the active workspace
    repository root. Later public functions read inputs and write outputs
    relative to this active workspace. Omit arguments to use their default.

    Packaged prerequisites are copied under the workspace ``data_raw/`` tree
    when missing or when ``refresh=True``. They include MRIO
    grouping templates, region matching tables for population and GDP
    processing, the EXIOBASE sector definition guide, EXIOBASE LCIA
    characterization matrices, LCIA responsibility period tables, carbon
    consumption based accounts coefficients of variation (CoV) tables, static
    carrying capacity CSVs and templates, local README guides for editable
    prerequisite folders, methodological PDF references, the recommended
    citation guide, and the functional unit and allocation method guide.
    Methodological assets are copied into ``data_raw/methodological_notes``
    from installed package resources.

    Args:
        top_path: Parent directory where the workspace repository root
            ``pyaesa/`` is created or reused. Accepts a string path or
            ``pathlib.Path``.
        refresh: If ``True``, overwrite the prerequisite files copied by
            ``set_workspace(...)`` under the resolved workspace ``data_raw``
            tree and rewrite the setup summary log. The scope is limited to
            prerequisite files such as default carrying capacities,
            methodological notes, citation guidance, grouping templates,
            matching tables, LCIA templates, and README guides. Raw downloads,
            processed outputs, and project outputs are not refreshed. Defaults
            to ``False``.

    Returns:
        None.

    Raises:
        OSError: If workspace directory or prerequisite file creation, copying,
            or refresh fails.

    Example:
        Initialize a workspace::

            from pathlib import Path
            from pyaesa import set_workspace

            # Windows example.
            set_workspace(Path(r"C:/Users/username/Documents/aesa_workspace"))
            # macOS example.
            # set_workspace(Path("/Users/username/Documents/aesa_workspace"))
    """
    repo_root = resolve_repo_root(top_path)
    repo_root.mkdir(parents=True, exist_ok=True)
    imported = import_prerequisites(repo_root=repo_root, refresh=refresh)
    summary_changed = _write_summary_log(repo_root=repo_root)
    set_default_repo_root(repo_root)
    if imported or summary_changed:
        for line in _setup_message_lines(repo_root=repo_root, imported=imported):
            print(line)
    else:
        for line in _reuse_message_lines(repo_root=repo_root):
            print(line)
