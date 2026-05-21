"""Public function for preparing project scoped external input scaffolds."""

from dataclasses import dataclass
from pathlib import Path

from pyaesa.asocc.runtime.paths.external import get_asocc_external_root
from pyaesa.external_inputs.asocc.templates.templates import ensure_external_asocc_templates
from pyaesa.external_inputs.lca.paths import external_lca_root
from pyaesa.external_inputs.lca.templates import ensure_external_lca_templates
from pyaesa.workspace_initialisation.workspace import project_outputs_root
from pyaesa.shared.runtime.manifest_contract import manifest_digest, manifest_json_value
from pyaesa.shared.runtime.metadata.json import write_json_dict
from pyaesa.shared.runtime.reporting.composite_phase_index import PHASE_A_LCA, PHASE_B1_ASOCC
from pyaesa.shared.runtime.reporting.output_inventory import inventory_item, inventory_lines
from pyaesa.shared.runtime.reporting.summary import document, render_summary, section
from pyaesa.shared.runtime.reporting.summary_log import summary_log_path, write_summary_log
from pyaesa.shared.runtime.text import print_user_text_line

_SCOPE_MANIFEST_FILENAME = "scope_manifest.json"

_EXTERNAL_ASOCC_INPUTS = (
    ("deterministic/", "runnable deterministic examples and project external aSoCC files"),
    ("monte_carlo/", "runnable Monte Carlo examples and project external aSoCC runs"),
)
_EXTERNAL_ASOCC_GUIDANCE = (
    ("README_external_asocc_templates.txt", "method syntax, filenames, schemas, and staging rules"),
)
_EXTERNAL_ASOCC_EXAMPLES = (
    ("deterministic/CO(S).csv", "deterministic one step example for base years"),
    ("deterministic/CO(S)__ssp2.csv", "deterministic one step SSP2 example"),
    (
        "deterministic/l1_AR(E)_l2_UT(S)__ef_3.1.csv",
        "deterministic two step EF 3.1 example",
    ),
    ("monte_carlo/CO(S).csv", "normal CSV Monte Carlo one step example"),
    ("monte_carlo/CO(S)/", "compact CSV Monte Carlo one step example"),
    (
        "monte_carlo/l1_AR(E)_l2_UT(S)__ef_3.1.csv",
        "normal CSV Monte Carlo two step EF 3.1 example",
    ),
    (
        "monte_carlo/l1_AR(E)_l2_UT(S)__ef_3.1/",
        "compact CSV Monte Carlo two step EF 3.1 example",
    ),
)
_EXTERNAL_LCA_INPUTS = (
    ("deterministic/", "runnable deterministic examples and project external LCA files"),
    ("monte_carlo/", "runnable Monte Carlo examples and project external LCA runs"),
)
_EXTERNAL_LCA_GUIDANCE = (
    ("README_external_lca_templates.txt", "version syntax, filenames, schemas, and staging rules"),
)
_EXTERNAL_LCA_EXAMPLES = (
    ("deterministic/template__ef_3.1.csv", "deterministic EF 3.1 external LCA example"),
    ("deterministic/template__ef_3.1__ssp2.csv", "deterministic EF 3.1 SSP2 example"),
    ("monte_carlo/template__ef_3.1.csv", "normal CSV Monte Carlo EF 3.1 example"),
    ("monte_carlo/template__ef_3.1/", "compact CSV Monte Carlo EF 3.1 example"),
)


@dataclass(frozen=True)
class ExternalInputPreparationReport:
    """Summary returned by ``prepare_external_inputs(...)``."""

    project_root: Path
    external_asocc_root: Path
    external_asocc_templates_dir: Path
    external_lca_root: Path
    external_lca_templates_dir: Path
    metadata_path: Path
    summary_log: Path
    summary_lines: tuple[str, ...] = tuple()

    def __str__(self) -> str:
        return render_summary(
            document(
                "prepare_external_inputs",
                lines=self.summary_lines,
                sections=(
                    section(
                        PHASE_A_LCA,
                        children=(
                            section(
                                "external_lca",
                                lines=(
                                    f"Input folder: {self.external_lca_root}",
                                    *inventory_lines(
                                        [
                                            inventory_item(
                                                folder="templates",
                                                content="external LCA README guidance",
                                            ),
                                            inventory_item(
                                                folder="deterministic",
                                                content="deterministic external LCA examples",
                                            ),
                                            inventory_item(
                                                folder="monte_carlo",
                                                content="Monte Carlo external LCA examples",
                                            ),
                                        ]
                                    ),
                                ),
                            ),
                        ),
                    ),
                    section(
                        PHASE_B1_ASOCC,
                        children=(
                            section(
                                "external_asocc",
                                lines=(
                                    f"Input folder: {self.external_asocc_root}",
                                    *inventory_lines(
                                        [
                                            inventory_item(
                                                folder="templates",
                                                content="external aSoCC README guidance",
                                            ),
                                            inventory_item(
                                                folder="deterministic",
                                                content="deterministic external aSoCC examples",
                                            ),
                                            inventory_item(
                                                folder="monte_carlo",
                                                content="Monte Carlo external aSoCC examples",
                                            ),
                                        ]
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        )

    __repr__ = __str__


@dataclass(frozen=True)
class _ResolvedExternalInputScope:
    """Resolved project scoped external input scaffold target."""

    project_root: Path
    external_asocc_root: Path
    external_lca_root: Path
    arguments: dict[str, str]


def _print_inventory_section(
    *,
    title: str,
    folder: Path,
    entries: tuple[tuple[str, str], ...],
) -> None:
    """Print one external input inventory section."""
    print_user_text_line(title)
    print_user_text_line(str(folder))
    for name, description in entries:
        print_user_text_line(f"  -> {name}: {description}.")


def _normalize_project_name(project_name: str) -> str:
    """Return the validated project name for external input preparation."""
    if not isinstance(project_name, str):
        raise ValueError("project_name must be a non-empty string.")
    project_name_clean = project_name.strip()
    if not project_name_clean:
        raise ValueError("project_name must be a non-empty string.")
    return project_name_clean


def _resolve_preparation_scope(
    *,
    project_name: str,
) -> _ResolvedExternalInputScope:
    """Resolve project scoped external roots for one preparation request."""
    project_name_clean = _normalize_project_name(project_name)
    project_root = project_outputs_root(project_name=project_name_clean)
    return _ResolvedExternalInputScope(
        project_root=project_root,
        external_asocc_root=get_asocc_external_root(proj_base=project_root),
        external_lca_root=external_lca_root(project_base=project_root),
        arguments={"project_name": project_name_clean},
    )


def _stage_external_templates(
    *,
    resolved_scope: _ResolvedExternalInputScope,
) -> tuple[Path, Path]:
    """Ensure packaged external guidance and runnable examples exist for the scope."""
    return (
        ensure_external_asocc_templates(external_dir=resolved_scope.external_asocc_root),
        ensure_external_lca_templates(external_dir=resolved_scope.external_lca_root),
    )


def _build_preparation_report(
    *,
    resolved_scope: _ResolvedExternalInputScope,
    external_asocc_templates_dir: Path,
    external_lca_templates_dir: Path,
) -> ExternalInputPreparationReport:
    """Build the public external input preparation report."""
    proj_base = resolved_scope.project_root
    logs_dir = _preparation_logs_dir(project_root=proj_base)
    summary_lines = (
        f"Output folder: {proj_base}",
        *inventory_lines(
            [
                inventory_item(folder="logs", content="summary log"),
            ]
        ),
    )
    return ExternalInputPreparationReport(
        project_root=proj_base,
        external_asocc_root=resolved_scope.external_asocc_root,
        external_asocc_templates_dir=external_asocc_templates_dir,
        external_lca_root=resolved_scope.external_lca_root,
        external_lca_templates_dir=external_lca_templates_dir,
        metadata_path=_preparation_manifest_path(logs_dir=logs_dir),
        summary_log=summary_log_path(logs_dir=logs_dir),
        summary_lines=summary_lines,
    )


def _preparation_logs_dir(*, project_root: Path) -> Path:
    """Return the logs folder for one external input preparation scope."""
    return Path(project_root) / "prepare_external_inputs_log"


def _preparation_manifest_path(*, logs_dir: Path) -> Path:
    """Return the scope manifest path for one external input preparation scope."""
    return Path(logs_dir) / _SCOPE_MANIFEST_FILENAME


def _write_preparation_metadata(
    *,
    resolved_scope: _ResolvedExternalInputScope,
    report: ExternalInputPreparationReport,
) -> None:
    """Persist the public external input preparation manifest and summary."""
    arguments = manifest_json_value(resolved_scope.arguments)
    identity_key = manifest_digest(
        {"function": "prepare_external_inputs", **arguments},
    )
    payload = {
        "function": "prepare_external_inputs",
        "arguments": arguments,
        "execution": {"status": "complete"},
        "reuse": {"identity_key": identity_key},
        "artifacts": {
            "external_asocc_templates_dir": str(report.external_asocc_templates_dir),
            "external_lca_templates_dir": str(report.external_lca_templates_dir),
            "scope_manifest": str(report.metadata_path),
            "summary_log": str(report.summary_log),
        },
        "provenance": {
            "project_root": str(report.project_root),
            "external_asocc_root": str(report.external_asocc_root),
            "external_lca_root": str(report.external_lca_root),
        },
    }
    write_json_dict(report.metadata_path, payload)
    write_summary_log(path=report.summary_log, summary=str(report))


def prepare_external_inputs(
    *,
    project_name: str,
) -> ExternalInputPreparationReport:
    """Import external allocated shares of carrying capacities (aSoCC) and LCA inputs.

    The function writes README guidance and runnable CSV examples under
    ``project_name`` so expected file formats and real input folders can be
    inspected directly. External allocated shares of carrying capacities
    (aSoCC) inputs let aSoCC, allocated carrying capacity (aCC), and absolute
    sustainability ratio (ASR) workflows mix user provided allocated shares
    with package computed shares. External LCA inputs can be used as the ASR
    numerator instead of IO-LCA. The imported guidance and examples cover
    deterministic and Monte Carlo external inputs, retrospective years, and
    prospective SSP scenario files. Existing user staged files are preserved.
    Omit arguments to use their default.

    Args:
        project_name: Required project name used to build
            ``<repo>/<project_name>``. External input folders are shared by the
            downstream aSoCC, aCC, and ASR calls that use the same project
            name.

    Returns:
        ``ExternalInputPreparationReport`` with the resolved project root and
        external aSoCC and external LCA scaffold locations.

    Raises:
        ValueError: If ``project_name`` is not a non-empty string.

    Example:
        Prepare external input guidance and runnable examples for one project::

            from pyaesa import prepare_external_inputs

            prepare_external_inputs(project_name="demo")
    """
    resolved_scope = _resolve_preparation_scope(
        project_name=project_name,
    )
    print_user_text_line("[prepare_external_inputs] External input guidance and examples imported")
    external_asocc_templates_dir, external_lca_templates_dir = _stage_external_templates(
        resolved_scope=resolved_scope,
    )
    report = _build_preparation_report(
        resolved_scope=resolved_scope,
        external_asocc_templates_dir=external_asocc_templates_dir,
        external_lca_templates_dir=external_lca_templates_dir,
    )
    _write_preparation_metadata(resolved_scope=resolved_scope, report=report)
    print_user_text_line("")
    _print_inventory_section(
        title="External aSoCC input folders:",
        folder=report.external_asocc_root,
        entries=_EXTERNAL_ASOCC_INPUTS,
    )
    print_user_text_line("")
    _print_inventory_section(
        title="External aSoCC runnable examples:",
        folder=report.external_asocc_root,
        entries=_EXTERNAL_ASOCC_EXAMPLES,
    )
    print_user_text_line("")
    _print_inventory_section(
        title="External aSoCC README guidance:",
        folder=report.external_asocc_templates_dir,
        entries=_EXTERNAL_ASOCC_GUIDANCE,
    )
    print_user_text_line("")
    _print_inventory_section(
        title="External LCA input folders:",
        folder=report.external_lca_root,
        entries=_EXTERNAL_LCA_INPUTS,
    )
    print_user_text_line("")
    _print_inventory_section(
        title="External LCA runnable examples:",
        folder=report.external_lca_root,
        entries=_EXTERNAL_LCA_EXAMPLES,
    )
    print_user_text_line("")
    _print_inventory_section(
        title="External LCA README guidance:",
        folder=report.external_lca_templates_dir,
        entries=_EXTERNAL_LCA_GUIDANCE,
    )
    return report
