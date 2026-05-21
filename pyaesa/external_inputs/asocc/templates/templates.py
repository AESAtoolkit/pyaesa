"""Guidance and runnable example scaffold for project external aSoCC inputs."""

from pathlib import Path

from pyaesa.external_inputs.templates import copy_packaged_file, copy_packaged_files
from pyaesa.shared.runtime.io.filesystem import ensure_dir

_EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples" / "asocc"
_GUIDANCE_FILE = (
    Path(__file__).resolve().parents[2]
    / "example_guidance"
    / "asocc"
    / "external_asocc_guidance.txt"
)


def ensure_external_asocc_templates(*, external_dir: Path) -> Path:
    """Ensure project external aSoCC README guidance and runnable examples exist."""
    external_dir = ensure_dir(external_dir)
    copy_packaged_files(source_dir=_EXAMPLES_ROOT, target_dir=external_dir)
    templates_dir = ensure_dir(external_dir / "templates")
    copy_packaged_file(
        source_file=_GUIDANCE_FILE,
        target_file=templates_dir / "README_external_asocc_templates.txt",
    )
    return templates_dir
