"""Persist final public summary text for deterministic and uncertainty scopes."""

from pathlib import Path

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

SUMMARY_LOG_FILENAME = "summary.log"


def summary_log_path(*, logs_dir: Path) -> Path:
    """Return the canonical summary log path inside one logs directory."""
    return ensure_file_parent(Path(logs_dir) / SUMMARY_LOG_FILENAME)


def write_summary_log(*, path: Path, summary: str) -> Path:
    """Write one final public summary text block and return its path."""
    resolved = ensure_file_parent(path)
    resolved.write_text(str(summary).rstrip() + "\n", encoding="utf-8")
    return resolved
