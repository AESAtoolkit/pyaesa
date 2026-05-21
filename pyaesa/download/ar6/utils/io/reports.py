"""Report objects for AR6 raw downloads."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DownloadReportAR6:
    """Outcome of one AR6 raw data download run."""

    database: str
    raw_root: Path
    logs_dir: Path
    metadata_path: Path
    downloaded_assets: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        """Return a human readable run summary."""
        lines = [
            f"[{self.database}] Summary:",
            f"  Raw data folder: {self.raw_root}",
            f"  Logs folder: {self.logs_dir}",
        ]
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)} asset(s)")
            lines.append("  See report.errors for details.")
        return "\n".join(lines)

    __repr__ = __str__
