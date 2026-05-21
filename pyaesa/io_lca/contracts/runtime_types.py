"""Typed reports and immutable runtime payloads for IO-LCA runs."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class IOLCAReport:
    """Summary report returned by ``deterministic_io_lca``."""

    source: str
    fu_code: str
    years: list[int]
    lcia_methods: list[str]
    main_result_paths: list[Path]
    origin_paths: list[Path]
    stage_paths: list[Path]
    skipped_method_years: dict[str, dict[int, str]]
    metadata_path: Path
    figure_paths: list[Path] = field(default_factory=list)
    summary_lines: list[str] = field(default_factory=list)
    reuse_status: str = "computed"

    def __str__(self) -> str:
        """Return run summary text."""
        return "\n".join(self.summary_lines)

    __repr__ = __str__
