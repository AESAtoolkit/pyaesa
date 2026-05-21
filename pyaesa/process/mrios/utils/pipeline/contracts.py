"""Dataclasses and source configuration for MRIO processing."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from pyaesa.download.mrios.utils.source_registry import (
    get_mrio_entry,
    iter_mrio_entries,
)
from pyaesa.process.mrios.utils.raw_corrections.runtime import build_scope_summary
from pyaesa.shared.runtime.text import extend_user_text_lines


@dataclass(frozen=True)
class SourceConfig:
    """Per source processing configuration."""

    requires_characterization: bool
    required_core: Tuple[str, ...]
    required_extensions: Tuple[str, ...]
    exio_system: str = ""


@dataclass
class ProcessReportMRIO:
    """Outcome of a MRIO processing run."""

    source: str
    requested: List[int]
    saved_root: Path | None = None
    processed: List[int] = field(default_factory=list)
    skipped_already_saved: List[int] = field(default_factory=list)
    errors: Dict[int, str] = field(default_factory=dict)
    saved_dirs: Dict[int, Path] = field(default_factory=dict)
    clipping_log_path: Path | None = None
    clipping_unit: str | None = None
    y_clip_count: int = 0
    y_clip_abs_sum: float = 0.0
    y_clip_abs_max: float = 0.0
    f_clip_count: int = 0
    f_clip_abs_sum: float = 0.0
    f_clip_abs_max: float = 0.0
    lcia_missing_by_year: Dict[int, Dict[str, List[str]]] = field(default_factory=dict)
    raw_corrected_value_row_count: int = 0
    raw_corrected_value_scopes: List[Dict[str, object]] = field(default_factory=list)
    raw_corrected_value_log_paths: List[Path] = field(default_factory=list)

    def missing(self) -> List[int]:
        """Return requested years that were neither processed nor skipped."""
        handled = set(self.processed) | set(self.skipped_already_saved)
        return [year for year in self.requested if year not in handled]

    def _format_year_ranges(self, years: List[int]) -> str:
        """Format years as compact ranges (e.g. 1995-1998, 2000)."""
        if not years:
            return "[]"
        values = sorted({int(year) for year in years})
        ranges: List[str] = []
        start = values[0]
        prev = values[0]
        for year in values[1:]:
            if year == prev + 1:
                prev = year
                continue
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = year
            prev = year
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ", ".join(ranges)

    def _format_year_ranges_with_count(self, years: List[int]) -> str:
        """Format years as compact ranges and append the year count."""
        unique_years = sorted({int(year) for year in years})
        return f"{self._format_year_ranges(unique_years)} ({len(unique_years)} year(s))"

    def _build_raw_corrected_value_summaries(self) -> List[str]:
        """Return grouped raw corrected values summaries for the final report."""
        if not self.raw_corrected_value_scopes:
            return []
        grouped: Dict[Tuple[str, str, str, str, str], List[int]] = {}
        for scope in self.raw_corrected_value_scopes:
            key = (
                str(scope["region"]),
                str(scope["extension"]),
                str(scope["stressor_family"]),
                str(scope["correction_method"]),
                str(scope["correction_reason"]),
            )
            grouped.setdefault(key, []).append(int(str(scope["year"])))
        summaries: List[str] = []
        for key in sorted(grouped):
            region, extension, stressor_family, correction_method, _correction_reason = key
            years = sorted({int(year) for year in grouped[key]})
            summaries.append(
                build_scope_summary(
                    source=self.source,
                    region=region,
                    extension=extension,
                    stressor_family=stressor_family,
                    correction_method=correction_method,
                    years=years,
                )
            )
        return summaries

    def __str__(self) -> str:
        """Return a human readable summary."""
        source_key = str(self.source).strip()
        source_reason_label = get_mrio_entry(source_key).display_label

        lines = [
            f"[{self.source}] Summary:",
            f"  Requested: {self._format_year_ranges_with_count(self.requested)}",
            f"  Processed: {self._format_year_ranges_with_count(self.processed)}",
        ]
        if self.skipped_already_saved:
            lines.append(
                "  Skipped existing files: "
                f"{self._format_year_ranges_with_count(self.skipped_already_saved)}"
            )
        if self.saved_root is not None:
            lines.append(f"  Saved to: {self.saved_root}")
        raw_corrected_value_summaries = self._build_raw_corrected_value_summaries()
        if raw_corrected_value_summaries:
            lines.append("")
            lines.append("  EXIOBASE data corrections applied:")
            for summary in raw_corrected_value_summaries:
                extend_user_text_lines(lines, f"    - {summary}")
            if self.raw_corrected_value_log_paths:
                if len(self.raw_corrected_value_log_paths) == 1:
                    lines.append(
                        "  See raw corrected values log for full row details: "
                        f"{self.raw_corrected_value_log_paths[0]}"
                    )
                else:
                    lines.append("  See raw corrected values logs for full row details:")
                    for path in self.raw_corrected_value_log_paths:
                        lines.append(f"    - {path}")
        if self.lcia_missing_by_year:
            lines.append("")
            lines.append("  LCIA skipped:")
            for year in sorted(self.lcia_missing_by_year):
                missing_by_method = self.lcia_missing_by_year.get(year, {})
                if not missing_by_method:
                    continue
                lcia_method_parts: List[str] = []
                for lcia_method, missing in sorted(missing_by_method.items()):
                    missing_unique = sorted({str(name) for name in missing})
                    if not missing_unique:
                        continue
                    if len(missing_unique) == 1:
                        reason = f"{missing_unique[0]} extension missing in {source_reason_label}"
                    else:
                        listed = ", ".join(missing_unique[:3])
                        if len(missing_unique) > 3:
                            listed = f"{listed}, +{len(missing_unique) - 3} more"
                        reason = f"extensions missing in {source_reason_label}: {listed}"
                    lcia_method_parts.append(f"{lcia_method}: {reason}")
                if lcia_method_parts:
                    extend_user_text_lines(lines, f"    {year}: {'; '.join(lcia_method_parts)}")
        if self.processed and self.clipping_log_path is not None:
            lines.append("")
            lines.append("  Clipping policy:")
            lines.append(
                "    - Y and F_factor_inputs, after summing across all their "
                "categories, are clipped at 0."
            )
            lines.append("    - Purpose: avoid negative allocated shares in deterministic_asocc.")
            lines.append(f"  Clipping log saved to: {self.clipping_log_path}")
        if self.errors:
            failed_years = sorted(self.errors.keys())
            lines.append(f"  Errors: {self._format_year_ranges_with_count(failed_years)}")
            lines.append("  See report.errors for details.")
        return "\n".join(lines)

    __repr__ = __str__


_CORE_MATRICES: Tuple[str, ...] = ("A", "G", "L", "Y", "Z", "x", "unit")
UNCASEXT_INTERMEDIATE_CORE_MATRICES: Tuple[str, ...] = ("A", "G", "L", "Z", "unit")


def build_source_configs() -> Dict[str, SourceConfig]:
    """Build source configuration map keyed by MRIO source."""
    configs: Dict[str, SourceConfig] = {}
    for entry in iter_mrio_entries():
        if entry.family == "exiobase":
            configs[entry.source_key] = SourceConfig(
                requires_characterization=True,
                required_core=_CORE_MATRICES,
                required_extensions=(),
                exio_system=str(entry.system),
            )
            continue
        configs[entry.source_key] = SourceConfig(
            requires_characterization=False,
            required_core=_CORE_MATRICES,
            required_extensions=(),
        )
    return configs


SOURCE_CONFIGS = build_source_configs()
