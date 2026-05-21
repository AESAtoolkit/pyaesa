"""Shared output tree helpers for figure products."""

import re
from pathlib import Path

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def figures_root_for_run(*, run_root: Path) -> Path:
    """Return the canonical figure root for one resolved run scope."""
    return Path(run_root) / "figures"


def output_file_path(*, base_path: Path, output_format: str) -> Path:
    """Return the canonical output file path for one figure export request."""
    base = Path(base_path)
    suffix = str(output_format).strip().lstrip(".")
    return ensure_file_parent(base.parent / f"{base.name}.{suffix}")


def output_paths(*, base_path: Path, output_format: str) -> list[Path]:
    """Return the canonical output file paths for one figure export request."""
    return [output_file_path(base_path=base_path, output_format=output_format)]


def year_token(years: list[int]) -> str:
    """Return a deterministic filename token from a year list."""
    unique_years = sorted({int(year) for year in years})
    if not unique_years:
        return "years_none"
    ranges: list[tuple[int, int]] = []
    start = unique_years[0]
    end = unique_years[0]
    for year in unique_years[1:]:
        if year == end + 1:
            end = year
            continue
        ranges.append((start, end))
        start = year
        end = year
    ranges.append((start, end))
    pieces: list[str] = []
    for start_year, end_year in ranges:
        pieces.append(str(start_year) if start_year == end_year else f"{start_year}-{end_year}")
    text = "_".join(pieces)
    text = _TOKEN_RE.sub("_", text).strip("._-")
    return f"years_{text or 'unknown'}"


def top_level_figure_dir(*, figures_root: Path, folder: str) -> Path:
    """Return one top level figure folder."""
    return Path(figures_root) / str(folder).strip()


def deterministic_figure_dir(
    *,
    figures_root: Path,
    timescale: str,
    role: str | None = None,
) -> Path:
    """Return a deterministic figure directory."""
    ts = str(timescale).strip()
    if role is None:
        return Path(figures_root) / ts
    return Path(figures_root) / str(role).strip() / ts


def uncertainty_figure_dir(
    *,
    figures_root: Path,
    timescale: str,
    family: str,
    role: str | None = None,
) -> Path:
    """Return an uncertainty figure directory."""
    ts = str(timescale).strip()
    fam = str(family).strip()
    if role is None:
        return Path(figures_root) / ts / fam
    return Path(figures_root) / str(role).strip() / ts / fam


def family_figure_dir(
    *,
    figures_root: Path,
    family: str,
    role: str | None = None,
    granularity: str | None = None,
) -> Path:
    """Return the canonical uncertainty family directory."""
    family_token = str(family).strip()
    return uncertainty_figure_dir(
        figures_root=figures_root,
        timescale=str(granularity).strip(),
        family=family_token,
        role=role,
    )


def strip_lcia_method_suffix(*, stem: str, lcia_methods: list[str] | None) -> str:
    """Return one per method folder stem without a trailing LCIA method token."""
    normalized = str(stem).strip()
    if not normalized or not lcia_methods:
        return normalized
    suffixes = sorted(
        {
            suffix
            for lcia_method_label in lcia_methods
            for suffix in (
                f"__{str(lcia_method_label).strip()}",
                f"_{str(lcia_method_label).strip()}",
            )
            if str(lcia_method_label).strip()
        },
        key=len,
        reverse=True,
    )
    for suffix in suffixes:
        if normalized.endswith(suffix):
            candidate = normalized[: -len(suffix)].strip("_")
            return candidate or normalized
    return normalized


def scope_filename_stem(
    *,
    base_stem: str,
    lcia_method: str | None = None,
) -> str:
    """Return one figure filename stem with an LCIA method token when applicable."""
    base = str(base_stem).strip()
    if lcia_method is None:
        return base
    method_token = _TOKEN_RE.sub("_", str(lcia_method).strip()).strip("._-")
    suffix = method_token or "item"
    if base.endswith(f"__{suffix}") or base.endswith(f"_{suffix}"):
        return base
    return f"{base}__{suffix}"
