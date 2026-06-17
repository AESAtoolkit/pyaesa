"""Shared output tree helpers for figure products."""

from pathlib import Path

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.tabular.scalars import sanitize_token


def figures_root_for_run(*, run_root: Path) -> Path:
    """Return the canonical figure root for one resolved run scope."""
    return Path(run_root) / "figs"


def output_file_path(*, base_path: Path, output_format: str) -> Path:
    """Return the canonical output file path for one figure export request."""
    base = Path(base_path)
    suffix = str(output_format).strip().lstrip(".")
    return ensure_file_parent(base.parent / f"{base.name}.{suffix}")


def output_paths(*, base_path: Path, output_format: str) -> list[Path]:
    """Return the canonical output file paths for one figure export request."""
    return [output_file_path(base_path=base_path, output_format=output_format)]


def top_level_figure_dir(*, figures_root: Path, folder: str) -> Path:
    """Return one top level figure folder."""
    return Path(figures_root) / str(folder).strip()


def strip_lcia_method_suffix(*, stem: str, lcia_methods: list[str] | None) -> str:
    """Return one per method folder stem without a trailing LCIA method token."""
    normalized = str(stem).strip()
    if not normalized or not lcia_methods:
        return normalized
    suffixes = sorted(
        {
            f"__{str(lcia_method_label).strip()}"
            for lcia_method_label in lcia_methods
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
    method_text = str(lcia_method).strip()
    suffix = sanitize_token(method_text) if method_text else "item"
    if base.endswith(f"__{suffix}"):
        return base
    return f"{base}__{suffix}"
