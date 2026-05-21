"""Helpers for strict persisted file path handling."""

from pathlib import Path
from typing import cast

_TABLE_SUFFIXES = {".csv", ".parquet", ".pickle"}


def normalize_persisted_paths(*, raw_paths: object) -> list[Path]:
    """Return normalized package-owned persisted paths from one metadata field."""
    if raw_paths is None:
        return []
    out: list[Path] = []
    for raw_path in cast(list[object], raw_paths):
        if isinstance(raw_path, Path):
            out.append(raw_path)
            continue
        out.append(Path(str(raw_path).strip()))
    return out


def scoped_existing_table_paths(
    *,
    raw_paths: object,
    root: Path,
    field_name: str,
) -> list[Path]:
    """Return existing persisted table paths under one root."""
    root_path = Path(root).resolve()
    seen: set[Path] = set()
    ordered: list[Path] = []
    outside_root: list[str] = []
    unsupported_suffixes: list[str] = []
    missing_paths: list[str] = []
    duplicate_paths: list[str] = []
    for path in normalize_persisted_paths(raw_paths=raw_paths):
        resolved = path.resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError:
            outside_root.append(str(resolved))
            continue
        if resolved.suffix.lower() not in _TABLE_SUFFIXES:
            unsupported_suffixes.append(str(resolved))
            continue
        if not resolved.exists():
            missing_paths.append(str(resolved))
            continue
        if resolved in seen:
            duplicate_paths.append(str(resolved))
            continue
        seen.add(resolved)
        ordered.append(resolved)
    if outside_root or unsupported_suffixes or missing_paths or duplicate_paths:
        details: list[str] = []
        if outside_root:
            details.append(f"outside root {root_path}: {outside_root[:10]}")
        if unsupported_suffixes:
            details.append(
                f"unsupported tabular suffixes (allowed: {sorted(_TABLE_SUFFIXES)}): "
                f"{unsupported_suffixes[:10]}"
            )
        if missing_paths:
            details.append(f"missing files: {missing_paths[:10]}")
        if duplicate_paths:
            details.append(f"duplicate entries: {duplicate_paths[:10]}")
        raise ValueError(
            f"Metadata field '{field_name}' contains invalid persisted table paths; "
            + "; ".join(details)
            + "."
        )
    return sorted(ordered)
