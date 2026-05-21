"""Shared contracts for package wide tabular file formats."""

TABULAR_OUTPUT_FORMATS: tuple[str, ...] = ("csv", "pickle", "parquet")
TABULAR_OUTPUT_FORMAT_SET: set[str] = set(TABULAR_OUTPUT_FORMATS)
TABULAR_SUFFIXES: tuple[str, ...] = tuple(f".{fmt}" for fmt in TABULAR_OUTPUT_FORMATS)
TABULAR_SUFFIX_SET: set[str] = set(TABULAR_SUFFIXES)


def normalize_tabular_output_format(output_format: str) -> str:
    """Validate and normalize one tabular output format."""
    normalized = str(output_format).strip().lower()
    if normalized not in TABULAR_OUTPUT_FORMAT_SET:
        raise ValueError(
            f"Unsupported output_format '{output_format}'. "
            f"Use one of: {sorted(TABULAR_OUTPUT_FORMAT_SET)}."
        )
    return normalized


def suffix_for_tabular_output(output_format: str) -> str:
    """Return the filename suffix for one normalized tabular output format."""
    return f".{normalize_tabular_output_format(output_format)}"
