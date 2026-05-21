"""Output format contract for uncertainty public artifacts."""

UNCERTAINTY_OUTPUT_FORMATS: tuple[str, ...] = ("csv_compact", "parquet")
UNCERTAINTY_OUTPUT_FORMAT_SET: set[str] = set(UNCERTAINTY_OUTPUT_FORMATS)


def normalize_uncertainty_output_format(output_format: str) -> str:
    """Validate and normalize one uncertainty output format.

    Args:
        output_format: User requested table format.

    Returns:
        Canonical uncertainty output format.

    Raises:
        ValueError: If the format is not supported for uncertainty outputs.
    """
    normalized = str(output_format).strip().lower()
    if normalized not in UNCERTAINTY_OUTPUT_FORMAT_SET:
        raise ValueError(
            f"Unsupported uncertainty output_format '{output_format}'. "
            f"Use one of: {sorted(UNCERTAINTY_OUTPUT_FORMAT_SET)}."
        )
    return normalized


def suffix_for_uncertainty_output(output_format: str) -> str:
    """Return the filename suffix for one uncertainty output format."""
    normalized = normalize_uncertainty_output_format(output_format)
    return ".csv" if normalized == "csv_compact" else ".parquet"


def is_csv_compact_output(output_format: str) -> bool:
    """Return whether one normalized uncertainty output uses CSV storage."""
    return normalize_uncertainty_output_format(output_format) == "csv_compact"
