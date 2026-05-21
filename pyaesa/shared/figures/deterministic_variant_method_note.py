"""Shared deterministic variant compression method note."""

from pathlib import Path

import pandas as pd

from pyaesa.shared.figures.deterministic_variant_compressor import (
    ROLE_COLUMN,
    VARIANT_COLUMNS,
    compress_variants,
)

VARIANT_COMPRESSION_METHOD_FILENAME = "variant_compression_method.txt"

VARIANT_COMPRESSION_METHOD_TEXT = """Deterministic variant compression method

Applies to deterministic aSoCC, aCC, and ASR figures that retain reference year
and/or L2 reuse year variants.

Variant compression is used when the plotted deterministic figure scope contains
more than one reference year and/or L2 reuse year combination. Candidate
combinations are evaluated over comparable cells i. A comparable cell has the
same visible scientific identity in the figure scope, including allocation
method, impact category, SSP scope, year, carrying capacity bound, and model
scenario where those fields are present.

Reference year candidates are restricted before scoring. A reference year can
be retained only when it has rows for every plotted year in the deterministic
study window. Reference years that do not cover the full plotted window are not
candidate combinations for variant compression.

For each candidate combination v and comparable cell i:

p_i(v) = (x_i(v) - min_i) / (max_i - min_i)

where min_i and max_i are the minimum and maximum values across candidate
combinations for the same comparable cell. Cells with max_i = min_i are
excluded from the score.

Each comparable cell is weighted by its relative variant range:

w_i = (max_i - min_i) / (abs(max_i) + abs(min_i))

The score for each candidate combination is:

score(v) = sum_i(w_i p_i(v)) / sum_i(w_i)

The retained lower combination is the candidate with the lowest score. Figure
families that display an upper retained combination use the candidate with the
highest score.

When no comparable cell varies across candidate combinations, the candidates
are value equivalent in the displayed scope and deterministic variant ordering
selects the retained pair.
"""


def write_variant_compression_method_note(
    *,
    figures_root: Path,
    rows: pd.DataFrame,
) -> Path | None:
    """Write the shared method note when deterministic compression is visible."""
    if not _has_visible_variant_compression(rows):
        return None
    figures_root.mkdir(parents=True, exist_ok=True)
    path = figures_root / VARIANT_COMPRESSION_METHOD_FILENAME
    path.write_text(VARIANT_COMPRESSION_METHOD_TEXT, encoding="utf-8")
    return path


def _has_visible_variant_compression(rows: pd.DataFrame) -> bool:
    if rows.empty or not any(column in rows.columns for column in VARIANT_COLUMNS):
        return False
    compressed = compress_variants(rows)
    return ROLE_COLUMN in compressed.columns
