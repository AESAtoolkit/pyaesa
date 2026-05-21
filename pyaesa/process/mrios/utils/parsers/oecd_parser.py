"""OECD specific ownership for MRIO processing."""

from pathlib import Path
from typing import Any, Callable, cast

import pymrio
from pyaesa.download.mrios.utils.paths import _get_oecd_csv_path


def _parse_oecd_year(
    full_dir: Path,
    year: int,
    *,
    parser: Callable[[str], Any] = pymrio.parse_oecd,
) -> pymrio.IOSystem:
    """Parse a single OECD ICIO v2025 CSV.

    Args:
        full_dir (Path): Directory containing downloaded OECD CSVs.
        year (int): Four digit year to parse.

    Returns:
        pymrio.IOSystem: Parsed IOSystem instance.

    Raises:
        FileNotFoundError: If the expected CSV is missing.
    """
    csv_path = _get_oecd_csv_path(Path(full_dir), int(year))
    if not csv_path.exists():
        raise FileNotFoundError(
            f"OECD ICIO2025 CSV for {year} not found at {csv_path}. Run MRIO downloads first."
        )
    return cast(pymrio.IOSystem, parser(str(csv_path)))
