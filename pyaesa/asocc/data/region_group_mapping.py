"""MRIO region grouping ownership for aSoCC data loaders and equations."""

from pandas.errors import ParserError

from pyaesa.process.mrios.utils.grouping.grouping import read_group_map
from pyaesa.process.mrios.utils.io.paths import _get_group_map_path


def load_region_group_mapping(
    *,
    source_key: str,
    group_version: str,
) -> dict[str, str]:
    """Load the original-to-grouped MRIO region mapping for one source/version."""
    group_map_path = _get_group_map_path(
        source_key,
        kind="reg",
        group_version=group_version,
    )
    try:
        group_df = read_group_map(group_map_path)
    except (FileNotFoundError, OSError, UnicodeError, ParserError, ValueError) as exc:
        raise ValueError(f"Failed to read region grouping CSV: {group_map_path}. {exc}") from exc
    return dict(zip(group_df["original_classification"], group_df["grouped_mrio"]))
