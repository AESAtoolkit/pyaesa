"""MRIO region aggregation ownership for aSoCC data loaders and equations."""

from pandas.errors import ParserError

from pyaesa.process.mrios.utils.aggregation.aggregation import read_agg_map
from pyaesa.process.mrios.utils.io.paths import _get_agg_map_path


def load_region_agg_mapping(
    *,
    source_key: str,
    agg_version: str,
) -> dict[str, str]:
    """Load the original-to-aggregated MRIO region mapping for one source/version."""
    agg_map_path = _get_agg_map_path(
        source_key,
        kind="reg",
        agg_version=agg_version,
    )
    try:
        agg_df = read_agg_map(agg_map_path)
    except (FileNotFoundError, OSError, UnicodeError, ParserError, ValueError) as exc:
        raise ValueError(
            f"Failed to read region MRIO aggregation and disaggregation CSV: {agg_map_path}. {exc}"
        ) from exc
    return dict(zip(agg_df["original_classification"], agg_df["aggregated_mrio"]))
