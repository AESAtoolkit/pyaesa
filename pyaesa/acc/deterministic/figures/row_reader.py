"""Public deterministic aCC table reader for figures."""

from pathlib import Path
from typing import Any, cast

import pandas as pd

from pyaesa.acc.figures.common import attach_common_columns
from pyaesa.acc.deterministic.state.metadata import load_run_metadata
from pyaesa.shared.tabular.l2_reuse_years import canonicalize_l2_reuse_year_column
from pyaesa.shared.tabular.table_io import read_table
from pyaesa.shared.tabular.wide_tables import melt_requested_year_value_rows


def load_deterministic_figure_rows(
    *, metadata_path: Path, coverage: dict[str, list[Any]] | None = None
) -> tuple[pd.DataFrame, list[int], str]:
    """Read deterministic aCC output files recorded in one branch manifest."""
    metadata = load_run_metadata(metadata_path)
    provenance = metadata["provenance"]
    artifacts = metadata["artifacts"]
    coverage_payload = coverage or {}
    requested_years = [
        int(year)
        for year in cast(list[Any], coverage_payload.get("years", provenance["requested_years"]))
    ]
    cc_source = str(provenance["cc_source"])
    frames = [
        _read_one_output(path=Path(str(path)), requested_years=requested_years, cc_source=cc_source)
        for path in artifacts["output_files"]
    ]
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for column, values in (coverage or {}).items():
        if column == "years" or column not in rows.columns:
            continue
        rows = rows.loc[rows[column].astype(str).isin([str(value) for value in values])].copy()
    rows["fu_code"] = str(metadata["arguments"]["fu_code"])
    return attach_common_columns(rows), requested_years, str(provenance["cc_type"])


def _read_one_output(*, path: Path, requested_years: list[int], cc_source: str) -> pd.DataFrame:
    """Read one deterministic aCC output table into long figure rows."""
    raw = canonicalize_l2_reuse_year_column(read_table(path=path), path=path)
    raw["lcia_method"] = str(cc_source)
    long_frame = melt_requested_year_value_rows(raw, requested_years=requested_years)
    long_frame["__source_path"] = str(path)
    return long_frame
