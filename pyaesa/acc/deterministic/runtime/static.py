"""Static aCC branch execution."""

from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

from pyaesa.shared.acc_asr_common.deterministic.downstream.inputs import LoadedAsoccShare
from pyaesa.shared.acc_asr_common.deterministic.downstream.selection import (
    static_compatible_share_frame,
)
from pyaesa.shared.acc_asr_common.deterministic.downstream.tabular_io import requested_year_columns
from pyaesa.shared.acc_asr_common.deterministic.status_labels import cc_branch_status_label
from pyaesa.shared.lcia.static_cc import read_static_cc, require_static_cc_bounds_available
from pyaesa.shared.lcia.paths import static_cc_csv_path
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.acc.shared.runtime.paths import build_acc_output_stem

from pyaesa.acc.deterministic.runtime.paths import (
    ACCDeterministicPathContext,
    acc_output_relative_dir,
    get_acc_output_dir,
)
from .static_cc import match_cc_for_share
from .tables import (
    materialize_acc_scope,
    resolve_acc_l1_l2_method,
    write_acc_output,
)


def process_static_acc(
    *,
    path_context: ACCDeterministicPathContext,
    public_result_root_name: str | None,
    cc_source: str,
    years: list[int],
    asocc_shares: list[LoadedAsoccShare],
    fmt: str,
    static_cc_bounds: list[str],
    status: StatusSink,
) -> tuple[int, int, list[str], list[Path], list[Path], Path]:
    """Run the static aCC branch writer."""
    cc_csv_path = static_cc_csv_path(lcia_method=cc_source)
    cc_df = read_static_cc(cc_csv_path)
    require_static_cc_bounds_available(
        cc_df=cc_df,
        requested_bounds=static_cc_bounds,
        context="deterministic_acc static carrying capacity request",
    )
    n_share = 0
    n_written = 0
    impacts: set[str] = set()
    output_root = get_acc_output_dir(
        context=path_context,
        public_result_root_name=public_result_root_name,
    )
    output_dirs: list[Path] = [output_root]
    output_files: list[Path] = []
    derive_max = "max_cc" in static_cc_bounds
    status_label = cc_branch_status_label(cc_source=cc_source, cc_type="static")
    for asocc_share in asocc_shares:
        status.show(f"[deterministic_acc] {status_label}: {asocc_share.display_name}")
        share_df = static_compatible_share_frame(
            asocc_share=asocc_share,
            share_frame=asocc_share.frame_wide,
            cc_source=cc_source,
        )
        if share_df is None:
            continue
        year_cols = requested_year_columns(
            share_df,
            requested_years=years,
        )
        if not year_cols:
            continue
        year_matrix = _numeric_year_matrix(frame=share_df, year_cols=year_cols)
        share_rows = share_df.drop(columns=year_cols).reset_index(drop=True)
        l1_l2_method = resolve_acc_l1_l2_method(
            frame=share_df,
            source_label=f"Static aCC aSoCC share '{asocc_share.display_name}'",
        )
        matches = match_cc_for_share(
            asocc_share.reference_path,
            cc_df,
            forced_impacts=(None if not asocc_share.impacts else list(asocc_share.impacts)),
        )
        row_blocks: list[pd.DataFrame] = []
        value_blocks: list[np.ndarray] = []
        for impact_code, min_val, max_val, impact_unit in matches:
            positions = _impact_row_positions(
                share_frame=share_df,
                impact_code=impact_code,
            )
            if positions is None:
                continue
            scoped_rows = share_rows.iloc[positions].reset_index(drop=True)
            scoped_values = year_matrix[positions]
            min_rows = materialize_acc_scope(
                scoped_rows,
                l1_l2_method=l1_l2_method,
                impact=impact_code,
                impact_unit=impact_unit,
            )
            min_rows["cc_bound"] = "min_cc"
            row_blocks.append(min_rows)
            value_blocks.append(scoped_values * float(min_val))
            if derive_max:
                max_rows = min_rows.copy()
                max_rows["cc_bound"] = "max_cc"
                row_blocks.append(max_rows)
                value_blocks.append(scoped_values * float(cast(float, max_val)))
            impacts.add(impact_code)
        branch_dir = output_root / acc_output_relative_dir(
            upstream_relative_dir=asocc_share.relative_dir
        )
        if not row_blocks:
            continue
        n_share += 1
        file_stem = build_acc_output_stem(
            base_stem=asocc_share.file_stem,
            cc_source=cc_source,
            cc_type="static",
        )
        file_path = branch_dir / f"{file_stem}.{fmt}"
        output = pd.concat(row_blocks, ignore_index=True)
        values = np.vstack(value_blocks)
        for column_index, year_col in enumerate(year_cols):
            output[year_col] = values[:, column_index]
        write_acc_output(
            output,
            file_path,
            fmt,
        )
        output_files.append(file_path)
        n_written += 1
    return n_share, n_written, sorted(impacts), output_dirs, output_files, cc_csv_path


def _numeric_year_matrix(*, frame: pd.DataFrame, year_cols: list[str]) -> np.ndarray:
    """Return requested year values for one static share table."""
    numeric = pd.DataFrame(frame.loc[:, year_cols].apply(pd.to_numeric, errors="raise"))
    return numeric.to_numpy(dtype=np.float64, copy=True)


def _impact_row_positions(
    *,
    share_frame: pd.DataFrame,
    impact_code: str,
) -> np.ndarray | None:
    """Return row positions compatible with one static CC impact."""
    if "impact" not in share_frame.columns:
        return np.arange(len(share_frame), dtype=np.int64)
    impact_series = pd.Series(share_frame.loc[:, "impact"], copy=False).astype("string").fillna("")
    non_empty = {str(value).strip() for value in impact_series.tolist() if str(value).strip()}
    if not non_empty:
        return np.arange(len(share_frame), dtype=np.int64)
    if str(impact_code).strip() not in non_empty:
        return None
    return np.flatnonzero(
        impact_series.str.strip().eq(str(impact_code).strip()).to_numpy(dtype=bool)
    )
