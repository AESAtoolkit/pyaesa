"""Static deterministic ASR branch execution."""

from pathlib import Path
from typing import Any

import pandas as pd

from pyaesa.acc.deterministic.runtime.paths import (
    acc_output_relative_dir,
    build_acc_path_context,
    get_acc_output_dir,
)
from pyaesa.acc.shared.runtime.paths import public_result_root_name_for_fu_code
from pyaesa.shared.acc_asr_common.deterministic.status_labels import cc_branch_status_label
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.tabular.table_io import read_table

from .compute import deterministic_asr_for_acc_file
from .tables import requested_year_columns, write_asr_output
from ...shared.runtime.paths import (
    build_asr_path_context,
    get_asr_results_dir,
)
from .common import (
    ASRProcessResult,
    build_external_transition,
    require_single_method_identity,
)


def process_static_asr(
    *,
    proj_base: Path,
    fu_code: str,
    cc_source: str,
    source_label: str,
    base_allocate_args: dict[str, Any],
    years: list[int],
    fmt: str,
    lca_type: str,
    lca_version_name: str | None,
    static_cc_bounds: list[str],
    acc_output_files: list[Path],
    allowed_l1_l2_methods: set[str],
    lca_rows: pd.DataFrame,
    status: StatusSink,
    return_lca_rows: bool = False,
) -> ASRProcessResult:
    """Build ASR outputs for static aCC branches."""
    n_matched = 0
    n_written = 0
    impacts: set[str] = set()
    output_dirs: list[Path] = []
    output_files: list[Path] = []
    external_transition = build_external_transition(lca_rows, lca_type=lca_type)
    acc_path_context = build_acc_path_context(
        proj_base=proj_base,
        source_label=source_label,
        agg_version=base_allocate_args["agg_version"],
        cc_source=cc_source,
        cc_type="static",
    )
    acc_dir = get_acc_output_dir(
        context=acc_path_context,
        public_result_root_name=public_result_root_name_for_fu_code(fu_code=fu_code),
    )
    acc_files = acc_output_files
    results_dir = get_asr_results_dir(
        context=build_asr_path_context(
            proj_base=proj_base,
            source_label=source_label,
            agg_version=base_allocate_args["agg_version"],
            fu_code=fu_code,
            lca_type=lca_type,
            cc_source=cc_source,
            cc_type="static",
            lca_version_name=lca_version_name,
        ),
    )
    output_dirs.append(results_dir)
    status_label = cc_branch_status_label(cc_source=cc_source, cc_type="static")
    for acc_path in acc_files:
        status.show(f"[deterministic_asr] {status_label}: {acc_path.name}")
        acc_df = read_table(path=acc_path)
        l1_l2_method = require_single_method_identity(acc_df, path=acc_path)
        if l1_l2_method not in allowed_l1_l2_methods:
            continue
        requested_bounds = sorted({str(value) for value in static_cc_bounds})
        acc_df = acc_df.loc[acc_df["cc_bound"].astype(str).isin(requested_bounds)].copy()
        year_cols = requested_year_columns(acc_df, requested_years=years)
        if not year_cols:
            continue
        asr_frames = []
        for impact_code, impact_acc_df in acc_df.groupby("impact", dropna=False, sort=False):
            impact_text = str(impact_code).strip()
            asr_frames.append(
                deterministic_asr_for_acc_file(
                    acc_df=impact_acc_df.reset_index(drop=True),
                    year_cols=year_cols,
                    impact_code=impact_text,
                    lca_rows=lca_rows,
                    lca_type=lca_type,
                )
            )
            impacts.add(impact_text)
        relative_output_dir = acc_output_relative_dir(
            upstream_relative_dir=acc_path.relative_to(acc_dir).parent
        )
        asr_path = results_dir / relative_output_dir / f"{acc_path.stem}.{fmt}"
        write_asr_output(pd.concat(asr_frames, ignore_index=True), asr_path, fmt)
        output_files.append(asr_path)
        n_written += 1
        n_matched += 1
    return ASRProcessResult(
        n_matched=n_matched,
        n_written=n_written,
        impacts=sorted(impacts),
        output_dirs=output_dirs,
        output_files=output_files,
        external_lca_transition=external_transition,
        lca_rows=lca_rows if return_lca_rows else None,
        dynamic_component_frame=None,
    )
