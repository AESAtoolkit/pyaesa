"""Dynamic deterministic ASR branch execution."""

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
from pyaesa.shared.tabular.deterministic_companion_stems import (
    parse_deterministic_companion_stem,
)
from pyaesa.shared.tabular.table_io import read_table

from .compute import build_deterministic_asr_component_frame
from .dynamic_cumulative import (
    PendingDynamicAsrOutput,
    dynamic_group_parent,
    write_dynamic_asr_outputs,
)
from .tables import requested_year_columns
from ...shared.runtime.paths import (
    build_asr_path_context,
    get_asr_results_dir,
)
from .common import (
    ASRProcessResult,
    asocc_ssp_transition_start_year,
    build_external_transition,
    lca_transition_start_year,
    require_single_method_identity,
)


def process_dynamic_asr(
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
    acc_output_files: list[Path],
    allowed_l1_l2_methods: set[str],
    share_transition_meta: dict[str, dict[str, object]],
    lca_rows: pd.DataFrame,
    status: StatusSink,
    return_lca_rows: bool = False,
) -> ASRProcessResult:
    """Build ASR outputs for dynamic aCC branches."""
    acc_path_context = build_acc_path_context(
        proj_base=proj_base,
        source_label=source_label,
        group_version=base_allocate_args["group_version"],
        cc_source=cc_source,
        cc_type="dynamic_ar6",
    )
    acc_dir = get_acc_output_dir(
        context=acc_path_context,
        public_result_root_name=public_result_root_name_for_fu_code(fu_code=fu_code),
    )
    acc_files = acc_output_files
    out_dir = get_asr_results_dir(
        context=build_asr_path_context(
            proj_base=proj_base,
            source_label=source_label,
            group_version=base_allocate_args["group_version"],
            fu_code=fu_code,
            lca_type=lca_type,
            cc_source=cc_source,
            cc_type="dynamic_ar6",
            lca_version_name=lca_version_name,
        ),
    )
    n_matched = 0
    n_written = 0
    impacts: set[str] = set()
    output_files: list[Path] = []
    pending_outputs: list[PendingDynamicAsrOutput] = []
    external_transition = build_external_transition(lca_rows, lca_type=lca_type)
    resolved_lca_ssp_start_year = lca_transition_start_year(
        lca_rows=lca_rows,
        lca_type=lca_type,
    )
    status_label = cc_branch_status_label(cc_source=cc_source, cc_type="dynamic_ar6")
    for acc_path in acc_files:
        status.show(f"[deterministic_asr] {status_label}: {acc_path.name}")
        acc_df = read_table(path=acc_path)
        l1_l2_method = require_single_method_identity(acc_df, path=acc_path)
        if l1_l2_method not in allowed_l1_l2_methods:
            continue
        year_cols = requested_year_columns(acc_df, requested_years=years)
        if not year_cols:
            continue
        impact_code = str(acc_df["impact"].iloc[0]).strip()
        resolved_asocc_ssp_start_year = asocc_ssp_transition_start_year(
            output_stem=acc_path.stem,
            share_transition_meta=share_transition_meta,
        )
        asr_df = build_deterministic_asr_component_frame(
            acc_df=acc_df,
            year_cols=year_cols,
            impact_code=impact_code,
            lca_rows=lca_rows,
            lca_type=lca_type,
        )
        if resolved_asocc_ssp_start_year is not None:
            asr_df["asocc_ssp_start_year"] = int(resolved_asocc_ssp_start_year)
        if resolved_lca_ssp_start_year is not None:
            asr_df["lca_ssp_start_year"] = int(resolved_lca_ssp_start_year)
        relative_output_dir = acc_output_relative_dir(
            upstream_relative_dir=acc_path.relative_to(acc_dir).parent
        )
        asr_result_stem = acc_path.stem
        asr_path = out_dir / relative_output_dir / f"{asr_result_stem}.{fmt}"
        pending_outputs.append(
            PendingDynamicAsrOutput(
                path=asr_path,
                relative_parent=dynamic_group_parent(relative_output_dir),
                base_stem=parse_deterministic_companion_stem(asr_result_stem).base_stem,
                frame=asr_df,
                year_cols=year_cols,
            )
        )
        output_files.append(asr_path)
        impacts.add(impact_code)
        n_matched += 1
        n_written += 1
    dynamic_component_frame = write_dynamic_asr_outputs(
        outputs=pending_outputs,
        fmt=fmt,
    )
    return ASRProcessResult(
        n_matched=n_matched,
        n_written=n_written,
        impacts=sorted(impacts),
        output_dirs=[out_dir],
        output_files=output_files,
        external_lca_transition=external_transition,
        lca_rows=lca_rows if return_lca_rows else None,
        dynamic_component_frame=dynamic_component_frame,
    )
