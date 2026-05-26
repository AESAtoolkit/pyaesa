"""Writer for UT(GVAa) identity closure audit rows."""

from pathlib import Path

import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent

from ....runtime.paths.deterministic import _get_allocate_ut_gvaa_identity_closure_path
from pyaesa.asocc.orchestration.write.writers.progress import tick_write_progress

_KEY_COLUMNS = [
    "source",
    "fu_code",
    "year",
    "ssp_scenario",
    "l2_method",
    "comparator_method",
    "l1_method",
    "impact",
    "lcia_key",
    "reference_year",
    "l2_reuse_year",
    "r_p",
    "s_p",
]

_OUTPUT_COLUMNS = [
    "projection_branch",
    "source",
    "fu_code",
    "year",
    "ssp_scenario",
    "l2_method",
    "comparator_method",
    "l1_method",
    "impact",
    "lcia_key",
    "reference_year",
    "l2_reuse_year",
    "r_p",
    "s_p",
    "ut_gvaa_raw",
    "ut_gva_floor",
    "ut_gvaa_final",
    "delta_added",
    "adjustment_note",
]


def _reorder_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic public column order for closure audit rows."""
    out = frame.copy()
    for column in _OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    return out.loc[:, _OUTPUT_COLUMNS]


def write_ut_gvaa_identity_closure_audit(
    *,
    context,
    state,
    refresh_effective: bool,
) -> Path | None:
    """Write UT(GVAa) identity closure audit rows with deterministic upsert semantics."""
    output_source = context.output_source
    path = _get_allocate_ut_gvaa_identity_closure_path(
        proj_base=context.proj_base,
        source=output_source,
        agg_version=context.agg_version,
    )
    if not state.ut_gvaa_identity_closure_rows:
        return path if path.exists() else None

    path = ensure_file_parent(path)
    out = pd.DataFrame(state.ut_gvaa_identity_closure_rows)
    if out.empty:
        return path if path.exists() else None
    if path.exists() and not refresh_effective:
        prior = pd.read_csv(path)
        out = pd.concat([prior, out], ignore_index=True)
    out = out.drop_duplicates(subset=_KEY_COLUMNS, keep="last")
    out = out.sort_values(_KEY_COLUMNS).reset_index(drop=True)
    out = _reorder_columns(out)
    out.to_csv(path, index=False)
    tick_write_progress(context=context, state=state)
    return path
