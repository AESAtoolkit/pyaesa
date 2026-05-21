"""Shared deterministic ASR branch runtime."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from pyaesa.shared.acc_asr_common.deterministic.downstream.scenarios import (
    share_transition_payload_for_output_stem,
)
from pyaesa.shared.tabular.wide_tables import (
    first_non_null_scenario_year,
    resolve_single_allocation_method_identity,
)
from pyaesa.shared.runtime.scenario.columns import EXT_LCA_SSP_SCENARIO_COLUMN


@dataclass(frozen=True)
class ASRProcessResult:
    """Internal deterministic ASR branch process result."""

    n_matched: int
    n_written: int
    impacts: list[str]
    output_dirs: list[Path]
    output_files: list[Path]
    external_lca_transition: dict[str, object] | None
    lca_rows: pd.DataFrame | None
    dynamic_component_frame: pd.DataFrame | None


def require_single_method_identity(frame: pd.DataFrame, *, path: Path) -> str:
    """Return one canonical allocation method identity from a persisted aCC file."""
    return resolve_single_allocation_method_identity(
        frame,
        where=f"Deterministic aCC input '{path.name}'",
    )


def build_external_transition(lca_rows: pd.DataFrame, *, lca_type: str) -> dict[str, object] | None:
    """Return the figure marker payload for external LCA transitions when relevant."""
    if lca_type != "external":
        return None
    return {
        "switch_year": first_non_null_scenario_year(
            lca_rows,
            scenario_column=EXT_LCA_SSP_SCENARIO_COLUMN,
        ),
        "marker_label": "external LCA SSP-dependent switch",
        "marker_color": "#375a7f",
    }


def asocc_ssp_transition_start_year(
    *,
    output_stem: str,
    share_transition_meta: dict[str, dict[str, object]],
) -> int | None:
    """Return the denominator SSP start year for one deterministic downstream file."""
    payload = share_transition_payload_for_output_stem(
        output_stem=output_stem,
        share_transition_meta=share_transition_meta,
    )
    start_year = payload.get("ssp_start_year")
    if start_year is None:
        return None
    missing = pd.isna(start_year)
    if isinstance(missing, bool) and missing:
        return None
    if isinstance(start_year, str):
        return int(start_year.strip())
    return int(cast(int | float, start_year))


def lca_transition_start_year(*, lca_rows: pd.DataFrame, lca_type: str) -> int | None:
    """Return the numerator SSP start year for one deterministic ASR branch."""
    if lca_type != "external":
        return None
    return first_non_null_scenario_year(
        lca_rows,
        scenario_column=EXT_LCA_SSP_SCENARIO_COLUMN,
        year_column="year",
    )
