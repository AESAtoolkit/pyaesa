"""ASR uncertainty run planning."""

import pandas as pd

from pyaesa.acc.uncertainty.evaluation.summary import (
    acc_dynamic_category_uncertainty_active_from_manifest,
    acc_summary_excluded_columns,
)
from pyaesa.acc.uncertainty.io.artifacts import (
    acc_run_layout_from_manifest,
    acc_run_paths_from_manifest,
)
from pyaesa.asr.uncertainty.evaluation.alignment import build_asr_alignment
from pyaesa.asr.uncertainty.evaluation.cumulative import cumulative_period_identity_groups
from pyaesa.asr.uncertainty.evaluation.summary import (
    asr_summary_identity_groups,
    cumulative_summary_identity_with_metrics,
    summary_identity_with_metrics,
)
from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyPlan, LCAUncertaintyInput
from pyaesa.asr.uncertainty.sources.source_keys import acc_source_name
from pyaesa.shared.figures.dynamic_ar6 import DYNAMIC_AR6_CC_TYPE
from pyaesa.shared.runtime.scenario.columns import ASOCC_TIME_ROUTE_PUBLIC_COLUMN
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def build_asr_uncertainty_plan(
    *,
    acc_manifest: UncertaintyManifest,
    lca_input: LCAUncertaintyInput,
    output_format: str,
) -> ASRUncertaintyPlan:
    """Build ASR public identity and vectorized component position maps."""
    acc_paths = acc_run_paths_from_manifest(manifest=acc_manifest)
    acc_identity = read_uncertainty_table(
        path=acc_paths.public_row_identity,
        output_format=output_format,
    )
    alignment = build_asr_alignment(
        acc_identity=acc_identity,
        lca_identity=lca_input.identity,
        lca_type=lca_input.lca_type,
    )
    active_sources = tuple(acc_source_name(name) for name in acc_manifest.active_sources) + tuple(
        lca_input.active_sources
    )
    dynamic_category_uncertainty_active = acc_dynamic_category_uncertainty_active_from_manifest(
        manifest=acc_manifest
    )
    base_summary_identity, summary_groups = asr_summary_identity_groups(
        identity=alignment.identity,
        active_sources=acc_manifest.active_sources,
        dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
    )
    if _identity_is_dynamic_ar6(alignment.identity):
        cumulative_excluded_columns = acc_summary_excluded_columns(
            active_sources=acc_manifest.active_sources,
            dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
        )
        cumulative_excluded_columns.add(ASOCC_TIME_ROUTE_PUBLIC_COLUMN)
        cumulative_excluded_columns.difference_update({"l1_l2_method", "l1_method", "l2_method"})
        cumulative_identity, cumulative_groups = cumulative_period_identity_groups(
            identity=alignment.identity,
            excluded_columns=cumulative_excluded_columns,
        )
        cumulative_summary_identity, cumulative_summary_groups = asr_summary_identity_groups(
            identity=cumulative_identity,
            active_sources=acc_manifest.active_sources,
            dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
        )
        cumulative_summary_identity = cumulative_summary_identity_with_metrics(
            cumulative_summary_identity
        )
    else:
        cumulative_identity = alignment.identity.iloc[0:0].copy()
        cumulative_summary_identity = base_summary_identity.iloc[0:0].copy()
        cumulative_groups = ()
        cumulative_summary_groups = ()
    return ASRUncertaintyPlan(
        identity=alignment.identity,
        summary_identity=summary_identity_with_metrics(base_summary_identity),
        summary_public_row_groups=summary_groups,
        cumulative_identity=cumulative_identity,
        cumulative_summary_identity=cumulative_summary_identity,
        cumulative_summary_public_row_groups=cumulative_summary_groups,
        cumulative_public_row_groups=cumulative_groups,
        acc_positions=alignment.acc_positions,
        lca_positions=alignment.lca_positions,
        lca_unit_factors=alignment.lca_unit_factors,
        acc_manifest=acc_manifest,
        lca_input=lca_input,
        asr_run_layout=acc_run_layout_from_manifest(manifest=acc_manifest),
        source_method_rows=pd.DataFrame(),
        active_sources=active_sources,
    )


def _identity_is_dynamic_ar6(identity: pd.DataFrame) -> bool:
    """Return whether ASR identity rows are backed by dynamic AR6 CC."""
    values = {
        str(value).strip()
        for value in identity["cc_type"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    return values == {DYNAMIC_AR6_CC_TYPE}
