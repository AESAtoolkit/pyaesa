"""aCC uncertainty run planning."""

from typing import Any, cast

import pandas as pd

from pyaesa.acc.uncertainty.evaluation.branches import (
    build_acc_branch_plans,
    combined_acc_identity,
)
from pyaesa.acc.uncertainty.evaluation.summary import acc_summary_identity_groups
from pyaesa.acc.uncertainty.io.source_methods import build_acc_source_methods
from pyaesa.acc.uncertainty.runtime.models import (
    ACCAsoccInput,
    ACCDynamicCCInput,
    ACCUncertaintyPlan,
)
from pyaesa.acc.uncertainty.sources.source_keys import ar6_cc_source_name, asocc_source_name
from pyaesa.ar6_cc.deterministic.request.contracts import CC_FLOW_NEGATIVE
from pyaesa.ar6_cc.uncertainty.io.artifacts import (
    ar6_cc_run_layout_from_manifest,
    ar6_cc_run_paths_from_manifest,
)
from pyaesa.asocc.uncertainty.io.artifacts import (
    asocc_run_layout_from_manifest,
    asocc_run_paths_from_manifest,
)
from pyaesa.shared.uncertainty_assessment.io.tables import read_uncertainty_table
from pyaesa.shared.uncertainty_assessment.run_state.manifest import UncertaintyManifest


def build_acc_uncertainty_plan(
    *,
    asocc_input: ACCAsoccInput,
    dynamic_cc_input: ACCDynamicCCInput | None,
    branches: list[dict[str, Any]],
    output_format: str,
) -> ACCUncertaintyPlan:
    """Build compact row alignment maps for one aCC uncertainty request."""
    asocc_identity = _asocc_identity(asocc_input=asocc_input, output_format=output_format)
    asocc_layout = _asocc_layout(asocc_input=asocc_input)
    cc_identity = _dynamic_cc_identity(
        dynamic_cc_input=dynamic_cc_input,
        output_format=output_format,
    )
    cc_layout = _dynamic_cc_layout(dynamic_cc_input=dynamic_cc_input)
    branch_plans = build_acc_branch_plans(
        asocc_identity=asocc_identity,
        cc_identity=cc_identity,
        branches=branches,
    )
    identity = combined_acc_identity(branch_plans=branch_plans)
    active_sources = _active_sources(
        asocc_input=asocc_input,
        dynamic_cc_input=dynamic_cc_input,
    )
    dynamic_category_uncertainty_active = _dynamic_category_uncertainty_active(
        dynamic_cc_input=dynamic_cc_input
    )
    summary_identity, summary_public_row_groups = acc_summary_identity_groups(
        identity=identity,
        active_sources=active_sources,
        dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
    )
    source_methods = build_acc_source_methods(
        asocc_input=asocc_input,
        dynamic_cc_input=dynamic_cc_input,
    )
    return ACCUncertaintyPlan(
        identity=identity,
        summary_identity=summary_identity,
        summary_public_row_groups=summary_public_row_groups,
        branch_plans=branch_plans,
        asocc_input=asocc_input,
        dynamic_cc_input=dynamic_cc_input,
        acc_run_layout=(
            "sparse_selected_rows"
            if "sparse_selected_rows" in {asocc_layout, cc_layout}
            else "compact_run_matrix"
        ),
        deterministic_cc_values=(
            None if dynamic_cc_input is None else dynamic_cc_input.deterministic_values
        ),
        source_method_rows=source_methods,
        active_sources=active_sources,
        dynamic_category_uncertainty_active=dynamic_category_uncertainty_active,
    )


def _asocc_identity(*, asocc_input: ACCAsoccInput, output_format: str) -> pd.DataFrame:
    if asocc_input.manifest is None:
        return cast(pd.DataFrame, asocc_input.identity)
    paths = asocc_run_paths_from_manifest(manifest=cast(UncertaintyManifest, asocc_input.manifest))
    return read_uncertainty_table(path=paths.public_row_identity, output_format=output_format)


def _asocc_layout(*, asocc_input: ACCAsoccInput) -> str:
    if asocc_input.manifest is None:
        return "fixed_values"
    return asocc_run_layout_from_manifest(manifest=cast(UncertaintyManifest, asocc_input.manifest))


def _dynamic_cc_identity(
    *,
    dynamic_cc_input: ACCDynamicCCInput | None,
    output_format: str,
) -> pd.DataFrame | None:
    if dynamic_cc_input is None:
        return None
    if dynamic_cc_input.manifest is None:
        return cast(pd.DataFrame, dynamic_cc_input.identity)
    paths = ar6_cc_run_paths_from_manifest(
        manifest=cast(UncertaintyManifest, dynamic_cc_input.manifest)
    )
    identity = read_uncertainty_table(path=paths.public_row_identity, output_format=output_format)
    return identity.loc[identity["cc_flow"].astype(str) != CC_FLOW_NEGATIVE].reset_index(drop=True)


def _dynamic_cc_layout(*, dynamic_cc_input: ACCDynamicCCInput | None) -> str:
    if dynamic_cc_input is None or dynamic_cc_input.manifest is None:
        return "fixed_values"
    return ar6_cc_run_layout_from_manifest(
        manifest=cast(UncertaintyManifest, dynamic_cc_input.manifest)
    )


def _active_sources(
    *,
    asocc_input: ACCAsoccInput,
    dynamic_cc_input: ACCDynamicCCInput | None,
) -> tuple[str, ...]:
    names: list[str] = []
    if asocc_input.manifest is not None:
        names.extend(
            asocc_source_name(name)
            for name in cast(UncertaintyManifest, asocc_input.manifest).active_sources
        )
    cc_manifest = None if dynamic_cc_input is None else dynamic_cc_input.manifest
    if cc_manifest is not None:
        names.extend(ar6_cc_source_name(name) for name in cc_manifest.active_sources)
    return tuple(names)


def _dynamic_category_uncertainty_active(
    *,
    dynamic_cc_input: ACCDynamicCCInput | None,
) -> bool:
    if dynamic_cc_input is None or dynamic_cc_input.manifest is None:
        return False
    manifest = cast(UncertaintyManifest, dynamic_cc_input.manifest)
    source_parameters = manifest.source_parameters or {}
    dynamic_source = source_parameters.get("dynamic_ar6_cc_uncertainty", {})
    return bool(cast(dict[str, Any], dynamic_source).get("category_uncertainty", False))
