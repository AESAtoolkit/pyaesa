"""Dynamic AR6 CC inputs for aCC uncertainty."""

from typing import Any, cast

import pandas as pd

from pyaesa.acc.shared.runtime.dynamic_units import dynamic_acc_unit_factors
from pyaesa.acc.uncertainty.runtime.models import ACCDynamicCCInput
from pyaesa.ar6_cc.uncertainty.runtime.prerequisites import (
    load_deterministic_ar6_cc_rows,
    prepare_ar6_cc_deterministic_prerequisite,
)
from pyaesa.ar6_cc.uncertainty.request.normalization import (
    AR6_DYNAMIC_CC_SOURCE,
    normalize_ar6_cc_uncertainty_request,
)
from pyaesa.ar6_cc.uncertainty.evaluation.sampling import deterministic_ar6_cc_identity_and_values
from pyaesa.ar6_cc.uncertainty.runner import run_uncertainty_ar6_cc_component
from pyaesa.shared.lcia.contracts import dynamic_cc_match
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import RunProgressPrinter
from pyaesa.shared.runtime.reporting.status import StatusSink


def dynamic_cc_base_args(
    *,
    branch: dict[str, Any],
    years: int | list[int] | range,
) -> dict[str, Any]:
    """Return AR6 CC public selector arguments from one dynamic aCC branch."""
    return {
        "years": years,
        "harmonization": bool(branch["harmonization"]),
        "harmonization_method": str(branch["harmonization_method"]),
        "category": branch["category"],
        "ssp_scenario": branch["ssp_scenario"],
        "emission_type": str(branch["emission_type"]),
        "include_afolu": bool(branch["include_afolu"]),
        "emissions_mode": str(branch["emissions_mode"]),
        "subset_version": branch["subset_version"],
    }


def dynamic_ar6_cc_uncertainty_input(
    *,
    branch: dict[str, Any],
    years: int | list[int] | range,
    source_parameters: dict[str, Any],
    mc_parameters: Any,
    output_format: str,
    figures: bool,
    figure_format: dict[str, Any] | None,
    component_inventory: dict[str, Any] | None = None,
    run_id: str | None = None,
    show_progress: bool = True,
    phase: PhasePrinter | NullPhasePrinter | None = None,
    refresh: bool = False,
    progress: RunProgressPrinter | None = None,
    component_session: Any | None = None,
    finalize_component_inventory: bool = False,
) -> tuple[ACCDynamicCCInput, Any | None]:
    """Run or reuse AR6 CC uncertainty and expose it as an aCC component."""
    run = run_uncertainty_ar6_cc_component(
        base_ar6_cc_args=dynamic_cc_base_args(branch=branch, years=years),
        uncertainty_config={
            "mc_parameters": mc_parameters,
            AR6_DYNAMIC_CC_SOURCE: source_parameters,
        },
        output_format=output_format,
        figures=figures,
        figure_options=None,
        figure_format=figure_format,
        refresh=refresh,
        component_inventory=component_inventory,
        run_id=run_id,
        show_progress=show_progress,
        phase=phase,
        progress=progress,
        component_session=component_session,
        finalize_component_inventory=finalize_component_inventory,
    )
    return (
        ACCDynamicCCInput(
            identity=None,
            deterministic_values=None,
            manifest=run.report.manifest,
            deterministic_manifest_path=None,
            reuse_status=run.report.reuse_status,
        ),
        run.session,
    )


def deterministic_dynamic_cc_input(
    *,
    branch: dict[str, Any],
    years: int | list[int] | range,
    figures: bool = False,
    figure_format: dict[str, Any] | None = None,
    status: StatusSink | None = None,
    refresh: bool = False,
) -> ACCDynamicCCInput:
    """Load deterministic AR6 CC rows as fixed carrying capacity values."""
    request = normalize_ar6_cc_uncertainty_request(
        base_ar6_cc_args=dynamic_cc_base_args(branch=branch, years=years),
        source_parameters={},
    )
    prerequisite = prepare_ar6_cc_deterministic_prerequisite(
        request=request,
        refresh=refresh,
        figures=figures,
        figure_format=figure_format,
        status=status,
    )
    deterministic_rows = load_deterministic_ar6_cc_rows(request=request, scope=prerequisite)
    identity, values = deterministic_ar6_cc_identity_and_values(
        request=request,
        deterministic_rows=deterministic_rows,
    )
    match = cast(dict[str, str], dynamic_cc_match(lcia_method=str(branch["cc_source"])))
    impact = str(match["impact"]).strip()
    target_unit, factors = dynamic_acc_unit_factors(
        source_units=pd.Series(identity["impact_unit"], copy=False),
        cc_source=str(branch["cc_source"]),
        impact=impact,
        source_path=prerequisite.metadata_path,
    )
    identity = identity.copy()
    identity["impact_unit"] = target_unit
    return ACCDynamicCCInput(
        identity=identity,
        deterministic_values=values * factors,
        manifest=None,
        deterministic_manifest_path=prerequisite.metadata_path,
        reuse_status=prerequisite.reuse_status,
        process_ar6=prerequisite.process_ar6,
    )
