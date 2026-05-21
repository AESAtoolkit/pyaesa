"""aSoCC Sobol README and manifest method metadata."""

from pathlib import Path

from pyaesa.asocc.uncertainty.inputs.external_rows import EXTERNAL_ASOCC_RUN_SOURCE
from pyaesa.shared.uncertainty_assessment.sobol.plan import SobolPlan
from pyaesa.shared.uncertainty_assessment.sobol.reporting import (
    sobol_method_payload,
    write_sobol_readme,
)


def asocc_sobol_method_payload(
    *,
    source_names: tuple[str, ...],
    plan: SobolPlan,
    selected_years: tuple[int, ...],
) -> dict[str, object]:
    """Return aSoCC Sobol method metadata for the run manifest."""
    return sobol_method_payload(
        source_names=source_names,
        plan=plan,
        selected_scope={"selected_output_years": list(selected_years)},
    )


def write_asocc_sobol_readme(
    *,
    path: Path,
    output_format: str,
    source_names: tuple[str, ...],
    selected_years: tuple[int, ...],
    plan: SobolPlan,
) -> None:
    """Write the aSoCC Sobol README."""
    method_notes = (
        (
            (
                f"- {EXTERNAL_ASOCC_RUN_SOURCE} appears when selected external aSoCC "
                "methods are provided as Monte Carlo run files. It is the empirical "
                "source dimension for the user supplied external run_index values. LCIA, "
                "projection, and inter-MRIO uncertainty still skip external method rows."
            ),
        )
        if EXTERNAL_ASOCC_RUN_SOURCE in source_names
        else ()
    )
    write_sobol_readme(
        path=path,
        output_format=output_format,
        family_label="aSoCC",
        source_names=source_names,
        selected_scope_line=(
            "The selected Sobol output years are "
            f"{';'.join(str(year) for year in selected_years)}. Sobol indices explain "
            "variance only for those output years."
        ),
        plan=plan,
        source_summary_notes=(
            (
                "- LCIA method, impact category, region selector, sector selector, "
                "studied output year, and SSP scenario are retained when they are present "
                "in the evaluated aSoCC output identity."
            ),
            (
                "- When method choice is an active uncertainty source, method columns are "
                "removed before Sobol accumulation. The reported value is the value after "
                "method choice on the remaining public output identity."
            ),
            (
                "- contains_ssp_invariant_outputs=True means this SSP row includes "
                "evaluated aSoCC outputs whose value is invariant across SSP scenarios. "
                "Those outputs are included in SSP rows only when the same selected year "
                "and remaining public selector identity also has actual SSP specific "
                "outputs. Non SSP years remain blank on the SSP axis."
            ),
            (
                "- summary_level=selector keeps the selected output year, LCIA method, "
                "impact category, functional unit selectors, and SSP scenario where those "
                "axes exist."
            ),
            (
                "- summary_level=lcia_method keeps the same selectors but aggregates over "
                "impact categories within each LCIA method. The impact column is empty on "
                "these rows by construction."
            ),
        ),
        indices_notes=(),
        method_notes=method_notes,
    )
