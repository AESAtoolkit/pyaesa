"""User facing reports for completed uncertainty runs."""

from dataclasses import dataclass
from pathlib import Path

from pyaesa.shared.runtime.reporting.labels import (
    labelled_values_line,
)
from pyaesa.shared.runtime.reporting.reuse_status import public_reuse_status
from pyaesa.shared.runtime.reporting.summary import (
    SummaryDocument,
    SummaryWarning,
    document,
    render_summary,
    warning,
)
from pyaesa.shared.runtime.reporting.summary_log import summary_log_path, write_summary_log
from pyaesa.shared.runtime.reporting.values import (
    as_sequence,
    format_report_value,
    format_summary_value,
    format_values,
)
from pyaesa.shared.uncertainty_assessment.run_state.report_arguments import scope_arguments
from pyaesa.shared.uncertainty_assessment.run_state.report_dependencies import payload_source
from pyaesa.shared.uncertainty_assessment.run_state.report_sections import phase_sections
from pyaesa.shared.uncertainty_assessment.run_state.manifest import (
    UncertaintyManifest,
)


@dataclass(frozen=True)
class UncertaintyRunReport:
    """Compact printable report returned by public uncertainty functions."""

    manifest: UncertaintyManifest
    reuse_status: str

    def __str__(self) -> str:
        """Return one deterministic summary block for the completed run."""
        return render_summary(
            _summary_document(manifest=self.manifest, reuse_status=self.reuse_status)
        )

    __repr__ = __str__


def uncertainty_report(
    *,
    manifest: UncertaintyManifest,
    reuse_status: str,
) -> UncertaintyRunReport:
    """Return the public report wrapper for one uncertainty manifest."""
    report = UncertaintyRunReport(manifest=manifest, reuse_status=str(reuse_status))
    scope_manifest = manifest.artifacts["scope_manifest"]
    write_summary_log(
        path=summary_log_path(logs_dir=Path(str(scope_manifest)).parent),
        summary=str(report),
    )
    return report


def _summary_document(
    *,
    manifest: UncertaintyManifest,
    reuse_status: str,
) -> SummaryDocument:
    return document(
        f"uncertainty_{manifest.family}",
        lines=_run_lines(manifest=manifest, reuse_status=reuse_status),
        sections=phase_sections(
            manifest=manifest,
            run_warnings=_run_warnings(manifest=manifest),
            warning_builder=lambda nested_manifest: _run_warnings(manifest=nested_manifest),
        ),
    )


def _run_lines(
    *,
    manifest: UncertaintyManifest,
    reuse_status: str,
) -> tuple[str, ...]:
    lines = [f"Run status: {public_reuse_status(reuse_status)}"]
    lines.extend(_common_scope_lines(manifest=manifest))
    lines.extend(_active_source_lines(manifest=manifest))
    lines.extend(_monte_carlo_lines(manifest=manifest))
    lines.extend(_sobol_lines(manifest=manifest))
    return tuple(lines)


def _common_scope_lines(*, manifest: UncertaintyManifest) -> list[str]:
    arguments = scope_arguments(manifest=manifest)
    lines: list[str] = []
    project_name = arguments.get("project_name")
    if project_name is not None:
        lines.append(f"Project: {format_report_value(project_name)}")
    years = arguments.get("years")
    if years is not None and manifest.family != "ar6_cc":
        year_values = as_sequence(years)
        int_years = [int(str(year)) for year in year_values]
        lines.append(
            labelled_values_line(
                "Studied year",
                "Studied years",
                tuple(int_years),
                format_summary_value(key="years", value=int_years),
            )
        )
    lcia_method = arguments.get("lcia_method")
    if lcia_method is not None:
        methods = as_sequence(lcia_method)
        lines.append(
            labelled_values_line(
                "LCIA method",
                "LCIA methods",
                methods,
                format_values(methods),
            )
        )
    for label, key in (
        ("Functional unit", "fu_code"),
        ("Carrying capacity type", "cc_type"),
    ):
        value = arguments.get(key)
        if value is not None:
            lines.append(f"{label}: {format_report_value(value)}")
    lca_route = arguments.get("lca_route")
    if lca_route is not None:
        lines.append(f"LCA route: {format_report_value(lca_route)}")
    version_name = arguments.get("version_name")
    if version_name is not None:
        lines.append(f"Version: {format_report_value(version_name)}")
    ssp_scenarios = arguments.get("ssp_scenarios") or arguments.get("ssp_scenario")
    if ssp_scenarios is not None:
        scenarios = as_sequence(ssp_scenarios)
        lines.append(
            labelled_values_line(
                "SSP scenario",
                "SSP scenarios",
                scenarios,
                format_values(scenarios),
            )
        )
    return lines


def _active_source_lines(*, manifest: UncertaintyManifest) -> list[str]:
    if not manifest.active_sources:
        return ["Active uncertainty sources: none"]
    grouped: dict[str, list[str]] = {}
    for source in manifest.active_sources:
        owner, source_name = _source_owner_and_name(source=str(source), family=manifest.family)
        grouped.setdefault(owner, []).append(source_name)
    lines = ["Active uncertainty sources:"]
    lines.extend(
        f"  {owner}: {format_values(tuple(source_names))}"
        for owner, source_names in grouped.items()
    )
    return lines


def _monte_carlo_lines(*, manifest: UncertaintyManifest) -> list[str]:
    mc_parameters = manifest.mc_parameters or {}
    requested = int(mc_parameters.get("requested_runs", manifest.requested_runs) or 0)
    if manifest.mode == "fixed":
        return [f"Monte Carlo: fixed; completed fixed runs {manifest.completed_runs}"]
    maximum = int(mc_parameters.get("max_runs", requested or manifest.requested_runs) or 0)
    convergence = manifest.convergence or {}
    reached = "not reached"
    if convergence:
        reached = "reached" if bool(convergence.get("reached")) else "not reached"
        maximum = int(convergence.get("max_runs", maximum) or maximum)
    line = (
        f"Monte Carlo: convergence; completed runs {manifest.completed_runs}; "
        f"maximum allowed runs {maximum}; convergence {reached}"
    )
    lines = [line]
    context = manifest.compatibility_context or {}
    if str(context.get("artifact_contract", "")).endswith("_branch_set"):
        lines.append(
            "Convergence scope: independent per branch; each branch scope manifest "
            "records its own convergence status."
        )
    reason = convergence.get("reason") if convergence else None
    if reason is not None:
        lines.append(f"Convergence reason: {reason}")
    return lines


def _sobol_lines(*, manifest: UncertaintyManifest) -> list[str]:
    sobol = manifest.sobol or {}
    if not sobol:
        return ["Sobol: not requested"]
    if not bool(sobol.get("ran")):
        reason = sobol.get("reason")
        if reason is None:
            return ["Sobol: not requested"]
        return [f"Sobol: not run ({reason})"]
    mode = str(sobol.get("mode", "fixed"))
    samples = sobol.get("n_base_samples")
    dimension_count = sobol.get("active_source_count")
    evaluations = None
    if samples is not None and dimension_count is not None:
        evaluations = int(samples) * (int(dimension_count) + 2)
    if mode == "fixed":
        line = f"Sobol: fixed; base samples {samples}"
    else:
        reached = "reached" if bool(sobol.get("reached")) else "not reached"
        max_samples = sobol.get("max_base_samples", samples)
        line = (
            f"Sobol: convergence; base samples {samples}; "
            f"maximum base samples {max_samples}; convergence {reached}"
        )
    if evaluations is not None:
        line = f"{line}; design evaluations {evaluations}"
    return [line]


def _run_warnings(*, manifest: UncertaintyManifest) -> tuple[SummaryWarning, ...]:
    warnings: list[SummaryWarning] = []
    convergence = manifest.convergence or {}
    if manifest.mode == "convergence" and convergence and not bool(convergence.get("reached")):
        completed = convergence.get("completed_runs", manifest.completed_runs)
        maximum = convergence.get("max_runs", manifest.requested_runs)
        reason = convergence.get("reason")
        message = f"Monte Carlo convergence was not reached after {completed} of {maximum} runs."
        if reason is not None:
            message = f"{message} Reason: {reason}."
        warnings.append(warning(message))
    sobol = manifest.sobol or {}
    if sobol.get("mode") == "convergence" and sobol.get("reached") is False:
        samples = sobol.get("n_base_samples", sobol.get("max_base_samples", "unknown"))
        warnings.append(warning(f"Sobol convergence was not reached at {samples} base samples."))
    warnings.extend(_lineage_records(manifest=manifest, severity="WARNING"))
    warnings.extend(_ar6_uncertainty_availability_warnings(manifest=manifest))
    return tuple(warnings)


def _lineage_records(*, manifest: UncertaintyManifest, severity: str) -> tuple[SummaryWarning, ...]:
    """Return structured warning records persisted in uncertainty lineage."""
    lineage = manifest.lineage or {}
    records = lineage.get("summary_records") or ()
    out: list[SummaryWarning] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("severity", "")).strip().upper() != severity:
            continue
        message = str(record.get("message", "")).strip()
        if message:
            out.append(warning(message))
    return tuple(out)


def _ar6_uncertainty_availability_warnings(
    *,
    manifest: UncertaintyManifest,
) -> tuple[SummaryWarning, ...]:
    if manifest.family != "ar6_cc":
        return ()
    lineage = manifest.lineage or {}
    inventory = lineage.get("source_inventory", {})
    if not isinstance(inventory, dict):
        return ()
    raw_messages = inventory.get("scope_availability_messages") or ()
    messages = [str(message).strip() for message in raw_messages if str(message).strip()]
    if not messages:
        return ()
    deterministic_missing = any(
        payload.get("missing_pathway_combinations")
        for payload in manifest.deterministic_prerequisites
        if payload_source(payload) == "deterministic_ar6_cc"
    )
    warnings: list[SummaryWarning] = []
    for message in messages:
        lower = message.lower()
        duplicate_selector_warning = deterministic_missing and (
            "category" in lower or "ssp" in lower
        )
        if not duplicate_selector_warning:
            warnings.append(warning(message))
    return tuple(warnings)


def _source_owner_and_name(*, source: str, family: str) -> tuple[str, str]:
    if "::" in source:
        owner_key, source_name = source.split("::", 1)
    else:
        owner_key, source_name = family, source
    owners = {
        "asocc": "aSoCC",
        "ar6_cc": "dynamic AR6 CC",
        "io_lca": "IO-LCA",
        "external_lca": "external LCA",
        "acc": "aCC",
        "asr": "ASR",
    }
    return owners[owner_key], source_name
