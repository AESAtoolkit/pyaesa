"""Internal runner for the ``deterministic_ar6_cc(...)`` entrypoint."""

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from pyaesa.ar6_cc.deterministic.io.paths import (
    get_cc_figures_dir,
    get_cc_logs_dir,
    get_cc_metadata_path,
    get_cc_output_path,
    get_cc_post_study_output_path,
    get_cc_summary_log_path,
    get_cc_scope_dir,
)
from pyaesa.ar6_cc.deterministic.io.tables import (
    build_cc_table,
    cc_output_exists,
    filter_pathways,
    read_cc_output,
    read_harmonized_pathways,
    select_cc_year_columns,
    write_cc_output,
)
from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NEGATIVE,
    cc_positive_flow,
    cc_sequestration_variable,
    cc_variable,
    normalize_emission_type,
    normalize_emissions_mode,
)
from pyaesa.process.ar6.process_ar6 import process_ar6
from pyaesa.process.ar6.utils.io.reports import ProcessReportAR6
from pyaesa.process.ar6.utils.pipeline.runtime_helpers import validate_harmonization_method
from pyaesa.process.ar6.utils.pipeline.study_period import resolve_study_period
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter, PhasePrinter
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.reporting.summary_log import write_summary_log
from pyaesa.shared.runtime.manifest_contract import manifest_digest
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens
from pyaesa.shared.tabular.contracts import normalize_tabular_output_format

from ..shared.runtime.signatures import (
    build_cc_scope_signature,
    normalize_cc_category,
    normalize_cc_ssp_scenario,
)
from .figures.render import render_cc_pathway_figures
from .runtime.metadata import (
    build_cached_report,
    build_run_metadata_payload,
    clear_figure_state_paths,
    figure_state_matches,
    load_run_metadata,
    save_run_metadata,
    set_figure_state,
)
from .runtime.reports import AR6CCPathwayCount, ComputeAR6CCReport


def _build_figure_signature(
    *,
    figure_format: dict[str, object],
) -> dict[str, object]:
    """Return deterministic AR6 CC figure request signature."""
    return {
        "function": "deterministic_ar6_cc_figures",
        "contract": "ssp_overlay_categories_study_post_study_budget_panel_v1",
        "figure_format": dict(figure_format),
    }


def _show_status(*, status: StatusSink, message: str) -> None:
    """Render one transient deterministic AR6 CC status line."""
    status.show(message)


def _write_report_summary(*, report: ComputeAR6CCReport, summary_log: Path) -> ComputeAR6CCReport:
    """Persist the final deterministic AR6 CC summary and return the report."""
    write_summary_log(path=summary_log, summary=str(report))
    return report


def _filter_companion_to_positive(
    *,
    companion: pd.DataFrame,
    positive: pd.DataFrame,
) -> pd.DataFrame:
    """Return sequestration rows for the selected positive model-scenario scope."""
    key_columns = ["model", "scenario", "Category", "Ssp_family"]
    # Sequestration companions of gross emissionsare read from
    # the same processed sheet as their gross counterparts.
    keys = pd.MultiIndex.from_frame(positive.loc[:, key_columns])
    companion_keys = pd.MultiIndex.from_frame(companion.loc[:, key_columns])
    matched = companion.loc[companion_keys.isin(keys)].copy()
    positive_units = positive.loc[:, [*key_columns, "unit"]].drop_duplicates()
    matched = matched.merge(
        positive_units,
        on=key_columns,
        how="left",
        suffixes=("", "_selected_cc"),
    )
    matched["unit"] = matched["unit_selected_cc"]
    return matched.drop(columns=["unit_selected_cc"])


def _post_study_years(study_period: list[int]) -> list[int]:
    """Return post study years retained for AR6 CC reporting."""
    return list(range(int(study_period[1]) + 1, 2101))


def _cc_output_scope_ready(
    *,
    output_file: Path,
    post_study_output_file: Path | None,
) -> bool:
    """Return whether all deterministic CC output artifacts exist."""
    if not cc_output_exists(output_file=output_file):
        return False
    if post_study_output_file is None:
        return True
    return cc_output_exists(output_file=post_study_output_file)


def _ensure_processed_ar6_scope(
    *,
    study_period: list[int],
    harmonization: bool,
    harmonization_method: str,
    refresh: bool,
    status: PhasePrinter | NullPhasePrinter | None = None,
) -> ProcessReportAR6:
    """Ensure the matching processed AR6 workbook exists for this CC scope."""
    return process_ar6(
        years=range(study_period[0], study_period[1] + 1),
        figures=False,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        refresh=refresh,
        _status=status,
    )


def _process_ar6_payload(report: ProcessReportAR6) -> dict[str, object]:
    """Return structured process_ar6 prerequisite summary payload."""
    period = f"{int(report.study_period[0])}-{int(report.study_period[1])}"
    return {
        "reuse_status": report.reuse_status,
        "study_period": period,
        "categories": list(report.categories),
        "ssps": [int(value) for value in report.ssps],
        "harmonization": bool(report.harmonization),
        "harmonization_method": report.harmonization_method,
        "latest_historical_year": report.latest_historical_year,
        "harmonization_year_requested": report.harmonization_year_requested,
        "harmonization_year": report.harmonization_year,
        "harmonization_year_message": report.harmonization_year_message,
        "output_root": str(report.processed_dir),
        "output_files_available": report._public_output_file_count(),
        "figures_available": len(report.figure_files) if report.figure_files else None,
        "variable_coverage": [
            {
                "variable": entry.variable,
                "retained_model_scenario_pairs": int(entry.retained_model_scenario_pairs),
            }
            for entry in report.variable_coverage_summaries
        ],
    }


def _attach_process_ar6_payload(
    *,
    payload: dict[str, Any],
    process_report: ProcessReportAR6,
) -> None:
    """Persist structured process_ar6 prerequisite information in AR6 CC metadata."""
    provenance = payload.setdefault("provenance", {})
    provenance["process_ar6"] = _process_ar6_payload(process_report)


def _pathway_counts(
    *,
    cc_table: pd.DataFrame,
    requested_categories: list[str],
    requested_ssps: list[str],
) -> tuple[list[AR6CCPathwayCount], list[AR6CCPathwayCount]]:
    """Return retained and missing AR6 CC pathway counts for requested combinations."""
    pairs = cc_table.loc[
        :,
        ["cc_category", "ssp_scenario", "cc_model", "cc_scenario"],
    ].drop_duplicates()
    grouped = pairs.groupby(["cc_category", "ssp_scenario"], sort=True).size()
    retained: list[AR6CCPathwayCount] = []
    missing: list[AR6CCPathwayCount] = []
    for category in requested_categories:
        for ssp in requested_ssps:
            count = int(grouped.get((str(category), str(ssp)), 0))
            item = AR6CCPathwayCount(
                category=str(category),
                ssp_scenario=str(ssp),
                model_scenario_pairs=count,
            )
            if count > 0:
                retained.append(item)
            else:
                missing.append(item)
    return retained, missing


def run_deterministic_ar6_cc(
    *,
    years: list[int] | range,
    harmonization: bool,
    harmonization_method: str,
    category: str | list[str] | None,
    ssp_scenario: str | list[str] | None,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str,
    subset_version: str | None,
    output_format: str,
    figures: bool,
    figure_format: dict[str, Any],
    refresh: bool,
    _status: PhasePrinter | NullPhasePrinter,
) -> ComputeAR6CCReport:
    """Execute deterministic AR6 dynamic carrying capacity extraction."""
    study_period = resolve_study_period(years)
    harmonization_method = validate_harmonization_method(
        harmonization=harmonization,
        harmonization_method=harmonization_method,
    )
    fmt = normalize_tabular_output_format(output_format)
    figure_output_format = str(figure_format["format"])
    figure_dpi = int(figure_format["dpi"])
    cats = normalize_cc_category(category)
    ssps = normalize_cc_ssp_scenario(ssp_scenario)
    emission_type_norm = normalize_emission_type(emission_type)
    emissions_mode_norm = normalize_emissions_mode(emissions_mode)

    cc_dir = get_cc_scope_dir(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        emission_type=emission_type_norm,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode_norm,
        subset_version=subset_version,
        category=cats,
        ssp_scenario=ssps,
    )
    logs_dir = get_cc_logs_dir(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        emission_type=emission_type_norm,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode_norm,
        subset_version=subset_version,
        category=cats,
        ssp_scenario=ssps,
    )
    figures_dir = get_cc_figures_dir(
        study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        emission_type=emission_type_norm,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode_norm,
        subset_version=subset_version,
        category=cats,
        ssp_scenario=ssps,
    )
    out_file = get_cc_output_path(cc_dir=cc_dir, output_format=fmt)
    study_years = list(range(int(study_period[0]), int(study_period[1]) + 1))
    post_years = _post_study_years(study_period)
    post_out_file = (
        None if not post_years else get_cc_post_study_output_path(cc_dir=cc_dir, output_format=fmt)
    )
    meta_file = get_cc_metadata_path(cc_dir=cc_dir)
    summary_log_file = get_cc_summary_log_path(cc_dir=cc_dir)

    if refresh and cc_dir.exists():
        shutil.rmtree(cc_dir)

    requested_signature = build_cc_scope_signature(
        study_period=study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        emission_type=emission_type_norm,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode_norm,
        category=cats,
        ssp_scenario=ssps,
        subset_version=subset_version,
    )
    requested_identity = dict(requested_signature)
    requested_coverage = {"cc_category": list(cats), "ssp_scenario": list(ssps)}
    figure_compute_signature = {
        "identity_key": manifest_digest(requested_identity),
        "coverage": requested_coverage,
    }
    existing_meta_raw = load_run_metadata(meta_file)
    existing_meta = existing_meta_raw or {}
    figure_signature = _build_figure_signature(
        figure_format=figure_format,
    )
    reuse_mode = "compute"
    if existing_meta and not refresh:
        if str(existing_meta["reuse"]["identity_key"]) != manifest_digest(requested_identity):
            raise ValueError(
                "deterministic_ar6_cc cannot reuse deterministic scope "
                f"'{cc_dir}' because the AR6 CC selector identity changed. "
                "Use refresh=True for this selector scope."
            )
        reuse_mode = "reuse"
    if reuse_mode == "reuse":
        if not _cc_output_scope_ready(output_file=out_file, post_study_output_file=post_out_file):
            raise ValueError(
                "Existing deterministic_ar6_cc metadata marks this scope as complete, but one "
                f"or more output files are missing in '{cc_dir}'. Use refresh=True for this "
                "selector scope."
            )
    if reuse_mode == "reuse" and not refresh:
        if not figures:
            return _write_report_summary(
                report=build_cached_report(
                    payload=existing_meta,
                    study_period=study_period,
                    harmonization=harmonization,
                    harmonization_method=harmonization_method,
                    emission_type=emission_type_norm,
                    include_afolu=include_afolu,
                    emissions_mode=emissions_mode_norm,
                    subset_version=subset_version,
                    meta_file=meta_file,
                    cc_dir=cc_dir,
                    logs_dir=logs_dir,
                    figure_paths=[
                        Path(str(path))
                        for path in existing_meta["artifacts"].get("figure_paths", [])
                    ],
                ),
                summary_log=summary_log_file,
            )
        if figure_state_matches(
            payload=existing_meta,
            request_signature=figure_signature,
            compute_signature=figure_compute_signature,
        ):
            return _write_report_summary(
                report=build_cached_report(
                    payload=existing_meta,
                    study_period=study_period,
                    harmonization=harmonization,
                    harmonization_method=harmonization_method,
                    emission_type=emission_type_norm,
                    include_afolu=include_afolu,
                    emissions_mode=emissions_mode_norm,
                    subset_version=subset_version,
                    meta_file=meta_file,
                    cc_dir=cc_dir,
                    logs_dir=logs_dir,
                    figure_paths=[
                        Path(str(path))
                        for path in existing_meta["artifacts"].get("figure_paths", [])
                    ],
                ),
                summary_log=summary_log_file,
            )
        clear_figure_state_paths(payload=existing_meta)
        cc_table = read_cc_output(output_file=out_file, output_format=fmt)
        cc_table = cc_table.loc[
            cc_table["cc_category"].astype(str).isin(cats)
            & cc_table["ssp_scenario"].astype(str).isin(ssps)
        ].copy()
        post_cc_table = (
            None
            if post_out_file is None
            else read_cc_output(output_file=post_out_file, output_format=fmt)
        )
        if post_cc_table is not None:
            post_cc_table = post_cc_table.loc[
                post_cc_table["cc_category"].astype(str).isin(cats)
                & post_cc_table["ssp_scenario"].astype(str).isin(ssps)
            ].copy()
        figure_paths = render_cc_pathway_figures(
            cc_table=cc_table,
            post_study_cc_table=post_cc_table,
            variable_name=cc_variable(
                emission_type=emission_type_norm,
                include_afolu=include_afolu,
                emissions_mode=emissions_mode_norm,
            ),
            output_dir=figures_dir,
            dpi=figure_dpi,
            output_format=figure_output_format,
            requested_years=study_years,
            status=_status,
        )
        set_figure_state(
            payload=existing_meta,
            request_signature=figure_signature,
            compute_signature=figure_compute_signature,
            paths=figure_paths,
        )
        save_run_metadata(meta_file, existing_meta)
        return _write_report_summary(
            report=build_cached_report(
                payload=existing_meta,
                study_period=study_period,
                harmonization=harmonization,
                harmonization_method=harmonization_method,
                emission_type=emission_type_norm,
                include_afolu=include_afolu,
                emissions_mode=emissions_mode_norm,
                subset_version=subset_version,
                meta_file=meta_file,
                cc_dir=cc_dir,
                logs_dir=logs_dir,
                figure_paths=figure_paths,
                reuse_status="partially_reused",
            ),
            summary_log=summary_log_file,
        )

    process_report = _ensure_processed_ar6_scope(
        study_period=study_period,
        harmonization=harmonization,
        harmonization_method=harmonization_method,
        refresh=refresh,
        status=_status,
    )
    ar6_processed_dir = process_report.processed_dir

    figure_paths: list[Path] = []
    _show_status(
        status=_status,
        message="[deterministic_ar6_cc] Reading processed AR6 workbook",
    )
    pathways_df = read_harmonized_pathways(
        processed_dir=ar6_processed_dir,
        harmonization=harmonization,
    )

    variable = cc_variable(
        emission_type=emission_type_norm,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode_norm,
    )
    _show_status(
        status=_status,
        message="[deterministic_ar6_cc] Preparing AR6 CC pathways",
    )
    filtered = filter_pathways(
        pathways_df,
        variable=variable,
        category=cats,
        ssp_scenario=ssps,
        subset_version=subset_version,
        processed_dir=ar6_processed_dir,
    )

    all_output_years = [*study_years, *post_years]
    full_cc_tables = [
        build_cc_table(
            filtered,
            all_output_years,
            cc_flow=cc_positive_flow(emissions_mode=emissions_mode_norm),
            cc_variable=variable,
        )
    ]
    sequestration_variable = cc_sequestration_variable(emissions_mode=emissions_mode_norm)
    if sequestration_variable is not None:
        sequestration_filtered = filter_pathways(
            pathways_df,
            variable=sequestration_variable,
            category=cats,
            ssp_scenario=ssps,
            subset_version=subset_version,
            processed_dir=ar6_processed_dir,
        )
        sequestration_filtered = _filter_companion_to_positive(
            companion=sequestration_filtered,
            positive=filtered,
        )
        full_cc_tables.append(
            build_cc_table(
                sequestration_filtered,
                all_output_years,
                cc_flow=CC_FLOW_NEGATIVE,
                cc_variable=sequestration_variable,
                sign=-1.0,
            )
        )
    full_cc_table = pd.concat(full_cc_tables, ignore_index=True)
    cc_table = select_cc_year_columns(full_cc_table, study_years)
    post_cc_table = None if not post_years else select_cc_year_columns(full_cc_table, post_years)
    actual_cats = sorted(set(cc_table["cc_category"].astype(str)))
    actual_ssps = sorted(set(normalize_ssp_tokens(cc_table["ssp_scenario"].tolist())))
    n_pairs = len(cc_table[["cc_model", "cc_scenario"]].drop_duplicates())
    pathway_counts, missing_pathway_combinations = _pathway_counts(
        cc_table=cc_table,
        requested_categories=cats,
        requested_ssps=ssps,
    )
    _show_status(
        status=_status,
        message="[deterministic_ar6_cc] Writing outputs",
    )
    write_cc_output(
        cc_table,
        out_file,
        fmt,
    )
    if post_cc_table is not None and post_out_file is not None:
        write_cc_output(
            post_cc_table,
            post_out_file,
            fmt,
        )

    if figures and existing_meta_raw is not None:
        clear_figure_state_paths(payload=existing_meta)
    save_run_metadata(
        meta_file,
        build_run_metadata_payload(
            signature=requested_signature,
            identity_payload=requested_identity,
            coverage=requested_coverage,
            write_scope_identity=requested_signature,
            emission_type=emission_type_norm,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode_norm,
            cc_categories=actual_cats,
            ssp_scenarios=actual_ssps,
            total_model_scenario_pairs=n_pairs,
            pathway_counts=pathway_counts,
            missing_pathway_combinations=missing_pathway_combinations,
            output_file=out_file,
            post_study_output_file=post_out_file,
            process_ar6=_process_ar6_payload(process_report),
        ),
    )
    if figures:
        figure_paths = render_cc_pathway_figures(
            cc_table=cc_table,
            post_study_cc_table=post_cc_table,
            variable_name=variable,
            output_dir=figures_dir,
            dpi=figure_dpi,
            output_format=figure_output_format,
            requested_years=study_years,
            status=_status,
        )
        payload = load_run_metadata(meta_file) or {}
        set_figure_state(
            payload=payload,
            request_signature=figure_signature,
            compute_signature=figure_compute_signature,
            paths=figure_paths,
        )
        _attach_process_ar6_payload(payload=payload, process_report=process_report)
        save_run_metadata(meta_file, payload)
    return _write_report_summary(
        report=ComputeAR6CCReport(
            study_period=study_period,
            harmonization=harmonization,
            harmonization_method=harmonization_method,
            emission_type=emission_type_norm,
            include_afolu=include_afolu,
            emissions_mode=emissions_mode_norm,
            variable=variable,
            categories=actual_cats,
            ssp_scenarios=actual_ssps,
            subset_version=subset_version,
            total_model_scenario_pairs=n_pairs,
            pathway_counts=pathway_counts,
            missing_pathway_combinations=missing_pathway_combinations,
            output_file=out_file,
            post_study_output_file=post_out_file,
            figure_paths=figure_paths,
            meta_file=meta_file,
            cc_dir=cc_dir,
            logs_dir=logs_dir,
            process_ar6=_process_ar6_payload(process_report),
        ),
        summary_log=summary_log_file,
    )
