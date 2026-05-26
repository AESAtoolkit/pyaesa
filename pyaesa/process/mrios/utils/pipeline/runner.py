"""Process MRIO runtime orchestration below the public entry point."""

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, cast

import pandas as pd
import pymrio

from pyaesa.download.mrios.utils.logging import suppress_pymrio_logging
from pyaesa.download.mrios.utils.paths import _get_full_dir
from pyaesa.download.mrios.utils.source_registry import (
    is_exio_mrio_source,
    normalize_mrio_source_key,
)
from pyaesa.download.mrios.utils.year_selection import (
    YearSelection,
    normalize_mrio_years,
)
from pyaesa.process.mrios.utils.io.metadata import (
    _get_year_entry,
    _metadata_satisfies,
    _read_metadata,
    _remove_year_entry,
    _set_year_entry,
    _write_metadata,
)
from pyaesa.process.mrios.utils.io.paths import (
    _get_metadata_path,
    _get_mrio_clipping_log_path,
    _get_saved_dir,
    _get_year_saved_path,
    _resolve_version_tag,
)
from pyaesa.process.mrios.utils.aggregation.aggregation import AggregationSpec
from pyaesa.process.mrios.utils.parsers.exio_parser import (
    ExioCharacterizationOptions,
    _build_characterization_jobs,
    _calc_characterized_extensions_minimal,
)
from pyaesa.process.mrios.utils.pipeline.contracts import (
    SOURCE_CONFIGS,
    ProcessReportMRIO,
    SourceConfig,
    UNCASEXT_INTERMEDIATE_CORE_MATRICES,
)
from pyaesa.process.mrios.utils.pipeline.aggregation_validation import validate_metadata_aggregation
from pyaesa.process.mrios.utils.pipeline.lcia_tracking import (
    build_year_entry_payload,
    expected_pyaesa_lcia_methods,
    merge_lcia_method_lists,
    resolve_lcia_units_for_methods,
)
from pyaesa.process.mrios.utils.pipeline.matrix_ops import _calc_core_system_minimal
from pyaesa.process.mrios.utils.pipeline.metadata_payloads import (
    extract_preclip_extension_payload,
    extract_root_extension_payload,
)
from pyaesa.process.mrios.utils.pipeline.persistence import (
    preclip_calc_all_outputs_exist,
    preclip_core_outputs_exist,
    preclip_extension_outputs_exist,
    prune_saved_dir,
    save_lcia_fy_pickles,
    save_minimal_core_pickles,
    save_preclip_core_pickles,
    save_pymrio_calc_all_extensions,
    save_pyaesa_extension_pickles,
    summarize_preclip_core,
    summarize_saved,
    pyaesa_core_outputs_exist,
    pyaesa_extension_outputs_exist,
    pyaesa_outputs_exist,
)
from pyaesa.process.mrios.utils.pipeline.process_setup import (
    _normalize_lcia_methods,
    _resolve_aggregation_inputs,
    _resolve_year_characterization_jobs,
    _update_report_clipping_stats,
)
from pyaesa.process.mrios.utils.pipeline.runtime_environment import runtime_env_versions
from pyaesa.process.mrios.utils.pipeline.year_orchestrator import parse_and_calc_year
from pyaesa.process.mrios.utils.raw_corrections.reporting import record_raw_correction_payload
from pyaesa.process.mrios.utils.uncasext_metrics.enacting_metric import (
    _precompute_enacting_metrics_uncasext,
)
from pyaesa.process.mrios.utils.uncasext_metrics.utility_propagation_metrics import (
    _precompute_utility_propag_uncasext,
)
from pyaesa.shared.runtime.reporting.progress import YearProgressPrinter

_RECOVERABLE_YEAR_EXCEPTIONS = (
    TypeError,
    ValueError,
    FileNotFoundError,
    OSError,
    pd.errors.ParserError,
    pd.errors.EmptyDataError,
)


@dataclass(frozen=True)
class YearReusePlan:
    """Resolved processed year reuse state for one process_mrio call."""

    years_to_process: list[int]
    year_saved_dir_map: dict[int, Path]
    year_meta_map: dict[int, dict[str, object] | None]


@dataclass
class YearProcessContext:
    """Shared state used while processing selected MRIO years."""

    source_key: str
    cfg: SourceConfig
    full_dir: Path
    metadata: dict[str, object]
    metadata_path: Path
    matrix_version: str | None
    version_tag: str
    is_exio: bool
    refresh: bool
    agg_reg: bool
    agg_sec: bool
    agg_reg_df: pd.DataFrame | None
    agg_sec_df: pd.DataFrame | None
    agg_reg_path: Path | None
    agg_sec_path: Path | None
    keep_intermediate_uncasext: bool
    pymrio_calc_all: bool
    requested_lcia_methods: list[str]
    char_jobs_cache: dict[str, ExioCharacterizationOptions]
    reg_vec_cache: dict[tuple[str, ...], AggregationSpec]
    sec_vec_cache: dict[tuple[str, ...], AggregationSpec]
    runtime_env: dict[str, str]
    report: ProcessReportMRIO
    progress: YearProgressPrinter


def run_process_mrio(
    source: str,
    years: YearSelection = None,
    *,
    refresh: bool = False,
    lcia_method: Optional[str | Sequence[str]] = None,
    agg_reg: bool = False,
    agg_sec: bool = False,
    agg_version: Optional[str] = None,
    keep_intermediate_uncasext: bool = False,
    pymrio_calc_all: bool = False,
) -> ProcessReportMRIO | None:
    """Process MRIO archives into processed assets for selected years."""
    source_key = normalize_mrio_source_key(source)
    cfg = SOURCE_CONFIGS[source_key]
    is_exio = is_exio_mrio_source(source_key)
    lcia_methods = _normalize_lcia_methods(lcia_method)
    if (not is_exio) and lcia_methods:
        raise ValueError(
            "lcia_method is only supported for EXIOBASE MRIO sources. "
            "OECD ICIO does not support LCIA characterization."
        )

    agg_version_clean = _resolve_agg_version(
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version,
    )
    matrix_version = agg_version_clean if (agg_reg or agg_sec) else None
    version_tag = _resolve_version_tag(matrix_version)
    years_list = [int(year) for year in normalize_mrio_years(years, source_key=source_key)]
    report = ProcessReportMRIO(source=source_key, requested=years_list)
    report.saved_root = _get_saved_dir(source_key, matrix_version=matrix_version)
    report.clipping_log_path = _get_mrio_clipping_log_path(
        source_key,
        matrix_version=matrix_version,
    )
    metadata = _read_metadata(source_key, matrix_version=matrix_version)
    metadata_path = _get_metadata_path(source_key, matrix_version=matrix_version)

    (
        agg_reg_path,
        agg_sec_path,
        agg_reg_df,
        agg_sec_df,
        aggregation_payload,
    ) = _resolve_aggregation_inputs(
        source=source_key,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_version=agg_version_clean,
    )
    validate_metadata_aggregation(
        metadata=metadata,
        version_tag=version_tag,
        aggregation_payload=aggregation_payload,
        agg_reg=agg_reg,
        agg_sec=agg_sec,
        agg_reg_df=agg_reg_df,
        agg_sec_df=agg_sec_df,
        agg_reg_path=agg_reg_path,
        agg_sec_path=agg_sec_path,
        metadata_path=metadata_path,
    )
    if not metadata.get("version_tag"):
        metadata["version_tag"] = version_tag
    if not metadata.get("aggregation"):
        metadata["aggregation"] = aggregation_payload

    requested_char_jobs: dict[str, ExioCharacterizationOptions] = {}
    char_jobs_cache: dict[str, ExioCharacterizationOptions] = {}
    if cfg.requires_characterization:
        requested_char_jobs = _build_characterization_jobs(
            source_key=source_key,
            lcia_methods=lcia_methods,
        )
        char_jobs_cache.update(requested_char_jobs)
    requested_lcia_methods = list(requested_char_jobs.keys())

    reuse_plan = _resolve_year_reuse_plan(
        years_list=years_list,
        source_key=source_key,
        matrix_version=matrix_version,
        metadata=metadata,
        cfg=cfg,
        is_exio=is_exio,
        requested_lcia_methods=requested_lcia_methods,
        requested_char_jobs=requested_char_jobs,
        refresh=refresh,
        keep_intermediate_uncasext=keep_intermediate_uncasext,
        pymrio_calc_all=pymrio_calc_all,
        report=report,
    )
    progress = YearProgressPrinter(
        source=source_key,
        action="processing",
        total=len(reuse_plan.years_to_process),
    )
    try:
        context = YearProcessContext(
            source_key=source_key,
            cfg=cfg,
            full_dir=_get_full_dir(source_key),
            metadata=metadata,
            metadata_path=metadata_path,
            matrix_version=matrix_version,
            version_tag=version_tag,
            is_exio=is_exio,
            refresh=refresh,
            agg_reg=agg_reg,
            agg_sec=agg_sec,
            agg_reg_df=agg_reg_df,
            agg_sec_df=agg_sec_df,
            agg_reg_path=agg_reg_path,
            agg_sec_path=agg_sec_path,
            keep_intermediate_uncasext=keep_intermediate_uncasext,
            pymrio_calc_all=pymrio_calc_all,
            requested_lcia_methods=requested_lcia_methods,
            char_jobs_cache=char_jobs_cache,
            reg_vec_cache={},
            sec_vec_cache={},
            runtime_env=runtime_env_versions(),
            report=report,
            progress=progress,
        )
        for year in reuse_plan.years_to_process:
            _process_mrio_year(
                context=context,
                year=year,
                saved_dir=reuse_plan.year_saved_dir_map[year],
                year_meta=reuse_plan.year_meta_map.get(year),
            )
    finally:
        progress.finish()

    _write_metadata(source_key, metadata, matrix_version=matrix_version)
    if report.processed or report.errors:
        return report
    return None


def _resolve_agg_version(
    *,
    agg_reg: bool,
    agg_sec: bool,
    agg_version: Optional[str],
) -> str | None:
    """Return normalized aggregate version after public aggregation argument validation."""
    cleaned = "" if agg_version is None else str(agg_version).strip()
    if agg_reg or agg_sec:
        if not cleaned:
            raise ValueError("agg_version must be provided when agg_reg or agg_sec is True.")
        return cleaned
    if cleaned:
        raise ValueError(
            "agg_version was provided but aggregation is disabled. "
            "Either enable aggregation or omit agg_version."
        )
    return None


def _resolve_year_reuse_plan(
    *,
    years_list: list[int],
    source_key: str,
    matrix_version: str | None,
    metadata: dict[str, object],
    cfg: SourceConfig,
    is_exio: bool,
    requested_lcia_methods: list[str],
    requested_char_jobs: dict[str, ExioCharacterizationOptions],
    refresh: bool,
    keep_intermediate_uncasext: bool,
    pymrio_calc_all: bool,
    report: ProcessReportMRIO,
) -> YearReusePlan:
    """Return years that require processing and record satisfied years in the report."""
    year_saved_dir_map: dict[int, Path] = {}
    year_meta_map: dict[int, dict[str, object] | None] = {}
    years_to_process: list[int] = []

    for year in years_list:
        saved_dir = _get_year_saved_path(source_key, year, matrix_version=matrix_version)
        saved_exists = saved_dir.exists()
        year_saved_dir_map[year] = saved_dir
        year_meta = _get_year_entry(metadata, year)
        year_meta_map[year] = year_meta
        expected_lcia_methods = (
            expected_pyaesa_lcia_methods(
                year_meta=year_meta,
                requested_methods=requested_lcia_methods,
            )
            if is_exio and bool(requested_lcia_methods)
            else None
        )
        if refresh:
            if saved_dir.exists():
                shutil.rmtree(saved_dir)
            _remove_year_entry(metadata, year)
            years_to_process.append(year)
            continue

        if _year_outputs_satisfied(
            saved_dir=saved_dir,
            saved_exists=saved_exists,
            year_meta=year_meta,
            cfg=cfg,
            expected_lcia_methods=expected_lcia_methods,
            requested_char_jobs=requested_char_jobs,
            keep_intermediate_uncasext=keep_intermediate_uncasext,
            pymrio_calc_all=pymrio_calc_all,
        ):
            report.skipped_already_saved.append(year)
            report.saved_dirs[year] = saved_dir
            continue
        years_to_process.append(year)

    return YearReusePlan(
        years_to_process=years_to_process,
        year_saved_dir_map=year_saved_dir_map,
        year_meta_map=year_meta_map,
    )


def _year_outputs_satisfied(
    *,
    saved_dir: Path,
    saved_exists: bool,
    year_meta: dict[str, object] | None,
    cfg: SourceConfig,
    expected_lcia_methods: list[str] | None,
    requested_char_jobs: dict[str, ExioCharacterizationOptions],
    keep_intermediate_uncasext: bool,
    pymrio_calc_all: bool,
) -> bool:
    """Return whether a processed year already satisfies the requested output scope."""
    pyaesa_ready = True
    preclip_ready = True
    pyaesa_core_ready = True
    pyaesa_extensions_ready = True
    if saved_exists:
        pyaesa_ready = pyaesa_outputs_exist(
            saved_dir,
            is_exio_source=cfg.requires_characterization,
            lcia_methods=expected_lcia_methods,
        )
        if pymrio_calc_all:
            preclip_ready = preclip_core_outputs_exist(saved_dir, core_matrices=cfg.required_core)
        if preclip_ready and pymrio_calc_all:
            preclip_ready = preclip_calc_all_outputs_exist(saved_dir)
        if preclip_ready and pymrio_calc_all:
            preclip_ready = preclip_extension_outputs_exist(
                saved_dir,
                extension_payload=extract_preclip_extension_payload(year_meta),
            )
        if keep_intermediate_uncasext:
            pyaesa_core_ready = pyaesa_core_outputs_exist(
                saved_dir,
                core_matrices=UNCASEXT_INTERMEDIATE_CORE_MATRICES,
            )
            if pyaesa_core_ready:
                pyaesa_extensions_ready = pyaesa_extension_outputs_exist(
                    saved_dir,
                    extension_payload=extract_root_extension_payload(year_meta),
                )
    return (
        _metadata_satisfies(
            year_meta,
            saved_exists=saved_exists,
            required_core=(
                UNCASEXT_INTERMEDIATE_CORE_MATRICES if keep_intermediate_uncasext else ()
            ),
            required_extensions=cfg.required_extensions if pymrio_calc_all else (),
            required_lcia_method=None,
            required_lcia_methods=(
                expected_lcia_methods
                if (
                    cfg.requires_characterization
                    and requested_char_jobs
                    and keep_intermediate_uncasext
                )
                else None
            ),
        )
        and pyaesa_ready
        and preclip_ready
        and pyaesa_core_ready
        and pyaesa_extensions_ready
    )


def _process_mrio_year(
    *,
    context: YearProcessContext,
    year: int,
    saved_dir: Path,
    year_meta: dict[str, object] | None,
) -> None:
    """Process one MRIO year and update report plus metadata in place."""
    context.progress.begin_year(year)
    year_lcia_methods = _year_lcia_methods(context=context, year_meta=year_meta)
    year_char_jobs: dict[str, ExioCharacterizationOptions] = {}
    year_lcia_units_by_method: dict[str, dict[str, str]] = {}
    if context.cfg.requires_characterization:
        (
            year_char_jobs,
            year_lcia_units_by_method,
        ) = _resolve_year_characterization_jobs(
            source=context.source_key,
            year_lcia_methods=year_lcia_methods,
            char_jobs_cache=context.char_jobs_cache,
        )

    try:
        with suppress_pymrio_logging():
            (
                iosys,
                applied_methods,
                missing_by_method,
                regions_original,
                sectors_original,
                regions_used,
                sectors_used,
            ) = parse_and_calc_year(
                source=context.source_key,
                cfg=context.cfg,
                full_dir=context.full_dir,
                year=year,
                char_jobs=year_char_jobs,
                agg_reg=context.agg_reg,
                agg_sec=context.agg_sec,
                agg_reg_df=context.agg_reg_df,
                agg_sec_df=context.agg_sec_df,
                agg_reg_path=context.agg_reg_path,
                agg_sec_path=context.agg_sec_path,
                reg_vec_cache=context.reg_vec_cache,
                sec_vec_cache=context.sec_vec_cache,
                pymrio_calc_all=context.pymrio_calc_all,
                keep_postclip_ghosh=context.keep_intermediate_uncasext,
            )
    except _RECOVERABLE_YEAR_EXCEPTIONS as exc:
        context.report.errors[year] = f"{exc}"
        context.progress.complete_year(year)
        context.progress.log_message(f"[{context.source_key}] {year}: failed -> {exc}")
        return

    if missing_by_method:
        context.report.lcia_missing_by_year[year] = {
            str(lcia_method): sorted({str(name) for name in missing})
            for lcia_method, missing in missing_by_method.items()
            if missing
        }

    labels_payload = {
        "regions_original": regions_original,
        "sectors_original": sectors_original,
        "regions_used": regions_used,
        "sectors_used": sectors_used,
    }
    _record_label_payload(
        metadata=context.metadata,
        labels_payload=labels_payload,
        source_key=context.source_key,
        version_tag=context.version_tag,
        year=year,
        metadata_path=context.metadata_path,
    )

    raw_correction_payload = record_raw_correction_payload(
        iosys=iosys,
        report=context.report,
        source_key=context.source_key,
        matrix_version=context.matrix_version,
        saved_dir=saved_dir,
        year=year,
    )
    lcia_methods_for_enacting_metric = (
        list(applied_methods) if context.is_exio and applied_methods else None
    )
    preclip_extensions_summary = _save_optional_outputs(
        iosys=iosys,
        context=context,
        saved_dir=saved_dir,
        year_lcia_methods=year_lcia_methods,
        applied_methods=lcia_methods_for_enacting_metric,
    )
    lcia_units_for_enacting_metric = resolve_lcia_units_for_methods(
        lcia_method_names=lcia_methods_for_enacting_metric,
        units_by_method=year_lcia_units_by_method,
    )
    enacting_metric_units = _precompute_enacting_metrics_uncasext(
        iosys=iosys,
        saved_dir=saved_dir,
        source_key=context.source_key,
        refresh=context.refresh,
        lcia_methods=lcia_methods_for_enacting_metric,
        matrix_version=context.matrix_version,
        lcia_units_by_method=lcia_units_for_enacting_metric,
    )
    context.report.clipping_unit = (
        str(enacting_metric_units.get("mrio_default_monetary", "")).strip()
        or context.report.clipping_unit
    )
    _update_report_clipping_stats(context.report, iosys)
    if context.is_exio and lcia_methods_for_enacting_metric:
        save_lcia_fy_pickles(
            iosys=iosys,
            saved_dir=saved_dir,
            lcia_method_names=lcia_methods_for_enacting_metric,
        )
    if not context.keep_intermediate_uncasext and not context.pymrio_calc_all:
        prune_saved_dir(saved_dir, keep_dirs=("utility_propag_uncasext", "enacting_metrics"))

    del iosys

    matrices_summary = (
        {"core": [], "extensions": {}}
        if (not context.keep_intermediate_uncasext and not context.pymrio_calc_all)
        else summarize_saved(saved_dir)
    )
    year_entry = build_year_entry_payload(
        saved_dir_name=saved_dir.name,
        core_matrices=matrices_summary["core"],
        extension_payload=matrices_summary["extensions"],
        updated_iso=datetime.now().isoformat(),
        uncasext_only=not context.keep_intermediate_uncasext and not context.pymrio_calc_all,
        preclip_core_matrices=summarize_preclip_core(saved_dir),
        preclip_extension_payload=preclip_extensions_summary,
        pymrio_calc_all=context.pymrio_calc_all,
        enacting_metric_units=enacting_metric_units,
        applied_methods=lcia_methods_for_enacting_metric,
        is_exio=context.is_exio,
        requires_characterization=context.cfg.requires_characterization,
        year_char_jobs=year_char_jobs,
        missing_by_method=missing_by_method,
        runtime_env=context.runtime_env,
        raw_correction_payload=raw_correction_payload,
    )

    _set_year_entry(context.metadata, year, year_entry)
    _write_metadata(context.source_key, context.metadata, matrix_version=context.matrix_version)
    context.report.processed.append(year)
    context.report.saved_dirs[year] = saved_dir
    context.progress.complete_year(year)


def _year_lcia_methods(
    *,
    context: YearProcessContext,
    year_meta: dict[str, object] | None,
) -> list[str]:
    """Return LCIA methods to attempt for one processed year."""
    if (not context.refresh) and context.requested_lcia_methods:
        return (
            expected_pyaesa_lcia_methods(
                year_meta=year_meta,
                requested_methods=context.requested_lcia_methods,
            )
            or []
        )
    return merge_lcia_method_lists((), context.requested_lcia_methods)


def _record_label_payload(
    *,
    metadata: dict[str, object],
    labels_payload: dict[str, list[str]],
    source_key: str,
    version_tag: str,
    year: int,
    metadata_path: Path,
) -> None:
    """Record processed axis labels and fail on cross year label drift."""
    existing_labels = metadata.get("labels", {})
    if existing_labels and existing_labels != labels_payload:
        label_keys = ("regions_original", "sectors_original", "regions_used", "sectors_used")
        existing_label_payload = cast(dict[str, list[str]], existing_labels)
        stored_label_values = {
            key: [str(value) for value in existing_label_payload[key]] for key in label_keys
        }
        current_label_values = {
            key: [str(value) for value in labels_payload[key]] for key in label_keys
        }
        stored_counts = {key: len(stored_label_values[key]) for key in label_keys}
        current_counts = {key: len(current_label_values[key]) for key in label_keys}
        stored_only = {
            key: sorted(set(stored_label_values[key]) - set(current_label_values[key]))[:5]
            for key in label_keys
        }
        current_only = {
            key: sorted(set(current_label_values[key]) - set(stored_label_values[key]))[:5]
            for key in label_keys
        }
        action = (
            "Process a compatible year set for this MRIO source. For aggregated scopes, "
            "use a different agg_version for incompatible aggregated axes or refresh "
            "the custom classified processed MRIO scope."
        )
        raise ValueError(
            "MRIO label metadata is incompatible across processed years. "
            f"source='{source_key}', processed_scope='{version_tag}', year={year}, "
            f"metadata={metadata_path}. Stored counts={stored_counts}. "
            f"Current counts={current_counts}. Stored only sample={stored_only}. "
            f"Current only sample={current_only}. {action}"
        )
    if not existing_labels:
        metadata["labels"] = labels_payload


def _save_optional_outputs(
    *,
    iosys: pymrio.IOSystem,
    context: YearProcessContext,
    saved_dir: Path,
    year_lcia_methods: Sequence[str],
    applied_methods: list[str] | None,
) -> dict[str, list[str]]:
    """Persist optional PyMRIO and UNCASExt intermediate outputs."""
    preclip_extensions_summary: dict[str, list[str]] = {}
    if context.pymrio_calc_all:
        preclip_include_extensions: Sequence[str] | None = None
        if context.cfg.requires_characterization and year_lcia_methods:
            preclip_include_extensions = list(
                dict.fromkeys(["factor_inputs"] + list(applied_methods or []))
            )
        save_preclip_core_pickles(
            iosys=iosys,
            saved_dir=saved_dir,
            core_matrices=context.cfg.required_core,
        )
        preclip_extensions_summary = save_pymrio_calc_all_extensions(
            iosys=iosys,
            saved_dir=saved_dir,
            include_extensions=preclip_include_extensions,
        )
        _calc_core_system_minimal(
            iosys=iosys,
            include_ghosh=context.keep_intermediate_uncasext,
        )
        if applied_methods:
            _calc_characterized_extensions_minimal(
                iosys,
                applied_methods,
                keep_direct_intensities=context.keep_intermediate_uncasext,
            )

    if context.keep_intermediate_uncasext:
        save_minimal_core_pickles(
            iosys=iosys,
            saved_dir=saved_dir,
            core_matrices=UNCASEXT_INTERMEDIATE_CORE_MATRICES,
        )
        save_pyaesa_extension_pickles(
            iosys=iosys,
            saved_dir=saved_dir,
            lcia_methods=applied_methods,
        )

    _precompute_utility_propag_uncasext(
        iosys=iosys,
        saved_dir=saved_dir,
        refresh=context.refresh,
        source_key=context.source_key,
        matrix_version=context.matrix_version,
    )
    if not context.keep_intermediate_uncasext and not context.pymrio_calc_all:
        setattr(iosys, "L", None)
    return preclip_extensions_summary
