"""Per year parse/calc orchestration for MRIO processing."""

from pathlib import Path
from typing import Any, Optional, Tuple

import pymrio

from pyaesa.download.mrios.utils.logging import suppress_pymrio_logging
from pyaesa.process.mrios.utils.raw_corrections.runtime import (
    apply_raw_corrected_values,
)
from pyaesa.process.mrios.utils.aggregation.aggregation import (
    AggregationSpec,
    build_aggregation_spec,
)
from pyaesa.process.mrios.utils.parsers.exio_parser import (
    ExioCharacterizationOptions,
    _calc_characterized_extensions_minimal,
    _characterize_exiobase_io,
    _find_missing_characterization_extensions,
    _parse_exio_year,
    _retain_extension_instances,
)
from pyaesa.process.mrios.utils.parsers.oecd_parser import _parse_oecd_year

from pyaesa.process.mrios.utils.pipeline.contracts import SourceConfig
from .matrix_ops import (
    _aggregate_iosys_fast,
    _calc_aggregated_full_system_after_fast_aggregation,
    _calc_core_system_minimal,
    _labels_from_product_index,
)

_BASE_RETAINED_EXIO_EXTENSIONS: tuple[str, ...] = ("factor_inputs",)


def parse_and_calc_year(
    *,
    source: str,
    cfg: SourceConfig,
    full_dir: Path,
    year: int,
    char_jobs: Optional[dict[str, ExioCharacterizationOptions]],
    agg_reg: bool,
    agg_sec: bool,
    agg_reg_df,
    agg_sec_df,
    agg_reg_path: Optional[Path],
    agg_sec_path: Optional[Path],
    reg_vec_cache: Optional[dict[tuple[str, ...], AggregationSpec]] = None,
    sec_vec_cache: Optional[dict[tuple[str, ...], AggregationSpec]] = None,
    pymrio_calc_all: bool = False,
    keep_postclip_ghosh: bool = True,
    parse_exio_func=_parse_exio_year,
    parse_oecd_func=_parse_oecd_year,
    apply_raw_corrections_func=apply_raw_corrected_values,
) -> Tuple[
    pymrio.IOSystem,
    Optional[list[str]],
    Optional[dict[str, list[str]]],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    """Parse one year, optionally aggregate, then compute MRIO matrices."""

    def _ensure_label_list(value: Any, *, label_kind: str) -> list[str]:
        if value is None:
            raise ValueError(f"MRIO {label_kind} are missing (None).")
        try:
            items = list(value)
        except TypeError as exc:
            raise ValueError(f"MRIO {label_kind} are not iterable.") from exc
        if any(item is None for item in items):
            raise ValueError(f"MRIO {label_kind} contain None values.")
        return [str(item) for item in items]

    if cfg.requires_characterization:
        iosys = parse_exio_func(full_dir, year, system=cfg.exio_system)
        correction_summary = apply_raw_corrections_func(
            iosys=iosys,
            source=source,
            year=year,
        )
        if correction_summary is not None:
            setattr(iosys, "_raw_corrected_values_summary", correction_summary)
    else:
        iosys = parse_oecd_func(full_dir, year)

    regions_original = _ensure_label_list(iosys.get_regions(), label_kind="regions")
    sectors_original = _ensure_label_list(iosys.get_sectors(), label_kind="sectors")

    reg_spec = None
    sec_spec = None
    if agg_reg:
        reg_key = tuple(regions_original)
        if reg_vec_cache is not None and reg_key in reg_vec_cache:
            reg_spec = reg_vec_cache[reg_key]
        else:
            reg_spec = build_aggregation_spec(
                regions_original,
                agg_reg_df,
                label_kind="region",
                csv_path=agg_reg_path or "agg_reg",
            )
            if reg_vec_cache is not None:
                reg_vec_cache[reg_key] = reg_spec

    if agg_sec:
        sec_key = tuple(sectors_original)
        if sec_vec_cache is not None and sec_key in sec_vec_cache:
            sec_spec = sec_vec_cache[sec_key]
        else:
            sec_spec = build_aggregation_spec(
                sectors_original,
                agg_sec_df,
                label_kind="sector",
                csv_path=agg_sec_path or "agg_sec",
            )
            if sec_vec_cache is not None:
                sec_vec_cache[sec_key] = sec_spec

    aggregation_active = reg_spec is not None or sec_spec is not None
    if aggregation_active:
        _aggregate_iosys_fast(
            iosys=iosys,
            agg_reg=agg_reg,
            region_spec=reg_spec,
            sector_spec=sec_spec,
        )

    regions_used, sectors_used = _labels_from_product_index(iosys.Z.index)

    if cfg.requires_characterization and char_jobs:
        retained_instances: list[str] = []
        for options in char_jobs.values():
            for instance in options.retain_instances:
                if instance not in retained_instances:
                    retained_instances.append(instance)

        applied_methods: list[str] = []
        missing_by_method: dict[str, list[str]] = {}
        added_method_extensions: list[str] = []

        for lcia_method, options in char_jobs.items():
            char_matrix = options.char_matrix
            missing = _find_missing_characterization_extensions(iosys, options.requested_extensions)
            if missing:
                missing_by_method[lcia_method] = missing
                continue
            _characterize_exiobase_io(
                iosys,
                char_matrix=char_matrix,
                new_extension_name=lcia_method,
                retain_instances=options.retain_instances,
                prune=False,
                source_key=source,
                year=year,
            )
            added_method_extensions.append(lcia_method)
            applied_methods.append(lcia_method)

        if pymrio_calc_all:
            calc_all_methods = list(dict.fromkeys(added_method_extensions))
            retained_for_uncasext = [
                str(instance).strip() for instance in retained_instances if str(instance).strip()
            ]
            # For LCIA runs, calc_all is restricted to characterized LCIA methods only.
            # Retained UNCASExt extensions (e.g. factor_inputs) are restored afterward.
            retained_payload: dict[str, Any] = {}
            for inst_name in retained_for_uncasext:
                ext_obj = getattr(iosys, inst_name, None)
                if ext_obj is not None:
                    retained_payload[inst_name] = ext_obj

            _retain_extension_instances(iosys, calc_all_methods)
            if calc_all_methods:
                if aggregation_active:
                    _calc_aggregated_full_system_after_fast_aggregation(iosys=iosys)
                else:
                    with suppress_pymrio_logging():
                        iosys.calc_all(include_ghosh=True)
            else:
                _calc_core_system_minimal(
                    iosys=iosys,
                    include_ghosh=keep_postclip_ghosh,
                )
            for inst_name, payload in retained_payload.items():
                if getattr(iosys, inst_name, None) is None:
                    setattr(iosys, inst_name, payload)
        else:
            keep_instances = list(dict.fromkeys(retained_instances + added_method_extensions))
            if keep_instances:
                _retain_extension_instances(iosys, keep_instances)
                _calc_core_system_minimal(
                    iosys=iosys,
                    include_ghosh=keep_postclip_ghosh,
                )
                _calc_characterized_extensions_minimal(
                    iosys,
                    added_method_extensions,
                    keep_direct_intensities=keep_postclip_ghosh,
                )
            else:
                _calc_core_system_minimal(
                    iosys=iosys,
                    include_ghosh=keep_postclip_ghosh,
                )

        return (
            iosys,
            applied_methods,
            missing_by_method,
            regions_original,
            sectors_original,
            regions_used,
            sectors_used,
        )

    if pymrio_calc_all:
        if cfg.requires_characterization:
            # Without LCIA requests, exclude environmental extensions and keep
            # only retained non environmental accounts (factor_inputs).
            _retain_extension_instances(iosys, _BASE_RETAINED_EXIO_EXTENSIONS)
        if aggregation_active:
            _calc_aggregated_full_system_after_fast_aggregation(iosys=iosys)
        else:
            with suppress_pymrio_logging():
                iosys.calc_all(include_ghosh=True)
    else:
        if cfg.requires_characterization:
            _retain_extension_instances(iosys, _BASE_RETAINED_EXIO_EXTENSIONS)
        _calc_core_system_minimal(
            iosys=iosys,
            include_ghosh=keep_postclip_ghosh,
        )
    return (
        iosys,
        None,
        None,
        regions_original,
        sectors_original,
        regions_used,
        sectors_used,
    )
