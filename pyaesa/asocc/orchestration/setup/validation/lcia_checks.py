"""LCIA-specific setup validation."""

from pyaesa.process.mrios.utils.io.metadata import (
    _get_year_entry,
    _read_metadata,
)

from ....data.lcia_status import resolve_lcia_status
from ....data.paths import _get_mrio_year_dir
from pyaesa.asocc.orchestration.setup.formatting.formatting import (
    _format_year_ranges,
    _format_years_arg,
)


def _validate_lcia_requirements(
    *,
    source: str,
    is_exio: bool,
    needs_lcia_flag: bool,
    lcia_methods: list[str] | None,
) -> None:
    """Validate LCIA prerequisites for selected methods."""
    if lcia_methods and not is_exio:
        raise ValueError(
            f"source='{source}' does not support lcia_method. "
            "OECD ICIO routes do not support LCIA compatibility for deterministic "
            "aSoCC. Use an EXIOBASE source for LCIA-dependent aSoCC workflows."
        )


def _validate_lcia_ready_for_domain(
    *,
    source: str,
    years: list[int],
    lcia_methods: list[str] | None,
    matrix_version: str | None,
    domain_label: str,
    agg_reg: bool | None = None,
    agg_sec: bool | None = None,
) -> None:
    """Validate LCIA availability for one MRIO domain."""

    methods = list(lcia_methods or [])
    if not methods:
        return
    meta = _read_metadata(source, matrix_version=matrix_version)
    missing_year_dirs: list[int] = []
    available_count_by_method: dict[str, int] = {m: 0 for m in methods}
    missing_by_method: dict[str, list[int]] = {m: [] for m in methods}
    for year in years:
        year_dir = _get_mrio_year_dir(
            source=source,
            year=year,
            agg_version=matrix_version,
        )
        if not year_dir.exists():
            missing_year_dirs.append(year)
            for lcia_method in methods:
                missing_by_method[lcia_method].append(int(year))
            continue
        year_entry = _get_year_entry(meta, year)
        if year_entry is None:
            for lcia_method in methods:
                missing_by_method[lcia_method].append(int(year))
            continue
        for lcia_method in methods:
            available, _ = resolve_lcia_status(year_entry, lcia_method)
            if available:
                available_count_by_method[lcia_method] += 1
                continue
            missing_by_method[lcia_method].append(int(year))
    methods_without_any_lcia = sorted(
        lcia_method for lcia_method, count in available_count_by_method.items() if count == 0
    )
    if missing_year_dirs or methods_without_any_lcia:
        missing_ranges = "; ".join(
            f"{lcia_method}: {_format_year_ranges(missing_by_method.get(lcia_method, []))}"
            for lcia_method in methods_without_any_lcia
        )
        methods_hint = methods_without_any_lcia or methods
        methods_arg = f"'{methods_hint[0]}'" if len(methods_hint) == 1 else str(methods_hint)
        years_hint = sorted(
            set(missing_year_dirs).union(
                *(set(missing_by_method.get(m, [])) for m in methods_without_any_lcia)
            )
        )
        if not years_hint:
            years_hint = sorted(set(int(y) for y in years))
        years_arg = _format_years_arg(years_hint)
        if matrix_version is None:
            process_hint = (
                f"process_mrio(source='{source}', years={years_arg}, lcia_method={methods_arg})"
            )
        else:
            process_hint = (
                f"process_mrio(source='{source}', years={years_arg}, "
                f"lcia_method={methods_arg}, agg_version='{matrix_version}', "
                f"agg_reg={bool(agg_reg)}, agg_sec={bool(agg_sec)})"
            )
        missing_parts: list[str] = []
        if missing_year_dirs:
            missing_parts.append(f"missing MRIO years={_format_year_ranges(missing_year_dirs)}")
        if missing_ranges:
            missing_parts.append(f"missing LCIA methods/years={missing_ranges}")
        missing_msg = "; ".join(missing_parts) if missing_parts else "LCIA unavailable"
        raise ValueError(
            "LCIA-dependent allocation methods require LCIA data, but it is missing. "
            f"{missing_msg}. Domain={domain_label}. Run: {process_hint}"
        )


def _validate_aggregated_lcia_ready(
    *,
    source: str,
    years: list[int],
    lcia_methods: list[str] | None,
    agg_version: str | None,
    agg_reg: bool | None = None,
    agg_sec: bool | None = None,
) -> None:
    """Validate active aggregated domain LCIA inputs used in the run."""
    _validate_lcia_ready_for_domain(
        source=source,
        years=years,
        lcia_methods=lcia_methods,
        matrix_version=agg_version,
        domain_label=("aggregated" if agg_version else "original"),
        agg_reg=agg_reg,
        agg_sec=agg_sec,
    )


def _validate_original_lcia_ready(
    *,
    source: str,
    years: list[int],
    lcia_methods: list[str] | None,
) -> None:
    """Validate original domain LCIA inputs required by post mode L1 methods."""
    _validate_lcia_ready_for_domain(
        source=source,
        years=years,
        lcia_methods=lcia_methods,
        matrix_version=None,
        domain_label="original",
    )
