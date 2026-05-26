"""Formatting for setup validation messages."""

from ...common_formatting import format_year_ranges


def _format_year_ranges(years: list[int]) -> str:
    """Format sorted years into compact ranges (e.g. 1995-2004, 2006)."""
    return format_year_ranges(years)


def _format_years_arg(years: list[int]) -> str:
    """Format years as a Python call argument string."""
    if not years:
        return "[]"
    ordered = sorted(set(int(y) for y in years))
    if len(ordered) > 1 and ordered[-1] - ordered[0] + 1 == len(ordered):
        return f"list(range({ordered[0]}, {ordered[-1] + 1}))"
    return str(ordered)


def _format_lcia_arg(methods: list[str] | None) -> str:
    """Format lcia_method argument for a Python call string."""
    clean = [str(m).strip() for m in (methods or []) if str(m).strip()]
    if not clean:
        return "None"
    if len(clean) == 1:
        return f"'{clean[0]}'"
    return str(sorted(set(clean)))


def _process_mrio_hint(
    *,
    source: str,
    years: list[int],
    agg_version: str | None,
    agg_reg: bool | None,
    agg_sec: bool | None,
    lcia_methods: list[str] | None = None,
    keep_intermediate_uncasext: bool = False,
) -> str:
    """Build a concrete process_mrio(...) call hint."""
    years_arg = _format_years_arg(years)
    lcia_arg = _format_lcia_arg(lcia_methods)
    extra_args = ", keep_intermediate_uncasext=True" if keep_intermediate_uncasext else ""
    if agg_version is None:
        if lcia_arg == "None":
            return f"process_mrio(source='{source}', years={years_arg}{extra_args})"
        return (
            f"process_mrio(source='{source}', years={years_arg}, "
            f"lcia_method={lcia_arg}{extra_args})"
        )
    if lcia_arg == "None":
        return (
            f"process_mrio(source='{source}', years={years_arg}, "
            f"agg_version='{agg_version}', agg_reg={bool(agg_reg)}, "
            f"agg_sec={bool(agg_sec)}{extra_args})"
        )
    return (
        f"process_mrio(source='{source}', years={years_arg}, "
        f"lcia_method={lcia_arg}, agg_version='{agg_version}', "
        f"agg_reg={bool(agg_reg)}, agg_sec={bool(agg_sec)}{extra_args})"
    )
