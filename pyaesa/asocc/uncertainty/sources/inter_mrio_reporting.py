"""Inter-MRIO uncertainty route reporting."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InterMrioRouteReport:
    """Comparable and skipped year sets for inter-MRIO interpolation."""

    interpolated_years: tuple[int, ...]
    skipped_years: tuple[int, ...]
    skipped_route_pairs: tuple[str, ...]
    skipped_scopes: tuple[str, ...]


def years_from_report(*, frame: pd.DataFrame) -> tuple[int, ...]:
    """Return sorted year values from an inter-MRIO route report frame."""
    if frame.empty:
        return ()
    year_column = "_year" if "_year" in frame.columns else "year"
    years = pd.Series(frame.loc[:, year_column], copy=False)
    return tuple(sorted(int(year) for year in pd.unique(years)))


def route_pairs_from_skipped(*, frame: pd.DataFrame) -> tuple[str, ...]:
    """Return compact main and alternate route pairs for skipped rows."""
    if frame.empty:
        return ()
    pairs = (
        frame.loc[:, ["_main_route", "_alternate_route"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(by=["_main_route", "_alternate_route"])
    )
    return tuple(
        f"main={main_route}; alternate={alternate_route}"
        for main_route, alternate_route in pairs.to_numpy(dtype=str)
    )


def scopes_from_skipped(*, frame: pd.DataFrame) -> tuple[str, ...]:
    """Return method and year scopes for skipped inter-MRIO rows."""
    if frame.empty:
        return ()
    scopes = (
        frame.loc[:, ["_method_label", "_year"]]
        .drop_duplicates()
        .sort_values(by=["_method_label", "_year"])
    )
    return tuple(f"{method} {int(year)}" for method, year in scopes.to_numpy(dtype=object))


def inter_mrio_notes(
    *,
    alternate_source: str,
    route_report: InterMrioRouteReport,
) -> str:
    """Return compact source method notes for inter-MRIO uncertainty."""
    base = (
        f"Alternate source: {alternate_source}. Eligible rows are final non LCIA rows with "
        "comparable deterministic time routes."
    )
    notes = f"{base} Interpolated years: {_format_years(route_report.interpolated_years)}."
    if not route_report.skipped_years:
        return notes
    return (
        f"{notes} Skipped years: {_format_years(route_report.skipped_years)}. "
        "Skipped years keep the main deterministic aSoCC value. Inter-MRIO interpolation "
        "is year level: all eligible rows in a year must have matched alternate rows, and "
        "each matched row must have the same deterministic time route on both endpoints. "
        "Incompatible route pairs: "
        f"{_format_route_pairs(route_report.skipped_route_pairs)}. Skipped method year "
        f"scopes grouped by method: {_format_skipped_scopes(route_report.skipped_scopes)}."
    )


def _format_years(years: tuple[int, ...]) -> str:
    return ", ".join(str(year) for year in years) if years else "none"


def _format_route_pairs(route_pairs: tuple[str, ...]) -> str:
    return " | ".join(route_pairs) if route_pairs else "none"


def _format_skipped_scopes(scopes: tuple[str, ...]) -> str:
    years_by_method: dict[str, list[int]] = {}
    for scope in scopes:
        method, year = scope.rsplit(" ", maxsplit=1)
        years_by_method.setdefault(method, []).append(int(year))
    return "; ".join(
        f"{method}: {_format_year_ranges(tuple(years))}"
        for method, years in sorted(years_by_method.items())
    )


def _format_year_ranges(years: tuple[int, ...]) -> str:
    ordered = sorted(set(years))
    ranges: list[str] = []
    start = ordered[0]
    previous = start
    for year in ordered[1:]:
        if year == previous + 1:
            previous = year
            continue
        ranges.append(_format_year_range(start=start, stop=previous))
        start = year
        previous = year
    ranges.append(_format_year_range(start=start, stop=previous))
    return ", ".join(ranges)


def _format_year_range(*, start: int, stop: int) -> str:
    return str(start) if start == stop else f"{start} to {stop}"
