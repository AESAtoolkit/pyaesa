"""Progress and console formatting ownership for IO-LCA workflows."""

from pyaesa.asocc.orchestration.common_formatting import format_year_ranges
from pyaesa.shared.runtime.reporting.progress import YearProgressPrinter
from pyaesa.shared.runtime.reporting.status import StatusSink
from pyaesa.shared.runtime.text import compact_user_text, print_user_text_line


def source_prefix(*, source: str) -> str:
    """Return source tag prefix used by run time terminal messages."""
    return f"[{str(source).strip()}]"


def format_year_ranges_with_count(years: list[int]) -> str:
    """Format years as compact ranges with explicit cardinality."""
    unique_years = sorted({int(year) for year in years})
    return f"{format_year_ranges(unique_years)} ({len(unique_years)} year(s))"


def format_method_labels(method_labels: list[str]) -> str:
    """Return deterministic comma separated LCIA method labels."""
    clean = sorted(
        {
            str(lcia_method_label).strip()
            for lcia_method_label in method_labels
            if str(lcia_method_label).strip()
        }
    )
    return compact_user_text(", ".join(clean), max_chars=72) if clean else "none"


def format_indices_label(filters: dict[str, list[str] | None]) -> str:
    """Format selected index values as a concise deterministic label."""
    keys = ("r_p", "s_p", "r_c", "r_f")
    if all(not filters.get(key) for key in keys):
        return "all"
    parts: list[str] = []
    for key in keys:
        values = filters.get(key)
        if not values:
            continue
        joined = "+".join(str(value) for value in values)
        parts.append(f"{key}={joined}")
    return compact_user_text(", ".join(parts), max_chars=72)


def io_lca_banner(
    *,
    source: str,
    years: list[int],
    methods: list[str],
    fu_code: str,
    filters: dict[str, list[str] | None],
    upstream_analysis: bool,
    upstream_stages: int,
    status: StatusSink | None = None,
) -> None:
    """Print concise deterministic_io_lca run header in package consistent format."""
    prefix = "[deterministic_io_lca]"
    indices_label = format_indices_label(filters)
    _emit_status_line(
        f"{prefix} Starting deterministic_io_lca: fu={fu_code}, indices={indices_label}",
        status=status,
    )
    _emit_status_line(f"{prefix} Requested: {format_year_ranges_with_count(years)}", status=status)
    _emit_status_line(f"{prefix} LCIA methods: {format_method_labels(methods)}", status=status)
    if upstream_analysis:
        _emit_status_line(
            f"{prefix} Upstream analysis: years=all requested, stages={int(upstream_stages)}",
            status=status,
        )


def io_lca_mode_banner(
    *,
    source: str,
    fu_code: str,
    filters: dict[str, list[str] | None],
    mode_tag: str | None,
    status: StatusSink | None = None,
) -> None:
    """Print one concise branch line for deterministic_io_lca."""
    _emit_status_line(
        io_lca_mode_line(
            source=source,
            fu_code=fu_code,
            filters=filters,
            mode_tag=mode_tag,
        ),
        status=status,
    )


def io_lca_mode_line(
    *,
    source: str,
    fu_code: str,
    filters: dict[str, list[str] | None],
    mode_tag: str | None,
) -> str:
    """Return one concise branch line for deterministic_io_lca."""
    prefix = "[deterministic_io_lca]"
    indices_label = format_indices_label(filters)
    if mode_tag is None:
        return f"{prefix} Branch: fu={fu_code}, indices={indices_label}"
    return f"{prefix} Branch: fu={fu_code}, indices={indices_label}, group_indices={mode_tag}"


def year_progress(*, source: str, action: str, total: int) -> YearProgressPrinter:
    """Return shared year based progress printer."""
    return YearProgressPrinter(source=source, action=action, total=total, show_timing=False)


def _emit_status_line(line: str, *, status: StatusSink | None) -> None:
    if status is None:
        print_user_text_line(line)
        return
    status.show(line)
