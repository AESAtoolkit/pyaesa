"""Shared public formatting for process_ar6 variable coverage lines."""

from collections.abc import Iterable, Mapping


def process_ar6_coverage_line(*, variable: object, retained: object) -> str:
    """Return one process_ar6 retained model scenario count line."""
    return f"  {variable}: {retained} model-scenario pairs"


def process_ar6_coverage_lines_from_payload(
    coverage: Iterable[object],
) -> list[str]:
    """Return retained count lines from persisted process_ar6 coverage payloads."""
    lines: list[str] = []
    for item in coverage:
        if isinstance(item, Mapping):
            variable = item.get("variable")
            retained = item.get("retained_model_scenario_pairs")
            lines.append(process_ar6_coverage_line(variable=variable, retained=retained))
    return lines
