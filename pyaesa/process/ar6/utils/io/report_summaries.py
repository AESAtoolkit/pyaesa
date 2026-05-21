"""AR6 variable coverage summary ownership for process reports and metadata."""

from dataclasses import dataclass
from typing import cast

from pyaesa.download.ar6.utils.config import PROCESSED_OUTPUT_VARIABLES

SUMMARY_VARIABLE_ORDER = PROCESSED_OUTPUT_VARIABLES


@dataclass(frozen=True)
class VariableCoverageSummaryAR6:
    """One variable specific coverage summary line for ``ProcessReportAR6``."""

    variable: str
    retained_model_scenario_pairs: int
    missing_reason_counts: dict[str, int]


def _summary_count_int(value: object) -> int:
    """Return one integer count from a processing owned summary payload."""
    return int(cast(int, value))


def _normalize_missing_reason_counts(value: object) -> dict[str, int]:
    """Return validated missing reason counts from a summary payload."""
    normalized: dict[str, int] = {}
    for reason_code, count in cast(dict[object, object], value).items():
        normalized_reason_code = str(reason_code)
        normalized_count = _summary_count_int(count)
        if normalized_reason_code and normalized_count > 0:
            normalized[normalized_reason_code] = normalized_count
    return dict(sorted(normalized.items()))


def serialize_variable_coverage_summary_counts(
    summary_counts: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    """Return a JSON safe representation of AR6 variable coverage summaries."""

    def _serialize_entry(variable: str, counts: dict[str, object]) -> dict[str, object]:
        available = _summary_count_int(counts.get("available_model_scenario_pairs"))
        retained = _summary_count_int(counts.get("retained_model_scenario_pairs"))
        missing_reason_counts = _normalize_missing_reason_counts(
            counts.get("missing_reason_counts"),
        )
        return {
            "variable": variable,
            "available_model_scenario_pairs": available,
            "retained_model_scenario_pairs": retained,
            "missing_reason_counts": [
                {"reason_code": reason_code, "count": count}
                for reason_code, count in sorted(missing_reason_counts.items())
                if count > 0
            ],
        }

    serialized: list[dict[str, object]] = []
    seen: set[str] = set()
    for variable in SUMMARY_VARIABLE_ORDER:
        counts = summary_counts.get(variable)
        if counts is None:
            continue
        serialized.append(_serialize_entry(variable, counts))
        seen.add(variable)
    for variable in sorted(set(summary_counts) - seen):
        serialized.append(_serialize_entry(variable, summary_counts[variable]))
    return serialized


def deserialize_variable_coverage_summary_counts(
    payload: object,
) -> dict[str, dict[str, object]]:
    """Parse package-owned AR6 variable coverage summary metadata."""
    out: dict[str, dict[str, object]] = {}
    for entry in cast(list[dict[str, object]], payload):
        variable = entry.get("variable")
        available = entry.get("available_model_scenario_pairs")
        retained = entry.get("retained_model_scenario_pairs")
        missing_reason_counts_payload = entry.get("missing_reason_counts", [])
        missing_reason_counts: dict[str, int] = {}
        for item in cast(list[dict[str, object]], missing_reason_counts_payload):
            reason_code = item.get("reason_code")
            count = item.get("count")
            if int(cast(int, count)) > 0:
                missing_reason_counts[str(reason_code)] = int(cast(int, count))
        out[str(variable)] = {
            "available_model_scenario_pairs": int(cast(int, available)),
            "retained_model_scenario_pairs": int(cast(int, retained)),
            "missing_reason_counts": missing_reason_counts,
        }
    return out


def summarize_variable_model_scenario_pairs(
    summary_counts: dict[str, dict[str, object]] | None,
) -> list[VariableCoverageSummaryAR6]:
    """Return ordered variable coverage summaries for ``ProcessReportAR6``."""
    if not summary_counts:
        return []
    summaries: list[VariableCoverageSummaryAR6] = []
    seen: set[str] = set()

    def _append_summary(variable: str, counts: dict[str, object]) -> None:
        retained = _summary_count_int(counts.get("retained_model_scenario_pairs"))
        if retained < 0:
            return
        missing_reason_counts = _normalize_missing_reason_counts(
            counts.get("missing_reason_counts"),
        )
        summaries.append(
            VariableCoverageSummaryAR6(
                variable=variable,
                retained_model_scenario_pairs=retained,
                missing_reason_counts=missing_reason_counts,
            )
        )

    for variable in SUMMARY_VARIABLE_ORDER:
        counts = summary_counts.get(variable)
        if counts is None:
            continue
        _append_summary(variable, counts)
        seen.add(variable)
    for variable in sorted(set(summary_counts) - seen):
        _append_summary(variable, summary_counts[variable])
    return summaries
