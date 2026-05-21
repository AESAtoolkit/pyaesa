"""Resolve explicit LCIA method rule bindings and bundled static CC metadata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from .static_cc_schema import NormalizedStaticCCRow, StaticCCSchemaKind, standardize_static_cc_rows

STATIC_CC_SUFFIX = "_cc_steady_state.csv"
CHARACTERIZATION_FACTORS_MATRICES_DIRNAME = "characterization_factors_matrices"
RESPONSIBILITY_PERIODS_DIRNAME = "responsibility_periods"


@dataclass(frozen=True)
class LCIAMethodRule:
    """Explicit LCIA runtime rules that cannot be inferred from files."""

    dynamic_cc_match: dict[str, str] | None = None


_LCIA_METHOD_RULES = {
    "ef_3.1": LCIAMethodRule(dynamic_cc_match={"impact": "GWP_100"}),
    "gwp100_lcia": LCIAMethodRule(dynamic_cc_match={"impact": "GWP_100"}),
}


def normalize_lcia_method_name(lcia_method: str) -> str:
    """Return one required LCIA method token as stripped text."""
    cleaned = str(lcia_method).strip()
    if not cleaned:
        raise ValueError("LCIA method name cannot be empty.")
    return cleaned


def _normalize_dynamic_cc_match(
    *,
    lcia_method: str,
    match: dict[str, str] | None,
) -> dict[str, str] | None:
    """Return one validated dynamic CC binding or ``None`` when absent."""
    if match is None:
        return None
    impact = str(match.get("impact", "")).strip()
    if not impact:
        raise ValueError(
            "LCIA dynamic CC rule definitions must provide a non-empty 'impact' "
            f"binding. Method='{lcia_method}'."
        )
    return {"impact": impact}


def _normalize_lcia_method_rules(
    rules: dict[str, LCIAMethodRule],
) -> dict[str, LCIAMethodRule]:
    """Return canonical LCIA rules with validated dynamic CC bindings."""
    normalized: dict[str, LCIAMethodRule] = {}
    for lcia_method, rule in rules.items():
        normalized[lcia_method] = LCIAMethodRule(
            dynamic_cc_match=_normalize_dynamic_cc_match(
                lcia_method=lcia_method,
                match=rule.dynamic_cc_match,
            ),
        )
    return normalized


_LCIA_METHOD_RULES = _normalize_lcia_method_rules(_LCIA_METHOD_RULES)


def load_bundled_static_cc_rows(
    *,
    lcia_method: str,
) -> tuple[Path, StaticCCSchemaKind, tuple[NormalizedStaticCCRow, ...]]:
    """Load and normalize one bundled static carrying capacity CSV."""
    from .availability import require_static_cc_csv_path

    cc_csv_path = require_static_cc_csv_path(lcia_method=lcia_method)
    cc_df = pd.read_csv(cc_csv_path)
    schema_kind, rows = standardize_static_cc_rows(frame=cc_df, path=cc_csv_path)
    return cc_csv_path, schema_kind, rows


def bundled_cc_expected_impact_units(*, lcia_method: str) -> tuple[Path, list[tuple[str, str]]]:
    """Return exact bundled ``(impact, impact_unit)`` pairs for one LCIA method."""
    cc_csv_path, _schema_kind, rows = load_bundled_static_cc_rows(lcia_method=lcia_method)
    expected = sorted({(row.impact, row.impact_unit) for row in rows})
    return cc_csv_path, expected


def bundled_cc_expected_impacts(*, lcia_method: str) -> tuple[Path, list[str]]:
    """Return exact bundled impact codes for one LCIA method."""
    cc_csv_path, _schema_kind, rows = load_bundled_static_cc_rows(lcia_method=lcia_method)
    expected = sorted({row.impact for row in rows})
    return cc_csv_path, expected


def bundled_cc_impact_unit(*, lcia_method: str, impact: str) -> tuple[Path, str]:
    """Return the bundled static carrying capacity unit for one impact."""
    cc_csv_path, _schema_kind, rows = load_bundled_static_cc_rows(lcia_method=lcia_method)
    target_impact = str(impact).strip()
    matches = [row.impact_unit for row in rows if row.impact == target_impact]
    if not matches:
        available = sorted({row.impact for row in rows})
        raise ValueError(
            "Bundled static carrying capacity CSV does not define the required impact "
            f"'{target_impact}' for lcia_method='{lcia_method}'. "
            f"CSV: '{cc_csv_path}'. Available impacts: {available}."
        )
    return cc_csv_path, matches[0]


def dynamic_cc_match(*, lcia_method: str) -> dict[str, str] | None:
    """Return the explicit dynamic climate binding for one LCIA method."""
    rule = _LCIA_METHOD_RULES.get(normalize_lcia_method_name(lcia_method), LCIAMethodRule())
    if rule.dynamic_cc_match is None:
        return None
    return dict(rule.dynamic_cc_match)


def dynamic_cc_compatible_methods(*, method_specs: Sequence[str]) -> list[str]:
    """Return the ordered subset of methods that have an explicit dynamic climate binding."""
    compatible: list[str] = []
    for method_spec in method_specs:
        cleaned = normalize_lcia_method_name(method_spec)
        if dynamic_cc_match(lcia_method=cleaned) is not None:
            compatible.append(cleaned)
    return compatible
