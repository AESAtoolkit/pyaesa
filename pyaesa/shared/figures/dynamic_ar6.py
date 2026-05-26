"""Shared dynamic AR6 figure identity contracts."""

from collections.abc import Iterable
import re
from typing import Any

from pyaesa.shared.tabular.scalars import sanitize_token

DYNAMIC_AR6_CC_TYPE = "dynamic_ar6"
MODEL_SCENARIO_PAIR_COUNT_COLUMN = "__model_scenario_pair_count"
MODEL_SCENARIO_SAMPLING_METHOD_COLUMN = "__model_scenario_sampling_method"
AR6_CATEGORY_SCOPE_COLUMN = "__ar6_category_scope"


def category_scope_label(categories: Iterable[str]) -> str:
    """Return a compact AR6 category scope label."""
    cleaned = sorted({str(value).strip() for value in categories if str(value).strip()})
    if not cleaned:
        return ""
    matches = [re.fullmatch(r"C(\d+)", value) for value in cleaned]
    if not all(matches):
        return ", ".join(cleaned)
    ordered_numbers = sorted(int(match.group(1)) for match in matches if match is not None)
    expected = list(range(ordered_numbers[0], ordered_numbers[-1] + 1))
    if ordered_numbers == expected and len(ordered_numbers) > 1:
        return f"C{ordered_numbers[0]}-C{ordered_numbers[-1]}"
    return ", ".join(f"C{number}" for number in ordered_numbers)


def model_scenario_pair_token(
    *,
    models: Iterable[str],
    scenarios: Iterable[str],
) -> str | None:
    """Return a filesystem safe token for one AR6 model-scenario pair."""
    model = _single_visible_value(models)
    scenario = _single_visible_value(scenarios)
    if model is None or scenario is None:
        return None
    return f"{sanitize_token(model)}_{sanitize_token(scenario)}"


def model_scenario_pair_label(
    *,
    models: Iterable[str],
    scenarios: Iterable[str],
) -> str | None:
    """Return a readable label for one AR6 model-scenario pair."""
    model = _single_visible_value(models)
    scenario = _single_visible_value(scenarios)
    if model is None or scenario is None:
        return None
    return f"{model} / {scenario}"


def model_scenario_sampling_method(frame: Any) -> str | None:
    """Return sampled pathway method metadata when figure rows carry it."""
    if MODEL_SCENARIO_SAMPLING_METHOD_COLUMN not in frame:
        return None
    values = frame[MODEL_SCENARIO_SAMPLING_METHOD_COLUMN].dropna()
    if values.empty:
        return None
    return str(values.iloc[0]).strip().lower()


def dynamic_ar6_detail_line(
    *,
    categories: Iterable[str],
    models: Iterable[str],
    scenarios: Iterable[str],
) -> str:
    """Return the second title line for one deterministic dynamic AR6 figure scope."""
    pair_label = model_scenario_pair_label(models=models, scenarios=scenarios)
    if pair_label is None:
        return ""
    category_scope = category_scope_label(categories)
    parts = []
    if category_scope:
        noun = "categories" if _is_multi_category_scope(category_scope) else "category"
        parts.append(f"AR6 {noun}: {category_scope}")
    parts.append(f"Model-scenario pair: {pair_label}")
    return " | ".join(parts)


def _single_visible_value(values: Iterable[str]) -> str | None:
    cleaned = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
    return cleaned[0] if len(cleaned) == 1 else None


def _is_multi_category_scope(scope: str) -> bool:
    text = str(scope).strip()
    return "-" in text or "," in text
