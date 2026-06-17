"""Canonical deterministic companion stem parsing."""

from collections.abc import Sequence
from dataclasses import dataclass
import re

from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens, partition_token_to_ssp_token

_CANONICAL_STEM_SSP_RE = re.compile(r"^ssp[0-9]+$", re.IGNORECASE)


@dataclass(frozen=True)
class DeterministicCompanionStem:
    """Parsed deterministic companion stem identity."""

    normalized_stem: str
    base_stem: str
    ssp_scenario: str | None


def parse_deterministic_companion_stem(
    stem: str,
    *,
    scenario_tokens: Sequence[str] | None = None,
) -> DeterministicCompanionStem:
    """Return canonical companion stem identity for historical and projected files."""
    normalized = str(stem).strip()
    base_stem, ssp_scenario = _split_companion_scenario_suffix(
        normalized,
        scenario_tokens=scenario_tokens,
    )
    return DeterministicCompanionStem(
        normalized_stem=normalized,
        base_stem=base_stem,
        ssp_scenario=ssp_scenario,
    )


def _split_companion_scenario_suffix(
    stem: str,
    *,
    scenario_tokens: Sequence[str] | None,
) -> tuple[str, str | None]:
    normalized = str(stem).strip()
    normalized_tokens = normalize_ssp_tokens(
        list(scenario_tokens) if scenario_tokens is not None else None
    )
    explicit_tokens = {token.lower() for token in normalized_tokens}
    base_stem, separator, suffix = normalized.rpartition("_")
    if not separator or not base_stem:
        return normalized, None
    suffix_is_scenario = (
        suffix.lower() in explicit_tokens
        if explicit_tokens
        else bool(_CANONICAL_STEM_SSP_RE.fullmatch(suffix.strip()))
    )
    if not suffix_is_scenario:
        return normalized, None
    _base_prefix, base_separator, base_suffix = base_stem.rpartition("_")
    base_suffix_is_scenario = (
        base_suffix.lower() in explicit_tokens
        if explicit_tokens
        else bool(_CANONICAL_STEM_SSP_RE.fullmatch(base_suffix.strip()))
    )
    if base_separator and base_suffix_is_scenario:
        raise ValueError(
            "Deterministic companion stems must contain at most one SSP filename token. "
            f"Got stem='{normalized}'."
        )
    scenario_token = partition_token_to_ssp_token(
        suffix,
        context=f"Deterministic companion stem '{normalized}'",
    )
    return base_stem, scenario_token
