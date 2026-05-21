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
    pieces = [piece for piece in normalized.split("__") if piece]
    normalized_tokens = normalize_ssp_tokens(
        list(scenario_tokens) if scenario_tokens is not None else None
    )
    if len(pieces) <= 1:
        return normalized, None
    explicit_tokens = {token.lower() for token in normalized_tokens}
    scenario_positions = [
        index
        for index, piece in enumerate(pieces)
        if (
            piece.lower() in explicit_tokens
            if explicit_tokens
            else bool(_CANONICAL_STEM_SSP_RE.fullmatch(piece.strip()))
        )
    ]
    if not scenario_positions:
        return normalized, None
    if len(scenario_positions) > 1:
        raise ValueError(
            "Deterministic companion stems must contain at most one SSP filename token. "
            f"Got stem='{normalized}'."
        )
    scenario_position = scenario_positions[0]
    scenario_token = partition_token_to_ssp_token(
        pieces[scenario_position],
        context=f"Deterministic companion stem '{normalized}'",
    )
    base_tokens = [piece for index, piece in enumerate(pieces) if index != scenario_position]
    return "__".join(base_tokens), scenario_token
