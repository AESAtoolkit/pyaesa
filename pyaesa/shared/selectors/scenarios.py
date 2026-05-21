"""Shared SSP selector normalization helpers."""

from collections.abc import Sequence

DEFAULT_SSP_SCENARIOS: list[str] = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]


def normalize_ssp_token(value: object, *, context: str = "SSP selector") -> str:
    """Return one canonical ``SSP<n>`` token."""
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context} contains an empty SSP value.")
    upper = text.upper()
    suffix = upper[3:] if upper.startswith("SSP") else text
    suffix = str(suffix).strip()
    if not suffix.isdigit():
        raise ValueError(f"{context} must use canonical SSP tokens like 'SSP2'.")
    return f"SSP{int(suffix)}"


def normalize_ssp_tokens(
    values: Sequence[object] | str | None,
    *,
    context: str = "SSP selector",
) -> list[str]:
    """Return canonical ``SSP<n>`` tokens from one SSP selector sequence."""
    if values is None:
        return []
    sequence: Sequence[object] = [values] if isinstance(values, str) else values
    tokens = [
        normalize_ssp_token(value, context=context) for value in sequence if str(value).strip()
    ]
    return sorted(set(tokens))


def normalize_optional_ssp_selector(
    values: object,
    *,
    argument_name: str,
) -> list[str] | None:
    """Validate one optional public SSP selector list and return canonical tokens."""
    if values is None:
        return None
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raise ValueError(f"'{argument_name}' must be None, a string, or a list.")
    tokens = normalize_ssp_tokens(raw_values, context=f"'{argument_name}'")
    if not tokens:
        raise ValueError(f"'{argument_name}' must be a non-empty string or list.")
    return tokens


def ssp_partition_token(value: object, *, context: str = "SSP selector") -> str:
    """Return the internal lowercase partition token for one canonical SSP value."""
    return normalize_ssp_token(value, context=context).lower()


def partition_token_to_ssp_token(token: object, *, context: str) -> str:
    """Return one canonical ``SSP<n>`` token from an internal partition token."""
    text = str(token).strip()
    if not text:
        raise ValueError(f"{context} is missing the SSP partition token.")
    lower = text.lower()
    if not lower.startswith("ssp"):
        raise ValueError(f"{context} uses invalid SSP partition token '{token}'.")
    return normalize_ssp_token(lower, context=context)
