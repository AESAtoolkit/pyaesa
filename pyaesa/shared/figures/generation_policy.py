"""Figure generation policy helpers shared by figure contracts."""


def resolve_polar_years(
    studied_years: list[int],
    user_override: list[int] | None = None,
    *,
    argument_name: str = "figure_options.polar_years",
) -> list[int]:
    """Return ASR polar checkpoint years.

    Args:
        studied_years: Full list of studied years.
        user_override: Explicit user supplied polar years (takes precedence).

    Returns:
        Sorted list of years for which polar figures are materialized.
    """
    studied = sorted(set(studied_years))
    if user_override is not None:
        requested = sorted(set(user_override))
        unsupported = sorted(set(requested) - set(studied))
        if unsupported:
            raise ValueError(
                f"{argument_name} must be selected from the studied years. "
                f"Unsupported year(s): {unsupported}. Studied years: {studied}."
            )
        return requested

    if len(studied) <= 1:
        return list(studied)

    return [studied[0], studied[-1]]
