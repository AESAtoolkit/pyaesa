"""Shared scientific notation tick formatters for package figures."""


def scientific_tick_text(value: float | int | str, *, suffix: str = "") -> str:
    """Return compact scientific notation text for one axis tick."""
    numeric = float(value)
    if numeric == 0.0:
        return f"0{suffix}"
    mantissa, exponent = f"{numeric:.3e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent):+03d}{suffix}"


def scientific_tick_formatter(value, _pos) -> str:
    """Format axis ticks in explicit scientific notation."""
    return scientific_tick_text(value)


def scientific_percent_tick_formatter(value, _pos) -> str:
    """Format percentage axis ticks in explicit scientific notation."""
    return scientific_tick_text(value, suffix="%")
