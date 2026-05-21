"""Scientific text formatting for visible figure labels."""

import re

_CHEMICAL_FORMULAS = {
    "CO2": r"CO$_2$",
    "CH4": r"CH$_4$",
    "N2O": r"N$_2$O",
    "NH3": r"NH$_3$",
    "NO2": r"NO$_2$",
    "NO3": r"NO$_3$",
    "O3": r"O$_3$",
    "PO4": r"PO$_4$",
    "SO2": r"SO$_2$",
}
_CHEMICAL_REPLACEMENTS = tuple(sorted(_CHEMICAL_FORMULAS.items(), key=lambda item: -len(item[0])))
_UNIT_TOKENS = (
    "year",
    "yr",
    "km",
    "Mm",
    "cm",
    "mm",
    "m",
    "kg",
    "g",
    "Mt",
    "Gt",
    "Tg",
    "t",
    "mol",
    "MJ",
    "J",
    "kBq",
    "Bq",
    "W",
)
_POSITIVE_UNIT_EXPONENT_PATTERN = re.compile(r"\b(?P<unit>km|Mm|cm|mm|m)\^?(?P<exp>[23])\b")
_SIGNED_UNIT_EXPONENT_PATTERN = re.compile(
    rf"\b(?P<unit>{'|'.join(_UNIT_TOKENS)})\^?(?P<exp>[+-]\d+)\b"
)
_SLASH_UNIT_PATTERN = re.compile(rf"\s*/\s*(?P<unit>{'|'.join(_UNIT_TOKENS)})\b")
_POSITIVE_ION_PATTERN = re.compile(r"\bH\+(?=$|\s|[,;:)\]])")


def format_scientific_figure_text(value: object) -> str:
    """Return one label using compact scientific math text where applicable."""
    text = str(value)
    text = text.replace("fNT", r"$f^{\mathrm{NT}}$")
    for formula, formatted in _CHEMICAL_REPLACEMENTS:
        text = text.replace(formula, formatted)
    text = _POSITIVE_ION_PATTERN.sub(r"H$^+$", text)
    text = _SLASH_UNIT_PATTERN.sub(
        lambda match: rf" {match.group('unit')}$^{{-1}}$",
        text,
    )
    text = _POSITIVE_UNIT_EXPONENT_PATTERN.sub(
        lambda match: rf"{match.group('unit')}$^{match.group('exp')}$",
        text,
    )
    return _SIGNED_UNIT_EXPONENT_PATTERN.sub(
        lambda match: rf"{match.group('unit')}$^{{{match.group('exp')}}}$",
        text,
    )
