"""Canonical aSoCC uncertainty source names."""

from pyaesa.shared.lcia.uncertainty_source import LCIA_SOURCE

INTER_METHOD_SOURCE = "inter_method_uncertainty"
INTER_MRIO_SOURCE = "inter_mrio_uncertainty"
PROJECTION_SOURCE = "projection_uncertainty"
REFERENCE_YEAR_SOURCE = "reference_year_uncertainty"

ASOCC_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    INTER_METHOD_SOURCE,
    INTER_MRIO_SOURCE,
    LCIA_SOURCE,
    PROJECTION_SOURCE,
    REFERENCE_YEAR_SOURCE,
)

DEFAULT_ASOCC_UNCERTAINTY_SOURCES: tuple[str, ...] = (
    INTER_METHOD_SOURCE,
    PROJECTION_SOURCE,
    REFERENCE_YEAR_SOURCE,
)
