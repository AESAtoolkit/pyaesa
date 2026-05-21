"""Canonical aCC uncertainty source keys."""

from pyaesa.asocc.uncertainty.sources.names import (
    INTER_METHOD_SOURCE,
    PROJECTION_SOURCE,
    REFERENCE_YEAR_SOURCE,
)
from pyaesa.ar6_cc.uncertainty.request.normalization import AR6_DYNAMIC_CC_SOURCE

ASOCC_SOURCE_PREFIX = "asocc::"
AR6_CC_SOURCE_PREFIX = "ar6_cc::"

ASOCC_INTER_METHOD_SOURCE = f"{ASOCC_SOURCE_PREFIX}{INTER_METHOD_SOURCE}"
ASOCC_PROJECTION_SOURCE = f"{ASOCC_SOURCE_PREFIX}{PROJECTION_SOURCE}"
ASOCC_REFERENCE_YEAR_SOURCE = f"{ASOCC_SOURCE_PREFIX}{REFERENCE_YEAR_SOURCE}"
AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE = f"{AR6_CC_SOURCE_PREFIX}{AR6_DYNAMIC_CC_SOURCE}"


def asocc_source_name(source_name: str) -> str:
    """Return the aCC source key for one upstream aSoCC source."""
    return f"{ASOCC_SOURCE_PREFIX}{str(source_name).strip()}"


def ar6_cc_source_name(source_name: str) -> str:
    """Return the aCC source key for one upstream AR6 CC source."""
    return f"{AR6_CC_SOURCE_PREFIX}{str(source_name).strip()}"
