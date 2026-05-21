"""Shared contracts for dynamic AR6 carrying capacity selectors."""

import pandas as pd

from pyaesa.download.ar6.utils.config import (
    GROSS_ALT_CO2_WITH_AFOLU,
    GROSS_ALT_CO2_WO_AFOLU,
    GROSS_ALT_KYOTO_WITH_AFOLU,
    GROSS_ALT_KYOTO_WO_AFOLU,
    GROSS_CO2_WITH_AFOLU,
    GROSS_CO2_WO_AFOLU,
    GROSS_KYOTO_WITH_AFOLU,
    GROSS_KYOTO_WO_AFOLU,
    NET_CO2_WITH_AFOLU,
    NET_CO2_WO_AFOLU,
    NET_KYOTO_WITH_AFOLU,
    NET_KYOTO_WO_AFOLU,
    SEQUESTRATION_SUBTOTAL,
    SEQUESTRATION_TOTAL,
)

CC_FLOW_NET = "net_emissions"
CC_FLOW_POSITIVE = "positive_emissions"
CC_FLOW_NEGATIVE = "negative_sequestration"
DOWNSTREAM_CC_FLOWS = (CC_FLOW_NET, CC_FLOW_POSITIVE)

_VALID_EMISSION_BASES = {"kyoto_gases", "co2"}
_VALID_EMISSIONS_MODES = {"net", "gross", "gross_alt"}


def normalize_emission_type(emission_type: str) -> str:
    """Validate and normalize the public dynamic AR6 emission type selector."""
    normalized = str(emission_type).strip().lower()
    if normalized not in _VALID_EMISSION_BASES:
        raise ValueError(
            f"emission_type must be 'kyoto_gases' or 'co2'. Received '{emission_type}'."
        )
    return normalized


def emission_type_tag(*, emission_type: str) -> str:
    """Return the filesystem tag for one dynamic AR6 emission type."""
    return normalize_emission_type(emission_type)


def normalize_emissions_mode(emissions_mode: str) -> str:
    """Validate and normalize the public dynamic AR6 emissions mode selector."""
    normalized = str(emissions_mode).strip().lower()
    if normalized not in _VALID_EMISSIONS_MODES:
        raise ValueError(
            f"emissions_mode must be 'net', 'gross', or 'gross_alt'. Received '{emissions_mode}'."
        )
    return normalized


def emissions_mode_tag(*, emissions_mode: str) -> str:
    """Return the filesystem tag for one dynamic AR6 emissions mode."""
    return normalize_emissions_mode(emissions_mode)


def cc_variable(
    *,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str = "gross_alt",
) -> str:
    """Return the processed AR6 variable for the selected dynamic AR6 CC scope."""
    emission_type_norm = normalize_emission_type(emission_type)
    mode = normalize_emissions_mode(emissions_mode)
    variable_map = {
        ("kyoto_gases", False, "net"): NET_KYOTO_WO_AFOLU,
        ("kyoto_gases", True, "net"): NET_KYOTO_WITH_AFOLU,
        ("co2", False, "net"): NET_CO2_WO_AFOLU,
        ("co2", True, "net"): NET_CO2_WITH_AFOLU,
        ("kyoto_gases", False, "gross"): GROSS_KYOTO_WO_AFOLU,
        ("kyoto_gases", True, "gross"): GROSS_KYOTO_WITH_AFOLU,
        ("co2", False, "gross"): GROSS_CO2_WO_AFOLU,
        ("co2", True, "gross"): GROSS_CO2_WITH_AFOLU,
        ("kyoto_gases", False, "gross_alt"): GROSS_ALT_KYOTO_WO_AFOLU,
        ("kyoto_gases", True, "gross_alt"): GROSS_ALT_KYOTO_WITH_AFOLU,
        ("co2", False, "gross_alt"): GROSS_ALT_CO2_WO_AFOLU,
        ("co2", True, "gross_alt"): GROSS_ALT_CO2_WITH_AFOLU,
    }
    return variable_map[(emission_type_norm, bool(include_afolu), mode)]


def cc_sequestration_variable(*, emissions_mode: str) -> str | None:
    """Return the sequestration companion variable for a gross CC mode."""
    mode = normalize_emissions_mode(emissions_mode)
    if mode == "net":
        return None
    if mode == "gross":
        return SEQUESTRATION_TOTAL
    return SEQUESTRATION_SUBTOTAL


def cc_positive_flow(*, emissions_mode: str) -> str:
    """Return the public CC flow label for the positive denominator row."""
    if normalize_emissions_mode(emissions_mode) == "net":
        return CC_FLOW_NET
    return CC_FLOW_POSITIVE


def cc_denominator_variable(
    *,
    emission_type: str,
    include_afolu: bool,
    emissions_mode: str,
) -> str:
    """Return the AR6 CC variable used by aCC and ASR denominators."""
    return cc_variable(
        emission_type=emission_type,
        include_afolu=include_afolu,
        emissions_mode=emissions_mode,
    )


def downstream_cc_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return AR6 CC rows that can act as aCC or ASR denominator values."""
    return frame.loc[frame["cc_flow"].isin(DOWNSTREAM_CC_FLOWS)].copy()
