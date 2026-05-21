"""Variable group ownership for AR6 process figures."""

from collections.abc import Iterable

from pyaesa.download.ar6.utils.config import (
    RAW_SEQUESTRATION_COMPONENTS,
    SEQUESTRATION_SUBTOTAL,
    SEQUESTRATION_TOTAL,
)

SEQUESTRATION_FIGURE_VARIABLES = (SEQUESTRATION_TOTAL, SEQUESTRATION_SUBTOTAL)
SEQUESTRATION_CONTRIBUTION_VARIABLES = (
    SEQUESTRATION_TOTAL,
    SEQUESTRATION_SUBTOTAL,
    *tuple(reversed(RAW_SEQUESTRATION_COMPONENTS)),
)


def ordered_available_variables(
    *,
    requested_variables: Iterable[str],
    available_variables: Iterable[str],
) -> list[str]:
    """Return requested variables that are present, preserving request order."""
    available_set = set(available_variables)
    return [variable for variable in requested_variables if variable in available_set]


def missing_requested_variables(
    *,
    requested_variables: Iterable[str],
    available_variables: Iterable[str],
) -> list[str]:
    """Return requested variables absent from the available variable set."""
    available_set = set(available_variables)
    return [variable for variable in requested_variables if variable not in available_set]


def emission_variable_groups(variables: Iterable[str]) -> tuple[tuple[str, list[str]], ...]:
    """Group processed emissions variables using the v2 reference figure families."""
    co2_variables = [variable for variable in variables if "CO2" in variable]
    ghg_variables = [variable for variable in variables if "Kyoto" in variable]
    return tuple(
        (label, group) for label, group in (("CO2", co2_variables), ("GHG", ghg_variables)) if group
    )


def emission_mode_variable_groups(variables: Iterable[str]) -> tuple[tuple[str, list[str]], ...]:
    """Group processed emissions variables by net, gross, and gross alternative modes."""
    variables_list = list(variables)
    groups = (
        ("net", [variable for variable in variables_list if variable.startswith("Emissions(net)")]),
        (
            "gross",
            [variable for variable in variables_list if variable.startswith("Emissions(gross)")],
        ),
        (
            "gross_alt",
            [
                variable
                for variable in variables_list
                if variable.startswith("Emissions(gross_alt)")
            ],
        ),
    )
    return tuple((label, group) for label, group in groups if group)
