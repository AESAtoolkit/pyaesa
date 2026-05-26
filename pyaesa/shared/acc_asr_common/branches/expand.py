"""Shared CC branch expansion for composite aCC and ASR entrypoints."""

from typing import Any

from pyaesa.process.ar6.utils.pipeline.study_period import resolve_study_period
from pyaesa.shared.lcia.contracts import dynamic_cc_compatible_methods


def has_dynamic_ar6_branch(*, branches: list[dict[str, Any]]) -> bool:
    """Return whether expanded CC branches include dynamic AR6 carrying capacity."""
    return any(branch["cc_type"] == "dynamic_ar6" for branch in branches)


def iter_cc_method_branches(
    *,
    lcia_methods: list[str],
    base_cc_args: dict[str, Any],
    years: int | list[int] | range,
) -> list[dict[str, Any]]:
    """Expand normalized CC config into one branch payload per family and method."""
    branches: list[dict[str, Any]] = []
    if "static" in base_cc_args:
        for lcia_method in lcia_methods:
            branches.append(
                {
                    "cc_source": lcia_method,
                    "cc_type": "static",
                    "static_cc_bounds": list(base_cc_args["static"]["bounds"]),
                }
            )
    if "dynamic_ar6" in base_cc_args:
        resolve_study_period(years)
        compatible = dynamic_cc_compatible_methods(method_specs=lcia_methods)
        if not compatible:
            raise ValueError(
                "base_cc_args.dynamic_ar6 requires at least one LCIA method with a "
                "static carrying capacity row for impact 'GWP_100'. "
                f"Requested lcia_method values: {lcia_methods}. "
                "Requested methods without that row use steady state carrying capacities only."
            )
        for lcia_method in compatible:
            branch = dict(base_cc_args["dynamic_ar6"])
            branch["cc_source"] = lcia_method
            branch["cc_type"] = "dynamic_ar6"
            branches.append(branch)
    return branches
