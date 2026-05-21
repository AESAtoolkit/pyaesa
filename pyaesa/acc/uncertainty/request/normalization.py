"""Request source routing for aCC uncertainty."""

from typing import Any

from pyaesa.asocc.uncertainty.sources.names import (
    ASOCC_UNCERTAINTY_SOURCES,
    DEFAULT_ASOCC_UNCERTAINTY_SOURCES,
)
from pyaesa.ar6_cc.uncertainty.request.normalization import (
    AR6_DYNAMIC_CC_SOURCE,
)
from pyaesa.shared.uncertainty_assessment.request.sources import build_source_activation_plan

ACC_UNCERTAINTY_SOURCES = (*ASOCC_UNCERTAINTY_SOURCES, AR6_DYNAMIC_CC_SOURCE)
DEFAULT_ACC_UNCERTAINTY_SOURCES = (
    *DEFAULT_ASOCC_UNCERTAINTY_SOURCES,
    AR6_DYNAMIC_CC_SOURCE,
)


def normalize_acc_uncertainty_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return one validated aCC uncertainty configuration mapping."""
    payload = dict(config)
    mc_parameters = payload.pop("mc_parameters", None)
    sources = build_source_activation_plan(
        uncertainty_config=payload,
        allowed_sources=ACC_UNCERTAINTY_SOURCES,
        default_sources=DEFAULT_ACC_UNCERTAINTY_SOURCES,
    )
    out = {source.name: source.parameters for source in sources.sources}
    if mc_parameters is not None:
        out["mc_parameters"] = mc_parameters
    return out


def asocc_uncertainty_config_for_acc(config: dict[str, Any]) -> dict[str, Any]:
    """Return the aSoCC source subset from a shared aCC uncertainty request."""
    out = {
        key: value
        for key, value in config.items()
        if key in ASOCC_UNCERTAINTY_SOURCES or key == "mc_parameters"
    }
    for source in DEFAULT_ASOCC_UNCERTAINTY_SOURCES:
        out.setdefault(source, {"active": False})
    return out


def dynamic_cc_source_parameters(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return normalized dynamic AR6 CC source parameters for aCC."""
    if value is None:
        return None
    return dict(value)
