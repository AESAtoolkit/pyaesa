"""ASR uncertainty source routing across independent component lanes."""

from dataclasses import dataclass
from typing import Any

from pyaesa.ar6_cc.uncertainty.request.normalization import AR6_DYNAMIC_CC_SOURCE
from pyaesa.asocc.uncertainty.sources.names import (
    ASOCC_UNCERTAINTY_SOURCES,
    DEFAULT_ASOCC_UNCERTAINTY_SOURCES,
)
from pyaesa.shared.lcia.uncertainty_source import LCIA_SOURCE
from pyaesa.shared.uncertainty_assessment.request.sources import build_source_activation_plan

ASR_ASOCC_UNCERTAINTY_SOURCES_KEY = "asocc_uncertainty_sources"
ASR_AR6_CC_UNCERTAINTY_SOURCES_KEY = "ar6_cc_uncertainty_sources"
ASR_IO_LCA_UNCERTAINTY_SOURCES_KEY = "io_lca_uncertainty_sources"
ASR_DYNAMIC_CC_UNCERTAINTY_SOURCES = (AR6_DYNAMIC_CC_SOURCE,)
DEFAULT_ASR_DYNAMIC_CC_UNCERTAINTY_SOURCES = (AR6_DYNAMIC_CC_SOURCE,)
ASR_IO_LCA_UNCERTAINTY_SOURCES = (LCIA_SOURCE,)
_ASR_COMPONENT_KEYS = {
    ASR_ASOCC_UNCERTAINTY_SOURCES_KEY,
    ASR_AR6_CC_UNCERTAINTY_SOURCES_KEY,
    ASR_IO_LCA_UNCERTAINTY_SOURCES_KEY,
}


@dataclass(frozen=True)
class ASRSourceConfig:
    """Component source configuration for one ASR uncertainty request."""

    acc_config: dict[str, Any]
    lca_config: dict[str, Any]


def split_asr_uncertainty_config(config: dict[str, Any]) -> ASRSourceConfig:
    """Route ASR uncertainty source blocks to aCC and LCA components.

    ASR owns three independent component lanes:

    - ``asocc_uncertainty_sources`` controls aSoCC allocation share uncertainty.
    - ``ar6_cc_uncertainty_sources`` controls dynamic AR6 CC uncertainty.
    - ``io_lca_uncertainty_sources`` controls IO-LCA uncertainty.
    """
    payload = dict(config)
    mc_parameters = payload.pop("mc_parameters", None)
    asocc_sources = payload.pop(ASR_ASOCC_UNCERTAINTY_SOURCES_KEY, None)
    dynamic_cc_sources = payload.pop(ASR_AR6_CC_UNCERTAINTY_SOURCES_KEY, None)
    io_lca_sources = payload.pop(ASR_IO_LCA_UNCERTAINTY_SOURCES_KEY, None)
    unknown = sorted(set(payload) - _ASR_COMPONENT_KEYS)
    if unknown:
        allowed = sorted({*_ASR_COMPONENT_KEYS, "mc_parameters"})
        raise ValueError(
            f"Unsupported ASR uncertainty_config keys: {unknown}. Accepted keys: {allowed}."
        )

    asocc_config = _component_source_config(
        component=asocc_sources,
        field=ASR_ASOCC_UNCERTAINTY_SOURCES_KEY,
        allowed_sources=ASOCC_UNCERTAINTY_SOURCES,
        default_sources=DEFAULT_ASOCC_UNCERTAINTY_SOURCES,
    )
    dynamic_cc_config = _component_source_config(
        component=dynamic_cc_sources,
        field=ASR_AR6_CC_UNCERTAINTY_SOURCES_KEY,
        allowed_sources=ASR_DYNAMIC_CC_UNCERTAINTY_SOURCES,
        default_sources=DEFAULT_ASR_DYNAMIC_CC_UNCERTAINTY_SOURCES,
    )
    return ASRSourceConfig(
        acc_config=_with_mc_parameters(
            {**asocc_config, **dynamic_cc_config},
            mc_parameters=mc_parameters,
        ),
        lca_config=_with_mc_parameters(
            _component_source_config(
                component=io_lca_sources,
                field=ASR_IO_LCA_UNCERTAINTY_SOURCES_KEY,
                allowed_sources=ASR_IO_LCA_UNCERTAINTY_SOURCES,
                default_sources=(),
            ),
            mc_parameters=mc_parameters,
        ),
    )


def _component_source_config(
    *,
    component: object,
    field: str,
    allowed_sources: tuple[str, ...],
    default_sources: tuple[str, ...],
) -> dict[str, Any]:
    component_sources = _component_mapping(component=component, field=field)
    sources = build_source_activation_plan(
        uncertainty_config=component_sources,
        allowed_sources=allowed_sources,
        default_sources=default_sources,
    )
    out = {source.name: source.parameters for source in sources.sources}
    for source in default_sources:
        if source in component_sources and source not in out:
            out[source] = {"active": False}
    return out


def _with_mc_parameters(config: dict[str, Any], *, mc_parameters: object) -> dict[str, Any]:
    out = dict(config)
    if mc_parameters is not None:
        out["mc_parameters"] = mc_parameters
    return out


def _component_mapping(*, component: object, field: str) -> dict[str, Any]:
    if component is None:
        return {}
    if not isinstance(component, dict):
        raise ValueError(f"uncertainty_config.{field} must be a dictionary when provided.")
    return dict(component)
