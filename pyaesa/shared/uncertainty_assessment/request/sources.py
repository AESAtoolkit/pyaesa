"""Family-neutral active uncertainty source planning."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActiveSource:
    """One active uncertainty source and its normalized public configuration."""

    name: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class SourceActivationPlan:
    """Ordered active uncertainty sources for one public request."""

    sources: tuple[ActiveSource, ...]

    @property
    def names(self) -> tuple[str, ...]:
        """Return active source names in execution order."""
        return tuple(source.name for source in self.sources)

    def parameters_for(self, source_name: str) -> dict[str, Any]:
        """Return normalized parameters for one active source."""
        for source in self.sources:
            if source.name == source_name:
                return dict(source.parameters)
        return {}

    def is_active(self, source_name: str) -> bool:
        """Return whether one source is active."""
        return source_name in self.names


def build_source_activation_plan(
    *,
    uncertainty_config: object,
    allowed_sources: tuple[str, ...],
    default_sources: tuple[str, ...] = (),
) -> SourceActivationPlan:
    """Build an ordered active-source plan from public source configuration.

    Args:
        uncertainty_config: Public source configuration. A source mapping is
            active when its ``active`` field is ``True`` or omitted. A mapping
            with ``active=False`` disables the source.
        allowed_sources: Canonical source order for the public family.
        default_sources: Sources active by default unless explicitly disabled
            by `uncertainty_config`.

    Returns:
        Ordered active source plan.
    """
    allowed = tuple(allowed_sources)
    allowed_set = set(allowed)
    default_set = set(default_sources)
    if uncertainty_config is None:
        source_config: dict[object, object] = {}
    elif isinstance(uncertainty_config, dict):
        source_config = dict(uncertainty_config)
    else:
        raise ValueError("uncertainty_config must be a dictionary when provided.")
    unknown = sorted(str(key) for key in source_config if str(key) not in allowed_set)
    if unknown:
        raise ValueError(f"Unsupported uncertainty source names: {unknown}.")
    sources: list[ActiveSource] = []
    for source in allowed:
        raw = source_config.get(source)
        if raw is None:
            if source in default_set:
                sources.append(ActiveSource(name=source, parameters={}))
            continue
        if isinstance(raw, dict):
            parameters = dict(raw)
            active = parameters.pop("active", True)
            if not isinstance(active, bool):
                raise ValueError(f"Uncertainty source '{source}'.active must be a boolean.")
            if not active:
                continue
            if "alternate_source" in parameters:
                alternate_source = parameters.pop("alternate_source")
                if alternate_source is not None:
                    parameters["source"] = alternate_source
            sources.append(ActiveSource(name=source, parameters=parameters))
            continue
        if isinstance(raw, bool):
            raise ValueError(
                f"Uncertainty source '{source}' must be a dictionary with an active boolean."
            )
        raise ValueError(f"Uncertainty source '{source}' must be configured with a dictionary.")
    return SourceActivationPlan(sources=tuple(sources))
