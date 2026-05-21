"""ASR public LCA route request normalization."""

from typing import Any

from pyaesa.external_inputs.lca.naming import normalize_external_lca_version_name

_EXTERNAL_LCA_ROUTE = "external_lca"
_IO_LCA_ROUTE = "io_lca"
_ALLOWED_LCA_KEYS = {_EXTERNAL_LCA_ROUTE, _IO_LCA_ROUTE}
_EXTERNAL_KEYS = {"active", "version_name"}
_IO_LCA_KEYS = {"active"}


def normalize_lca_args(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the public ASR LCA family block."""
    if not isinstance(raw, dict):
        raise ValueError("'lca_args' must be a dictionary describing the ASR numerator route.")
    unknown = sorted(set(raw) - _ALLOWED_LCA_KEYS)
    if unknown:
        raise ValueError(
            f"'lca_args' contains unknown key(s): {unknown}. "
            "Provide only 'external_lca' and 'io_lca' route blocks."
        )
    external = _normalize_external_lca_block(raw.get(_EXTERNAL_LCA_ROUTE), default_active=not raw)
    io_lca = _normalize_io_lca_block(raw.get(_IO_LCA_ROUTE))
    active_routes = [
        route
        for route, block in ((_EXTERNAL_LCA_ROUTE, external), (_IO_LCA_ROUTE, io_lca))
        if bool(block["active"])
    ]
    if len(active_routes) != 1:
        raise ValueError("Exactly one lca_args route block must be active.")
    if external["active"]:
        external["version_name"] = normalize_external_lca_version_name(
            external["version_name"],
            argument_name="lca_args.external_lca.version_name",
        )
    elif external["version_name"] is not None:
        raise ValueError(
            "lca_args.external_lca.version_name is only valid when "
            "lca_args.external_lca.active=True."
        )
    return {_EXTERNAL_LCA_ROUTE: external, _IO_LCA_ROUTE: io_lca}


def selected_lca_type(*, lca_args: dict[str, Any]) -> str:
    """Return the internal ASR numerator route selected by normalized LCA args."""
    if bool(lca_args[_IO_LCA_ROUTE]["active"]):
        return _IO_LCA_ROUTE
    return "external"


def selected_lca_version_name(*, lca_args: dict[str, Any]) -> str | None:
    """Return the external LCA version selected by normalized LCA args."""
    if selected_lca_type(lca_args=lca_args) == _IO_LCA_ROUTE:
        return None
    return str(lca_args[_EXTERNAL_LCA_ROUTE]["version_name"])


def _normalize_external_lca_block(value: object, *, default_active: bool) -> dict[str, Any]:
    if value is None:
        return {"active": default_active, "version_name": None}
    if not isinstance(value, dict):
        raise ValueError("lca_args.external_lca must be a dictionary.")
    block = {"active": True, "version_name": None, **value}
    unknown = sorted(set(block) - _EXTERNAL_KEYS)
    if unknown:
        raise ValueError(f"Unsupported lca_args.external_lca keys: {unknown}.")
    if not isinstance(block["active"], bool):
        raise ValueError("lca_args.external_lca.active must be a boolean.")
    return block


def _normalize_io_lca_block(value: object) -> dict[str, Any]:
    if value is None:
        return {"active": False}
    if not isinstance(value, dict):
        raise ValueError("lca_args.io_lca must be a dictionary.")
    block = {"active": True, **value}
    unknown = sorted(set(block) - _IO_LCA_KEYS)
    if unknown:
        raise ValueError(f"Unsupported lca_args.io_lca keys: {unknown}.")
    if not isinstance(block["active"], bool):
        raise ValueError("lca_args.io_lca.active must be a boolean.")
    return block
