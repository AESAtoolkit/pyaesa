"""Functional unit mapping for IO-LCA matrix routing and selector contracts."""

from dataclasses import dataclass
from typing import Literal

from pyaesa.asocc.data.load_mrio import _LCIA_L1_SCHEMA, _LCIA_L2_SCHEMA
from pyaesa.asocc.methods.registry.registry import METHOD_SPECS, normalize_fu_code
from pyaesa.asocc.methods.registry.model.input_requirements import (
    lcia_enacting_metric_l1_metrics,
    lcia_enacting_metric_l2_metrics,
)


@dataclass(frozen=True)
class IOLCAFUSpec:
    """IO-LCA routing metadata for one supported FU."""

    fu_code: str
    level: str
    family: str
    lcia_matrix_key: str
    selector_axes: tuple[str, ...]
    upstream_driver: Literal["y_fd", "x_to_rc"]
    upstream_supported: bool
    fy_relevant: bool


# Keep L1 support aligned with deterministic_asocc public FU contract.
_SUPPORTED_L1_FUS: tuple[str, ...] = ("L1.a", "L1.b")
_SUPPORTED_L2_FUS: tuple[str, ...] = tuple(
    sorted(
        {
            str(spec.fu_code)
            for spec in METHOD_SPECS
            if str(spec.level) == "L2" and spec.fu_code is not None
        }
    )
)
_SUPPORTED_FUS: tuple[str, ...] = _SUPPORTED_L1_FUS + _SUPPORTED_L2_FUS


def _lcia_kind_for_fu(*, fu_code: str) -> str:
    """Return canonical LCIA boundary kind for one FU code."""
    if fu_code == "L1.a":
        return "CBA_FD"
    if fu_code == "L1.b":
        return "PBA"
    if fu_code.startswith("L2.") and fu_code.endswith(".a"):
        return "CBA_FD"
    if fu_code.startswith("L2.") and fu_code.endswith(".b"):
        return "CBA_TD"
    return "PBA"


def _family_from_kind(*, lcia_kind: str) -> str:
    """Map canonical LCIA kind to IO-LCA family tag."""
    if lcia_kind == "CBA_FD":
        return "fd"
    if lcia_kind == "CBA_TD":
        return "td"
    return "pba"


def _lcia_matrix_key_for_fu(*, fu_code: str, lcia_kind: str) -> str:
    """Resolve canonical LCIA enacting metric matrix key for one FU."""
    if fu_code.startswith("L1."):
        keys = lcia_enacting_metric_l1_metrics(lcia_kinds={lcia_kind})
    else:
        keys = lcia_enacting_metric_l2_metrics(
            lcia_kinds={lcia_kind},
            fu_code=fu_code,
            l1_weighting=False,
        )
    return str(next(iter(keys)))


def _selector_axes_for_lcia_key(*, matrix_key: str) -> tuple[str, ...]:
    """Resolve selector axes from shared processed MRIO LCIA schema."""
    schema = _LCIA_L1_SCHEMA | _LCIA_L2_SCHEMA
    index_names, column_names = schema[matrix_key]
    selectors: list[str] = [name for name in index_names if name != "impact"]
    selectors.extend(name for name in (column_names or []) if name != "impact")
    return tuple(selectors)


def _build_fu_spec(*, fu_code: str) -> IOLCAFUSpec:
    """Build one IO-LCA FU spec from centralized allocation/process contracts."""
    lcia_kind = _lcia_kind_for_fu(fu_code=fu_code)
    family = _family_from_kind(lcia_kind=lcia_kind)
    matrix_key = _lcia_matrix_key_for_fu(fu_code=fu_code, lcia_kind=lcia_kind)
    selector_axes = _selector_axes_for_lcia_key(matrix_key=matrix_key)
    level = "L1" if fu_code.startswith("L1.") else "L2"
    return IOLCAFUSpec(
        fu_code=fu_code,
        level=level,
        family=family,
        lcia_matrix_key=matrix_key,
        selector_axes=selector_axes,
        upstream_driver=("y_fd" if family == "fd" else "x_to_rc"),
        upstream_supported=not (level == "L2" and family == "pba"),
        fy_relevant=(level == "L1"),
    )


_FU_SPECS: dict[str, IOLCAFUSpec] = {
    fu_code: _build_fu_spec(fu_code=fu_code) for fu_code in _SUPPORTED_FUS
}


def resolve_fu_spec(*, fu_code: str) -> IOLCAFUSpec:
    """Return FU mapping for IO-LCA, or fail for unsupported FUs.

    Args:
        fu_code: User provided functional unit code.

    Returns:
        Immutable FU specification.

    Raises:
        ValueError: If the FU code is invalid or unsupported by IO-LCA.
    """
    fu_norm = normalize_fu_code(fu_code)
    spec = _FU_SPECS.get(fu_norm)
    if spec is None:
        supported = sorted(_FU_SPECS.keys())
        raise ValueError(
            f"IO-LCA does not support this fu_code. Got '{fu_norm}'. Supported values: {supported}."
        )
    return spec
