"""Query operations for the allocation method registry."""

from typing import Iterable, Optional, cast

from pyaesa.asocc.methods.registry.build.build import _L2_FU_L1_KIND_MAP
from pyaesa.asocc.methods.registry.model.family_checks import MethodRegistryFamilyChecksMixin
from pyaesa.asocc.methods.registry.model.input_requirements import (
    l2_base_enacting_metrics,
    lcia_enacting_metric_l1_metrics,
    lcia_enacting_metric_l2_metrics,
    lcia_kinds_for_method,
    method_requires_contiguous_history,
    method_requires_lcia_percap,
    method_requires_pr_hr_cumulative,
)
from pyaesa.asocc.methods.registry.model.types import MethodSpec, normalize_fu_code


class MethodRegistry(MethodRegistryFamilyChecksMixin):
    """Registry of valid L1/L2 allocation methods and input requirements."""

    def __init__(self, methods: list[MethodSpec]) -> None:
        """Store method specs."""
        self._methods = methods

    def all_methods(self) -> list[MethodSpec]:
        """Return all registry method specs."""
        return list(self._methods)

    def list_l1_methods(self) -> list[str]:
        """Return sorted unique L1 method names."""
        return sorted({m.name for m in self._methods if m.level == "L1"})

    def list_l2_methods(
        self,
        *,
        fu_code: str,
        l1_weighting: Optional[bool] = None,
    ) -> list[str]:
        """Return sorted unique L2 method names for one FU."""
        names: set[str] = set()
        for method_spec in self._methods:
            if method_spec.level != "L2":
                continue
            if method_spec.fu_code != fu_code:
                continue
            if l1_weighting is not None and method_spec.l1_weighting != l1_weighting:
                continue
            names.add(method_spec.name)
        return sorted(names)

    def get_method(
        self,
        name: str,
        *,
        level: Optional[str] = None,
    ) -> list[MethodSpec]:
        """Return method specs matching name and optional level."""
        return [
            method_spec
            for method_spec in self._methods
            if method_spec.name == name and (level is None or method_spec.level == level)
        ]

    def has_method(
        self,
        name: str,
        *,
        level: Optional[str] = None,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> bool:
        """Return whether a method exists under given filters."""
        for method_spec in self._methods:
            if method_spec.name != name:
                continue
            if level and method_spec.level != level:
                continue
            if fu_code and method_spec.fu_code != fu_code:
                continue
            if l1_weighting is not None and method_spec.l1_weighting != l1_weighting:
                continue
            return True
        return False

    def required_indices(
        self,
        method_name: str,
        fu_code: Optional[str],
        *,
        l1_weighting: Optional[bool] = None,
    ) -> tuple[str, ...]:
        """Return required indices for one method under selected filters."""
        indices: set[str] = set()
        for method_spec in self._methods:
            if method_spec.name != method_name:
                continue
            if fu_code and method_spec.fu_code != fu_code:
                continue
            if l1_weighting is not None and method_spec.l1_weighting != l1_weighting:
                continue
            indices.update(method_spec.indices)
            if method_spec.level == "L1":
                l1_kind = method_spec.l1_kind
                if l1_kind is None and fu_code:
                    fu_norm = normalize_fu_code(fu_code)
                    if fu_norm == "L1.a":
                        l1_kind = "CBA_FD"
                    elif fu_norm == "L1.b":
                        l1_kind = "PBA"
                    else:
                        l1_kind = _L2_FU_L1_KIND_MAP.get(fu_norm)
                if l1_kind == "PBA":
                    indices.add("r_p")
                elif l1_kind in {"CBA_FD", "CBA_TD"}:
                    indices.add("r_f")
        return tuple(sorted(indices))

    def validate_selection(
        self,
        fu_code: str,
        selected_methods: Iterable[str],
        *,
        l1_weighting: Optional[bool] = None,
    ) -> None:
        """Validate that selected L2 methods exist for the FU."""
        missing: list[str] = []
        for name in selected_methods:
            if not self.has_method(
                name,
                level="L2",
                fu_code=fu_code,
                l1_weighting=l1_weighting,
            ):
                missing.append(name)
        if missing:
            raise ValueError(f"Methods not found for FU {fu_code}: {sorted(set(missing))}")

    def method_requires_lcia(self, name: str, fu_code: Optional[str]) -> bool:
        """Return whether method requires LCIA inputs."""
        return any(
            method_spec.name == name
            and (fu_code is None or method_spec.fu_code == fu_code)
            and method_spec.needs_lcia
            for method_spec in self._methods
        )

    def method_requires_rp(self, name: str, fu_code: Optional[str]) -> bool:
        """Return whether method requires responsibility period data."""
        return any(
            method_spec.name == name
            and (fu_code is None or method_spec.fu_code == fu_code)
            and method_spec.needs_rp
            for method_spec in self._methods
        )

    def l1_kind_for_l2_method(self, name: str) -> str:
        """Return the L1 boundary kind required by an L2 method."""
        candidates: set[str] = set()
        for method_spec in self._methods:
            if method_spec.level != "L2" or method_spec.name != name:
                continue
            candidates.add(cast(str, method_spec.l1_kind))
        if len(candidates) == 1:
            return next(iter(candidates))
        if not candidates:
            raise ValueError(f"L2 method '{name}' is not registered with L1 boundary metadata.")
        raise ValueError(
            f"Ambiguous L1 boundary metadata for L2 method '{name}': {sorted(candidates)}"
        )

    def expand_ar_years_for_method(self, name: str) -> bool:
        """Return whether AR year expansion is enabled for this method."""
        flags = {
            method_spec.expand_ar_years for method_spec in self._methods if method_spec.name == name
        }
        if len(flags) == 1:
            return next(iter(flags))
        if not flags:
            raise ValueError(f"No method metadata found for '{name}'.")
        raise ValueError(f"Ambiguous AR expansion metadata for method '{name}': {sorted(flags)}")

    def l2_weight_axis_for_method(self, l2_method: str, fu_code: str) -> str:
        """Return canonical L1-weighting axis for a two step L2 method."""
        candidates = [
            method_spec
            for method_spec in self._methods
            if (
                method_spec.level == "L2"
                and method_spec.name == l2_method
                and method_spec.fu_code == fu_code
                and method_spec.l1_weighting
            )
        ]
        if not candidates:
            raise ValueError(
                f"No two-step registry metadata for method '{l2_method}' on {fu_code}."
            )
        axes = {str(method_spec.l2_weight_axis) for method_spec in candidates}
        if len(axes) != 1:
            raise ValueError(
                "Ambiguous L2 weight-axis metadata for method "
                f"'{l2_method}' on {fu_code}: {sorted(axes)}"
            )
        return next(iter(axes))

    def method_family(
        self,
        name: str,
        *,
        level: Optional[str] = None,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> str:
        """Return canonical method family for one method under optional filters."""
        families: set[str] = set()
        for method_spec in self._methods:
            if method_spec.name != name:
                continue
            if level and method_spec.level != level:
                continue
            if fu_code and method_spec.fu_code != fu_code:
                continue
            if l1_weighting is not None and method_spec.l1_weighting != l1_weighting:
                continue
            families.add(method_spec.family)
        if len(families) == 1:
            return next(iter(families))
        if not families:
            raise ValueError(f"No method metadata found for '{name}'.")
        raise ValueError(f"Ambiguous family metadata for method '{name}': {sorted(families)}")

    def method_requires_contiguous_history(
        self,
        name: str,
        *,
        level: Optional[str] = None,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> bool:
        """Return whether a method needs contiguous historical MRIO coverage."""
        family = self.method_family(
            name,
            level=level,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
        )
        return method_requires_contiguous_history(family=family)

    def method_requires_lcia_percap(
        self,
        name: str,
        *,
        level: Optional[str] = None,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> bool:
        """Return whether method needs LCIA per capita enacting metrics."""
        family = self.method_family(
            name,
            level=level,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
        )
        return method_requires_lcia_percap(family=family)

    def method_requires_pr_hr_cumulative(
        self,
        name: str,
        *,
        level: Optional[str] = None,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> bool:
        """Return whether method needs PR-HR cumulative per capita inputs."""
        family = self.method_family(
            name,
            level=level,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
        )
        return method_requires_pr_hr_cumulative(family=family)

    def l2_base_enacting_metrics(self, name: str, *, fu_code: str) -> tuple[str, ...]:
        """Return base (non LCIA) enacting metrics required by an L2 method."""
        family = self.method_family(name, level="L2", fu_code=fu_code)
        return l2_base_enacting_metrics(family=family, fu_code=fu_code)

    def lcia_enacting_metric_l1_metrics(
        self,
        name: str,
        *,
        level: str,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> tuple[str, ...]:
        """Return required level-1 LCIA enacting metric keys for one method."""
        lcia_kinds = self._lcia_kinds_for_method(
            name=name,
            level=level,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
        )
        return lcia_enacting_metric_l1_metrics(lcia_kinds=lcia_kinds)

    def lcia_enacting_metric_l2_metrics(
        self,
        name: str,
        *,
        level: str,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> tuple[str, ...]:
        """Return required level-2 LCIA enacting metric keys for one method."""
        lcia_kinds = self._lcia_kinds_for_method(
            name=name,
            level=level,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
        )
        return lcia_enacting_metric_l2_metrics(
            lcia_kinds=lcia_kinds,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
        )

    def _lcia_kinds_for_method(
        self,
        *,
        name: str,
        level: str,
        fu_code: Optional[str],
        l1_weighting: Optional[bool],
    ) -> set[str]:
        """Resolve canonical LCIA boundary kinds for one method selection."""
        return lcia_kinds_for_method(
            methods=self._methods,
            name=name,
            level=level,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
        )

    def l1_kinds_for_method(self, name: str) -> list[str]:
        """Return sorted LCIA boundary kinds used by one L1 method."""
        return sorted(
            self._lcia_kinds_for_method(
                name=name,
                level="L1",
                fu_code=None,
                l1_weighting=None,
            )
        )
