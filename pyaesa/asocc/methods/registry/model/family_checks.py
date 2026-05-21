"""Family check mixin for registry method groups."""

from abc import ABC, abstractmethod
from typing import Optional


class MethodRegistryFamilyChecksMixin(ABC):
    """Family checks shared by `MethodRegistry`."""

    @abstractmethod
    def method_family(
        self,
        name: str,
        *,
        level: Optional[str] = None,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> str:
        """Return canonical method family for one method selection."""

    def method_is_ar(
        self,
        name: str,
        *,
        level: Optional[str] = None,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> bool:
        """Return whether a method belongs to AR families."""
        family = self.method_family(
            name,
            level=level,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
        )
        return family in {"AR_E", "AR_ECAP"}

    def method_is_ar_cap(
        self,
        name: str,
        *,
        level: Optional[str] = None,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> bool:
        """Return whether a method belongs to AR(Ecap) family."""
        return (
            self.method_family(
                name,
                level=level,
                fu_code=fu_code,
                l1_weighting=l1_weighting,
            )
            == "AR_ECAP"
        )

    def method_is_ut(
        self,
        name: str,
        *,
        level: Optional[str] = None,
        fu_code: Optional[str] = None,
        l1_weighting: Optional[bool] = None,
    ) -> bool:
        """Return whether a method belongs to UT families."""
        family = self.method_family(
            name,
            level=level,
            fu_code=fu_code,
            l1_weighting=l1_weighting,
        )
        return family in {"UT_FD", "UT_FDA", "UT_GVAA", "UT_TD", "UT_GVA"}
