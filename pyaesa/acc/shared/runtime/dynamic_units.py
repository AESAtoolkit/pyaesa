"""Dynamic aCC carrying capacity unit conversion."""

from pathlib import Path

import numpy as np
import pandas as pd

from pyaesa.shared.lcia.contracts import bundled_cc_impact_unit
from pyaesa.shared.lcia.units import try_unit_conversion


def dynamic_acc_target_impact_unit(*, cc_source: str, impact: str) -> str:
    """Return the public dynamic aCC unit from the bundled static CC prerequisite."""
    _path, unit = bundled_cc_impact_unit(lcia_method=cc_source, impact=impact)
    return unit


def dynamic_acc_unit_factors(
    *,
    source_units: pd.Series,
    cc_source: str,
    impact: str,
    source_path: Path | None = None,
) -> tuple[str, np.ndarray]:
    """Return source CC to dynamic aCC unit factors for public aCC values."""
    target_unit = dynamic_acc_target_impact_unit(cc_source=cc_source, impact=impact)
    source_text = pd.Series(source_units, copy=False).astype("string").str.strip()
    unique_units = sorted(set(source_text.dropna().tolist()))
    factors_by_unit: dict[str, float] = {}
    for source_unit in unique_units:
        factor = try_unit_conversion(str(source_unit), target_unit)
        if factor is None:
            path_text = "" if source_path is None else f" Source table: '{source_path}'."
            raise ValueError(
                "Dynamic aCC cannot convert AR6 carrying capacity units to the bundled "
                "static carrying capacity unit. "
                f"cc_source='{cc_source}', impact='{impact}', "
                f"source_unit='{source_unit}', target_unit='{target_unit}'.{path_text}"
            )
        factors_by_unit[str(source_unit)] = float(factor)
    factors = np.asarray(
        [factors_by_unit[str(value)] for value in source_text.tolist()],
        dtype=np.float64,
    )
    return target_unit, factors
