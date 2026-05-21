"""Unit interval source coordinates for Sobol evaluation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SourceUnitIntervalSamples:
    """Values in [0, 1] keyed by active uncertainty source name."""

    values_by_source: dict[str, np.ndarray]

    def values_for(self, source: str) -> np.ndarray | None:
        """Return source coordinates or None when run streams own sampling."""
        return self.values_by_source.get(source)
