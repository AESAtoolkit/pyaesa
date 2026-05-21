"""aSoCC uncertainty figure transition marker policy."""

import pandas as pd

from pyaesa.shared.figures.asocc_transition_policy import asocc_transition_year


def transition_year(group: pd.DataFrame) -> int | None:
    """Return at most one retrospective to prospective transition year."""
    return asocc_transition_year(group)
