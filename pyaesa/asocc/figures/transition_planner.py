"""aSoCC deterministic figure transition marker policy."""

import pandas as pd

from pyaesa.shared.figures.asocc_transition_policy import asocc_transition_year


def transition_year(group: pd.DataFrame) -> int | None:
    """Return the transition year when retrospective and prospective rows are both visible."""
    return asocc_transition_year(group)
