"""AR6 CC uncertainty summary identity grouping."""

import pandas as pd

from pyaesa.shared.uncertainty_assessment.evaluation.summary_groups import (
    identity_groups_from_excluded_columns,
)


def ar6_cc_summary_identity_groups(
    *,
    identity: pd.DataFrame,
    category_uncertainty: bool,
) -> tuple[pd.DataFrame, tuple[tuple[str, ...], ...]]:
    """Return AR6 CC summary rows and backing public row groups."""
    excluded = {"cc_model", "cc_scenario"}
    if category_uncertainty:
        excluded.add("cc_category")
    return identity_groups_from_excluded_columns(identity=identity, excluded_columns=excluded)
