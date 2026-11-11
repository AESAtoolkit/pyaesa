"""Restrict internal ASR calculations to characterized numerator categories."""

import pandas as pd

from pyaesa.io_lca.data.contracts import IO_LCA_FAMILY
from pyaesa.shared.runtime.reporting.status import StatusSink


def compatible_numerator_categories(
    *,
    denominator: pd.DataFrame,
    numerator: pd.DataFrame,
    lca_type: str,
    source: str,
    status: StatusSink,
) -> pd.DataFrame:
    """Keep actual internal IO categories; external numerator matching stays strict.

    A category represented anywhere in the numerator remains subject to exact
    year and selector matching. Missing rows within such a category are errors.
    """
    if lca_type != IO_LCA_FAMILY:
        return denominator
    available = set(numerator["impact"].astype(str).str.strip())
    impacts = denominator["impact"].astype(str).str.strip()
    missing = sorted(set(impacts) - available)
    if not missing:
        return denominator
    labels = sorted(set(denominator.loc[impacts.isin(missing), "l1_l2_method"].astype(str)))
    message = (
        f"Source '{source}': internal IO-LCA cannot characterize categories {missing}; "
        f"ASR excludes these categories for allocation methods {labels}. "
        "An external LCA numerator with compatible category coverage can assess them."
    )
    selected = denominator.loc[impacts.isin(available)].copy()
    if selected.empty:
        raise ValueError(message + " No compatible ASR categories remain.")
    status.log_message(message)
    return selected
