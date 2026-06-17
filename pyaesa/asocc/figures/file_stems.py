"""aSoCC figure file stem ownership."""

import pandas as pd

from pyaesa.shared.figures.scope_values import visible_scope_values
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.tabular.scalars import sanitize_token


def asocc_scope_stem(
    label: str,
    frame: pd.DataFrame,
    *,
    include_impact: bool,
    selector_token: str = "all",
    studied_year: int | None = None,
) -> str:
    """Return one deterministic aSoCC figure file stem."""
    prefix = [label]
    if str(selector_token).strip() and selector_token != "all":
        prefix.append(str(selector_token).strip())
    lcia_method = visible_scope_values(frame, "lcia_method")[:1]
    parts = []
    if include_impact:
        parts.extend(visible_scope_values(frame, "impact")[:1])
    parts.extend(visible_scope_values(frame, ASOCC_SSP_SCENARIO_COLUMN)[:1])
    if studied_year is not None:
        parts.append(str(int(studied_year)))
    stem = "_".join(sanitize_token(part) for part in prefix if str(part).strip())
    if lcia_method:
        lcia_token = sanitize_token(lcia_method[0])
        stem = f"{stem}__{lcia_token}" if stem else lcia_token
    suffix = [sanitize_token(part) for part in parts if str(part).strip()]
    return "_".join([stem, *suffix] if stem else suffix)
