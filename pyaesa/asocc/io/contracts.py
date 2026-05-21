"""Shared deterministic aSoCC branch contract tokens."""

GROUPED_MODE_LABEL = "grouped"
UNGROUPED_MODE_LABEL = "ungrouped"
ASOCC_FAMILY = "asocc"
EXTERNAL_ASOCC_DIRNAME = "external_asocc"
EXTERNAL_ASOCC_FAMILY = EXTERNAL_ASOCC_DIRNAME


def mode_label(*, aggreg_indices: bool) -> str:
    """Return the canonical grouped or ungrouped branch label."""
    return GROUPED_MODE_LABEL if bool(aggreg_indices) else UNGROUPED_MODE_LABEL
