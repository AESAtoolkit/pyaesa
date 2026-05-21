"""Canonical AR6 CC figure style policy."""

AR6_CATEGORY_COLORS = {
    "C1": "#7FBC41",
    "C2": "#2654D2",
    "C3": "#F39B1F",
    "C4": "#5A0418",
}

AR6_CC_POSITIVE_FLOW_COLOR = "#54A24B"
AR6_CC_NEGATIVE_FLOW_COLOR = "#E68613"
AR6_CC_FLOW_COLORS = {
    "net": AR6_CC_POSITIVE_FLOW_COLOR,
    "positive": AR6_CC_POSITIVE_FLOW_COLOR,
    "net_emissions": AR6_CC_POSITIVE_FLOW_COLOR,
    "positive_emissions": AR6_CC_POSITIVE_FLOW_COLOR,
    "negative": AR6_CC_NEGATIVE_FLOW_COLOR,
    "negative_sequestration": AR6_CC_NEGATIVE_FLOW_COLOR,
}


def ar6_category_color(*, category: str) -> str:
    """Return the stable AR6 CC color for one retained category."""
    return AR6_CATEGORY_COLORS[str(category)]


def ar6_cc_flow_color(flow: str) -> str:
    """Return the stable AR6 CC flow color for positive and negative flow plots."""
    return AR6_CC_FLOW_COLORS[str(flow).strip()]
