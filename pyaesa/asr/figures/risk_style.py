"""Shared risk style color rules for boundary and ratio interpretation plots.

The colormaps and scaling helpers here provide a common visual treatment for
values interpreted against carrying capacity or ASR thresholds so equivalent
risk semantics are rendered consistently across figure families.
"""

from matplotlib import colors as mcolors

SAFE_COLOR = "#39ba38"
RAMP_END_COLOR = "#7c281e"
MAX_RISK_COLOR = "#7b1d1d"

SAFE_FRAC = 0.23
RISK_RAMP_END = 0.83
SCALE_LIGHTEN = 0.70

RISK_RAMP_STOPS = [
    (0.00, "#f6efad"),
    (0.10, "#f4e083"),
    (0.24, "#f1cb56"),
    (0.40, "#eaa73a"),
    (0.58, "#df8428"),
    (0.72, "#cd6a22"),
    (0.84, "#b9541f"),
    (0.92, "#a6451f"),
    (0.97, "#92381f"),
    (0.99, "#842f1f"),
    (1.00, RAMP_END_COLOR),
]


def make_bg_risk_cmap(n: int = 512) -> mcolors.LinearSegmentedColormap:
    """Build a colormap for background risk bands."""
    return mcolors.LinearSegmentedColormap.from_list("pb_risk", RISK_RAMP_STOPS, N=n)
