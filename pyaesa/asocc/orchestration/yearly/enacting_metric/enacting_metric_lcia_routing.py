"""Routing contracts for enacting metric LCIA-derived metrics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class _PerCapMetricPair:
    source_metric: str
    output_metric: str
    region_label: str


@dataclass(frozen=True)
class _PrHrCumulativeMetricContract:
    output_metric: str
    region_label: str


def _iter_lcia_percap_metric_pairs(*, required_kinds: set[str]) -> list[_PerCapMetricPair]:
    """Return enacting metric per-cap LCIA routing pairs for the required boundary kinds."""
    pairs: list[_PerCapMetricPair] = []
    if "CBA_FD" in required_kinds or "CBA_TD" in required_kinds:
        pairs.append(
            _PerCapMetricPair(
                source_metric="e_cba_fd_reg",
                output_metric="e_cba_fd_reg_cap",
                region_label="r_f",
            )
        )
    if "PBA" in required_kinds:
        pairs.append(
            _PerCapMetricPair(
                source_metric="e_pba_reg",
                output_metric="e_pba_reg_cap",
                region_label="r_p",
            )
        )
    return pairs


def _resolve_pr_hr_cumulative_metric_contract(
    *,
    lcia_kind: str,
) -> _PrHrCumulativeMetricContract:
    """Return output metric and region label for PR-HR cumulative enacting metric outputs."""
    if lcia_kind == "PBA":
        return _PrHrCumulativeMetricContract(
            output_metric="e_pba_reg_cap_cum",
            region_label="r_p",
        )
    return _PrHrCumulativeMetricContract(
        output_metric="e_cba_fd_reg_cap_cum",
        region_label="r_f",
    )
