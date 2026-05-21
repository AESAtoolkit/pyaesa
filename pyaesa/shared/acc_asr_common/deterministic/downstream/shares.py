"""Shared downstream aSoCC share context for deterministic aCC and ASR."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyaesa.shared.tabular.wide_tables import resolve_single_allocation_method_identity

from .inputs import LoadedAsoccShare, combined_asocc_shares, load_asocc_share
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens

from .scenarios import share_transition_metadata


@dataclass(frozen=True)
class DownstreamAsoccShareContext:
    """Resolved deterministic downstream aSoCC shares and transition metadata."""

    asocc_shares: list[LoadedAsoccShare]
    share_transition_meta: dict[str, dict[str, object]]

    @property
    def allowed_l1_l2_methods(self) -> set[str]:
        """Return the allowed canonical allocation identities for downstream consumers."""
        return {
            resolve_single_allocation_method_identity(
                item.frame_wide,
                where=f"Deterministic downstream aSoCC share '{item.display_name}'",
            )
            for item in self.asocc_shares
        }


def build_downstream_asocc_share_context(
    *,
    proj_base: Path,
    source_label: str,
    base_allocate_args: dict[str, Any],
    fu_code: str,
    external_method: dict[str, Any] | None,
    years: list[int],
    lcia_method: str | None,
    output_source_label: str,
    branch_ssp_scenario: list[str] | None = None,
    switch_label: str = "Switch year for SSP-dependent series",
) -> DownstreamAsoccShareContext:
    """Build the shared deterministic downstream aSoCC share context for aCC and ASR."""
    asocc_shares = [
        load_asocc_share(asocc_share)
        for asocc_share in combined_asocc_shares(
            proj_base=proj_base,
            source_label=source_label,
            base_allocate_args=base_allocate_args,
            fu_code=fu_code,
            external_method=external_method,
            years=years,
            lcia_method=lcia_method,
            output_source_label=output_source_label,
        )
    ]
    share_transition_meta = share_transition_metadata(
        asocc_shares=asocc_shares,
        scenario_tokens=_resolved_downstream_scenario_tokens(
            base_allocate_args=base_allocate_args,
            branch_ssp_scenario=branch_ssp_scenario,
        ),
        switch_label=switch_label,
    )
    return DownstreamAsoccShareContext(
        asocc_shares=asocc_shares,
        share_transition_meta=share_transition_meta,
    )


def _resolved_downstream_scenario_tokens(
    *,
    base_allocate_args: dict[str, Any],
    branch_ssp_scenario: list[str] | None,
) -> list[str]:
    """Return the effective SSP tokens for one deterministic downstream branch."""
    return sorted(
        {
            *normalize_ssp_tokens(base_allocate_args["ssp_scenario"]),
            *normalize_ssp_tokens(branch_ssp_scenario),
        }
    )
