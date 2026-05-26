"""Deterministic aSoCC figure scope planning."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pyaesa.shared.runtime.io.persisted_paths import scoped_existing_table_paths

from pyaesa.asocc.runtime.paths.published import (
    _get_asocc_l1_dir,
    _get_asocc_l2_dir,
    _owning_fu_level_for_code,
)


@dataclass(frozen=True)
class RunScope:
    """Persisted deterministic aSoCC run scope used by figure generation."""

    proj_base: Path
    source: str
    agg_version: str | None
    agg_reg: bool
    group_indices: bool
    l1_reg_aggreg: str

    @classmethod
    def from_signature(
        cls,
        *,
        proj_base: Path,
        source: str,
        signature: dict[str, Any],
    ) -> "RunScope":
        """Build a scope from the deterministic run signature."""
        return cls(
            proj_base=Path(proj_base),
            source=str(source),
            agg_version=_optional_text(signature.get("agg_version")),
            agg_reg=bool(signature.get("agg_reg")),
            group_indices=bool(signature.get("group_indices")),
            l1_reg_aggreg=str(signature.get("l1_reg_aggreg", "post")),
        )


def figure_state_key(*, fu_code: str) -> str:
    """Return the persisted figure state key for one functional unit level."""
    level = "level_1" if str(fu_code).startswith("L1.") else "level_2"
    return f"current_figure_state__{level}"


def figure_signature(
    *,
    requested_years: list[int],
    lcia_methods: list[str] | None,
    dpi: int,
    output_format: str,
    figure_external_method: dict[str, Any] | None,
    figure_options: dict[str, bool],
) -> dict[str, Any]:
    """Return the deterministic aSoCC figure request signature."""
    return {
        "function": "deterministic_asocc_figures",
        "contract": "deterministic_asocc_scientific_percent_ticks",
        "years": [int(year) for year in requested_years],
        "lcia_methods": sorted(
            {str(method).strip() for method in (lcia_methods or []) if str(method).strip()}
        ),
        "figure_output_format": str(output_format),
        "figure_dpi": int(dpi),
        "figure_external_method": figure_external_method,
        "figure_options": dict(figure_options),
    }


def requested_ssp_scenarios(
    *,
    options_by_year: dict[int, list[str | None]] | None,
    compute_signature: dict[str, Any],
) -> list[str | None]:
    """Return requested SSP scenarios for final figure scopes."""
    requested = compute_signature.get("ssp_scenario_input")
    if requested:
        return [str(value).strip().upper() for value in requested if str(value).strip()]
    resolved_options = cast(dict[int, list[str | None]], options_by_year)
    values = {
        None if option is None else str(option).strip().upper()
        for options in resolved_options.values()
        for option in options
        if option is None or str(option).strip()
    }
    return sorted(values, key=lambda value: "" if value is None else str(value))


def scoped_output_paths(
    *,
    scope: RunScope,
    fu_code: str,
    output_paths: list[str],
) -> list[Path]:
    """Return persisted deterministic output tables for one figure request."""
    l1_root = _get_asocc_l1_dir(
        proj_base=scope.proj_base,
        source=scope.source,
        agg_version=scope.agg_version,
        lcia_sub=None,
        owning_fu_level=_owning_fu_level_for_code(fu_code=fu_code),
    )
    l2_root = _get_asocc_l2_dir(
        proj_base=scope.proj_base,
        source=scope.source,
        agg_version=scope.agg_version,
        bucket="l2_vs_global",
        lcia_sub=None,
    )
    output_paths_resolved = [Path(path) for path in output_paths]
    paths = [
        *scoped_existing_table_paths(
            raw_paths=_paths_within_root(root=l1_root, output_paths=output_paths_resolved),
            root=l1_root,
            field_name="artifacts.outputs",
        ),
        *scoped_existing_table_paths(
            raw_paths=_paths_within_root(root=l2_root, output_paths=output_paths_resolved),
            root=l2_root,
            field_name="artifacts.outputs",
        ),
    ]
    if not paths:
        raise ValueError(
            "Persisted deterministic aSoCC outputs did not resolve to any existing figure inputs."
        )
    return paths


def _paths_within_root(*, root: Path, output_paths: list[Path]) -> list[Path]:
    root_resolved = root.resolve()
    scoped: list[Path] = []
    for path in output_paths:
        try:
            Path(path).resolve().relative_to(root_resolved)
        except ValueError:
            continue
        scoped.append(path)
    return scoped


def _optional_text(value: object) -> str | None:
    return None if value is None or str(value).strip() == "" else str(value).strip()
