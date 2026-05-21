"""Assign requested years to historical or scenario tagged external files.

The routing and validation helpers here enforce the shared external file
contract used by external aSoCC and external LCA inputs: one file per scenario
variant, no overlapping historical versus SSP coverage, and exactly one file
assignment for every requested year.
"""

from dataclasses import dataclass
from pathlib import Path

from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens


@dataclass(frozen=True)
class ScenarioTaggedFileSpec:
    """One scenario aware external file candidate."""

    path: Path
    scenario: str | None
    years: tuple[int, ...]


def allowed_scenarios_for_year(
    *,
    year: int,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> set[str]:
    """Return the resolved non null scenario tokens allowed for one year."""
    if ssp_scenario_options_by_year is None:
        return set()
    return set(
        normalize_ssp_tokens(
            [
                str(value)
                for value in (ssp_scenario_options_by_year.get(int(year), [None]) or [None])
                if value is not None
            ]
        )
    )


def validate_scenario_inventory(
    *,
    specs: tuple[ScenarioTaggedFileSpec, ...],
    family_label: str,
    item_label: str,
) -> None:
    """Validate one file per scenario and historical vs SSP coverage rules."""
    if not specs:
        return
    by_scenario: dict[str | None, ScenarioTaggedFileSpec] = {}
    for spec in specs:
        if spec.scenario in by_scenario:
            scenario_label = "historical" if spec.scenario is None else str(spec.scenario)
            existing = by_scenario[spec.scenario]
            raise ValueError(
                f"{family_label} allows only one file per scenario variant. "
                f"{item_label} has duplicate files for '{scenario_label}': "
                f"'{existing.path}' and '{spec.path}'."
            )
        by_scenario[spec.scenario] = spec
    non_scenario_years = set(by_scenario.get(None, ScenarioTaggedFileSpec(Path(), None, ())).years)
    scenario_year_sets = {
        str(scenario): set(spec.years)
        for scenario, spec in by_scenario.items()
        if scenario is not None
    }
    if scenario_year_sets:
        expected_sets = {frozenset(years) for years in scenario_year_sets.values()}
        if len(expected_sets) > 1:
            raise ValueError(
                f"All SSP-tagged {family_label} files for {item_label} must cover the same "
                f"year set. Observed coverage: {scenario_year_sets}."
            )
    overlaps = {
        scenario: sorted(non_scenario_years & years)
        for scenario, years in scenario_year_sets.items()
        if non_scenario_years & years
    }
    if overlaps:
        raise ValueError(
            f"Historical and SSP-tagged {family_label} files must not overlap on year coverage. "
            f"{item_label} has overlapping years: {overlaps}."
        )


def resolve_year_assignments(
    *,
    specs: tuple[ScenarioTaggedFileSpec, ...],
    years: list[int],
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
    family_label: str,
    item_label: str,
    expected_stems: list[str] | None = None,
) -> dict[Path, list[int]]:
    """Assign each requested year to exactly one external file."""
    assignments: dict[Path, list[int]] = {spec.path: [] for spec in specs}
    expected_suffix = "" if not expected_stems else f" Expected stems: {expected_stems}."
    for year in years:
        matching = [spec for spec in specs if int(year) in spec.years]
        if not matching:
            raise ValueError(
                f"{family_label} coverage does not include requested year {year} for "
                f"{item_label}.{expected_suffix}"
            )
        historical = [spec for spec in matching if spec.scenario is None]
        if historical:
            assignments[historical[0].path].append(int(year))
            continue
        allowed = allowed_scenarios_for_year(
            year=int(year),
            ssp_scenario_options_by_year=ssp_scenario_options_by_year,
        )
        scenario_matches = [spec for spec in matching if spec.scenario in allowed]
        matched_scenarios = {
            str(spec.scenario) for spec in scenario_matches if spec.scenario is not None
        }
        if not allowed or matched_scenarios != {str(value) for value in allowed}:
            raise ValueError(
                f"{family_label} could not resolve SSP-tagged files matching the full allowed "
                f"scenario set for year {year} for {item_label}. "
                f"Allowed scenarios: {sorted(allowed)}.{expected_suffix}"
            )
        for spec in scenario_matches:
            assignments[spec.path].append(int(year))
    return {path: year_list for path, year_list in assignments.items() if year_list}
