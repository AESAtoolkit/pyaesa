"""Parsed file specifications for external aSoCC inputs."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from pyaesa.asocc.methods.registry.registry import normalize_fu_code
from pyaesa.asocc.runtime.paths.external import (
    get_asocc_external_method_level_dir,
)
from pyaesa.shared.tabular.empty_rows import drop_fully_empty_rows
from pyaesa.shared.lcia.availability import discover_static_cc_methods
from pyaesa.shared.lcia.contracts import bundled_cc_expected_impacts
from pyaesa.shared.runtime.scenario.file_routing import (
    ScenarioTaggedFileSpec,
    resolve_year_assignments,
    validate_scenario_inventory,
)
from pyaesa.shared.tabular.contracts import TABULAR_SUFFIX_SET
from pyaesa.shared.tabular.wide_tables import (
    detect_year_columns,
    validate_complete_wide_year_values,
)

from pyaesa.external_inputs.asocc.schema.contracts import ExternalMethodSelection
from pyaesa.external_inputs.shared.compact_matrix import is_compact_run_matrix_dir
from pyaesa.external_inputs.shared.scenario_tokens import (
    external_file_ssp_token,
    external_row_ssp_token,
)
from pyaesa.external_inputs.shared.tabular import year_columns_from_schema


def external_asocc_runtime_file_stem(
    *,
    fu_code: str,
    file_method_token: str,
    l1_method: str | None,
    lcia_method: str | None,
    scenario: str | None,
) -> str:
    """Return the deterministic runtime file stem for one external aSoCC table."""
    fu_norm = normalize_fu_code(fu_code)
    if fu_norm.startswith("L1."):
        stem = f"l1_{file_method_token}"
    elif l1_method is None:
        stem = str(file_method_token)
    else:
        stem = f"l1_{l1_method}_l2_{file_method_token}"
    if lcia_method is not None:
        stem = f"{stem}__{lcia_method}"
    if scenario is not None:
        stem = f"{stem}__{external_file_ssp_token(scenario, family_label='External aSoCC')}"
    return stem


def external_asocc_expected_stems(
    *,
    fu_code: str,
    file_method_token: str,
    l1_method: str | None,
    lcia_methods: list[str] | None,
    years: list[int],
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> list[str]:
    """Return expected deterministic external aSoCC stems for one selection."""
    del years
    methods = [
        None,
        *sorted({str(value).strip() for value in lcia_methods or [] if str(value).strip()}),
    ]
    scenarios = sorted(
        {
            external_file_ssp_token(value, family_label="External aSoCC")
            for value_by_year in (ssp_scenario_options_by_year or {}).values()
            for value in value_by_year or [None]
            if value is not None
        }
    )
    stems: set[str] = set()
    for lcia_method in methods:
        stems.add(
            external_asocc_runtime_file_stem(
                fu_code=fu_code,
                file_method_token=file_method_token,
                l1_method=l1_method,
                lcia_method=lcia_method,
                scenario=None,
            )
        )
        for scenario in scenarios:
            stems.add(
                external_asocc_runtime_file_stem(
                    fu_code=fu_code,
                    file_method_token=file_method_token,
                    l1_method=l1_method,
                    lcia_method=lcia_method,
                    scenario=scenario,
                )
            )
    return sorted(stems)


@dataclass(frozen=True)
class ExternalAsoCCFileSpec(ScenarioTaggedFileSpec):
    """Parsed filename contract for one external aSoCC file."""

    lcia_method: str | None


def read_external_asocc_table(path: Path) -> pd.DataFrame:
    """Read one external aSoCC table by suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".pickle":
        frame = cast(pd.DataFrame, pd.read_pickle(path))
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported external aSoCC file format '{path.suffix}' at {path}. "
            "Use one of: ['.csv', '.parquet', '.pickle']."
        )
    return drop_fully_empty_rows(frame=frame)


def requested_lcia_methods(lcia_methods: list[str] | None) -> tuple[str, ...]:
    """Return normalized requested LCIA methods."""
    return tuple(sorted({str(value).strip() for value in lcia_methods or [] if str(value).strip()}))


def frame_years(frame: pd.DataFrame) -> list[int]:
    """Return years covered by one external aSoCC table."""
    if "year" in frame.columns:
        return sorted({int(value) for value in frame["year"].dropna().tolist()})
    year_columns = detect_year_columns(frame)
    validate_complete_wide_year_values(
        frame,
        year_columns=year_columns,
        where="External aSoCC wide tables",
    )
    return [int(column) for column in year_columns]


def _file_years(*, path: Path, storage_mode: str) -> tuple[int, ...]:
    if storage_mode == "monte_carlo":
        return ()
    return tuple(sorted(year_columns_from_schema(path)))


def _parse_suffix(
    *,
    suffix: str,
    requested_methods: tuple[str, ...],
) -> tuple[str | None, str | None]:
    text = str(suffix).strip()
    if not text:
        return None, None
    tokens = [token.strip() for token in text[2:].split("__") if token.strip()]
    if not tokens:
        return None, None
    candidate_methods = sorted(
        {*(discover_static_cc_methods()), *requested_methods},
        key=len,
        reverse=True,
    )
    first = tokens[0]
    if first in candidate_methods:
        scenario = (
            external_row_ssp_token(tokens[1], family_label="External aSoCC")
            if len(tokens) > 1
            else None
        )
        return first, scenario
    return None, external_row_ssp_token(first, family_label="External aSoCC")


def _parse_candidate_file(
    *,
    path: Path,
    selection: ExternalMethodSelection,
    requested_methods: tuple[str, ...],
    storage_mode: str = "deterministic",
) -> ExternalAsoCCFileSpec | None:
    base_stem = external_asocc_runtime_file_stem(
        fu_code=selection.fu_code,
        file_method_token=selection.file_method_token,
        l1_method=selection.l1_method,
        lcia_method=None,
        scenario=None,
    )
    stem = path.name if path.is_dir() else path.stem
    if stem == base_stem:
        suffix = ""
    elif stem.startswith(f"{base_stem}__"):
        suffix = stem[len(base_stem) :]
    else:
        return None
    if storage_mode == "monte_carlo" and suffix:
        tokens = [token.strip() for token in suffix[2:].split("__") if token.strip()]
        candidate_methods = {*(discover_static_cc_methods()), *requested_methods}
        if len(tokens) != 1 or tokens[0] not in candidate_methods:
            return None
    lcia_method, scenario = _parse_suffix(suffix=suffix, requested_methods=requested_methods)
    return ExternalAsoCCFileSpec(
        path=path,
        scenario=scenario,
        years=_file_years(path=path, storage_mode=storage_mode),
        lcia_method=lcia_method,
    )


def candidate_files(
    *,
    proj_base: Path,
    selection: ExternalMethodSelection,
    lcia_methods: list[str] | None,
    storage_mode: str = "deterministic",
) -> tuple[ExternalAsoCCFileSpec, ...]:
    """Return parsed file specs for one external aSoCC selection."""
    directory = get_asocc_external_method_level_dir(
        proj_base=proj_base,
        storage_mode=storage_mode,
        level=str(selection.level),
    )
    if not directory.exists():
        return tuple()
    requested_methods = requested_lcia_methods(lcia_methods)
    out: list[ExternalAsoCCFileSpec] = []
    candidates = [
        item
        for item in directory.iterdir()
        if (item.is_file() and item.suffix.lower() in TABULAR_SUFFIX_SET)
        or (
            storage_mode == "monte_carlo"
            and is_compact_run_matrix_dir(item, run_file_name="asocc_runs.csv")
        )
    ]
    for path in sorted(candidates):
        spec = _parse_candidate_file(
            path=path,
            selection=selection,
            requested_methods=requested_methods,
            storage_mode=storage_mode,
        )
        if spec is not None:
            out.append(spec)
    return tuple(out)


def validate_lcia_inventory(
    *,
    specs: tuple[ExternalAsoCCFileSpec, ...],
    selection: ExternalMethodSelection,
    lcia_methods: list[str] | None,
) -> None:
    """Validate LCIA token inventory for one external aSoCC selection."""
    if not specs:
        return
    lcia_tokens = {spec.lcia_method for spec in specs}
    if None in lcia_tokens and len(lcia_tokens) > 1:
        raise ValueError(
            "External aSoCC candidate files must not mix LCIA token and non LCIA stems. "
            f"Selection='{selection.asocc_method_label}'."
        )
    requested = set(requested_lcia_methods(lcia_methods))
    unexpected = sorted(
        {token for token in lcia_tokens if token is not None and token not in requested}
    )
    if unexpected:
        raise ValueError(
            "External aSoCC LCIA tokens must match the requested lcia_method scope. "
            f"Selection='{selection.asocc_method_label}', unexpected tokens={unexpected}, "
            f"requested={sorted(requested)}."
        )


def resolve_external_asocc_year_assignments(
    *,
    specs: tuple[ExternalAsoCCFileSpec, ...],
    selection: ExternalMethodSelection,
    years: list[int],
    lcia_methods: list[str] | None,
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> dict[Path, list[int]]:
    """Resolve requested years for one external aSoCC file-spec set."""
    validate_scenario_inventory(
        specs=tuple(
            ScenarioTaggedFileSpec(path=spec.path, scenario=spec.scenario, years=spec.years)
            for spec in specs
        ),
        family_label="external aSoCC",
        item_label=f"selection '{selection.asocc_method_label}'",
    )
    return resolve_year_assignments(
        specs=tuple(
            ScenarioTaggedFileSpec(path=spec.path, scenario=spec.scenario, years=spec.years)
            for spec in specs
        ),
        years=years,
        ssp_scenario_options_by_year=ssp_scenario_options_by_year,
        family_label="external aSoCC",
        item_label=f"selection '{selection.asocc_method_label}'",
        expected_stems=external_asocc_expected_stems(
            fu_code=selection.fu_code,
            file_method_token=selection.file_method_token,
            l1_method=selection.l1_method,
            lcia_methods=lcia_methods,
            years=years,
            ssp_scenario_options_by_year=ssp_scenario_options_by_year,
        ),
    )


def validate_impact_contract(
    *,
    frame: pd.DataFrame,
    path: Path,
    lcia_method: str | None,
) -> None:
    """Validate external aSoCC impact metadata against the bundled CC contract."""
    validate_lcia_axis_columns(columns=frame.columns, path=path, lcia_method=lcia_method)
    if lcia_method is None:
        return
    cc_csv_path, expected = bundled_cc_expected_impacts(lcia_method=lcia_method)
    found = sorted({str(value).strip() for value in frame["impact"].dropna().astype(str).tolist()})
    if found != expected:
        raise ValueError(
            "External aSoCC LCIA impacts must match the bundled carrying capacity CSV exactly. "
            f"External file: '{path}'. Validation CSV: '{cc_csv_path}'. "
            f"Expected impacts: {expected}. Found: {found}."
        )


def validate_lcia_axis_columns(
    *,
    columns: Iterable[object],
    path: Path,
    lcia_method: str | None,
) -> None:
    """Validate staged external aSoCC LCIA and reference year axis columns."""
    column_set = {str(column) for column in columns}
    if lcia_method is None:
        conflicts = sorted(
            column for column in ("impact", "reference_year") if column in column_set
        )
        if conflicts:
            if conflicts == ["impact"]:
                conflict_text = "an impact column"
            elif conflicts == ["reference_year"]:
                conflict_text = "a reference_year column"
            else:
                conflict_text = "impact or reference_year columns"
            raise ValueError(
                f"External aSoCC file '{path}' is not tied to an LCIA method in its "
                f"filename, so it must not contain {conflict_text}. Remove "
                "LCIA specific columns or use an LCIA method filename suffix."
            )
        return
    if "impact" not in column_set:
        raise ValueError(
            f"LCIA based external aSoCC file '{path}' is missing the required 'impact' column."
        )
