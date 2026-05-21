"""Shared I/O and filename contracts for external LCA inputs."""

from dataclasses import dataclass
from pathlib import Path
import re
from collections.abc import Iterable
from typing import Any, cast

import pandas as pd

from pyaesa.shared.figures.contracts import SELECTOR_COLUMNS
from pyaesa.shared.lcia.contracts import bundled_cc_expected_impact_units
from pyaesa.shared.selectors.fu_axes import expected_fu_selector_columns
from pyaesa.shared.runtime.scenario.file_routing import (
    ScenarioTaggedFileSpec,
    allowed_scenarios_for_year,
    validate_scenario_inventory,
)
from pyaesa.shared.tabular.empty_rows import drop_fully_empty_rows
from pyaesa.shared.tabular.contracts import TABULAR_SUFFIX_SET
from pyaesa.shared.tabular.wide_tables import (
    melt_requested_year_value_rows,
    validate_complete_wide_year_values,
)
from pyaesa.external_inputs.shared.scenario_tokens import (
    external_file_ssp_token,
    external_row_ssp_token,
)
from pyaesa.external_inputs.shared.tabular import tabular_columns, year_columns_from_schema
from .naming import normalize_external_lca_version_name
from pyaesa.shared.runtime.scenario.columns import (
    EXT_LCA_SSP_SCENARIO_COLUMN,
    LCA_SSP_START_YEAR_COLUMN,
)

_MANDATORY_COLUMNS = {"impact", "impact_unit"}
_STANDARD_SOURCE_COLUMNS = {
    "run_index",
    "lcia_method",
    *SELECTOR_COLUMNS,
    "impact",
    "impact_unit",
    "value",
    "ssp_scenario",
    EXT_LCA_SSP_SCENARIO_COLUMN,
    LCA_SSP_START_YEAR_COLUMN,
    "year",
}
_FILE_OWNED_OR_INTERNAL_COLUMNS = frozenset(
    {"lcia_method", "ssp_scenario", EXT_LCA_SSP_SCENARIO_COLUMN}
)
_DETERMINISTIC_LCA_FILENAME_RE = re.compile(
    r"^(?P<version>.+?)__(?P<method>.+?)(?:__(?P<scenario>ssp\d+))?$",
)
_SSP_SUFFIX_CANDIDATE_RE = re.compile(r"__(?P<scenario>ssp\d+)$", re.IGNORECASE)


def _ordered_external_lca_columns(*, frame: pd.DataFrame) -> list[str]:
    """Return the canonical public column order for normalized external LCA rows."""
    base_columns = ["year", "impact", "impact_unit", "value"]
    selector_columns = [column for column in SELECTOR_COLUMNS if column in frame.columns]
    scenario_columns = [
        column
        for column in [EXT_LCA_SSP_SCENARIO_COLUMN, LCA_SSP_START_YEAR_COLUMN]
        if column in frame.columns
    ]
    extra_columns = [
        column
        for column in external_source_driver_columns(frame)
        if column not in {*base_columns, *selector_columns, *scenario_columns}
    ]
    return [*base_columns, *selector_columns, *scenario_columns, *extra_columns]


def finalize_external_lca_loaded_rows(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Return canonical runtime rows for loaded deterministic LCA data."""
    out = frame.copy()
    out["year"] = cast(pd.Series, out["year"]).astype(int).astype(str)
    out["impact"] = out["impact"].astype(str)
    out["impact_unit"] = out["impact_unit"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    return out.loc[
        :,
        _ordered_external_lca_columns(frame=out),
    ]


@dataclass(frozen=True)
class ExternalLCAFileSpec(ScenarioTaggedFileSpec):
    """Parsed filename contract for one external LCA file."""

    version_name: str
    lcia_method: str


def discover_external_lca_files(directory: Path) -> tuple[Path, ...]:
    """Return direct external LCA files from one storage directory."""
    if not directory.exists():
        return tuple()
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in TABULAR_SUFFIX_SET
        )
    )


def external_lca_expected_stems(
    *,
    version_name: str,
    lcia_method: str,
    years: list[int],
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> list[str]:
    """Return representative expected deterministic external LCA stems."""
    del years
    version = normalize_external_lca_version_name(
        version_name,
        argument_name="external LCA version_name",
    )
    stems = {f"{version}__{str(lcia_method)}"}
    scenarios = sorted(
        {
            external_file_ssp_token(value, family_label="External LCA")
            for value_by_year in (ssp_scenario_options_by_year or {}).values()
            for value in value_by_year or [None]
            if value is not None
        }
    )
    stems.update({f"{version}__{lcia_method}__{scenario}" for scenario in scenarios})
    return sorted(stems)


def read_external_lca(path: Path) -> pd.DataFrame:
    """Read one external LCA file and validate required identifier columns."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(path)
    elif suffix == ".parquet":
        raw = pd.read_parquet(path)
    elif suffix == ".pickle":
        raw = pd.read_pickle(path)
    else:
        raise ValueError(
            f"Unsupported external LCA file format for '{path}'. "
            "Use one of: .csv, .parquet, .pickle."
        )
    df = drop_fully_empty_rows(frame=cast(pd.DataFrame, raw))
    validate_external_lca_required_columns(
        columns=df.columns,
        path=path,
        required_columns=_MANDATORY_COLUMNS,
        file_label="External LCA file",
    )
    return df


def parse_external_lca_filename(*, path: Path) -> ExternalLCAFileSpec:
    """Parse one external LCA filename into its exact runtime contract."""
    stem = path.stem
    match = _DETERMINISTIC_LCA_FILENAME_RE.match(stem)
    if match is None or stem.lower().startswith("runs__"):
        raise ValueError(
            f"External LCA deterministic file '{path.name}' must use "
            "'<version_name>__<lcia_method>' or "
            "'<version_name>__<lcia_method>__<ssp_scenario>' stems."
        )
    ssp_suffix_match = _SSP_SUFFIX_CANDIDATE_RE.search(stem)
    if ssp_suffix_match is not None and match.group("scenario") is None:
        external_row_ssp_token(
            str(ssp_suffix_match.group("scenario")).strip(),
            family_label="External LCA",
        )
    columns = tabular_columns(path)
    validate_external_lca_required_columns(
        columns=columns,
        path=path,
        required_columns=_MANDATORY_COLUMNS,
        file_label="External LCA file",
    )
    return ExternalLCAFileSpec(
        path=path,
        scenario=(
            None
            if match.group("scenario") is None
            else external_row_ssp_token(
                str(match.group("scenario")).strip(),
                family_label="External LCA",
            )
        ),
        years=tuple(year_columns_from_schema(path)),
        version_name=normalize_external_lca_version_name(
            str(match.group("version")).strip(),
            argument_name="external LCA filename version_name",
        ),
        lcia_method=str(match.group("method")).strip(),
    )


def matching_external_lca_specs(
    *,
    directory: Path,
    version_name: str,
    lcia_method: str,
) -> tuple[ExternalLCAFileSpec, ...]:
    """Return parsed deterministic external LCA specs matching one LCIA method."""
    version = normalize_external_lca_version_name(
        version_name,
        argument_name="external LCA version_name",
    )
    specs: list[ExternalLCAFileSpec] = []
    for path in discover_external_lca_files(directory):
        spec = parse_external_lca_filename(path=path)
        if spec.version_name == version and spec.lcia_method == str(lcia_method):
            specs.append(spec)
    return tuple(specs)


def validate_external_lca_contract(
    *,
    frame: pd.DataFrame,
    path: Path,
    lcia_method: str,
) -> None:
    """Validate exact external LCA impact and unit names against the bundled CC CSV."""
    cc_csv_path, expected_pairs = bundled_cc_expected_impact_units(lcia_method=lcia_method)
    metadata_rows = frame.loc[:, ["impact", "impact_unit"]].dropna(how="any")
    found_pairs = sorted(
        {
            (str(impact).strip(), str(impact_unit).strip())
            for impact, impact_unit in metadata_rows.itertuples(index=False, name=None)
        }
    )
    if found_pairs != expected_pairs:
        raise ValueError(
            "External LCA impact metadata must match the bundled carrying capacity CSV exactly. "
            f"External file: '{path}'. Validation CSV: '{cc_csv_path}'. "
            f"Expected (impact, impact_unit) pairs: {expected_pairs}. Found: {found_pairs}."
        )


def detect_year_columns_external(df: pd.DataFrame) -> list[int]:
    """Detect year-like column names in an external LCA DataFrame."""
    if "year" in df.columns:
        numeric_years = cast(
            pd.Series,
            pd.to_numeric(pd.Series(df["year"], copy=False), errors="coerce"),
        )
        return sorted({int(value) for value in numeric_years.dropna().astype(int).tolist()})
    years = []
    for col in df.columns:
        try:
            val = int(col)
        except (TypeError, ValueError):
            continue
        if 1900 < val < 2200:
            years.append(val)
    return sorted(set(years))


def validate_external_lca_extra_columns(
    *,
    frame: pd.DataFrame,
    path: Path,
) -> pd.DataFrame:
    """Fail only on extra columns that conflict with canonical internal ownership."""
    out = frame.copy()
    validate_external_lca_reserved_columns(
        columns=out.columns,
        path=path,
        forbidden_columns=_FILE_OWNED_OR_INTERNAL_COLUMNS,
        file_label="External LCA file",
    )
    return out


def validate_external_lca_required_columns(
    *,
    columns: Iterable[object],
    path: Path,
    required_columns: Iterable[str],
    file_label: str,
) -> None:
    """Validate required external LCA columns with one shared message."""
    required = {str(column) for column in required_columns}
    observed = {str(column) for column in columns}
    missing = sorted(required - observed)
    if missing:
        raise ValueError(
            f"{file_label} '{path}' is missing required columns {missing}. "
            f"Required columns are {sorted(required)}."
        )


def validate_external_lca_reserved_columns(
    *,
    columns: Iterable[object],
    path: Path,
    forbidden_columns: Iterable[str],
    file_label: str,
) -> None:
    """Validate columns whose values are derived from filename or package outputs."""
    forbidden = {str(column) for column in forbidden_columns}
    observed = {str(column) for column in columns}
    conflicts = sorted(forbidden & observed)
    if conflicts:
        raise ValueError(f"{file_label} '{path}' contains reserved columns {conflicts}.")


def external_lca_no_requested_year_rows_error(*, path: Path, years: list[int]) -> ValueError:
    """Return the shared missing requested year rows error for external LCA runs."""
    return ValueError(
        f"External LCA Monte Carlo source '{path}' has no rows for requested years {years}."
    )


def validate_external_lca_selector_columns(
    *,
    frame: pd.DataFrame,
    path: Path,
    base_allocate_args: dict[str, Any] | None,
) -> None:
    """Validate staged external LCA selector axes against the requested FU."""
    if base_allocate_args is None:
        return
    required = expected_fu_selector_columns(fu_code=str(base_allocate_args["fu_code"]))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"External LCA file '{path}' must provide the selector columns required by "
            f"fu_code='{base_allocate_args['fu_code']}'. Missing={missing}."
        )
    empty = [
        column
        for column in required
        if pd.Series(frame.loc[:, column], copy=False).isna().any()
        or pd.Series(frame.loc[:, column], copy=False).astype(str).str.strip().eq("").any()
    ]
    if empty:
        raise ValueError(
            f"External LCA file '{path}' contains empty selector values in required "
            f"columns {empty}."
        )
    unexpected: list[str] = []
    for column in SELECTOR_COLUMNS:
        if column in required or column not in frame.columns:
            continue
        series = pd.Series(frame.loc[:, column], copy=False)
        if bool(series.notna().any()) and bool(series.astype(str).str.strip().ne("").any()):
            unexpected.append(column)
    if unexpected:
        raise ValueError(
            f"External LCA file '{path}' contains selector columns outside the requested "
            f"fu_code='{base_allocate_args['fu_code']}'. Expected={list(required)}, "
            f"unexpected={unexpected}."
        )


def normalize_external_lca_deterministic_rows(
    *,
    frame: pd.DataFrame,
    path: Path,
    lcia_method: str,
    base_allocate_args: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalize one deterministic external LCA table to canonical long-form rows."""
    validate_external_lca_contract(frame=frame, path=path, lcia_method=lcia_method)
    normalized = validate_external_lca_extra_columns(
        frame=frame,
        path=path,
    )
    validate_external_lca_selector_columns(
        frame=normalized,
        path=path,
        base_allocate_args=base_allocate_args,
    )
    if {"year", "value"}.issubset(set(normalized.columns)):
        raise ValueError(
            f"Deterministic external LCA file '{path}' must use wide year columns, not long "
            "'year'/'value' rows."
        )
    year_columns = [str(year) for year in detect_year_columns_external(normalized)]
    if not year_columns:
        raise ValueError(f"External LCA file '{path}' must include at least one year column.")
    validate_complete_wide_year_values(
        normalized,
        year_columns=year_columns,
        where=f"External LCA file '{path}'",
    )
    long_frame = melt_requested_year_value_rows(
        normalized,
        requested_years=[int(year) for year in year_columns],
    ).copy()
    long_frame["year"] = long_frame["year"].astype(int)
    long_frame["impact"] = long_frame["impact"].astype(str)
    long_frame["impact_unit"] = long_frame["impact_unit"].astype(str)
    long_frame["value"] = pd.to_numeric(long_frame["value"], errors="raise")
    long_frame = long_frame.loc[
        :,
        _ordered_external_lca_columns(frame=long_frame),
    ]
    return long_frame.reset_index(drop=True)


def external_source_driver_columns(frame: pd.DataFrame) -> list[str]:
    """Return auto-discovered external LCA source driver columns."""
    year_columns = {str(year) for year in detect_year_columns_external(frame)}
    excluded = set(_STANDARD_SOURCE_COLUMNS) | year_columns
    return sorted(column for column in frame.columns if column not in excluded)


def resolve_external_lca_year_assignments(
    *,
    specs: tuple[ExternalLCAFileSpec, ...],
    version_name: str,
    lcia_method: str,
    years: list[int],
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
) -> dict[Path, list[int]]:
    """Resolve requested years for one external LCA file-spec set."""
    family_label = "external LCA"
    routing_specs = tuple(
        ScenarioTaggedFileSpec(path=spec.path, scenario=spec.scenario, years=spec.years)
        for spec in specs
    )
    validate_scenario_inventory(
        specs=routing_specs,
        family_label=family_label,
        item_label=f"LCIA method '{lcia_method}'",
    )
    return _resolve_external_lca_year_assignments(
        specs=routing_specs,
        years=[int(year) for year in years],
        ssp_scenario_options_by_year=ssp_scenario_options_by_year,
        item_label=f"LCIA method '{lcia_method}'",
        expected_stems=external_lca_expected_stems(
            version_name=version_name,
            lcia_method=lcia_method,
            years=years,
            ssp_scenario_options_by_year=ssp_scenario_options_by_year,
        ),
    )


def _resolve_external_lca_year_assignments(
    *,
    specs: tuple[ScenarioTaggedFileSpec, ...],
    years: list[int],
    ssp_scenario_options_by_year: dict[int, list[str | None]] | None,
    item_label: str,
    expected_stems: list[str],
) -> dict[Path, list[int]]:
    """Assign external LCA years from the LCA inventory and optional scenario filter."""
    assignments: dict[Path, list[int]] = {spec.path: [] for spec in specs}
    expected_suffix = f" Expected stems: {expected_stems}."
    for year in years:
        matching = [spec for spec in specs if int(year) in spec.years]
        if not matching:
            raise ValueError(
                f"external LCA coverage does not include requested year {year} for "
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
        selected = [
            spec
            for spec in matching
            if spec.scenario is not None and (not allowed or spec.scenario in allowed)
        ]
        matched_scenarios = {str(spec.scenario) for spec in selected}
        if allowed and matched_scenarios != {str(value) for value in allowed}:
            raise ValueError(
                "external LCA could not resolve SSP tagged files matching the full allowed "
                f"scenario set for year {year} for {item_label}. "
                f"Allowed scenarios: {sorted(allowed)}.{expected_suffix}"
            )
        for spec in selected:
            assignments[spec.path].append(int(year))
    return {path: year_list for path, year_list in assignments.items() if year_list}
