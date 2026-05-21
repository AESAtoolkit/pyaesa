"""Canonical row shaping for external aSoCC owner-side loaders."""

from collections.abc import Iterable

import pandas as pd
from typing import cast

from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.selectors.scenarios import normalize_ssp_tokens
from pyaesa.shared.selectors.fu_axes import expected_fu_selector_columns
from pyaesa.shared.tabular.wide_tables import melt_requested_year_value_rows

from pyaesa.external_inputs.asocc.schema.contracts import ExternalMethodSelection

_BASE_DETERMINISTIC_ROW_COLUMNS = [
    "year",
    "level",
    "l1_l2_method",
    "l2_method",
    "l1_method",
    "lcia_method",
    "impact",
    ASOCC_SSP_SCENARIO_COLUMN,
    "reference_year",
    "value",
]
_ALL_SELECTOR_COLUMNS = ("r_p", "s_p", "r_c", "r_f")
_OPTIONAL_SELECTOR_COLUMNS = ("reference_year", "impact")
_INTERNAL_OR_FILE_OWNED_COLUMNS = frozenset(
    {
        "level",
        "bucket",
        "l1_l2_method",
        "l2_method",
        "l1_method",
        "lcia_method",
        "ssp_scenario",
        ASOCC_SSP_SCENARIO_COLUMN,
    }
)
_MONTE_CARLO_REQUIRED_COLUMNS = frozenset({"run_index", "year", "value", ASOCC_SSP_SCENARIO_COLUMN})


def external_asocc_deterministic_row_columns(
    *,
    selection: ExternalMethodSelection,
    include_asocc_ssp_scenario: bool,
) -> list[str]:
    """Return the canonical deterministic external aSoCC row columns."""
    selectors = expected_external_selector_columns(fu_code=selection.fu_code)
    columns = [
        *_BASE_DETERMINISTIC_ROW_COLUMNS[:-2],
        *selectors,
        *_BASE_DETERMINISTIC_ROW_COLUMNS[-2:],
    ]
    if include_asocc_ssp_scenario:
        return columns
    return [column for column in columns if column != ASOCC_SSP_SCENARIO_COLUMN]


def external_asocc_render_row_columns(
    *,
    selection: ExternalMethodSelection,
    include_asocc_ssp_scenario: bool,
) -> list[str]:
    """Return the canonical Monte Carlo external aSoCC row columns."""
    return [
        "run_index",
        *external_asocc_deterministic_row_columns(
            selection=selection,
            include_asocc_ssp_scenario=include_asocc_ssp_scenario,
        ),
    ]


def empty_external_asocc_rows(
    *,
    selection: ExternalMethodSelection,
    include_asocc_ssp_scenario: bool = True,
) -> pd.DataFrame:
    """Return the canonical empty external aSoCC deterministic row frame."""
    return pd.DataFrame(
        columns=external_asocc_deterministic_row_columns(
            selection=selection,
            include_asocc_ssp_scenario=include_asocc_ssp_scenario,
        )
    )


def empty_external_asocc_render_rows(
    *,
    selection: ExternalMethodSelection,
    include_asocc_ssp_scenario: bool = True,
) -> pd.DataFrame:
    """Return the canonical empty external aSoCC Monte Carlo row frame."""
    return pd.DataFrame(
        columns=external_asocc_render_row_columns(
            selection=selection,
            include_asocc_ssp_scenario=include_asocc_ssp_scenario,
        )
    )


def validate_external_asocc_extra_columns(
    *,
    frame: pd.DataFrame,
    allow_asocc_ssp_scenario: bool = False,
) -> None:
    """Fail when staged external aSoCC files define pyaesa owned columns."""
    validate_external_asocc_extra_column_names(
        columns=frame.columns,
        allow_asocc_ssp_scenario=allow_asocc_ssp_scenario,
    )


def validate_external_asocc_extra_column_names(
    *,
    columns: Iterable[object],
    allow_asocc_ssp_scenario: bool = False,
    path: object | None = None,
) -> None:
    """Fail when staged external aSoCC columns collide with pyaesa owned axes."""
    forbidden = set(_INTERNAL_OR_FILE_OWNED_COLUMNS)
    if allow_asocc_ssp_scenario:
        forbidden.discard(ASOCC_SSP_SCENARIO_COLUMN)
    conflicts = sorted({str(column) for column in columns}.intersection(forbidden))
    if conflicts:
        path_text = "" if path is None else f" '{path}'"
        raise ValueError(
            f"External aSoCC staged file{path_text} contains reserved columns {conflicts}."
        )


def validate_external_asocc_monte_carlo_column_names(
    *,
    columns: Iterable[object],
    selection: ExternalMethodSelection,
    path: object | None = None,
) -> None:
    """Validate long row external aSoCC Monte Carlo column names."""
    observed = {str(column) for column in columns}
    missing = sorted(_MONTE_CARLO_REQUIRED_COLUMNS - observed)
    if missing:
        path_text = "" if path is None else f" '{path}'"
        raise ValueError(
            f"Monte Carlo external aSoCC file{path_text} must provide long run rows with "
            f"columns {sorted(_MONTE_CARLO_REQUIRED_COLUMNS)}. "
            f"fu_code='{selection.fu_code}', method='{selection.user_label}', missing={missing}."
        )
    validate_external_asocc_extra_column_names(
        columns=columns,
        allow_asocc_ssp_scenario=True,
        path=path,
    )


def expected_external_selector_columns(*, fu_code: str) -> tuple[str, ...]:
    """Return public selector columns required by one external aSoCC FU."""
    return expected_fu_selector_columns(fu_code=fu_code)


def _validate_required_selectors(
    *, frame: pd.DataFrame, selection: ExternalMethodSelection
) -> None:
    required = expected_external_selector_columns(fu_code=selection.fu_code)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            "External aSoCC rows must provide the public selector columns expected "
            f"for fu_code='{selection.fu_code}' and method "
            f"'{selection.user_label}'. Expected={list(required)}, missing={missing}."
        )
    empty = [
        column
        for column in required
        if pd.Series(frame.loc[:, column], copy=False).isna().any()
        or pd.Series(frame.loc[:, column], copy=False).astype(str).str.strip().eq("").any()
    ]
    if empty:
        raise ValueError(
            "External aSoCC public selector columns must be non empty so external rows "
            "can align with pyaesa owned method rows. "
            f"fu_code='{selection.fu_code}', method='{selection.user_label}', empty={empty}."
        )
    unexpected = []
    for column in _ALL_SELECTOR_COLUMNS:
        if column in required or column not in frame.columns:
            continue
        series = pd.Series(frame.loc[:, column], copy=False)
        if bool(series.notna().any()) and bool(series.astype(str).str.strip().ne("").any()):
            unexpected.append(column)
    if unexpected:
        raise ValueError(
            "External aSoCC rows must not provide selector columns outside the requested "
            f"functional unit identity. fu_code='{selection.fu_code}', "
            f"method='{selection.user_label}', expected={list(required)}, "
            f"unexpected={unexpected}."
        )


def _apply_external_asocc_deterministic_row_metadata(
    *,
    frame: pd.DataFrame,
    selection: ExternalMethodSelection,
    lcia_method: str | None,
    ssp_scenario: str | None,
    include_asocc_ssp_scenario_column: bool = True,
) -> pd.DataFrame:
    out = frame.copy()
    _validate_required_selectors(frame=out, selection=selection)
    missing_optional = {
        column: None for column in _OPTIONAL_SELECTOR_COLUMNS if column not in out.columns
    }
    out = out.assign(**missing_optional)
    out["year"] = out["year"].astype(int)
    out["level"] = selection.level
    out["l1_l2_method"] = selection.l1_l2_method
    out["l2_method"] = selection.l2_method
    out["l1_method"] = selection.l1_method
    out["lcia_method"] = lcia_method
    if include_asocc_ssp_scenario_column:
        out[ASOCC_SSP_SCENARIO_COLUMN] = ssp_scenario
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    return out.loc[
        :,
        external_asocc_deterministic_row_columns(
            selection=selection,
            include_asocc_ssp_scenario=include_asocc_ssp_scenario_column,
        ),
    ]


def normalize_external_asocc_wide_rows(
    *,
    frame: pd.DataFrame,
    years: list[int],
    selection: ExternalMethodSelection,
    lcia_method: str | None,
    ssp_scenario: str | None,
    include_asocc_ssp_scenario_column: bool = True,
) -> pd.DataFrame:
    """Normalize one wide deterministic external aSoCC table to canonical rows."""
    validate_external_asocc_extra_columns(frame=frame)
    if {"year", "value"}.issubset(frame.columns):
        raise ValueError(
            "Deterministic external aSoCC files must use wide year columns, not long "
            f"'year'/'value' rows. fu_code='{selection.fu_code}', "
            f"method='{selection.user_label}'."
        )
    melted = melt_requested_year_value_rows(
        frame,
        requested_years=years,
    )
    if melted.empty:
        return empty_external_asocc_rows(
            selection=selection,
            include_asocc_ssp_scenario=include_asocc_ssp_scenario_column,
        )
    return _apply_external_asocc_deterministic_row_metadata(
        frame=melted,
        selection=selection,
        lcia_method=lcia_method,
        ssp_scenario=ssp_scenario,
        include_asocc_ssp_scenario_column=include_asocc_ssp_scenario_column,
    )


def normalize_external_asocc_long_rows(
    *,
    frame: pd.DataFrame,
    selection: ExternalMethodSelection,
    lcia_method: str | None,
    ssp_scenario: str | None,
    include_asocc_ssp_scenario_column: bool = True,
) -> pd.DataFrame:
    """Normalize already unpivoted deterministic external aSoCC rows."""
    validate_external_asocc_extra_columns(frame=frame)
    return _apply_external_asocc_deterministic_row_metadata(
        frame=frame,
        selection=selection,
        lcia_method=lcia_method,
        ssp_scenario=ssp_scenario,
        include_asocc_ssp_scenario_column=include_asocc_ssp_scenario_column,
    )


def normalize_external_asocc_render_rows(
    *,
    frame: pd.DataFrame,
    selection: ExternalMethodSelection,
    lcia_method: str | None,
    ssp_scenario: str | None,
    requested_years: list[int],
    include_asocc_ssp_scenario_column: bool = True,
) -> pd.DataFrame:
    """Normalize one long external aSoCC Monte Carlo table to canonical rows."""
    validate_external_asocc_monte_carlo_column_names(
        columns=frame.columns,
        selection=selection,
    )
    out = frame.copy()
    _validate_required_selectors(frame=out, selection=selection)
    for column in _OPTIONAL_SELECTOR_COLUMNS:
        if column not in out.columns:
            out[column] = None
    run_index = cast(
        pd.Series,
        pd.to_numeric(pd.Series(out.loc[:, "run_index"], copy=False), errors="raise"),
    )
    year = cast(
        pd.Series,
        pd.to_numeric(pd.Series(out.loc[:, "year"], copy=False), errors="raise"),
    )
    out["run_index"] = run_index.astype(int)
    out["year"] = year.astype(int)
    out = out.loc[out["year"].isin([int(year) for year in requested_years])].reset_index(drop=True)
    if out.empty:
        return empty_external_asocc_render_rows(
            selection=selection,
            include_asocc_ssp_scenario=include_asocc_ssp_scenario_column,
        )
    out["level"] = selection.level
    out["l1_l2_method"] = selection.l1_l2_method
    out["l2_method"] = selection.l2_method
    out["l1_method"] = selection.l1_method
    out["lcia_method"] = lcia_method
    if include_asocc_ssp_scenario_column:
        if ssp_scenario is None:
            out[ASOCC_SSP_SCENARIO_COLUMN] = normalize_external_ssp_scenario_series(
                out[ASOCC_SSP_SCENARIO_COLUMN]
            )
        else:
            out[ASOCC_SSP_SCENARIO_COLUMN] = ssp_scenario
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    return out.loc[
        :,
        [
            "run_index",
            *external_asocc_deterministic_row_columns(
                selection=selection,
                include_asocc_ssp_scenario=include_asocc_ssp_scenario_column,
            ),
        ],
    ]


def normalize_external_ssp_scenario_series(series: pd.Series) -> pd.Series:
    """Return normalized row-owned SSP labels for external Monte Carlo rows."""
    raw = pd.Series(series, copy=False)
    text = raw.astype("string").str.strip()
    missing = raw.isna() | text.isna() | text.eq("")
    unique_values = sorted({str(value) for value in text.loc[~missing].dropna().unique().tolist()})
    normalized_by_value = {
        value: normalize_ssp_tokens(
            [value],
            context="External aSoCC asocc_ssp_scenario column",
        )[0]
        for value in unique_values
    }
    normalized = text.map(normalized_by_value).astype(object)
    normalized.loc[missing] = None
    return pd.Series(normalized, index=series.index, dtype="object")
