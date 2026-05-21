"""I/O ownership for deterministic carrying-capacity outputs."""

from pathlib import Path
from typing import cast

import pandas as pd

from pyaesa.ar6_cc.deterministic.request.contracts import (
    CC_FLOW_NET,
    CC_FLOW_POSITIVE,
)
from pyaesa.ar6_cc.deterministic.io.paths import (
    get_subset_csv_path,
)
from pyaesa.process.ar6.utils.io import (
    contracts as ar6_contracts,
)
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent, write_via_atomic_temp
from pyaesa.shared.selectors.scenarios import normalize_ssp_token, normalize_ssp_tokens
from pyaesa.shared.tabular.contracts import normalize_tabular_output_format

_CC_ID_COLUMNS = [
    "cc_model",
    "cc_scenario",
    "cc_category",
    "ssp_scenario",
    "cc_flow",
    "cc_variable",
    "impact_unit",
]


def _cc_year_columns(frame: pd.DataFrame) -> list[int]:
    """Return canonical integer year columns from one deterministic CC table."""
    return sorted(
        {
            int(column)
            for column in frame.columns
            if isinstance(column, int) or (isinstance(column, str) and column.isdigit())
        }
    )


def _require_unique_cc_identity(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Return one CC table with unique deterministic row identity."""
    duplicated = frame.duplicated(subset=_CC_ID_COLUMNS, keep=False)
    if not bool(duplicated.any()):
        return frame
    duplicate_rows = frame.loc[duplicated, _CC_ID_COLUMNS].drop_duplicates().reset_index(drop=True)
    raise ValueError(
        f"{context} requires unique deterministic CC rows per identity "
        f"{_CC_ID_COLUMNS}. Duplicate identities: {duplicate_rows.to_dict(orient='records')}"
    )


def _require_cc_output_contract(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Return one deterministic CC table that satisfies the canonical output contract."""
    missing = [column for column in _CC_ID_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}.")
    if frame.empty:
        return frame
    for column in _CC_ID_COLUMNS:
        series = pd.Series(frame.loc[:, column], copy=False)
        empty = [
            value is None or pd.isna(value) or not str(value).strip() for value in series.tolist()
        ]
        if any(empty):
            raise ValueError(f"{context} has empty values in required column '{column}'.")
    return _require_unique_cc_identity(frame, context=context)


def read_harmonized_pathways(
    *,
    processed_dir: Path,
    harmonization: bool,
) -> pd.DataFrame:
    """Read the harmonized AR6 pathway DataFrame from the processed workbook.

    Returns:
        DataFrame with MultiIndex (model, scenario, variable) and metadata
        columns (Category, Ssp_family, unit, region, ...) plus year columns.
    """
    workbook_name = ar6_contracts.processed_workbook_name(harmonization=harmonization)
    workbook_path = processed_dir / workbook_name
    sheet = ar6_contracts.final_pathways_sheet_name(harmonization=harmonization)
    with pd.ExcelFile(workbook_path, engine="calamine") as xl:
        df = pd.read_excel(xl, sheet_name=sheet, index_col=[0, 1, 2])
    return df


def filter_pathways(
    df: pd.DataFrame,
    *,
    variable: str,
    category: list[str] | None,
    ssp_scenario: list[str] | None,
    subset_version: str | None,
    processed_dir: Path,
) -> pd.DataFrame:
    """Filter harmonized pathways to the requested CC scope.

    Args:
        df: Full harmonized pathway DataFrame with MultiIndex
            (model, scenario, variable).
        variable: AR6 variable to select.
        category: Category filter (e.g. ["C1", "C2"]). None means all.
        ssp_scenario: SSP filter (e.g. ["SSP1", "SSP2"]). None means all.
        subset_version: Model-scenario subset CSV version name.
        processed_dir: Processed directory where subset CSVs live.

    Returns:
        Filtered DataFrame.
    """
    # The processed AR6 workbook arrive as a fragmented wide frame.
    # Copy once before resetting the index so later selector work stays fast
    # and does not emit fragmentation warnings on large study windows.
    df_reset = df.copy().reset_index()
    mask = df_reset["variable"] == variable
    if mask.sum() == 0:
        available = sorted(set(df_reset["variable"]))
        raise ValueError(
            f"Variable '{variable}' not found in processed AR6 workbook. "
            f"Processed directory: {processed_dir}. Available variables: {available}."
        )
    if category is not None:
        mask = mask & df_reset["Category"].isin(category)
    if ssp_scenario is not None:
        ssp_labels = set(normalize_ssp_tokens(ssp_scenario))
        ssp_series = pd.Series(
            [
                normalize_ssp_token(value, context="Processed AR6 Ssp_family")
                for value in df_reset["Ssp_family"].tolist()
            ],
            index=df_reset.index,
            copy=False,
        )
        mask = mask & ssp_series.isin(sorted(ssp_labels))
    if subset_version is not None:
        subset_df = _read_subset_csv(processed_dir, subset_version)
        subset_pairs = pd.MultiIndex.from_frame(subset_df.loc[:, ["model", "scenario"]])
        pathway_pairs = pd.MultiIndex.from_frame(df_reset.loc[:, ["model", "scenario"]])
        mask = mask & pathway_pairs.isin(subset_pairs)
    filtered = df_reset.loc[mask]
    if filtered.empty:
        raise ValueError(
            "No AR6 pathways remain after filtering deterministic carrying capacity inputs. "
            f"variable='{variable}', category={category}, ssp_scenario={ssp_scenario}, "
            f"subset_version={subset_version}, processed_dir='{processed_dir}'."
        )
    return filtered


def _read_subset_csv(processed_dir: Path, subset_version: str) -> pd.DataFrame:
    """Read and validate a model-scenario subset CSV."""
    subset_path = get_subset_csv_path(processed_dir, subset_version)
    if not subset_path.exists():
        raise FileNotFoundError(
            f"Model-scenario subset CSV not found: {subset_path}. "
            "The subset must be staged under the processed AR6 scope that matches the "
            "current study period and harmonization settings. Generate a template with "
            "process_ar6() for that scope and rename it to "
            f"model_scenario_subset__{subset_version}.csv."
        )
    subset_df = pd.read_csv(
        subset_path, comment="#", dtype={"model": "string", "scenario": "string"}
    )
    required_cols = {"model", "scenario"}
    missing = required_cols - set(subset_df.columns)
    if missing:
        raise ValueError(f"Subset CSV {subset_path} is missing required columns: {missing}")
    return subset_df


def build_cc_table(
    filtered_df: pd.DataFrame,
    years: list[int],
    *,
    cc_flow: str,
    cc_variable: str,
    sign: float = 1.0,
) -> pd.DataFrame:
    """Build the final carrying capacity table from filtered pathways.

    Returns:
        DataFrame with columns: cc_model, cc_scenario, cc_category, ssp_scenario,
        impact_unit, plus one column per requested year.
    """
    requested_years = sorted({int(year) for year in years})
    year_pairs = sorted(
        [
            (int(c), c)
            for c in filtered_df.columns
            if isinstance(c, (int, float)) and int(c) in requested_years
        ],
        key=lambda item: item[0],
    )
    year_cols = [year for year, _source_column in year_pairs]
    source_year_cols = [source_column for _year, source_column in year_pairs]
    selected_columns: list[str | int | float] = [
        "model",
        "scenario",
        "Category",
        "Ssp_family",
        "unit",
        *source_year_cols,
    ]
    selected = filtered_df.loc[:, selected_columns].copy()
    result_columns: dict[str | int, object] = {
        "cc_model": selected["model"].to_numpy(copy=False),
        "cc_scenario": selected["scenario"].to_numpy(copy=False),
        "cc_category": selected["Category"].to_numpy(copy=False),
        "ssp_scenario": [
            normalize_ssp_token(value, context="Processed AR6 Ssp_family")
            for value in selected["Ssp_family"].tolist()
        ],
        "cc_flow": str(cc_flow),
        "cc_variable": str(cc_variable),
        "impact_unit": selected["unit"].to_numpy(copy=False),
    }
    for year_col, source_column in year_pairs:
        result_columns[year_col] = selected[source_column].to_numpy(dtype=float, copy=False) * sign
    result = pd.DataFrame(result_columns)
    col_order: list[str | int] = [
        "cc_model",
        "cc_scenario",
        "cc_category",
        "ssp_scenario",
        "cc_flow",
        "cc_variable",
        "impact_unit",
        *year_cols,
    ]
    result = result.loc[:, col_order]
    result = _require_unique_cc_identity(
        result,
        context="Deterministic AR6 CC table construction",
    )
    result = result.sort_values(
        ["cc_category", "ssp_scenario", "cc_flow", "cc_model", "cc_scenario"],
        kind="stable",
    ).reset_index(drop=True)
    return result


def filter_to_denominator_cc_rows(cc_table: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic AR6 CC rows valid for aCC and ASR denominator use."""
    mask = cc_table["cc_flow"].isin([CC_FLOW_NET, CC_FLOW_POSITIVE])
    return cc_table.loc[mask].reset_index(drop=True)


def select_cc_year_columns(cc_table: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Return one deterministic CC table with only the selected year columns."""
    requested = [int(year) for year in years]
    return cc_table.loc[:, [*_CC_ID_COLUMNS, *requested]].copy()


def write_cc_output(
    cc_table: pd.DataFrame,
    output_file: Path,
    output_format: str,
) -> None:
    """Write the carrying capacity table in the requested format."""
    normalize_tabular_output_format(output_format)
    output_file = ensure_file_parent(output_file)
    _write_deterministic_table(path=output_file, frame=cc_table)


def merge_cc_tables(*, existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical union of existing and incoming deterministic CC rows."""
    existing = _require_cc_output_contract(
        _normalize_cc_year_columns(existing),
        context="Deterministic AR6 CC merge existing input",
    )
    incoming = _require_cc_output_contract(
        _normalize_cc_year_columns(incoming),
        context="Deterministic AR6 CC merge incoming input",
    )
    existing_indexed = existing.set_index(_CC_ID_COLUMNS)
    incoming_indexed = incoming.set_index(_CC_ID_COLUMNS)
    merged = incoming_indexed.combine_first(existing_indexed).reset_index()
    year_columns = _cc_year_columns(merged)
    ordered = merged.loc[:, [*_CC_ID_COLUMNS, *year_columns]]
    ordered = _require_cc_output_contract(ordered, context="Deterministic AR6 CC merge result")
    return ordered.sort_values(
        ["cc_category", "ssp_scenario", "cc_flow", "cc_model", "cc_scenario"],
        kind="stable",
    ).reset_index(drop=True)


def read_cc_output(
    *,
    output_file: Path,
    output_format: str,
) -> pd.DataFrame:
    """Read one deterministic AR6 CC output table."""
    fmt = normalize_tabular_output_format(output_format)
    out = _read_deterministic_table(path=output_file, output_format=fmt)
    out = _require_cc_output_contract(
        out,
        context=f"Deterministic AR6 CC output table at {output_file}",
    )
    return out.reset_index(drop=True)


def cc_output_exists(*, output_file: Path) -> bool:
    """Return whether the deterministic AR6 CC output table exists."""
    return output_file.exists() and output_file.is_file()


def _write_deterministic_table(*, path: Path, frame: pd.DataFrame) -> None:
    """Write one deterministic AR6 CC table in a package tabular format."""
    path = ensure_file_parent(path)
    output_format = normalize_tabular_output_format(path.suffix.lower().removeprefix("."))
    if output_format == "csv":
        write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_csv(tmp_path, index=False))
        return
    if output_format == "parquet":
        write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_parquet(tmp_path, index=False))
        return
    write_via_atomic_temp(path, writer=lambda tmp_path: frame.to_pickle(tmp_path))


def _read_deterministic_table(*, path: Path, output_format: str) -> pd.DataFrame:
    """Read one deterministic AR6 CC table in a package tabular format."""
    if output_format == "csv":
        return pd.read_csv(path)
    elif output_format == "parquet":
        return pd.read_parquet(path)
    return cast(pd.DataFrame, pd.read_pickle(path))


def _normalize_cc_year_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one CC table with canonical integer year column labels."""
    renamed = frame.copy()
    renamed.columns = [
        int(column) if isinstance(column, str) and column.isdigit() else column
        for column in renamed.columns
    ]
    return renamed
