"""Parse the public AR6 archive exposed by the IIASA files endpoint."""

import io
import zipfile
from typing import cast

import pandas as pd

AR6_WORLD_ARCHIVE_DESCRIPTION = "AR6_Scenarios_Database_World_v1.1"
AR6_WORLD_ARCHIVE_FILENAME_SUFFIX = "AR6_Scenarios_Database_World_v1.1.csv.zip"
AR6_WORLD_TIMESERIES_MEMBER = "AR6_Scenarios_Database_World_v1.1.csv"
AR6_WORLD_METADATA_MEMBER = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
AR6_WORLD_METADATA_SHEET = "meta"
_ARCHIVE_CHUNK_SIZE = 5000


def _require_archive_member(archive: zipfile.ZipFile, member_name: str) -> bytes:
    """Return one required archive member and fail fast when it is absent."""
    if member_name not in archive.namelist():
        raise RuntimeError(
            f"The AR6 public archive did not contain the required member '{member_name}'."
        )
    return archive.read(member_name)


def _read_filtered_wide_timeseries(
    archive: zipfile.ZipFile,
    *,
    variables: list[str],
    region: str,
) -> pd.DataFrame:
    """Read only the requested AR6 world timeseries rows from the public archive."""
    csv_bytes = _require_archive_member(archive, AR6_WORLD_TIMESERIES_MEMBER)
    base_columns = {"Model", "Scenario", "Region", "Variable", "Unit"}
    filtered_chunks: list[pd.DataFrame] = []
    with io.BytesIO(csv_bytes) as csv_buffer:
        reader = pd.read_csv(
            csv_buffer,
            usecols=lambda column: column in base_columns or str(column).isdigit(),
            chunksize=_ARCHIVE_CHUNK_SIZE,
        )
        for chunk_df in reader:
            retained = chunk_df.loc[
                (chunk_df["Region"] == region) & (chunk_df["Variable"].isin(variables))
            ].copy()
            if not retained.empty:
                filtered_chunks.append(retained)
    if not filtered_chunks:
        raise RuntimeError(
            "The AR6 public archive did not contain any rows for the requested region and "
            "variable scope. "
            f"Archive member='{AR6_WORLD_TIMESERIES_MEMBER}', region='{region}', "
            f"variables={variables}."
        )
    return pd.concat(filtered_chunks, ignore_index=True)


def _wide_to_long_timeseries(filtered_wide_df: pd.DataFrame) -> pd.DataFrame:
    """Return the long timeseries contract used by ``download_ar6``."""
    year_columns = [column for column in filtered_wide_df.columns if column.isdigit()]
    long_df = filtered_wide_df.melt(
        id_vars=["Model", "Scenario", "Variable", "Unit", "Region"],
        value_vars=year_columns,
        var_name="year",
        value_name="value",
    )
    long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)
    renamed = long_df.rename(
        columns={
            "Model": "model",
            "Scenario": "scenario",
            "Variable": "variable",
            "Unit": "unit",
            "Region": "region",
        }
    )
    year_series = cast(pd.Series, pd.to_numeric(renamed["year"], errors="raise"))
    renamed["year"] = year_series.astype(int)
    return renamed.loc[:, ["model", "scenario", "variable", "unit", "region", "year", "value"]]


def _read_metadata_table(
    archive: zipfile.ZipFile,
    *,
    meta_columns: list[str],
) -> pd.DataFrame:
    """Return the scenario metadata contract used by ``download_ar6``."""
    metadata_bytes = _require_archive_member(archive, AR6_WORLD_METADATA_MEMBER)
    metadata_df = pd.read_excel(
        io.BytesIO(metadata_bytes),
        sheet_name=AR6_WORLD_METADATA_SHEET,
        engine="calamine",
    )
    required_identity_columns = ["Model", "Scenario"]
    missing_identity = [col for col in required_identity_columns if col not in metadata_df.columns]
    if missing_identity:
        raise RuntimeError(
            "The AR6 public metadata table is missing required identity columns: "
            f"{missing_identity}"
        )
    retained_meta_columns = [col for col in meta_columns if col in metadata_df]
    selected_columns = required_identity_columns + retained_meta_columns
    selected_df = metadata_df.loc[:, selected_columns].copy()
    selected_df = selected_df.rename(columns={"Model": "model", "Scenario": "scenario"})
    selected_df = selected_df.drop_duplicates(subset=["model", "scenario"], keep="first")
    selected_df = selected_df.set_index(["model", "scenario"])
    for meta_col in meta_columns:
        if meta_col not in selected_df.columns:
            selected_df[meta_col] = pd.NA
    return selected_df.loc[:, meta_columns]


def load_ar6_public_archive_data(
    archive_bytes: bytes,
    *,
    variables: list[str],
    region: str,
    meta_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return AR6 public timeseries and metadata parsed from the World archive."""
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        filtered_wide_df = _read_filtered_wide_timeseries(
            archive,
            variables=variables,
            region=region,
        )
        data_df = _wide_to_long_timeseries(filtered_wide_df)
        meta_df = _read_metadata_table(archive, meta_columns=meta_columns)
    return data_df, meta_df
