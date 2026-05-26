"""Published deterministic table loading and writing for disaggregation."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_HISTORICAL_REUSE,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.tabular.contracts import TABULAR_SUFFIX_SET
from pyaesa.shared.tabular.l2_reuse_years import canonicalize_l2_reuse_year_column
from pyaesa.shared.tabular.wide_tables import (
    detect_year_columns,
    melt_requested_year_value_rows,
    requested_year_columns,
)


@dataclass(frozen=True)
class PartitionSchema:
    """Target-owned partition schema for one persisted output file."""

    relative_parent: Path
    file_stem: str
    id_columns: list[str]
    year_columns: list[str]


_TIME_ROUTE_BRIDGE_COLUMN = "_disaggregation_time_route_bridge"


def _write_table(*, path: Path, frame: pd.DataFrame, output_format: str) -> None:
    """Persist one deterministic output table in the requested format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        frame.to_csv(path, index=False)
        return
    if output_format == "pickle":
        frame.to_pickle(path)
        return
    frame.to_parquet(path, index=False)


def _read_deterministic_table(path: Path) -> pd.DataFrame:
    """Read one deterministic table using its persisted suffix."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".pickle":
        return cast(pd.DataFrame, pd.read_pickle(path))
    return pd.read_parquet(path)


def _output_path(*, root: Path, relative_parent: Path, file_stem: str, output_format: str) -> Path:
    """Return one persisted output path in the requested target partition."""
    suffix = {"csv": ".csv", "pickle": ".pickle", "parquet": ".parquet"}[output_format]
    return root / relative_parent / f"{file_stem}{suffix}"


def _matching_table_paths(*, root: Path, stem_prefix: str) -> list[Path]:
    """Return deterministic output files whose first owned token matches ``stem_prefix``."""
    if not root.exists():
        return []
    matches = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TABULAR_SUFFIX_SET
        and path.stem.split("__", 1)[0] == stem_prefix
    ]
    return sorted(matches)


def _partition_schema_for_path(*, root: Path, path: Path, frame: pd.DataFrame) -> PartitionSchema:
    """Return target-owned partition metadata for one deterministic table."""
    relative = path.relative_to(root)
    year_cols = detect_year_columns(frame)
    id_cols = [column for column in frame.columns if str(column) not in year_cols]
    return PartitionSchema(
        relative_parent=relative.parent,
        file_stem=path.stem,
        id_columns=[str(column) for column in id_cols],
        year_columns=[str(column) for column in year_cols],
    )


def load_partitioned_rows(
    *,
    root: Path,
    stem_prefix: str,
    requested_years: list[int],
    require_requested_coverage: bool,
) -> tuple[pd.DataFrame, dict[tuple[str, str], PartitionSchema]]:
    """Load deterministic wide outputs as long rows for one method stem prefix."""
    frames: list[pd.DataFrame] = []
    schemas: dict[tuple[str, str], PartitionSchema] = {}
    for path in _matching_table_paths(root=root, stem_prefix=stem_prefix):
        raw = _read_deterministic_table(path)
        schema = _partition_schema_for_path(root=root, path=path, frame=raw)
        frame = canonicalize_l2_reuse_year_column(raw, path=path)
        years = requested_year_columns(frame, requested_years=requested_years)
        if not years:
            continue
        rows = melt_requested_year_value_rows(frame, requested_years=requested_years)
        if rows.empty:
            continue
        key = (schema.relative_parent.as_posix(), schema.file_stem)
        schemas[key] = PartitionSchema(
            relative_parent=schema.relative_parent,
            file_stem=schema.file_stem,
            id_columns=[
                str(column) for column in rows.columns if str(column) not in {"year", "value"}
            ],
            year_columns=years,
        )
        if "l2_reuse_year" not in rows.columns:
            rows["l2_reuse_year"] = None
        year_series = pd.Series(pd.to_numeric(rows["year"], errors="raise"), copy=False)
        rows["year"] = year_series.astype(int)
        rows["relative_parent"] = schema.relative_parent.as_posix()
        rows["file_stem"] = schema.file_stem
        frames.append(rows)
    if not frames:
        if require_requested_coverage:
            raise ValueError(
                "Missing required deterministic output files for method stem "
                f"'{stem_prefix}' in {root}."
            )
        return pd.DataFrame(), {}
    return pd.concat(frames, ignore_index=True), schemas


def _require_unique_variants(
    *,
    frame: pd.DataFrame,
    match_keys: list[str],
    label: str,
    allowed_variant_keys: list[str] | None = None,
) -> None:
    """Fail when a broadcast match would require collapsing multiple variants."""
    if frame.empty:
        return
    group_keys = [*match_keys, *(allowed_variant_keys or [])]
    counts = frame.groupby(group_keys, dropna=False).size()
    if not bool((counts > 1).any()):
        return
    duplicates = counts.loc[counts > 1]
    sample = duplicates.index.tolist()[:3]
    raise ValueError(
        f"{label} has multiple incompatible published variants for one target row. "
        f"Sample duplicate keys: {sample}"
    )


def _merge_reference(
    *,
    target: pd.DataFrame,
    reference: pd.DataFrame,
    exact_keys: list[str],
    label: str,
    allowed_variant_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Merge one reference row family onto target rows with optional broadcast rules."""
    target_frame = target.copy()
    ref_frame = reference.copy()
    for optional_column in (ASOCC_SSP_SCENARIO_COLUMN, "l2_reuse_year"):
        if optional_column not in target_frame.columns:
            target_frame[optional_column] = None
        if optional_column not in ref_frame.columns:
            ref_frame[optional_column] = None
    target_frame["_target_row_id"] = range(len(target_frame))
    ref_columns = [column for column in ref_frame.columns if column not in exact_keys]
    ref_renames = {column: f"_ref_{column}" for column in ref_columns}
    merged = target_frame.merge(
        ref_frame.rename(columns=ref_renames),
        on=exact_keys,
        how="left",
        sort=False,
    )
    for optional_column in (ASOCC_SSP_SCENARIO_COLUMN, "l2_reuse_year"):
        ref_column = ref_renames[optional_column]
        row_ids = merged["_target_row_id"]
        target_has_value = merged[optional_column].notna()
        exact = target_has_value & merged[ref_column].eq(merged[optional_column])
        null = merged[ref_column].isna()
        exact_available = exact.groupby(row_ids, sort=False).transform("any")
        null_available = null.groupby(row_ids, sort=False).transform("any")
        keep = (target_has_value & ((exact_available & exact) | (~exact_available & null))) | (
            ~target_has_value & ((null_available & null) | ~null_available)
        )
        kept = merged.loc[keep].copy()
        missing_ids = row_ids.loc[~row_ids.isin(kept["_target_row_id"])].drop_duplicates()
        if not missing_ids.empty:
            missing = (
                merged.loc[merged["_target_row_id"].isin(missing_ids)]
                .drop_duplicates(subset=["_target_row_id"], keep="first")
                .copy()
            )
            for column in ref_renames.values():
                missing[column] = np.nan
            kept = pd.concat([kept, missing], ignore_index=True)
        merged = kept
    # Each source can be valid alone while their optional SSP/reuse axes still
    # leave more than one reference row for the same target row.
    _require_unique_variants(
        frame=merged,
        match_keys=["_target_row_id"],
        label=label,
        allowed_variant_keys=[ref_renames[key] for key in allowed_variant_keys or []],
    )
    output = merged.loc[:, list(target.columns)].copy()
    output[f"{label}_value"] = merged[ref_renames["value"]].to_numpy()
    output[f"{label}_s_p"] = merged[ref_renames["s_p"]].to_numpy()
    bridge = pd.Series(False, index=merged.index)
    if (
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN in reference.columns
        and ASOCC_TIME_ROUTE_PUBLIC_COLUMN in target.columns
    ):
        source_route = merged[ref_renames[ASOCC_TIME_ROUTE_PUBLIC_COLUMN]].astype("string")
        target_route = merged[ASOCC_TIME_ROUTE_PUBLIC_COLUMN].astype("string")
        bridge = bridge | (source_route.notna() & source_route.ne(target_route))
    if _TIME_ROUTE_BRIDGE_COLUMN in reference.columns:
        bridge = bridge | merged[ref_renames[_TIME_ROUTE_BRIDGE_COLUMN]].fillna(False).astype(bool)
    output[f"{label}_time_route_bridge"] = bridge.to_numpy()
    return output.reset_index(drop=True)


def _shared_exact_keys(*, left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """Return exact-match keys shared by two long deterministic row frames."""
    ignored = {
        "value",
        "s_p",
        "relative_parent",
        "file_stem",
        ASOCC_SSP_SCENARIO_COLUMN,
        ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
        "l2_reuse_year",
        _TIME_ROUTE_BRIDGE_COLUMN,
    }
    return [
        column for column in left.columns if column in right.columns and str(column) not in ignored
    ]


def _bridge_time_route_rows(*, target: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add source rows needed when target/source historical reuse starts in different years."""
    required = {ASOCC_TIME_ROUTE_PUBLIC_COLUMN, "l2_reuse_year"}
    if not required.issubset(target.columns) or not required.issubset(reference.columns):
        return reference
    target_reuse = target.loc[
        target[ASOCC_TIME_ROUTE_PUBLIC_COLUMN]
        .astype("string")
        .eq(ASOCC_TIME_ROUTE_HISTORICAL_REUSE)
        & target["l2_reuse_year"].notna()
    ].copy()
    source_reuse = reference.loc[
        reference[ASOCC_TIME_ROUTE_PUBLIC_COLUMN]
        .astype("string")
        .eq(ASOCC_TIME_ROUTE_HISTORICAL_REUSE)
        & reference["l2_reuse_year"].notna()
    ].copy()
    if target_reuse.empty or source_reuse.empty:
        return reference
    for frame in (target_reuse, source_reuse):
        frame["year"] = pd.Series(
            pd.to_numeric(frame["year"], errors="raise"),
            index=frame.index,
        ).astype(int)
        frame["l2_reuse_year"] = pd.Series(
            pd.to_numeric(frame["l2_reuse_year"], errors="raise"),
            index=frame.index,
        ).astype(int)
    match_columns = [
        column for column in _shared_exact_keys(left=target, right=reference) if column != "year"
    ]
    if (
        ASOCC_SSP_SCENARIO_COLUMN in target_reuse.columns
        and ASOCC_SSP_SCENARIO_COLUMN in source_reuse.columns
    ):
        match_columns.append(ASOCC_SSP_SCENARIO_COLUMN)
    key_columns = ["l2_reuse_year", *match_columns]
    target_keys = target_reuse.drop_duplicates(subset=["year", *key_columns])
    source_keys = source_reuse.loc[:, ["year", *key_columns]].drop_duplicates()
    missing = target_keys.merge(
        source_keys.assign(_bridge_existing=True),
        on=["year", *key_columns],
        how="left",
        sort=False,
    )
    missing = missing.loc[missing["_bridge_existing"].isna(), ["year", *key_columns]]
    if missing.empty:
        return reference
    source_years = source_reuse.groupby(key_columns, dropna=False)["year"].transform("min")
    donors = source_reuse.loc[source_reuse["year"].eq(source_years)].drop(columns=["year"])
    additions = missing.merge(donors, on=key_columns, how="inner", sort=False)
    additions.loc[:, _TIME_ROUTE_BRIDGE_COLUMN] = True
    out = reference.copy()
    out.loc[:, _TIME_ROUTE_BRIDGE_COLUMN] = out.get(_TIME_ROUTE_BRIDGE_COLUMN, False)
    return pd.concat([out, additions.loc[:, out.columns]], ignore_index=True)


def _disaggregate_values(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return disaggregated output values and disaggregate ratios for one merged frame."""
    target_values = pd.Series(
        pd.to_numeric(frame["target_value"], errors="raise"),
        copy=False,
    ).to_numpy(dtype="float64")
    aggregated_values = pd.Series(
        pd.to_numeric(frame["ref_aggregated_value"], errors="raise"),
        copy=False,
    ).to_numpy(dtype="float64")
    disaggregate_values = pd.Series(
        pd.to_numeric(frame["ref_disaggregate_value"], errors="raise"),
        copy=False,
    ).to_numpy(dtype="float64")
    missing_aggregated = pd.Series(frame["ref_aggregated_value"], copy=False).isna().to_numpy()
    missing_disaggregate = pd.Series(frame["ref_disaggregate_value"], copy=False).isna().to_numpy()
    if bool((missing_aggregated | missing_disaggregate).any()):
        sample_columns: list[str] = []
        for column in ["year", *frame.columns[:5]]:
            if column not in sample_columns:
                sample_columns.append(column)
        sample = frame.loc[missing_aggregated | missing_disaggregate, sample_columns].head(5)
        sample_values = list(sample.itertuples(index=False, name=None))
        raise ValueError(
            "Missing reference values for required disaggregation key/year. "
            f"Sample row columns: {list(sample.columns)}. Sample row values: {sample_values}"
        )
    aggregated_zero_disaggregate_positive = (aggregated_values == 0.0) & (disaggregate_values > 0.0)
    if bool(aggregated_zero_disaggregate_positive.any()):
        sample = frame.loc[
            aggregated_zero_disaggregate_positive,
            [
                column
                for column in [
                    "year",
                    "relative_parent",
                    "file_stem",
                    ASOCC_SSP_SCENARIO_COLUMN,
                    "l2_reuse_year",
                    "aggregated_sector_label",
                    "s_p",
                    "target_value",
                    "ref_aggregated_value",
                    "ref_disaggregate_value",
                ]
                if column in frame.columns
            ],
        ].head(5)
        raise ValueError(
            "Disaggregation cannot compute disaggregate ratios where the "
            "aggregated reference value is zero and the disaggregate reference "
            "value is positive. "
            f"Sample row columns: {list(sample.columns)}. "
            f"Sample row values: {list(sample.itertuples(index=False, name=None))}"
        )
    aggregated_disaggregate_zero_target_positive = (
        (aggregated_values == 0.0) & (disaggregate_values == 0.0) & (target_values > 0.0)
    )
    if bool(aggregated_disaggregate_zero_target_positive.any()):
        sample = frame.loc[
            aggregated_disaggregate_zero_target_positive,
            [
                column
                for column in [
                    "year",
                    "relative_parent",
                    "file_stem",
                    ASOCC_SSP_SCENARIO_COLUMN,
                    "l2_reuse_year",
                    "aggregated_sector_label",
                    "s_p",
                    "target_value",
                    "ref_aggregated_value",
                    "ref_disaggregate_value",
                ]
                if column in frame.columns
            ],
        ].head(5)
        raise ValueError(
            "Disaggregation cannot allocate a positive aggregated target value "
            "when both aggregated and disaggregate reference values are zero. "
            f"Sample row columns: {list(sample.columns)}. "
            f"Sample row values: {list(sample.itertuples(index=False, name=None))}"
        )
    ratio = np.divide(
        disaggregate_values,
        aggregated_values,
        out=np.zeros_like(disaggregate_values, dtype=float),
        where=(aggregated_values != 0.0),
    )
    output = target_values * ratio
    triple_zero = (target_values == 0.0) & (aggregated_values == 0.0) & (disaggregate_values == 0.0)
    output = np.where(triple_zero, 0.0, output)
    return output, ratio


def disaggregate_rows(
    *,
    target_rows: pd.DataFrame,
    ref_aggregated_rows: pd.DataFrame,
    ref_disaggregate_rows: pd.DataFrame,
    aggregated_sector_by_disaggregate: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the disaggregation equation to one published output family."""
    if target_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    target = target_rows.copy()
    ref_aggregated = ref_aggregated_rows.copy()
    ref_disaggregate = ref_disaggregate_rows.copy()
    target["aggregated_sector_label"] = target["s_p"]
    ref_aggregated["aggregated_sector_label"] = ref_aggregated["s_p"]
    ref_disaggregate["aggregated_sector_label"] = ref_disaggregate["s_p"].map(
        lambda value: aggregated_sector_by_disaggregate.get(str(value))
    )
    if bool(pd.Series(ref_disaggregate["aggregated_sector_label"], copy=False).isna().any()):
        missing = sorted(
            set(
                ref_disaggregate.loc[
                    ref_disaggregate["aggregated_sector_label"].isna(), "s_p"
                ].astype(str)
            )
        )
        raise ValueError(
            "Reference disaggregate rows contain sector labels not declared "
            "in disaggregation_specs: "
            f"{missing}"
        )
    ref_aggregated = _bridge_time_route_rows(target=target, reference=ref_aggregated)
    ref_disaggregate = _bridge_time_route_rows(target=target, reference=ref_disaggregate)
    shared_with_aggregated = _shared_exact_keys(left=target, right=ref_aggregated)
    merged = _merge_reference(
        target=target.rename(columns={"value": "target_value"}),
        reference=ref_aggregated.rename(columns={"value": "value"}),
        exact_keys=shared_with_aggregated,
        label="ref_aggregated",
    )
    shared_with_disaggregate = _shared_exact_keys(left=merged, right=ref_disaggregate)
    merged = _merge_reference(
        target=merged.rename(columns={"target_value": "target_value"}),
        reference=ref_disaggregate.rename(columns={"value": "value"}),
        exact_keys=shared_with_disaggregate,
        label="ref_disaggregate",
        allowed_variant_keys=["s_p"],
    )
    output_values, ratios = _disaggregate_values(frame=merged)
    result = merged.copy()
    result["value"] = output_values
    result["ratio"] = ratios
    result["s_p"] = result["ref_disaggregate_s_p"]
    keep_columns = [
        column for column in target.columns if column not in {"aggregated_sector_label", "value"}
    ]
    result_rows = result.loc[:, [*keep_columns, "value"]].copy()
    audit = result.loc[
        :,
        [
            column
            for column in [
                "relative_parent",
                "file_stem",
                ASOCC_SSP_SCENARIO_COLUMN,
                "l2_reuse_year",
                "year",
                "aggregated_sector_label",
                "s_p",
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
                "target_value",
                "ref_aggregated_value",
                "ref_disaggregate_value",
                "ref_aggregated_time_route_bridge",
                "ref_disaggregate_time_route_bridge",
                "ratio",
                "value",
            ]
            if column in result.columns
        ],
    ].copy()
    return result_rows, audit


def write_partitioned_rows(
    *,
    rows: pd.DataFrame,
    schemas: dict[tuple[str, str], PartitionSchema],
    output_root: Path,
    output_format: str,
) -> list[Path]:
    """Write disaggregated long rows back to target-owned wide output partitions."""
    written: list[Path] = []
    if rows.empty:
        return written
    for (relative_parent, file_stem), schema in sorted(schemas.items()):
        partition_rows = rows.loc[
            rows["relative_parent"].eq(relative_parent) & rows["file_stem"].eq(file_stem)
        ].copy()
        if partition_rows.empty:
            continue
        id_columns = [
            column for column in schema.id_columns if column not in {"relative_parent", "file_stem"}
        ]
        wide = (
            partition_rows.pivot_table(
                index=id_columns,
                columns="year",
                values="value",
                aggfunc="first",
                dropna=False,
            )
            .reset_index()
            .rename_axis(columns=None)
        )
        wide.columns = [
            str(column) if isinstance(column, int) else column for column in wide.columns
        ]
        for year_column in schema.year_columns:
            if year_column not in wide.columns:
                wide[year_column] = np.nan
        ordered = [*id_columns, *schema.year_columns]
        path = _output_path(
            root=output_root,
            relative_parent=Path(relative_parent),
            file_stem=file_stem,
            output_format=output_format,
        )
        _write_table(path=path, frame=wide.loc[:, ordered], output_format=output_format)
        written.append(path)
    return written
