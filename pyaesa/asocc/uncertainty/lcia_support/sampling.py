"""LCIA interval sampling for deterministic aSoCC owner rows."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    ASOCC_VALUE_COLUMN,
    LoadedAsoccFinalRows,
)
from pyaesa.shared.lcia.cov_inputs import (
    LCIACoVInputs,
    country_cov_values,
    sector_cov_keys,
    sector_cov_values,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.asocc.uncertainty.io.source_methods import SourceMethodRow
from pyaesa.shared.lcia.uncertainty_keys import build_lcia_shared_u_key
from pyaesa.shared.selectors.aggregate_labels import aggregate_selector_label_or_none
from pyaesa.shared.uncertainty_assessment.request.shared_u import deterministic_shared_u_matrix

LCIA_FORMULA = (
    "lower = value * (1 - primary_cov_value) / (1 + reference_cov_value); "
    "upper = value * (1 + primary_cov_value) / (1 - reference_cov_value); "
    "sampled_value = lower + u_shared * (upper - lower)"
)
LCIA_SOURCE_METHOD_GROUP_COLUMNS = (
    "_applied_bucket",
    "_allocation_method",
    "lcia_method",
    "_primary_cov_kind",
    "_primary_cov_key",
    "_primary_cov_value",
    "_reference_cov_kind",
    "_reference_cov_key",
    "_reference_cov_value",
)
LCIA_SOURCE_METHOD_COLUMNS = (*LCIA_SOURCE_METHOD_GROUP_COLUMNS, "impact", "year")


@dataclass(frozen=True)
class LCIASampleBlock:
    """Numeric LCIA interval owner rows."""

    lower_bound: np.ndarray
    upper_bound: np.ndarray
    unique_shared_u_keys: np.ndarray
    shared_u_inverse: np.ndarray


@dataclass(frozen=True)
class LCIASharedUMatrix:
    """Batch random positions for LCIA shared driver keys."""

    key_positions: dict[str, int]
    values: np.ndarray


def attach_lcia_sampling_columns(
    *,
    rows: pd.DataFrame,
    loaded: LoadedAsoccFinalRows,
    covs: LCIACoVInputs,
    applied_bucket: str,
    allocation_column: str,
    weight_axis: str | None,
) -> pd.DataFrame:
    """Attach LCIA interval columns to deterministic owner rows."""
    out = rows.copy()
    interval = _interval_inputs_frame(
        rows=out,
        loaded=loaded,
        covs=covs,
        applied_bucket=applied_bucket,
        weight_axis=weight_axis,
    )
    value = out[ASOCC_VALUE_COLUMN].astype(float)
    lower = (
        value * (1.0 - interval["_primary_cov_value"]) / (1.0 + interval["_reference_cov_value"])
    )
    upper = (
        value * (1.0 + interval["_primary_cov_value"]) / (1.0 - interval["_reference_cov_value"])
    )
    out = pd.concat([out, interval], axis=1)
    out["_lower_bound"] = lower.to_numpy(dtype="float64")
    out["_upper_bound"] = upper.to_numpy(dtype="float64")
    out["_applied_bucket"] = applied_bucket
    out["_allocation_method"] = out[allocation_column].astype(str)
    out["_shared_u_key"] = _shared_u_keys(rows=out, loaded=loaded)
    return out


def lcia_sample_block(*, template: pd.DataFrame) -> LCIASampleBlock:
    """Return compact numeric LCIA interval owner rows."""
    shared_u_keys = template["_shared_u_key"].astype(str).to_numpy()
    shared_u_inverse, unique_shared_u_keys = pd.factorize(shared_u_keys, sort=False)
    return LCIASampleBlock(
        lower_bound=template["_lower_bound"].to_numpy(dtype="float64"),
        upper_bound=template["_upper_bound"].to_numpy(dtype="float64"),
        unique_shared_u_keys=np.asarray(unique_shared_u_keys, dtype=object),
        shared_u_inverse=shared_u_inverse.astype(np.int64, copy=False),
    )


def lcia_source_method_template(*, rows: pd.DataFrame) -> pd.DataFrame:
    """Return the narrow LCIA source method log template for sampled rows."""
    if rows.empty:
        return pd.DataFrame(columns=LCIA_SOURCE_METHOD_COLUMNS)
    return rows.loc[:, LCIA_SOURCE_METHOD_COLUMNS].drop_duplicates(ignore_index=True)


def shared_u_matrix_for_lcia_blocks(
    *,
    blocks: tuple[LCIASampleBlock, ...],
    batch: RunBatch,
) -> LCIASharedUMatrix:
    """Return one shared random matrix for all LCIA blocks in a run batch."""
    keys = np.unique(np.concatenate([block.unique_shared_u_keys for block in blocks]))
    return LCIASharedUMatrix(
        key_positions={str(key): index for index, key in enumerate(keys.tolist())},
        values=_shared_u_by_key(unique_keys=keys, run_indices=batch.run_indices()),
    )


def sample_lcia_block_matrix(
    *,
    block: LCIASampleBlock,
    shared_u: LCIASharedUMatrix,
) -> np.ndarray:
    """Return sampled LCIA values from compact numeric interval rows."""
    positions = np.fromiter(
        (shared_u.key_positions[str(key)] for key in block.unique_shared_u_keys),
        dtype=np.int64,
        count=len(block.unique_shared_u_keys),
    )
    u_by_key = shared_u.values[:, positions]
    out = u_by_key[:, block.shared_u_inverse]
    out *= (block.upper_bound - block.lower_bound)[None, :]
    out += block.lower_bound[None, :]
    return out


def _shared_u_by_key(*, unique_keys: np.ndarray, run_indices: np.ndarray) -> np.ndarray:
    return deterministic_shared_u_matrix(
        shared_u_keys=unique_keys,
        run_indices=run_indices,
    )


def lcia_source_method_rows(
    *,
    loaded: LoadedAsoccFinalRows,
    templates: list[pd.DataFrame],
    source_name: str,
) -> list[SourceMethodRow]:
    """Return compact non run indexed LCIA source method rows."""
    groups: dict[tuple[object, ...], tuple[set[str], int, int]] = {}
    for template in templates:
        if template.empty:
            continue
        work = template.loc[:, LCIA_SOURCE_METHOD_COLUMNS].drop_duplicates(ignore_index=True)
        work["year"] = pd.Series(pd.to_numeric(work.loc[:, "year"], errors="raise")).astype("int64")
        for key, group in work.groupby(
            list(LCIA_SOURCE_METHOD_GROUP_COLUMNS),
            dropna=False,
            sort=False,
        ):
            values = key if isinstance(key, tuple) else (key,)
            impacts = {str(value) for value in group["impact"].loc[group["impact"].notna()]}
            year_min = int(group["year"].min())
            year_max = int(group["year"].max())
            current = groups.setdefault(values, (set[str](), year_min, year_max))
            current[0].update(impacts)
            groups[values] = (current[0], min(current[1], year_min), max(current[2], year_max))
    rows: list[SourceMethodRow] = []
    for values, (impacts, year_min, year_max) in groups.items():
        payload = dict(zip(LCIA_SOURCE_METHOD_GROUP_COLUMNS, values, strict=True))
        rows.append(
            SourceMethodRow(
                source_component="asocc",
                source_name=source_name,
                scope=str(loaded.base_asocc_args["fu_code"]),
                applied_bucket=str(payload["_applied_bucket"]),
                allocation_method=str(payload["_allocation_method"]),
                lcia_method=str(payload["lcia_method"]),
                impact_categories=";".join(sorted(impacts)),
                year_min=year_min,
                year_max=year_max,
                primary_cov_kind=str(payload["_primary_cov_kind"]),
                primary_cov_key=str(payload["_primary_cov_key"]),
                primary_cov_value=float(str(payload["_primary_cov_value"])),
                reference_cov_kind=str(payload["_reference_cov_kind"]),
                reference_cov_key=str(payload["_reference_cov_key"]),
                reference_cov_value=float(str(payload["_reference_cov_value"])),
                distribution="uniform",
                shared_random_variable="driver kind and driver key by run",
                formula=LCIA_FORMULA,
                notes="LCIA uncertainty is sampled at deterministic LCIA owner rows.",
            )
        )
    return rows


def _interval_inputs_frame(
    *,
    rows: pd.DataFrame,
    loaded: LoadedAsoccFinalRows,
    covs: LCIACoVInputs,
    applied_bucket: str,
    weight_axis: str | None,
) -> pd.DataFrame:
    if applied_bucket == "level_1":
        key = _country_keys(rows=rows, loaded=loaded)
        return _cov_frame(
            primary_kind="country",
            primary_key=key,
            primary_value=country_cov_values(covs=covs, country_key=key),
            reference_kind="world",
            reference_key=pd.Series("World", index=rows.index),
            reference_value=pd.Series(covs.world_cov, index=rows.index, dtype="float64"),
        )
    sector_label = _axis_cov_keys(rows=rows, loaded=loaded, column="s_p")
    sector_key = sector_cov_keys(covs=covs, sector_label=sector_label)
    sector_cov = sector_cov_values(covs=covs, sector_key=sector_key)
    if weight_axis is None:
        return _cov_frame(
            primary_kind="sector",
            primary_key=sector_key,
            primary_value=sector_cov,
            reference_kind="world",
            reference_key=pd.Series("World", index=rows.index),
            reference_value=pd.Series(covs.world_cov, index=rows.index, dtype="float64"),
        )
    country_key = _axis_cov_keys(rows=rows, loaded=loaded, column=weight_axis)
    return _cov_frame(
        primary_kind="sector",
        primary_key=sector_key,
        primary_value=sector_cov,
        reference_kind="country",
        reference_key=country_key,
        reference_value=country_cov_values(covs=covs, country_key=country_key),
    )


def _cov_frame(
    *,
    primary_kind: str,
    primary_key: pd.Series,
    primary_value: pd.Series,
    reference_kind: str,
    reference_key: pd.Series,
    reference_value: pd.Series,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_primary_cov_kind": primary_kind,
            "_primary_cov_key": primary_key.astype(str),
            "_primary_cov_value": primary_value.astype("float64"),
            "_reference_cov_kind": reference_kind,
            "_reference_cov_key": reference_key.astype(str),
            "_reference_cov_value": reference_value.astype("float64"),
        },
        index=primary_key.index,
    )


def _country_keys(*, rows: pd.DataFrame, loaded: LoadedAsoccFinalRows) -> pd.Series:
    key = pd.Series(pd.NA, index=rows.index, dtype="object")
    for column in ("r_f", "r_p", "r_c", "r_u"):
        if column in rows.columns:
            missing = key.isna()
            values = _axis_cov_keys(rows=rows, loaded=loaded, column=column)
            key.loc[missing] = values.loc[missing]
    return key.astype(str)


def _axis_cov_keys(
    *,
    rows: pd.DataFrame,
    loaded: LoadedAsoccFinalRows,
    column: str,
) -> pd.Series:
    aggregate_label = _aggregate_axis_label(loaded=loaded, column=column)
    if aggregate_label is not None:
        return pd.Series(aggregate_label, index=rows.index, dtype="object")
    return pd.Series(rows.loc[:, column], copy=False).astype(str)


def _aggregate_axis_label(*, loaded: LoadedAsoccFinalRows, column: str) -> str | None:
    if not bool(loaded.base_asocc_args.get("aggreg_indices")):
        return None
    return aggregate_selector_label_or_none(loaded.base_asocc_args.get(column))


def _shared_u_keys(*, rows: pd.DataFrame, loaded: LoadedAsoccFinalRows) -> list[str]:
    driver_pairs = pd.MultiIndex.from_frame(
        rows.loc[:, ["_primary_cov_kind", "_primary_cov_key"]].astype(str)
    )
    codes, unique_pairs = pd.factorize(driver_pairs, sort=False)
    unique_keys = np.array(
        [
            build_lcia_shared_u_key(
                project_name=str(loaded.base_asocc_args["project_name"]),
                source=str(loaded.base_asocc_args["source"]),
                group_reg=bool(loaded.base_asocc_args["group_reg"]),
                group_sec=bool(loaded.base_asocc_args["group_sec"]),
                group_version=loaded.base_asocc_args["group_version"],
                driver_kind=str(kind),
                driver_key=str(key),
            )
            for kind, key in unique_pairs
        ],
        dtype=object,
    )
    return unique_keys[codes].tolist()
