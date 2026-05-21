"""LCIA uncertainty source planning for IO-LCA."""

from typing import cast

import numpy as np
import pandas as pd

from pyaesa.io_lca.uncertainty.runtime.models import IOLCAUncertaintyPlan, IOLCAUncertaintyRequest
from pyaesa.shared.lcia.cov_inputs import (
    LCIACoVInputs,
    country_cov_values,
    load_lcia_cov_inputs,
    sector_cov_keys,
    sector_cov_values,
)
from pyaesa.shared.lcia.uncertainty_keys import build_lcia_shared_u_key
from pyaesa.shared.lcia.uncertainty_source import LCIA_SOURCE
from pyaesa.shared.selectors.aggregate_labels import aggregate_selector_label_or_none
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.uncertainty_assessment.request.shared_u import deterministic_shared_u_matrix

_REGION_SELECTOR_COLUMNS = ("r_f", "r_p", "r_c", "r_u")

IO_LCA_LCIA_FORMULA = (
    "lower = value * (1 - cov_value); "
    "upper = value * (1 + cov_value); "
    "sampled_value = lower + u_shared * (upper - lower)"
)


def build_io_lca_lcia_plan(
    *,
    request: IOLCAUncertaintyRequest,
    public_rows: pd.DataFrame,
) -> IOLCAUncertaintyPlan:
    """Return the compact LCIA uncertainty plan for one IO-LCA request."""
    covs = load_lcia_cov_inputs(
        sector_cov_mapping=cast(dict[str, str], request.source_parameters["sector_cov_mapping"]),
        group_reg=request.group_reg,
        group_version=request.group_version,
        aggregate_region_covs=_uses_aggregate_region_covs(request=request),
    )
    public = _canonical_public_rows(request=request, public_rows=public_rows)
    components = _attach_component_drivers(request=request, components=public, covs=covs)
    identity = _public_identity(request=request, public=public)
    lower = components["lca_value"].to_numpy(dtype=np.float64) * (
        1.0 - components["_cov_value"].to_numpy(dtype=np.float64)
    )
    upper = components["lca_value"].to_numpy(dtype=np.float64) * (
        1.0 + components["_cov_value"].to_numpy(dtype=np.float64)
    )
    shared_u_keys = _shared_u_keys(request=request, components=components)
    shared_u_inverse, unique_shared_u_keys = pd.factorize(shared_u_keys, sort=False)
    return IOLCAUncertaintyPlan(
        identity=identity,
        lower_bound=lower,
        upper_bound=upper,
        unique_shared_u_keys=np.asarray(unique_shared_u_keys, dtype=object),
        shared_u_inverse=shared_u_inverse.astype(np.int64, copy=False),
        source_method_rows=_source_method_rows(
            request=request,
            components=components,
        ),
    )


def sample_io_lca_lcia_matrix(
    *,
    plan: IOLCAUncertaintyPlan,
    batch: RunBatch,
    unit_values: np.ndarray | None = None,
) -> np.ndarray:
    """Return sampled IO-LCA values as run by public row matrix."""
    u_values = _shared_u_values(
        plan=plan,
        run_indices=batch.run_indices(),
        unit_values=unit_values,
    )
    components = u_values[:, plan.shared_u_inverse]
    components *= (plan.upper_bound - plan.lower_bound)[None, :]
    components += plan.lower_bound[None, :]
    return components


def _shared_u_values(
    *,
    plan: IOLCAUncertaintyPlan,
    run_indices: np.ndarray,
    unit_values: np.ndarray | None,
) -> np.ndarray:
    if unit_values is None:
        return deterministic_shared_u_matrix(
            shared_u_keys=plan.unique_shared_u_keys,
            run_indices=run_indices,
        )
    values = np.asarray(unit_values, dtype=np.float64)
    return np.broadcast_to(
        values[:, None],
        (len(values), len(plan.unique_shared_u_keys)),
    )


def _canonical_public_rows(
    *,
    request: IOLCAUncertaintyRequest,
    public_rows: pd.DataFrame,
) -> pd.DataFrame:
    sort_columns = _join_columns(request=request, public=public_rows)
    out = public_rows.copy()
    out["year"] = _numeric_series(out, "year").astype("int64")
    out["lca_value"] = _numeric_series(out, "lca_value").astype("float64")
    return out.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)


def _attach_component_drivers(
    *,
    request: IOLCAUncertaintyRequest,
    components: pd.DataFrame,
    covs: LCIACoVInputs,
) -> pd.DataFrame:
    out = components.copy()
    if request.fu_spec.level == "L1":
        axis = _l1_country_axis(request=request)
        key = _series(out, axis).astype(str)
        out["_driver_kind"] = "country"
        out["_driver_key"] = key
        out["_cov_value"] = country_cov_values(covs=covs, country_key=key)
        return out
    sector_label = _series(out, "s_p").astype(str)
    sector_key = sector_cov_keys(covs=covs, sector_label=sector_label)
    out["_driver_kind"] = "sector"
    out["_driver_key"] = sector_key.astype(str)
    out["_cov_value"] = sector_cov_values(covs=covs, sector_key=sector_key)
    return out


def _public_identity(
    *,
    request: IOLCAUncertaintyRequest,
    public: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["lcia_method", "year", "impact", *_public_selector_columns(request, public)]
    columns.append("impact_unit")
    identity = public.loc[:, columns].copy().reset_index(drop=True)
    identity.insert(0, "public_row_id", np.arange(len(identity), dtype=np.int64))
    return identity


def _join_columns(*, request: IOLCAUncertaintyRequest, public: pd.DataFrame) -> list[str]:
    columns = ["lcia_method", "year", "impact"]
    columns.extend(_public_selector_columns(request, public))
    columns.append("impact_unit")
    return columns


def _public_selector_columns(
    request: IOLCAUncertaintyRequest,
    public: pd.DataFrame,
) -> list[str]:
    return [axis for axis in request.fu_spec.selector_axes if axis in public.columns]


def _shared_u_keys(
    *,
    request: IOLCAUncertaintyRequest,
    components: pd.DataFrame,
) -> np.ndarray:
    pairs = pd.MultiIndex.from_frame(components.loc[:, ["_driver_kind", "_driver_key"]])
    codes, unique_pairs = pd.factorize(pairs, sort=False)
    keys = np.array(
        [
            build_lcia_shared_u_key(
                project_name=request.project_name,
                source=request.source,
                group_reg=request.group_reg,
                group_sec=request.group_sec,
                group_version=request.group_version,
                driver_kind=str(kind),
                driver_key=str(key),
            )
            for kind, key in unique_pairs
        ],
        dtype=object,
    )
    return keys[codes]


def _source_method_rows(
    *,
    request: IOLCAUncertaintyRequest,
    components: pd.DataFrame,
) -> pd.DataFrame:
    group_columns = ["lcia_method", "_driver_kind", "_driver_key", "_cov_value"]
    rows = components.loc[:, [*group_columns, "impact", "year"]].drop_duplicates()
    records: list[dict[str, object]] = []
    for key, group in rows.groupby(group_columns, dropna=False, sort=False):
        lcia_method, driver_kind, driver_key, cov_value = key
        records.append(
            {
                "source_component": "io_lca",
                "source_name": LCIA_SOURCE,
                "scope": request.fu_spec.fu_code,
                "lcia_method": str(lcia_method),
                "impact_categories": _joined_values(group, "impact"),
                "year_min": int(group["year"].min()),
                "year_max": int(group["year"].max()),
                "primary_cov_kind": str(driver_kind),
                "primary_cov_key": str(driver_key),
                "primary_cov_value": float(cov_value),
                "distribution": "uniform",
                "shared_random_variable": "driver kind and driver key by run",
                "formula": IO_LCA_LCIA_FORMULA,
                "notes": ("IO-LCA LCIA uncertainty samples deterministic rows"),
            }
        )
    return pd.DataFrame.from_records(records)


def _l1_country_axis(*, request: IOLCAUncertaintyRequest) -> str:
    return "r_f" if request.fu_spec.fu_code == "L1.a" else "r_p"


def _uses_aggregate_region_covs(*, request: IOLCAUncertaintyRequest) -> bool:
    if not request.aggreg_indices:
        return False
    return any(
        column in request.fu_spec.selector_axes
        and aggregate_selector_label_or_none(request.filters.get(column)) is not None
        for column in _REGION_SELECTOR_COLUMNS
    )


def _joined_values(frame: pd.DataFrame, column: str) -> str:
    values = sorted({str(value) for value in frame[column].dropna().tolist()})
    return ";".join(values)


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.Series(frame.loc[:, column], copy=False)


def _numeric_series(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    return cast(pd.Series, pd.to_numeric(_series(frame, column), errors="raise"))
