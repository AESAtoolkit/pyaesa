"""Public aSoCC Monte Carlo row schema ownership."""

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)

ASOCC_PUBLIC_VALUE_COLUMN = "asocc"
ASOCC_UNCERTAINTY_CSV_DTYPES: dict[str, str] = {
    "l1_l2_method": "string",
    "l1_method": "string",
    "l2_method": "string",
    "lcia_method": "string",
    "impact": "string",
    ASOCC_SSP_SCENARIO_COLUMN: "string",
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN: "string",
    "r_p": "string",
    "s_p": "string",
    "r_c": "string",
    "r_f": "string",
}
ASOCC_PUBLIC_COLUMN_ORDER: tuple[str, ...] = (
    "run_index",
    "l1_l2_method",
    "l1_method",
    "l2_method",
    "r_c",
    "r_p",
    "r_f",
    "s_p",
    "lcia_method",
    "impact",
    "reference_year",
    "l2_reuse_year",
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
    ASOCC_SSP_SCENARIO_COLUMN,
    "year",
    ASOCC_PUBLIC_VALUE_COLUMN,
)
ASOCC_PUBLIC_IDENTITY_ORDER: tuple[str, ...] = tuple(
    column
    for column in ASOCC_PUBLIC_COLUMN_ORDER
    if column not in {"run_index", ASOCC_PUBLIC_VALUE_COLUMN}
)


def normalize_asocc_public_row_identity(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows with canonical nullable aSoCC public identity columns."""
    out = frame.copy()
    normalize_asocc_public_row_identity_inplace(frame=out)
    return out


def normalize_asocc_public_row_identity_inplace(*, frame: pd.DataFrame) -> None:
    """Normalize nullable aSoCC public identity columns in place."""
    if ASOCC_SSP_SCENARIO_COLUMN not in frame.columns:
        return
    series = pd.Series(frame.loc[:, ASOCC_SSP_SCENARIO_COLUMN], copy=False)
    frame[ASOCC_SSP_SCENARIO_COLUMN] = series.astype("object").where(series.notna(), None)


def expand_rows_to_reference_lcia_axis(
    *,
    rows: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    """Repeat non LCIA rows across the represented public LCIA axis."""
    axis = lcia_public_axis(frame=reference)
    if axis.empty:
        return rows
    expanded, _values = align_asocc_lcia_public_axis(
        frame=rows,
        values=np.empty((0, len(rows)), dtype=np.float64),
        reference_axis=axis,
    )
    return expanded


def lcia_public_axis(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Return represented LCIA method and impact pairs from public aSoCC rows."""
    if "lcia_method" not in frame.columns or "impact" not in frame.columns:
        return pd.DataFrame(columns=["lcia_method", "impact"])
    method = _normalized_optional_text(series=pd.Series(frame.loc[:, "lcia_method"], copy=False))
    impact = _normalized_optional_text(series=pd.Series(frame.loc[:, "impact"], copy=False))
    present = method.ne("") & impact.ne("")
    if not bool(present.any()):
        return pd.DataFrame(columns=["lcia_method", "impact"])
    return (
        pd.DataFrame(
            {
                "lcia_method": method.loc[present].astype(object),
                "impact": impact.loc[present].astype(object),
            }
        )
        .drop_duplicates(ignore_index=True)
        .reset_index(drop=True)
    )


def align_asocc_lcia_public_axis(
    *,
    frame: pd.DataFrame,
    values: np.ndarray,
    reference_axis: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Repeat generic aSoCC rows across the active LCIA public axis."""
    axis = (
        lcia_public_axis(frame=frame)
        if reference_axis is None
        else reference_axis.loc[:, ["lcia_method", "impact"]].drop_duplicates(ignore_index=True)
    )
    if axis.empty:
        return frame, values
    if "lcia_method" not in frame.columns or "impact" not in frame.columns:
        generic = pd.Series(True, index=frame.index, dtype=bool)
    else:
        method = _normalized_optional_text(
            series=pd.Series(frame.loc[:, "lcia_method"], copy=False)
        )
        impact = _normalized_optional_text(series=pd.Series(frame.loc[:, "impact"], copy=False))
        generic = method.eq("") & impact.eq("")
    if not bool(generic.any()):
        return frame, values
    generic_positions = np.flatnonzero(generic.to_numpy(dtype=bool))
    constrained_positions = np.flatnonzero(~generic.to_numpy(dtype=bool))
    repeated_rows = frame.iloc[np.repeat(generic_positions, len(axis))].reset_index(drop=True)
    repeated_rows["lcia_method"] = list(axis["lcia_method"]) * len(generic_positions)
    repeated_rows["impact"] = list(axis["impact"]) * len(generic_positions)
    output = pd.concat(
        [frame.iloc[constrained_positions].reset_index(drop=True), repeated_rows],
        ignore_index=True,
    )
    output_values = np.concatenate(
        [
            values[:, constrained_positions],
            np.repeat(values[:, generic_positions], repeats=len(axis), axis=1),
        ],
        axis=1,
    )
    return output, output_values


def finalize_asocc_public_row_identity(
    *,
    frame: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Return compact public aSoCC row identity with stable row ids."""
    out = normalize_asocc_public_row_identity(frame=frame)
    out = out.drop(columns=["run_index", value_column], errors="ignore")
    out = _drop_private_columns(frame=out)
    out = _drop_all_null_public_columns(frame=out)
    ordered = [column for column in ASOCC_PUBLIC_IDENTITY_ORDER if column in out.columns]
    extra = [column for column in out.columns if column not in ordered]
    identity = out.loc[:, [*ordered, *extra]].reset_index(drop=True).copy()
    identity.insert(0, "public_row_id", range(len(identity)))
    return identity


def _drop_all_null_public_columns(*, frame: pd.DataFrame) -> pd.DataFrame:
    protected = {"run_index", ASOCC_PUBLIC_VALUE_COLUMN}
    drop_columns = [
        column
        for column in frame.columns
        if column not in protected and pd.Series(frame.loc[:, column], copy=False).isna().all()
    ]
    return frame.drop(columns=drop_columns) if drop_columns else frame


def _drop_private_columns(*, frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in frame.columns if not str(column).startswith("_")]
    return frame.loc[:, columns].copy()


def _normalized_optional_text(*, series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()
