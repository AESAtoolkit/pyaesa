"""ASR uncertainty numerator denominator row alignment."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyaesa.asr.deterministic.runtime.compute import (
    merge_lca_acc_rows,
    required_match_selectors,
    unit_factors_for_matches,
)


@dataclass(frozen=True)
class ASRAlignment:
    """Vectorized ASR numerator denominator alignment."""

    identity: pd.DataFrame
    acc_positions: np.ndarray
    lca_positions: np.ndarray
    lca_unit_factors: np.ndarray


def build_asr_alignment(
    *,
    acc_identity: pd.DataFrame,
    lca_identity: pd.DataFrame,
    lca_type: str,
) -> ASRAlignment:
    """Return one ASR public identity aligned to ACC and LCA public rows."""
    acc = acc_identity.copy().reset_index(drop=True)
    acc["_asr_eval_id"] = np.arange(len(acc), dtype=np.int64)
    acc["_acc_position"] = acc["public_row_id"].to_numpy(dtype=np.int64)
    acc = acc.drop(columns=["public_row_id"])
    acc["year"] = acc["year"].astype(int).astype(str)
    acc["acc_impact_unit"] = acc["impact_unit"]
    lca = lca_identity.copy().reset_index(drop=True)
    lca["_lca_position"] = lca["public_row_id"].to_numpy(dtype=np.int64)
    lca = lca.drop(columns=["public_row_id"])
    lca["year"] = lca["year"].astype(int).astype(str)
    lca["lca_impact_unit"] = lca["impact_unit"]
    selectors = required_match_selectors(acc)
    matched = merge_lca_acc_rows(
        acc_long=acc,
        lca=lca,
        selectors=selectors,
        lca_type=lca_type,
    ).sort_values("_asr_eval_id", kind="mergesort")
    factors = unit_factors_for_matches(matched)
    identity = _asr_identity_from_matches(matches=matched)
    return ASRAlignment(
        identity=identity,
        acc_positions=matched["_acc_position"].to_numpy(dtype=np.int64),
        lca_positions=matched["_lca_position"].to_numpy(dtype=np.int64),
        lca_unit_factors=factors,
    )


def _asr_identity_from_matches(*, matches: pd.DataFrame) -> pd.DataFrame:
    drop_prefixes = ("_asr_key_",)
    drop = {
        "_asr_eval_id",
        "_acc_position",
        "_lca_position",
        "acc_impact_unit",
        "lca_impact_unit",
        "_asr_lca_ssp",
    }
    out = matches.drop(
        columns=[
            column
            for column in matches.columns
            if column in drop or any(column.startswith(prefix) for prefix in drop_prefixes)
        ],
    ).copy()
    if "lca_ssp_scenario" in out.columns and lca_ssp_column_empty(out):
        out = out.drop(columns=["lca_ssp_scenario"])
    out.insert(0, "public_row_id", np.arange(len(out), dtype=np.int64))
    return out.reset_index(drop=True)


def lca_ssp_column_empty(frame: pd.DataFrame) -> bool:
    """Return whether the external LCA scenario column carries no values."""
    series = pd.Series(frame.loc[:, "lca_ssp_scenario"], copy=False)
    text = series.astype("string").str.strip()
    return bool((series.isna() | text.isna() | text.eq("")).all())
