"""Processed MRIO input loaders for L2*b validation expectations."""

from pathlib import Path
from typing import NamedTuple, TypedDict, cast

import numpy as np
import pandas as pd

from pyaesa.process.mrios.utils.io.paths import _get_year_saved_dir


class L2StarBTotals(TypedDict):
    """Deterministic totals used by L2*b overlap checks."""

    fd_w: float
    gva_w: float
    x_w: float
    overlap_fd_by_rf: pd.Series
    overlap_gvaa_by_ru: pd.Series


_L2_STAR_B_TOTALS_CACHE: dict[tuple[str, int, str | None], L2StarBTotals] = {}


def _find_first_pickle(year_dir: Path, filename: str) -> Path:
    """Return the shortest matching pickle path under one processed MRIO year dir."""
    hits = list(year_dir.rglob(filename))
    if not hits:
        raise FileNotFoundError(f"Cannot find '{filename}' under {year_dir}")
    return sorted(hits, key=lambda p: len(str(p)))[0]


def _numeric_series(value: object) -> pd.Series:
    """Return numeric float Series from array like payload."""
    return pd.Series(pd.to_numeric(pd.Series(value), errors="coerce"), copy=False).astype(float)


def _as_frame(value: object, *, label: str) -> pd.DataFrame:
    """Return DataFrame payload for matrix computations."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    raise TypeError(f"{label} must be a pandas DataFrame or Series payload.")


def _compute_fd_overlap(*, fd_rf: object, x_to_rc: pd.DataFrame, kappa: pd.DataFrame) -> pd.Series:
    """Return UT(FDa) overlap by ``r_f``."""
    fd_denom = _numeric_series(fd_rf)
    fd_denom.index = fd_denom.index.map(str)

    fd_blocks: list[pd.DataFrame] = []
    for rc in x_to_rc.columns:
        k_block = kappa.xs(rc, level="r_c")
        if isinstance(k_block, pd.Series):
            k_block = k_block.to_frame()
        contrib = k_block.mul(x_to_rc[rc], axis=0)
        if isinstance(contrib, pd.Series):
            contrib = contrib.to_frame()
        fd_blocks.append(contrib)
    contrib_all = cast(
        pd.DataFrame,
        pd.concat(fd_blocks, axis=0).groupby(level=0).sum(min_count=1),
    )
    contrib_sum = cast(pd.Series, contrib_all.sum(axis=0, min_count=1))
    fd_numer = pd.Series(
        pd.to_numeric(contrib_sum, errors="coerce"),
        copy=False,
    ).astype(float)
    fd_numer.index = fd_numer.index.map(str)
    return fd_numer.div(fd_denom.replace(0.0, np.nan)).astype(float)


def _compute_gvaa_overlap(
    *,
    gva_rp: object,
    x_to_rc: pd.DataFrame,
    omega_reg: pd.DataFrame,
) -> pd.Series:
    """Return UT(GVAa) overlap by ``r_u``."""
    gva_denom = _numeric_series(gva_rp)
    gva_denom.index = gva_denom.index.map(str)

    x_vec = pd.Series(
        pd.to_numeric(x_to_rc.sum(axis=1, min_count=1), errors="coerce"),
        copy=False,
    ).astype(float)
    omega_weighted = omega_reg.mul(x_vec, axis=1)
    gva_numer = pd.Series(
        pd.to_numeric(omega_weighted.sum(axis=1, min_count=1), errors="coerce"),
        copy=False,
    ).astype(float)
    gva_numer.index = gva_numer.index.map(str)
    return gva_numer.div(gva_denom.replace(0.0, np.nan)).astype(float)


class _L2StarBInputs(NamedTuple):
    """Raw processed MRIO payloads needed for L2*b expected totals."""

    fd_rf: object
    gva_rp: object
    x_to_rc: pd.DataFrame
    kappa: pd.DataFrame
    omega_reg: pd.DataFrame


def _load_inputs(year_dir: Path) -> _L2StarBInputs:
    """Load and normalize required processed MRIO payloads."""
    return _L2StarBInputs(
        fd_rf=pd.read_pickle(_find_first_pickle(year_dir, "fd_rf.pickle")),
        gva_rp=pd.read_pickle(_find_first_pickle(year_dir, "gva_rp.pickle")),
        x_to_rc=_as_frame(
            pd.read_pickle(_find_first_pickle(year_dir, "x_to_rc.pickle")),
            label="x_to_rc",
        ),
        kappa=_as_frame(
            pd.read_pickle(_find_first_pickle(year_dir, "kappa.pickle")),
            label="kappa",
        ),
        omega_reg=_as_frame(
            pd.read_pickle(_find_first_pickle(year_dir, "omega_reg.pickle")),
            label="omega_reg",
        ),
    )


def load_l2_star_b_totals(
    source: str,
    year: int,
    *,
    matrix_version: str | None = None,
) -> L2StarBTotals:
    """Load deterministic totals/overlaps from processed MRIO metrics."""

    key = (source, year, matrix_version)
    if key in _L2_STAR_B_TOTALS_CACHE:
        return _L2_STAR_B_TOTALS_CACHE[key]

    year_dir = _get_year_saved_dir(source, year, matrix_version=matrix_version)
    inputs = _load_inputs(year_dir)
    fd_w = float(pd.Series(inputs.fd_rf).sum(min_count=1))
    gva_w = float(pd.Series(inputs.gva_rp).sum(min_count=1))
    x_w = float(np.nansum(inputs.x_to_rc.to_numpy()))
    overlap_fd_by_rf = _compute_fd_overlap(
        fd_rf=inputs.fd_rf,
        x_to_rc=inputs.x_to_rc,
        kappa=inputs.kappa,
    )
    overlap_gvaa_by_ru = _compute_gvaa_overlap(
        gva_rp=inputs.gva_rp,
        x_to_rc=inputs.x_to_rc,
        omega_reg=inputs.omega_reg,
    )

    out: L2StarBTotals = {
        "fd_w": fd_w,
        "gva_w": gva_w,
        "x_w": x_w,
        "overlap_fd_by_rf": overlap_fd_by_rf,
        "overlap_gvaa_by_ru": overlap_gvaa_by_ru,
    }
    _L2_STAR_B_TOTALS_CACHE[key] = out
    return out
