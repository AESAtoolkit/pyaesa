"""Utility propagation ownership for UNCASExt extension matrices."""

from datetime import datetime
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_dir, ensure_file_parent

from ..io.paths import _get_mrio_calc_log_path
from .common import (
    _get_prepared_uncasext_inputs,
    _set_column_names,
    _set_index_names,
    _write_pickle,
)
from .enacting_metric_clip_log import write_distribution_normalization_log

_ASSERT_ATOL = 1e-5


def _utility_log_name(saved_dir: Path) -> str:
    """Return utility diagnostics log filename namespaced by source/version/year."""
    source_label = saved_dir.parent.parent.name
    version_label = saved_dir.parent.name
    year_label = saved_dir.name
    return f"{source_label}__{version_label}__{year_label}_utility_propag_uncasext_error.log"


def _write_error_log(
    saved_dir: Path,
    exc: BaseException,
    *,
    source_key: str | None = None,
    matrix_version: str | None = None,
) -> None:
    """Append exception details to the MRIO calc log directory."""
    log_name = _utility_log_name(saved_dir)
    resolved_source_key = saved_dir.parent.parent.name if source_key is None else source_key
    resolved_matrix_version = saved_dir.parent.name if matrix_version is None else matrix_version
    log_path = _get_mrio_calc_log_path(
        log_name,
        source_key=resolved_source_key,
        matrix_version=resolved_matrix_version,
    )
    log_path = ensure_file_parent(log_path)
    timestamp = datetime.now().isoformat()
    message = f"[{timestamp}] {type(exc).__name__}: {exc}\n{traceback.format_exc()}\n"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _write_diagnostic_log(
    saved_dir: Path,
    message: str,
    *,
    source_key: str | None = None,
    matrix_version: str | None = None,
) -> None:
    """Append a diagnostic message to the utility propagation log."""
    log_name = _utility_log_name(saved_dir)
    resolved_source_key = saved_dir.parent.parent.name if source_key is None else source_key
    resolved_matrix_version = saved_dir.parent.name if matrix_version is None else matrix_version
    log_path = _get_mrio_calc_log_path(
        log_name,
        source_key=resolved_source_key,
        matrix_version=resolved_matrix_version,
    )
    log_path = ensure_file_parent(log_path)
    timestamp = datetime.now().isoformat()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def _precompute_utility_propag_uncasext(
    *,
    iosys: Any,
    saved_dir: Path,
    refresh: bool,
    source_key: str,
    matrix_version: str | None,
) -> None:
    """Compute and persist utility propagation matrices for one saved MRIO year."""
    log_name = _utility_log_name(saved_dir)
    log_path = _get_mrio_calc_log_path(
        log_name,
        source_key=source_key,
        matrix_version=matrix_version,
    )
    outdir = saved_dir / "utility_propag_uncasext"
    x_to_rc_path = outdir / "x_to_rc.pickle"
    kappa_path = outdir / "kappa.pickle"
    omega_reg_path = outdir / "omega_reg.pickle"

    if not refresh and x_to_rc_path.exists() and kappa_path.exists() and omega_reg_path.exists():
        return

    outdir = ensure_dir(outdir)

    try:
        z = iosys.Z
        leontief_inv = iosys.L
        prepared = _get_prepared_uncasext_inputs(iosys)
        x_vec = prepared.x_vec
        y_fd = prepared.y_fd
        z_reg = prepared.z_reg

        regions = list(y_fd.columns)
        y_fd = y_fd.reindex(columns=regions)
        z_reg = z_reg.reindex(columns=regions)

        x_to_rc = z_reg + _set_column_names(y_fd, ["r_c"])

        _write_pickle(x_to_rc_path, x_to_rc)

        x_np = x_vec.to_numpy(dtype=float)
        y_fd_np = y_fd.to_numpy(dtype=float)
        leontief_np = leontief_inv.to_numpy(dtype=float)
        pi_np = leontief_np @ y_fd_np
        pi_np = np.divide(pi_np, x_np[:, None], out=np.zeros_like(pi_np), where=x_np[:, None] != 0)

        x_to_rc_np = x_to_rc.to_numpy(dtype=float)
        theta_to_rc_np = np.divide(
            y_fd_np,
            x_to_rc_np,
            out=np.zeros_like(y_fd_np),
            where=x_to_rc_np != 0,
        )

        z_np = z.to_numpy(dtype=float, copy=False)

        col_regions = z.columns.get_level_values("region").to_numpy()
        row_regions = z.index.get_level_values("region").to_numpy()

        rc_col_pos = {rc: np.where(col_regions == rc)[0] for rc in regions}
        rc_row_pos = {rc: np.where(row_regions == rc)[0] for rc in regions}

        n = z_np.shape[0]
        num_regions = len(regions)
        kappa_np = np.empty((num_regions * n, num_regions), dtype=float)

        for j, rc in enumerate(regions):
            z_block = z_np[:, rc_col_pos[rc]]
            pi_block = pi_np[rc_row_pos[rc], :]
            denom = x_to_rc_np[:, j]

            norm = np.divide(
                z_block,
                denom[:, None],
                out=np.zeros_like(z_block),
                where=denom[:, None] != 0,
            )

            indirect = norm @ pi_block
            indirect[:, j] += theta_to_rc_np[:, j]
            kappa_np[j * n : (j + 1) * n, :] = indirect

        rc_level = np.repeat(regions, n)
        prod_r = np.tile(z.index.get_level_values("region").to_numpy(), num_regions)
        prod_i = np.tile(z.index.get_level_values("sector").to_numpy(), num_regions)

        kappa_index = pd.MultiIndex.from_arrays(
            [rc_level, prod_r, prod_i],
            names=["r_c", "r_p", "s_p"],
        )

        kappa = pd.DataFrame(
            kappa_np,
            index=kappa_index,
            columns=pd.Index(regions, name="r_f"),
        )

        zero_mask = np.concatenate([(x_to_rc_np[:, j] == 0) for j in range(num_regions)])
        row_sums = kappa_np.sum(axis=1)

        expected_row_sums = np.where(zero_mask, 0.0, 1.0)
        if not np.allclose(row_sums, expected_row_sums, atol=_ASSERT_ATOL, rtol=0.0):
            max_diff = float(np.abs(row_sums - expected_row_sums).max())
            _write_diagnostic_log(
                saved_dir,
                (
                    "kappa row-sum check failed "
                    f"(max_abs_diff={max_diff:.6e}, atol={_ASSERT_ATOL}) "
                    f"log={log_path}"
                ),
                source_key=source_key,
                matrix_version=matrix_version,
            )
            raise RuntimeError(
                "kappa row sums mismatch against expected values "
                "(1 for nonzero rows, 0 for zero rows) "
                f"(max_abs_diff={max_diff:.6e}, atol={_ASSERT_ATOL}). "
                f"See diagnostic log: {log_path}"
            )

        _write_pickle(kappa_path, kappa)

        gva_by_prod = prepared.gva_by_prod
        alpha_np = np.divide(
            gva_by_prod.to_numpy(dtype=float),
            x_np,
            out=np.zeros_like(x_np),
            where=x_np != 0.0,
        )
        omega_reg_raw_np = np.zeros((num_regions, leontief_np.shape[1]), dtype=float)
        for idx, region in enumerate(regions):
            rows = rc_row_pos[region]
            omega_reg_raw_np[idx, :] = (leontief_np[rows, :] * alpha_np[rows, None]).sum(axis=0)
        omega_reg_raw = pd.DataFrame(
            omega_reg_raw_np,
            index=pd.Index(regions, name="r_u"),
            columns=leontief_inv.columns,
        )
        omega_reg_raw = _set_index_names(omega_reg_raw, ["r_u"])
        omega_reg_raw = _set_column_names(omega_reg_raw, ["r_p", "s_p"])

        colsum = omega_reg_raw.sum(axis=0)
        omega_reg = omega_reg_raw.div(colsum.replace(0, np.nan), axis=1).fillna(0.0)
        intermediate_inputs = pd.Series(z_np.sum(axis=0), index=z.columns, copy=False).reindex(
            omega_reg_raw.columns
        )
        producer_output = x_vec.reindex(omega_reg_raw.columns)
        value_added = gva_by_prod.reindex(omega_reg_raw.columns)
        absolute_context = pd.DataFrame(
            {
                "processed_output_abs": producer_output.to_numpy(dtype=float),
                "processed_input_side_value_added_abs": value_added.to_numpy(dtype=float),
                "processed_intermediate_input_total_abs": intermediate_inputs.to_numpy(dtype=float),
                "processed_output_side_value_added_abs": (
                    producer_output - intermediate_inputs
                ).to_numpy(dtype=float),
            },
            index=omega_reg_raw.columns,
        )
        # Clipping can make the raw omega_reg shares stop summing to a valid
        # distribution even though the normalized result remains usable. For a
        # producer pair (r_p, s_p), the raw regional shares sum to alpha^T L.
        # After clipping changes x and/or the clipped factor_inputs.F proxy
        # used for alpha, alpha^T may no longer equal 1^T(I A), so the raw
        # omega_reg column can stop summing to the expected total. Log every
        # such renormalization in the shared clipping CSV so users can see
        # when omega_reg required an explicit adjustment.
        write_distribution_normalization_log(
            before=omega_reg_raw,
            after=omega_reg,
            matrix_name="omega_reg",
            distribution_axis="r_u",
            unit=prepared.clipping_unit,
            source_key=source_key,
            matrix_version=matrix_version,
            saved_dir=saved_dir,
            expected_sum=1.0,
            absolute_context=absolute_context,
        )

        _write_pickle(omega_reg_path, omega_reg)
    except Exception as exc:
        _write_error_log(
            saved_dir,
            exc,
            source_key=source_key,
            matrix_version=matrix_version,
        )
        _write_diagnostic_log(
            saved_dir,
            f"Utility propagation error logged to {log_path}",
            source_key=source_key,
            matrix_version=matrix_version,
        )
        raise
