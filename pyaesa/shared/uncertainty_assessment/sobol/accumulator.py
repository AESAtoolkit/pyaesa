"""Streaming Sobol estimator accumulation."""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from pyaesa.shared.runtime.memory import memory_bounded_rows


@dataclass(frozen=True)
class SobolIndexEstimate:
    """Sobol estimates and confidence precision by source and output."""

    s1: np.ndarray
    st: np.ndarray
    variance: np.ndarray
    s1_confidence_half_width: np.ndarray
    st_confidence_half_width: np.ndarray
    s1_resamples: np.ndarray
    st_resamples: np.ndarray


class SobolMomentAccumulator:
    """Accumulate Saltelli estimator moments without retaining model outputs."""

    def __init__(
        self,
        *,
        output_count: int,
        dimension_count: int,
        confidence_resamples: int,
    ) -> None:
        self._output_count = output_count
        self._dimension_count = dimension_count
        self._confidence_resamples = confidence_resamples
        self._variance_count = np.zeros(output_count, dtype=np.float64)
        self._variance_sum = np.zeros(output_count, dtype=np.float64)
        self._variance_sumsq = np.zeros(output_count, dtype=np.float64)
        self._center_count = np.zeros(output_count, dtype=np.float64)
        self._center_sum = np.zeros(output_count, dtype=np.float64)
        self._s1_count = np.zeros((dimension_count, output_count), dtype=np.float64)
        self._s1_b_delta_sum = np.zeros_like(self._s1_count)
        self._s1_delta_sum = np.zeros_like(self._s1_count)
        self._st_count = np.zeros_like(self._s1_count)
        self._st_sum = np.zeros_like(self._s1_count)
        shape = (self._confidence_resamples, self._output_count)
        dim_shape = (self._confidence_resamples, self._dimension_count, self._output_count)
        self._boot_variance_count = np.zeros(shape, dtype=np.float64)
        self._boot_variance_sum = np.zeros(shape, dtype=np.float64)
        self._boot_variance_sumsq = np.zeros(shape, dtype=np.float64)
        self._boot_center_count = np.zeros(shape, dtype=np.float64)
        self._boot_center_sum = np.zeros(shape, dtype=np.float64)
        self._boot_s1_count = np.zeros(dim_shape, dtype=np.float64)
        self._boot_s1_b_delta_sum = np.zeros(dim_shape, dtype=np.float64)
        self._boot_s1_delta_sum = np.zeros(dim_shape, dtype=np.float64)
        self._boot_st_count = np.zeros(dim_shape, dtype=np.float64)
        self._boot_st_sum = np.zeros(dim_shape, dtype=np.float64)

    def add(
        self,
        *,
        a_values: np.ndarray,
        b_values: np.ndarray,
        mixed_values: tuple[np.ndarray, ...],
        row_weights: np.ndarray | None,
    ) -> None:
        """Add one evaluated Saltelli chunk."""
        _add_unweighted_moments(
            count=self._variance_count,
            sums=self._variance_sum,
            sumsq=self._variance_sumsq,
            values=(a_values, b_values),
        )
        _add_unweighted_moments(
            count=self._center_count,
            sums=self._center_sum,
            sumsq=None,
            values=(a_values, b_values, *mixed_values),
        )
        if row_weights is not None:
            _add_weighted_moments(
                weights=row_weights,
                count=self._boot_variance_count,
                sums=self._boot_variance_sum,
                sumsq=self._boot_variance_sumsq,
                values=(a_values, b_values),
            )
            _add_weighted_moments(
                weights=row_weights,
                count=self._boot_center_count,
                sums=self._boot_center_sum,
                sumsq=None,
                values=(a_values, b_values, *mixed_values),
            )
        for dimension_index, mixed in enumerate(mixed_values):
            self._add_dimension(
                dimension_index=dimension_index,
                a_values=a_values,
                b_values=b_values,
                mixed=mixed,
                row_weights=row_weights,
            )

    def _add_dimension(
        self,
        *,
        dimension_index: int,
        a_values: np.ndarray,
        b_values: np.ndarray,
        mixed: np.ndarray,
        row_weights: np.ndarray | None,
    ) -> None:
        delta = mixed - a_values
        s1_valid = np.isfinite(b_values) & np.isfinite(delta)
        st_terms = 0.5 * delta * delta
        st_valid = np.isfinite(st_terms)
        self._s1_count[dimension_index, :] += s1_valid.sum(axis=0)
        self._s1_b_delta_sum[dimension_index, :] += np.where(
            s1_valid,
            b_values * delta,
            0.0,
        ).sum(axis=0)
        self._s1_delta_sum[dimension_index, :] += np.where(s1_valid, delta, 0.0).sum(axis=0)
        self._st_count[dimension_index, :] += st_valid.sum(axis=0)
        self._st_sum[dimension_index, :] += np.where(st_valid, st_terms, 0.0).sum(axis=0)
        if row_weights is not None:
            self._boot_s1_count[:, dimension_index, :] += row_weights @ s1_valid.astype(np.float64)
            self._boot_s1_b_delta_sum[:, dimension_index, :] += row_weights @ np.where(
                s1_valid,
                b_values * delta,
                0.0,
            )
            self._boot_s1_delta_sum[:, dimension_index, :] += row_weights @ np.where(
                s1_valid,
                delta,
                0.0,
            )
            self._boot_st_count[:, dimension_index, :] += row_weights @ st_valid.astype(np.float64)
            self._boot_st_sum[:, dimension_index, :] += row_weights @ np.where(
                st_valid,
                st_terms,
                0.0,
            )

    def estimates(self, *, confidence_level: float) -> SobolIndexEstimate:
        """Return Sobol estimates and bootstrap confidence half widths."""
        variance = _variance_from_moments(
            count=self._variance_count,
            sums=self._variance_sum,
            sumsq=self._variance_sumsq,
        )
        center = _mean_from_moments(count=self._center_count, sums=self._center_sum)
        s1, st = _indices_from_moments(
            s1_count=self._s1_count,
            s1_b_delta_sum=self._s1_b_delta_sum,
            s1_delta_sum=self._s1_delta_sum,
            st_count=self._st_count,
            st_sum=self._st_sum,
            center=center,
            variance=variance,
        )
        boot_variance = _variance_from_moments(
            count=self._boot_variance_count,
            sums=self._boot_variance_sum,
            sumsq=self._boot_variance_sumsq,
        )
        boot_center = _mean_from_moments(
            count=self._boot_center_count,
            sums=self._boot_center_sum,
        )
        s1_resamples, st_resamples = _bootstrap_indices_from_moments(
            s1_count=self._boot_s1_count,
            s1_b_delta_sum=self._boot_s1_b_delta_sum,
            s1_delta_sum=self._boot_s1_delta_sum,
            st_count=self._boot_st_count,
            st_sum=self._boot_st_sum,
            center=boot_center,
            variance=boot_variance,
        )
        z_value = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
        return SobolIndexEstimate(
            s1=s1,
            st=st,
            variance=variance,
            s1_confidence_half_width=_confidence_half_width_array(
                values=s1_resamples,
                z_value=z_value,
            ),
            st_confidence_half_width=_confidence_half_width_array(
                values=st_resamples,
                z_value=z_value,
            ),
            s1_resamples=s1_resamples,
            st_resamples=st_resamples,
        )


def sobol_confidence_converged(
    *,
    values: np.ndarray,
    half_widths: np.ndarray,
    rtol: float,
    abs_tol: float,
    scale_floor: float,
) -> bool:
    """Return whether confidence precision satisfies the requested tolerance."""
    finite = np.isfinite(values) & np.isfinite(half_widths)
    if not finite.any():
        return True
    thresholds = abs_tol + rtol * np.maximum(np.abs(values[finite]), scale_floor)
    return bool(np.all(half_widths[finite] <= thresholds))


def _add_unweighted_moments(
    *,
    count: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray | None,
    values: tuple[np.ndarray, ...],
) -> None:
    for start, stop in _column_chunks(width=values[0].shape[1], row_count=values[0].shape[0]):
        for block in values:
            chunk = block[:, start:stop]
            valid = np.isfinite(chunk)
            clean = np.where(valid, chunk, 0.0)
            count[start:stop] += valid.sum(axis=0)
            sums[start:stop] += clean.sum(axis=0)
            if sumsq is not None:
                sumsq[start:stop] += (clean * clean).sum(axis=0)


def _add_weighted_moments(
    *,
    weights: np.ndarray,
    count: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray | None,
    values: tuple[np.ndarray, ...],
) -> None:
    for start, stop in _column_chunks(
        width=values[0].shape[1],
        row_count=values[0].shape[0],
        confidence_resamples=weights.shape[0],
    ):
        for block in values:
            chunk = block[:, start:stop]
            valid = np.isfinite(chunk)
            clean = np.where(valid, chunk, 0.0)
            count[:, start:stop] += weights @ valid.astype(np.float64)
            sums[:, start:stop] += weights @ clean
            if sumsq is not None:
                sumsq[:, start:stop] += weights @ (clean * clean)


def _variance_from_moments(*, count: np.ndarray, sums: np.ndarray, sumsq: np.ndarray) -> np.ndarray:
    out = np.full(sums.shape, np.nan, dtype=np.float64)
    mean = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    variance = sumsq / np.maximum(count, 1.0) - mean * mean
    out[count > 1] = variance[count > 1]
    return out


def _mean_from_moments(*, count: np.ndarray, sums: np.ndarray) -> np.ndarray:
    out = np.full(sums.shape, np.nan, dtype=np.float64)
    np.divide(sums, count, out=out, where=count > 0)
    return out


def _indices_from_moments(
    *,
    s1_count: np.ndarray,
    s1_b_delta_sum: np.ndarray,
    s1_delta_sum: np.ndarray,
    st_count: np.ndarray,
    st_sum: np.ndarray,
    center: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    safe_variance = np.where(variance > 0.0, variance, np.nan)
    b_delta_mean = np.divide(
        s1_b_delta_sum,
        s1_count,
        out=np.full_like(s1_b_delta_sum, np.nan),
        where=s1_count > 0,
    )
    delta_mean = np.divide(
        s1_delta_sum,
        s1_count,
        out=np.full_like(s1_delta_sum, np.nan),
        where=s1_count > 0,
    )
    st_mean = np.divide(
        st_sum,
        st_count,
        out=np.full_like(st_sum, np.nan),
        where=st_count > 0,
    )
    return (b_delta_mean - center[None, :] * delta_mean) / safe_variance[None, :], (
        st_mean / safe_variance[None, :]
    )


def _bootstrap_indices_from_moments(
    *,
    s1_count: np.ndarray,
    s1_b_delta_sum: np.ndarray,
    s1_delta_sum: np.ndarray,
    st_count: np.ndarray,
    st_sum: np.ndarray,
    center: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    safe_variance = np.where(variance > 0.0, variance, np.nan)
    b_delta_mean = np.divide(
        s1_b_delta_sum,
        s1_count,
        out=np.full_like(s1_b_delta_sum, np.nan),
        where=s1_count > 0,
    )
    delta_mean = np.divide(
        s1_delta_sum,
        s1_count,
        out=np.full_like(s1_delta_sum, np.nan),
        where=s1_count > 0,
    )
    st_mean = np.divide(
        st_sum,
        st_count,
        out=np.full_like(st_sum, np.nan),
        where=st_count > 0,
    )
    return (b_delta_mean - center[:, None, :] * delta_mean) / safe_variance[:, None, :], (
        st_mean / safe_variance[:, None, :]
    )


def _confidence_half_width_array(*, values: np.ndarray, z_value: float) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=0)
    clean = np.where(valid, values, 0.0)
    sums = clean.sum(axis=0)
    sumsq = (clean * clean).sum(axis=0)
    out = np.full(values.shape[1:], np.nan, dtype=np.float64)
    enough = counts > 1
    variance = np.zeros(values.shape[1:], dtype=np.float64)
    variance[enough] = (sumsq[enough] - sums[enough] * sums[enough] / counts[enough]) / (
        counts[enough] - 1
    )
    out[enough] = z_value * np.sqrt(np.maximum(variance[enough], 0.0))
    return out


def _column_chunks(
    *,
    width: int,
    row_count: int,
    confidence_resamples: int = 0,
) -> tuple[tuple[int, int], ...]:
    chunk_columns = _moment_chunk_columns(
        row_count=row_count,
        confidence_resamples=confidence_resamples,
    )
    return tuple(
        (start, min(start + chunk_columns, width)) for start in range(0, width, chunk_columns)
    )


def _moment_chunk_columns(*, row_count: int, confidence_resamples: int) -> int:
    row_bytes = np.dtype(np.float64).itemsize * int(row_count) * len(("clean_values",))
    row_bytes += np.dtype(np.bool_).itemsize * int(row_count) * len(("valid_mask",))
    if confidence_resamples:
        row_bytes += (
            np.dtype(np.float64).itemsize
            * int(confidence_resamples)
            * len(("weighted_count", "weighted_sum", "weighted_sumsq"))
        )
    return memory_bounded_rows(bytes_per_row=max(1, row_bytes))
