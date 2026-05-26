"""Clipping diagnostics logging for UNCASExt metric preprocessing."""

import re
from pathlib import Path
from typing import cast

import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.runtime.text import join_user_text_lines

from pyaesa.process.mrios.utils.io.paths import (
    _get_mrio_clipping_log_columns_explanation_path,
    _get_mrio_clipping_log_path,
)

_LOG_COLUMNS: tuple[str, ...] = (
    "source",
    "matrix_version",
    "matrix",
    "event_type",
    "event_detail",
    "year",
    "unit",
    "r_p",
    "s_p",
    "r_f",
    "r_u",
    "distribution_axis",
    "original_value",
    "clipped_value",
    "original_sum",
    "adjusted_sum",
    "expected_sum",
    "processed_output_abs",
    "processed_input_side_value_added_abs",
    "processed_intermediate_input_total_abs",
    "processed_output_side_value_added_abs",
)


def clipping_log_columns_explanation_text() -> str:
    """Return the shared MRIO clipping log column explanation TXT content."""
    return join_user_text_lines(
        [
            "MRIO clipping and normalization log columns explanation",
            "",
            "source: MRIO source identifier used by process_mrio(...).",
            "matrix_version: MRIO version lane. 'original_classification' means no "
            "aggregation version.",
            "matrix: Matrix or derived object that triggered the log row.",
            "event_type: Diagnostic event kind such as 'clip_negative_value' or "
            "'normalize_distribution'.",
            "event_detail: explanation of why the row was logged.",
            "year: MRIO year attached to the processed-year folder.",
            "unit: Monetary unit.",
            "r_p: Producer region for the affected producer column or row.",
            "s_p: Producer sector for the affected producer column or row.",
            "r_f: Final demand region label when the logged event refers to final demand.",
            "r_u: Upstream region label when the logged event refers to omega_reg shares.",
            "distribution_axis: Axis name that was normalized for distribution rows.",
            "original_value: Raw negative value before clipping to zero.",
            "clipped_value: Value after clipping to zero.",
            "original_sum: Distribution-column sum before normalization.",
            "adjusted_sum: Distribution-column sum after normalization.",
            "expected_sum: Expected sum for the normalized distribution.",
            "processed_output_abs: Processed producer output x_j used in the omega_reg "
            "calculation. This is the processed MRIO value, not a raw source file value.",
            "processed_input_side_value_added_abs: Processed input-side value-added amount "
            "used to build alpha_j for omega_reg.",
            "processed_intermediate_input_total_abs: Processed total intermediate inputs "
            "into producer j, sum_i Z_ij.",
            "processed_output_side_value_added_abs: Processed output-side implied "
            "value added, x_j - sum_i Z_ij.",
            "",
            "For normalize_distribution rows, the processed_* absolute columns are context "
            "for the producer column that was normalized. They are values used by the "
            "processed MRIO calculation, and they are not changed by the final omega_reg "
            "share renormalization step.",
        ],
    )


def _ensure_clipping_log_columns_explanation_file(
    *,
    source_key: str,
    matrix_version: str | None,
) -> None:
    """Write the shared MRIO clipping log schema TXT when missing or outdated."""
    explanation_path = _get_mrio_clipping_log_columns_explanation_path(
        source_key,
        matrix_version=matrix_version,
    )
    explanation_path = ensure_file_parent(explanation_path)
    expected_text = clipping_log_columns_explanation_text()
    if explanation_path.exists():
        current_text = explanation_path.read_text(encoding="utf-8")
        if current_text == expected_text:
            return
    explanation_path.write_text(expected_text, encoding="utf-8")


def _resolve_clip_log_context(
    *,
    source_key: str,
    matrix_version: str | None,
    saved_dir: Path,
) -> tuple[Path, str, int | None]:
    """Resolve log path, matrix version label, and year for clip diagnostics."""
    _ensure_clipping_log_columns_explanation_file(
        source_key=source_key,
        matrix_version=matrix_version,
    )
    version_label = "original_classification"
    if matrix_version is not None and str(matrix_version).strip():
        version_label = str(matrix_version).strip()
    log_path = _get_mrio_clipping_log_path(source_key, matrix_version=matrix_version)
    log_path = ensure_file_parent(log_path)
    year_tokens = re.findall(r"(?:19|20)\d{2}", str(saved_dir.name))
    log_year: int | None = int(year_tokens[-1]) if year_tokens else None
    return log_path, version_label, log_year


def _append_clip_log_rows(log_path: Path, rows: pd.DataFrame) -> None:
    """Append rows to the clipping CSV using the stable shared schema."""
    payload = rows.reindex(columns=list(_LOG_COLUMNS))
    mode = "a" if log_path.exists() else "w"
    payload.to_csv(log_path, mode=mode, index=False, header=mode == "w")


def _expand_labels(index: pd.Index | pd.MultiIndex) -> pd.DataFrame:
    """Expand an index into label columns suitable for the shared log schema."""
    labels = index.to_frame(index=False)
    if labels.shape[1] == 1:
        labels.columns = ["r_p"]
        return labels
    if labels.shape[1] == 2:
        labels.columns = ["r_p", "s_p"]
        return labels
    renamed = [f"label_{pos}" for pos in range(labels.shape[1])]
    labels.columns = renamed
    return labels


def write_clipping_log(
    *,
    before: pd.Series | pd.DataFrame,
    matrix_name: str,
    unit: str | None,
    source_key: str,
    matrix_version: str | None,
    saved_dir: Path,
) -> None:
    """Append clipped negative entries for one matrix into clipping diagnostics."""
    s_before = (
        cast(pd.Series, before.stack(future_stack=True))
        if isinstance(before, pd.DataFrame)
        else cast(pd.Series, before)
    )
    clipped = cast(pd.Series, s_before[s_before < 0]).to_frame(name="original_value")
    if clipped.empty:
        return
    log_path, version_label, log_year = _resolve_clip_log_context(
        source_key=source_key,
        matrix_version=matrix_version,
        saved_dir=saved_dir,
    )
    clipped["clipped_value"] = clipped["original_value"].clip(lower=0.0)
    clipped = clipped.reset_index()
    id_cols = [col for col in clipped.columns if col not in {"original_value", "clipped_value"}]
    rename_map: dict[object, str] = dict(zip(id_cols[:3], ("r_p", "s_p", "r_f")))
    clipped = clipped.rename(columns=rename_map)
    for axis_col in ("r_p", "s_p", "r_f", "r_u", "distribution_axis"):
        if axis_col not in clipped.columns:
            clipped[axis_col] = pd.NA
    clipped.insert(0, "source", str(source_key))
    clipped.insert(1, "matrix_version", version_label)
    clipped.insert(2, "matrix", matrix_name)
    clipped.insert(3, "event_type", "clip_negative_value")
    clipped.insert(
        4,
        "event_detail",
        f"Negative values in {matrix_name} were clipped to zero before UNCASExt metrics.",
    )
    clipped.insert(5, "year", log_year)
    clipped.insert(6, "unit", str(unit).strip() if unit is not None else "")
    clipped["original_sum"] = pd.NA
    clipped["adjusted_sum"] = pd.NA
    clipped["expected_sum"] = pd.NA
    _append_clip_log_rows(log_path, clipped)


def write_distribution_normalization_log(
    *,
    before: pd.DataFrame,
    after: pd.DataFrame,
    matrix_name: str,
    distribution_axis: str,
    unit: str | None,
    source_key: str,
    matrix_version: str | None,
    saved_dir: Path,
    expected_sum: float,
    absolute_context: pd.DataFrame | None = None,
) -> None:
    """Append one row per normalized distribution column to the clipping diagnostics CSV."""
    original_sum = before.sum(axis=0)
    adjusted_sum = after.sum(axis=0)
    abs_adjustment = after.sub(before).abs().max(axis=0)
    changed = cast(pd.Series, abs_adjustment.ne(0.0))
    if not bool(changed.any()):
        return

    log_path, version_label, log_year = _resolve_clip_log_context(
        source_key=source_key,
        matrix_version=matrix_version,
        saved_dir=saved_dir,
    )
    selected_original_sum = cast(pd.Series, original_sum.loc[changed])
    selected_adjusted_sum = cast(pd.Series, adjusted_sum.loc[changed])
    labels = _expand_labels(selected_original_sum.index)
    rows = labels.copy()
    for axis_col in ("r_p", "s_p", "r_f", "r_u"):
        if axis_col not in rows.columns:
            rows[axis_col] = pd.NA
    rows["source"] = str(source_key)
    rows["matrix_version"] = version_label
    rows["matrix"] = matrix_name
    rows["event_type"] = "normalize_distribution"
    rows["event_detail"] = (
        f"{matrix_name} was renormalized because clipping can create a "
        "disequilibrium between the value-added input side and the clipped output side "
        "for one producer pair (r_p, s_p). For one omega_reg column, the raw regional "
        "shares sum to alpha^T L; after clipping changes x and/or the clipped "
        "factor_inputs.F proxy used for alpha, alpha^T may no longer equal 1^T(I-A), "
        f"so the raw {distribution_axis} shares may stop summing to the expected total. "
        "The processed_* absolute columns on this row are processed MRIO values used in "
        "the diagnostic, not raw source-file values."
    )
    rows["year"] = log_year
    rows["unit"] = str(unit).strip() if unit is not None else ""
    rows["distribution_axis"] = distribution_axis
    rows["original_value"] = pd.NA
    rows["clipped_value"] = pd.NA
    rows["original_sum"] = selected_original_sum.to_numpy(dtype=float)
    rows["adjusted_sum"] = selected_adjusted_sum.to_numpy(dtype=float)
    rows["expected_sum"] = float(expected_sum)
    if absolute_context is None:
        rows["processed_output_abs"] = pd.NA
        rows["processed_input_side_value_added_abs"] = pd.NA
        rows["processed_intermediate_input_total_abs"] = pd.NA
        rows["processed_output_side_value_added_abs"] = pd.NA
    else:
        selected_context = absolute_context.reindex(selected_original_sum.index)
        for column in (
            "processed_output_abs",
            "processed_input_side_value_added_abs",
            "processed_intermediate_input_total_abs",
            "processed_output_side_value_added_abs",
        ):
            rows[column] = selected_context[column].to_numpy()
    _append_clip_log_rows(log_path, rows)
