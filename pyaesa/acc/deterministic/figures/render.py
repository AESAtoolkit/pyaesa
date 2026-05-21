"""Deterministic aCC figure orchestration."""

from pathlib import Path
from typing import Any

from pyaesa.acc.deterministic.figures.metadata import (
    clear_deterministic_figure_scope,
    figure_request_signature,
    figure_state_matches,
    load_figure_metadata,
    recorded_figure_paths,
    write_branch_figure_paths,
)
from pyaesa.acc.deterministic.state.metadata import load_run_metadata
from pyaesa.acc.deterministic.figures.product_renderers import render_products
from pyaesa.acc.deterministic.figures.row_reader import load_deterministic_figure_rows
from pyaesa.shared.figures.request_validation import (
    validate_consecutive_multi_year_figure_request,
)
from pyaesa.shared.runtime.reporting.status import StatusSink


def render_acc_deterministic_figures(
    *,
    metadata_path: Path,
    dpi: int,
    output_format: str,
    figure_options: dict[str, bool],
    coverage: dict[str, list[Any]] | None = None,
    status: StatusSink | None = None,
) -> tuple[list[Path], bool]:
    """Render deterministic aCC figures from persisted branch output tables."""
    payload = load_run_metadata(metadata_path)
    request_signature = figure_request_signature(
        dpi=dpi,
        output_format=output_format,
        figure_options=figure_options,
    )
    reuse = payload["reuse"]
    compute_signature = {
        "identity_key": reuse["identity_key"],
        "coverage": coverage or reuse["coverage"],
    }
    figure_payload = load_figure_metadata(metadata_path=metadata_path)
    if figure_state_matches(
        payload=figure_payload,
        request_signature=request_signature,
        compute_signature=compute_signature,
    ):
        return recorded_figure_paths(payload=figure_payload), True
    if not (bool(figure_options["per_method"]) or bool(figure_options["multi_method"])):
        clear_deterministic_figure_scope(metadata_path=metadata_path)
        write_branch_figure_paths(
            metadata_path=metadata_path,
            figure_paths=[],
            request_signature=request_signature,
            compute_signature=compute_signature,
        )
        return [], False
    rows, requested_years, cc_type = load_deterministic_figure_rows(
        metadata_path=metadata_path,
        coverage=coverage,
    )
    validate_consecutive_multi_year_figure_request(
        requested_years=requested_years,
        family_label="deterministic aCC",
    )
    clear_deterministic_figure_scope(metadata_path=metadata_path)
    figures_root = metadata_path.parent.parent / "figures"
    figure_paths = render_products(
        rows=rows,
        figures_root=figures_root,
        requested_years=requested_years,
        cc_type=cc_type,
        dpi=dpi,
        output_format=output_format,
        per_method=bool(figure_options["per_method"]),
        multi_method=bool(figure_options["multi_method"]),
        status=status,
    )
    write_branch_figure_paths(
        metadata_path=metadata_path,
        figure_paths=figure_paths,
        request_signature=request_signature,
        compute_signature=compute_signature,
    )
    return figure_paths, False
