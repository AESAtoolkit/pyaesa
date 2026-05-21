"""Figure output ownership for processed AR6 climate outputs."""

from pathlib import Path
from typing import Callable

import pandas as pd
from pyaesa.shared.runtime.io.filesystem import ensure_dir

from .generate_figures import generate_ar6_figures
from ..io.contracts import final_pathways_sheet_name
from ..io.metadata import read_json, signature_matches, write_figure_metadata
from ..io.text_outputs import figure_sampling_log_columns_explanation_text


def figure_signature(
    *,
    study_period: list[int],
    database: str,
    categories: list[str],
    variables_output: list[str],
    figure_output_format: str,
    figure_dpi: int,
    harmonization_method: str,
    figure_convergence_tol: float,
    figure_convergence_max_runs: int,
) -> dict:
    """Return the figure output signature."""
    return {
        "study_period": [int(study_period[0]), int(study_period[1])],
        "database": database,
        "categories": list(categories),
        "variables_output": list(variables_output),
        "figure_output_format": str(figure_output_format).lower(),
        "figure_dpi": int(figure_dpi),
        "harmonization_method": str(harmonization_method),
        "figure_convergence_tol": float(figure_convergence_tol),
        "figure_convergence_max_runs": int(figure_convergence_max_runs),
    }


def load_saved_figure_files(
    *,
    figures_metadata_file: Path,
    figures_dir: Path,
    study_period: list[int],
    database: str,
    categories: list[str],
    variables_output: list[str],
    figure_output_format: str,
    figure_dpi: int,
    harmonization_method: str,
    figure_convergence_tol: float,
    figure_convergence_max_runs: int,
) -> list[str] | None:
    """Return saved figure files when the current signature is fully reusable."""
    existing_metadata = read_json(figures_metadata_file)
    sig = figure_signature(
        study_period=study_period,
        database=database,
        categories=categories,
        variables_output=variables_output,
        figure_output_format=figure_output_format,
        figure_dpi=figure_dpi,
        harmonization_method=harmonization_method,
        figure_convergence_tol=figure_convergence_tol,
        figure_convergence_max_runs=figure_convergence_max_runs,
    )
    existing_files = [
        str(path) for path in sorted(figures_dir.glob(f"*.{figure_output_format.lower()}"))
    ]
    if existing_files and existing_metadata is None:
        raise RuntimeError(
            "Figure files exist but their metadata JSON is missing. "
            f"Metadata={figures_metadata_file}. Existing figure examples={existing_files[:5]}. "
            "Rerun process_ar6(..., figures=True, refresh=True) to rebuild this figure scope."
        )
    execution = {} if existing_metadata is None else existing_metadata["execution"]
    artifacts = {} if existing_metadata is None else existing_metadata["artifacts"]
    sampling_log_csv = artifacts.get("sampling_convergence_log_csv")
    sampling_log_columns_txt = artifacts.get("sampling_convergence_log_columns_txt")
    if (
        existing_metadata is not None
        and signature_matches(existing_metadata, sig)
        and bool(execution["complete"])
        and all(Path(path).exists() for path in artifacts["figure_files"])
        and isinstance(sampling_log_csv, str)
        and bool(sampling_log_csv)
        and Path(sampling_log_csv).exists()
        and isinstance(sampling_log_columns_txt, str)
        and bool(sampling_log_columns_txt)
        and Path(sampling_log_columns_txt).exists()
    ):
        return list(artifacts["figure_files"])
    return None


def ensure_figures(
    *,
    out_file: Path,
    log_file: Path | None,
    figures_dir: Path,
    figures_metadata_file: Path,
    logs_dir: Path,
    variables_output: list[str],
    study_period: list[int],
    database: str,
    categories: list[str],
    figure_output_format: str,
    figure_dpi: int,
    harmonization_method: str,
    figure_convergence_tol: float,
    figure_convergence_max_runs: int,
    refresh: bool,
    source_metadata: pd.DataFrame,
    raw_data_dir: Path,
    status_callback: Callable[[str], None] | None = None,
    generate_figures_func=generate_ar6_figures,
) -> tuple[list[str], bool]:
    """Generate figures when needed and return ``(figure_files, outputs_reused)``."""
    figures_dir = ensure_dir(figures_dir)
    logs_dir = ensure_dir(logs_dir)
    if refresh:
        for path in figures_dir.glob("*"):
            if path.is_file():
                path.unlink()
    sig = figure_signature(
        study_period=study_period,
        database=database,
        categories=categories,
        variables_output=variables_output,
        figure_output_format=figure_output_format,
        figure_dpi=figure_dpi,
        harmonization_method=harmonization_method,
        figure_convergence_tol=figure_convergence_tol,
        figure_convergence_max_runs=figure_convergence_max_runs,
    )
    if not refresh:
        saved_files = load_saved_figure_files(
            figures_metadata_file=figures_metadata_file,
            figures_dir=figures_dir,
            study_period=study_period,
            database=database,
            categories=categories,
            variables_output=variables_output,
            figure_output_format=figure_output_format,
            figure_dpi=figure_dpi,
            harmonization_method=harmonization_method,
            figure_convergence_tol=figure_convergence_tol,
            figure_convergence_max_runs=figure_convergence_max_runs,
        )
        if saved_files is not None:
            return saved_files, True

    with pd.ExcelFile(out_file, engine="calamine") as xl:
        harmonized_data = pd.read_excel(
            xl,
            sheet_name=final_pathways_sheet_name(harmonization=True),
            index_col=[0, 1, 2],
        )
        original_data = pd.read_excel(
            xl,
            sheet_name="ORIGINAL_AR6",
            index_col=[0, 1, 2],
        )
        historical_data = pd.read_excel(
            xl,
            sheet_name="HISTORICAL_PRIMAP_GCP",
            index_col=0,
        )
    if log_file is None or not log_file.exists():
        raise RuntimeError(
            "The harmonization log workbook is missing. Rebuild the harmonized AR6 outputs "
            "before generating figures."
        )
    harmonization_log = pd.read_excel(
        log_file,
        sheet_name="HARMONIZATION_LOG",
        index_col=[0, 1, 2],
        engine="calamine",
    )

    def persist_partial_metadata(current_figure_files: list[str]) -> None:
        write_figure_metadata(
            figures_metadata_file=figures_metadata_file,
            signature=sig,
            figure_files=current_figure_files,
            generation_complete=False,
        )

    figure_files, convergence_log_df = generate_figures_func(
        output_dir=str(figures_dir),
        harmonized_data=harmonized_data,
        original_data=original_data,
        harmonization_log=harmonization_log,
        historical_data=historical_data,
        source_metadata=source_metadata,
        raw_data_dir=str(raw_data_dir),
        study_period=study_period,
        database=database,
        categories=categories,
        variables_output=variables_output,
        figure_output_format=figure_output_format,
        dpi=figure_dpi,
        figure_convergence_tol=figure_convergence_tol,
        figure_convergence_max_runs=figure_convergence_max_runs,
        status_callback=status_callback,
        metadata_callback=persist_partial_metadata,
    )
    sampling_log_csv_file = logs_dir / "figure_sampling_convergence_log.csv"
    sampling_log_columns_txt_file = (
        logs_dir / "figure_sampling_convergence_log_columns_explanation.txt"
    )
    convergence_log_df.to_csv(sampling_log_csv_file, index=False)
    sampling_log_columns_txt_file.write_text(
        figure_sampling_log_columns_explanation_text(),
        encoding="utf-8",
    )
    write_figure_metadata(
        figures_metadata_file=figures_metadata_file,
        signature=sig,
        figure_files=figure_files,
        generation_complete=True,
        sampling_log_csv_file=sampling_log_csv_file,
        sampling_log_columns_txt_file=sampling_log_columns_txt_file,
    )
    return figure_files, False
