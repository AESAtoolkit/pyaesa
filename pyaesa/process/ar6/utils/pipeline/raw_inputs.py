"""Validation ownership for downloaded AR6 raw inputs."""

from pathlib import Path

from pyaesa.download.ar6.utils.io.paths import (
    explorer_csv_path_for_raw_dir,
)
from pyaesa.download.ar6.utils.sources import (
    historical_sources,
)


def require_downloaded_ar6_raw_inputs(
    *,
    raw_data_dir: Path,
    citation_txt_path: Path,
    database: str,
) -> None:
    """Fail fast when AR6 raw inputs required by ``process_ar6`` are missing."""
    raw_dir = Path(raw_data_dir)
    expected_paths: list[Path] = [
        explorer_csv_path_for_raw_dir(raw_dir, database),
        Path(citation_txt_path),
        raw_dir / historical_sources.PRIMAP_FINAL_LOCAL_NAME,
        raw_dir / historical_sources.PRIMAP_FINAL_NO_ROUNDING_LOCAL_NAME,
        raw_dir / historical_sources.GCP_NATIONAL_FOSSIL_LOCAL_NAME,
    ]
    missing = [path.name for path in expected_paths if not path.exists()]
    if missing:
        missing_s = ", ".join(sorted(missing))
        raise RuntimeError(
            "AR6 raw inputs are missing for process_ar6: "
            f"{missing_s}. Run download_ar6() before process_ar6()."
        )


def require_ar6_historical_figure_reference(*, raw_data_dir: Path) -> None:
    """Fail fast when the figure only AR6 historical comparison file is missing."""
    overlay_file = (
        Path(raw_data_dir) / historical_sources.AR6_HISTORICAL_FIGURE_REFERENCE_LOCAL_NAME
    )
    if overlay_file.exists():
        return
    raise RuntimeError(
        "The AR6 historical figure reference file is missing from the raw-data folder. "
        "Run download_ar6() before process_ar6(..., figures=True)."
    )
