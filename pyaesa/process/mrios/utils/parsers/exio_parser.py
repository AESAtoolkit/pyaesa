"""EXIOBASE parser and LCIA job configuration ownership."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, cast

import pandas as pd
import pymrio
from pyaesa.download.mrios.utils.paths import _get_exio_archive_path
from pyaesa.shared.runtime.io.filesystem import ensure_file_parent
from pyaesa.shared.lcia.prerequisite_tables import clean_characterization_matrix_frame

from pyaesa.process.mrios.utils.io.paths import (
    _get_characterization_matrix_path,
    _get_mrio_calc_log_path,
)

from .exio_characterization import (
    _calc_characterized_extensions_minimal as _core_calc_characterized_extensions_minimal,
    _characterize_exiobase_io_core,
    _find_missing_characterization_extensions as _core_find_missing_characterization_extensions,
    _retain_extension_instances as _core_retain_extension_instances,
)

_DEFAULT_RETAINED_EXTENSIONS: tuple[str, ...] = ("factor_inputs",)


def _parse_exio_year(
    full_dir: Path,
    year: int,
    *,
    system: str,
    parser: Callable[[str], Any] = pymrio.parse_exiobase3,
) -> pymrio.IOSystem:
    """Parse a single EXIOBASE archive for ``year``."""
    archive_path = _get_exio_archive_path(
        Path(full_dir),
        int(year),
        system=str(system).strip(),
    )
    if not archive_path.exists():
        raise FileNotFoundError(
            f"EXIO archive for {year} not found at {archive_path}. Run MRIO downloads first."
        )
    return cast(pymrio.IOSystem, parser(str(archive_path)))


@dataclass
class CharacterizationSummary:
    """Summary of a characterization run."""

    keep_instances: list[str]
    requested_extensions: list[str]


@dataclass
class ExioCharacterizationOptions:
    """Configuration for EXIOBASE characterization."""

    lcia_method: str
    matrix_path: Path
    char_matrix: pd.DataFrame
    retain_instances: tuple[str, ...] = _DEFAULT_RETAINED_EXTENSIONS
    requested_extensions: list[str] = field(default_factory=list)


def _build_characterization_options(
    *,
    source_key: str,
    requested_lcia_method: Optional[str],
) -> Optional[ExioCharacterizationOptions]:
    """Return normalized characterization options with loaded matrix payload."""
    lcia_method = str(requested_lcia_method or "").strip()
    if not lcia_method:
        return None
    matrix_path = _get_characterization_matrix_path(
        source_key=source_key,
        lcia_method=lcia_method,
    )
    char_matrix = _load_characterization_matrix(matrix_path)
    return ExioCharacterizationOptions(
        lcia_method=lcia_method,
        matrix_path=matrix_path,
        char_matrix=char_matrix,
        retain_instances=_DEFAULT_RETAINED_EXTENSIONS,
        requested_extensions=_requested_extensions_from_matrix(char_matrix),
    )


def _build_characterization_jobs(
    *,
    source_key: str,
    lcia_methods: Optional[Sequence[str]],
) -> dict[str, ExioCharacterizationOptions]:
    """Return characterization jobs keyed by method name."""
    if not lcia_methods:
        return {}

    lcia_method_names: list[str] = []
    seen: set[str] = set()
    for lcia_method in lcia_methods:
        cleaned = str(lcia_method).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        lcia_method_names.append(cleaned)

    jobs: dict[str, ExioCharacterizationOptions] = {}
    for lcia_method in lcia_method_names:
        options = _build_characterization_options(
            source_key=source_key,
            requested_lcia_method=lcia_method,
        )
        options = cast(ExioCharacterizationOptions, options)
        jobs[options.lcia_method] = options
    return jobs


def _load_characterization_matrix(path: Path) -> pd.DataFrame:
    """Load and normalize an LCIA characterization matrix CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Characterization matrix not found at {path}")
    df = clean_characterization_matrix_frame(
        frame=pd.read_csv(path),
        path=path,
    )
    required = {"extension", "stressor", "impact", "factor", "impact_unit"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Characterization matrix at {path} is missing required columns: {missing}."
        )
    df["extension"] = df["extension"].astype(str).str.strip()
    required_text = df.loc[:, ["extension", "stressor", "impact", "impact_unit"]].astype(str)
    required_text = required_text.apply(lambda series: series.str.strip())
    usable = (
        required_text["extension"].ne("")
        & required_text["stressor"].ne("")
        & required_text["impact"].ne("")
        & required_text["impact_unit"].ne("")
    )
    if not bool(usable.any()):
        raise ValueError(
            f"Characterization matrix at {path} has no usable impact rows with extension, "
            "stressor, impact, and impact_unit values."
        )
    df.attrs["source_path"] = str(path)
    return df


def _requested_extensions_from_matrix(char_df: pd.DataFrame) -> list[str]:
    """Return unique extension names referenced by ``char_df``."""
    return sorted({str(value) for value in char_df["extension"].dropna()})


def _characterize_exiobase_io(
    io_system: pymrio.IOSystem,
    *,
    char_matrix: pd.DataFrame,
    new_extension_name: str,
    retain_instances: Iterable[str],
    prune: bool = True,
    source_key: str | None = None,
    year: int | None = None,
) -> CharacterizationSummary:
    """Attach characterized extension ``new_extension_name`` to ``io_system``."""
    if not new_extension_name.isidentifier():
        raise ValueError(
            "The public lcia_method value becomes a PyMRIO extension name during "
            "MRIO processing and must be a valid Python identifier. "
            f"Received lcia_method='{new_extension_name}'."
        )

    def _validate(validation_df: pd.DataFrame) -> None:
        _ensure_validation_success(
            validation_df,
            source_key=source_key,
            year=year,
            lcia_method=new_extension_name,
        )

    keep_instances, requested_extensions = _characterize_exiobase_io_core(
        io_system,
        char_matrix=char_matrix,
        new_extension_name=new_extension_name,
        retain_instances=retain_instances,
        prune=prune,
        validate=_validate,
    )
    return CharacterizationSummary(
        keep_instances=keep_instances,
        requested_extensions=requested_extensions,
    )


def _find_missing_characterization_extensions(
    io_system: pymrio.IOSystem,
    requested_extensions: Sequence[str],
) -> list[str]:
    """Return requested extensions absent from ``io_system``."""
    return _core_find_missing_characterization_extensions(io_system, requested_extensions)


def _retain_extension_instances(
    io_system: pymrio.IOSystem,
    retain_instances: Iterable[str],
) -> list[str]:
    """Remove EXIO satellite accounts other than ``retain_instances``."""
    return _core_retain_extension_instances(io_system, retain_instances)


def _calc_characterized_extensions_minimal(
    io_system: pymrio.IOSystem,
    lcia_method_names: Sequence[str],
    *,
    keep_direct_intensities: bool,
) -> None:
    """Compute only LCIA attributes required by UNCASExt enacting metric outputs."""
    _core_calc_characterized_extensions_minimal(
        io_system,
        lcia_method_names,
        keep_direct_intensities=keep_direct_intensities,
    )


def _safe_filename_token(value: str) -> str:
    """Return a filesystem safe token for diagnostics filenames."""
    cleaned = "".join(
        char if (char.isalnum() or char in {"-", "_"}) else "_" for char in str(value).strip()
    ).strip("_")
    return cleaned or "unknown"


def _write_validation_mismatch_log(
    *,
    validation_df: pd.DataFrame,
    mask: pd.Series,
    source_key: str,
    year: int,
    lcia_method: str,
    matrix_version: str | None = None,
) -> Path:
    """Write mismatch rows to a CSV log and return its path."""
    mismatch = validation_df.loc[mask].copy()
    mismatch.insert(0, "method", str(lcia_method))
    mismatch.insert(0, "year", int(year))
    mismatch.insert(0, "source", str(source_key))

    source_token = _safe_filename_token(source_key)
    method_token = _safe_filename_token(lcia_method)
    filename = f"{source_token}_{int(year)}_{method_token}_characterization_validation_mismatch.csv"
    log_path = _get_mrio_calc_log_path(
        filename,
        source_key=source_key,
        matrix_version=matrix_version,
    )
    log_path = ensure_file_parent(log_path)
    mismatch.to_csv(log_path, index=False)
    return log_path


def _ensure_validation_success(
    validation_df: pd.DataFrame,
    *,
    source_key: str | None = None,
    year: int | None = None,
    lcia_method: str | None = None,
    matrix_version: str | None = None,
) -> None:
    """Raise when validation reports extension/stressor/unit mismatch flags."""
    cols = validation_df.columns
    flag_columns = [c for c in cols if c.startswith("error")]
    mask = cast(pd.Series, validation_df[flag_columns].any(axis=1))
    if not bool(mask.any()):
        return

    preview_cols = [col for col in ["stressor", "extension", "reason"] if col in cols]
    preview_cols += flag_columns[:3]
    preview = validation_df.loc[mask, preview_cols].head(10).to_string(index=False)

    log_msg = ""
    if source_key is not None and year is not None and lcia_method is not None:
        log_path = _write_validation_mismatch_log(
            validation_df=validation_df,
            mask=mask,
            source_key=source_key,
            year=int(year),
            lcia_method=lcia_method,
            matrix_version=matrix_version,
        )
        log_msg = f"\nMismatch rows were written to: {log_path}"

    raise RuntimeError(
        "Characterization validation failed; at least one row did not match "
        "the parsed extensions.\n"
        f"Sample rows:\n{preview}{log_msg}"
    )
