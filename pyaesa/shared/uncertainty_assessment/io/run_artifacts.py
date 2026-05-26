"""Public Monte Carlo run artifact contracts and interval indexes."""

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, cast

import numpy as np
import pandas as pd

from pyaesa.shared.runtime.io.filesystem import ensure_file_parent, write_via_atomic_temp
from pyaesa.shared.uncertainty_assessment.io.csv_fragments import (
    CSV_COMPACT_RUN_FRAGMENT_SUFFIX,
)
from pyaesa.shared.uncertainty_assessment.io.formats import (
    is_csv_compact_output,
    normalize_uncertainty_output_format,
    suffix_for_uncertainty_output,
)

RUN_INTERVAL_INDEX_STEM_SUFFIX = ".run_intervals"
RUN_INTERVAL_COLUMNS = (
    "batch_index",
    "run_start",
    "run_stop",
    "row_start",
    "row_count",
    "fragment",
)


def run_interval_index_path(*, path: Path, output_format: str) -> Path:
    """Return the public run interval index path for a run artifact."""
    table_path = Path(path)
    suffix = suffix_for_uncertainty_output(output_format)
    return table_path.with_name(f"{table_path.stem}{RUN_INTERVAL_INDEX_STEM_SUFFIX}{suffix}")


def public_run_artifact_contract(*, path: Path, output_format: str) -> dict[str, str]:
    """Return metadata for one public Monte Carlo run artifact."""
    table_path = Path(path)
    if is_csv_compact_output(output_format):
        return {
            "artifact_kind": "csv_compact_dataset_directory",
            "path": str(table_path),
            "fragment_pattern": f"part-*{CSV_COMPACT_RUN_FRAGMENT_SUFFIX}",
            "interval_index_path": str(
                run_interval_index_path(path=table_path, output_format=output_format)
            ),
            "interval_index_kind": "csv_file",
        }
    return {
        "artifact_kind": "parquet_dataset_directory",
        "path": str(table_path),
        "fragment_pattern": "part-*.parquet",
        "interval_index_path": str(
            run_interval_index_path(path=table_path, output_format=output_format)
        ),
        "interval_index_kind": "parquet_file",
    }


def public_run_artifact_readme_lines(*, run_name: str) -> list[str]:
    """Return README lines for the shared public run artifact contract."""
    return [
        f"- {run_name}: public Monte Carlo run values.",
        "  For csv_compact output this path is a dataset directory",
        "  containing compressed part-*.csv.zst CSV fragments.",
        "  For Parquet output this path is a dataset directory containing",
        "  part-*.parquet fragments.",
        f"- {run_name}.run_intervals.<suffix>: interval index for run windows",
        "  artifact row ranges, and fragment names.",
        "  Public readers use it to read only requested run windows without",
        "  concatenating or scanning unrelated batches.",
        "  For csv_compact and Parquet it maps each run interval to its fragment file.",
        "  The manifest records the interval index path.",
    ]


def read_run_interval_index(*, path: Path, output_format: str) -> pd.DataFrame:
    """Read the public run interval index for a run artifact."""
    index_path = run_interval_index_path(path=path, output_format=output_format)
    if is_csv_compact_output(output_format):
        frame = pd.read_csv(index_path)
    else:
        frame = pd.read_parquet(index_path)
    return _normalize_run_interval_index(frame=frame)


@dataclass
class RunIntervalWriterState:
    """Mutable interval state for one generated public run artifact."""

    path: Path
    output_format: str
    append_existing: bool
    intervals: list[dict[str, Any]]
    next_row_offset: int
    next_fragment_index: int
    prepared: bool = False
    dirty: bool = False

    @classmethod
    def create(
        cls,
        *,
        path: Path,
        output_format: str,
        append_existing: bool,
    ) -> "RunIntervalWriterState":
        """Create interval state for a new or appended artifact."""
        table_path = Path(path)
        fmt = normalize_uncertainty_output_format(output_format)
        intervals = _existing_run_intervals(
            path=table_path,
            output_format=fmt,
            append_existing=bool(append_existing),
        )
        return cls(
            path=table_path,
            output_format=fmt,
            append_existing=bool(append_existing),
            intervals=intervals,
            next_row_offset=_next_interval_row_offset(intervals=intervals),
            next_fragment_index=len(intervals),
        )

    def close(self) -> None:
        """Write the interval index when recorded intervals changed."""
        if self.dirty:
            _write_run_interval_index(
                path=self.path,
                output_format=self.output_format,
                intervals=self.intervals,
            )
            self.dirty = False

    def prepare(self) -> None:
        """Prepare the dataset directory before the first fragment write."""
        if self.prepared:
            return
        _prepare_run_artifact(
            path=self.path,
            output_format=self.output_format,
            append_existing=self.append_existing,
        )
        self.prepared = True

    def next_fragment(self) -> str:
        """Return the next fragment name and advance the fragment counter."""
        fragment = run_fragment_name(
            index=self.next_fragment_index,
            output_format=self.output_format,
        )
        self.next_fragment_index += 1
        return fragment

    def record_interval(
        self,
        *,
        batch_index: int,
        run_index: np.ndarray,
        row_count: int,
        fragment: str,
    ) -> None:
        """Record one non empty fragment in the interval index."""
        if row_count == 0 or len(run_index) == 0:
            return
        run_values = np.asarray(run_index, dtype=np.int64)
        self.intervals.append(
            {
                "batch_index": int(batch_index),
                "run_start": int(np.min(run_values)),
                "run_stop": int(np.max(run_values)) + 1,
                "row_start": int(self.next_row_offset),
                "row_count": int(row_count),
                "fragment": str(fragment),
            }
        )
        self.next_row_offset += int(row_count)
        self.dirty = True


def first_run_fragment_path(*, path: Path, output_format: str) -> Path:
    """Return the first generated run artifact fragment."""
    suffix = run_fragment_suffix(output_format=output_format)
    return sorted(Path(path).glob(f"part-*{suffix}"))[0]


def run_fragment_name(*, index: int, output_format: str) -> str:
    """Return the generated fragment file name for one fragment index."""
    return f"part-{int(index):08d}{run_fragment_suffix(output_format=output_format)}"


def run_fragment_suffix(*, output_format: str) -> str:
    """Return the generated fragment suffix for one output format."""
    return CSV_COMPACT_RUN_FRAGMENT_SUFFIX if is_csv_compact_output(output_format) else ".parquet"


def _existing_run_intervals(
    *,
    path: Path,
    output_format: str,
    append_existing: bool,
) -> list[dict[str, Any]]:
    """Return already written interval records when appending to an artifact."""
    if not append_existing or not path.exists():
        return []
    frame = read_run_interval_index(path=path, output_format=output_format)
    return [
        {
            "batch_index": int(record["batch_index"]),
            "run_start": int(record["run_start"]),
            "run_stop": int(record["run_stop"]),
            "row_start": int(record["row_start"]),
            "row_count": int(record["row_count"]),
            "fragment": str(record["fragment"]),
        }
        for record in cast(list[dict[str, Any]], frame.to_dict("records"))
    ]


def _next_interval_row_offset(*, intervals: list[dict[str, Any]]) -> int:
    """Return the next artifact row offset after existing intervals."""
    if not intervals:
        return 0
    last = intervals[-1]
    return int(last["row_start"]) + int(last["row_count"])


def _prepare_run_artifact(*, path: Path, output_format: str, append_existing: bool) -> None:
    """Prepare the artifact directory for append or replacement writing."""
    ensure_file_parent(path)
    if append_existing:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return
    index_path = run_interval_index_path(path=path, output_format=output_format)
    if index_path.exists():
        index_path.unlink()
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _write_run_interval_index(
    *,
    path: Path,
    output_format: str,
    intervals: list[dict[str, Any]],
) -> None:
    """Persist interval records beside the generated artifact directory."""
    frame = pd.DataFrame.from_records(intervals, columns=RUN_INTERVAL_COLUMNS)
    index_path = run_interval_index_path(path=path, output_format=output_format)
    if is_csv_compact_output(output_format):
        write_via_atomic_temp(
            index_path,
            writer=lambda tmp_path: frame.to_csv(tmp_path, index=False),
        )
        return
    write_via_atomic_temp(
        index_path,
        writer=lambda tmp_path: frame.to_parquet(tmp_path, index=False),
    )


def _normalize_run_interval_index(*, frame: pd.DataFrame) -> pd.DataFrame:
    """Return an interval index with canonical columns and dtypes."""
    out = frame.loc[:, list(RUN_INTERVAL_COLUMNS)].copy()
    for column in ("batch_index", "run_start", "run_stop", "row_start", "row_count"):
        numeric = pd.Series(pd.to_numeric(out[column], errors="raise"), index=out.index)
        out[column] = numeric.astype("int64")
    out["fragment"] = out["fragment"].fillna("").astype(str)
    return out
