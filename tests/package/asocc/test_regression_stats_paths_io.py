from pathlib import Path

import pandas as pd
import pytest

from pyaesa.asocc.orchestration.write.regression_stats import columns as defs_mod
from pyaesa.asocc.orchestration.write.regression_stats import paths_io as paths_mod
from pyaesa.asocc.runtime.paths.deterministic import (
    _get_allocate_logs_dir,
    columns_defs_path_for_stats,
)


def _logs_dir(tmp_path: Path) -> Path:
    return _get_allocate_logs_dir(
        tmp_path,
        source="oecd_v2025",
        group_version=None,
    )


def _fit_window_frame(*, start: int = 2000, end: int = 2005) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fit_start_year": [start],
            "fit_end_year": [end],
            "value": [1.0],
        }
    )


def test_existing_scoped_paths_cover_missing_logs_and_sorted_matches(tmp_path: Path) -> None:
    assert (
        paths_mod.existing_scoped_stats_paths(
            proj_base=tmp_path,
            output_format="csv",
            source="oecd_v2025",
            group_version=None,
        )
        == []
    )
    logs_dir = _logs_dir(tmp_path)
    regression_dir = logs_dir / "regression_proj"
    regression_dir.mkdir(parents=True, exist_ok=True)

    stats_path = regression_dir / "regression_stats.csv"
    stats_path.write_text("value\n1\n", encoding="utf-8")

    assert paths_mod.existing_scoped_stats_paths(
        proj_base=tmp_path,
        output_format="csv",
        source="oecd_v2025",
        group_version=None,
    ) == [stats_path]


def test_fit_window_cover_reordering_and_window_validation() -> None:
    reordered = paths_mod._reorder_fit_inputs_columns(
        pd.DataFrame(
            {
                "projection_branch": ["regression"],
                "source": ["oecd"],
                "fit_start_year": [2000],
                "fit_end_year": [2005],
                "x_value": [4.0],
                "extra_column": ["ignored"],
            }
        )
    )
    assert list(reordered.columns) == list(paths_mod._FIT_INPUT_OUTPUT_COLUMNS)  # noqa: SLF001
    assert reordered.loc[0, "x_value"] == 4.0
    assert pd.isna(reordered.loc[0, "y_value"])

    assert (
        paths_mod._fit_windows_in_frame(  # noqa: SLF001
            frame=pd.DataFrame(),
            label="empty_rows",
        )
        == set()
    )
    with pytest.raises(ValueError):
        paths_mod._fit_windows_in_frame(  # noqa: SLF001
            frame=pd.DataFrame({"fit_start_year": [2000]}),
            label="stats_rows",
        )
    assert paths_mod._fit_windows_in_frame(  # noqa: SLF001
        frame=pd.DataFrame(
            {
                "fit_start_year": [2000, 2000, 2001],
                "fit_end_year": [2005, 2005, 2006],
            }
        ),
        label="stats_rows",
    ) == {(2000, 2005), (2001, 2006)}

    assert paths_mod._resolve_single_fit_window(windows=set()) is None  # noqa: SLF001
    assert paths_mod._resolve_single_fit_window(windows={(2000, 2005)}) == (  # noqa: SLF001
        2000,
        2005,
    )
    with pytest.raises(ValueError):
        paths_mod._resolve_single_fit_window(windows={(2000, 2005), (2001, 2006)})  # noqa: SLF001

    assert paths_mod._resolve_fit_window_for_write(  # noqa: SLF001
        stats_frame=_fit_window_frame(),
        uncertainty_frame=_fit_window_frame(),
        fit_inputs_frame=_fit_window_frame(),
    ) == (2000, 2005)
    with pytest.raises(ValueError):
        paths_mod._resolve_fit_window_for_write(  # noqa: SLF001
            stats_frame=_fit_window_frame(),
            uncertainty_frame=pd.DataFrame({"fit_start_year": [2000]}),
            fit_inputs_frame=pd.DataFrame(),
        )
    with pytest.raises(ValueError):
        paths_mod._resolve_fit_window_for_write(  # noqa: SLF001
            stats_frame=_fit_window_frame(start=2000, end=2005),
            uncertainty_frame=_fit_window_frame(start=2001, end=2006),
            fit_inputs_frame=pd.DataFrame(),
        )


@pytest.mark.parametrize("output_format", ["csv", "pickle", "parquet"])
def test_table_io_and_columns_defs_cover_all_output_formats(
    tmp_path: Path,
    output_format: str,
) -> None:
    suffix = paths_mod.suffix_for_output_format(output_format=output_format)
    table_path = tmp_path / f"regression_stats{suffix}"
    frame = pd.DataFrame(
        {
            "fit_start_year": [2000, 2001],
            "fit_end_year": [2005, 2006],
            "value": [1.25, 2.5],
        }
    )

    paths_mod._write_table(path=table_path, output_format=output_format, frame=frame)  # noqa: SLF001
    assert table_path.exists()

    paths_mod._write_regression_columns_defs(stats_path=table_path)  # noqa: SLF001
    defs_path = columns_defs_path_for_stats(stats_path=table_path)
    assert defs_path.exists()
    rendered_defs = defs_mod.render_regression_columns_defs(
        columns=defs_mod.REGRESSION_MODELS_COLUMNS
    )
    assert defs_path.read_text(encoding="utf-8") == rendered_defs
    assert all(len(line) <= 100 for line in rendered_defs.splitlines())
    assert all(len(line) <= 100 for line in defs_path.read_text(encoding="utf-8").splitlines())


def test_render_regression_columns_defs_rejects_unknown_columns() -> None:
    with pytest.raises(ValueError):
        defs_mod.render_regression_columns_defs(columns=["missing_column"])
