from pathlib import Path
from types import SimpleNamespace

from pyaesa.asocc.runtime.paths import deterministic as paths_mod


def test_suffix_cover_error_branches() -> None:
    assert paths_mod.suffix_for_output_format(output_format="csv") == ".csv"
    assert paths_mod.suffix_for_output_format(output_format="pickle") == ".pickle"
    assert paths_mod.suffix_for_output_format(output_format="parquet") == ".parquet"


def test_runtime_and_diagnostics_paths_cover_state_validation_and_filename_tags(
    tmp_path: Path,
) -> None:
    state = SimpleNamespace(
        runtime_proj_base=tmp_path,
        runtime_output_source="oecd_v2025",
        runtime_group_version="demo",
    )
    runtime_logs_dir = paths_mod.runtime_regression_logs_dir(
        state=state,
    )
    assert runtime_logs_dir == paths_mod.allocate_regression_logs_dir(
        proj_base=tmp_path,
        source="oecd_v2025",
        group_version="demo",
    )
    assert (
        paths_mod.projection_clipping_log_path(
            state=state,
        )
        == runtime_logs_dir / "projection_clipping_log.csv"
    )
    assert (
        paths_mod.share_fit_window_log_path(
            state=state,
        )
        == runtime_logs_dir / "share_fit_window_log.csv"
    )


def test_allocate_scope_paths_cover_public_diagnostics_contracts(tmp_path: Path) -> None:
    common = {
        "proj_base": tmp_path,
        "source": "oecd_v2025",
        "group_version": None,
    }
    local_common = {key: value for key, value in common.items() if key != "proj_base"}
    refresh_root = paths_mod._get_allocate_refresh_scope_root(**common)  # noqa: SLF001
    logs_dir = paths_mod._get_allocate_logs_dir(tmp_path, **local_common)  # noqa: SLF001
    assert logs_dir == refresh_root / "logs"
    assert paths_mod._get_allocate_summary_log_path(tmp_path, **local_common) == (  # noqa: SLF001
        logs_dir / "summary.log"
    )
    assert paths_mod._get_allocate_run_metadata_path(tmp_path, **local_common) == (
        logs_dir / "scope_manifest.json"
    )  # noqa: SLF001
    assert paths_mod._get_allocate_ut_gvaa_identity_closure_path(**common) == (
        logs_dir / "ut_gvaa_identity_closure_audit.csv"
    )  # noqa: SLF001
    assert paths_mod._get_asocc_figure_metadata_path(**common) == (logs_dir / "figure.json")  # noqa: SLF001

    regression_dir = paths_mod._get_projection_regression_dir(  # noqa: SLF001
        tmp_path,
        source="oecd_v2025",
        group_version=None,
    )
    assert regression_dir == logs_dir / "regression_proj"
    assert (
        paths_mod._get_allocate_regression_stats_path(  # noqa: SLF001
            output_format="csv",
            **common,
        )
        == regression_dir / "regression_stats.csv"
    )
    assert (
        paths_mod._get_allocate_regression_fit_inputs_path(  # noqa: SLF001
            output_format="pickle",
            **common,
        )
        == regression_dir / "regression_fit_inputs.pickle"
    )

    stats_path = paths_mod.stats_path_for_format(
        output_format="parquet",
        **common,
    )
    fit_inputs_path = paths_mod.fit_inputs_path_for_format(
        output_format="csv",
        **common,
    )
    assert stats_path == logs_dir / "regression_proj" / "regression_stats.parquet"
    assert fit_inputs_path == logs_dir / "regression_proj" / "regression_fit_inputs.csv"
    assert paths_mod.columns_defs_path_for_stats(stats_path=stats_path) == (
        stats_path.parent / "regression_stats_columns_defs.txt"
    )
