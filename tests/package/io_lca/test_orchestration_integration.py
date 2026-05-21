from pathlib import Path

import pandas as pd

from pyaesa import deterministic_io_lca
from pyaesa.io_lca.data.paths import (
    resolve_io_lca_paths,
)
from pyaesa.io_lca.orchestration.pipeline.method_runner import run_io_lca_method
from pyaesa.io_lca.orchestration.reporting.summary import (
    build_io_lca_summary,
)
from pyaesa.io_lca.contracts.fu_mapping import resolve_fu_spec
from pyaesa.io_lca.data.loaders import load_domain_metadata


def _filters() -> dict[str, list[str] | None]:
    return {"r_f": ["FR", "DE"], "r_c": None, "r_p": None, "s_p": None}


def test_run_io_lca_method_covers_bannerless_main_only_branch(
    io_lca_dummy_repo,
) -> None:
    spec = resolve_fu_spec(fu_code="L1.a")
    paths = resolve_io_lca_paths(
        project_name="io_lca_runner_slice",
        group_reg=False,
        group_sec=False,
        group_version=None,
    )
    metadata, metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        group_version=None,
    )
    result = run_io_lca_method(
        lcia_method=io_lca_dummy_repo.lcia_method,
        source=io_lca_dummy_repo.source,
        group_version=None,
        group_reg=False,
        group_sec=False,
        spec=spec,
        filters=_filters(),
        metadata=metadata,
        domain_metadata_path=metadata_path,
        paths=paths,
        scope={},
        resolved_years=io_lca_dummy_repo.years,
        upstream_analysis=False,
        upstream_stages=1,
        aggreg_indices=False,
        output_format="csv",
        refresh=True,
        method_progress=None,
    )

    assert result.main_paths and result.main_paths[0].exists()
    assert result.origin_paths == []
    assert result.stage_paths == []
    assert result.done_main_years == [2019]
    assert result.done_origin_years == []
    assert result.done_stage_years == []
    assert result.skipped_years == {2020: "extension missing"}
    frame = pd.read_csv(result.main_paths[0])
    assert set(frame["year"].tolist()) == {2019}
    assert set(frame["impact"]) == {"AAL", "BI FD"}


def test_run_io_lca_method_computes_origin_outputs_from_existing_main_year(
    io_lca_dummy_repo,
) -> None:
    spec = resolve_fu_spec(fu_code="L1.a")
    paths = resolve_io_lca_paths(
        project_name="io_lca_runner_origin_extension",
        group_reg=False,
        group_sec=False,
        group_version=None,
    )
    metadata, metadata_path = load_domain_metadata(
        source=io_lca_dummy_repo.source,
        group_version=None,
    )
    initial_main_only = run_io_lca_method(
        lcia_method=io_lca_dummy_repo.lcia_method,
        source=io_lca_dummy_repo.source,
        group_version=None,
        group_reg=False,
        group_sec=False,
        spec=spec,
        filters=_filters(),
        metadata=metadata,
        domain_metadata_path=metadata_path,
        paths=paths,
        scope={},
        resolved_years=[2019],
        upstream_analysis=False,
        upstream_stages=1,
        aggreg_indices=False,
        output_format="csv",
        refresh=True,
        method_progress=None,
    )

    assert initial_main_only.main_paths and initial_main_only.main_paths[0].exists()

    extended = run_io_lca_method(
        lcia_method=io_lca_dummy_repo.lcia_method,
        source=io_lca_dummy_repo.source,
        group_version=None,
        group_reg=False,
        group_sec=False,
        spec=spec,
        filters=_filters(),
        metadata=metadata,
        domain_metadata_path=metadata_path,
        paths=paths,
        scope={
            "status": {
                "main": {io_lca_dummy_repo.lcia_method: {"years_done": [2019]}},
                "origin": {},
                "stages": {},
            }
        },
        resolved_years=[2019],
        upstream_analysis=True,
        upstream_stages=1,
        aggreg_indices=False,
        output_format="csv",
        refresh=False,
        method_progress=None,
    )

    assert extended.main_paths == [initial_main_only.main_paths[0]]
    assert len(extended.origin_paths) == 2
    assert len(extended.stage_paths) == 1
    assert extended.done_main_years == [2019]
    assert extended.done_origin_years == [2019]
    assert extended.done_stage_years == [2019]


def test_reporting_contracts_cover_generated_figure_paths_and_multi_folder_summaries(
    io_lca_dummy_repo,
) -> None:
    project_name = "io_lca_reporting_slice"
    deterministic_io_lca(
        project_name=project_name,
        source=io_lca_dummy_repo.source,
        years=io_lca_dummy_repo.years,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_code="L1.a",
        r_f=["FR", "DE"],
        upstream_analysis=True,
        upstream_stages=1,
        aggreg_indices=False,
        output_format="csv",
        figures=False,
        refresh=True,
    )

    figure_report = deterministic_io_lca(
        project_name=project_name,
        source=io_lca_dummy_repo.source,
        years=[2019],
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_code="L1.a",
        r_f=["FR", "DE"],
        upstream_analysis=True,
        upstream_stages=1,
        aggreg_indices=False,
        output_format="csv",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    )
    generated_paths = figure_report.figure_paths
    assert generated_paths
    assert all(path.exists() for path in generated_paths)

    multi_summary = build_io_lca_summary(
        source=io_lca_dummy_repo.source,
        output_root=Path("multi"),
        resolved_years=[2019, 2020],
        covered_main_years={2019},
        covered_origin_years={2019},
        covered_stage_years={2019},
        skipped_method_years={"pb_lcia__mode_a": {2020: "missing extension"}},
        aggreg_indices=True,
        upstream_analysis=True,
        stage_outputs_enabled=True,
        reuse_status="computed",
        lca_results_dirs={Path("main_a"), Path("main_b")},
        origin_dirs={Path("origin_a"), Path("origin_b")},
        stages_dirs={Path("stages_a"), Path("stages_b")},
        figure_paths=[Path("figures_a") / "figure_1.png", Path("figures_b") / "figure_2.png"],
    )
    assert "\n".join(multi_summary)
    assert len(multi_summary) > 1

    single_figure_summary = build_io_lca_summary(
        source=io_lca_dummy_repo.source,
        output_root=Path("single"),
        resolved_years=[2019],
        covered_main_years={2019},
        covered_origin_years=set(),
        covered_stage_years=set(),
        skipped_method_years={},
        aggreg_indices=False,
        upstream_analysis=False,
        stage_outputs_enabled=False,
        reuse_status="computed",
        lca_results_dirs={Path("main_only")},
        origin_dirs=set(),
        stages_dirs=set(),
        figure_paths=[Path("figures_only") / "figure.png"],
    )
    assert len(single_figure_summary) > 1
