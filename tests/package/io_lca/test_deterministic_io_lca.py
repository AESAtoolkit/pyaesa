import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa import deterministic_io_lca
from pyaesa.io_lca.orchestration.pipeline.progress import (
    format_indices_label,
    format_method_labels,
    format_year_ranges_with_count,
    io_lca_banner,
    io_lca_mode_banner,
    io_lca_mode_line,
    source_prefix,
    year_progress,
)
from pyaesa.io_lca.orchestration.reporting.summary import (
    build_io_lca_summary,
)


def _run_deterministic_io_lca(*, io_lca_dummy_repo, refresh: bool):
    return deterministic_io_lca(
        project_name="io_lca_public",
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
        refresh=refresh,
    )


def test_io_lca_progress_helpers_use_count_only_live_lines(capsys) -> None:
    filters = {"r_p": None, "s_p": ["Electricity"], "r_c": ["FR"], "r_f": None}

    assert source_prefix(source=" exiobase_3102_ixi ") == "[exiobase_3102_ixi]"
    assert format_year_ranges_with_count([2019, 2021, 2020, 2020]) == "2019-2021 (3 year(s))"
    assert format_method_labels([" gwp100_lcia ", "", "pb_lcia"]) == "gwp100_lcia, pb_lcia"
    assert format_method_labels(["", " "]) == "none"
    assert format_indices_label({}) == "all"
    assert format_indices_label(filters) == "s_p=Electricity, r_c=FR"
    assert "indices=s_p=Electricity, r_c=FR" in io_lca_mode_line(
        source="source", fu_code="L2.c.b", filters=filters, mode_tag=None
    )
    assert "aggreg_indices=post" in io_lca_mode_line(
        source="source", fu_code="L2.c.b", filters=filters, mode_tag="post"
    )

    for upstream_analysis in (True, False):
        io_lca_banner(
            source="source",
            years=[2019],
            methods=["gwp100_lcia"],
            fu_code="L2.c.b",
            filters=filters,
            upstream_analysis=upstream_analysis,
            upstream_stages=2,
        )
    io_lca_mode_banner(source="source", fu_code="L2.c.b", filters=filters, mode_tag="post")
    progress = year_progress(source="exiobase_3102_ixi", action="processing[gwp100_lcia]", total=1)
    progress.begin_year(2019)
    progress.complete_year(2019)
    progress.finish()

    captured = capsys.readouterr()
    assert captured.out


def test_deterministic_io_lca_end_to_end_reuse_and_refresh(
    io_lca_dummy_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = _run_deterministic_io_lca(io_lca_dummy_repo=io_lca_dummy_repo, refresh=True)
    first_output = capsys.readouterr().out

    assert report is not None
    assert first_output
    assert report.source == io_lca_dummy_repo.source
    assert report.fu_code == "L1.a"
    assert report.years == io_lca_dummy_repo.years
    assert report.lcia_methods == [io_lca_dummy_repo.lcia_method]
    assert report.metadata_path.exists()
    summary_log = report.metadata_path.parent / "summary.log"
    assert summary_log.read_text(encoding="utf-8").strip()
    assert not (report.metadata_path.parent.parent / "figures").exists()
    assert len(report.main_result_paths) == 1
    assert len(report.origin_paths) == 2
    assert len(report.stage_paths) == 1
    skipped_years = {
        year for skipped_by_year in report.skipped_method_years.values() for year in skipped_by_year
    }
    assert io_lca_dummy_repo.unavailable_year in skipped_years
    assert report.summary_lines

    metadata_payload = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    assert metadata_payload["function"] == "deterministic_io_lca"
    assert metadata_payload["arguments"]["project_name"] == "io_lca_public"
    assert metadata_payload["execution"]["complete"] is True

    main_path = report.main_result_paths[0]
    ungrouped_main = pd.read_csv(main_path)
    assert ungrouped_main.columns.tolist() == [
        "lcia_method",
        "year",
        "impact",
        "r_f",
        "lca_value",
        "impact_unit",
    ]
    ungrouped_values = ungrouped_main.sort_values(["impact", "r_f"])["lca_value"].tolist()
    assert ungrouped_values == [43.0, 11.0, 32.0, 9.0]
    assert set(ungrouped_main["impact"]) == {"AAL", "BI FD"}

    origin_ratio_path = next(path for path in report.origin_paths if "ratio" in path.name)
    origin_ratio = pd.read_csv(origin_ratio_path)
    ratio_totals = cast(
        pd.Series,
        origin_ratio.groupby(["impact", "r_f"], sort=True)["2019"].sum(),
    )
    assert ratio_totals.tolist() == pytest.approx([1.0] * len(ratio_totals))

    ungrouped_stage_path = report.stage_paths[0]
    ungrouped_stage = pd.read_csv(ungrouped_stage_path)
    assert "r_f" in ungrouped_stage.columns
    assert "direct_final_demand_FY" in set(ungrouped_stage["stage"])

    reused_report = _run_deterministic_io_lca(
        io_lca_dummy_repo=io_lca_dummy_repo,
        refresh=False,
    )
    capsys.readouterr()
    assert reused_report.reuse_status == "reused_exact"

    partial_payload = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    partial_payload["execution"]["complete"] = False
    partial_payload["execution"]["status"] = "running"
    report.metadata_path.write_text(json.dumps(partial_payload, indent=2), encoding="utf-8")
    resumed_report = _run_deterministic_io_lca(
        io_lca_dummy_repo=io_lca_dummy_repo,
        refresh=False,
    )
    assert resumed_report.reuse_status == "computed"
    assert resumed_report.origin_paths
    assert resumed_report.stage_paths

    with pytest.raises(ValueError):
        deterministic_io_lca(
            project_name="io_lca_public",
            source=io_lca_dummy_repo.source,
            years=io_lca_dummy_repo.years,
            lcia_method=io_lca_dummy_repo.lcia_method,
            fu_code="L1.a",
            r_f=["FR", "DE"],
            upstream_analysis=True,
            upstream_stages=1,
            aggreg_indices=True,
            output_format="csv",
            figures=False,
            refresh=False,
        )
    with pytest.raises(ValueError):
        deterministic_io_lca(
            project_name="io_lca_public",
            source=io_lca_dummy_repo.source,
            years=io_lca_dummy_repo.years,
            lcia_method=io_lca_dummy_repo.lcia_method,
            fu_code="L1.a",
            r_f=["FR", "DE"],
            upstream_analysis=True,
            upstream_stages=1,
            aggreg_indices=True,
            output_format="csv",
            figures=False,
            refresh=True,
        )

    refreshed_report = _run_deterministic_io_lca(io_lca_dummy_repo=io_lca_dummy_repo, refresh=True)

    assert refreshed_report is not None
    assert len(refreshed_report.main_result_paths) == 1


def test_deterministic_io_lca_covers_pba_aggregated_upstream_and_missing_output_failure(
    io_lca_dummy_repo,
) -> None:
    aggregated_report = deterministic_io_lca(
        project_name="io_lca_pba_aggregated",
        source=io_lca_dummy_repo.source,
        years=io_lca_dummy_repo.years,
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_code="L1.b",
        r_p=["FR", "DE"],
        upstream_analysis=True,
        upstream_stages=1,
        aggreg_indices=True,
        output_format="csv",
        figures=False,
        refresh=True,
    )
    assert aggregated_report is not None
    assert len(aggregated_report.main_result_paths) == 1
    assert len(aggregated_report.origin_paths) == 2
    assert aggregated_report.stage_paths == []

    aggregated_main = pd.read_csv(aggregated_report.main_result_paths[0])
    assert aggregated_main.columns.tolist() == [
        "lcia_method",
        "year",
        "impact",
        "r_p",
        "lca_value",
        "impact_unit",
    ]
    assert aggregated_main["r_p"].unique().tolist() == ["DE, FR"]
    assert aggregated_main.sort_values("impact")["lca_value"].tolist() == pytest.approx(
        [54.0, 41.0]
    )

    aggregated_ratio_path = next(
        path for path in aggregated_report.origin_paths if "ratio" in path.name
    )
    aggregated_ratio = pd.read_csv(aggregated_ratio_path)
    assert aggregated_ratio.columns.tolist() == [
        "impact",
        "origin_r_p",
        "origin_s_p",
        "impact_unit",
        "2019",
    ]
    aggregated_ratio_totals = cast(
        pd.Series,
        aggregated_ratio.groupby(["impact"], sort=True)["2019"].sum(),
    )
    assert aggregated_ratio_totals.tolist() == pytest.approx([1.0] * len(aggregated_ratio_totals))

    complete_report = _run_deterministic_io_lca(io_lca_dummy_repo=io_lca_dummy_repo, refresh=True)
    missing_ratio_path = next(path for path in complete_report.origin_paths if "ratio" in path.name)
    missing_ratio_path.unlink()

    with pytest.raises(ValueError):
        _run_deterministic_io_lca(io_lca_dummy_repo=io_lca_dummy_repo, refresh=False)


def test_deterministic_io_lca_generates_figures_when_requested(io_lca_dummy_repo) -> None:
    base_kwargs = {
        "project_name": "io_lca_public_figures",
        "source": io_lca_dummy_repo.source,
        "years": io_lca_dummy_repo.years,
        "lcia_method": io_lca_dummy_repo.lcia_method,
        "fu_code": "L1.a",
        "r_f": ["FR", "DE"],
        "upstream_analysis": False,
        "aggreg_indices": False,
        "output_format": "csv",
    }
    report = deterministic_io_lca(
        **base_kwargs,
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    )

    assert report is not None
    assert report.figure_paths
    assert all(path.exists() for path in report.figure_paths)
    assert any(path.suffix.lower() == ".png" for path in report.figure_paths)

    skipped_report = deterministic_io_lca(
        **base_kwargs,
        figures=False,
        refresh=False,
    )

    assert skipped_report.reuse_status == "reused_exact"
    assert all(path.exists() for path in report.figure_paths)
    reused_figure_report = deterministic_io_lca(
        **base_kwargs,
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    )
    assert reused_figure_report.reuse_status == "reused_exact"
    assert reused_figure_report.figure_paths == report.figure_paths
    narrower_figure_report = deterministic_io_lca(
        **{**base_kwargs, "years": [io_lca_dummy_repo.available_year]},
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    )
    assert narrower_figure_report.figure_paths == report.figure_paths
    restyled_figure_report = deterministic_io_lca(
        **base_kwargs,
        figures=True,
        figure_format={"format": "svg", "dpi": 1},
        refresh=False,
    )
    assert all(path.suffix == ".svg" for path in restyled_figure_report.figure_paths)

    refreshed_figure_report = deterministic_io_lca(
        **base_kwargs,
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    )

    assert refreshed_figure_report.figure_paths
    assert all(path.exists() for path in refreshed_figure_report.figure_paths)


def test_deterministic_io_lca_generates_single_year_figures_when_requested(
    io_lca_dummy_repo,
) -> None:
    report = deterministic_io_lca(
        project_name="io_lca_public_single_year_figures",
        source=io_lca_dummy_repo.source,
        years=[io_lca_dummy_repo.available_year],
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_code="L1.a",
        r_f=["FR"],
        upstream_analysis=False,
        aggreg_indices=False,
        output_format="csv",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    )

    assert report.figure_paths
    assert all(path.exists() for path in report.figure_paths)
    assert all(
        path.name.endswith(f"__{io_lca_dummy_repo.available_year}.png")
        for path in report.figure_paths
    )


def test_deterministic_io_lca_public_figures_cover_odd_multi_impact_multi_year(
    io_lca_dummy_repo_factory,
) -> None:
    repo = io_lca_dummy_repo_factory(
        name="io_lca_public_multi_year_odd_impacts",
        impacts=["aal_child", "bifd_child", "oa_child"],
        parent_by_impact={"aal_child": "AAL", "bifd_child": "BI FD", "oa_child": "OA"},
        available_years=[2019, 2020],
        unavailable_years=[],
    )

    single = deterministic_io_lca(
        project_name="io_lca_public_single_year_odd_impacts",
        source=repo.source,
        years=[repo.available_year],
        lcia_method=repo.lcia_method,
        fu_code="L1.a",
        r_f=["FR"],
        upstream_analysis=False,
        aggreg_indices=False,
        output_format="csv",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    )

    assert single.figure_paths
    assert all(path.exists() for path in single.figure_paths)
    assert all(path.name.endswith(f"__{repo.available_year}.png") for path in single.figure_paths)

    report = deterministic_io_lca(
        project_name="io_lca_public_multi_year_odd_impacts",
        source=repo.source,
        years=repo.years,
        lcia_method=repo.lcia_method,
        fu_code="L1.a",
        r_f=["FR"],
        upstream_analysis=False,
        aggreg_indices=False,
        output_format="csv",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    )

    assert report.figure_paths
    assert all(path.exists() for path in report.figure_paths)
    assert all(not path.stem.endswith("__2020") for path in report.figure_paths)


def test_deterministic_io_lca_public_figures_cover_single_impact_layout(
    io_lca_dummy_repo_factory,
) -> None:
    repo = io_lca_dummy_repo_factory(
        name="io_lca_public_single_impact_figures",
        impacts=["AAL"],
        parent_by_impact={"AAL": "AAL"},
        available_years=[2019],
        unavailable_years=[],
    )

    report = deterministic_io_lca(
        project_name="io_lca_public_single_impact_figures",
        source=repo.source,
        years=[repo.available_year],
        lcia_method=repo.lcia_method,
        fu_code="L1.a",
        r_f=["FR"],
        upstream_analysis=False,
        aggreg_indices=False,
        output_format="csv",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    )

    assert report.figure_paths
    assert all(path.exists() for path in report.figure_paths)


def test_deterministic_io_lca_public_figures_return_no_paths_without_renderable_years(
    io_lca_dummy_repo,
) -> None:
    report = deterministic_io_lca(
        project_name="io_lca_public_unavailable_figure_year",
        source=io_lca_dummy_repo.source,
        years=[io_lca_dummy_repo.unavailable_year],
        lcia_method=io_lca_dummy_repo.lcia_method,
        fu_code="L1.a",
        r_f=["FR"],
        upstream_analysis=False,
        aggreg_indices=False,
        output_format="csv",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    )

    assert report.figure_paths == []


def test_build_io_lca_summary_covers_multi_folder_branches(tmp_path: Path) -> None:
    figure_path = tmp_path / "figures_a" / "plot.png"
    summary = build_io_lca_summary(
        source="exiobase_396_ixi",
        output_root=tmp_path / "outputs",
        resolved_years=[2019, 2020],
        covered_main_years={2019, 2020},
        covered_origin_years={2019},
        covered_stage_years={2020},
        skipped_method_years={},
        aggreg_indices=False,
        upstream_analysis=True,
        stage_outputs_enabled=True,
        reuse_status="computed",
        lca_results_dirs={tmp_path / "main_a", tmp_path / "main_b"},
        origin_dirs={tmp_path / "origin_a", tmp_path / "origin_b"},
        stages_dirs={tmp_path / "stage_a", tmp_path / "stage_b"},
        figure_paths=[figure_path],
    )

    rendered = "\n".join(summary)
    assert rendered


def test_deterministic_io_lca_validation_errors_are_public_and_fail_fast() -> None:
    with pytest.raises(ValueError):
        deterministic_io_lca(
            project_name="proj",
            source="exiobase_396_ixi",
            years=[2019],
            lcia_method="pb_lcia",
            fu_code="L1.a",
            aggreg_indices=cast(Any, "both"),
            figures=False,
        )

    with pytest.raises(ValueError):
        deterministic_io_lca(
            project_name="proj",
            source="oecd_v2025",
            years=[2019],
            lcia_method="pb_lcia",
            fu_code="L1.a",
            figures=False,
        )

    with pytest.raises(ValueError):
        deterministic_io_lca(
            project_name="proj",
            source="exiobase_396_ixi",
            years=[2019],
            lcia_method="pb_lcia",
            fu_code="L2.a.c",
            upstream_analysis=True,
            figures=False,
        )
