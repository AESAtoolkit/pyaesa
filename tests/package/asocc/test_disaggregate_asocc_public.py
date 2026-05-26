import pandas as pd

from pyaesa import deterministic_asocc, disaggregate_asocc


def _run_prerequisite_asocc(
    *,
    project_name: str,
    source: str,
    sectors: list[str],
    agg_sec: bool = False,
    agg_version: str | None = None,
) -> None:
    deterministic_asocc(
        project_name=project_name,
        source=source,
        years=[2005, 2006],
        fu_code="L2.a.a",
        s_p=sectors,
        agg_sec=agg_sec,
        agg_version="" if agg_version is None else agg_version,
        method_plan="one_step",
        one_step_methods=["UT(FD)"],
        output_format="csv",
        figures=False,
        refresh=True,
    )


def _run_prerequisite_pair_asocc(
    *,
    project_name: str,
    source: str,
    sectors: list[str],
    agg_sec: bool = False,
    agg_version: str | None = None,
) -> None:
    deterministic_asocc(
        project_name=project_name,
        source=source,
        years=[2005],
        fu_code="L2.c.b",
        r_c=["FR"],
        s_p=sectors,
        agg_sec=agg_sec,
        agg_version="" if agg_version is None else agg_version,
        method_plan="pairs",
        l1_l2_pairs=["PR(GDPcap)::UT(GVAa)"],
        output_format="csv",
        figures=False,
        refresh=True,
    )


def _disaggregation_config() -> dict:
    return {
        "target_agg_run": {
            "source": "oecd_v2025",
            "s_p": ["Energy", "Other"],
        },
        "ref_agg_run": {
            "source": "exiobase_396_ixi",
            "agg_sec": True,
            "agg_version": "energy_aggregate",
            "s_p": ["Energy", "Other"],
        },
        "ref_disagg_run": {
            "source": "exiobase_396_ixi",
            "s_p": ["Coal", "Gas"],
        },
        "disaggregation_specs": [
            {"agg_sector_label": "Energy", "disagg_sector_label": "Coal"},
            {"agg_sector_label": "Other", "disagg_sector_label": "Gas"},
        ],
        "new_disagg_version_name": "disagg_oecd_energy",
    }


def _base_asocc_args(project_name: str) -> dict:
    return {
        "project_name": project_name,
        "years": [2005, 2006],
        "fu_code": "L2.a.a",
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
    }


def _base_pair_asocc_args(project_name: str) -> dict:
    return {
        "project_name": project_name,
        "years": [2005],
        "fu_code": "L2.c.b",
        "r_c": ["FR"],
        "method_plan": "pairs",
        "l1_l2_pairs": ["PR(GDPcap)::UT(GVAa)"],
    }


def _prepare_repo(allocation_dummy_repo) -> None:
    years = [2005, 2006]
    allocation_dummy_repo.write_mrio_metadata(
        source="oecd_v2025",
        matrix_version=None,
        sectors_used=["Energy", "Other"],
        regions_used=["FR", "US"],
        years=years,
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_396_ixi",
        matrix_version=None,
        sectors_used=["Coal", "Gas"],
        regions_used=["FR", "US"],
        years=years,
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_396_ixi",
        matrix_version="energy_aggregate",
        sectors_used=["Energy", "Other"],
        regions_used=["FR", "US"],
        years=years,
    )
    allocation_dummy_repo.write_mrio_history(
        source="oecd_v2025",
        matrix_version=None,
        years=years,
    )
    allocation_dummy_repo.write_mrio_history(
        source="exiobase_396_ixi",
        matrix_version=None,
        years=years,
    )
    allocation_dummy_repo.write_mrio_history(
        source="exiobase_396_ixi",
        matrix_version="energy_aggregate",
        years=years,
    )


def test_disaggregate_asocc_public_write_reuse_and_refresh(allocation_dummy_repo) -> None:
    project_name = "disagg_public"
    _prepare_repo(allocation_dummy_repo)
    _run_prerequisite_asocc(
        project_name=project_name,
        source="oecd_v2025",
        sectors=["Energy", "Other"],
    )
    _run_prerequisite_asocc(
        project_name=project_name,
        source="exiobase_396_ixi",
        sectors=["Energy", "Other"],
        agg_sec=True,
        agg_version="energy_aggregate",
    )
    _run_prerequisite_asocc(
        project_name=project_name,
        source="exiobase_396_ixi",
        sectors=["Coal", "Gas"],
    )

    report = disaggregate_asocc(
        disaggregation_config=_disaggregation_config(),
        base_asocc_args=_base_asocc_args(project_name),
        output_format="csv",
        figures=False,
        refresh=True,
    )

    assert report is not None
    branch = report.branch_reports[0]
    assert branch.metadata_path.exists()
    assert branch.disaggregation_audit_path.exists()
    metadata = branch.metadata_path.read_text(encoding="utf-8")
    assert "target_agg_run" in metadata
    assert "ref_agg_run" in metadata
    assert "ref_disagg_run" in metadata

    output_root = (
        allocation_dummy_repo.repo_root / f"{project_name}" / "B1_asocc" / "disagg_oecd_energy"
    )
    output_files = sorted(output_root.rglob("UT(FD).csv"))
    assert len(output_files) == 1
    rows = pd.read_csv(output_files[0])
    assert set(rows["s_p"]) == {"Coal", "Gas"}
    assert {"2005", "2006"}.issubset(rows.columns)
    values = rows[["2005", "2006"]].to_numpy(dtype=float)
    assert bool((values > 0).all())

    reused = disaggregate_asocc(
        disaggregation_config=_disaggregation_config(),
        base_asocc_args=_base_asocc_args(project_name),
        output_format="csv",
        figures=False,
        refresh=False,
    )
    assert reused is not None
    assert reused.branch_reports[0].run_status == "reused_exact"
    assert reused.branch_reports[0].metadata_path == branch.metadata_path

    figure_report = disaggregate_asocc(
        disaggregation_config=_disaggregation_config(),
        base_asocc_args=_base_asocc_args(project_name),
        output_format="csv",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=False,
    )
    assert figure_report is not None
    assert figure_report.branch_reports[0].figure_paths

    stale_marker = output_root / "deterministic" / "stale.txt"
    stale_marker.write_text("old", encoding="utf-8")
    refreshed = disaggregate_asocc(
        disaggregation_config=_disaggregation_config(),
        base_asocc_args=_base_asocc_args(project_name),
        output_format="csv",
        figures=False,
        refresh=True,
    )
    assert refreshed is not None
    assert not stale_marker.exists()


def test_disaggregate_asocc_public_combined_method_branch(allocation_dummy_repo) -> None:
    project_name = "disagg_public_pair"
    _prepare_repo(allocation_dummy_repo)
    _run_prerequisite_pair_asocc(
        project_name=project_name,
        source="oecd_v2025",
        sectors=["Energy", "Other"],
    )
    _run_prerequisite_pair_asocc(
        project_name=project_name,
        source="exiobase_396_ixi",
        sectors=["Energy", "Other"],
        agg_sec=True,
        agg_version="energy_aggregate",
    )
    _run_prerequisite_pair_asocc(
        project_name=project_name,
        source="exiobase_396_ixi",
        sectors=["Coal", "Gas"],
    )

    report = disaggregate_asocc(
        disaggregation_config=_disaggregation_config(),
        base_asocc_args=_base_pair_asocc_args(project_name),
        output_format="csv",
        figures=False,
        refresh=True,
    )

    assert report is not None
    output_root = (
        allocation_dummy_repo.repo_root / f"{project_name}" / "B1_asocc" / "disagg_oecd_energy"
    )
    assert sorted(output_root.rglob("PR(GDPcap)_UT(GVAa).csv"))
