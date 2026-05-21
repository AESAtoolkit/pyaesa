import json
import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from pyaesa import uncertainty_io_lca
from pyaesa.process.mrios.utils.io.paths import (
    _get_group_map_path,
    _get_metadata_path,
    _get_year_saved_dir,
)
from pyaesa.shared.lcia.paths import carbon_account_cov_path


@pytest.mark.parametrize(
    "mc_parameters",
    [
        {"fixed": []},
        {"fixed": {"active": True, "n_runs": 1, "extra": 1}},
        {"fixed": {"active": "yes", "n_runs": 1}, "convergence": {"active": False}},
    ],
)
def test_uncertainty_io_lca_rejects_invalid_public_mc_parameter_blocks(
    mc_parameters: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args={},
            uncertainty_config={"mc_parameters": mc_parameters},
            figures=False,
        )


def _write_sector_group_map(*, io_lca_dummy_repo, rows: list[tuple[str, str]]) -> None:
    group_map_path = _get_group_map_path(
        io_lca_dummy_repo.source,
        kind="sec",
        group_version="elec",
    )
    group_map_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "original_classification": [row[0] for row in rows],
            "grouped_mrio": [row[1] for row in rows],
        }
    ).to_csv(group_map_path, index=False)


def _append_country_cov(
    label: str,
    cov: float = 0.2,
    *,
    asset_name: str = "reg_cbca_covs.csv",
) -> None:
    path = carbon_account_cov_path(asset_name=asset_name)
    if path.exists():
        frame = pd.read_csv(path)
    else:
        base = pd.read_csv(carbon_account_cov_path(asset_name="reg_cbca_covs.csv"))
        frame = base.loc[base["exio_code"].astype(str).eq("World")].copy()
    pd.concat(
        [
            frame.loc[frame["exio_code"].astype(str).ne(label)],
            pd.DataFrame({"exio_code": [label], "cov": [cov]}),
        ],
        ignore_index=True,
    ).to_csv(path, index=False)


def _add_grouped_sector_scope(*, io_lca_dummy_repo, year: int | None = None) -> None:
    effective_year = io_lca_dummy_repo.available_year if year is None else int(year)
    _write_sector_group_map(
        io_lca_dummy_repo=io_lca_dummy_repo,
        rows=[("A", "Electricity"), ("B", "Electricity")],
    )
    metadata_path = _get_metadata_path(io_lca_dummy_repo.source, matrix_version="elec")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "source": io_lca_dummy_repo.source,
                "version_tag": "elec",
                "labels": {"sectors_used": ["Electricity"]},
                "years": {
                    str(effective_year): {
                        "core": ["A", "L"],
                        "extensions": {io_lca_dummy_repo.lcia_method: {"available": True}},
                        "lcia_status": {io_lca_dummy_repo.lcia_method: {"available": True}},
                        "enacting_metrics": {
                            "units": {
                                "mrio_default_monetary": "EUR",
                                "mrio_by_metric": {
                                    "fd_rp_sp_rf": "EUR",
                                    "x_to_rc": "EUR",
                                },
                                "lcia_by_method": {
                                    io_lca_dummy_repo.lcia_method: {
                                        "AAL": "kg",
                                        "BI FD": "kg",
                                    }
                                },
                            }
                        },
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    grouped_dir = _get_year_saved_dir(
        io_lca_dummy_repo.source,
        effective_year,
        matrix_version="elec",
    )
    target_dir = grouped_dir / "enacting_metrics" / "level_2" / io_lca_dummy_repo.lcia_method
    target_dir.mkdir(parents=True, exist_ok=True)
    impacts = pd.Index(["AAL", "BI FD GHG"], name="impact")
    columns = pd.MultiIndex.from_tuples([("FR", "Electricity")], names=["r_c", "s_p"])
    pd.DataFrame(
        [[12.0], [36.0]],
        index=impacts,
        columns=columns,
    ).to_pickle(target_dir / "e_cba_td_rc_sp.pickle")


def _add_grouped_region_scope(*, io_lca_dummy_repo, include_2020: bool = False) -> None:
    group_map_path = _get_group_map_path(
        io_lca_dummy_repo.source,
        kind="reg",
        group_version="eu27",
    )
    group_map_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "original_classification": ["FR", "DE"],
            "grouped_mrio": ["EU27", "EU27"],
        }
    ).to_csv(group_map_path, index=False)
    metadata_path = _get_metadata_path(io_lca_dummy_repo.source, matrix_version="eu27")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    years_payload = {
        "2019": {
            "core": ["A", "L"],
            "extensions": {io_lca_dummy_repo.lcia_method: {"available": True}},
            "lcia_status": {io_lca_dummy_repo.lcia_method: {"available": True}},
            "enacting_metrics": {
                "units": {
                    "mrio_default_monetary": "EUR",
                    "mrio_by_metric": {
                        "fd_rp_sp_rf": "EUR",
                        "x_to_rc": "EUR",
                    },
                    "lcia_by_method": {
                        io_lca_dummy_repo.lcia_method: {
                            "AAL": "kg",
                            "BI FD": "kg",
                        }
                    },
                }
            },
        }
    }
    if include_2020:
        years_payload["2020"] = years_payload["2019"]
    metadata_path.write_text(
        json.dumps(
            {
                "source": io_lca_dummy_repo.source,
                "version_tag": "eu27",
                "labels": {"sectors_used": ["A", "B"]},
                "years": years_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for year in [2019, 2020] if include_2020 else [2019]:
        grouped_dir = _get_year_saved_dir(
            io_lca_dummy_repo.source,
            year,
            matrix_version="eu27",
        )
        target_dir = grouped_dir / "enacting_metrics" / "level_1" / io_lca_dummy_repo.lcia_method
        target_dir.mkdir(parents=True, exist_ok=True)
        impacts = pd.Index(["AAL", "BI FD GHG"], name="impact")
        pd.DataFrame(
            [[54.0], [41.0]],
            index=impacts,
            columns=pd.Index(["EU27"], name="r_f"),
        ).to_pickle(target_dir / "e_cba_fd_reg.pickle")


def _base_args(*, io_lca_dummy_repo, project_name: str) -> dict[str, Any]:
    return {
        "project_name": project_name,
        "source": io_lca_dummy_repo.source,
        "years": io_lca_dummy_repo.years,
        "lcia_method": io_lca_dummy_repo.lcia_method,
        "fu_code": "L1.a",
        "r_f": ["FR", "DE"],
    }


def _uncertainty_config(mc_parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "mc_parameters": mc_parameters,
        "lcia_uncertainty": {"active": True, "sector_cov_mapping": {}},
    }


def test_uncertainty_io_lca_fixed_outputs_and_reuse(
    io_lca_dummy_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    kwargs = {
        "base_io_lca_args": _base_args(
            io_lca_dummy_repo=io_lca_dummy_repo,
            project_name="uncertainty_io_lca_fixed",
        ),
        "uncertainty_config": _uncertainty_config(
            {"fixed": {"active": True, "n_runs": 4}, "convergence": {"active": False}}
        ),
        "output_format": "csv_compact",
    }

    manifest = uncertainty_io_lca(
        refresh=True,
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        **cast(Any, kwargs),
    ).manifest
    first_output = capsys.readouterr().out

    assert manifest.family == "io_lca"
    assert first_output
    assert manifest.active_sources == ("lcia_uncertainty",)
    assert manifest.completed_runs == 4
    assert manifest.sobol is None
    assert manifest.artifacts is not None
    assert "figure_paths" in manifest.artifacts
    assert "figure_request" in manifest.artifacts
    figure_paths = [Path(str(path)) for path in manifest.artifacts["figure_paths"]]
    assert all(path.exists() for path in figure_paths)
    old_timestamp = 1_700_000_000
    for path in figure_paths:
        os.utime(path, (old_timestamp, old_timestamp))
    cached_mtimes = {path: path.stat().st_mtime_ns for path in figure_paths}
    identity = pd.read_csv(manifest.artifacts["public_row_identity"])
    assert identity.columns.tolist() == [
        "public_row_id",
        "lcia_method",
        "year",
        "impact",
        "r_f",
        "impact_unit",
    ]
    runs = pd.read_csv(manifest.artifacts["lca_runs"])
    assert runs.columns.tolist() == ["run_index", "0", "1", "2", "3"]
    assert runs["run_index"].tolist() == [0, 1, 2, 3]
    summary = pd.read_csv(manifest.artifacts["summary_stats_runs"])
    assert {"mean", "std", "p5", "median", "p95"}.issubset(summary.columns)
    methods = pd.read_csv(manifest.artifacts["source_methods"])
    assert set(methods["primary_cov_key"]) == {"FR", "DE"}
    readme = Path(str(manifest.artifacts["results_readme"]))
    readme_text = readme.read_text(encoding="utf-8")
    assert readme_text.strip()
    assert all(len(line) <= 100 for line in readme_text.splitlines())

    reused = uncertainty_io_lca(
        refresh=False,
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        **cast(Any, kwargs),
    ).manifest
    reuse_output = capsys.readouterr().out

    assert reused.run_id == manifest.run_id
    assert reused.artifacts is not None
    assert "figure_paths" in reused.artifacts
    assert reused.artifacts["figure_request"] == manifest.artifacts["figure_request"]
    assert {path: path.stat().st_mtime_ns for path in figure_paths} == cached_mtimes
    assert reuse_output

    rerendered = uncertainty_io_lca(
        refresh=False,
        figures=True,
        figure_format={"format": "png", "dpi": 11},
        **cast(Any, kwargs),
    ).manifest
    assert rerendered.run_id == manifest.run_id
    assert rerendered.artifacts is not None
    assert rerendered.artifacts["figure_request"] != manifest.artifacts["figure_request"]
    assert any(path.stat().st_mtime_ns != cached_mtimes[path] for path in figure_paths)

    missing_path = Path(str(rerendered.artifacts["figure_paths"][0]))
    missing_path.unlink()
    restored = uncertainty_io_lca(
        refresh=False,
        figures=True,
        figure_format={"format": "png", "dpi": 11},
        **cast(Any, kwargs),
    ).manifest
    assert restored.run_id == manifest.run_id
    assert missing_path.exists()
    stale_run_file = Path(str(manifest.artifacts["scope_manifest"])).parents[1] / "stale.txt"
    stale_run_file.write_text("stale", encoding="utf-8")
    stale_upstream_file = (
        Path(str(manifest.deterministic_prerequisites[0]["deterministic_paths"][0])).parent
        / "stale.txt"
    )
    stale_upstream_file.write_text("stale", encoding="utf-8")
    refreshed = uncertainty_io_lca(
        refresh=True,
        figures=False,
        **cast(Any, kwargs),
    ).manifest
    assert refreshed.status == "complete"
    assert not stale_run_file.exists()
    assert not stale_upstream_file.exists()


def test_uncertainty_io_lca_writes_parquet_outputs(io_lca_dummy_repo) -> None:
    manifest = uncertainty_io_lca(
        base_io_lca_args={
            **_base_args(
                io_lca_dummy_repo=io_lca_dummy_repo,
                project_name="uncertainty_io_lca_parquet",
            ),
            "years": [io_lca_dummy_repo.available_year],
        },
        uncertainty_config=_uncertainty_config(
            {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
        ),
        output_format="parquet",
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    identity = pd.read_parquet(manifest.artifacts["public_row_identity"])
    runs = pd.read_parquet(manifest.artifacts["lca_runs"])
    summary = pd.read_parquet(manifest.artifacts["summary_stats_runs"])
    assert identity["public_row_id"].tolist() == [0, 1, 2, 3]
    assert runs["run_index"].tolist() == [0, 1]
    assert summary["public_row_id"].tolist() == [0, 1, 2, 3]


def test_uncertainty_io_lca_generates_single_year_figures_when_requested(io_lca_dummy_repo) -> None:
    manifest = uncertainty_io_lca(
        base_io_lca_args={
            **_base_args(
                io_lca_dummy_repo=io_lca_dummy_repo,
                project_name="uncertainty_io_lca_single_year_figures",
            ),
            "years": [io_lca_dummy_repo.available_year],
        },
        uncertainty_config=_uncertainty_config(
            {"fixed": {"active": True, "n_runs": 3}, "convergence": {"active": False}}
        ),
        output_format="csv_compact",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    paths = [Path(path) for path in manifest.artifacts["figure_paths"]]
    assert paths
    assert all(path.exists() for path in paths)
    assert all(path.name.endswith(f"__{io_lca_dummy_repo.available_year}.png") for path in paths)


def test_uncertainty_io_lca_public_figures_cover_odd_impact_panels(
    io_lca_dummy_repo_factory,
) -> None:
    repo = io_lca_dummy_repo_factory(
        name="uncertainty_io_lca_odd_impact_figures",
        impacts=["aal_child", "bifd_child", "oa_child"],
        parent_by_impact={"aal_child": "AAL", "bifd_child": "BI FD", "oa_child": "OA"},
        available_years=[2019, 2020],
        unavailable_years=[],
    )

    single = uncertainty_io_lca(
        base_io_lca_args={
            **_base_args(io_lca_dummy_repo=repo, project_name="uncertainty_iolca_odd_single"),
            "years": [repo.available_year],
            "r_f": ["FR"],
        },
        uncertainty_config=_uncertainty_config(
            {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
        ),
        output_format="csv_compact",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    ).manifest
    multi = uncertainty_io_lca(
        base_io_lca_args={
            **_base_args(io_lca_dummy_repo=repo, project_name="uncertainty_iolca_odd_multi"),
            "r_f": ["FR"],
        },
        uncertainty_config=_uncertainty_config(
            {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
        ),
        output_format="csv_compact",
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    ).manifest

    assert single.artifacts is not None
    assert multi.artifacts is not None
    paths = [
        Path(path) for path in [*single.artifacts["figure_paths"], *multi.artifacts["figure_paths"]]
    ]
    assert paths
    assert all(path.exists() for path in paths)


def test_uncertainty_io_lca_convergence_reports_unreached_and_cleans_cache(
    io_lca_dummy_repo,
) -> None:
    manifest = uncertainty_io_lca(
        base_io_lca_args={
            **_base_args(
                io_lca_dummy_repo=io_lca_dummy_repo,
                project_name="uncertainty_io_lca_convergence",
            ),
            "years": [io_lca_dummy_repo.available_year],
        },
        uncertainty_config=_uncertainty_config(
            {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 4, "rtol": 1e-12, "stable_runs": 3},
            }
        ),
        refresh=True,
    ).manifest

    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is False
    assert manifest.convergence["completed_runs"] == 4
    assert manifest.artifacts is not None
    run_root = Path(str(manifest.artifacts["scope_manifest"])).parent.parent
    assert list(run_root.glob(".summary_values_*.dat")) == []

    reused = uncertainty_io_lca(
        base_io_lca_args={
            **_base_args(
                io_lca_dummy_repo=io_lca_dummy_repo,
                project_name="uncertainty_io_lca_convergence",
            ),
            "years": [io_lca_dummy_repo.available_year],
        },
        uncertainty_config=_uncertainty_config(
            {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 4, "rtol": 1e-12, "stable_runs": 3},
            }
        ),
        refresh=False,
    ).manifest

    assert reused.run_id == manifest.run_id


def test_uncertainty_io_lca_convergence_can_reach_stability(io_lca_dummy_repo) -> None:
    manifest = uncertainty_io_lca(
        base_io_lca_args={
            **_base_args(
                io_lca_dummy_repo=io_lca_dummy_repo,
                project_name="uncertainty_io_lca_convergence_reached",
            ),
            "years": [io_lca_dummy_repo.available_year],
        },
        uncertainty_config=_uncertainty_config(
            {
                "fixed": {"active": False},
                "convergence": {"active": True, "max_runs": 4, "rtol": 10.0, "stable_runs": 2},
            }
        ),
        refresh=True,
    ).manifest

    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is True
    assert manifest.completed_runs == 4


def test_uncertainty_io_lca_l1_aggregated_rows_use_aggregate_country_cov(
    io_lca_dummy_repo,
) -> None:
    _append_country_cov("DE, FR", asset_name="reg_cbca_covs_aggreg_indices.csv")
    manifest = uncertainty_io_lca(
        base_io_lca_args={
            "project_name": "uncertainty_io_lca_aggregated_l1",
            "source": io_lca_dummy_repo.source,
            "years": io_lca_dummy_repo.years,
            "lcia_method": io_lca_dummy_repo.lcia_method,
            "fu_code": "L1.b",
            "r_p": ["FR", "DE"],
            "aggreg_indices": True,
        },
        uncertainty_config=_uncertainty_config(
            {"fixed": {"active": True, "n_runs": 3}, "convergence": {"active": False}}
        ),
        figures=True,
        figure_format={"format": "png", "dpi": 10},
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    identity = pd.read_csv(manifest.artifacts["public_row_identity"])
    assert identity.columns.tolist() == [
        "public_row_id",
        "lcia_method",
        "year",
        "impact",
        "r_p",
        "impact_unit",
    ]
    assert identity["r_p"].unique().tolist() == ["DE, FR"]
    methods = pd.read_csv(manifest.artifacts["source_methods"])
    assert methods["primary_cov_key"].tolist() == ["DE, FR"]
    runs = pd.read_csv(manifest.artifacts["lca_runs"])
    assert runs.shape == (3, 3)
    figure_paths = [Path(path) for path in manifest.artifacts["figure_paths"]]
    assert figure_paths
    assert all(path.exists() for path in figure_paths)


def test_uncertainty_io_lca_l2_requires_and_uses_sector_mapping(io_lca_dummy_repo) -> None:
    base_args = {
        "project_name": "uncertainty_io_lca_l2",
        "source": io_lca_dummy_repo.source,
        "years": [io_lca_dummy_repo.available_year],
        "lcia_method": io_lca_dummy_repo.lcia_method,
        "fu_code": "L2.c.b",
        "r_c": ["FR"],
        "s_p": ["A", "B"],
    }

    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args=base_args,
            uncertainty_config=_uncertainty_config(
                {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
            ),
            refresh=True,
        )

    manifest = uncertainty_io_lca(
        base_io_lca_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "lcia_uncertainty": {"sector_cov_mapping": {"A": "Electricity", "B": "Electricity"}},
        },
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    identity = pd.read_csv(manifest.artifacts["public_row_identity"])
    assert identity[["r_c", "s_p"]].drop_duplicates().values.tolist() == [
        ["FR", "A"],
        ["FR", "B"],
    ]
    methods = pd.read_csv(manifest.artifacts["source_methods"])
    assert methods["primary_cov_kind"].tolist() == ["sector"]
    assert methods["primary_cov_key"].tolist() == ["Electricity"]


def test_uncertainty_io_lca_grouped_sector_uses_grouped_public_cov(
    io_lca_dummy_repo,
) -> None:
    _add_grouped_sector_scope(io_lca_dummy_repo=io_lca_dummy_repo)

    manifest = uncertainty_io_lca(
        base_io_lca_args={
            "project_name": "uncertainty_io_lca_grouped_sector",
            "source": io_lca_dummy_repo.source,
            "group_sec": True,
            "group_version": "elec",
            "years": [io_lca_dummy_repo.available_year],
            "lcia_method": io_lca_dummy_repo.lcia_method,
            "fu_code": "L2.c.b",
            "r_c": ["FR"],
            "s_p": ["Electricity"],
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "lcia_uncertainty": {"sector_cov_mapping": {"Electricity": "Electricity"}},
        },
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    identity = pd.read_csv(manifest.artifacts["public_row_identity"])
    assert identity["s_p"].tolist() == ["Electricity", "Electricity"]
    methods = pd.read_csv(manifest.artifacts["source_methods"])
    assert methods["primary_cov_key"].tolist() == ["Electricity"]
    runs = pd.read_csv(manifest.artifacts["lca_runs"])
    assert runs.shape == (2, 3)


def test_uncertainty_io_lca_grouped_region_uses_group_cov(
    io_lca_dummy_repo,
) -> None:
    _add_grouped_region_scope(io_lca_dummy_repo=io_lca_dummy_repo)

    manifest = uncertainty_io_lca(
        base_io_lca_args={
            "project_name": "uncertainty_io_lca_grouped_region",
            "source": io_lca_dummy_repo.source,
            "group_reg": True,
            "group_version": "eu27",
            "years": [io_lca_dummy_repo.available_year],
            "lcia_method": io_lca_dummy_repo.lcia_method,
            "fu_code": "L1.a",
            "r_f": ["EU27"],
        },
        uncertainty_config=_uncertainty_config(
            {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
        ),
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    identity = pd.read_csv(manifest.artifacts["public_row_identity"])
    assert identity["r_f"].tolist() == ["EU27", "EU27"]
    methods = pd.read_csv(manifest.artifacts["source_methods"])
    assert methods["primary_cov_key"].tolist() == ["EU27"]


def test_uncertainty_io_lca_validation_errors_are_public(io_lca_dummy_repo) -> None:
    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args={
                "source": io_lca_dummy_repo.source,
                "years": [io_lca_dummy_repo.available_year],
                "lcia_method": io_lca_dummy_repo.lcia_method,
                "fu_code": "L1.a",
            },
            uncertainty_config=_uncertainty_config(
                {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
            ),
            refresh=True,
        )

    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args={
                **_base_args(
                    io_lca_dummy_repo=io_lca_dummy_repo,
                    project_name="uncertainty_io_lca_bad_bool",
                ),
                "group_reg": "false",
            },
            uncertainty_config=_uncertainty_config(
                {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
            ),
            refresh=True,
        )

    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args={
                **_base_args(
                    io_lca_dummy_repo=io_lca_dummy_repo,
                    project_name=" ",
                ),
            },
            uncertainty_config=_uncertainty_config(
                {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
            ),
            refresh=True,
        )

    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args={
                **_base_args(
                    io_lca_dummy_repo=io_lca_dummy_repo,
                    project_name="uncertainty_io_lca_bad_base",
                ),
                "figures": False,
            },
            uncertainty_config=_uncertainty_config(
                {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
            ),
            refresh=True,
        )

    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args=_base_args(
                io_lca_dummy_repo=io_lca_dummy_repo,
                project_name="uncertainty_io_lca_disabled",
            ),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "lcia_uncertainty": {"active": False},
            },
            refresh=True,
        )

    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args={
                **_base_args(
                    io_lca_dummy_repo=io_lca_dummy_repo,
                    project_name="uncertainty_io_lca_all_skipped",
                ),
                "years": [io_lca_dummy_repo.unavailable_year],
            },
            uncertainty_config=_uncertainty_config(
                {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
            ),
            refresh=True,
        )

    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args=_base_args(
                io_lca_dummy_repo=io_lca_dummy_repo,
                project_name="uncertainty_io_lca_bad_mapping",
            ),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "lcia_uncertainty": {"sector_cov_mapping": ["bad"]},
            },
            refresh=True,
        )

    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args=_base_args(
                io_lca_dummy_repo=io_lca_dummy_repo,
                project_name="uncertainty_io_lca_bad_lcia_param",
            ),
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "lcia_uncertainty": {"unknown": True},
            },
            refresh=True,
        )

    country_cov_path = carbon_account_cov_path(asset_name="reg_cbca_covs.csv")
    country_covs = pd.read_csv(country_cov_path)
    country_covs.loc[country_covs["exio_code"].ne("FR")].to_csv(country_cov_path, index=False)
    try:
        with pytest.raises(ValueError):
            uncertainty_io_lca(
                base_io_lca_args={
                    **_base_args(
                        io_lca_dummy_repo=io_lca_dummy_repo,
                        project_name="uncertainty_io_lca_missing_country_cov",
                    ),
                    "years": [io_lca_dummy_repo.available_year],
                    "r_f": ["FR"],
                },
                uncertainty_config=_uncertainty_config(
                    {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}}
                ),
                refresh=True,
            )
    finally:
        country_covs.to_csv(country_cov_path, index=False)

    with pytest.raises(ValueError):
        uncertainty_io_lca(
            base_io_lca_args={
                "project_name": "uncertainty_io_lca_missing_sector_cov",
                "source": io_lca_dummy_repo.source,
                "years": [io_lca_dummy_repo.available_year],
                "lcia_method": io_lca_dummy_repo.lcia_method,
                "fu_code": "L2.c.b",
                "r_c": ["FR"],
                "s_p": ["A"],
            },
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 2},
                    "convergence": {"active": False},
                },
                "lcia_uncertainty": {"sector_cov_mapping": {"A": "unknown"}},
            },
            refresh=True,
        )
