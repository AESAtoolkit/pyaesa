from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from pyaesa import uncertainty_asocc
from pyaesa.asocc.runtime.scope.branch_resolution import AsoccDeterministicPathScope, asocc_l2_dir
from pyaesa.asocc.runtime.scope.persisted_scope import (
    AsoccPersistedComputeSignature,
    AsoccPersistedRunScope,
)
from pyaesa.asocc.runtime.request.scope import AsoccScope
from pyaesa.asocc.figures.product_renderers import prepare_plot_rows as prepare_asocc_plot_rows
from pyaesa.asocc.figures.multi_method_renderer import expand_generic_lcia_rows
from pyaesa.asocc.figures.product_renderers import plot_scope as plot_asocc_scope
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import (
    ASOCC_VALUE_COLUMN,
    LoadedAsoccFinalRows,
    read_deterministic_asocc_rows,
    validate_single_l2_reuse_year_per_identity,
    validate_single_reference_year_per_identity,
)
from pyaesa.asocc.uncertainty.schema.public_rows import (
    align_asocc_lcia_public_axis,
    expand_rows_to_reference_lcia_axis,
)
from pyaesa.asocc.uncertainty.figures.product_renderers import (
    plot_band_scope,
    plot_mean_line_scope,
    plot_violin_scope,
)
from pyaesa.asocc.uncertainty.figures.scope_planner import FigureContext, VALUE_ARRAY_COLUMN
from pyaesa.asocc.uncertainty.figures.row_reader import collapsed_violin_rows
from pyaesa.asocc.uncertainty.figures.render import _multi_year_jobs, _single_year_jobs
from pyaesa.asocc.uncertainty.io.paths import AsoccUncertaintyRunPaths
from pyaesa.asocc.uncertainty.sources.names import INTER_METHOD_SOURCE
from pyaesa.asocc.uncertainty.engine.evaluation.summary_identity import (
    ASOCC_SUMMARY_SCOPE_COLUMN,
    ASOCC_SUMMARY_SCOPE_INTER_METHOD,
    ASOCC_SUMMARY_SCOPE_PER_METHOD,
)
from pyaesa.asocc.uncertainty.sources.lcia import LCIASupportRowCache
from pyaesa.asocc.uncertainty.sources.lcia_support import (
    combined_route_coefficients,
    support_rows,
)
from pyaesa.asocc.uncertainty.sources.projection import (
    apply_projection_uncertainty_to_matrix,
    build_projection_plan,
    projection_value_matrix_for_indices,
    sample_projection_indices,
)
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.uncertainty_assessment.io.run_writers import (
    CompactRunMatrixWriter,
    SparseRunRows,
    SparseRunRowsWriter,
)
from pyaesa.shared.uncertainty_assessment.io.run_matrix_reader import iter_compact_run_matrix
from pyaesa.shared.uncertainty_assessment.run_state.manifest import build_manifest
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.lcia.paths import carbon_account_cov_path
from tests.package.helpers.acc_dummy_repo import prepare_exiobase_repo_with_years
from tests.package.helpers.asr_dummy_repo import prepare_static_asr_pb_lcia_repo


def _run_root(repo_root: Path, *, project_name: str, run_id: str) -> Path:
    return repo_root / f"{project_name}" / "B1_asocc" / "exiobase_396_ixi" / "monte_carlo" / run_id


def _read_asocc_runs(run_root: Path) -> pd.DataFrame:
    identity = pd.read_csv(run_root / "results" / "public_row_identity.csv")
    matrix = _read_asocc_run_matrix(run_root=run_root, column_count=len(identity))
    runs = matrix.melt(id_vars="run_index", var_name="public_row_id", value_name="asocc")
    runs["public_row_id"] = runs["public_row_id"].astype(int)
    return runs.merge(identity, on="public_row_id", how="left").sort_values(
        ["run_index", "public_row_id"],
        ignore_index=True,
    )


def _read_asocc_run_matrix(*, run_root: Path, column_count: int) -> pd.DataFrame:
    columns = [str(index) for index in range(column_count)]
    chunks: list[pd.DataFrame] = []
    for run_indices, values in iter_compact_run_matrix(
        path=run_root / "results" / "asocc_runs.csv",
        output_format="csv_compact",
        column_count=column_count,
    ):
        chunk = pd.DataFrame(values, columns=columns)
        chunk.insert(0, "run_index", run_indices)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def _lcia_base_args(*, project_name: str, reference_years: list[int] | None) -> dict:
    return {
        "project_name": project_name,
        "source": "exiobase_396_ixi",
        "years": [2005, 2006],
        "reference_years": reference_years,
        "fu_code": "L2.a.a",
        "method_plan": "one_step_pairs",
        "one_step_methods": ["AR(E^{CBA_FD})"],
        "l1_l2_pairs": [
            "AR(E^{CBA_FD})::UT(FD)",
            "EG(Pop)::AR(E^{CBA_FD})",
            "EG(Pop)::UT(FD)",
        ],
        "lcia_method": "gwp100_lcia",
        "r_p": ["FR"],
        "s_p": ["D"],
        "r_f": ["FR"],
        "l1_reg_aggreg": "pre",
        "ssp_scenario": ["SSP2"],
    }


def _pb_lcia_base_args(
    *,
    project_name: str,
    years: list[int],
    reference_years: list[int] | None = None,
) -> dict:
    args = _lcia_base_args(project_name=project_name, reference_years=reference_years)
    args["years"] = years
    args["lcia_method"] = "pb_lcia"
    return args


def _lcia_config(*, n_runs: int = 2) -> dict:
    return {
        "mc_parameters": {
            "fixed": {"active": True, "n_runs": n_runs},
            "convergence": {"active": False},
        },
        "lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}},
        "projection_uncertainty": {"active": False},
        "reference_year_uncertainty": {"active": False},
        "inter_method_uncertainty": {"active": False},
    }


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


def _manifest_figure_paths(manifest) -> list[Path]:
    assert manifest.artifacts is not None
    return [Path(str(path)) for path in manifest.artifacts.get("figure_paths", [])]


def _fast_figure_format() -> dict:
    return {"format": "svg", "dpi": 1}


def _asocc_uncertainty_figure_frame(*, impacts: list[str], years: list[int]) -> pd.DataFrame:
    rows = []
    for method_index, method in enumerate(["UT(FD)", "AR(E^{CBA_FD})"]):
        for impact_index, impact in enumerate(impacts):
            for year_index, year in enumerate(years):
                value = 0.2 + method_index * 0.08 + impact_index * 0.04 + year_index * 0.02
                route = "" if year_index == 0 else "regression_proj"
                rows.append(
                    {
                        "__method": method,
                        "l1_l2_method": method,
                        "lcia_method": "pb_lcia",
                        "impact": impact,
                        "year": year,
                        ASOCC_SSP_SCENARIO_COLUMN: "" if year_index == 0 else "SSP2",
                        ASOCC_TIME_ROUTE_PUBLIC_COLUMN: route,
                        "mean": value,
                        "median": value,
                        "p5": value * 0.8,
                        "p25": value * 0.9,
                        "p75": value * 1.1,
                        "p95": value * 1.2,
                        VALUE_ARRAY_COLUMN: np.array([value * 0.85, value, value * 1.15]),
                    }
                )
    return pd.DataFrame(rows)


def test_uncertainty_asocc_product_renderers_cover_panel_and_single_axis_paths(
    read_only_project_repo: Path,
    tmp_path: Path,
) -> None:
    del read_only_project_repo
    single = _asocc_uncertainty_figure_frame(impacts=["AAL"], years=[2005, 2006])
    multi = _asocc_uncertainty_figure_frame(
        impacts=["AAL", "BI FD", "OA"],
        years=[2005, 2006],
    )
    no_transition_multi = multi.copy()
    no_transition_multi[ASOCC_TIME_ROUTE_PUBLIC_COLUMN] = ""
    no_transition_multi[ASOCC_SSP_SCENARIO_COLUMN] = ""
    single_year_multi = multi.loc[multi["year"].eq(2005)].copy()
    single_year_single = single.loc[single["year"].eq(2005)].copy()

    single_band_paths = plot_band_scope(
        frame=single,
        output_stem=tmp_path / "single_band",
        title="aSoCC single band",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=True,
        include_method_in_label=True,
    )
    panel_band_paths = plot_band_scope(
        frame=multi,
        output_stem=tmp_path / "panel_band",
        title="aSoCC panel band",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    no_transition_panel_band_paths = plot_band_scope(
        frame=no_transition_multi,
        output_stem=tmp_path / "panel_band_no_transition",
        title="aSoCC panel band no transition",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    mean_frame = prepare_asocc_plot_rows(
        multi.drop(columns=[VALUE_ARRAY_COLUMN]).assign(value=multi["mean"])
    )
    mean_paths = plot_mean_line_scope(
        frame=mean_frame,
        requested_years=[2005, 2006],
        output_stem=tmp_path / "mean_line",
        title="aSoCC mean line",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=True,
        include_method_in_label=True,
    )
    single_violin_paths = plot_violin_scope(
        frame=single_year_multi,
        output_stem=tmp_path / "single_violin",
        title="aSoCC single violin",
        dpi=1,
        output_format="svg",
        group_legend=False,
        include_impact_in_label=True,
        include_method_in_label=True,
    )
    grouped_single_violin_paths = plot_violin_scope(
        frame=single_year_single,
        output_stem=tmp_path / "grouped_single_violin",
        title="aSoCC grouped single violin",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )
    panel_violin_paths = plot_violin_scope(
        frame=single_year_multi,
        output_stem=tmp_path / "panel_violin",
        title="aSoCC panel violin",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=False,
        include_method_in_label=True,
    )

    paths = [
        *single_band_paths,
        *panel_band_paths,
        *no_transition_panel_band_paths,
        *mean_paths,
        *single_violin_paths,
        *grouped_single_violin_paths,
        *panel_violin_paths,
    ]
    assert paths
    assert all(path.exists() for path in paths)


def test_deterministic_asocc_renderers_expand_generic_lcia_and_hide_unused_panel(
    read_only_project_repo: Path,
    tmp_path: Path,
) -> None:
    del read_only_project_repo
    generic = pd.DataFrame(
        {
            "__method": ["UT(FD)", "UT(FD)", "UT(FD)", "UT(FD)"],
            "l1_l2_method": ["UT(FD)", "UT(FD)", "UT(FD)", "UT(FD)"],
            "lcia_method": [None, "pb_lcia", "pb_lcia", "gwp100_lcia"],
            "impact": [None, "AAL", "SOD", "GWP_100"],
            "year": [2030, 2030, 2030, 2030],
            "value": [0.1, 0.2, 0.3, 0.4],
        }
    )
    expanded = expand_generic_lcia_rows(generic)
    assert len(expanded) == 6

    rows = prepare_asocc_plot_rows(
        pd.DataFrame(
            {
                "__method": ["UT(FD)", "UT(FD)", "UT(FD)"],
                "l1_l2_method": ["UT(FD)", "UT(FD)", "UT(FD)"],
                "lcia_method": ["pb_lcia", "pb_lcia", "gwp100_lcia"],
                "impact": ["AAL", "SOD", "GWP_100"],
                "impact_unit": ["ratio", "ratio", "ratio"],
                "year": [2030, 2030, 2030],
                "value": [0.2, 0.3, 0.4],
            }
        )
    )
    paths = plot_asocc_scope(
        frame=rows,
        requested_years=[2030],
        output_stem=tmp_path / "single_year_panels",
        title="aSoCC single year panels",
        dpi=1,
        output_format="svg",
        group_legend=True,
        include_impact_in_label=True,
    )

    assert paths == [tmp_path / "single_year_panels.svg"]
    assert paths[0].exists()


def test_uncertainty_asocc_accepts_multi_lcia_public_scope(allocation_dummy_repo) -> None:
    prepare_static_asr_pb_lcia_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        years=[2005],
        impacts=["AAL"],
    )
    base_args = _lcia_base_args(
        project_name="asocc_uncertainty_multi_lcia_public_scope",
        reference_years=[2005],
    )
    base_args["years"] = [2005]
    base_args["lcia_method"] = ["gwp100_lcia", "pb_lcia"]

    manifest = uncertainty_asocc(
        base_asocc_args=base_args,
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "lcia_uncertainty": {"active": False},
            "projection_uncertainty": {"active": False},
            "reference_year_uncertainty": {"active": False},
            "inter_method_uncertainty": {"active": False},
        },
        sobol_parameters={"active": False},
        output_format="csv_compact",
        figures=False,
        refresh=True,
    ).manifest
    assert manifest.artifacts is not None
    identity = pd.read_csv(manifest.artifacts["public_row_identity"])

    assert manifest.completed_runs == 2
    assert set(identity["lcia_method"].dropna().astype(str)) == {"gwp100_lcia", "pb_lcia"}


def test_non_lcia_rows_expand_with_values_when_lcia_public_axis_exists() -> None:
    non_lcia_only = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FD)"],
            "lcia_method": [None],
            "impact": [None],
            "allocated_share": [0.4],
        }
    )
    values = pd.DataFrame([[0.4]]).to_numpy(dtype="float64")
    unchanged, unchanged_values = align_asocc_lcia_public_axis(
        frame=non_lcia_only,
        values=values,
    )
    assert unchanged.equals(non_lcia_only)
    assert unchanged_values.tolist() == [[0.4]]

    mixed = pd.DataFrame(
        {
            "l1_l2_method": ["AR(E^{CBA_FD})", "AR(E^{CBA_FD})", "UT(FD)"],
            "lcia_method": ["gwp100_lcia", "pb_lcia", None],
            "impact": ["GWP_100", "climate", None],
            "allocated_share": [0.3, 0.5, 0.4],
        }
    )
    expanded, expanded_values = align_asocc_lcia_public_axis(
        frame=mixed,
        values=pd.DataFrame([[0.3, 0.5, 0.4]]).to_numpy(dtype="float64"),
    )

    assert expanded["lcia_method"].tolist() == [
        "gwp100_lcia",
        "pb_lcia",
        "gwp100_lcia",
        "pb_lcia",
    ]
    assert expanded["impact"].tolist() == ["GWP_100", "climate", "GWP_100", "climate"]
    assert expanded_values.tolist() == [[0.3, 0.5, 0.4, 0.4]]
    reference_expanded = expand_rows_to_reference_lcia_axis(
        rows=non_lcia_only,
        reference=mixed,
    )
    assert reference_expanded["lcia_method"].tolist() == ["gwp100_lcia", "pb_lcia"]


def test_uncertainty_asocc_public_figures_cover_no_source_and_lcia_runs(
    allocation_dummy_repo,
) -> None:
    lcia_manifest = uncertainty_asocc(
        base_asocc_args=_lcia_base_args(
            project_name="asocc_uncertainty_figures_lcia",
            reference_years=[2005],
        ),
        uncertainty_config=_lcia_config(n_runs=2),
        output_format="csv_compact",
        figures=True,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest

    lcia_paths = _manifest_figure_paths(lcia_manifest)

    assert lcia_paths
    assert all(path.exists() for path in lcia_paths)
    assert any("multi_method" in path.parts for path in lcia_paths)
    assert any("per_method" in path.parts for path in lcia_paths)

    no_product_manifest = uncertainty_asocc(
        base_asocc_args=_lcia_base_args(
            project_name="asocc_uncertainty_figures_lcia",
            reference_years=[2005],
        ),
        uncertainty_config=_lcia_config(n_runs=2),
        output_format="csv_compact",
        figures=True,
        figure_options={"per_method": False, "multi_method": False, "inter_method": False},
        figure_format=_fast_figure_format(),
        refresh=False,
    ).manifest
    assert _manifest_figure_paths(no_product_manifest) == []


def test_uncertainty_asocc_public_figures_cover_scalar_year_and_single_method(
    allocation_dummy_repo,
) -> None:
    scalar_manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": "asocc_uncertainty_figures_scalar_non_lcia",
            "source": "exiobase_396_ixi",
            "years": 2005,
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
        },
        output_format="csv_compact",
        figures=True,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest
    scalar_multi_method_manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": "asocc_uncertainty_figures_scalar_multi_method_non_lcia",
            "source": "exiobase_396_ixi",
            "years": [2005],
            "fu_code": "L2.a.b",
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {"active": False},
        },
        output_format="csv_compact",
        figures=True,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest
    active_lcia_single_method_manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": "asocc_uncertainty_figures_single_lcia_method_active",
            "source": "exiobase_396_ixi",
            "years": [2005],
            "reference_years": [2005],
            "fu_code": "L2.a.b",
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)"],
            "lcia_method": "gwp100_lcia",
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config=_lcia_config(n_runs=2),
        output_format="csv_compact",
        figures=True,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=[2020, 2021, 2022, 2023, 2024],
        scenario_years=[2025],
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_3102_ixi",
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=[2020, 2021, 2022, 2023, 2024],
    )
    allocation_dummy_repo.write_mrio_history(
        source="exiobase_3102_ixi",
        matrix_version=None,
        years=[2020, 2021, 2022, 2023, 2024],
    )
    active_single_method_manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": "asocc_uncertainty_figures_single_method_active",
            "source": "exiobase_3102_ixi",
            "years": [2025],
            "reference_years": [2020],
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
            "projection_mode": "regression",
            "reg_window": [2020, 2021, 2022],
            "ssp_scenario": ["SSP2"],
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "projection_uncertainty": {},
        },
        output_format="csv_compact",
        figures=True,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest

    scalar_paths = _manifest_figure_paths(scalar_manifest)
    scalar_multi_method_paths = _manifest_figure_paths(scalar_multi_method_manifest)
    active_paths = _manifest_figure_paths(active_single_method_manifest)
    active_lcia_paths = _manifest_figure_paths(active_lcia_single_method_manifest)

    assert (
        Path(str(scalar_manifest.artifacts["scope_manifest"])).parent / "composite_phase_index.json"
    ).exists()
    assert scalar_paths
    assert scalar_multi_method_paths
    assert active_paths
    assert active_lcia_paths
    assert any("multi_method" in path.parts for path in scalar_multi_method_paths)
    assert all(
        path.exists()
        for path in [
            *scalar_paths,
            *scalar_multi_method_paths,
            *active_paths,
            *active_lcia_paths,
        ]
    )
    assert all("multi_method" not in path.parts for path in [*active_paths, *active_lcia_paths])


def test_uncertainty_asocc_public_figures_validate_inactive_reference_axis(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args=_lcia_base_args(
                project_name="asocc_uncertainty_figures_inactive_reference_axis",
                reference_years=[2005, 2006],
            ),
            uncertainty_config=_lcia_config(n_runs=2),
            output_format="csv_compact",
            figures=True,
            figure_format=_fast_figure_format(),
            refresh=True,
        )


def test_uncertainty_asocc_public_figures_cover_inter_method_sparse_runs(
    allocation_dummy_repo,
) -> None:
    prepare_exiobase_repo_with_years(
        allocation_dummy_repo,
        historical_years=list(range(1995, 2007)),
        scenario_years=[2030, 2031],
    )
    multi_manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": "asocc_uncertainty_figures_inter_method_multi_year",
            "source": "exiobase_396_ixi",
            "years": [2005, 2006],
            "reference_years": [2005],
            "fu_code": "L2.a.b",
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "lcia_method": "gwp100_lcia",
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
        },
        output_format="csv_compact",
        figures=True,
        figure_options={"per_method": False, "multi_method": False, "inter_method": True},
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest

    paths = _manifest_figure_paths(multi_manifest)
    assert multi_manifest.artifacts is not None
    summary = pd.read_csv(multi_manifest.artifacts["summary_stats_runs"])

    assert paths
    assert all(path.exists() for path in paths)
    assert any("inter_method" in path.parts for path in paths)
    assert set(summary[ASOCC_SUMMARY_SCOPE_COLUMN]) == {
        ASOCC_SUMMARY_SCOPE_PER_METHOD,
        ASOCC_SUMMARY_SCOPE_INTER_METHOD,
    }
    inter_summary = summary.loc[
        summary[ASOCC_SUMMARY_SCOPE_COLUMN].eq(ASOCC_SUMMARY_SCOPE_INTER_METHOD)
    ]
    assert inter_summary[["l1_l2_method", "l1_method", "l2_method"]].isna().all().all()
    assert (
        summary.loc[
            summary[ASOCC_SUMMARY_SCOPE_COLUMN].eq(ASOCC_SUMMARY_SCOPE_PER_METHOD),
            "l1_l2_method",
        ]
        .notna()
        .any()
    )

    single_manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": "asocc_uncertainty_figures_inter_method_single_year",
            "source": "exiobase_396_ixi",
            "years": [2005],
            "reference_years": [2005],
            "fu_code": "L2.a.b",
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
            "lcia_method": "gwp100_lcia",
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
        },
        output_format="csv_compact",
        figures=True,
        figure_options={"per_method": True, "multi_method": True, "inter_method": True},
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest
    single_paths = _manifest_figure_paths(single_manifest)
    assert any("inter_method" in path.parts for path in single_paths)
    assert any("multi_method" in path.parts for path in single_paths)
    assert any("per_method" in path.parts for path in single_paths)
    assert all(path.exists() for path in single_paths)


def test_uncertainty_asocc_collapsed_violin_rows_cover_ssp_ownership() -> None:
    collapsed = collapsed_violin_rows(
        rows=pd.DataFrame(
            {
                "year": [2030, 2030],
                "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
                "impact": ["GWP_100", "GWP_100"],
                ASOCC_SSP_SCENARIO_COLUMN: ["", "SSP2"],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical", "l2_reuse_year"],
                "l1_l2_method": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
                "l1_method": ["EG(Pop)", "PR(GDPcap)"],
                "l2_method": ["UT(FD)", "UT(FD)"],
                "__method": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
                VALUE_ARRAY_COLUMN: [
                    np.array([0.1, 0.2], dtype=np.float64),
                    np.array([0.3, 0.4], dtype=np.float64),
                ],
            }
        )
    )

    assert collapsed[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2"]
    assert collapsed[ASOCC_TIME_ROUTE_PUBLIC_COLUMN].tolist() == ["historical"]
    np.testing.assert_allclose(collapsed[VALUE_ARRAY_COLUMN].iloc[0], [0.1, 0.2, 0.3, 0.4])


def test_uncertainty_asocc_job_planner_covers_compact_single_year_multi_method(
    tmp_path: Path,
) -> None:
    paths = AsoccUncertaintyRunPaths(
        run_root=tmp_path / "run",
        public_row_identity=tmp_path / "run" / "results" / "public_row_identity.csv",
        public_runs=tmp_path / "run" / "results" / "asocc_runs.csv",
        summary_stats_runs=tmp_path / "run" / "results" / "summary_stats_runs.csv",
        results_readme=tmp_path / "run" / "results" / "README.txt",
        source_methods=tmp_path / "run" / "logs" / "source_methods.csv",
        inter_method_tree_csv=tmp_path / "run" / "figures" / "tree.csv",
        inter_method_tree_figure_base=tmp_path / "run" / "figures" / "tree",
        sobol_indices=tmp_path / "run" / "results" / "sobol" / "sobol_indices.csv",
        sobol_source_summary=tmp_path / "run" / "results" / "sobol" / "summary.csv",
        sobol_readme=tmp_path / "run" / "results" / "sobol" / "README.txt",
        scope_manifest=tmp_path / "run" / "logs" / "scope_manifest.json",
    )
    paths.public_row_identity.parent.mkdir(parents=True, exist_ok=True)
    identity = pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "year": [2005, 2005],
            "l1_l2_method": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "l1_method": ["EG(Pop)", "PR(GDPcap)"],
            "l2_method": ["UT(FD)", "UT(FD)"],
            "r_p": ["FR", "FR"],
            "s_p": ["D", "D"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical", "historical"],
        }
    )
    identity.to_csv(paths.public_row_identity, index=False)
    with CompactRunMatrixWriter(path=paths.public_runs, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=np.array([0], dtype=np.int64),
            values=np.array([[0.1, 0.2]], dtype=np.float64),
            batch_index=0,
        )
    context = FigureContext(
        manifest=build_manifest(
            family="asocc",
            mode="fixed",
            output_format="csv_compact",
            completed_runs=1,
            active_sources=(),
            arguments={"years": [2005], "fu_code": "L2.a.b"},
        ),
        paths=paths,
        figures_root=tmp_path / "figures",
        requested_years=(2005,),
        requested_ssps=(),
        fu_code="L2.a.b",
        output_format="csv_compact",
        figure_output_format="svg",
        figure_dpi=1,
        per_method=True,
        multi_method=True,
        inter_method=False,
        active_sources=(),
    )

    jobs = list(_single_year_jobs(context=context, identity=identity, summary=pd.DataFrame()))
    assert {job.kind for job in jobs} == {"multi_method", "per_method"}
    no_per_jobs = list(
        _single_year_jobs(
            context=FigureContext(
                manifest=context.manifest,
                paths=context.paths,
                figures_root=context.figures_root,
                requested_years=context.requested_years,
                requested_ssps=context.requested_ssps,
                fu_code=context.fu_code,
                output_format=context.output_format,
                figure_output_format=context.figure_output_format,
                figure_dpi=context.figure_dpi,
                per_method=False,
                multi_method=True,
                inter_method=False,
                active_sources=(),
            ),
            identity=identity,
            summary=pd.DataFrame(),
        )
    )
    assert {job.kind for job in no_per_jobs} == {"multi_method"}

    sparse_paths = AsoccUncertaintyRunPaths(
        run_root=tmp_path / "sparse_run",
        public_row_identity=tmp_path / "sparse_run" / "results" / "public_row_identity.csv",
        public_runs=tmp_path / "sparse_run" / "results" / "asocc_runs.csv",
        summary_stats_runs=tmp_path / "sparse_run" / "results" / "summary_stats_runs.csv",
        results_readme=tmp_path / "sparse_run" / "results" / "README.txt",
        source_methods=tmp_path / "sparse_run" / "logs" / "source_methods.csv",
        inter_method_tree_csv=tmp_path / "sparse_run" / "figures" / "tree.csv",
        inter_method_tree_figure_base=tmp_path / "sparse_run" / "figures" / "tree",
        sobol_indices=tmp_path / "sparse_run" / "results" / "sobol" / "sobol_indices.csv",
        sobol_source_summary=tmp_path / "sparse_run" / "results" / "sobol" / "summary.csv",
        sobol_readme=tmp_path / "sparse_run" / "results" / "sobol" / "README.txt",
        scope_manifest=tmp_path / "sparse_run" / "logs" / "scope_manifest.json",
    )
    sparse_paths.public_row_identity.parent.mkdir(parents=True, exist_ok=True)
    identity.to_csv(sparse_paths.public_row_identity, index=False)
    with SparseRunRowsWriter(path=sparse_paths.public_runs, output_format="csv_compact") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.array([0, 0], dtype=np.int64),
                public_row_id=np.array([0, 1], dtype=np.int64),
                values=np.array([0.1, 0.2], dtype=np.float64),
                value_column="asocc",
            ),
            batch_index=0,
        )
    inactive_inter_jobs = list(
        _single_year_jobs(
            context=FigureContext(
                manifest=context.manifest,
                paths=sparse_paths,
                figures_root=tmp_path / "sparse_figures",
                requested_years=context.requested_years,
                requested_ssps=context.requested_ssps,
                fu_code=context.fu_code,
                output_format=context.output_format,
                figure_output_format=context.figure_output_format,
                figure_dpi=context.figure_dpi,
                per_method=False,
                multi_method=False,
                inter_method=False,
                active_sources=(INTER_METHOD_SOURCE,),
            ),
            identity=identity,
            summary=pd.DataFrame(),
        )
    )
    assert inactive_inter_jobs == []

    summary = identity.drop(columns=["public_row_id"]).assign(
        **{
            ASOCC_SUMMARY_SCOPE_COLUMN: ASOCC_SUMMARY_SCOPE_PER_METHOD,
            "mean": [0.1, 0.2],
            "std": [0.0, 0.0],
            "min": [0.1, 0.2],
            "p5": [0.1, 0.2],
            "p25": [0.1, 0.2],
            "median": [0.1, 0.2],
            "p75": [0.1, 0.2],
            "p95": [0.1, 0.2],
            "max": [0.1, 0.2],
        }
    )
    multi_year_context = FigureContext(
        manifest=context.manifest,
        paths=context.paths,
        figures_root=context.figures_root,
        requested_years=(2005, 2006),
        requested_ssps=(),
        fu_code=context.fu_code,
        output_format=context.output_format,
        figure_output_format=context.figure_output_format,
        figure_dpi=context.figure_dpi,
        per_method=False,
        multi_method=False,
        inter_method=False,
        active_sources=(INTER_METHOD_SOURCE,),
    )
    assert (
        list(_multi_year_jobs(context=multi_year_context, identity=identity, summary=summary)) == []
    )
    non_inter_multi_year_context = FigureContext(
        manifest=context.manifest,
        paths=context.paths,
        figures_root=context.figures_root,
        requested_years=(2005, 2006),
        requested_ssps=(),
        fu_code=context.fu_code,
        output_format=context.output_format,
        figure_output_format=context.figure_output_format,
        figure_dpi=context.figure_dpi,
        per_method=False,
        multi_method=False,
        inter_method=False,
        active_sources=(),
    )
    assert (
        list(
            _multi_year_jobs(
                context=non_inter_multi_year_context,
                identity=identity,
                summary=summary,
            )
        )
        == []
    )


def test_uncertainty_asocc_public_figures_cover_inter_method_transition_bands(
    allocation_dummy_repo,
) -> None:
    allocation_dummy_repo.set_processed_pop_gdp_years(
        historical_years=[2020, 2021, 2022, 2023, 2024],
        scenario_years=[2025],
    )
    allocation_dummy_repo.write_mrio_metadata(
        source="exiobase_3102_ixi",
        matrix_version=None,
        sectors_used=["D", "X"],
        regions_used=["FR", "US"],
        years=[2020, 2021, 2022, 2023, 2024],
    )
    allocation_dummy_repo.write_mrio_history(
        source="exiobase_3102_ixi",
        matrix_version=None,
        years=[2020, 2021, 2022, 2023, 2024],
    )

    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": "asocc_uncertainty_figures_inter_method_transition",
            "source": "exiobase_3102_ixi",
            "years": [2024, 2025],
            "reference_years": [2020],
            "fu_code": "L2.a.a",
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
            "projection_mode": "regression",
            "reg_window": [2020, 2021, 2022],
            "ssp_scenario": ["SSP2"],
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "inter_method_uncertainty": {},
        },
        output_format="csv_compact",
        figures=True,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest
    paths = _manifest_figure_paths(manifest)

    assert paths
    assert any("inter_method" in path.parts for path in paths)
    assert all(path.exists() for path in paths)


def test_uncertainty_asocc_multi_impact_band_marks_transitions(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "year": [2024, 2025, 2024, 2025],
            "lcia_method": ["pb_lcia"] * 4,
            "impact": ["SOD", "SOD", "AAL", "AAL"],
            "l1_l2_method": ["UT(FD)"] * 4,
            "l1_method": [pd.NA] * 4,
            "l2_method": ["UT(FD)"] * 4,
            ASOCC_SSP_SCENARIO_COLUMN: ["", "SSP2", "", "SSP2"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [
                "historical",
                "regression_proj",
                "historical",
                "regression_proj",
            ],
            **{
                column: np.array([0.1, 0.2, 0.15, 0.25], dtype=np.float64)
                for column in ("mean", "std", "min", "p5", "p25", "median", "p75", "p95", "max")
            },
        }
    )

    paths = plot_band_scope(
        frame=frame,
        output_stem=tmp_path / "transition_band",
        title="transition band",
        dpi=10,
        output_format="png",
        group_legend=True,
        include_impact_in_label=False,
    )

    assert all(path.exists() for path in paths)


def test_uncertainty_asocc_public_figures_cover_multi_impact_panels(
    allocation_dummy_repo,
) -> None:
    prepare_exiobase_repo_with_years(
        allocation_dummy_repo,
        historical_years=list(range(1995, 2007)),
        scenario_years=[],
    )
    allocation_dummy_repo.write_lcia_support(
        source="exiobase_396_ixi",
        matrix_version=None,
        lcia_method="pb_lcia",
        available_years=[2005, 2006],
        impacts=["aal_child", "bifd_child", "oa_child"],
        impact_parents={"aal_child": "AAL", "bifd_child": "BI FD", "oa_child": "OA"},
    )
    multi_manifest = uncertainty_asocc(
        base_asocc_args=_pb_lcia_base_args(
            project_name="asocc_uncertainty_figures_pb_multi",
            years=[2005, 2006],
            reference_years=[2005],
        ),
        uncertainty_config=_lcia_config(n_runs=2),
        output_format="csv_compact",
        figures=True,
        figure_format=_fast_figure_format(),
        refresh=True,
    ).manifest

    multi_paths = _manifest_figure_paths(multi_manifest)

    assert any("multi_method" in path.parts for path in multi_paths)
    assert all(path.exists() for path in multi_paths)


def test_public_lcia_axis_owner_expands_full_non_lcia_reference_axis() -> None:
    rows = pd.DataFrame({"l1_l2_method": ["UT(FD)"], "allocated_share": [0.4]})
    reference_axis = pd.DataFrame(
        {
            "lcia_method": ["gwp100_lcia", "pb_lcia"],
            "impact": ["GWP_100", "climate"],
        }
    )

    expanded, values = align_asocc_lcia_public_axis(
        frame=rows,
        values=pd.DataFrame([[0.4]]).to_numpy(dtype="float64"),
        reference_axis=reference_axis,
    )

    assert expanded["lcia_method"].tolist() == ["gwp100_lcia", "pb_lcia"]
    assert expanded["impact"].tolist() == ["GWP_100", "climate"]
    assert values.tolist() == [[0.4, 0.4]]


def test_inactive_projection_reuse_validation_uses_final_l2_reuse_year() -> None:
    validate_single_l2_reuse_year_per_identity(
        rows=pd.DataFrame({"year": [2030], "allocated_share": [0.2]})
    )

    stable = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FDa)", "UT(FDa)"],
            "year": [2030, 2030],
            "l2_reuse_year": [2005, 2005],
            "allocated_share": [0.2, 0.2],
        }
    )
    validate_single_l2_reuse_year_per_identity(rows=stable)

    conflicting = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FDa)", "UT(FDa)"],
            "year": [2030, 2030],
            "l2_reuse_year": [2005, 2006],
            "allocated_share": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError):
        validate_single_l2_reuse_year_per_identity(rows=conflicting)


def test_inactive_reference_year_validation_uses_represented_identity() -> None:
    stable = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FDa)", "UT(GVAa)"],
            "year": [2030, 2030],
            "reference_year": [2005, 2006],
            "allocated_share": [0.2, 0.3],
        }
    )
    validate_single_reference_year_per_identity(rows=stable)

    conflicting = pd.DataFrame(
        {
            "l1_l2_method": ["UT(FDa)", "UT(GVAa)"],
            "year": [2030, 2030],
            "reference_year": [2005, 2006],
            "allocated_share": [0.2, 0.3],
        }
    )
    with pytest.raises(ValueError):
        validate_single_reference_year_per_identity(
            rows=conflicting,
            sampled_identity_columns=("l1_l2_method",),
        )


def test_uncertainty_asocc_lcia_writes_public_runs_logs_and_summaries(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_public"
    manifest = uncertainty_asocc(
        base_asocc_args=_lcia_base_args(project_name=project_name, reference_years=[2005]),
        uncertainty_config=_lcia_config(),
        output_format="csv_compact",
        refresh=True,
    ).manifest
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )

    runs = _read_asocc_runs(run_root)
    source_methods = pd.read_csv(run_root / "logs" / "source_methods.csv")
    summary = pd.read_csv(run_root / "results" / "summary_stats_runs.csv")

    assert manifest.active_sources == ("lcia_uncertainty",)
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1]
    assert runs.groupby("run_index").size().nunique() == 1
    assert runs["lcia_method"].isna().sum() == 0
    assert runs["impact"].isna().sum() == 0
    assert set(runs["l1_l2_method"]) == {
        "AR(E^{CBA_FD})",
        "AR(E^{CBA_FD})_UT(FD)",
        "EG(Pop)_AR(E^{CBA_FD})",
        "EG(Pop)_UT(FD)",
    }
    sampled_identity = runs.loc[
        (runs["l1_l2_method"] == "AR(E^{CBA_FD})")
        & (runs["year"] == 2005)
        & (runs["reference_year"] == 2005),
        ["run_index", "asocc"],
    ].sort_values("run_index")
    assert sampled_identity["asocc"].nunique() == 2
    repeated_non_lcia = runs.loc[
        runs["l1_l2_method"].eq("EG(Pop)_UT(FD)"),
        ["run_index", "year", "lcia_method", "impact", "asocc"],
    ]
    assert repeated_non_lcia["impact"].nunique() == 1
    assert repeated_non_lcia.groupby(["run_index", "year"])["asocc"].nunique().max() == 1
    assert (
        repeated_non_lcia.groupby(["year", "lcia_method", "impact"])["asocc"].nunique().max() == 1
    )

    cov_scope = set(
        source_methods[
            [
                "applied_bucket",
                "primary_cov_kind",
                "primary_cov_key",
                "reference_cov_kind",
                "reference_cov_key",
            ]
        ].itertuples(index=False, name=None)
    )
    assert cov_scope == {
        ("level_1", "country", "FR", "world", "World"),
        ("l2_in_l1", "sector", "Electricity", "country", "FR"),
        ("l2_vs_global", "sector", "Electricity", "world", "World"),
    }
    assert source_methods["formula"].str.contains("primary_cov_value").all()
    assert {
        "mean",
        "std",
        "min",
        "p5",
        "p25",
        "median",
        "p75",
        "p95",
        "max",
    }.issubset(summary.columns)
    assert summary[ASOCC_SUMMARY_SCOPE_COLUMN].unique().tolist() == [ASOCC_SUMMARY_SCOPE_PER_METHOD]


def test_uncertainty_asocc_lcia_aggregated_indices_use_full_cov_label(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_aggregate_cov"
    aggregate_label = "FR, US"
    _append_country_cov(aggregate_label, asset_name="reg_cbca_covs_group_indices.csv")

    manifest = uncertainty_asocc(
        base_asocc_args={
            **_lcia_base_args(project_name=project_name, reference_years=[2005]),
            "r_p": ["FR", "US"],
            "r_f": ["FR", "US"],
            "group_indices": True,
        },
        uncertainty_config=_lcia_config(n_runs=1),
        output_format="csv_compact",
        figures=False,
        refresh=True,
    ).manifest

    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    identity = pd.read_csv(run_root / "results" / "public_row_identity.csv")
    source_methods = pd.read_csv(run_root / "logs" / "source_methods.csv")

    assert aggregate_label in set(identity["r_p"].dropna().astype(str))
    assert "etc." not in set(identity["r_p"].dropna().astype(str))
    assert aggregate_label in set(source_methods["primary_cov_key"].astype(str)) | set(
        source_methods["reference_cov_key"].astype(str)
    )


def test_uncertainty_asocc_lcia_source_is_removed_when_no_lcia_methods(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_no_targets"
    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "years": [2005],
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config=_lcia_config(n_runs=2),
        output_format="csv_compact",
        refresh=True,
    ).manifest

    assert manifest.active_sources == ()
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)
    assert runs["asocc"].nunique() == 1


def test_uncertainty_asocc_projection_source_is_removed_when_no_targets(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_projection_no_targets"
    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "years": [2005],
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "projection_uncertainty": {},
        },
        output_format="csv_compact",
        refresh=True,
    ).manifest

    assert manifest.active_sources == ()
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)
    assert runs["asocc"].nunique() == 1


def test_uncertainty_asocc_reference_year_source_is_removed_when_no_targets(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_reference_year_no_targets"
    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "years": [2005],
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "r_p": ["FR"],
            "s_p": ["D"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "reference_year_uncertainty": {},
        },
        output_format="csv_compact",
        refresh=True,
    ).manifest

    assert manifest.active_sources == ()
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)
    assert runs["asocc"].nunique() == 1


def test_uncertainty_asocc_lcia_and_reference_year_use_compact_outputs(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_reference_year_compact"
    manifest = uncertainty_asocc(
        base_asocc_args=_lcia_base_args(project_name=project_name, reference_years=[2005, 2006]),
        uncertainty_config={
            **_lcia_config(n_runs=2),
            "reference_year_uncertainty": {},
        },
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 4},
            "convergence": {"active": False},
        },
        figures=False,
        refresh=True,
    ).manifest
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    identity = pd.read_csv(run_root / "results" / "public_row_identity.csv")
    matrix = _read_asocc_run_matrix(run_root=run_root, column_count=len(identity))
    runs = _read_asocc_runs(run_root)

    assert matrix.columns.tolist() == ["run_index", *identity["public_row_id"].astype(str)]
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1]
    assert bool(runs["asocc"].notna().to_numpy().all())


def test_uncertainty_asocc_lcia_samples_l1_final_rows(allocation_dummy_repo) -> None:
    project_name = "asocc_uncertainty_lcia_l1"
    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "years": [2005],
            "reference_years": [2005],
            "fu_code": "L1.a",
            "method_plan": "default",
            "l1_methods": ["AR(E^{CBA_FD})"],
            "lcia_method": "gwp100_lcia",
            "r_f": ["FR"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config=_lcia_config(n_runs=2),
        output_format="csv_compact",
        refresh=True,
    ).manifest
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)
    source_methods = pd.read_csv(run_root / "logs" / "source_methods.csv")

    assert runs["run_index"].tolist() == [0, 1]
    assert runs["asocc"].nunique() == 2
    assert source_methods["applied_bucket"].tolist() == ["level_1"]


def test_uncertainty_asocc_lcia_samples_combined_rows_without_direct_rows(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_combined_only"
    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "years": [2005],
            "reference_years": [2005],
            "fu_code": "L2.a.a",
            "method_plan": "pairs",
            "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FD)"],
            "lcia_method": "gwp100_lcia",
            "r_p": ["FR"],
            "s_p": ["D"],
            "r_f": ["FR"],
            "l1_reg_aggreg": "pre",
        },
        uncertainty_config=_lcia_config(n_runs=2),
        output_format="csv_compact",
        refresh=True,
    ).manifest
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)

    assert runs["l1_l2_method"].unique().tolist() == ["AR(E^{CBA_FD})_UT(FD)"]
    assert runs["asocc"].nunique() == 2


def test_uncertainty_asocc_lcia_historical_reuse_uses_final_l2_reuse_year(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_historical_reuse"
    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "years": [2030],
            "reference_years": [2005],
            "fu_code": "L2.a.b",
            "method_plan": "pairs",
            "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FDa)"],
            "lcia_method": "gwp100_lcia",
            "r_p": ["FR"],
            "s_p": ["D"],
            "r_f": ["FR"],
            "l1_reg_aggreg": "pre",
            "ssp_scenario": ["SSP2"],
            "projection_mode": "historical_reuse",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005],
        },
        uncertainty_config=_lcia_config(n_runs=1),
        output_format="csv_compact",
        refresh=True,
    ).manifest
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)

    assert runs["year"].unique().tolist() == [2030]
    assert runs["l2_reuse_year"].unique().tolist() == [2005]
    assert bool(runs["asocc"].notna().to_numpy().all())
    assert bool(runs["asocc"].gt(0.0).to_numpy().all())


def test_uncertainty_asocc_requires_projection_uncertainty_for_multiple_l2_reuse_years(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_multiple_l2_reuse_years"
    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args={
                "project_name": project_name,
                "source": "exiobase_396_ixi",
                "years": [2030],
                "reference_years": [2005],
                "fu_code": "L2.a.b",
                "method_plan": "pairs",
                "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FDa)"],
                "lcia_method": "gwp100_lcia",
                "r_p": ["FR"],
                "s_p": ["D"],
                "r_f": ["FR"],
                "l1_reg_aggreg": "pre",
                "ssp_scenario": ["SSP2"],
                "projection_mode": "historical_reuse",
                "reg_window": [2005, 2006],
                "l2_reuse_years": [2005, 2006],
            },
            uncertainty_config=_lcia_config(n_runs=1),
            output_format="csv_compact",
            refresh=True,
        )


def test_uncertainty_asocc_projection_uncertainty_samples_l2_reuse_year_axis(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_projection_l2_reuse_year"
    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "years": [2005, 2030],
            "reference_years": [2005],
            "fu_code": "L2.a.b",
            "method_plan": "pairs",
            "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FDa)"],
            "lcia_method": "gwp100_lcia",
            "r_p": ["FR"],
            "s_p": ["D"],
            "r_f": ["FR"],
            "l1_reg_aggreg": "pre",
            "ssp_scenario": ["SSP2"],
            "projection_mode": "historical_reuse",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005, 2006],
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 4},
                "convergence": {"active": False},
            },
            "projection_uncertainty": {},
        },
        output_format="csv_compact",
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 4},
            "convergence": {"active": False},
        },
        figures=False,
        refresh=True,
    ).manifest
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)
    identity = pd.read_csv(run_root / "results" / "public_row_identity.csv")
    source_methods = pd.read_csv(run_root / "logs" / "source_methods.csv")
    summary = pd.read_csv(run_root / "results" / "summary_stats_runs.csv")

    assert manifest.active_sources == ("projection_uncertainty",)
    assert manifest.sobol is not None
    assert manifest.sobol["mode"] == "fixed"
    assert "l2_reuse_year" not in identity.columns
    assert "l2_reuse_year" not in summary.columns
    assert summary[ASOCC_SUMMARY_SCOPE_COLUMN].unique().tolist() == [ASOCC_SUMMARY_SCOPE_PER_METHOD]
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1, 2, 3]
    assert runs.groupby("run_index").size().nunique() == 1
    assert source_methods["source_name"].tolist() == ["projection_uncertainty"]
    assert source_methods["notes"].str.contains("2005;2006").all()


def test_uncertainty_asocc_projection_and_reference_year_use_selected_rows(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_projection_reference_year"
    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "years": [2030],
            "reference_years": [2005, 2006],
            "fu_code": "L2.a.b",
            "method_plan": "pairs",
            "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FDa)"],
            "lcia_method": "gwp100_lcia",
            "r_p": ["FR"],
            "s_p": ["D"],
            "r_f": ["FR"],
            "l1_reg_aggreg": "pre",
            "ssp_scenario": ["SSP2"],
            "projection_mode": "historical_reuse",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005, 2006],
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "projection_uncertainty": {},
            "reference_year_uncertainty": {},
        },
        output_format="csv_compact",
        refresh=True,
    ).manifest
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    runs = _read_asocc_runs(run_root)
    source_methods = pd.read_csv(run_root / "logs" / "source_methods.csv")

    assert manifest.active_sources == ("projection_uncertainty", "reference_year_uncertainty")
    assert runs["run_index"].tolist() == [0, 1]
    assert source_methods["source_name"].tolist() == [
        "projection_uncertainty",
        "reference_year_uncertainty",
    ]


def test_projection_owner_keeps_historical_passthrough_rows() -> None:
    loaded = cast(
        LoadedAsoccFinalRows,
        type(
            "ProjectionLoadedRows",
            (),
            {
                "base_asocc_args": {"fu_code": "L2.a.b"},
                "final_bucket": "l2_vs_global",
                "requested_years": [2005, 2030],
                "rows": pd.DataFrame(
                    {
                        "l1_l2_method": ["UT(FDa)", "UT(FDa)", "UT(FDa)"],
                        "year": [2005, 2030, 2030],
                        "l2_reuse_year": [None, 2005, 2006],
                        "allocated_share": [0.1, 0.2, 0.4],
                    }
                ),
            },
        )(),
    )

    plan = build_projection_plan(loaded=loaded)
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=3)
    values = projection_value_matrix_for_indices(
        plan=plan,
        batch=batch,
        selected_indices=sample_projection_indices(plan=plan, batch=batch),
    )

    assert plan.sampled_rows["year"].tolist() == [2030]
    assert values[:, 0].tolist() == [0.1, 0.1]
    assert values.shape == (2, 2)


def test_projection_owner_collapses_existing_compact_matrix() -> None:
    loaded = cast(
        LoadedAsoccFinalRows,
        type(
            "ProjectionLoadedRows",
            (),
            {
                "base_asocc_args": {"fu_code": "L2.a.b"},
                "final_bucket": "l2_vs_global",
                "requested_years": [2030],
                "rows": pd.DataFrame(
                    {
                        "l1_l2_method": ["UT(FDa)", "UT(FDa)", "UT(FDa)"],
                        "year": [2005, 2030, 2030],
                        "l2_reuse_year": [None, 2005, 2006],
                        "allocated_share": [0.1, 0.2, 0.4],
                    }
                ),
            },
        )(),
    )
    plan = build_projection_plan(loaded=loaded)
    batch = RunBatch(batch_index=0, start_run_index=0, stop_run_index=2, rng_seed=3)
    selected = pd.Series([0, 1]).to_numpy()

    template, values = apply_projection_uncertainty_to_matrix(
        template=loaded.rows,
        values=pd.DataFrame([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]]).to_numpy(),
        plan=plan,
        batch=batch,
        selected_indices=selected,
    )

    assert "l2_reuse_year" not in template.columns
    assert values.tolist() == [[1.0, 2.0], [10.0, 40.0]]


def test_uncertainty_asocc_lcia_plus_projection_collapses_reuse_axis(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_projection"
    manifest = uncertainty_asocc(
        base_asocc_args={
            "project_name": project_name,
            "source": "exiobase_396_ixi",
            "years": [2030],
            "reference_years": [2005],
            "fu_code": "L2.a.b",
            "method_plan": "pairs",
            "l1_l2_pairs": ["AR(E^{CBA_FD})::UT(FDa)"],
            "lcia_method": "gwp100_lcia",
            "r_p": ["FR"],
            "s_p": ["D"],
            "r_f": ["FR"],
            "l1_reg_aggreg": "pre",
            "ssp_scenario": ["SSP2"],
            "projection_mode": "historical_reuse",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005, 2006],
        },
        uncertainty_config={
            **_lcia_config(n_runs=2),
            "projection_uncertainty": {},
        },
        output_format="csv_compact",
        refresh=True,
    ).manifest
    run_root = _run_root(
        allocation_dummy_repo.repo_root,
        project_name=project_name,
        run_id=manifest.run_id,
    )
    identity = pd.read_csv(run_root / "results" / "public_row_identity.csv")
    runs = _read_asocc_runs(run_root)

    assert manifest.active_sources == ("lcia_uncertainty", "projection_uncertainty")
    assert "l2_reuse_year" not in identity.columns
    assert sorted(runs["run_index"].unique().tolist()) == [0, 1]


def test_uncertainty_asocc_deterministic_row_loader_preserves_ssp_path_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "UT(TD)__ssp2.csv"
    pd.DataFrame(
        {
            "l1_l2_method": ["UT(TD)"],
            "l2_method": ["UT(TD)"],
            "s_p": ["D"],
            "r_p": ["FR"],
            ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical"],
            "2030": [0.25],
        }
    ).to_csv(path, index=False)

    rows = read_deterministic_asocc_rows(path=path, requested_years=[2030])

    assert rows["asocc_ssp_scenario"].tolist() == ["SSP2"]


def test_lcia_support_rows_cache_reuses_run_scoped_tables(tmp_path: Path) -> None:
    path_scope = AsoccDeterministicPathScope(
        proj_base=tmp_path,
        source_label="exiobase_396_ixi",
        agg_version=None,
    )
    output_path = asocc_l2_dir(scope=path_scope, bucket="l2_in_l1", lcia_sub=None) / (
        "l2_UT(FDa).csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "l2_method": ["UT(FDa)"],
            "r_c": ["FR"],
            "s_p": ["D"],
            "2005": [0.5],
        }
    ).to_csv(output_path, index=False)
    loaded = LoadedAsoccFinalRows(
        base_asocc_args={"fu_code": "L2.c.b"},
        asocc_scope=cast(AsoccScope, None),
        path_scope=path_scope,
        persisted_scopes=(
            AsoccPersistedRunScope(
                scope_key="scope",
                compute_signature=AsoccPersistedComputeSignature(payload={}),
                completed_years=[2005],
                outputs=[str(output_path)],
                ssp_scenarios=[],
            ),
        ),
        deterministic_manifest_path=tmp_path / "scope_manifest.json",
        requested_years=[2005],
        final_bucket="l2_vs_global",
        rows=pd.DataFrame(),
    )
    cache = LCIASupportRowCache()

    first = support_rows(
        loaded=loaded,
        bucket="l2_in_l1",
        stem="l2_UT(FDa)",
        requested_years=[2005],
        support_cache=cache,
    )
    output_path.unlink()
    second = support_rows(
        loaded=loaded,
        bucket="l2_in_l1",
        stem="l2_UT(FDa)",
        requested_years=[2005],
        support_cache=cache,
    )

    assert second.equals(first)


def test_combined_lcia_support_coefficients_use_sparse_code_lookup() -> None:
    years = list(range(2000, 2030))
    axis_values = [f"S{index}" for index in range(len(years))]
    final_rows = pd.DataFrame(
        {
            "year": years,
            "s_p": axis_values,
            ASOCC_SSP_SCENARIO_COLUMN: [None] * len(years),
            ASOCC_VALUE_COLUMN: np.arange(len(years), dtype=np.float64),
        }
    )
    l2_rows = pd.DataFrame(
        {
            "year": years,
            "s_p": axis_values,
            ASOCC_SSP_SCENARIO_COLUMN: [None] * len(years),
            ASOCC_VALUE_COLUMN: np.linspace(1.0, 2.0, len(years)),
        }
    )
    l1_rows = pd.DataFrame(
        {
            "year": years,
            "r_c": axis_values,
            ASOCC_SSP_SCENARIO_COLUMN: [None] * len(years),
            ASOCC_VALUE_COLUMN: np.linspace(3.0, 4.0, len(years)),
        }
    )

    l1_coefficients, l2_coefficients = combined_route_coefficients(
        final_rows=final_rows,
        l1_rows=l1_rows,
        l2_rows=l2_rows,
        weight_axis="s_p",
        l1_axis="r_c",
        l1_sampled=True,
    )

    assert l2_coefficients is None
    assert l1_coefficients is not None
    assert l1_coefficients.nnz == len(years)
    np.testing.assert_allclose(l1_coefficients.diagonal(), l2_rows[ASOCC_VALUE_COLUMN])
    unmatched_l1_rows = l1_rows.copy()
    unmatched_l1_rows["r_c"] = [f"R{index}" for index in range(len(years))]
    unmatched_coefficients, unmatched_l2_coefficients = combined_route_coefficients(
        final_rows=final_rows,
        l1_rows=unmatched_l1_rows,
        l2_rows=l2_rows,
        weight_axis="s_p",
        l1_axis="r_c",
        l1_sampled=True,
    )
    assert unmatched_l2_coefficients is None
    assert unmatched_coefficients is not None
    assert unmatched_coefficients.nnz == 0


def test_uncertainty_asocc_lcia_requires_sector_cov_mapping(allocation_dummy_repo) -> None:
    project_name = "asocc_uncertainty_lcia_missing_sector_cov"
    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args={
                **_lcia_base_args(project_name=project_name, reference_years=[2005]),
                "method_plan": "one_step",
                "l1_l2_pairs": [],
                "r_f": None,
            },
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 1},
                    "convergence": {"active": False},
                },
                "lcia_uncertainty": {},
            },
            output_format="csv_compact",
            refresh=True,
        )


def test_uncertainty_asocc_lcia_rejects_unavailable_sector_cov_code(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_unavailable_sector_cov"
    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args={
                **_lcia_base_args(project_name=project_name, reference_years=[2005]),
                "method_plan": "one_step",
                "l1_l2_pairs": [],
                "r_f": None,
            },
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 1},
                    "convergence": {"active": False},
                },
                "lcia_uncertainty": {"sector_cov_mapping": {"D": "not_a_sector"}},
            },
            output_format="csv_compact",
            refresh=True,
        )


def test_uncertainty_asocc_lcia_requires_sector_cov_mapping_shape(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_sector_cov_shape"
    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args={
                **_lcia_base_args(project_name=project_name, reference_years=[2005]),
                "method_plan": "one_step",
                "l1_l2_pairs": [],
                "r_f": None,
            },
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 1},
                    "convergence": {"active": False},
                },
                "lcia_uncertainty": {"sector_cov_mapping": ["D", "Electricity"]},
            },
            output_format="csv_compact",
            refresh=True,
        )


def test_uncertainty_asocc_lcia_rejects_unknown_source_parameters(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_lcia_unknown_parameter"
    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args={
                **_lcia_base_args(project_name=project_name, reference_years=[2005]),
                "method_plan": "one_step",
                "l1_l2_pairs": [],
                "r_f": None,
            },
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 1},
                    "convergence": {"active": False},
                },
                "lcia_uncertainty": {"unknown": True},
            },
            output_format="csv_compact",
            refresh=True,
        )


def test_uncertainty_asocc_rejects_inactive_future_reference_year(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_future_reference_year"
    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args={
                **_lcia_base_args(project_name=project_name, reference_years=[2006]),
                "years": [2005],
            },
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 1},
                    "convergence": {"active": False},
                }
            },
            output_format="csv_compact",
            refresh=True,
        )


def test_uncertainty_asocc_reference_year_requires_compatible_requested_year(
    allocation_dummy_repo,
) -> None:
    project_name = "asocc_uncertainty_reference_year_no_compatible"
    with pytest.raises(ValueError):
        uncertainty_asocc(
            base_asocc_args={
                **_lcia_base_args(project_name=project_name, reference_years=[2006]),
                "years": [2005],
            },
            uncertainty_config={
                "mc_parameters": {
                    "fixed": {"active": True, "n_runs": 1},
                    "convergence": {"active": False},
                },
                "reference_year_uncertainty": {},
            },
            output_format="csv_compact",
            refresh=True,
        )
