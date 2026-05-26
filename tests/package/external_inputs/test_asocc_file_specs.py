from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest

from pyaesa.asocc.runtime.scope.branch_resolution import AsoccDeterministicPathScope
from pyaesa.asocc.runtime.scope.branch_resolution import outputs_project_root
from pyaesa.asocc.runtime.request.scope import AsoccScope
from pyaesa.asocc.runtime.paths.external import (
    external_asocc_relative_dir,
    get_asocc_external_method_level_dir,
)
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.asocc.uncertainty.inputs.deterministic_rows import LoadedAsoccFinalRows
from pyaesa.asocc.uncertainty.inputs.external_rows import (
    ExternalAsoccRowsPlan,
    append_external_monte_carlo_matrix,
    external_asocc_has_monte_carlo_rows,
    resolve_external_asocc_rows,
)
from pyaesa.external_inputs.asocc.deterministic import downstream_shares as downstream_mod
from pyaesa.external_inputs.asocc.deterministic import files as deterministic_mod
from pyaesa.external_inputs.asocc.monte_carlo import files as monte_carlo_mod
from pyaesa.external_inputs.asocc.schema import file_specs
from pyaesa.external_inputs.asocc.schema import row_schema as row_schema_mod
from pyaesa.external_inputs.asocc.schema.contracts import ExternalMethodSelection
from pyaesa.external_inputs.shared import matrix_identity as matrix_identity_mod
from pyaesa.shared.runtime.scenario.columns import ASOCC_SSP_SCENARIO_COLUMN
from pyaesa.shared.uncertainty_assessment.monte_carlo.runs import RunBatch
from pyaesa.shared.lcia.contracts import bundled_cc_expected_impacts


def _selection(*, fu_code: str = "L2.a.a") -> ExternalMethodSelection:
    return ExternalMethodSelection(
        fu_code=fu_code,
        l2_method="UT(FD)",
        l1_method="CO(S)",
        level=("level_1" if fu_code.startswith("L1.") else "level_2"),
    )


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix == ".pickle":
        frame.to_pickle(path)
    elif path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise AssertionError(f"unsupported test suffix: {path.suffix}")


def _selectors(count: int = 1) -> dict[str, list[str]]:
    return {
        "r_p": ["FR"] * count,
        "s_p": ["D"] * count,
    }


def _external_lcia_render_frame(
    *,
    lcia_method: str,
    run_indices: list[int],
    value_start: float,
) -> pd.DataFrame:
    _cc_csv_path, expected_impacts = bundled_cc_expected_impacts(lcia_method=lcia_method)
    rows = []
    value = value_start
    for run_index in run_indices:
        for impact in expected_impacts:
            rows.append(
                {
                    "run_index": run_index,
                    "year": 2019,
                    ASOCC_SSP_SCENARIO_COLUMN: None,
                    "r_p": "FR",
                    "s_p": "D",
                    "impact": impact,
                    "value": value,
                }
            )
            value += 1.0
    return pd.DataFrame(rows)


def _loaded_external_asocc_scope(
    *,
    proj_base: Path,
    base_allocate_args: dict,
    years: list[int],
) -> LoadedAsoccFinalRows:
    return LoadedAsoccFinalRows(
        base_asocc_args=base_allocate_args,
        asocc_scope=cast(AsoccScope, SimpleNamespace(target_selector_payload={"methods": []})),
        path_scope=cast(
            AsoccDeterministicPathScope,
            SimpleNamespace(
                proj_base=proj_base,
                source_label=str(base_allocate_args["source"]),
            ),
        ),
        persisted_scopes=(),
        deterministic_manifest_path=proj_base / "unused.json",
        requested_years=years,
        final_bucket="l2_vs_global",
        rows=pd.DataFrame(),
    )


def test_runtime_stems_expected_stems_and_descriptions_cover_deterministic_modes(
    project_repo: Path,
) -> None:
    selection = _selection()
    assert external_asocc_relative_dir(level="level_1") == Path("results")
    assert external_asocc_relative_dir(level="level_2") == Path("results_l2_vs_global")

    assert (
        file_specs.external_asocc_runtime_file_stem(
            fu_code="L1.a",
            file_method_token="CO(S)",
            l1_method="CO(S)",
            lcia_method=None,
            scenario=None,
        )
        == "l1_CO(S)"
    )
    assert (
        file_specs.external_asocc_runtime_file_stem(
            fu_code="L2.a.a",
            file_method_token="UT(FD)",
            l1_method=None,
            lcia_method="pb_lcia",
            scenario="SSP2",
        )
        == "UT(FD)__pb_lcia__ssp2"
    )
    assert (
        file_specs.external_asocc_runtime_file_stem(
            fu_code="L2.a.a",
            file_method_token="UT(FD)",
            l1_method="CO(S)",
            lcia_method=None,
            scenario=None,
        )
        == "l1_CO(S)_l2_UT(FD)"
    )

    deterministic_stems = file_specs.external_asocc_expected_stems(
        fu_code=selection.fu_code,
        file_method_token=selection.file_method_token,
        l1_method=selection.l1_method,
        lcia_methods=["pb_lcia", "pb_lcia", " "],
        years=[2019, 2020],
        ssp_scenario_options_by_year={2019: [None, "SSP2"], 2020: ["SSP3"]},
    )
    assert deterministic_stems == sorted(
        {
            "l1_CO(S)_l2_UT(FD)",
            "l1_CO(S)_l2_UT(FD)__ssp2",
            "l1_CO(S)_l2_UT(FD)__ssp3",
            "l1_CO(S)_l2_UT(FD)__pb_lcia",
            "l1_CO(S)_l2_UT(FD)__pb_lcia__ssp2",
            "l1_CO(S)_l2_UT(FD)__pb_lcia__ssp3",
        }
    )


def test_read_parse_and_candidate_file_discovery_cover_all_supported_suffixes(
    project_repo: Path,
) -> None:
    selection = _selection()
    assert (
        file_specs.candidate_files(
            proj_base=project_repo,
            selection=selection,
            lcia_methods=["pb_lcia"],
        )
        == tuple()
    )

    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="deterministic",
        level=selection.level,
    )
    assert get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    ) == (project_repo / "B1_asocc" / "external_asocc" / "monte_carlo")

    csv_frame = pd.DataFrame({"year": [2019], "value": [1.0]})
    pickle_frame = pd.DataFrame({"2019": [2.0]})
    parquet_frame = pd.DataFrame({"year": [2020], "value": [3.0]})
    _write_table(directory / "l1_CO(S)_l2_UT(FD).csv", csv_frame)
    _write_table(directory / "l1_CO(S)_l2_UT(FD)__pb_lcia.pickle", pickle_frame)
    _write_table(directory / "l1_CO(S)_l2_UT(FD)__pb_lcia__ssp2.parquet", parquet_frame)
    _write_table(directory / "nonmatching.csv", csv_frame)
    (directory / "ignore.txt").write_text("x", encoding="utf-8")

    assert file_specs.read_external_asocc_table(directory / "l1_CO(S)_l2_UT(FD).csv").equals(
        csv_frame
    )
    assert file_specs.read_external_asocc_table(
        directory / "l1_CO(S)_l2_UT(FD)__pb_lcia.pickle"
    ).equals(pickle_frame)
    assert file_specs.read_external_asocc_table(
        directory / "l1_CO(S)_l2_UT(FD)__pb_lcia__ssp2.parquet"
    ).equals(parquet_frame)

    with pytest.raises(ValueError):
        file_specs.read_external_asocc_table(directory / "bad.xlsx")
    assert file_specs.requested_lcia_methods(["pb_lcia", " ", "pb_lcia"]) == ("pb_lcia",)
    assert file_specs.frame_years(pd.DataFrame({"year": [2021, 2019, 2021, None]})) == [2019, 2021]
    assert file_specs.frame_years(pd.DataFrame({"2018": [1.0], "name": ["x"], 2205: [2.0]})) == [
        2018
    ]
    with pytest.raises(
        ValueError,
    ):
        file_specs.frame_years(pd.DataFrame({"2018": [1.0], "2019": [pd.NA]}))

    assert file_specs._parse_suffix(suffix="", requested_methods=("pb_lcia",)) == (None, None)
    assert file_specs._parse_suffix(suffix="__pb_lcia", requested_methods=("pb_lcia",)) == (
        "pb_lcia",
        None,
    )
    assert file_specs._parse_suffix(
        suffix="__pb_lcia__ssp2",
        requested_methods=("pb_lcia",),
    ) == ("pb_lcia", "SSP2")
    assert file_specs._parse_suffix(
        suffix="__ssp2",
        requested_methods=("pb_lcia",),
    ) == (None, "SSP2")
    assert file_specs._parse_suffix(suffix="__", requested_methods=("pb_lcia",)) == (None, None)
    with pytest.raises(ValueError):
        file_specs._parse_suffix(
            suffix="__scenario_only",
            requested_methods=("pb_lcia",),
        )
    with pytest.raises(ValueError):
        file_specs._parse_suffix(
            suffix="__pb_lcia__SSP2",
            requested_methods=("pb_lcia",),
        )

    assert (
        file_specs._parse_candidate_file(
            path=directory / "l1_CO(S)_l2_UT(FD)__pb_lcia__ssp2.parquet",
            selection=selection,
            requested_methods=("pb_lcia",),
        )
        is not None
    )
    assert (
        file_specs._parse_candidate_file(
            path=directory / "different.parquet",
            selection=selection,
            requested_methods=("pb_lcia",),
        )
        is None
    )

    candidates = file_specs.candidate_files(
        proj_base=project_repo,
        selection=selection,
        lcia_methods=["pb_lcia"],
    )
    assert len(candidates) == 3
    assert [spec.scenario for spec in candidates] == [None, None, "SSP2"]


def test_external_monte_carlo_loader_covers_render_file_contracts(project_repo: Path) -> None:
    selection = _selection()
    assert (
        monte_carlo_mod.resolve_external_monte_carlo_source(
            proj_base=project_repo,
            selection=selection,
            years=[2019],
            lcia_methods=None,
            ssp_scenario_options_by_year=None,
        )
        is None
    )

    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    csv_path = directory / "l1_CO(S)_l2_UT(FD).csv"
    _write_table(
        csv_path,
        pd.DataFrame(
            {
                "run_index": [0, 1],
                "year": [2019, 2019],
                **_selectors(2),
                ASOCC_SSP_SCENARIO_COLUMN: [None, None],
                "value": [0.42, 0.43],
            }
        ),
    )
    assert [
        spec.path.name
        for spec in file_specs.candidate_files(
            proj_base=project_repo,
            selection=selection,
            lcia_methods=None,
            storage_mode="monte_carlo",
        )
    ] == ["l1_CO(S)_l2_UT(FD).csv"]
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)
    assert source.available_runs == 0
    assert materialized.available_runs == 2
    assert (
        monte_carlo_mod.resolve_external_monte_carlo_source(
            proj_base=project_repo,
            selection=selection,
            years=[],
            lcia_methods=None,
            ssp_scenario_options_by_year=None,
        )
        is None
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD)__pb_lcia.csv",
        pd.DataFrame(
            {
                "run_index": [0, 1],
                "year": [2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None],
                **_selectors(2),
                "impact": ["GWP_100", "GWP_100"],
                "value": [0.52, 0.53],
            }
        ),
    )
    assert [
        spec.path.name
        for spec in file_specs.candidate_files(
            proj_base=project_repo,
            selection=selection,
            lcia_methods=["pb_lcia"],
            storage_mode="monte_carlo",
        )
    ] == ["l1_CO(S)_l2_UT(FD).csv", "l1_CO(S)_l2_UT(FD)__pb_lcia.csv"]
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD)__ssp2.csv",
        pd.DataFrame(
            {
                "run_index": [0, 1],
                "year": [2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP2", "SSP2"],
                **_selectors(2),
                "value": [0.62, 0.63],
            }
        ),
    )
    assert [
        spec.path.name
        for spec in file_specs.candidate_files(
            proj_base=project_repo,
            selection=selection,
            lcia_methods=["pb_lcia"],
            storage_mode="monte_carlo",
        )
    ] == ["l1_CO(S)_l2_UT(FD).csv", "l1_CO(S)_l2_UT(FD)__pb_lcia.csv"]


def test_external_monte_carlo_loader_covers_parquet_pickle_and_inventory_errors(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    parquet_path = directory / "parquet_runs.parquet"
    pickle_path = directory / "pickle_runs.pickle"
    frame = pd.DataFrame({"run_index": [0, 1], "year": [2019, 2019], "value": [1.0, 2.0]})
    _write_table(parquet_path, frame)
    _write_table(pickle_path, frame)

    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD)__pb_lcia.csv",
        _external_lcia_render_frame(lcia_method="pb_lcia", run_indices=[0], value_start=3.0),
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD)__gwp100_lcia.csv",
        _external_lcia_render_frame(
            lcia_method="gwp100_lcia",
            run_indices=[0, 1],
            value_start=30.0,
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=["pb_lcia", "gwp100_lcia"],
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    _write_table(
        directory / "l1_CO(S)_l2_UT(FD)__pb_lcia.csv",
        _external_lcia_render_frame(lcia_method="pb_lcia", run_indices=[0, 2], value_start=3.0),
    )
    (directory / "l1_CO(S)_l2_UT(FD)__gwp100_lcia.csv").unlink()
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_loader_covers_scenario_and_malformed_runs(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0, 0, 1, 1],
                "year": [2030, 2030, 2030, 2030],
                ASOCC_SSP_SCENARIO_COLUMN: ["ssp1", "ssp2", "ssp1", "ssp2"],
                **_selectors(4),
                "value": [1.0, 2.0, 1.1, 2.1],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2030],
        lcia_methods=None,
        ssp_scenario_options_by_year={2030: ["SSP1", "SSP2"]},
    )
    assert source is not None
    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)
    assert materialized.run_matrix.template[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP1", "SSP2"]
    unrestricted_source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2030],
        lcia_methods=None,
        ssp_scenario_options_by_year={2029: ["SSP5"]},
    )
    assert unrestricted_source is not None
    unrestricted = monte_carlo_mod.materialize_external_monte_carlo_source(
        source=unrestricted_source
    )
    assert unrestricted.run_matrix.template[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP1", "SSP2"]
    template, values = append_external_monte_carlo_matrix(
        template=pd.DataFrame(
            {
                "year": [2030],
                "l1_l2_method": ["UT(FD)"],
                "allocated_share": [0.5],
            }
        ),
        values=pd.DataFrame({"value": [0.5]}).to_numpy(dtype="float64").T,
        plan=ExternalAsoccRowsPlan(monte_carlo_sources=(materialized,)),
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=1, rng_seed=7),
    )
    external_public = template.loc[template["l1_l2_method"].eq("CO(S)_UT(FD)")]
    assert sorted(external_public[ASOCC_SSP_SCENARIO_COLUMN].tolist()) == ["SSP1", "SSP2"]
    assert values.shape == (1, 3)

    mismatch_source = monte_carlo_mod.ExternalMonteCarloRowsSource(
        selection=selection,
        file_selections=(
            monte_carlo_mod.ExternalMonteCarloFileSelection(
                path=directory / "l1_CO(S)_l2_UT(FD).csv",
                lcia_method=None,
                requested_years=(2025,),
                ssp_scenario_options_by_year={2025: (None,)},
            ),
        ),
        run_indices=(0, 1),
    )
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=mismatch_source)
    scenario_mismatch = monte_carlo_mod.ExternalMonteCarloRowsSource(
        selection=selection,
        file_selections=(
            monte_carlo_mod.ExternalMonteCarloFileSelection(
                path=directory / "l1_CO(S)_l2_UT(FD).csv",
                lcia_method=None,
                requested_years=(2030,),
                ssp_scenario_options_by_year={2030: ("SSP5",)},
            ),
        ),
        run_indices=(0, 1),
    )
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=scenario_mismatch)
    missing_year_path = directory / "missing_year.pickle"
    _write_table(
        missing_year_path,
        pd.DataFrame(
            {
                "run_index": [0],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    missing_year_source = monte_carlo_mod.ExternalMonteCarloRowsSource(
        selection=selection,
        file_selections=(
            monte_carlo_mod.ExternalMonteCarloFileSelection(
                path=missing_year_path,
                lcia_method=None,
                requested_years=(2030,),
                ssp_scenario_options_by_year=None,
            ),
        ),
        run_indices=(0,),
    )
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=missing_year_source)
    missing_year_csv_path = directory / "missing_year.csv"
    _write_table(
        missing_year_csv_path,
        pd.DataFrame(
            {
                "run_index": [0],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    missing_year_csv_source = monte_carlo_mod.ExternalMonteCarloRowsSource(
        selection=selection,
        file_selections=(
            monte_carlo_mod.ExternalMonteCarloFileSelection(
                path=missing_year_csv_path,
                lcia_method=None,
                requested_years=(2030,),
                ssp_scenario_options_by_year=None,
            ),
        ),
        run_indices=(0,),
    )
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=missing_year_csv_source)


def test_external_monte_carlo_matrix_uses_empirical_sobol_run_units(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0, 1, 2],
                "year": [2019, 2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None, None],
                **_selectors(3),
                "value": [0.42, 0.52, 0.62],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    template, values = append_external_monte_carlo_matrix(
        template=pd.DataFrame(columns=["year", "l1_l2_method", "allocated_share"]),
        values=np.empty((3, 0), dtype=np.float64),
        plan=ExternalAsoccRowsPlan(monte_carlo_sources=(materialized,)),
        batch=RunBatch(batch_index=0, start_run_index=0, stop_run_index=3, rng_seed=7),
        unit_values=np.array([0.0, 0.5, np.nextafter(1.0, 0.0)]),
    )

    assert template["l1_l2_method"].tolist() == ["CO(S)_UT(FD)"]
    np.testing.assert_allclose(values[:, 0], [0.42, 0.52, 0.62])


@pytest.mark.parametrize("suffix", [".parquet", ".pickle"])
def test_external_monte_carlo_matrix_materializes_supported_table_formats(
    project_repo: Path,
    suffix: str,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / f"l1_CO(S)_l2_UT(FD){suffix}",
        pd.DataFrame(
            {
                "run_index": [0, 1],
                "year": [2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None],
                **_selectors(2),
                "value": [0.42, 0.52],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None

    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    assert materialized.run_matrix.template["l1_l2_method"].tolist() == ["CO(S)_UT(FD)"]
    np.testing.assert_allclose(materialized.run_matrix.values[:, 0], [0.42, 0.52])


def test_external_monte_carlo_compact_matrix_reports_empty_requested_year(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    compact_dir = directory / "l1_CO(S)_l2_UT(FD)"
    compact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "public_row_id": [0],
            "year": [2020],
            ASOCC_SSP_SCENARIO_COLUMN: [None],
            **_selectors(),
        }
    ).to_csv(compact_dir / "public_row_identity.csv", index=False)
    pd.DataFrame({"run_index": [0, 1], "0": [0.42, 0.52]}).to_csv(
        compact_dir / "asocc_runs.csv",
        index=False,
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_compact_matrix_materializes_selected_rows(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    compact_dir = directory / "l1_CO(S)_l2_UT(FD)"
    compact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "year": [2019, 2020],
            ASOCC_SSP_SCENARIO_COLUMN: [None, None],
            **_selectors(2),
        }
    ).to_csv(compact_dir / "public_row_identity.csv", index=False)
    pd.DataFrame(
        {
            "run_index": [0, 1],
            "0": [0.42, 0.52],
            "1": [0.43, 0.53],
        }
    ).to_csv(compact_dir / "asocc_runs.csv", index=False)
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2020],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None

    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    assert materialized.run_matrix.template["year"].tolist() == [2020]
    np.testing.assert_allclose(materialized.run_matrix.values, [[0.43], [0.53]])


def test_external_monte_carlo_matrix_materializes_frame_scenario_identities(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).pickle",
        pd.DataFrame(
            {
                "run_index": [0, 0, 0, 1, 1, 1],
                "year": [2030, 2030, 2031, 2030, 2030, 2031],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", "SSP2", None, "SSP1", "SSP2", None],
                **_selectors(6),
                "value": [0.42, 0.43, 0.44, 0.52, 0.53, 0.54],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2030, 2031],
        lcia_methods=None,
        ssp_scenario_options_by_year={2030: ["SSP1", "SSP2"]},
    )
    assert source is not None

    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    assert materialized.run_matrix.template[ASOCC_SSP_SCENARIO_COLUMN].tolist() == [
        "SSP1",
        "SSP2",
        None,
    ]
    np.testing.assert_allclose(
        materialized.run_matrix.values,
        [[0.42, 0.43, 0.44], [0.52, 0.53, 0.54]],
    )


def test_external_monte_carlo_source_year_filter_and_manifest(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0, 0, 1, 1],
                "year": [2019, 2020, 2019, 2020],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None, None, None],
                **_selectors(4),
                "value": [0.42, 0.43, 0.52, 0.53],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019, 2020],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    assert materialized.selection == selection
    assert materialized.file_selections == source.file_selections
    assert materialized.run_indices == (0, 1)
    assert materialized.available_runs == 2
    filtered = monte_carlo_mod.external_monte_carlo_source_for_years(
        source=materialized,
        years=(2020,),
    )
    assert filtered is not None
    assert filtered.run_matrix.template["year"].tolist() == [2020]
    np.testing.assert_allclose(filtered.run_matrix.values, [[0.43], [0.53]])
    assert (
        monte_carlo_mod.external_monte_carlo_source_for_years(
            source=materialized,
            years=(2030,),
        )
        is None
    )
    payload = monte_carlo_mod.external_monte_carlo_manifest_payload(source=materialized)
    assert payload["storage_mode"] == "monte_carlo"
    assert payload["selection"] == "CO(S)_UT(FD)"
    assert payload["run_indices"] == [0, 1]
    assert payload["files"][0]["requested_years"] == [2019, 2020]


def test_external_monte_carlo_matrix_rejects_empty_frame_inventory(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).pickle",
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2020],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None

    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_matrix_requires_same_identity_per_run(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0, 0, 1],
                "year": [2019, 2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", "SSP2", "SSP1"],
                **_selectors(3),
                "value": [0.42, 0.43, 0.52],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None

    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_matrix_rejects_duplicate_and_extra_identities(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    path = directory / "l1_CO(S)_l2_UT(FD).csv"
    _write_table(
        path,
        pd.DataFrame(
            {
                "run_index": [0, 0, 1],
                "year": [2019, 2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None, None],
                **_selectors(3),
                "value": [0.42, 0.43, 0.52],
            }
        ),
    )
    duplicate_source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert duplicate_source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=duplicate_source)

    _write_table(
        path,
        pd.DataFrame(
            {
                "run_index": [0, 0, 1, 1],
                "year": [2019, 2019, 2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", "SSP2", "SSP1", "SSP3"],
                **_selectors(4),
                "value": [0.42, 0.43, 0.52, 0.53],
            }
        ),
    )
    extra_source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert extra_source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=extra_source)

    _write_table(
        path,
        pd.DataFrame(
            {
                "run_index": [0, 1],
                "year": [2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None],
                "r_p": ["FR", "DE"],
                "s_p": ["D", "D"],
                "value": [0.42, 0.52],
            }
        ),
    )
    constant_mismatch_source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert constant_mismatch_source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=constant_mismatch_source)


def test_external_monte_carlo_matrix_rejects_frame_identity_mismatches(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    path = directory / "l1_CO(S)_l2_UT(FD).pickle"
    _write_table(
        path,
        pd.DataFrame(
            {
                "run_index": [0, 1],
                "year": [2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None],
                "r_p": ["FR", "DE"],
                "s_p": ["D", "D"],
                "value": [0.42, 0.52],
            }
        ),
    )
    constant_source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert constant_source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=constant_source)

    _write_table(
        path,
        pd.DataFrame(
            {
                "run_index": [0, 0, 1, 1],
                "year": [2019, 2019, 2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP1", "SSP2", "SSP1", "SSP3"],
                **_selectors(4),
                "value": [0.42, 0.43, 0.52, 0.53],
            }
        ),
    )
    missing_value_source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert missing_value_source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=missing_value_source)

    _write_table(
        path,
        pd.DataFrame(
            {
                "run_index": [0, 0, 1],
                "year": [2019, 2020, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None, None],
                **_selectors(3),
                "value": [0.42, 0.43, 0.52],
            }
        ),
    )
    absent_pair_source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019, 2020],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert absent_pair_source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=absent_pair_source)

    _template, lookup = matrix_identity_mod.template_lookup(
        template=pd.DataFrame({"run_index": [0, 0], "year": [2019, 2020], "value": [1.0, 2.0]})
    )
    with pytest.raises(ValueError):
        matrix_identity_mod.positions_from_codes(
            codes=np.array([2], dtype=np.int64),
            lookup=lookup,
        )


def test_external_monte_carlo_matrix_rejects_frame_duplicate_identity(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).pickle",
        pd.DataFrame(
            {
                "run_index": [0, 0, 1],
                "year": [2019, 2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None, None],
                **_selectors(3),
                "value": [0.42, 0.43, 0.52],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None

    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_matrix_rejects_duplicate_identity_across_chunks(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    filler_count = monte_carlo_mod.EXTERNAL_MONTE_CARLO_MATRIX_CHUNK_ROWS - 2
    frame = pd.DataFrame(
        {
            "run_index": [0, 1, *([0] * filler_count), 1],
            "year": [2019, 2019, *([2020] * filler_count), 2019],
            ASOCC_SSP_SCENARIO_COLUMN: [None, None, *([None] * filler_count), None],
            "r_p": ["FR"] * (filler_count + 3),
            "s_p": ["D"] * (filler_count + 3),
            "value": [0.42, 0.52, *([0.0] * filler_count), 0.62],
        }
    )
    _write_table(directory / "l1_CO(S)_l2_UT(FD).parquet", frame)
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None

    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_matrix_validates_selected_impact_contracts(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    base_path = directory / "l1_CO(S)_l2_UT(FD).csv"
    pb_path = directory / "l1_CO(S)_l2_UT(FD)__pb_lcia.csv"

    _write_table(
        base_path,
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "impact": ["AAL"],
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    base_path.unlink()
    _write_table(
        pb_path,
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    _write_table(
        pb_path,
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "impact": ["wrong"],
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    lcia_reference = _external_lcia_render_frame(
        lcia_method="pb_lcia",
        run_indices=[0, 1],
        value_start=2.0,
    )
    lcia_reference["reference_year"] = 2018
    _write_table(pb_path, lcia_reference)
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)
    assert set(materialized.run_matrix.template["reference_year"].tolist()) == {2018}


def test_external_monte_carlo_matrix_validates_frame_impact_contracts(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    base_path = directory / "l1_CO(S)_l2_UT(FD).pickle"
    pb_path = directory / "l1_CO(S)_l2_UT(FD)__pb_lcia.pickle"

    _write_table(
        base_path,
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "impact": ["AAL"],
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    base_path.unlink()
    _write_table(
        pb_path,
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    _write_table(
        pb_path,
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "impact": [None],
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    _write_table(
        pb_path,
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "impact": [None],
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    _write_table(
        pb_path,
        _external_lcia_render_frame(lcia_method="pb_lcia", run_indices=[0, 1], value_start=3.0),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)
    assert materialized.run_matrix.values.shape[0] == 2


@pytest.mark.parametrize(
    ("extra_columns", "error"),
    [
        ({"level": ["level_2"]}, "reserved columns"),
        ({}, "Expected=\\['r_p', 's_p'\\], missing=\\['s_p'\\]"),
        ({"s_p": [" "]}, "must be non empty"),
        ({"r_c": ["FR"]}, "outside the requested functional unit identity"),
    ],
)
def test_external_monte_carlo_csv_validates_schema_before_materialization(
    project_repo: Path,
    extra_columns: dict[str, list[str]],
    error: str,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    selectors = {"r_p": ["FR"], "s_p": ["D"]}
    selectors.update(extra_columns)
    if extra_columns == {}:
        selectors.pop("s_p")
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **selectors,
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_csv_rejects_missing_required_run_columns(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    csv_path = directory / "l1_CO(S)_l2_UT(FD).csv"
    _write_table(
        csv_path,
        pd.DataFrame(
            {
                "run_index": [0],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    csv_path.unlink()
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).pickle",
        pd.DataFrame(
            {
                "run_index": [0],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    frame_source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert frame_source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=frame_source)


def test_external_monte_carlo_matrix_rejects_empty_requested_year_inventory(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2020],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_matrix_rejects_inventory_without_run_zero(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [1],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_csv_requires_sorted_run_index(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0, 2, 1],
                "year": [2019, 2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None, None],
                **_selectors(3),
                "value": [1.0, 1.2, 1.1],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None

    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_csv_grows_streaming_matrix(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    run_indices = list(range(1025))
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": run_indices,
                "year": [2019] * len(run_indices),
                ASOCC_SSP_SCENARIO_COLUMN: [None] * len(run_indices),
                **_selectors(len(run_indices)),
                "value": [float(index) for index in run_indices],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert source is not None

    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    assert materialized.run_matrix.values.shape == (1025, 1)
    assert materialized.run_matrix.values[-1, 0] == 1024.0


def test_external_monte_carlo_csv_filters_requested_scenario_scope(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    filtered_path = directory / "l1_CO(S)_l2_UT(FD).csv"
    _write_table(
        filtered_path,
        pd.DataFrame(
            {
                "run_index": [0],
                "year": [2030],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"],
                **_selectors(),
                "value": [1.0],
            }
        ),
    )
    source = monte_carlo_mod.ExternalMonteCarloRowsSource(
        selection=selection,
        file_selections=(
            monte_carlo_mod.ExternalMonteCarloFileSelection(
                path=filtered_path,
                lcia_method=None,
                requested_years=(2030,),
                ssp_scenario_options_by_year={2030: ("SSP1",)},
            ),
        ),
        run_indices=(0,),
    )
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_external_monte_carlo_csv_materializes_null_scenario_scope(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0, 0, 1, 1],
                "year": [2019, 2019, 2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, "SSP1", None, "SSP1"],
                **_selectors(4),
                "value": [0.42, 0.43, 0.52, 0.53],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year={2019: [None]},
    )
    assert source is not None

    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    assert pd.isna(materialized.run_matrix.template.loc[0, ASOCC_SSP_SCENARIO_COLUMN])
    np.testing.assert_allclose(materialized.run_matrix.values[:, 0], [0.42, 0.52])


def test_external_monte_carlo_csv_materializes_all_null_scenario_scope(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0, 1],
                "year": [2019, 2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None],
                **_selectors(2),
                "value": [0.42, 0.52],
            }
        ),
    )
    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year={2019: [None]},
    )
    assert source is not None

    materialized = monte_carlo_mod.materialize_external_monte_carlo_source(source=source)

    assert pd.isna(materialized.run_matrix.template.loc[0, ASOCC_SSP_SCENARIO_COLUMN])
    np.testing.assert_allclose(materialized.run_matrix.values[:, 0], [0.42, 0.52])

    source = monte_carlo_mod.resolve_external_monte_carlo_source(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year={2019: ["SSP1"]},
    )
    assert source is not None
    with pytest.raises(ValueError):
        monte_carlo_mod.materialize_external_monte_carlo_source(source=source)


def test_validate_lcia_inventory_and_impact_contract_fail_fast(project_repo: Path) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="deterministic",
        level=selection.level,
    )

    mixed_specs = (
        file_specs.ExternalAsoCCFileSpec(
            path=directory / "a.csv",
            scenario=None,
            years=(2019,),
            lcia_method=None,
        ),
        file_specs.ExternalAsoCCFileSpec(
            path=directory / "b.csv",
            scenario=None,
            years=(2019,),
            lcia_method="pb_lcia",
        ),
    )
    with pytest.raises(ValueError):
        file_specs.validate_lcia_inventory(
            specs=mixed_specs,
            selection=selection,
            lcia_methods=["pb_lcia"],
        )

    with pytest.raises(ValueError):
        file_specs.validate_lcia_inventory(
            specs=(
                file_specs.ExternalAsoCCFileSpec(
                    path=directory / "c.csv",
                    scenario=None,
                    years=(2019,),
                    lcia_method="gwp100_lcia",
                ),
            ),
            selection=selection,
            lcia_methods=["pb_lcia"],
        )
    file_specs.validate_lcia_inventory(
        specs=(
            file_specs.ExternalAsoCCFileSpec(
                path=directory / "ok.csv",
                scenario=None,
                years=(2019,),
                lcia_method="pb_lcia",
            ),
        ),
        selection=selection,
        lcia_methods=["pb_lcia"],
    )
    file_specs.validate_lcia_inventory(
        specs=tuple(),
        selection=selection,
        lcia_methods=["pb_lcia"],
    )

    file_specs.validate_impact_contract(
        frame=pd.DataFrame({"year": [2019], "value": [1.0]}),
        path=directory / "non_lcia.csv",
        lcia_method=None,
    )
    with pytest.raises(ValueError):
        file_specs.validate_impact_contract(
            frame=pd.DataFrame({"impact": ["AAL"]}),
            path=directory / "non_lcia_bad.csv",
            lcia_method=None,
        )
    with pytest.raises(ValueError):
        file_specs.validate_impact_contract(
            frame=pd.DataFrame({"impact": [None]}),
            path=directory / "non_lcia_blank_impact.csv",
            lcia_method=None,
        )
    with pytest.raises(ValueError):
        file_specs.validate_impact_contract(
            frame=pd.DataFrame({"reference_year": [2018]}),
            path=directory / "non_lcia_reference.csv",
            lcia_method=None,
        )
    with pytest.raises(ValueError):
        file_specs.validate_impact_contract(
            frame=pd.DataFrame({"impact": ["AAL"], "reference_year": [2018]}),
            path=directory / "non_lcia_impact_reference.csv",
            lcia_method=None,
        )
    with pytest.raises(ValueError):
        file_specs.validate_impact_contract(
            frame=pd.DataFrame({"year": [2019], "value": [1.0]}),
            path=directory / "lcia_missing.csv",
            lcia_method="pb_lcia",
        )


def test_external_row_schema_and_deterministic_loader_cover_empty_and_edge_paths(
    project_repo: Path,
) -> None:
    selection = _selection()
    assert row_schema_mod.expected_external_selector_columns(fu_code="L1.a") == ("r_f",)
    assert row_schema_mod.expected_external_selector_columns(fu_code="L1.b") == ("r_p",)
    assert row_schema_mod.expected_external_selector_columns(fu_code="L2.a.a") == ("r_p", "s_p")
    assert row_schema_mod.expected_external_selector_columns(fu_code="L2.b.a") == (
        "r_p",
        "s_p",
        "r_f",
    )
    assert row_schema_mod.expected_external_selector_columns(fu_code="L2.b.b") == (
        "r_p",
        "s_p",
        "r_c",
    )
    assert row_schema_mod.expected_external_selector_columns(fu_code="L2.c.a") == ("s_p", "r_f")
    assert row_schema_mod.expected_external_selector_columns(fu_code="L2.c.b") == ("s_p", "r_c")
    with pytest.raises(ValueError):
        row_schema_mod.expected_external_selector_columns(fu_code="L3")
    assert list(row_schema_mod.empty_external_asocc_rows(selection=selection).columns) == (
        row_schema_mod.external_asocc_deterministic_row_columns(
            selection=selection,
            include_asocc_ssp_scenario=True,
        )
    )
    assert list(row_schema_mod.empty_external_asocc_render_rows(selection=selection).columns) == (
        row_schema_mod.external_asocc_render_row_columns(
            selection=selection,
            include_asocc_ssp_scenario=True,
        )
    )

    rows_with_scenario = row_schema_mod.normalize_external_asocc_wide_rows(
        frame=pd.DataFrame({**_selectors(), "2019": [1.0]}),
        years=[2019],
        selection=selection,
        lcia_method=None,
        ssp_scenario="SSP2",
        include_asocc_ssp_scenario_column=True,
    )
    assert rows_with_scenario.loc[0, ASOCC_SSP_SCENARIO_COLUMN] == "SSP2"
    empty_rows = row_schema_mod.normalize_external_asocc_wide_rows(
        frame=pd.DataFrame({**_selectors(), "note": ["kept"]}),
        years=[2019],
        selection=selection,
        lcia_method=None,
        ssp_scenario=None,
        include_asocc_ssp_scenario_column=False,
    )
    assert empty_rows.empty
    assert ASOCC_SSP_SCENARIO_COLUMN not in empty_rows.columns
    rows_with_empty_extra_selector = row_schema_mod.normalize_external_asocc_wide_rows(
        frame=pd.DataFrame({**_selectors(), "r_c": [None], "2019": [1.0]}),
        years=[2019],
        selection=selection,
        lcia_method=None,
        ssp_scenario=None,
        include_asocc_ssp_scenario_column=False,
    )
    assert "r_c" not in rows_with_empty_extra_selector.columns
    with pytest.raises(ValueError):
        row_schema_mod.normalize_external_asocc_wide_rows(
            frame=pd.DataFrame({"r_p": ["FR"], "2019": [1.0]}),
            years=[2019],
            selection=selection,
            lcia_method=None,
            ssp_scenario=None,
            include_asocc_ssp_scenario_column=False,
        )
    with pytest.raises(ValueError):
        row_schema_mod.normalize_external_asocc_wide_rows(
            frame=pd.DataFrame({"r_p": ["FR"], "s_p": [" "], "2019": [1.0]}),
            years=[2019],
            selection=selection,
            lcia_method=None,
            ssp_scenario=None,
            include_asocc_ssp_scenario_column=False,
        )
    with pytest.raises(ValueError):
        row_schema_mod.normalize_external_asocc_wide_rows(
            frame=pd.DataFrame({**_selectors(), "r_c": ["FR"], "2019": [1.0]}),
            years=[2019],
            selection=selection,
            lcia_method=None,
            ssp_scenario=None,
            include_asocc_ssp_scenario_column=False,
        )
    with pytest.raises(
        ValueError,
    ):
        row_schema_mod.normalize_external_asocc_wide_rows(
            frame=pd.DataFrame({"year": [2019], "value": [1.0]}),
            years=[2019],
            selection=selection,
            lcia_method=None,
            ssp_scenario=None,
            include_asocc_ssp_scenario_column=False,
        )
    with pytest.raises(ValueError):
        row_schema_mod.normalize_external_asocc_render_rows(
            frame=pd.DataFrame({"run_index": [0], "year": [2019]}),
            requested_years=[2019],
            selection=selection,
            lcia_method=None,
            ssp_scenario=None,
            include_asocc_ssp_scenario_column=False,
        )
    empty_filtered_render_rows = row_schema_mod.normalize_external_asocc_render_rows(
        frame=pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
        requested_years=[2020],
        selection=selection,
        lcia_method=None,
        ssp_scenario=None,
        include_asocc_ssp_scenario_column=True,
    )
    assert empty_filtered_render_rows.empty
    with pytest.raises(ValueError):
        row_schema_mod.normalize_external_asocc_render_rows(
            frame=pd.DataFrame({"run_index": [0], "year": [2019], "value": [1.0]}),
            requested_years=[2019],
            selection=selection,
            lcia_method=None,
            ssp_scenario=None,
            include_asocc_ssp_scenario_column=True,
        )
    render_rows_with_file_scenario = row_schema_mod.normalize_external_asocc_render_rows(
        frame=pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: ["ssp1"],
                **_selectors(),
                "value": [1.0],
            }
        ),
        requested_years=[2019],
        selection=selection,
        lcia_method=None,
        ssp_scenario="SSP2",
        include_asocc_ssp_scenario_column=True,
    )
    assert render_rows_with_file_scenario.loc[0, ASOCC_SSP_SCENARIO_COLUMN] == "SSP2"
    render_rows_without_scenario_column = row_schema_mod.normalize_external_asocc_render_rows(
        frame=pd.DataFrame(
            {
                "run_index": [0],
                "year": [2019],
                ASOCC_SSP_SCENARIO_COLUMN: [None],
                **_selectors(),
                "value": [1.0],
            }
        ),
        requested_years=[2019],
        selection=selection,
        lcia_method=None,
        ssp_scenario=None,
        include_asocc_ssp_scenario_column=False,
    )
    assert ASOCC_SSP_SCENARIO_COLUMN not in render_rows_without_scenario_column.columns

    assert (
        deterministic_mod.load_external_deterministic_rows(
            proj_base=project_repo,
            selection=selection,
            years=[2019],
            lcia_methods=None,
            ssp_scenario_options_by_year=None,
        )
        is None
    )

    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="deterministic",
        level=selection.level,
    )
    table_path = directory / "l1_CO(S)_l2_UT(FD).csv"

    _write_table(table_path, pd.DataFrame({**_selectors(), "2025": [1.0]}))
    with pytest.raises(ValueError):
        deterministic_mod.load_external_deterministic_rows(
            proj_base=project_repo,
            selection=selection,
            years=[2019],
            lcia_methods=None,
            ssp_scenario_options_by_year=None,
        )

    _write_table(table_path, pd.DataFrame({**_selectors(), "2019": [1.0]}))
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD)__ssp2.csv",
        pd.DataFrame({**_selectors(), "2030": [2.0]}),
    )
    loaded = deterministic_mod.load_external_deterministic_rows(
        proj_base=project_repo,
        selection=selection,
        years=[2019],
        lcia_methods=None,
        ssp_scenario_options_by_year=None,
    )
    assert loaded is not None
    assert loaded.loc[0, "l1_l2_method"] == "CO(S)_UT(FD)"
    assert loaded.loc[0, "year"] == 2019
    assert loaded.loc[0, "value"] == 1.0
    assert ASOCC_SSP_SCENARIO_COLUMN not in loaded.columns

    assert (
        deterministic_mod.load_external_deterministic_rows(
            proj_base=project_repo,
            selection=selection,
            years=[],
            lcia_methods=None,
            ssp_scenario_options_by_year=None,
        )
        is None
    )

    _write_table(
        table_path,
        pd.DataFrame({**_selectors(), "2019": [1.0], ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"]}),
    )
    with pytest.raises(ValueError):
        deterministic_mod.load_external_deterministic_rows(
            proj_base=project_repo,
            selection=selection,
            years=[2019],
            lcia_methods=None,
            ssp_scenario_options_by_year=None,
        )

    assert (
        deterministic_mod.describe_expected_external_deterministic_stems(
            proj_base=project_repo,
            selection=selection,
            stems=[" ", ""],
        )
        == "no valid deterministic filenames could be derived"
    )


def test_external_downstream_asocc_shares(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    selection = _selection()
    project_name = "external_inputs_share_loader"
    proj_base = outputs_project_root(project_name=project_name)
    base_allocate_args = normalize_base_allocate_args(
        {
            "project_name": project_name,
            "source": "oecd_v2025",
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "l1_reg_aggreg": "pre",
        }
    )

    with pytest.raises(ValueError):
        downstream_mod._wide_frame(
            frame=pd.DataFrame(
                {
                    "year": [2019, 2019],
                    "value": [1.0, 2.0],
                    **_selectors(2),
                }
            ),
            asocc_method_label="CO(S)_UT(FD)",
            scenario=None,
        )

    wide = downstream_mod._wide_frame(
        frame=pd.DataFrame(
            {
                "year": [2020, 2019],
                "value": [2.0, 1.0],
                **_selectors(2),
                "level": ["level_2", "level_2"],
                "bucket": ["l2_vs_global", "l2_vs_global"],
            }
        ),
        asocc_method_label="CO(S)_UT(FD)",
        scenario=None,
    )
    assert wide.loc[0, "l1_l2_method"] == "CO(S)_UT(FD)"
    assert [column for column in wide.columns if str(column).isdigit()] == ["2019", "2020"]

    no_impact_asocc_shares = downstream_mod._asocc_shares_for_frame(
        frame=pd.DataFrame(
            {
                "year": [2005],
                "value": [1.0],
                **_selectors(),
                "level": ["level_2"],
                "bucket": ["l2_vs_global"],
            }
        ),
        fu_code="L2.a.a",
        asocc_method_label="CO(S)_UT(FD)",
        level="level_2",
        lcia_method=None,
        file_method_token="UT(FD)",
        l1_method="CO(S)",
    )
    assert len(no_impact_asocc_shares) == 1
    assert no_impact_asocc_shares[0].impacts == tuple()

    impact_asocc_shares = downstream_mod._asocc_shares_for_frame(
        frame=pd.DataFrame(
            {
                "year": [2005, 2005],
                "value": [1.0, 2.0],
                **_selectors(2),
                "impact": [" Climate risk ", "Water stress "],
                "level": ["level_2", "level_2"],
                "bucket": ["l2_vs_global", "l2_vs_global"],
            }
        ),
        fu_code="L2.a.a",
        asocc_method_label="CO(S)_UT(FD)",
        level="level_2",
        lcia_method="pb_lcia",
        file_method_token="UT(FD)",
        l1_method="CO(S)",
    )
    assert [item.impacts for item in impact_asocc_shares] == [
        ("Climate risk",),
        ("Water stress",),
    ]
    assert {item.file_stem for item in impact_asocc_shares} == {
        "external__l1_CO(S)_l2_UT(FD)__pb_lcia"
    }

    scenario_partition_asocc_shares = downstream_mod._asocc_shares_for_frame(
        frame=pd.DataFrame(
            {
                "year": [2005, 2006],
                "value": [1.0, 2.0],
                **_selectors(2),
                ASOCC_SSP_SCENARIO_COLUMN: [None, "SSP2"],
                "level": ["level_2", "level_2"],
                "bucket": ["l2_vs_global", "l2_vs_global"],
            }
        ),
        fu_code="L2.a.a",
        asocc_method_label="CO(S)_UT(FD)",
        level="level_2",
        lcia_method=None,
        file_method_token="UT(FD)",
        l1_method="CO(S)",
    )
    assert [item.file_stem for item in scenario_partition_asocc_shares] == [
        "external__l1_CO(S)_l2_UT(FD)",
        "external__l1_CO(S)_l2_UT(FD)__ssp2",
    ]
    assert all(
        ASOCC_SSP_SCENARIO_COLUMN not in item.frame_wide.columns
        for item in scenario_partition_asocc_shares
    )

    scenario_only_asocc_shares = downstream_mod._asocc_shares_for_frame(
        frame=pd.DataFrame(
            {
                "year": [2005],
                "value": [1.0],
                **_selectors(),
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"],
                "level": ["level_2"],
                "bucket": ["l2_vs_global"],
            }
        ),
        fu_code="L2.a.a",
        asocc_method_label="CO(S)_UT(FD)",
        level="level_2",
        lcia_method=None,
        file_method_token="UT(FD)",
        l1_method="CO(S)",
    )
    assert [item.file_stem for item in scenario_only_asocc_shares] == [
        "external__l1_CO(S)_l2_UT(FD)__ssp2"
    ]
    assert scenario_only_asocc_shares[0].frame_wide.loc[0, "2005"] == 1.0

    with pytest.raises(ValueError):
        downstream_mod._scenario_slices(  # noqa: SLF001
            pd.DataFrame(
                {
                    "year": [2005, 2006],
                    "value": [1.0, 2.0],
                    ASOCC_SSP_SCENARIO_COLUMN: [" ", "SSP2"],
                }
            )
        )

    assert (
        downstream_mod.load_external_asocc_shares(
            proj_base=proj_base,
            fu_code="L2.a.a",
            external_method=None,
            years=[2005],
            lcia_method=None,
            base_allocate_args=base_allocate_args,
            output_source_label="oecd_v2025",
        )
        == []
    )

    with pytest.raises(ValueError):
        downstream_mod.load_external_asocc_shares(
            proj_base=proj_base,
            fu_code="L2.a.a",
            external_method={"l1_l2_pairs": ["CO(S)::UT(FD)"]},
            years=[2005],
            lcia_method=None,
            base_allocate_args=base_allocate_args,
            output_source_label="oecd_v2025",
        )

    deterministic_dir = get_asocc_external_method_level_dir(
        proj_base=proj_base,
        storage_mode="deterministic",
        level=selection.level,
    )
    _write_table(
        deterministic_dir / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame({**_selectors(), "2005": [1.0]}),
    )
    asocc_shares = downstream_mod.load_external_asocc_shares(
        proj_base=proj_base,
        fu_code="L2.a.a",
        external_method={"l1_l2_pairs": ["CO(S)::UT(FD)"]},
        years=[2005],
        lcia_method=None,
        base_allocate_args=base_allocate_args,
        output_source_label="oecd_v2025",
    )
    assert len(asocc_shares) == 1
    assert asocc_shares[0].relative_dir == Path("results_l2_vs_global")
    assert asocc_shares[0].frame_wide.loc[0, "2005"] == 1.0

    with pytest.raises(ValueError):
        file_specs.validate_impact_contract(
            frame=pd.DataFrame({"impact": ["wrong"]}),
            path=deterministic_dir / "lcia_wrong.csv",
            lcia_method="pb_lcia",
        )
    _cc_csv_path, expected_impacts = bundled_cc_expected_impacts(lcia_method="pb_lcia")
    file_specs.validate_impact_contract(
        frame=pd.DataFrame({"impact": expected_impacts}),
        path=deterministic_dir / "lcia_ok.csv",
        lcia_method="pb_lcia",
    )


def test_uncertainty_external_rows_cover_monte_carlo_and_missing_file_paths(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    project_name = "external_inputs_uncertainty_mc"
    proj_base = outputs_project_root(project_name=project_name)
    base_allocate_args = normalize_base_allocate_args(
        {
            "project_name": project_name,
            "source": "oecd_v2025",
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "l1_reg_aggreg": "pre",
        }
    )
    loaded = _loaded_external_asocc_scope(
        proj_base=proj_base,
        base_allocate_args=base_allocate_args,
        years=[2005],
    )
    external_method = {"l1_l2_pairs": ["CO(S)::UT(FD)"]}

    with pytest.raises(ValueError):
        resolve_external_asocc_rows(
            loaded=loaded,
            external_method=external_method,
            required_runs=None,
        )

    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=proj_base,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    _write_table(
        directory / "l1_CO(S)_l2_UT(FD).csv",
        pd.DataFrame(
            {
                "run_index": [0, 1],
                "year": [2005, 2005],
                ASOCC_SSP_SCENARIO_COLUMN: [None, None],
                **_selectors(2),
                "value": [0.42, 0.52],
            }
        ),
    )

    resolved, plan = resolve_external_asocc_rows(
        loaded=loaded,
        external_method=external_method,
        required_runs=None,
    )
    assert resolved.rows.empty
    assert plan.method_labels == ("CO(S)_UT(FD)",)
    assert len(plan.monte_carlo_sources) == 1
    assert plan.monte_carlo_sources[0].available_runs == 2

    with pytest.raises(ValueError):
        resolve_external_asocc_rows(
            loaded=loaded,
            external_method=external_method,
            required_runs=3,
        )


def test_uncertainty_external_rows_use_downstream_lcia_scope_for_compact_inputs(
    allocation_dummy_repo,
) -> None:
    del allocation_dummy_repo
    project_name = "external_inputs_uncertainty_downstream_lcia"
    proj_base = outputs_project_root(project_name=project_name)
    base_allocate_args = normalize_base_allocate_args(
        {
            "project_name": project_name,
            "source": "oecd_v2025",
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "l1_reg_aggreg": "pre",
        }
    )
    loaded = _loaded_external_asocc_scope(
        proj_base=proj_base,
        base_allocate_args=base_allocate_args,
        years=[2005],
    )
    external_method = {"l1_l2_pairs": ["CO(S)::UT(FD)"]}
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=proj_base,
        storage_mode="monte_carlo",
        level=selection.level,
    )
    compact_dir = directory / "l1_CO(S)_l2_UT(FD)__ef_3.1"
    compact_dir.mkdir(parents=True, exist_ok=True)
    _cc_csv_path, impacts = bundled_cc_expected_impacts(lcia_method="ef_3.1")
    identity = pd.DataFrame(
        {
            "public_row_id": range(len(impacts)),
            "year": [2005] * len(impacts),
            ASOCC_SSP_SCENARIO_COLUMN: [None] * len(impacts),
            **_selectors(len(impacts)),
            "impact": impacts,
        }
    )
    identity.to_csv(compact_dir / "public_row_identity.csv", index=False)
    pd.DataFrame(
        {
            "run_index": [0],
            **{str(index): [0.1 + index] for index in range(len(impacts))},
        }
    ).to_csv(compact_dir / "asocc_runs.csv", index=False)

    assert external_asocc_has_monte_carlo_rows(
        loaded=loaded,
        external_method=external_method,
        external_lcia_methods=["ef_3.1"],
    )
    resolved, plan = resolve_external_asocc_rows(
        loaded=loaded,
        external_method=external_method,
        required_runs=1,
        external_lcia_methods=["ef_3.1"],
    )

    assert resolved.rows.empty
    assert plan.method_labels == ("CO(S)_UT(FD)",)
    assert len(plan.monte_carlo_sources) == 1
    assert plan.monte_carlo_sources[0].file_selections[0].lcia_method == "ef_3.1"
    assert plan.monte_carlo_sources[0].available_runs == 1


def test_resolve_year_assignments_and_candidate_files_cover_scenario_routing(
    project_repo: Path,
) -> None:
    selection = _selection()
    directory = get_asocc_external_method_level_dir(
        proj_base=project_repo,
        storage_mode="deterministic",
        level=selection.level,
    )
    base_path = directory / "l1_CO(S)_l2_UT(FD)__pb_lcia.csv"
    scenario_path = directory / "l1_CO(S)_l2_UT(FD)__pb_lcia__ssp2.csv"
    _write_table(base_path, pd.DataFrame({"2019": [1.0], "2020": [2.0]}))
    _write_table(scenario_path, pd.DataFrame({"2021": [3.0]}))

    specs = file_specs.candidate_files(
        proj_base=project_repo,
        selection=selection,
        lcia_methods=["pb_lcia"],
    )
    year_map = file_specs.resolve_external_asocc_year_assignments(
        specs=specs,
        selection=selection,
        years=[2019, 2020, 2021],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year={2019: [None], 2020: [None], 2021: ["SSP2"]},
    )
    assert year_map[base_path] == [2019, 2020]
    assert year_map[scenario_path] == [2021]

    scenario_path_ssp1 = directory / "l1_CO(S)_l2_UT(FD)__pb_lcia__ssp1.csv"
    _write_table(scenario_path_ssp1, pd.DataFrame({"2021": [4.0], "2022": [4.5]}))
    _write_table(scenario_path, pd.DataFrame({"2021": [3.0], "2022": [5.0]}))
    specs = file_specs.candidate_files(
        proj_base=project_repo,
        selection=selection,
        lcia_methods=["pb_lcia"],
    )
    multi_year_map = file_specs.resolve_external_asocc_year_assignments(
        specs=specs,
        selection=selection,
        years=[2022],
        lcia_methods=["pb_lcia"],
        ssp_scenario_options_by_year={2022: ["SSP1", "SSP2"]},
    )
    assert multi_year_map[scenario_path_ssp1] == [2022]
    assert multi_year_map[scenario_path] == [2022]
