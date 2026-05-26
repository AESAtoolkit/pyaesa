from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq

from pyaesa import prepare_external_inputs
from pyaesa.external_inputs.lca import io as io_mod
from pyaesa.external_inputs.lca import figures as figures_mod
from pyaesa.external_inputs.lca import naming as naming_mod
from pyaesa.external_inputs.lca import paths as path_mod
from pyaesa.external_inputs.lca.deterministic import load_external_lca_deterministic_rows
from pyaesa.external_inputs.lca.monte_carlo import (
    ExternalLCAMonteCarloSource,
    external_lca_values_for_run_rows,
    external_lca_values_for_runs,
    load_external_lca_monte_carlo_source_from_path,
)
from pyaesa.external_inputs.lca.monte_carlo_stream import load_external_lca_long_matrix_source
from pyaesa.external_inputs.lca.io import ExternalLCAFileSpec
from pyaesa.external_inputs.lca.paths import external_lca_deterministic_dir
from pyaesa.external_inputs.shared.compact_matrix import compact_run_matrix_values_for_runs
from pyaesa.external_inputs.shared.compact_matrix import load_compact_run_matrix
from pyaesa.external_inputs.shared.compact_matrix import load_compact_run_matrix_source
from pyaesa.external_inputs.shared.tabular import read_projected_table
from pyaesa.asocc.runtime.request.normalization import normalize_base_allocate_args
from pyaesa.shared.lcia.contracts import bundled_cc_expected_impact_units
from pyaesa.shared.runtime.scenario.columns import EXT_LCA_SSP_SCENARIO_COLUMN
from pyaesa.shared.selectors.fu_axes import expected_fu_selector_columns


class _StatusRecorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def show(self, message: str) -> None:
        self.messages.append(message)

    def log_message(self, message: str, *, persistent: bool = True) -> None:
        self.messages.append(message)

    def clear_transient(self) -> None:
        pass

    def finish(self) -> None:
        pass


def _expected_pairs(*, lcia_method: str) -> list[tuple[str, str]]:
    _cc_path, pairs = bundled_cc_expected_impact_units(lcia_method=lcia_method)
    return pairs


def _wide_lca_frame(
    *,
    lcia_method: str,
    run_index: int | None = None,
    include_scenario_metadata: bool = True,
) -> pd.DataFrame:
    pairs = _expected_pairs(lcia_method=lcia_method)
    rows: list[dict[str, object]] = []
    for index, (impact, impact_unit) in enumerate(pairs):
        row: dict[str, object] = {
            "r_p": "FR" if index % 2 == 0 else "DE",
            "s_p": "D" if index % 2 == 0 else "X",
            "impact": impact,
            "impact_unit": impact_unit,
            "category": "category_a",
            "model": "model_a",
            "technology": "technology_a" if index % 2 == 0 else "technology_b",
            "missing_driver": float("nan"),
            "2019": float(index + 1),
            "2020": float(index + 11),
        }
        if include_scenario_metadata:
            row["scenario"] = "scenario_a"
            row[EXT_LCA_SSP_SCENARIO_COLUMN] = "SSP2"
        if run_index is not None:
            row["run_index"] = run_index
        rows.append(row)
    return pd.DataFrame(rows)


def test_external_lca_io_covers_discovery_parsing_normalization_and_reading(
    project_repo: Path,
) -> None:
    lcia_method = "pb_lcia"
    version_name = "supplier_v1"
    work_dir = project_repo / "external_lca_io"
    work_dir.mkdir(parents=True, exist_ok=True)
    frame = _wide_lca_frame(lcia_method=lcia_method, include_scenario_metadata=False)
    expected_pairs = _expected_pairs(lcia_method=lcia_method)
    expected_count = len(expected_pairs)
    csv_path = work_dir / f"{version_name}__{lcia_method}.csv"
    pickle_path = work_dir / f"{version_name}__{lcia_method}.pickle"
    parquet_path = work_dir / f"{version_name}__{lcia_method}.parquet"
    nested_dir = work_dir / "nested"
    nested_dir.mkdir()
    frame.to_csv(csv_path, index=False)
    frame.to_pickle(pickle_path)
    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(nested_dir / f"{version_name}__{lcia_method}.csv", index=False)
    (work_dir / "notes.txt").write_text("ignored", encoding="utf-8")

    assert io_mod.discover_external_lca_files(work_dir / "missing") == tuple()
    assert io_mod.discover_external_lca_files(work_dir) == (
        csv_path,
        parquet_path,
        pickle_path,
    )
    assert io_mod.matching_external_lca_specs(
        directory=work_dir,
        version_name=version_name,
        lcia_method=lcia_method,
    ) == (
        ExternalLCAFileSpec(
            path=csv_path,
            scenario=None,
            years=(2019, 2020),
            version_name=version_name,
            lcia_method=lcia_method,
        ),
        ExternalLCAFileSpec(
            path=parquet_path,
            scenario=None,
            years=(2019, 2020),
            version_name=version_name,
            lcia_method=lcia_method,
        ),
        ExternalLCAFileSpec(
            path=pickle_path,
            scenario=None,
            years=(2019, 2020),
            version_name=version_name,
            lcia_method=lcia_method,
        ),
    )
    assert io_mod.external_lca_expected_stems(
        version_name=version_name,
        lcia_method=lcia_method,
        years=[2019, 2020],
        ssp_scenario_options_by_year=None,
    ) == [f"{version_name}__{lcia_method}"]
    assert io_mod.external_lca_expected_stems(
        version_name=version_name,
        lcia_method=lcia_method,
        years=[2019, 2020],
        ssp_scenario_options_by_year={2020: []},
    ) == [f"{version_name}__{lcia_method}"]

    parsed = io_mod.parse_external_lca_filename(path=csv_path)
    assert parsed.lcia_method == lcia_method
    assert parsed.version_name == version_name
    assert parsed.scenario is None
    assert parsed.years == (2019, 2020)

    mc_path = work_dir / f"runs__{version_name}__{lcia_method}.csv"
    frame.assign(run_index=0).to_csv(mc_path, index=False)
    scenario_path = work_dir / f"{version_name}__{lcia_method}__ssp2.csv"
    frame.assign(run_index=0).to_csv(scenario_path, index=False)
    parsed_scenario = io_mod.parse_external_lca_filename(path=scenario_path)
    assert parsed_scenario.scenario == "SSP2"
    assert parsed_scenario.lcia_method == lcia_method
    uppercase_scenario_path = work_dir / f"{version_name}__{lcia_method}__SSP2.csv"
    frame.assign(run_index=0).to_csv(uppercase_scenario_path, index=False)
    with pytest.raises(
        ValueError,
    ):
        io_mod.parse_external_lca_filename(path=uppercase_scenario_path)

    with pytest.raises(
        ValueError,
    ):
        io_mod.parse_external_lca_filename(path=mc_path)

    pd.testing.assert_frame_equal(io_mod.read_external_lca(csv_path), frame, check_dtype=False)
    pd.testing.assert_frame_equal(
        io_mod.read_external_lca(pickle_path),
        frame,
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        io_mod.read_external_lca(parquet_path),
        frame,
        check_dtype=False,
    )
    with pytest.raises(ValueError):
        io_mod.read_external_lca(work_dir / "pb_lcia.txt")
    incomplete = work_dir / "supplier_v1__pb_lcia__missing.csv"
    frame.drop(columns=["impact_unit"]).to_csv(incomplete, index=False)
    with pytest.raises(ValueError):
        io_mod.read_external_lca(incomplete)
    missing_year = work_dir / "pb_lcia__missing_year.csv"
    frame.assign(**{"2020": [1.0] + [pd.NA] * (len(frame) - 1)}).to_csv(
        missing_year,
        index=False,
    )
    with pytest.raises(
        ValueError,
    ):
        io_mod.normalize_external_lca_deterministic_rows(
            frame=io_mod.read_external_lca(missing_year),
            path=missing_year,
            lcia_method=lcia_method,
        )
    with pytest.raises(ValueError):
        io_mod.validate_external_lca_contract(
            frame=frame.assign(impact_unit="mismatch"),
            path=csv_path,
            lcia_method=lcia_method,
        )

    normalized = io_mod.normalize_external_lca_deterministic_rows(
        frame=frame,
        path=csv_path,
        lcia_method=lcia_method,
    )
    assert list(normalized.columns[:4]) == ["year", "impact", "impact_unit", "value"]
    assert normalized["year"].tolist() == [2019] * expected_count + [2020] * expected_count
    expected_units = [pair[1] for pair in expected_pairs]
    assert normalized["value"].tolist() == [float(index + 1) for index in range(expected_count)] + [
        float(index + 11) for index in range(expected_count)
    ]
    assert normalized["impact_unit"].tolist() == expected_units + expected_units
    with pytest.raises(ValueError):
        io_mod.normalize_external_lca_deterministic_rows(
            frame=frame.assign(impact_unit="wrong_unit"),
            path=csv_path,
            lcia_method=lcia_method,
        )
    with pytest.raises(ValueError):
        io_mod.normalize_external_lca_deterministic_rows(
            frame=frame.assign(year=2019, value=1.0),
            path=csv_path,
            lcia_method=lcia_method,
        )

    with pytest.raises(ValueError):
        io_mod.normalize_external_lca_deterministic_rows(
            frame=frame.drop(columns=["2019", "2020"]),
            path=csv_path,
            lcia_method=lcia_method,
        )

    assert io_mod.detect_year_columns_external(frame) == [2019, 2020]
    assert io_mod.detect_year_columns_external(frame.rename(columns={"2020": "1800"})) == [2019]
    assert io_mod.detect_year_columns_external(
        pd.DataFrame({"year": ["2019", "2020", None, "2019"]})
    ) == [2019, 2020]
    assert io_mod.external_source_driver_columns(frame) == [
        "category",
        "missing_driver",
        "model",
        "technology",
    ]

    with pytest.raises(ValueError):
        io_mod.validate_external_lca_extra_columns(
            frame=frame.assign(lcia_method=lcia_method),
            path=csv_path,
        )
    with pytest.raises(ValueError):
        io_mod.validate_external_lca_extra_columns(
            frame=frame.assign(ssp_scenario="SSP2"),
            path=csv_path,
        )
    with pytest.raises(ValueError):
        io_mod.validate_external_lca_extra_columns(
            frame=frame.assign(**{EXT_LCA_SSP_SCENARIO_COLUMN: "SSP2"}),
            path=csv_path,
        )


def test_external_lca_year_routing() -> None:
    version_name = "supplier_v1"
    specs = (
        ExternalLCAFileSpec(
            path=Path("historical.csv"),
            scenario=None,
            years=(2019,),
            version_name=version_name,
            lcia_method="pb_lcia",
        ),
        ExternalLCAFileSpec(
            path=Path("ssp2.csv"),
            scenario="SSP2",
            years=(2020,),
            version_name=version_name,
            lcia_method="pb_lcia",
        ),
    )
    assignments = io_mod.resolve_external_lca_year_assignments(
        specs=specs,
        version_name=version_name,
        lcia_method="pb_lcia",
        years=[2019, 2020],
        ssp_scenario_options_by_year={2020: ["SSP2"]},
    )
    assert assignments == {
        Path("historical.csv"): [2019],
        Path("ssp2.csv"): [2020],
    }
    scenario_specs = (
        ExternalLCAFileSpec(
            path=Path("ssp1.csv"),
            scenario="SSP1",
            years=(2030,),
            version_name=version_name,
            lcia_method="pb_lcia",
        ),
        ExternalLCAFileSpec(
            path=Path("ssp2.csv"),
            scenario="SSP2",
            years=(2030,),
            version_name=version_name,
            lcia_method="pb_lcia",
        ),
    )
    assert io_mod.resolve_external_lca_year_assignments(
        specs=scenario_specs,
        version_name=version_name,
        lcia_method="pb_lcia",
        years=[2030],
        ssp_scenario_options_by_year=None,
    ) == {
        Path("ssp1.csv"): [2030],
        Path("ssp2.csv"): [2030],
    }
    assert io_mod.resolve_external_lca_year_assignments(
        specs=scenario_specs,
        version_name=version_name,
        lcia_method="pb_lcia",
        years=[2030],
        ssp_scenario_options_by_year={2030: ["SSP2"]},
    ) == {Path("ssp2.csv"): [2030]}
    with pytest.raises(ValueError, match="full allowed scenario set"):
        io_mod.resolve_external_lca_year_assignments(
            specs=scenario_specs,
            version_name=version_name,
            lcia_method="pb_lcia",
            years=[2030],
            ssp_scenario_options_by_year={2030: ["SSP1", "SSP5"]},
        )

    with pytest.raises(ValueError):
        io_mod.validate_scenario_inventory(
            specs=(
                ExternalLCAFileSpec(
                    path=Path("first.csv"),
                    scenario=None,
                    years=(2019,),
                    version_name=version_name,
                    lcia_method="pb_lcia",
                ),
                ExternalLCAFileSpec(
                    path=Path("second.csv"),
                    scenario=None,
                    years=(2019,),
                    version_name=version_name,
                    lcia_method="pb_lcia",
                ),
            ),
            family_label="external LCA",
            item_label="LCIA method 'pb_lcia'",
        )

    with pytest.raises(ValueError):
        io_mod.validate_scenario_inventory(
            specs=(
                ExternalLCAFileSpec(
                    path=Path("historical.csv"),
                    scenario=None,
                    years=(2019,),
                    version_name=version_name,
                    lcia_method="pb_lcia",
                ),
                ExternalLCAFileSpec(
                    path=Path("ssp2.csv"),
                    scenario="SSP2",
                    years=(2019,),
                    version_name=version_name,
                    lcia_method="pb_lcia",
                ),
            ),
            family_label="external LCA",
            item_label="LCIA method 'pb_lcia'",
        )

    with pytest.raises(ValueError):
        io_mod.resolve_external_lca_year_assignments(
            specs=specs,
            version_name=version_name,
            lcia_method="pb_lcia",
            years=[2030],
            ssp_scenario_options_by_year=None,
        )


def test_external_lca_path_owners_are_pure(
    project_repo: Path,
) -> None:
    root = path_mod.external_lca_root(project_base=project_repo)
    deterministic_dir = path_mod.external_lca_deterministic_dir(project_base=project_repo)
    monte_carlo_dir = path_mod.external_lca_monte_carlo_dir(project_base=project_repo)
    figures_dir = path_mod.external_lca_deterministic_figures_dir(project_base=project_repo)
    mc_figures_dir = path_mod.external_lca_monte_carlo_figures_dir(project_base=project_repo)

    assert root == project_repo / "A_lca" / "external_lca"
    assert deterministic_dir == project_repo / "A_lca" / "external_lca" / "deterministic"
    assert monte_carlo_dir == project_repo / "A_lca" / "external_lca" / "monte_carlo"
    assert figures_dir == project_repo / "A_lca" / "external_lca" / "deterministic" / "figures"
    assert mc_figures_dir == project_repo / "A_lca" / "external_lca" / "monte_carlo" / "figures"
    assert not deterministic_dir.exists()
    assert not monte_carlo_dir.exists()
    assert not figures_dir.exists()
    assert not mc_figures_dir.exists()


def test_external_lca_selector_version_and_fu_contracts(project_repo: Path) -> None:
    lcia_method = "pb_lcia"
    frame = _wide_lca_frame(lcia_method=lcia_method, include_scenario_metadata=False)
    base_args = {"fu_code": "L2.a.a"}
    path = project_repo / "supplier_v1__pb_lcia.csv"

    assert (
        naming_mod.normalize_external_lca_version_name(
            " supplier.v1 ",
            argument_name="version",
        )
        == "supplier.v1"
    )
    with pytest.raises(ValueError):
        naming_mod.normalize_external_lca_version_name(None, argument_name="version")
    with pytest.raises(ValueError):
        naming_mod.normalize_external_lca_version_name(" ", argument_name="version")
    with pytest.raises(ValueError):
        naming_mod.normalize_external_lca_version_name("bad__name", argument_name="version")

    assert expected_fu_selector_columns(fu_code="L1.a") == ("r_f",)
    assert expected_fu_selector_columns(fu_code="L1.b") == ("r_p",)
    assert expected_fu_selector_columns(fu_code="L2.a.a") == ("r_p", "s_p")
    assert expected_fu_selector_columns(fu_code="L2.b.a") == ("r_p", "s_p", "r_f")
    assert expected_fu_selector_columns(fu_code="L2.b.b") == ("r_p", "s_p", "r_c")
    assert expected_fu_selector_columns(fu_code="L2.c.a") == ("s_p", "r_f")
    assert expected_fu_selector_columns(fu_code="L2.c.b") == ("s_p", "r_c")
    with pytest.raises(ValueError):
        expected_fu_selector_columns(fu_code="bad")

    with pytest.raises(ValueError):
        io_mod.validate_external_lca_selector_columns(
            frame=frame.drop(columns=["s_p"]),
            path=path,
            base_allocate_args=base_args,
        )
    with pytest.raises(ValueError):
        io_mod.validate_external_lca_selector_columns(
            frame=frame.assign(s_p=" "),
            path=path,
            base_allocate_args=base_args,
        )
    with pytest.raises(ValueError):
        io_mod.validate_external_lca_selector_columns(
            frame=frame.assign(r_c="EU"),
            path=path,
            base_allocate_args=base_args,
        )
    io_mod.validate_external_lca_selector_columns(
        frame=frame.assign(r_c=pd.NA),
        path=path,
        base_allocate_args=base_args,
    )


def test_external_lca_figure_normalization_and_rendering(allocation_dummy_repo) -> None:
    del allocation_dummy_repo
    lcia_method = "gwp100_lcia"
    base_args = normalize_base_allocate_args(
        {
            "project_name": "external_lca_figures_multi",
            "source": "exiobase_396_ixi",
            "years": [2005, 2006],
            "fu_code": "L2.a.a",
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
            "lcia_method": lcia_method,
        }
    )
    report = prepare_external_inputs(project_name="external_lca_figures_multi")
    impact, impact_unit = _expected_pairs(lcia_method=lcia_method)[0]
    normalized = figures_mod._normalize_for_figures(  # noqa: SLF001
        frame=pd.DataFrame(
            {
                "year": [2005],
                "impact": [impact],
                "impact_unit": [impact_unit],
                "value": [1.0],
                "r_p": ["FR"],
                "s_p": ["D"],
                "l1_l2_method": ["UT(FD)"],
                "reference_year": [2005],
                EXT_LCA_SSP_SCENARIO_COLUMN: ["SSP2"],
            }
        ),
        lcia_method=lcia_method,
        value_column="value",
    )
    assert set(normalized["ssp_scenario"]) == {"SSP2"}
    assert set(normalized["lcia_method"]) == {lcia_method}
    assert "lca_value" in normalized.columns
    minimal_normalized = figures_mod._normalize_for_figures(  # noqa: SLF001
        frame=pd.DataFrame(
            {
                "year": [2005],
                "impact": [impact],
                "impact_unit": [impact_unit],
                "value": [1.0],
            }
        ),
        lcia_method=lcia_method,
        value_column="value",
    )
    assert "ssp_scenario" not in minimal_normalized.columns

    deterministic_dir = external_lca_deterministic_dir(project_base=report.project_root)
    deterministic_dir.mkdir(parents=True, exist_ok=True)
    multi_year = _wide_lca_frame(lcia_method=lcia_method, include_scenario_metadata=False)
    multi_year["2005"] = multi_year["2019"]
    multi_year.drop(columns=["2019", "2020"]).loc[
        :, ["r_p", "s_p", "impact", "impact_unit", "2005"]
    ].to_csv(
        deterministic_dir / "supplier_v1__gwp100_lcia.csv",
        index=False,
    )
    ssp_year = _wide_lca_frame(lcia_method=lcia_method, include_scenario_metadata=False)
    ssp_year["2006"] = ssp_year["2020"]
    ssp_year.drop(columns=["2019", "2020"]).loc[
        :, ["r_p", "s_p", "impact", "impact_unit", "2006"]
    ].to_csv(
        deterministic_dir / "supplier_v1__gwp100_lcia__ssp2.csv",
        index=False,
    )
    rows, _paths = load_external_lca_deterministic_rows(
        proj_base=report.project_root,
        version_name="supplier_v1",
        lcia_method=lcia_method,
        years=[2005, 2006],
        ssp_scenario_options_by_year={2006: ["SSP2"]},
        base_allocate_args=base_args,
    )
    assert rows is not None
    paths = figures_mod.render_external_lca_deterministic_figures_from_rows(
        proj_base=report.project_root,
        version_name="supplier_v1",
        lcia_method=lcia_method,
        rows=rows.rename(columns={"value": "lca_value"}),
        output_format="png",
        dpi=10,
    )
    assert paths
    assert all(path.exists() for path in paths)
    assert all(path.parent == deterministic_dir / "figures" for path in paths)
    assert all(path.name.startswith("supplier_v1__gwp100_lcia__") for path in paths)
    assert all("__SSP2" in path.stem for path in paths)
    assert all("rf_all" not in path.name and "rc_all" not in path.name for path in paths)
    deterministic_status = _StatusRecorder()
    assert figures_mod.render_external_lca_deterministic_figures_from_rows(
        proj_base=report.project_root,
        version_name="supplier_v1",
        lcia_method=lcia_method,
        rows=rows.loc[rows["year"].astype(int).eq(2005)].rename(columns={"value": "lca_value"}),
        output_format="png",
        dpi=10,
        status=deterministic_status,
    )
    assert len(deterministic_status.messages) == 2
    assert deterministic_status.messages[0].startswith("[external_lca] Generating figure")
    assert "Generating figures" not in deterministic_status.messages[0]
    assert deterministic_status.messages[1].startswith("[external_lca] Generated figure")

    status = _StatusRecorder()
    source_values = pd.DataFrame({"0": [1.0], "1": [2.0]}).to_numpy(dtype=float).T
    source = ExternalLCAMonteCarloSource(
        version_name="supplier_mc",
        lcia_method=lcia_method,
        identity=pd.DataFrame(
            {
                "public_row_id": [0],
                "year": [2005],
                "impact": [impact],
                "impact_unit": [impact_unit],
                "r_p": ["FR"],
                "s_p": ["D"],
                EXT_LCA_SSP_SCENARIO_COLUMN: ["SSP2"],
            }
        ),
        run_indices=pd.Series([0, 1]).to_numpy(dtype=int),
        paths=(deterministic_dir / "supplier_mc.csv",),
        values_for_runs=lambda run_indices: source_values[run_indices],
    )
    assert external_lca_values_for_runs(
        source=source,
        run_indices=source.run_indices,
    ).tolist() == [[1.0], [2.0]]
    assert external_lca_values_for_run_rows(
        source=source,
        run_indices=np.array([0], dtype=np.int64),
        row_positions=np.empty(0, dtype=np.int64),
    ).shape == (1, 0)
    assert external_lca_values_for_run_rows(
        source=source,
        run_indices=np.array([1], dtype=np.int64),
        row_positions=np.array([0], dtype=np.int64),
    ).tolist() == [[2.0]]
    with pytest.raises(ValueError):
        external_lca_values_for_run_rows(
            source=source,
            run_indices=np.array([2], dtype=np.int64),
            row_positions=np.array([0], dtype=np.int64),
        )
    mc_paths = figures_mod.render_external_lca_uncertainty_figures_from_source(
        proj_base=report.project_root,
        source=source,
        output_format="png",
        dpi=10,
        completed_runs=1,
        status=status,
    )
    assert mc_paths
    assert len(status.messages) == 2
    assert status.messages[0].startswith("[external_lca] Generating figure")
    assert "Generating figures" not in status.messages[0]
    assert status.messages[1].startswith("[external_lca] Generated figure")
    assert figures_mod.render_external_lca_uncertainty_figures_from_source(
        proj_base=report.project_root,
        source=source,
        output_format="svg",
        dpi=10,
        completed_runs=1,
    )

    scoped_requests: list[tuple[int, ...]] = []
    multi_year_values = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float64)

    def values_for_scoped_rows(
        run_indices: np.ndarray,
        row_positions: np.ndarray,
    ) -> np.ndarray:
        scoped_requests.append(tuple(int(value) for value in row_positions.tolist()))
        return multi_year_values[run_indices][:, row_positions]

    multi_year_source = ExternalLCAMonteCarloSource(
        version_name="supplier_mc_multi",
        lcia_method=lcia_method,
        identity=pd.DataFrame(
            {
                "public_row_id": [0, 1],
                "year": [2005, 2006],
                "impact": [impact, impact],
                "impact_unit": [impact_unit, impact_unit],
                "r_p": ["FR", "FR"],
                "s_p": ["D", "D"],
                EXT_LCA_SSP_SCENARIO_COLUMN: [None, "SSP2"],
            }
        ),
        run_indices=np.array([0, 1], dtype=np.int64),
        paths=(deterministic_dir / "supplier_mc_multi.csv",),
        values_for_runs=lambda run_indices: multi_year_values[run_indices],
        values_for_run_rows=values_for_scoped_rows,
    )
    assert figures_mod.render_external_lca_uncertainty_figures_from_source(
        proj_base=report.project_root,
        source=multi_year_source,
        output_format="png",
        dpi=10,
        completed_runs=2,
    )
    assert scoped_requests == [(0, 1)]


def test_external_lca_io_covers_contract_failures_and_year_detection(
    project_repo: Path,
) -> None:
    lcia_method = "pb_lcia"
    missing_dir = project_repo / "missing_external_lca"
    assert io_mod.discover_external_lca_files(missing_dir) == tuple()

    frame = _wide_lca_frame(lcia_method=lcia_method)
    pickle_path = project_repo / "external_lca_projected.pickle"
    frame.to_pickle(pickle_path)
    assert read_projected_table(pickle_path).num_rows == len(frame)
    mismatched = frame.copy()
    mismatched.loc[0, "impact_unit"] = "wrong_unit"
    with pytest.raises(ValueError):
        io_mod.validate_external_lca_contract(
            frame=mismatched,
            path=project_repo / "bad_external_lca.csv",
            lcia_method=lcia_method,
        )

    assert io_mod.detect_year_columns_external(
        pd.DataFrame(
            {
                "1899": [1.0],
                "2019": [2.0],
                "2200": [3.0],
                "label": ["not_a_year"],
            }
        )
    ) == [2019]


def test_external_lca_io_edges_cover_invalid_tokens_and_render_normalization(
    project_repo: Path,
) -> None:
    assert io_mod.detect_year_columns_external(
        pd.DataFrame({"year": ["2020", "bad", None, 2030]})
    ) == [2020, 2030]

    path = project_repo / "external_lca_long.csv"
    with pytest.raises(ValueError):
        io_mod.validate_external_lca_extra_columns(  # noqa: SLF001
            frame=pd.DataFrame({"lcia_method": ["pb_lcia"]}),
            path=path,
        )
    with pytest.raises(ValueError):
        io_mod.validate_external_lca_extra_columns(  # noqa: SLF001
            frame=pd.DataFrame({"ssp_scenario": ["SSP2"]}),
            path=path,
        )
    with pytest.raises(ValueError):
        io_mod.validate_external_lca_extra_columns(  # noqa: SLF001
            frame=pd.DataFrame({EXT_LCA_SSP_SCENARIO_COLUMN: ["SSP2"]}),
            path=path,
        )


def test_external_compact_run_matrix_reports_inventory_and_corrupt_rows(
    project_repo: Path,
) -> None:
    matrix_dir = project_repo / "external_compact_matrix"
    matrix_dir.mkdir()
    runs_path = matrix_dir / "lca_runs.csv"
    pd.DataFrame({"public_row_id": [0, 1]}).to_csv(
        matrix_dir / "public_row_identity.csv",
        index=False,
    )
    pd.DataFrame({"run_index": [0, 1], "0": [1.0, 2.0], "1": [3.0, 4.0]}).to_csv(
        runs_path,
        index=False,
    )

    source = load_compact_run_matrix_source(
        directory=matrix_dir,
        run_file_name="lca_runs.csv",
        context="External LCA",
    )
    matrix = load_compact_run_matrix(
        directory=matrix_dir,
        run_file_name="lca_runs.csv",
        context="External LCA",
    )

    assert matrix.values.tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert source.values_for_runs(np.empty(0, dtype=np.int64)).shape == (0, 2)
    assert source.values_for_run_rows(
        np.array([1], dtype=np.int64),
        np.array([1], dtype=np.int64),
    ).tolist() == [[4.0]]
    assert compact_run_matrix_values_for_runs(
        runs_path=runs_path,
        public_row_ids=np.array([1], dtype=np.int64),
        run_indices=np.array([0, 1], dtype=np.int64),
        requested_runs=np.array([1], dtype=np.int64),
    ).tolist() == [[4.0]]
    assert compact_run_matrix_values_for_runs(
        runs_path=runs_path,
        public_row_ids=np.array([1], dtype=np.int64),
        run_indices=np.array([0, 1], dtype=np.int64),
        requested_runs=np.empty(0, dtype=np.int64),
    ).shape == (0, 1)
    with pytest.raises(ValueError, match="missing requested run_index values \\[2\\]"):
        source.values_for_runs(np.array([2], dtype=np.int64))

    pd.DataFrame({"run_index": [0], "0": [1.0]}).to_csv(runs_path, index=False)
    with pytest.raises(ValueError, match="missing requested run rows"):
        compact_run_matrix_values_for_runs(
            runs_path=runs_path,
            public_row_ids=np.array([0], dtype=np.int64),
            run_indices=np.array([0, 1], dtype=np.int64),
            requested_runs=np.array([1], dtype=np.int64),
        )


def test_external_lca_monte_carlo_real_sources_cover_empty_and_streamed_requests(
    project_repo: Path,
) -> None:
    lcia_method = "gwp100_lcia"
    impact, impact_unit = _expected_pairs(lcia_method=lcia_method)[0]
    work_dir = project_repo / "external_lca_monte_carlo_real_sources"
    work_dir.mkdir()
    base_args = {"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]}
    rows = pd.DataFrame(
        {
            "run_index": [0, 1],
            "year": [2005, 2005],
            EXT_LCA_SSP_SCENARIO_COLUMN: [None, None],
            "impact": [impact, impact],
            "impact_unit": [impact_unit, impact_unit],
            "r_p": ["FR", "FR"],
            "s_p": ["D", "D"],
            "value": [1.0, 2.0],
        }
    )
    pickle_path = work_dir / f"supplier_v1__{lcia_method}.pickle"
    rows.to_pickle(pickle_path)
    frame_source = load_external_lca_monte_carlo_source_from_path(
        path=pickle_path,
        version_name="supplier_v1",
        lcia_method=lcia_method,
        years=[2005],
        base_allocate_args=base_args,
    )
    frame_values_for_runs = frame_source.values_for_runs
    frame_values_for_run_rows = frame_source.values_for_run_rows
    assert frame_values_for_runs is not None
    assert frame_values_for_run_rows is not None
    assert frame_values_for_runs(np.empty(0, dtype=np.int64)).shape == (0, 1)
    assert frame_values_for_run_rows(
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
    ).tolist() == [[1.0]]

    identity_count = 65_537
    regions = np.array([f"R{index:05d}" for index in range(identity_count)], dtype=object)
    row_count = identity_count * 2
    run_index = np.repeat(np.array([0, 1], dtype=np.int64), identity_count)
    stream_rows = pd.DataFrame(
        {
            "run_index": run_index,
            "year": np.full(row_count, 2005, dtype=np.int64),
            EXT_LCA_SSP_SCENARIO_COLUMN: np.full(row_count, None, dtype=object),
            "impact": np.full(row_count, impact, dtype=object),
            "impact_unit": np.full(row_count, impact_unit, dtype=object),
            "r_p": np.tile(regions, 2),
            "s_p": np.full(row_count, "D", dtype=object),
            "value": run_index.astype(np.float64) + np.arange(row_count, dtype=np.float64) / 1e9,
        }
    )
    stream_path = work_dir / f"supplier_v1__{lcia_method}.parquet"
    pq.write_table(
        pa.Table.from_pandas(stream_rows, preserve_index=False),
        stream_path,
        row_group_size=row_count,
    )
    stream_source = load_external_lca_long_matrix_source(
        path=stream_path,
        lcia_method=lcia_method,
        years=[2005],
        base_allocate_args={"fu_code": "L2.a.a", "r_p": regions.tolist(), "s_p": ["D"]},
    )
    values = stream_source.values_for_runs(np.array([1], dtype=np.int64))
    assert values.shape == (1, identity_count)
    np.testing.assert_allclose(
        values[0, [0, -1]],
        [1.0 + identity_count / 1e9, 1.0 + (row_count - 1) / 1e9],
    )
    selected_values = stream_source.values_for_run_rows(
        np.array([1], dtype=np.int64),
        np.array([identity_count - 1], dtype=np.int64),
    )
    np.testing.assert_allclose(selected_values, [[1.0 + (row_count - 1) / 1e9]])
    assert stream_source.values_for_run_rows(
        np.array([1], dtype=np.int64),
        np.empty(0, dtype=np.int64),
    ).shape == (1, 0)
