from pathlib import Path

import pandas as pd
import pytest

from pyaesa.external_inputs.lca import deterministic as det_mod
from pyaesa.external_inputs.lca import io as io_mod
from pyaesa.shared.lcia.contracts import bundled_cc_expected_impact_units
from pyaesa.shared.runtime.scenario.columns import EXT_LCA_SSP_SCENARIO_COLUMN


def _expected_pairs(*, lcia_method: str) -> list[tuple[str, str]]:
    _cc_path, pairs = bundled_cc_expected_impact_units(lcia_method=lcia_method)
    return pairs


def _wide_frame(
    *,
    lcia_method: str,
    year: int,
    include_scenario_metadata: bool = False,
) -> pd.DataFrame:
    _cc_path, pairs = bundled_cc_expected_impact_units(lcia_method=lcia_method)
    rows: list[dict[str, object]] = []
    for index, (impact, impact_unit) in enumerate(pairs):
        row = {
            "r_p": "FR" if index % 2 == 0 else "DE",
            "s_p": "D" if index % 2 == 0 else "X",
            "impact": impact,
            "impact_unit": impact_unit,
            "category": "category_a",
            "model": "model_a",
            "technology": "technology_a" if index % 2 == 0 else "technology_b",
            str(year): float(index + 1),
        }
        if include_scenario_metadata:
            row["scenario"] = "scenario_a"
            row[EXT_LCA_SSP_SCENARIO_COLUMN] = "SSP2"
        rows.append(row)
    return pd.DataFrame(rows)


def test_load_external_lca_deterministic_rows_handles_empty_and_year_routing(
    project_repo: Path,
) -> None:
    lcia_method = "pb_lcia"
    version_name = "supplier_v1"
    expected_count = len(_expected_pairs(lcia_method=lcia_method))
    empty_rows, empty_paths = det_mod.load_external_lca_deterministic_rows(
        proj_base=project_repo,
        version_name=version_name,
        lcia_method=lcia_method,
        years=[2019],
        ssp_scenario_options_by_year=None,
    )
    assert empty_rows is None
    assert empty_paths == tuple()

    deterministic_dir = project_repo / "A_lca" / "external_lca" / "deterministic"
    deterministic_dir.mkdir(parents=True, exist_ok=True)
    historical_path = deterministic_dir / f"{version_name}__{lcia_method}.csv"
    scenario_path = deterministic_dir / f"{version_name}__{lcia_method}__ssp2.csv"
    unrelated_path = deterministic_dir / f"{version_name}__ignored_method.csv"
    _wide_frame(lcia_method=lcia_method, year=2019).to_csv(historical_path, index=False)
    _wide_frame(lcia_method=lcia_method, year=2020).to_csv(scenario_path, index=False)
    _wide_frame(lcia_method=lcia_method, year=2030).to_csv(unrelated_path, index=False)

    rows, selected_paths = det_mod.load_external_lca_deterministic_rows(
        proj_base=project_repo,
        version_name=version_name,
        lcia_method=lcia_method,
        years=[2019, 2020],
        ssp_scenario_options_by_year={2020: ["ssp2"]},
    )

    assert selected_paths == (historical_path, scenario_path)
    assert rows is not None
    assert rows["year"].tolist() == ["2019"] * expected_count + ["2020"] * expected_count
    assert rows[EXT_LCA_SSP_SCENARIO_COLUMN].isna().tolist() == (
        [True] * expected_count + [False] * expected_count
    )
    assert rows.loc[
        rows[EXT_LCA_SSP_SCENARIO_COLUMN].notna(),
        EXT_LCA_SSP_SCENARIO_COLUMN,
    ].tolist() == (["SSP2"] * expected_count)
    assert set(rows["impact"]) == {
        impact for impact, _unit in _expected_pairs(lcia_method=lcia_method)
    }
    assert unrelated_path not in selected_paths

    historical_only_rows, historical_only_paths = det_mod.load_external_lca_deterministic_rows(
        proj_base=project_repo,
        version_name=version_name,
        lcia_method=lcia_method,
        years=[2019],
        ssp_scenario_options_by_year={2020: ["ssp2"]},
    )
    assert historical_only_rows is not None
    assert historical_only_paths == (historical_path,)
    assert historical_only_rows["year"].tolist() == ["2019"] * expected_count

    scenario_version = "supplier_scenarios"
    scenario_paths = (
        deterministic_dir / f"{scenario_version}__{lcia_method}__ssp1.csv",
        deterministic_dir / f"{scenario_version}__{lcia_method}__ssp2.csv",
    )
    for path in scenario_paths:
        _wide_frame(lcia_method=lcia_method, year=2030).to_csv(path, index=False)
    scenario_rows, selected_scenario_paths = det_mod.load_external_lca_deterministic_rows(
        proj_base=project_repo,
        version_name=scenario_version,
        lcia_method=lcia_method,
        years=[2030],
        ssp_scenario_options_by_year=None,
    )
    assert scenario_rows is not None
    assert selected_scenario_paths == scenario_paths
    assert scenario_rows[EXT_LCA_SSP_SCENARIO_COLUMN].tolist() == (
        ["SSP1"] * expected_count + ["SSP2"] * expected_count
    )

    no_year_rows, no_year_paths = det_mod.load_external_lca_deterministic_rows(
        proj_base=project_repo,
        version_name=version_name,
        lcia_method=lcia_method,
        years=[],
        ssp_scenario_options_by_year=None,
    )
    assert no_year_rows is None
    assert no_year_paths == tuple()


def test_load_external_lca_deterministic_rows_requires_matching_files(
    project_repo: Path,
) -> None:
    lcia_method = "pb_lcia"
    version_name = "supplier_v1"
    deterministic_dir = project_repo / "A_lca" / "external_lca" / "deterministic"
    deterministic_dir.mkdir(parents=True, exist_ok=True)
    (deterministic_dir / f"{version_name}__{lcia_method}.csv").write_text(
        "r_p,s_p,impact,impact_unit,2019\nFR,D,AAL,kg,1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        det_mod.load_external_lca_deterministic_rows(
            proj_base=project_repo,
            version_name=version_name,
            lcia_method=lcia_method,
            years=[2030],
            ssp_scenario_options_by_year=None,
        )


def test_external_lca_deterministic_loading_filters_unselected_specs_and_methods(
    project_repo: Path,
) -> None:
    lcia_method = "pb_lcia"
    version_name = "supplier_v1"
    other_lcia_method = "gwp100_lcia"
    deterministic_dir = project_repo / "A_lca" / "external_lca" / "deterministic"
    deterministic_dir.mkdir(parents=True, exist_ok=True)
    historical_path = deterministic_dir / f"{version_name}__{lcia_method}.csv"
    future_scenario_path = deterministic_dir / f"{version_name}__{lcia_method}__ssp2.csv"
    _wide_frame(lcia_method=lcia_method, year=2019).to_csv(historical_path, index=False)
    _wide_frame(lcia_method=lcia_method, year=2030).to_csv(future_scenario_path, index=False)
    _wide_frame(lcia_method=other_lcia_method, year=2019).to_csv(
        deterministic_dir / f"{version_name}__{other_lcia_method}.csv",
        index=False,
    )

    matched = io_mod.matching_external_lca_specs(
        directory=deterministic_dir,
        version_name=version_name,
        lcia_method=lcia_method,
    )
    assert tuple(spec.path for spec in matched) == (historical_path, future_scenario_path)

    rows, selected_paths = det_mod.load_external_lca_deterministic_rows(
        proj_base=project_repo,
        version_name=version_name,
        lcia_method=lcia_method,
        years=[2019],
        ssp_scenario_options_by_year=None,
    )

    assert rows is not None
    assert selected_paths == (historical_path,)
    assert set(rows["year"]) == {"2019"}
    assert EXT_LCA_SSP_SCENARIO_COLUMN in rows.columns
    assert bool(rows[EXT_LCA_SSP_SCENARIO_COLUMN].isna().all())

    pd.DataFrame(
        {
            "r_p": ["FR"],
            "s_p": ["D"],
            "impact": ["AAL"],
            "impact_unit": ["kg"],
            "year": [2019],
            "value": [1.0],
        }
    ).to_csv(historical_path, index=False)
    with pytest.raises(ValueError):
        det_mod.load_external_lca_deterministic_rows_from_paths(
            paths=(historical_path,),
            lcia_method=lcia_method,
            years=[2019],
        )

    _wide_frame(
        lcia_method=lcia_method,
        year=2019,
        include_scenario_metadata=True,
    ).to_csv(historical_path, index=False)
    with pytest.raises(ValueError):
        det_mod.load_external_lca_deterministic_rows(
            proj_base=project_repo,
            version_name=version_name,
            lcia_method=lcia_method,
            years=[2019],
            ssp_scenario_options_by_year=None,
        )

    parquet_version = "supplier_parquet"
    parquet_frame = _wide_frame(lcia_method=lcia_method, year=2019)
    parquet_frame["1800"] = 0.0
    parquet_path = deterministic_dir / f"{parquet_version}__{lcia_method}.parquet"
    parquet_frame.to_parquet(parquet_path, index=False)
    parquet_rows, parquet_paths = det_mod.load_external_lca_deterministic_rows(
        proj_base=project_repo,
        version_name=parquet_version,
        lcia_method=lcia_method,
        years=[2019],
        ssp_scenario_options_by_year=None,
    )
    assert parquet_rows is not None
    assert parquet_paths == (parquet_path,)
    assert set(parquet_rows["year"]) == {"2019"}

    pickle_version = "supplier_pickle"
    pickle_frame = _wide_frame(lcia_method=lcia_method, year=2019)
    pickle_frame["all_empty"] = pd.NA
    pickle_path = deterministic_dir / f"{pickle_version}__{lcia_method}.pickle"
    pickle_frame.to_pickle(pickle_path)
    pickle_rows, pickle_paths = det_mod.load_external_lca_deterministic_rows(
        proj_base=project_repo,
        version_name=pickle_version,
        lcia_method=lcia_method,
        years=[2019],
        ssp_scenario_options_by_year=None,
    )
    assert pickle_rows is not None
    assert pickle_paths == (pickle_path,)
    assert set(pickle_rows["year"]) == {"2019"}

    missing_values_path = deterministic_dir / f"supplier_missing__{lcia_method}.csv"
    missing_values = _wide_frame(lcia_method=lcia_method, year=2019)
    missing_values.loc[0, "2019"] = pd.NA
    missing_values.to_csv(missing_values_path, index=False)
    with pytest.raises(ValueError):
        det_mod.load_external_lca_deterministic_rows_from_paths(
            paths=(missing_values_path,),
            lcia_method=lcia_method,
            years=[2019],
        )
