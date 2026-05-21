from pathlib import Path
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from pyaesa import uncertainty_asr
from pyaesa.ar6_cc.deterministic.request.contracts import CC_FLOW_POSITIVE
from pyaesa.acc.uncertainty.sources.source_keys import (
    AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE,
    ASOCC_PROJECTION_SOURCE,
    ASOCC_REFERENCE_YEAR_SOURCE,
)
from pyaesa.asr.figures.dynamic_global_ar6 import (
    uncertainty_global_ar6_rows_from_source,
    uncertainty_global_ar6_source,
)
from pyaesa.asr.figures.common import VALUE_ARRAY_COLUMN
from pyaesa.asr.uncertainty.evaluation.alignment import build_asr_alignment
from pyaesa.asr.uncertainty.evaluation.planning import build_asr_uncertainty_plan
from pyaesa.asr.uncertainty.evaluation.scenario_groups import (
    scenario_identity_groups_from_excluded_columns,
)
from pyaesa.asr.uncertainty.evaluation.summary import (
    ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN,
    ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
    ASR_CUMULATIVE_VALUE_METRIC,
    ASR_FREQUENCY_VALUE_COLUMN,
    ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
    ASR_SUMMARY_SCOPE_COLUMN,
    ASR_SUMMARY_SCOPE_INTER_METHOD,
    ASR_SUMMARY_SCOPE_PER_METHOD,
    ASR_VALUE_METRIC,
)
from pyaesa.asr.uncertainty.figures.component_data import (
    _component_cumulative_value_rows,
    _lca_component_rows,
    component_scope_rows,
    cumulative_component_entries,
)
from pyaesa.asr.uncertainty.figures.render import (
    _collapsed_inter_method_value_rows,
    _uncertainty_jobs,
)
from pyaesa.asr.uncertainty.figures.row_reader import collapsed_value_rows, read_figure_tables
from pyaesa.asr.uncertainty.figures.scope_planner import FigureContext, build_figure_context
from pyaesa.asr.uncertainty.sources.lca_inputs import _external_lca_run_value_provider
from pyaesa.asr.uncertainty.sources.lca_inputs import _external_lca_unit_value_provider
from pyaesa.asr.uncertainty.sources.lca_inputs import _public_lca_run_value_provider
from pyaesa.asr.uncertainty.sources.lca_inputs import lca_values_for_runs
from pyaesa.asr.uncertainty.sources.lca_inputs import resolve_lca_uncertainty_component_input
from pyaesa.asr.uncertainty.sources.config import split_asr_uncertainty_config
from pyaesa.asr.uncertainty.runtime.models import LCAUncertaintyInput
from pyaesa.asr.uncertainty.runtime.checkpoints import run_asr_checkpoints
from pyaesa.asr.uncertainty.io.run_outputs import write_asr_run_outputs
from pyaesa.asr.uncertainty.io.manifest_payloads import build_asr_manifest_context
from pyaesa.asr.uncertainty.io.source_methods import build_asr_source_methods
from pyaesa.asr.uncertainty.runtime.models import ASRUncertaintyPlan, ASRUncertaintyRunPaths
from pyaesa.asr.uncertainty.sobol.runner import (
    ASRSobolEvaluationContext,
    _lca_values_for_units,
)
from pyaesa.download.ar6.utils.config import GROSS_ALT_KYOTO_WO_AFOLU
from pyaesa.external_inputs.lca.monte_carlo import (
    ExternalLCAMonteCarloSource,
    _materialize_matrix,
    _normalize_rows,
    external_lca_values_for_runs,
    external_lca_values_for_units,
    load_external_lca_monte_carlo_source,
    load_external_lca_monte_carlo_source_from_path,
)
from pyaesa.external_inputs.lca.monte_carlo_stream import load_external_lca_long_matrix_source
from pyaesa.shared.figures.dynamic_ar6 import (
    AR6_CATEGORY_SCOPE_COLUMN,
    MODEL_SCENARIO_PAIR_COUNT_COLUMN,
)
from pyaesa.shared.uncertainty_assessment.run_state.manifest import build_manifest, read_manifest
from pyaesa.shared.uncertainty_assessment.request.core import normalize_uncertainty_request
from pyaesa.external_inputs.lca.paths import (
    external_lca_deterministic_dir,
    external_lca_monte_carlo_dir,
)
from pyaesa.io_lca.data.paths import main_results_path, resolve_io_lca_paths
from pyaesa.shared.runtime.reporting.phase import NullPhasePrinter
from pyaesa.shared.runtime.reporting.run_progress import monte_carlo_run_progress
from pyaesa.workspace_initialisation.workspace import clear_default_repo_root, set_default_repo_root
from pyaesa.shared.runtime.scenario.columns import (
    ASOCC_SSP_SCENARIO_COLUMN,
    ASOCC_TIME_ROUTE_PUBLIC_COLUMN,
)
from pyaesa.shared.uncertainty_assessment.io.tables import (
    CompactRunMatrixWriter,
    SparseRunRows,
    SparseRunRowsWriter,
    read_uncertainty_table,
    write_uncertainty_table,
)
from tests.package.helpers.asr_dummy_repo import (
    prepare_dynamic_asr_io_lca_repo,
    prepare_static_asr_pb_lcia_repo,
    prepare_static_asr_external_lca_repo,
    prepare_static_asr_io_lca_repo,
)


def _static_asr_kwargs(*, project_name: str) -> dict:
    return {
        "project_name": project_name,
        "years": [2005],
        "lcia_method": "gwp100_lcia",
        "fu_code": "L2.a.a",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)", "AR(E^{CBA_FD})"],
            "l1_reg_aggreg": "pre",
            "projection_mode": "historical_reuse",
            "reg_window": [2005, 2006],
            "l2_reuse_years": [2005, 2006],
            "include_lcia_based_allocation_methods": False,
        },
        "base_cc_args": {"static": {"exclude_max_cc": True}},
        "lca_args": {
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        "uncertainty_config": {
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 1},
                "convergence": {"active": False},
            },
            "asocc_uncertainty_sources": {"inter_method_uncertainty": {}},
        },
        "output_format": "csv_compact",
        "subfigures": False,
    }


def _inactive_denominator_uncertainty_sources() -> dict:
    return {
        "asocc_uncertainty_sources": {
            "projection_uncertainty": {"active": False},
            "reference_year_uncertainty": {"active": False},
            "inter_method_uncertainty": {"active": False},
        },
        "ar6_cc_uncertainty_sources": {
            "dynamic_ar6_cc_uncertainty": {"active": False},
        },
    }


def test_uncertainty_asr_rejects_sobol_years_outside_studied_years() -> None:
    kwargs = _static_asr_kwargs(project_name="asr_bad_sobol_year")
    with pytest.raises(ValueError):
        uncertainty_asr(
            **kwargs,
            sobol_parameters={"sobol_years": [2006]},
            refresh=True,
        )


def test_uncertainty_asr_rejects_polar_years_outside_studied_years() -> None:
    kwargs = _static_asr_kwargs(project_name="asr_bad_polar_year")
    with pytest.raises(ValueError):
        uncertainty_asr(
            **kwargs,
            figure_options={"polar": {"polar_years": [2006]}},
            refresh=True,
        )


def test_asr_summary_groups_drop_active_dynamic_cc_category_axis(tmp_path: Path) -> None:
    acc_identity_path = tmp_path / "acc_identity.csv"
    source_methods_path = tmp_path / "source_methods.csv"
    source_methods_path.write_text("source_name\ndynamic_ar6_cc_uncertainty\n", encoding="utf-8")
    write_uncertainty_table(
        path=acc_identity_path,
        output_format="csv_compact",
        frame=pd.DataFrame(
            {
                "public_row_id": [0, 1],
                "cc_type": ["dynamic_ar6", "dynamic_ar6"],
                "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
                "cc_category": ["C1", "C2"],
                "ar6_cc_ssp_scenario": ["SSP1", "SSP1"],
                "year": [2030, 2030],
                "impact": ["climate_child", "climate_child"],
                "impact_unit": ["kg", "kg"],
                "cc_model": ["model_a", "model_b"],
                "cc_scenario": ["scenario_a", "scenario_b"],
            }
        ),
    )
    acc_manifest = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("ar6_cc::dynamic_ar6_cc_uncertainty",),
        source_parameters={"dynamic_cc_category_uncertainty": True},
        artifacts={
            "public_row_identity": str(acc_identity_path),
            "acc_runs": str(tmp_path / "acc_runs.csv"),
            "summary_stats_runs": str(tmp_path / "acc_summary.csv"),
            "results_readme": str(tmp_path / "README.txt"),
            "source_methods": str(source_methods_path),
            "scope_manifest": str(tmp_path / "logs" / "scope_manifest.json"),
            "public_output": {"acc_runs": {"layout": "compact_run_matrix"}},
        },
    )
    lca_input = LCAUncertaintyInput(
        identity=pd.DataFrame(
            {
                "public_row_id": [0],
                "lcia_method": ["gwp100_lcia"],
                "impact": ["climate_child"],
                "year": [2030],
                "impact_unit": ["kg"],
            }
        ),
        fixed_values=pd.Series([1.0]).to_numpy(dtype=float),
        manifest=None,
        external_inputs=(),
        source_method_rows=pd.DataFrame(),
        active_sources=(),
        lca_type="io_lca",
    )

    plan = build_asr_uncertainty_plan(
        acc_manifest=acc_manifest,
        lca_input=lca_input,
        output_format="csv_compact",
    )
    plan = replace(plan, source_method_rows=build_asr_source_methods(plan=plan))
    assert "cc_category" in plan.identity.columns
    assert "cc_category" not in plan.summary_identity.columns
    assert "cc_category" not in plan.cumulative_identity.columns
    assert "cc_model" not in plan.cumulative_identity.columns
    assert "cc_scenario" not in plan.cumulative_identity.columns
    assert plan.summary_public_row_groups == (("0", "1"),)
    assert plan.cumulative_public_row_groups == (("0", "1"),)
    assert plan.source_method_rows["source_component"].tolist() == ["acc", "asr"]
    assert plan.summary_identity["asr_metric"].tolist() == [
        "asr",
        "frequency_of_no_transgression",
    ]
    assert "year" not in plan.cumulative_identity.columns
    assert plan.cumulative_summary_identity["asr_metric"].tolist() == [
        ASR_CUMULATIVE_VALUE_METRIC,
        ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
    ]


def test_asr_cumulative_identity_keeps_method_axis_for_inter_method_figures(
    tmp_path: Path,
) -> None:
    acc_identity_path = tmp_path / "acc_identity.csv"
    write_uncertainty_table(
        path=acc_identity_path,
        output_format="csv_compact",
        frame=pd.DataFrame(
            {
                "public_row_id": [0, 1, 2, 3],
                "cc_type": ["dynamic_ar6", "dynamic_ar6", "dynamic_ar6", "dynamic_ar6"],
                "lcia_method": ["gwp100_lcia"] * 4,
                "impact": ["GWP_100"] * 4,
                "impact_unit": ["kg CO2-eq"] * 4,
                "year": [2024, 2030, 2024, 2030],
                "asocc_ssp_scenario": ["", "SSP1", "", "SSP1"],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [
                    "historical",
                    "l2_reuse_year",
                    "historical",
                    "l2_reuse_year",
                ],
                "l1_l2_method": ["A", "A", "B", "B"],
                "l1_method": ["A1", "A1", "B1", "B1"],
                "l2_method": ["A2", "A2", "B2", "B2"],
            }
        ),
    )
    acc_manifest = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("asocc::inter_method_uncertainty",),
        artifacts={
            "scope_manifest": str(tmp_path / "acc" / "logs" / "scope_manifest.json"),
            "public_row_identity": str(acc_identity_path),
            "acc_runs": str(tmp_path / "acc_runs.csv"),
            "summary_stats_runs": str(tmp_path / "acc_summary.csv"),
            "results_readme": str(tmp_path / "README.txt"),
            "source_methods": str(tmp_path / "source_methods.csv"),
            "public_output": {"acc_runs": {"layout": "compact_run_matrix"}},
        },
    )
    lca_input = LCAUncertaintyInput(
        identity=pd.DataFrame(
            {
                "public_row_id": [0, 1],
                "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
                "impact": ["GWP_100", "GWP_100"],
                "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
                "year": [2024, 2030],
            }
        ),
        fixed_values=pd.Series([10.0, 12.0]).to_numpy(dtype=float),
        manifest=None,
        external_inputs=(),
        source_method_rows=pd.DataFrame(),
        active_sources=(),
        lca_type="io_lca",
    )

    plan = build_asr_uncertainty_plan(
        acc_manifest=acc_manifest,
        lca_input=lca_input,
        output_format="csv_compact",
    )

    assert plan.cumulative_identity["l1_l2_method"].tolist() == ["A", "B"]
    assert plan.cumulative_identity["asocc_ssp_scenario"].tolist() == ["SSP1", "SSP1"]
    assert ASOCC_TIME_ROUTE_PUBLIC_COLUMN not in plan.cumulative_identity.columns
    assert plan.cumulative_public_row_groups == (("0", "1"), ("2", "3"))
    assert plan.cumulative_summary_public_row_groups == (("0",), ("1",), ("0", "1"))
    assert plan.cumulative_summary_identity[ASR_SUMMARY_SCOPE_COLUMN].tolist() == [
        ASR_SUMMARY_SCOPE_PER_METHOD,
        ASR_SUMMARY_SCOPE_PER_METHOD,
        ASR_SUMMARY_SCOPE_INTER_METHOD,
        ASR_SUMMARY_SCOPE_PER_METHOD,
        ASR_SUMMARY_SCOPE_PER_METHOD,
        ASR_SUMMARY_SCOPE_INTER_METHOD,
    ]


def test_asr_inter_method_summaries_use_visible_external_lca_scenario_scope(
    tmp_path: Path,
) -> None:
    acc_identity_path = tmp_path / "acc_identity.csv"
    write_uncertainty_table(
        path=acc_identity_path,
        output_format="csv_compact",
        frame=pd.DataFrame(
            {
                "public_row_id": [0, 1, 2, 3],
                "cc_type": ["dynamic_ar6", "dynamic_ar6", "dynamic_ar6", "dynamic_ar6"],
                "lcia_method": ["gwp100_lcia"] * 4,
                "impact": ["GWP_100"] * 4,
                "impact_unit": ["kg CO2-eq"] * 4,
                "year": [2024, 2030, 2024, 2030],
                "ar6_cc_ssp_scenario": ["SSP2"] * 4,
                "asocc_ssp_scenario": ["", "SSP2", "", ""],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [
                    "historical",
                    "regression_proj",
                    "historical",
                    "historical",
                ],
                "l1_l2_method": ["A", "A", "B", "B"],
                "l1_method": ["A1", "A1", "B1", "B1"],
                "l2_method": ["A2", "A2", "B2", "B2"],
            }
        ),
    )
    acc_manifest = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("asocc::inter_method_uncertainty",),
        artifacts={
            "scope_manifest": str(tmp_path / "acc" / "logs" / "scope_manifest.json"),
            "public_row_identity": str(acc_identity_path),
            "acc_runs": str(tmp_path / "acc_runs.csv"),
            "summary_stats_runs": str(tmp_path / "acc_summary.csv"),
            "results_readme": str(tmp_path / "README.txt"),
            "source_methods": str(tmp_path / "source_methods.csv"),
            "public_output": {"acc_runs": {"layout": "compact_run_matrix"}},
        },
    )
    lca_input = LCAUncertaintyInput(
        identity=pd.DataFrame(
            {
                "public_row_id": [0, 1],
                "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
                "impact": ["GWP_100", "GWP_100"],
                "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
                "year": [2024, 2030],
                "lca_ssp_scenario": ["", "SSP2"],
            }
        ),
        fixed_values=pd.Series([10.0, 12.0]).to_numpy(dtype=float),
        manifest=None,
        external_inputs=(),
        source_method_rows=pd.DataFrame(),
        active_sources=(),
        lca_type="external",
    )

    plan = build_asr_uncertainty_plan(
        acc_manifest=acc_manifest,
        lca_input=lca_input,
        output_format="csv_compact",
    )

    inter_asr = plan.summary_identity.loc[
        plan.summary_identity[ASR_SUMMARY_SCOPE_COLUMN].eq(ASR_SUMMARY_SCOPE_INTER_METHOD)
        & plan.summary_identity["asr_metric"].eq(ASR_VALUE_METRIC)
    ]
    cumulative_inter_asr = plan.cumulative_summary_identity.loc[
        plan.cumulative_summary_identity[ASR_SUMMARY_SCOPE_COLUMN].eq(
            ASR_SUMMARY_SCOPE_INTER_METHOD
        )
        & plan.cumulative_summary_identity["asr_metric"].eq(ASR_CUMULATIVE_VALUE_METRIC)
    ]

    assert pd.Series(pd.to_numeric(inter_asr["year"], errors="raise")).astype(int).tolist() == [
        2024,
        2030,
    ]
    assert inter_asr[ASOCC_TIME_ROUTE_PUBLIC_COLUMN].tolist() == [
        "historical",
        "regression_proj",
    ]
    assert plan.summary_public_row_groups[-2:] == (("0", "2"), ("1", "3"))
    assert plan.cumulative_public_row_groups == (("0", "1"), ("2", "3"))
    assert cumulative_inter_asr["asocc_ssp_scenario"].tolist() == ["SSP2"]
    assert cumulative_inter_asr["lca_ssp_scenario"].tolist() == ["SSP2"]
    assert plan.cumulative_summary_public_row_groups == (("0",), ("1",), ("0", "1"))


def test_asr_alignment_repeats_scenario_invariant_acc_for_external_lca_ssp() -> None:
    acc_identity = pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "cc_type": ["static", "static"],
            "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
            "impact": ["GWP_100", "GWP_100"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
            "year": [2023, 2024],
            "s_p": ["Electricity", "Electricity"],
            "r_c": ["FR", "FR"],
        }
    )
    lca_identity = pd.DataFrame(
        {
            "public_row_id": [0, 1, 2, 3],
            "lcia_method": ["gwp100_lcia"] * 4,
            "impact": ["GWP_100"] * 4,
            "impact_unit": ["kg CO2-eq"] * 4,
            "year": [2023, 2023, 2024, 2024],
            "lca_ssp_scenario": ["SSP1", "SSP2", "SSP1", "SSP2"],
            "s_p": ["Electricity"] * 4,
            "r_c": ["FR"] * 4,
        }
    )

    alignment = build_asr_alignment(
        acc_identity=acc_identity,
        lca_identity=lca_identity,
        lca_type="external",
    )

    assert alignment.acc_positions.tolist() == [0, 0, 1, 1]
    assert alignment.lca_positions.tolist() == [0, 1, 2, 3]
    assert alignment.identity["lca_ssp_scenario"].tolist() == ["SSP1", "SSP2", "SSP1", "SSP2"]


def test_asr_scenario_groups_resolve_invariant_rows_by_target_specificity() -> None:
    identity = pd.DataFrame(
        {
            "public_row_id": [0, 1, 2],
            "lca_ssp_scenario": ["", "SSP1", "SSP2"],
        }
    )

    grouped, public_row_groups = scenario_identity_groups_from_excluded_columns(
        identity=identity,
        excluded_columns=set(),
    )

    assert grouped["lca_ssp_scenario"].tolist() == ["SSP1", "SSP2"]
    assert public_row_groups == (("1",), ("2",))

    invariant_only = pd.DataFrame({"public_row_id": [0, 1], "lca_ssp_scenario": ["", None]})
    grouped_invariant, invariant_groups = scenario_identity_groups_from_excluded_columns(
        identity=invariant_only,
        excluded_columns=set(),
    )

    assert pd.isna(grouped_invariant.loc[0, "lca_ssp_scenario"])
    assert invariant_groups == (("0", "1"),)


def test_asr_cumulative_identity_repeats_invariant_rows_into_ssp_periods(
    tmp_path: Path,
) -> None:
    acc_identity_path = tmp_path / "acc_identity.csv"
    write_uncertainty_table(
        path=acc_identity_path,
        output_format="csv_compact",
        frame=pd.DataFrame(
            {
                "public_row_id": [0, 1, 2],
                "cc_type": ["dynamic_ar6", "dynamic_ar6", "dynamic_ar6"],
                "lcia_method": ["gwp100_lcia", "gwp100_lcia", "gwp100_lcia"],
                "impact": ["GWP_100", "GWP_100", "GWP_100"],
                "impact_unit": ["kg CO2-eq", "kg CO2-eq", "kg CO2-eq"],
                "year": [2024, 2030, 2030],
                "asocc_ssp_scenario": ["", "SSP1", "SSP2"],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: [
                    "historical",
                    "l2_reuse_year",
                    "l2_reuse_year",
                ],
                "reference_year": [2019, 2019, 2019],
                "l2_reuse_year": [pd.NA, 2022, 2022],
            }
        ),
    )
    acc_manifest = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(
            "asocc::inter_method_uncertainty",
            "asocc::projection_uncertainty",
            "asocc::reference_year_uncertainty",
        ),
        artifacts={
            "scope_manifest": str(tmp_path / "acc" / "logs" / "scope_manifest.json"),
            "public_row_identity": str(acc_identity_path),
            "acc_runs": str(tmp_path / "acc_runs.csv"),
            "summary_stats_runs": str(tmp_path / "acc_summary.csv"),
            "results_readme": str(tmp_path / "README.txt"),
            "source_methods": str(tmp_path / "source_methods.csv"),
            "public_output": {"acc_runs": {"layout": "compact_run_matrix"}},
        },
    )
    lca_input = LCAUncertaintyInput(
        identity=pd.DataFrame(
            {
                "public_row_id": [0, 1],
                "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
                "impact": ["GWP_100", "GWP_100"],
                "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
                "year": [2024, 2030],
            }
        ),
        fixed_values=pd.Series([10.0, 12.0]).to_numpy(dtype=float),
        manifest=None,
        external_inputs=(),
        source_method_rows=pd.DataFrame(),
        active_sources=(),
        lca_type="io_lca",
    )

    plan = build_asr_uncertainty_plan(
        acc_manifest=acc_manifest,
        lca_input=lca_input,
        output_format="csv_compact",
    )

    assert "year" not in plan.cumulative_identity.columns
    assert ASOCC_TIME_ROUTE_PUBLIC_COLUMN not in plan.cumulative_identity.columns
    assert plan.cumulative_identity["asocc_ssp_scenario"].tolist() == ["SSP1", "SSP2"]
    assert plan.cumulative_public_row_groups == (("0", "1"), ("0", "2"))

    invariant_identity_path = tmp_path / "acc_identity_invariant.csv"
    write_uncertainty_table(
        path=invariant_identity_path,
        output_format="csv_compact",
        frame=pd.DataFrame(
            {
                "public_row_id": [0, 1],
                "cc_type": ["dynamic_ar6", "dynamic_ar6"],
                "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
                "impact": ["GWP_100", "GWP_100"],
                "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
                "year": [2024, 2030],
                "asocc_ssp_scenario": ["", ""],
                ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical", "l2_reuse_year"],
                "reference_year": [2019, 2019],
                "l2_reuse_year": [pd.NA, 2022],
            }
        ),
    )
    invariant_manifest = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(
            "asocc::inter_method_uncertainty",
            "asocc::projection_uncertainty",
            "asocc::reference_year_uncertainty",
        ),
        artifacts={
            "scope_manifest": str(tmp_path / "acc_invariant" / "logs" / "scope_manifest.json"),
            "public_row_identity": str(invariant_identity_path),
            "acc_runs": str(tmp_path / "acc_invariant_runs.csv"),
            "summary_stats_runs": str(tmp_path / "acc_invariant_summary.csv"),
            "results_readme": str(tmp_path / "README_invariant.txt"),
            "source_methods": str(tmp_path / "source_methods_invariant.csv"),
            "public_output": {"acc_runs": {"layout": "compact_run_matrix"}},
        },
    )

    invariant_plan = build_asr_uncertainty_plan(
        acc_manifest=invariant_manifest,
        lca_input=lca_input,
        output_format="csv_compact",
    )

    assert invariant_plan.cumulative_public_row_groups == (("0", "1"),)
    assert bool(invariant_plan.cumulative_identity["asocc_ssp_scenario"].isna().all())


def test_asr_dynamic_figure_component_rows_collapse_sampled_axes(tmp_path: Path) -> None:
    sources = (
        ASOCC_REFERENCE_YEAR_SOURCE,
        ASOCC_PROJECTION_SOURCE,
        AR6_DYNAMIC_CC_UNCERTAINTY_SOURCE,
    )
    paths = ASRUncertaintyRunPaths(
        run_root=tmp_path / "asr_run",
        public_row_identity=tmp_path / "asr_run" / "results" / "identity.csv",
        public_runs=tmp_path / "asr_run" / "results" / "runs.csv",
        summary_stats_runs=tmp_path / "asr_run" / "results" / "summary.csv",
        cumulative_row_identity=tmp_path / "asr_run" / "results" / "cumulative_identity.csv",
        cumulative_runs=tmp_path / "asr_run" / "results" / "cumulative_runs.csv",
        cumulative_summary_stats_runs=tmp_path / "asr_run" / "results" / "cumulative_summary.csv",
        results_readme=tmp_path / "asr_run" / "results" / "README.txt",
        source_methods=tmp_path / "asr_run" / "logs" / "source_methods.csv",
        sobol_indices=tmp_path / "asr_run" / "results" / "sobol" / "indices.csv",
        sobol_source_summary=tmp_path / "asr_run" / "results" / "sobol" / "summary.csv",
        sobol_readme=tmp_path / "asr_run" / "results" / "sobol" / "README.txt",
        scope_manifest=tmp_path / "asr_run" / "logs" / "scope_manifest.json",
    )
    context = FigureContext(
        manifest=build_manifest(
            family="asr",
            mode="fixed",
            output_format="csv_compact",
            active_sources=sources,
        ),
        paths=paths,
        figures_root=tmp_path / "figures",
        requested_years=(2024, 2030),
        requested_asocc_ssps=("SSP2", "SSP3"),
        fu_code="L2.a.a",
        output_format="csv_compact",
        figure_output_format="png",
        figure_dpi=10,
        per_method=True,
        multi_method=True,
        inter_method=True,
        active_sources=sources,
        run_layout="compact_run_matrix",
        dynamic_category_uncertainty_active=True,
        polar_years=(),
        polar_style="violin",
    )
    rows = pd.DataFrame(
        {
            "public_row_id": [0, 1, 2],
            "year": [2024, 2030, 2030],
            VALUE_ARRAY_COLUMN: [
                np.array([1.0, 2.0]),
                np.array([3.0, 4.0]),
                np.array([5.0, 6.0]),
            ],
            "cc_type": ["dynamic_ar6", "dynamic_ar6", "dynamic_ar6"],
            "lcia_method": ["gwp100_lcia", "gwp100_lcia", "gwp100_lcia"],
            "impact": ["GWP_100", "GWP_100", "GWP_100"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq", "kg CO2-eq"],
            "reference_year": [2019, 2019, 2019],
            "l2_reuse_year": [pd.NA, 2022, 2022],
            "cc_model": ["model_h", "model_a", "model_b"],
            "cc_scenario": ["scenario_h", "scenario_a", "scenario_b"],
            "cc_category": ["C1", "C1", "C2"],
            ASOCC_SSP_SCENARIO_COLUMN: ["", "SSP2", "SSP3"],
            "ar6_cc_ssp_scenario": ["SSP2", "SSP2", "SSP3"],
            ASOCC_TIME_ROUTE_PUBLIC_COLUMN: ["historical", "l2_reuse_year", "l2_reuse_year"],
            "__method": ["A", "A", "A"],
        }
    )

    collapsed = collapsed_value_rows(rows=rows, context=context, include_method_axis=False)
    assert sorted(collapsed[ASOCC_SSP_SCENARIO_COLUMN].dropna().astype(str)) == ["SSP2", "SSP3"]
    mixed_collapsed = collapsed_value_rows(
        rows=pd.concat([rows, rows.assign(cc_type="static")], ignore_index=True),
        context=context,
        include_method_axis=False,
    )
    assert MODEL_SCENARIO_PAIR_COUNT_COLUMN in mixed_collapsed.columns
    static_multi_ssp_rows = rows.drop(
        columns=["cc_model", "cc_scenario", "cc_category", "ar6_cc_ssp_scenario"]
    )
    static_multi_ssp_rows = static_multi_ssp_rows.assign(cc_type="static", year=2030)
    static_multi_ssp = _collapsed_inter_method_value_rows(
        rows=static_multi_ssp_rows,
        context=context,
    )
    assert static_multi_ssp[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2", "SSP3"]
    assert [len(values) for values in static_multi_ssp[VALUE_ARRAY_COLUMN].tolist()] == [4, 4]
    static_single_ssp_context = replace(
        context,
        requested_asocc_ssps=("SSP2",),
        dynamic_category_uncertainty_active=False,
    )
    static_single_ssp = _collapsed_inter_method_value_rows(
        rows=static_multi_ssp_rows.iloc[[0, 1]].copy(),
        context=static_single_ssp_context,
    )
    assert len(static_single_ssp) == 1
    assert static_single_ssp[ASOCC_SSP_SCENARIO_COLUMN].tolist() == ["SSP2"]
    repo_root = tmp_path / "repo"
    cc_dir = repo_root / "data_raw" / "carrying_capacities"
    cc_dir.mkdir(parents=True)
    (cc_dir / "gwp100_lcia_cc_steady_state.csv").write_text(
        "impact_full_name,impact,impact_unit,min_cc,max_cc\n"
        "Climate change,GWP_100,kg CO2-eq,1.0,2.0\n"
        "Climate neutral,GWP_NEUTRAL,kg CO2-eq,1.0,1.0\n",
        encoding="utf-8",
    )
    threshold_rows = pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "year": [2024, 2024],
            VALUE_ARRAY_COLUMN: [np.array([1.0, 2.0]), np.array([2.0, 3.0])],
            "cc_type": ["static", "static"],
            "lcia_method": ["gwp100_lcia", "gwp100_lcia"],
            "impact": ["GWP_100", "GWP_NEUTRAL"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
            "cc_bound": ["max_cc", "max_cc"],
            "cc_category": ["C1", "C1"],
        }
    )
    set_default_repo_root(repo_root)
    try:
        threshold_collapsed = collapsed_value_rows(
            rows=threshold_rows,
            context=context,
            include_method_axis=False,
        )
    finally:
        clear_default_repo_root()
    assert threshold_collapsed["__asr_max_threshold"].dropna().tolist() == [2.0]
    assert threshold_collapsed[AR6_CATEGORY_SCOPE_COLUMN].dropna().tolist() == ["C1"]

    cumulative = _component_cumulative_value_rows(
        rows=rows,
        context=context,
        include_method_axis=True,
    ).rename(columns={VALUE_ARRAY_COLUMN: "__component_cumulative_values"})
    inter_cumulative = _component_cumulative_value_rows(
        rows=rows,
        context=context,
        include_method_axis=False,
    )
    static_context = replace(
        context,
        active_sources=(ASOCC_REFERENCE_YEAR_SOURCE, ASOCC_PROJECTION_SOURCE),
        dynamic_category_uncertainty_active=False,
    )
    static_cumulative = _component_cumulative_value_rows(
        rows=rows,
        context=static_context,
        include_method_axis=False,
    )
    assert "__method" not in inter_cumulative.columns
    assert "cc_model" in static_cumulative.columns
    scoped = component_scope_rows(
        cumulative.assign(__component="acc"),
        asr_frame=pd.DataFrame(
            {
                "cc_type": ["dynamic_ar6"],
                "lcia_method": ["gwp100_lcia"],
                "impact": ["GWP_100"],
                "cc_category": ["C1"],
                "ar6_cc_ssp_scenario": ["SSP2"],
                ASOCC_SSP_SCENARIO_COLUMN: ["SSP2"],
                "__method": ["A"],
            }
        ),
        include_method_axis=True,
    )
    assert scoped["ar6_cc_ssp_scenario"].tolist() == ["SSP2"]
    entries = cumulative_component_entries(scoped)
    assert entries[0][0] == "A"
    np.testing.assert_allclose(entries[0][1], [4.0, 6.0])


def test_asr_component_rows_load_deterministic_numerator_artifacts(
    tmp_path: Path,
    allocation_dummy_repo,
) -> None:
    repo_root = allocation_dummy_repo.repo_root
    base_args = {
        "project_name": "asr_component_deterministic",
        "years": [2020, 2021],
        "lcia_method": ["gwp100_lcia"],
        "fu_code": "L2.a.a",
        "r_p": ["FR"],
        "s_p": ["D"],
        "r_c": None,
        "r_f": None,
        "source": "exiobase_396_ixi",
        "group_reg": False,
        "group_sec": False,
        "group_version": None,
        "aggreg_indices": False,
        "base_asocc_args": {
            "method_plan": "one_step",
            "l1_methods": None,
            "one_step_methods": ["UT(FD)"],
            "two_step_methods": None,
            "l1_l2_pairs": None,
            "l1_reg_aggreg": "pre",
            "reference_years": None,
            "ssp_scenario": ["SSP2"],
            "projection_mode": "historical_reuse",
            "reg_window": None,
            "l2_reuse_years": None,
            "include_lcia_based_allocation_methods": False,
        },
    }
    run_paths = _asr_run_paths(tmp_path / "asr_component_run")
    manifest = build_manifest(
        family="asr",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        completed_runs=2,
        arguments=base_args,
        external_inputs=(
            {
                "type": "io_lca_deterministic",
                "source": "exiobase_396_ixi",
                "output_format": "csv",
            },
        ),
    )
    context = FigureContext(
        manifest=manifest,
        paths=run_paths,
        figures_root=tmp_path / "figures",
        requested_years=(2020, 2021),
        requested_asocc_ssps=("SSP2",),
        fu_code="L2.a.a",
        output_format="csv_compact",
        figure_output_format="svg",
        figure_dpi=10,
        per_method=True,
        multi_method=True,
        inter_method=True,
        active_sources=(),
        run_layout="compact_run_matrix",
        dynamic_category_uncertainty_active=False,
        polar_years=(),
        polar_style="violin",
    )
    set_default_repo_root(repo_root)
    try:
        io_paths = resolve_io_lca_paths(
            project_name="asr_component_deterministic",
            group_reg=False,
            group_sec=False,
            group_version=None,
        )
        io_path = main_results_path(
            paths=io_paths,
            source="exiobase_396_ixi",
            lcia_method="gwp100_lcia",
            extension="csv",
        )
        io_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "year": [2020, 2021],
                "impact": ["GWP_100", "GWP_100"],
                "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
                "r_p": ["FR", "FR"],
                "s_p": ["D", "D"],
                "lca_value": [2.0, 3.0],
            }
        ).to_csv(io_path, index=False)

        io_rows = _lca_component_rows(context=context)
    finally:
        clear_default_repo_root()
    assert set(io_rows["__component"]) == {"lca"}
    assert sorted(io_rows["mean"].tolist()) == [2.0, 3.0]

    external_project_base = repo_root / "asr_component_external"
    external_dir = external_lca_deterministic_dir(project_base=external_project_base)
    external_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "r_p": ["FR"],
            "s_p": ["D"],
            "impact": ["GWP_100"],
            "impact_unit": ["kg CO2-eq"],
            "2020": [4.0],
            "2021": [6.0],
        }
    ).to_csv(external_dir / "supplier_v1__gwp100_lcia.csv", index=False)
    external_args = {**base_args, "project_name": "asr_component_external"}
    set_default_repo_root(repo_root)
    try:
        lca_input = resolve_lca_uncertainty_component_input(
            proj_base=external_project_base,
            source_label="exiobase_396_ixi",
            lca_type="external",
            lca_version_name="supplier_v1",
            base_allocate_args=external_args,
            lcia_methods=["gwp100_lcia"],
            uncertainty_config={},
            output_format="csv_compact",
            refresh=False,
            phase=cast(Any, NullPhasePrinter()),
            status=cast(Any, NullPhasePrinter()),
            figures=True,
            figure_format={"format": "svg", "dpi": 1},
        ).input
    finally:
        clear_default_repo_root()
    external_manifest = build_manifest(
        family="asr",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        completed_runs=2,
        arguments=external_args,
        external_inputs=lca_input.external_inputs,
    )
    set_default_repo_root(repo_root)
    try:
        external_rows = _lca_component_rows(context=replace(context, manifest=external_manifest))
    finally:
        clear_default_repo_root()

    assert set(external_rows["__component"]) == {"lca"}
    assert sorted(external_rows["mean"].tolist()) == [4.0, 6.0]
    assert lca_input.external_inputs[0]["figures_available"] == 1

    mc_project_base = repo_root / "asr_component_external_mc"
    mc_dir = external_lca_monte_carlo_dir(project_base=mc_project_base)
    mc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_index": run_index,
                "year": year,
                "lca_ssp_scenario": "",
                "r_p": "FR",
                "s_p": "D",
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "value": value + run_index * 0.1,
            }
            for run_index in range(2)
            for year, value in [(2020, 5.0), (2021, 6.0)]
        ]
    ).to_csv(mc_dir / "supplier_v1__gwp100_lcia.csv", index=False)
    mc_args = {**base_args, "project_name": "asr_component_external_mc"}
    set_default_repo_root(repo_root)
    try:
        mc_input = resolve_lca_uncertainty_component_input(
            proj_base=mc_project_base,
            source_label="exiobase_396_ixi",
            lca_type="external",
            lca_version_name="supplier_v1",
            base_allocate_args=mc_args,
            lcia_methods=["gwp100_lcia"],
            uncertainty_config={},
            output_format="csv_compact",
            refresh=False,
            phase=cast(Any, NullPhasePrinter()),
            status=cast(Any, NullPhasePrinter()),
            figures=True,
            figure_format={"format": "svg", "dpi": 1},
        ).input
    finally:
        clear_default_repo_root()

    assert mc_input.external_inputs[0]["type"] == "external_lca_monte_carlo"
    assert mc_input.external_inputs[0]["figures_available"] == 1


def test_uncertainty_asr_static_io_lca_outputs(
    allocation_dummy_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    kwargs = _static_asr_kwargs(project_name="asr_uncertainty_static_io_lca")
    kwargs["years"] = [2005, 2006]
    kwargs["base_asocc_args"]["include_lcia_based_allocation_methods"] = True
    kwargs["base_asocc_args"]["reference_years"] = [2005]
    kwargs["lca_args"] = {
        "external_lca": {"active": False, "version_name": None},
        "io_lca": {"active": True},
    }
    kwargs["uncertainty_config"] = {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}},
        "asocc_uncertainty_sources": {"inter_method_uncertainty": {}},
        "io_lca_uncertainty_sources": {
            "lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}}
        },
    }

    manifest = uncertainty_asr(
        **kwargs,
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 1},
            "convergence": {"active": False},
        },
        figures=False,
        figure_options={"polar": {"polar_years": []}},
        refresh=True,
    ).manifest
    runtime_output = capsys.readouterr().out

    assert manifest.family == "asr"
    assert runtime_output.strip()
    assert manifest.completed_runs == 2
    assert manifest.artifacts is not None
    assert {
        item["component_inventory"]["component_name"]
        for item in manifest.deterministic_prerequisites
        if "component_inventory" in item
    } == {"acc", "io_lca"}
    assert manifest.artifacts["public_output"] is not None
    assert manifest.sobol is not None and manifest.sobol["ran"] is True
    assert manifest.sobol["mode"] == "fixed"
    identity = read_uncertainty_table(
        path=Path(manifest.artifacts["public_row_identity"]),
        output_format="csv_compact",
    )
    runs = pd.read_csv(manifest.artifacts["asr_runs"])
    summary = pd.read_csv(manifest.artifacts["summary_stats_runs"])
    source_methods = pd.read_csv(manifest.artifacts["source_methods"])
    readme_text = Path(manifest.artifacts["results_readme"]).read_text(encoding="utf-8")

    assert {"lcia_method", "impact", "impact_unit", "year", "cc_type"}.issubset(identity.columns)
    assert runs.shape[0] == 4
    assert "asr" in source_methods["source_component"].tolist()
    assert set(manifest.active_sources).issubset(set(source_methods["source_name"]))
    assert any(name.startswith("acc::") for name in manifest.active_sources)
    assert any(name.startswith("io_lca::") for name in manifest.active_sources)
    assert {"mean", "median"}.issubset(summary.columns)
    frequency = summary.loc[
        summary["asr_metric"].eq(ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC)
    ].reset_index(drop=True)
    assert ASR_FREQUENCY_VALUE_COLUMN in frequency.columns
    assert bool(frequency["mean"].isna().all())
    assert bool(frequency["median"].isna().all())
    defined_frequency = frequency[ASR_FREQUENCY_VALUE_COLUMN].dropna()
    assert bool(defined_frequency.between(0.0, 1.0).all())
    assert "cumulative_row_identity" not in manifest.artifacts
    assert "cumulative_asr_runs" not in manifest.artifacts
    assert "cumulative_summary_stats_runs" not in manifest.artifacts
    assert "cumulative_asr_runs" not in manifest.artifacts["public_output"]
    assert Path(manifest.artifacts["sobol_indices"]).exists()
    assert all(len(line) <= 100 for line in readme_text.splitlines())


def test_uncertainty_asr_static_pb_lcia_public_figures_cover_polar_and_frequency(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_pb_lcia_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        years=[2005, 2006],
        impacts=["AAL", "BI FD"],
    )

    kwargs = {
        "project_name": "asr_uncertainty_static_pb_lcia_public_figures",
        "years": [2005, 2006],
        "lcia_method": "pb_lcia",
        "fu_code": "L2.a.a",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "include_lcia_based_allocation_methods": False,
            "reference_years": [2005],
        },
        "base_cc_args": {"static": {}},
        "lca_args": {
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        "uncertainty_config": {
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 4},
                "convergence": {"active": False},
            },
            "asocc_uncertainty_sources": {"inter_method_uncertainty": {}},
        },
        "output_format": "csv_compact",
        "sobol_parameters": {"active": False},
        "figures": True,
        "figure_options": {
            "per_method": True,
            "multi_method": True,
            "inter_method": True,
            "polar": {"polar_style": "whisker", "polar_years": [2005]},
        },
        "figure_format": {"format": "svg", "dpi": 1},
        "subfigures": False,
    }
    manifest = uncertainty_asr(**kwargs, refresh=True).manifest

    assert manifest.artifacts is not None
    figure_paths = [Path(path) for path in manifest.artifacts["figure_paths"]]
    assert all(path.exists() for path in figure_paths)
    figure_names = {path.name for path in figure_paths}
    assert any(
        name.startswith("inter_method__") and name.endswith("__pb_lcia.svg")
        for name in figure_names
    )
    assert any(
        name.endswith("__pb_lcia__frequency_of_no_transgression.svg") for name in figure_names
    )
    assert any(
        name.startswith("polar_whisker_inter_method__") and "__pb_lcia__" in name
        for name in figure_names
    )
    assert any(name.startswith("multi_method__") for name in figure_names)
    assert any("per_method" in path.parts for path in figure_paths)
    no_product_manifest = uncertainty_asr(
        **{
            **kwargs,
            "figure_options": {
                "per_method": False,
                "multi_method": False,
                "inter_method": False,
                "polar": {"active": False},
            },
        },
        refresh=False,
    ).manifest
    assert no_product_manifest.artifacts is not None
    assert no_product_manifest.artifacts["figure_paths"] == []


def test_uncertainty_asr_static_pb_lcia_single_year_public_figures_cover_violin_products(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_pb_lcia_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        years=[2005],
        impacts=["AAL", "BI FD"],
    )

    kwargs = {
        "project_name": "asr_uncertainty_static_pb_lcia_single_year_public_figures",
        "years": [2005],
        "lcia_method": "pb_lcia",
        "fu_code": "L2.a.a",
        "r_p": ["FR"],
        "s_p": ["D"],
        "source": "exiobase_396_ixi",
        "base_asocc_args": {
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "include_lcia_based_allocation_methods": False,
            "reference_years": [2005],
        },
        "base_cc_args": {"static": {}},
        "lca_args": {
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        "uncertainty_config": {
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "asocc_uncertainty_sources": {"inter_method_uncertainty": {}},
            "io_lca_uncertainty_sources": {
                "lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}}
            },
        },
        "output_format": "csv_compact",
        "sobol_parameters": {"active": False},
        "figures": True,
        "figure_options": {"polar": {"polar_style": "violin"}},
        "figure_format": {"format": "svg", "dpi": 1},
        "subfigures": False,
        "refresh": True,
    }
    manifest = uncertainty_asr(**kwargs).manifest

    assert manifest.artifacts is not None
    figure_paths = [Path(path) for path in manifest.artifacts["figure_paths"]]
    assert all(path.exists() for path in figure_paths)
    figure_names = {path.name for path in figure_paths}
    assert any(name.startswith("polar_violin_inter_method__") for name in figure_names)
    assert any(
        name.startswith("polar_violin_") and not name.startswith("polar_violin_inter_method__")
        for name in figure_names
    )
    no_polar_manifest = uncertainty_asr(
        **{**kwargs, "figure_options": {"polar": {"active": False}}, "refresh": False}
    ).manifest
    no_polar_names = {Path(path).name for path in no_polar_manifest.artifacts["figure_paths"]}
    assert not any(name.startswith("polar_") for name in no_polar_names)


def test_uncertainty_asr_dynamic_io_lca_outputs_cumulative_artifacts(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
        historical_years=[2020, 2021],
        scenario_years=[2030],
    )
    explorer_path = (
        allocation_dummy_repo.repo_root
        / "data_raw"
        / "carrying_capacities"
        / "dynamic_climate_change_ar6"
        / "ar6_public_explorer.csv"
    )
    explorer = pd.read_csv(explorer_path)
    explorer.loc[explorer["model"].astype(str).eq("M1"), "Ssp_family"] = 2
    explorer.to_csv(explorer_path, index=False)

    manifest = uncertainty_asr(
        project_name="asr_uncertainty_dynamic_io_lca",
        years=range(2020, 2022),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "pairs",
            "l1_l2_pairs": ["EG(Pop)::UT(FD)", "PR(GDPcap)::UT(FD)"],
            "ssp_scenario": ["SSP2"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C1", "C2"], "ssp_scenario": ["SSP2"]},
        },
        lca_args={
            "external_lca": {"active": False, "version_name": None},
            "io_lca": {"active": True},
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
            "ar6_cc_uncertainty_sources": {
                "dynamic_ar6_cc_uncertainty": {"category_uncertainty": True}
            },
            "asocc_uncertainty_sources": {"inter_method_uncertainty": {}},
            "io_lca_uncertainty_sources": {
                "lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}}
            },
        },
        output_format="csv_compact",
        sobol_parameters={"active": False},
        figures=False,
        subfigures=False,
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    assert manifest.artifacts["public_output"] is not None
    assert "cumulative_row_identity" in manifest.artifacts
    assert "cumulative_asr_runs" in manifest.artifacts
    assert "cumulative_summary_stats_runs" in manifest.artifacts
    assert (
        manifest.artifacts["public_output"]["cumulative_asr_runs"]["layout"] == "compact_run_matrix"
    )
    identity = read_uncertainty_table(
        path=Path(manifest.artifacts["public_row_identity"]),
        output_format="csv_compact",
    )
    summary = pd.read_csv(manifest.artifacts["summary_stats_runs"])
    cumulative_identity = read_uncertainty_table(
        path=Path(manifest.artifacts["cumulative_row_identity"]),
        output_format="csv_compact",
    )
    cumulative_summary = pd.read_csv(manifest.artifacts["cumulative_summary_stats_runs"])
    for frame in (identity, summary, cumulative_identity, cumulative_summary):
        assert {"cc_flow", "cc_variable"}.issubset(frame.columns)
        assert set(frame["cc_flow"]) == {CC_FLOW_POSITIVE}
        assert set(frame["cc_variable"]) == {GROSS_ALT_KYOTO_WO_AFOLU}
    assert "year" not in cumulative_identity.columns
    assert ASOCC_TIME_ROUTE_PUBLIC_COLUMN not in cumulative_identity.columns
    assert set(cumulative_summary["asr_metric"]) == {
        ASR_CUMULATIVE_VALUE_METRIC,
        ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
    }
    cumulative_frequency = cumulative_summary.loc[
        cumulative_summary["asr_metric"].eq(ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC)
    ]
    assert ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN in cumulative_frequency.columns
    assert bool(cumulative_frequency["mean"].isna().all())
    defined_frequency = cumulative_frequency[ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN].dropna()
    assert bool(defined_frequency.between(0.0, 1.0).all())
    figure_context = build_figure_context(
        manifest=manifest,
        paths=_asr_paths_from_manifest(manifest),
        figure_options={"multi_method": False, "polar": {"polar_years": []}},
        figure_format={"format": "png", "dpi": 10},
    )
    figure_tables = read_figure_tables(context=figure_context)
    single_year_context = build_figure_context(
        manifest=replace(
            manifest,
            arguments={**dict(manifest.arguments or {}), "years": [2020]},
        ),
        paths=_asr_paths_from_manifest(manifest),
        figure_options={"multi_method": False, "polar": {"polar_years": []}},
        figure_format={"format": "png", "dpi": 10},
    )
    single_year_tables = read_figure_tables(
        context=single_year_context,
        include_cumulative=False,
    )
    assert single_year_tables.cumulative_identity.empty
    assert single_year_tables.cumulative_summary.empty
    planned_jobs = _uncertainty_jobs(
        context=figure_context,
        identity=figure_tables.identity,
        summary=figure_tables.summary,
        cumulative_identity=figure_tables.cumulative_identity,
        cumulative_summary=figure_tables.cumulative_summary,
    )
    assert planned_jobs
    ar6_rows = uncertainty_global_ar6_rows_from_source(
        source=uncertainty_global_ar6_source(manifest=manifest),
        asr_frame=figure_tables.summary,
        requested_years=list(figure_context.requested_years),
        target_unit="kg CO2-eq",
    )
    assert not ar6_rows.summary.empty


def test_uncertainty_asr_dynamic_external_lca_figures_component_diagnostics(
    allocation_dummy_repo,
) -> None:
    prepare_dynamic_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
        historical_years=[2020, 2021],
        scenario_years=[2030],
    )
    project_name = "asr_uncertainty_dynamic_external_lca_figures"
    prepare_static_asr_external_lca_repo(
        allocation_dummy_repo,
        project_name=project_name,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact="GWP_100",
        impact_unit="kg CO2-eq",
    )
    project_root = allocation_dummy_repo.repo_root / f"{project_name}"
    mc_dir = external_lca_monte_carlo_dir(project_base=project_root)
    compact_dir = mc_dir / "supplier_v1__gwp100_lcia"
    compact_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "year": [2020, 2021],
            "lca_ssp_scenario": [None, None],
            "r_p": ["FR", "FR"],
            "s_p": ["D", "D"],
            "impact": ["GWP_100", "GWP_100"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
        }
    ).to_csv(compact_dir / "public_row_identity.csv", index=False)
    pd.DataFrame(
        {
            "run_index": list(range(4)),
            "0": [1.0 + 0.01 * run for run in range(4)],
            "1": [1.1 + 0.01 * run for run in range(4)],
        }
    ).to_csv(compact_dir / "lca_runs.csv", index=False)

    manifest = uncertainty_asr(
        project_name=project_name,
        years=range(2020, 2022),
        lcia_method="gwp100_lcia",
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        source="exiobase_396_ixi",
        base_asocc_args={
            "method_plan": "one_step",
            "one_step_methods": ["UT(FD)"],
        },
        base_cc_args={
            "static": {"active": False},
            "dynamic_ar6": {"category": ["C1"], "ssp_scenario": ["SSP1"]},
        },
        lca_args={
            "external_lca": {"active": True, "version_name": "supplier_v1"},
            "io_lca": {"active": False},
        },
        uncertainty_config={
            "mc_parameters": {
                "fixed": {"active": True, "n_runs": 2},
                "convergence": {"active": False},
            },
        },
        output_format="csv_compact",
        sobol_parameters={"active": False},
        figures=True,
        figure_options={"polar": {"polar_years": []}},
        figure_format={"format": "svg", "dpi": 1},
        subfigures=False,
        refresh=True,
    ).manifest

    assert manifest.artifacts is not None
    figure_paths = [Path(path) for path in manifest.artifacts["figure_paths"]]
    assert figure_paths
    assert all(path.exists() for path in figure_paths)
    assert (
        Path(manifest.artifacts["scope_manifest"]).parent / "composite_phase_index.json"
    ).exists()
    assert "external_lca::supplier_v1" in manifest.active_sources
    mc_inputs = [
        item for item in manifest.external_inputs if item["type"] == "external_lca_monte_carlo"
    ]
    assert len(mc_inputs) == 1
    assert [Path(path) for path in mc_inputs[0]["paths"]] == [compact_dir]


def test_uncertainty_asr_static_io_lca_component_source_routing(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    lca_only_kwargs = _static_asr_kwargs(project_name="asr_uncertainty_io_lca_lca_only")
    lca_only_kwargs["years"] = [2005, 2006]
    lca_only_kwargs["fu_code"] = "L2.a.b"
    lca_only_kwargs["base_asocc_args"] = {
        "method_plan": "pairs",
        "l1_l2_pairs": ["EG(Pop)::UT(GVAa)", "PR(GDPcap)::UT(GVAa)"],
        "include_lcia_based_allocation_methods": False,
    }
    lca_only_kwargs["lca_args"] = {
        "external_lca": {"active": False, "version_name": None},
        "io_lca": {"active": True},
    }
    lca_only_kwargs["uncertainty_config"] = {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}},
        **_inactive_denominator_uncertainty_sources(),
        "io_lca_uncertainty_sources": {
            "lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}}
        },
    }

    lca_only = uncertainty_asr(
        **lca_only_kwargs,
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 4},
            "convergence": {"active": False},
        },
        figures=True,
        figure_format={"format": "svg", "dpi": 1},
        refresh=True,
    ).manifest

    assert lca_only.active_sources == ("io_lca::lcia_uncertainty",)
    assert lca_only.artifacts is not None
    assert "sobol_indices" not in lca_only.artifacts
    assert "sobol_source_summary" not in lca_only.artifacts
    assert "sobol_readme" not in lca_only.artifacts
    assert lca_only.sobol is not None
    assert lca_only.sobol["ran"] is False
    assert lca_only.sobol["active_source_count"] == 1
    assert lca_only.artifacts["figure_paths"]
    acc_only_kwargs = _static_asr_kwargs(project_name="asr_uncertainty_io_lca_acc_only")
    acc_only_kwargs["base_asocc_args"]["include_lcia_based_allocation_methods"] = True
    acc_only_kwargs["base_asocc_args"]["reference_years"] = [2005]
    acc_only_kwargs["lca_args"] = {
        "external_lca": {"active": False, "version_name": None},
        "io_lca": {"active": True},
    }
    acc_only_kwargs["uncertainty_config"] = {
        "mc_parameters": {"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}},
        "asocc_uncertainty_sources": {"inter_method_uncertainty": {}},
    }

    acc_only = uncertainty_asr(
        **acc_only_kwargs,
        sobol_parameters={"active": False},
        figures=False,
        refresh=True,
    ).manifest

    assert acc_only.active_sources == ("acc::asocc::inter_method_uncertainty",)
    assert acc_only.artifacts is not None
    assert "sobol_indices" not in acc_only.artifacts
    assert "sobol_source_summary" not in acc_only.artifacts
    assert "sobol_readme" not in acc_only.artifacts


def test_uncertainty_asr_static_io_lca_deterministic_numerator_compact(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    kwargs = _static_asr_kwargs(project_name="asr_uncertainty_static_io_lca_compact")
    kwargs["base_asocc_args"] = {
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
        "include_lcia_based_allocation_methods": False,
    }
    kwargs["lca_args"] = {
        "external_lca": {"active": False, "version_name": None},
        "io_lca": {"active": True},
    }
    kwargs["uncertainty_config"] = {
        "mc_parameters": {
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 4, "rtol": 1e-12, "stable_runs": 2},
        }
    }

    manifest = uncertainty_asr(
        **kwargs,
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 4},
            "convergence": {"active": False},
        },
        refresh=True,
    ).manifest

    assert manifest.artifacts["public_output"] is not None
    assert manifest.artifacts["public_output"]["asr_runs"]["layout"] == "compact_run_matrix"
    assert manifest.artifacts is not None
    assert "sobol_indices" not in manifest.artifacts
    assert "sobol_source_summary" not in manifest.artifacts
    assert "sobol_readme" not in manifest.artifacts
    assert manifest.sobol is not None and manifest.sobol["ran"] is False


def test_uncertainty_asr_static_external_lca_monte_carlo_outputs(
    allocation_dummy_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_static_asr_external_lca_repo(
        allocation_dummy_repo,
        project_name="asr_uncertainty_static_external_lca",
        source="oecd_v2025",
        lcia_method="gwp100_lcia",
        impact="GWP_100",
        impact_unit="kg CO2-eq",
        include_deterministic=True,
    )
    project_root = allocation_dummy_repo.repo_root / "asr_uncertainty_static_external_lca"
    mc_dir = external_lca_monte_carlo_dir(project_base=project_root)
    mc_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "run_index": run,
                "year": 2005,
                "lca_ssp_scenario": "",
                "r_p": "FR",
                "s_p": "D",
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "value": value,
            }
            for run, value in [(0, 1.0), (1, 1.2)]
        ]
    ).to_csv(mc_dir / "supplier_v1__gwp100_lcia.csv", index=False)
    kwargs = _static_asr_kwargs(project_name="asr_uncertainty_static_external_lca")
    kwargs["source"] = "oecd_v2025"
    kwargs["lca_args"] = {
        "external_lca": {"active": True, "version_name": "supplier_v1"},
        "io_lca": {"active": False},
    }
    kwargs["uncertainty_config"]["mc_parameters"]["fixed"]["n_runs"] = 2

    manifest = uncertainty_asr(
        **kwargs,
        sobol_parameters={
            "active": True,
            "fixed": {"active": True, "n_base_samples": 4},
            "convergence": {"active": False},
        },
        refresh=True,
    ).manifest
    runtime_output = capsys.readouterr().out

    assert manifest.artifacts is not None
    assert runtime_output.strip()
    assert "external_lca::supplier_v1" in manifest.active_sources
    assert manifest.sobol is not None
    assert manifest.sobol["ran"] is False
    assert manifest.sobol["active_source_count"] == 1
    identity = pd.read_csv(manifest.artifacts["public_row_identity"])
    runs = pd.read_csv(manifest.artifacts["asr_runs"])
    source_methods = pd.read_csv(manifest.artifacts["source_methods"])

    assert "lca_ssp_scenario" not in identity.columns
    assert runs["run_index"].tolist() == [0, 1]
    assert bool(runs.drop(columns=["run_index"]).notna().all().all())
    assert "external_lca::supplier_v1" in set(source_methods["source_name"])


def test_uncertainty_asr_static_external_lca_deterministic_repeat(
    allocation_dummy_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_static_asr_external_lca_repo(
        allocation_dummy_repo,
        project_name="asr_uncertainty_static_external_lca_deterministic",
        source="oecd_v2025",
        lcia_method="gwp100_lcia",
        impact="GWP_100",
        impact_unit="kg CO2-eq",
        include_deterministic=True,
    )
    kwargs = _static_asr_kwargs(project_name="asr_uncertainty_static_external_lca_deterministic")
    kwargs["source"] = "oecd_v2025"
    kwargs["base_asocc_args"] = {
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
        "include_lcia_based_allocation_methods": False,
    }
    kwargs["years"] = 2005
    kwargs["lca_args"] = {
        "external_lca": {"active": True, "version_name": "supplier_v1"},
        "io_lca": {"active": False},
    }
    kwargs["uncertainty_config"] = {
        "mc_parameters": {
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 4, "rtol": 1e-12, "stable_runs": 2},
        }
    }

    figure_format = {"format": "svg", "dpi": 1}
    no_figure_kwargs = {**kwargs, "figures": False, "subfigures": True}
    no_figure_manifest = uncertainty_asr(
        **no_figure_kwargs,
        figure_format=figure_format,
        refresh=True,
    ).manifest
    runtime_output = capsys.readouterr().out

    assert no_figure_manifest.artifacts is not None
    assert runtime_output.strip()
    assert "external_lca::supplier_v1" not in no_figure_manifest.active_sources
    assert "figure_paths" not in no_figure_manifest.artifacts
    manifest = uncertainty_asr(
        **kwargs,
        figures=True,
        figure_format=figure_format,
        refresh=False,
    ).manifest
    assert manifest.run_id == no_figure_manifest.run_id
    assert manifest.artifacts is not None
    assert manifest.artifacts["figure_paths"]
    runs = pd.read_csv(manifest.artifacts["asr_runs"])
    assert runs.shape[0] == 4
    reused = uncertainty_asr(
        **kwargs,
        figures=True,
        figure_format=figure_format,
        refresh=False,
    ).manifest
    assert reused.run_id == manifest.run_id
    reused_no_figures = uncertainty_asr(
        **no_figure_kwargs,
        figure_format=figure_format,
        refresh=False,
    ).manifest
    assert reused_no_figures.run_id == manifest.run_id

    missing_kwargs = dict(kwargs)
    missing_kwargs["lca_args"] = {
        "external_lca": {"active": True, "version_name": "missing_supplier"},
        "io_lca": {"active": False},
    }
    with pytest.raises(FileNotFoundError):
        uncertainty_asr(**missing_kwargs, refresh=True).manifest


def test_uncertainty_asr_convergence_reports_unreached_and_reuse(
    allocation_dummy_repo,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    kwargs = _static_asr_kwargs(project_name="asr_uncertainty_convergence_unreached")
    kwargs["years"] = [2005, 2006]
    kwargs["base_asocc_args"]["include_lcia_based_allocation_methods"] = True
    kwargs["base_asocc_args"]["reference_years"] = [2005]
    kwargs["lca_args"] = {
        "external_lca": {"active": False, "version_name": None},
        "io_lca": {"active": True},
    }
    kwargs["uncertainty_config"] = {
        "mc_parameters": {
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 2, "rtol": 1e-12, "stable_runs": 2},
        },
        "asocc_uncertainty_sources": {"inter_method_uncertainty": {}},
        "io_lca_uncertainty_sources": {
            "lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}}
        },
    }
    kwargs["figures"] = True
    kwargs["subfigures"] = True
    kwargs["figure_options"] = {
        "per_method": False,
        "multi_method": False,
        "inter_method": False,
        "polar": {"active": False},
    }
    kwargs["figure_format"] = {"format": "svg", "dpi": 1}
    kwargs["sobol_parameters"] = {
        "active": True,
        "fixed": {"active": False},
        "convergence": {"active": True, "max_base_samples": 1, "rtol": 1e-12},
    }

    manifest = uncertainty_asr(**kwargs, refresh=True).manifest
    first_output = capsys.readouterr().out

    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is False
    assert manifest.sobol is not None
    assert manifest.sobol["ran"] is True
    assert manifest.sobol["reached"] is False
    assert first_output.strip()

    reused = uncertainty_asr(**kwargs, refresh=False).manifest
    reuse_output = capsys.readouterr().out
    assert reused.run_id == manifest.run_id
    assert reuse_output.strip()
    stale_run_file = Path(manifest.artifacts["scope_manifest"]).parents[1] / "stale.txt"
    stale_run_file.write_text("stale", encoding="utf-8")
    stale_acc_file = (
        Path(manifest.deterministic_prerequisites[0]["scope_manifest"]).parents[1] / "stale.txt"
    )
    stale_acc_file.write_text("stale", encoding="utf-8")
    stale_lca_file = (
        Path(manifest.deterministic_prerequisites[1]["scope_manifest"]).parents[1] / "stale.txt"
    )
    stale_lca_file.write_text("stale", encoding="utf-8")
    refreshed = uncertainty_asr(**kwargs, refresh=True).manifest
    assert refreshed.status == "complete"
    assert not stale_run_file.exists()
    assert not stale_acc_file.exists()
    assert not stale_lca_file.exists()


def test_uncertainty_asr_compact_convergence_reaches(allocation_dummy_repo) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    kwargs = _static_asr_kwargs(project_name="asr_uncertainty_compact_convergence")
    kwargs["base_asocc_args"] = {
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
        "include_lcia_based_allocation_methods": False,
    }
    kwargs["lca_args"] = {
        "external_lca": {"active": False, "version_name": None},
        "io_lca": {"active": True},
    }
    kwargs["uncertainty_config"] = {
        "mc_parameters": {
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 4, "stable_runs": 2},
        }
    }

    manifest = uncertainty_asr(**kwargs, refresh=True).manifest

    assert manifest.convergence is not None
    assert manifest.convergence["reached"] is True


def test_asr_convergence_checks_frequency_mean_from_public_runs(tmp_path: Path) -> None:
    acc_manifest = _compact_acc_manifest(
        tmp_path=tmp_path,
        acc_values=pd.DataFrame({"run_index": [0, 1, 2, 3], "0": [1.0, 1.0, 1.0, 1.0]}),
    )
    identity = pd.DataFrame({"public_row_id": [0], "year": [2005]})
    plan = _compact_asr_plan(
        identity=identity,
        acc_manifest=acc_manifest,
        lca_values=np.array([[0.0], [2.0], [0.5], [1.0]], dtype=np.float64),
    )
    runtime = normalize_uncertainty_request(
        family="asr",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {
                "active": True,
                "max_runs": 4,
                "stable_runs": 2,
                "rtol": 0.1,
                "convergence_statistics": ["mean"],
            },
        },
    )
    paths = _asr_run_paths(tmp_path / "asr_frequency")

    completed, convergence = write_asr_run_outputs(
        paths=paths,
        plan=plan,
        runtime=runtime,
        show_progress=False,
    )
    summary = pd.read_csv(paths.summary_stats_runs)
    frequency = summary.loc[summary["asr_metric"].eq(ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC)]

    assert completed == 4
    assert convergence is not None
    assert convergence["reached"] is False
    assert bool(frequency["mean"].isna().all())
    np.testing.assert_allclose(frequency[ASR_FREQUENCY_VALUE_COLUMN], [0.75])


def test_asr_convergence_checks_dynamic_cumulative_metrics(tmp_path: Path) -> None:
    acc_manifest = _compact_acc_manifest(
        tmp_path=tmp_path,
        acc_values=pd.DataFrame({"run_index": [0, 1], "0": [100.0, 1.0], "1": [1.0, 100.0]}),
    )
    identity = pd.DataFrame({"public_row_id": [0, 1], "year": [2020, 2030]})
    cumulative_identity = pd.DataFrame({"public_row_id": [0]})
    plan = _compact_asr_plan(
        identity=identity,
        acc_manifest=acc_manifest,
        lca_values=np.array([[0.0, 2.0], [0.0, 200.0]], dtype=np.float64),
        cumulative_identity=cumulative_identity,
        cumulative_public_row_groups=(("0", "1"),),
    )
    runtime = normalize_uncertainty_request(
        family="asr",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {
                "active": True,
                "max_runs": 2,
                "stable_runs": 1,
                "rtol": 0.01,
                "convergence_statistics": ["mean"],
            },
        },
    )
    runtime = replace(runtime, batch_size=1)

    completed, convergence = write_asr_run_outputs(
        paths=_asr_run_paths(tmp_path / "asr_cumulative"),
        plan=plan,
        runtime=runtime,
        show_progress=False,
    )

    assert completed == 2
    assert convergence is not None
    assert convergence["reached"] is False


def test_asr_checkpoints_skip_final_component_refresh_without_live_sessions(
    tmp_path: Path,
) -> None:
    acc_manifest = _compact_acc_manifest(
        tmp_path=tmp_path,
        acc_values=pd.DataFrame({"run_index": [0], "0": [2.0]}),
    )
    identity = pd.DataFrame({"public_row_id": [0], "year": [2005]})
    plan = _compact_asr_plan(
        identity=identity,
        acc_manifest=acc_manifest,
        lca_values=np.array([[1.0]], dtype=np.float64),
    )
    runtime = normalize_uncertainty_request(
        family="asr",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": True, "n_runs": 1},
            "convergence": {"active": False},
        },
    )
    result = run_asr_checkpoints(
        paths=_asr_run_paths(tmp_path / "asr_checkpoint_no_components"),
        runtime=runtime,
        checkpoints=(1,),
        initial_plan=plan,
        initial_acc_manifest=acc_manifest,
        initial_acc_session=None,
        initial_lca_input=plan.lca_input,
        initial_lca_session=None,
        project_name="asr_checkpoint_no_components",
        years=[2005],
        shared_methods=["gwp100_lcia"],
        fu_code="L2.a.a",
        r_p=["FR"],
        s_p=["D"],
        r_c=None,
        r_f=None,
        mrio_scope={},
        asocc_config={},
        base_cc_args={},
        source_config=cast(Any, None),
        external_method=None,
        proj_base=tmp_path,
        source_label="exiobase_396_ixi",
        lca_type="io_lca",
        lca_version_name=None,
        base_allocate_args={},
        output_format="csv_compact",
        phase=NullPhasePrinter(),
        render_subfigures=False,
        subfigures=False,
        figure_options=None,
        figure_format=None,
        run_id=None,
        acc_progress=monte_carlo_run_progress(source="uncertainty_acc", enabled=False),
        lca_progress=monte_carlo_run_progress(source="uncertainty_io_lca", enabled=False),
        asr_progress=monte_carlo_run_progress(source="uncertainty_asr", enabled=False),
        progress_mode="fixed",
        progress_max_runs=1,
        progress_component=False,
    )

    assert result.completed_runs == 1
    assert result.acc_session is None
    assert result.lca_session is None


def _compact_acc_manifest(*, tmp_path: Path, acc_values: pd.DataFrame):
    acc_root = tmp_path / "acc_run"
    results = acc_root / "results"
    logs = acc_root / "logs"
    results.mkdir(parents=True)
    logs.mkdir(parents=True)
    row_columns = [column for column in acc_values.columns if column != "run_index"]
    identity = pd.DataFrame(
        {
            "public_row_id": [int(column) for column in row_columns],
            "year": [2005 + index for index in range(len(row_columns))],
        }
    )
    identity_path = results / "public_row_identity.csv"
    runs_path = results / "acc_runs.csv"
    summary_path = results / "summary_stats_runs.csv"
    readme_path = results / "README.txt"
    source_methods_path = logs / "source_methods.csv"
    scope_manifest_path = logs / "scope_manifest.json"
    identity.to_csv(identity_path, index=False)
    with CompactRunMatrixWriter(path=runs_path, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=acc_values["run_index"].to_numpy(dtype=np.int64),
            values=acc_values.loc[:, row_columns].to_numpy(dtype=np.float64),
            batch_index=0,
        )
    summary_path.write_text("", encoding="utf-8")
    readme_path.write_text("", encoding="utf-8")
    source_methods_path.write_text("source_component,source_name\nacc,test\n", encoding="utf-8")
    return build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        completed_runs=len(acc_values),
        artifacts={
            "scope_manifest": str(scope_manifest_path),
            "public_row_identity": str(identity_path),
            "acc_runs": str(runs_path),
            "summary_stats_runs": str(summary_path),
            "results_readme": str(readme_path),
            "source_methods": str(source_methods_path),
            "public_output": {"acc_runs": {"layout": "compact_run_matrix"}},
        },
    )


def _compact_asr_plan(
    *,
    identity: pd.DataFrame,
    acc_manifest,
    lca_values: np.ndarray,
    cumulative_identity: pd.DataFrame | None = None,
    cumulative_public_row_groups: tuple[tuple[str, ...], ...] = (),
) -> ASRUncertaintyPlan:
    cumulative = pd.DataFrame() if cumulative_identity is None else cumulative_identity
    return ASRUncertaintyPlan(
        identity=identity,
        summary_identity=pd.concat(
            [
                identity.assign(asr_metric=ASR_VALUE_METRIC),
                identity.assign(asr_metric=ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC),
            ],
            ignore_index=True,
        ),
        summary_public_row_groups=tuple(
            (str(public_id),) for public_id in identity["public_row_id"]
        ),
        cumulative_identity=cumulative,
        cumulative_summary_identity=(
            pd.concat(
                [
                    cumulative.assign(asr_metric=ASR_CUMULATIVE_VALUE_METRIC),
                    cumulative.assign(
                        asr_metric=ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC
                    ),
                ],
                ignore_index=True,
            )
            if not cumulative.empty
            else pd.DataFrame()
        ),
        cumulative_summary_public_row_groups=(
            tuple((str(public_id),) for public_id in cumulative["public_row_id"])
            if not cumulative.empty
            else ()
        ),
        cumulative_public_row_groups=cumulative_public_row_groups,
        acc_positions=identity["public_row_id"].to_numpy(dtype=np.int64),
        lca_positions=identity["public_row_id"].to_numpy(dtype=np.int64),
        lca_unit_factors=np.ones(len(identity), dtype=np.float64),
        acc_manifest=acc_manifest,
        lca_input=LCAUncertaintyInput(
            identity=identity,
            fixed_values=None,
            manifest=None,
            external_inputs=(),
            source_method_rows=pd.DataFrame(),
            active_sources=(),
            lca_type="io_lca",
            run_values_for_runs=lambda run_indices: lca_values[run_indices],
            run_inventory_size=len(lca_values),
        ),
        asr_run_layout="compact_run_matrix",
        source_method_rows=pd.DataFrame(),
        active_sources=(),
    )


def _asr_run_paths(root: Path) -> ASRUncertaintyRunPaths:
    return ASRUncertaintyRunPaths(
        run_root=root,
        public_row_identity=root / "results" / "public_row_identity.csv",
        public_runs=root / "results" / "asr_runs.csv",
        summary_stats_runs=root / "results" / "summary_stats_runs.csv",
        cumulative_row_identity=root / "results" / "cumulative_identity.csv",
        cumulative_runs=root / "results" / "cumulative_asr_runs.csv",
        cumulative_summary_stats_runs=root / "results" / "cumulative_summary_stats_runs.csv",
        results_readme=root / "results" / "README.txt",
        source_methods=root / "logs" / "source_methods.csv",
        sobol_indices=root / "results" / "sobol" / "sobol_indices.csv",
        sobol_source_summary=root / "results" / "sobol" / "sobol_source_summary.csv",
        sobol_readme=root / "results" / "sobol" / "README_sobol.txt",
        scope_manifest=root / "logs" / "scope_manifest.json",
    )


def _asr_paths_from_manifest(manifest) -> ASRUncertaintyRunPaths:
    artifacts = cast(dict[str, object], manifest.artifacts)
    scope_manifest = Path(str(artifacts["scope_manifest"]))
    run_root = scope_manifest.parents[1]
    sobol_root = run_root / "results" / "sobol"
    return ASRUncertaintyRunPaths(
        run_root=run_root,
        public_row_identity=Path(str(artifacts["public_row_identity"])),
        public_runs=Path(str(artifacts["asr_runs"])),
        summary_stats_runs=Path(str(artifacts["summary_stats_runs"])),
        cumulative_row_identity=Path(str(artifacts["cumulative_row_identity"])),
        cumulative_runs=Path(str(artifacts["cumulative_asr_runs"])),
        cumulative_summary_stats_runs=Path(str(artifacts["cumulative_summary_stats_runs"])),
        results_readme=Path(str(artifacts["results_readme"])),
        source_methods=Path(str(artifacts["source_methods"])),
        sobol_indices=sobol_root / "sobol_indices.csv",
        sobol_source_summary=sobol_root / "sobol_source_summary.csv",
        sobol_readme=sobol_root / "README_sobol.txt",
        scope_manifest=scope_manifest,
    )


def test_uncertainty_asr_convergence_uses_fixed_io_lca_component_inventory(
    allocation_dummy_repo,
) -> None:
    prepare_static_asr_io_lca_repo(
        allocation_dummy_repo,
        source="exiobase_396_ixi",
        lcia_method="gwp100_lcia",
        impact_parent="GWP_100",
        impact_unit="kg CO2-eq",
    )
    project_name = "asr_uncertainty_io_lca_convergence_inventory"
    kwargs = _static_asr_kwargs(project_name=project_name)
    kwargs["base_asocc_args"] = {
        "method_plan": "one_step",
        "one_step_methods": ["UT(FD)"],
        "include_lcia_based_allocation_methods": False,
    }
    kwargs["lca_args"] = {
        "external_lca": {"active": False, "version_name": None},
        "io_lca": {"active": True},
    }
    kwargs["uncertainty_config"] = {
        "mc_parameters": {
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 4, "stable_runs": 2},
        },
        "io_lca_uncertainty_sources": {
            "lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}}
        },
    }

    manifest = uncertainty_asr(**kwargs, refresh=True).manifest

    manifest_paths = sorted(
        (allocation_dummy_repo.repo_root / f"{project_name}" / "A_lca" / "io_lca").glob(
            "**/monte_carlo/mc_*/logs/scope_manifest.json"
        )
    )
    inventory_manifests = []
    for path in manifest_paths:
        stored_manifest = read_manifest(path=path)
        if stored_manifest.component_inventory is not None:
            inventory_manifests.append(stored_manifest)

    assert len(inventory_manifests) == 1
    for stored_manifest in inventory_manifests:
        assert stored_manifest.run_id == manifest.run_id
        assert stored_manifest.mode == "fixed"
        assert stored_manifest.component_inventory is not None
        assert stored_manifest.completed_runs == stored_manifest.component_inventory["target_runs"]


def test_external_lca_monte_carlo_errors_and_mixed_matrix(tmp_path: Path) -> None:
    identity = pd.DataFrame(
        {
            "public_row_id": [0],
            "lcia_method": ["gwp100_lcia"],
            "year": [2005],
            "impact": ["GWP_100"],
            "impact_unit": ["kg CO2-eq"],
            "r_p": ["FR"],
            "s_p": ["D"],
        }
    )
    source_values = pd.DataFrame({"value": [1.0, 2.0]}).to_numpy(dtype=float)
    source = ExternalLCAMonteCarloSource(
        version_name="supplier_v1",
        lcia_method="gwp100_lcia",
        identity=identity,
        run_indices=pd.Series([0, 1]).to_numpy(dtype=int),
        paths=(tmp_path / "supplier_v1__gwp100_lcia.csv",),
        values_for_runs=lambda run_indices: source_values[run_indices],
    )
    lca_input = LCAUncertaintyInput(
        identity=identity,
        fixed_values=None,
        manifest=None,
        external_inputs=(),
        source_method_rows=pd.DataFrame(),
        active_sources=("external_lca::supplier_v1",),
        lca_type="external",
        run_values_for_runs=lambda run_indices: external_lca_values_for_runs(
            source=source,
            run_indices=run_indices,
        ),
        run_values_for_units=lambda units: external_lca_values_for_units(
            source=source,
            unit_values=units,
        ),
        run_inventory_size=len(source.run_indices),
    )
    assert external_lca_values_for_runs(
        source=source,
        run_indices=pd.Series([], dtype=int).to_numpy(dtype=int),
    ).shape == (0, 1)
    assert external_lca_values_for_units(
        source=source,
        unit_values=pd.Series([0.0, 0.99]).to_numpy(dtype=float),
    ).tolist() == [[1.0], [2.0]]
    with pytest.raises(ValueError):
        external_lca_values_for_runs(
            source=source,
            run_indices=pd.Series([2]).to_numpy(dtype=int),
        )
    with pytest.raises(ValueError):
        lca_values_for_runs(lca_input=lca_input, run_indices=pd.Series([2]).to_numpy(dtype=int))

    bad_dir = external_lca_monte_carlo_dir(project_base=tmp_path)
    bad_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"year": 2005, "impact": "GWP_100", "impact_unit": "kg CO2-eq", "value": 1.0}]
    ).to_csv(bad_dir / "supplier_v1__gwp100_lcia.csv", index=False)
    with pytest.raises(ValueError):
        load_external_lca_monte_carlo_source(
            proj_base=tmp_path,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
            years=[2005],
            base_allocate_args={
                "fu_code": "L2.a.a",
                "r_p": ["FR"],
                "s_p": ["D"],
                "r_c": None,
                "r_f": None,
            },
        )

    valid_rows = pd.DataFrame(
        [
            {
                "run_index": 0,
                "year": 2005,
                "lca_ssp_scenario": "",
                "r_p": "FR",
                "s_p": "D",
                "impact": "GWP_100",
                "impact_unit": "kg CO2-eq",
                "value": 1.0,
            }
        ]
    )
    matrix_path = bad_dir / "supplier_v1__gwp100_lcia.csv"
    repo_root = tmp_path / "repo"
    cc_dir = repo_root / "data_raw" / "carrying_capacities"
    cc_dir.mkdir(parents=True)
    (cc_dir / "gwp100_lcia_cc_steady_state.csv").write_text(
        "impact_full_name,impact,impact_unit,min_cc,max_cc\n"
        "Climate change,GWP_100,kg CO2-eq,1.0,2.0\n",
        encoding="utf-8",
    )
    historical_and_ssp_rows = pd.concat(
        [
            valid_rows,
            valid_rows.assign(year=2006, lca_ssp_scenario="SSP2", value=3.0),
            valid_rows.assign(run_index=1, value=2.0),
            valid_rows.assign(run_index=1, year=2006, lca_ssp_scenario="SSP2", value=4.0),
        ],
        ignore_index=True,
    )
    historical_and_ssp_rows.to_csv(matrix_path, index=False)
    set_default_repo_root(repo_root)
    try:
        path_source = load_external_lca_monte_carlo_source_from_path(
            path=matrix_path,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
            years=[2005, 2006],
            base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
        )
    finally:
        clear_default_repo_root()
    assert path_source.identity["lca_ssp_scenario"].tolist() == [None, "SSP2"]
    assert external_lca_values_for_runs(
        source=path_source,
        run_indices=path_source.run_indices,
    ).tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert path_source.values_for_runs(np.empty(0, dtype=np.int64)).shape == (0, 2)
    base_args = {"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]}
    parquet_path = bad_dir / "supplier_v1__gwp100_lcia.parquet"
    historical_and_ssp_rows.to_parquet(parquet_path, index=False, row_group_size=2)
    set_default_repo_root(repo_root)
    try:
        parquet_source = load_external_lca_long_matrix_source(
            path=parquet_path,
            lcia_method="gwp100_lcia",
            years=[2005, 2006],
            base_allocate_args=base_args,
        )
    finally:
        clear_default_repo_root()
    parquet_identity = parquet_source.identity
    parquet_values = parquet_source.values_for_runs(parquet_source.run_indices)
    parquet_runs = parquet_source.run_indices
    assert parquet_identity["lca_ssp_scenario"].tolist() == [None, "SSP2"]
    assert parquet_values.tolist() == [[1.0, 3.0], [2.0, 4.0]]
    assert list(parquet_runs) == [0, 1]
    assert parquet_source.values_for_runs(np.empty(0, dtype=np.int64)).shape == (0, 2)
    assert parquet_source.values_for_runs(np.array([1], dtype=np.int64)).tolist() == [[2.0, 4.0]]
    with pytest.raises(ValueError, match="missing run_index values"):
        parquet_source.values_for_runs(np.array([2], dtype=np.int64))

    stream_error_cases = {
        "missing_selector": valid_rows.drop(columns=["s_p"]),
        "missing_year_column": valid_rows.drop(columns=["year"]),
        "empty_selector": valid_rows.assign(s_p=" "),
        "unexpected_selector": valid_rows.assign(r_c="FR"),
        "duplicate_identity": pd.concat([valid_rows, valid_rows], ignore_index=True),
        "duplicate_after_template": pd.concat(
            [
                valid_rows,
                valid_rows.assign(run_index=1),
                valid_rows.assign(run_index=1, value=2.0),
            ],
            ignore_index=True,
        ),
        "nondecreasing": pd.concat(
            [valid_rows.assign(run_index=1), valid_rows],
            ignore_index=True,
        ),
        "missing_run_zero": valid_rows.assign(run_index=1),
    }
    for stem, rows in stream_error_cases.items():
        case_path = bad_dir / f"{stem}.csv"
        rows.to_csv(case_path, index=False)
        set_default_repo_root(repo_root)
        try:
            with pytest.raises(ValueError):
                load_external_lca_long_matrix_source(
                    path=case_path,
                    lcia_method="gwp100_lcia",
                    years=[2005],
                    base_allocate_args=base_args,
                )
        finally:
            clear_default_repo_root()

    no_year_path = bad_dir / "no_requested_year.csv"
    valid_rows.to_csv(no_year_path, index=False)
    with pytest.raises(ValueError):
        load_external_lca_long_matrix_source(
            path=no_year_path,
            lcia_method="gwp100_lcia",
            years=[2030],
            base_allocate_args=base_args,
        )
    no_year_parquet_path = bad_dir / "no_requested_year.parquet"
    valid_rows.to_parquet(no_year_parquet_path, index=False)
    with pytest.raises(ValueError):
        load_external_lca_long_matrix_source(
            path=no_year_parquet_path,
            lcia_method="gwp100_lcia",
            years=[2030],
            base_allocate_args=base_args,
        )

    large_path = bad_dir / "large_inventory.csv"
    pd.concat(
        [valid_rows.assign(run_index=run, value=float(run)) for run in range(1026)],
        ignore_index=True,
    ).to_csv(large_path, index=False)
    set_default_repo_root(repo_root)
    try:
        large_source = load_external_lca_long_matrix_source(
            path=large_path,
            lcia_method="gwp100_lcia",
            years=[2005],
            base_allocate_args=base_args,
        )
    finally:
        clear_default_repo_root()
    large_values = large_source.values_for_runs(large_source.run_indices)
    large_runs = large_source.run_indices
    assert list(large_runs[:2]) == [0, 1]
    assert int(large_runs[-1]) == 1025
    assert large_values[-1].tolist() == [1025.0]

    compact_root = tmp_path / "compact_lca"
    compact_dir = external_lca_monte_carlo_dir(project_base=compact_root) / (
        "supplier_v1__gwp100_lcia"
    )
    compact_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "year": [2005, 2006],
            "lca_ssp_scenario": [None, "SSP2"],
            "r_p": ["FR", "FR"],
            "s_p": ["D", "D"],
            "impact": ["GWP_100", "GWP_100"],
            "impact_unit": ["kg CO2-eq", "kg CO2-eq"],
        }
    ).to_csv(compact_dir / "public_row_identity.csv", index=False)
    pd.DataFrame({"run_index": [0, 1], "0": [1.0, 2.0], "1": [3.0, 4.0]}).to_csv(
        compact_dir / "lca_runs.csv",
        index=False,
    )
    historical_and_ssp_rows.assign(value=99.0).to_csv(
        external_lca_monte_carlo_dir(project_base=compact_root) / "supplier_v1__gwp100_lcia.csv",
        index=False,
    )
    set_default_repo_root(repo_root)
    try:
        compact_source = load_external_lca_monte_carlo_source(
            proj_base=compact_root,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
            years=[2005, 2006],
            base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
        )
    finally:
        clear_default_repo_root()
    assert compact_source is not None
    assert compact_source.paths == (compact_dir,)
    assert compact_source.identity["lca_ssp_scenario"].tolist() == [None, "SSP2"]
    assert external_lca_values_for_runs(
        source=compact_source,
        run_indices=compact_source.run_indices,
    ).tolist() == [[1.0, 3.0], [2.0, 4.0]]
    set_default_repo_root(repo_root)
    try:
        reloaded_compact = load_external_lca_monte_carlo_source_from_path(
            path=compact_source.paths[0],
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
            years=[2005, 2006],
            base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
        )
    finally:
        clear_default_repo_root()
    assert external_lca_values_for_runs(
        source=reloaded_compact,
        run_indices=reloaded_compact.run_indices,
    ).tolist() == [[1.0, 3.0], [2.0, 4.0]]

    compact_error_root = tmp_path / "compact_lca_errors"
    compact_error_base = external_lca_monte_carlo_dir(project_base=compact_error_root)
    compact_identity = pd.read_csv(compact_dir / "public_row_identity.csv")
    compact_runs = pd.read_csv(compact_dir / "lca_runs.csv")
    compact_error_cases = {
        "missing_public_id": (
            compact_identity.drop(columns=["public_row_id"]),
            compact_runs,
        ),
        "bad_public_id": (
            compact_identity.assign(public_row_id=[0, 2]),
            compact_runs,
        ),
        "bad_run_index": (
            compact_identity,
            compact_runs.assign(run_index=[0, 2]),
        ),
        "bad_run_columns": (
            compact_identity,
            compact_runs.rename(columns={"1": "missing"}),
        ),
    }
    for stem, (identity_frame, runs_frame) in compact_error_cases.items():
        case_dir = compact_error_base / f"supplier_v1__gwp100_lcia_{stem}"
        case_dir.mkdir(parents=True)
        identity_frame.to_csv(case_dir / "public_row_identity.csv", index=False)
        runs_frame.to_csv(case_dir / "lca_runs.csv", index=False)
        set_default_repo_root(repo_root)
        try:
            with pytest.raises(ValueError):
                load_external_lca_monte_carlo_source_from_path(
                    path=case_dir,
                    version_name="supplier_v1",
                    lcia_method="gwp100_lcia",
                    years=[2005, 2006],
                    base_allocate_args=base_args,
                )
        finally:
            clear_default_repo_root()

    set_default_repo_root(repo_root)
    try:
        with pytest.raises(ValueError):
            load_external_lca_monte_carlo_source(
                proj_base=compact_root,
                version_name="supplier_v1",
                lcia_method="gwp100_lcia",
                years=[2030],
                base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
            )
    finally:
        clear_default_repo_root()

    _identity, values, _run_indices = _materialize_matrix(
        rows=valid_rows,
        path=matrix_path,
        version_name="supplier_v1",
        lcia_method="gwp100_lcia",
    )
    assert values.tolist() == [[1.0]]

    direct_identity = pd.DataFrame({"public_row_id": [0, 1]})
    fixed_lca_values = np.array([99.0], dtype=np.float64)
    direct_source = ExternalLCAMonteCarloSource(
        version_name="supplier_v1",
        lcia_method="gwp100_lcia",
        identity=pd.DataFrame({"public_row_id": [0]}),
        run_indices=np.array([0, 1], dtype=np.int64),
        paths=(matrix_path,),
        values_for_runs=lambda run_indices: np.array([[1.0], [2.0]], dtype=np.float64)[run_indices],
    )
    run_provider = _external_lca_run_value_provider(
        identity=direct_identity,
        value_blocks=[direct_source, fixed_lca_values],
        version_name="supplier_v1",
    )
    assert run_provider(np.empty(0, dtype=np.int64)).shape == (0, 2)
    np.testing.assert_allclose(
        run_provider(np.array([1, 0], dtype=np.int64)),
        [[2.0, 99.0], [1.0, 99.0]],
    )
    with pytest.raises(ValueError, match="inventory was exhausted"):
        run_provider(np.array([2], dtype=np.int64))

    unit_provider = _external_lca_unit_value_provider(
        identity=direct_identity,
        value_blocks=[direct_source, fixed_lca_values],
    )
    np.testing.assert_allclose(
        unit_provider(np.array([0.0, 0.75], dtype=np.float64)),
        [[1.0, 99.0], [2.0, 99.0]],
    )

    public_runs = bad_dir / "public_lca_runs.csv"
    with CompactRunMatrixWriter(path=public_runs, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=[0, 1],
            values=[[1.0, 3.0], [2.0, 4.0]],
            batch_index=0,
        )
    public_provider = _public_lca_run_value_provider(
        runs_path=public_runs,
        output_format="csv_compact",
        column_count=2,
        run_count=2,
    )
    assert public_provider(np.empty(0, dtype=np.int64)).shape == (0, 2)
    np.testing.assert_allclose(
        public_provider(np.array([1, 0], dtype=np.int64)),
        [[2.0, 4.0], [1.0, 3.0]],
    )
    with pytest.raises(ValueError, match="inventory was exhausted"):
        public_provider(np.array([2], dtype=np.int64))

    with pytest.raises(ValueError):
        _normalize_rows(
            frame=valid_rows.assign(lcia_method="gwp100_lcia"),
            path=bad_dir / "conflict.csv",
            lcia_method="gwp100_lcia",
            years=[2005],
            base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
        )
    set_default_repo_root(repo_root)
    try:
        with pytest.raises(ValueError):
            _normalize_rows(
                frame=valid_rows.assign(run_index=-1),
                path=bad_dir / "negative.csv",
                lcia_method="gwp100_lcia",
                years=[2005],
                base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
            )
        with pytest.raises(ValueError):
            _normalize_rows(
                frame=valid_rows,
                path=bad_dir / "empty_year.csv",
                lcia_method="gwp100_lcia",
                years=[2006],
                base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
            )
    finally:
        clear_default_repo_root()
    with pytest.raises(ValueError):
        _materialize_matrix(
            rows=pd.concat([valid_rows, valid_rows], ignore_index=True),
            path=matrix_path,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
        )
    with pytest.raises(ValueError):
        _materialize_matrix(
            rows=valid_rows.assign(run_index=1),
            path=matrix_path,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
        )
    missing_run = pd.concat(
        [
            valid_rows,
            valid_rows.assign(run_index=1, s_p="X"),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError):
        _materialize_matrix(
            rows=missing_run,
            path=matrix_path,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
        )
    incomplete_run = pd.concat(
        [
            valid_rows,
            valid_rows.assign(s_p="X"),
            valid_rows.assign(run_index=1),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError):
        _materialize_matrix(
            rows=incomplete_run,
            path=matrix_path,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
        )

    assert _lca_values_for_units(
        context=cast(
            ASRSobolEvaluationContext,
            type(
                "Context",
                (),
                {"lca_context": None, "lca_input": lca_input},
            )(),
        ),
        units=pd.Series([0.25]).to_numpy(dtype=float)[:, None],
    ).tolist() == [[1.0]]

    deterministic_input = LCAUncertaintyInput(
        identity=identity,
        fixed_values=pd.Series([4.0]).to_numpy(dtype=float),
        manifest=None,
        external_inputs=(),
        source_method_rows=pd.DataFrame(),
        active_sources=(),
        lca_type="external",
    )
    assert _lca_values_for_units(
        context=cast(
            ASRSobolEvaluationContext,
            type(
                "Context",
                (),
                {"lca_context": None, "lca_input": deterministic_input},
            )(),
        ),
        units=pd.Series([0.25]).to_numpy(dtype=float)[:, None],
    ).tolist() == [[4.0]]

    ssp_root = tmp_path / "ssp_lca"
    ssp_dir = external_lca_monte_carlo_dir(project_base=ssp_root)
    ssp_dir.mkdir(parents=True, exist_ok=True)
    ssp_path = ssp_dir / "supplier_v1__gwp100_lcia__ssp2.csv"
    valid_rows.to_csv(ssp_path, index=False)
    with pytest.raises(ValueError):
        load_external_lca_monte_carlo_source(
            proj_base=ssp_root,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
            years=[2005],
            base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
        )

    multi_root = tmp_path / "multi"
    multi_dir = external_lca_monte_carlo_dir(project_base=multi_root)
    multi_dir.mkdir(parents=True, exist_ok=True)
    multi_rows = valid_rows.assign(lca_ssp_scenario="SSP1")
    multi_rows.to_csv(multi_dir / "supplier_v1__gwp100_lcia.csv", index=False)
    multi_rows.to_pickle(multi_dir / "supplier_v1__gwp100_lcia.pickle")
    set_default_repo_root(repo_root)
    try:
        selected_source = load_external_lca_monte_carlo_source(
            proj_base=multi_root,
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
            years=[2005],
            base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
        )
    finally:
        clear_default_repo_root()
    assert selected_source is not None
    assert selected_source.paths[0].suffix == ".csv"
    set_default_repo_root(repo_root)
    try:
        pickle_source = load_external_lca_monte_carlo_source_from_path(
            path=multi_dir / "supplier_v1__gwp100_lcia.pickle",
            version_name="supplier_v1",
            lcia_method="gwp100_lcia",
            years=[2005],
            base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
        )
    finally:
        clear_default_repo_root()
    assert external_lca_values_for_runs(
        source=pickle_source,
        run_indices=pickle_source.run_indices,
    ).tolist() == [[1.0]]
    assert (
        load_external_lca_monte_carlo_source(
            proj_base=multi_root,
            version_name="supplier_v1",
            lcia_method="pb_lcia",
            years=[2005],
            base_allocate_args={"fu_code": "L2.a.a", "r_p": ["FR"], "s_p": ["D"]},
        )
        is None
    )


def test_asr_sparse_writer_preserves_empty_requested_runs(tmp_path: Path) -> None:
    acc_results = tmp_path / "acc_run" / "results"
    acc_logs = tmp_path / "acc_run" / "logs"
    acc_results.mkdir(parents=True)
    acc_logs.mkdir(parents=True)
    acc_identity_path = acc_results / "public_row_identity.csv"
    acc_runs_path = acc_results / "acc_runs.csv"
    acc_identity_path.write_text("public_row_id,year\n0,2005\n", encoding="utf-8")
    with SparseRunRowsWriter(path=acc_runs_path, output_format="csv_compact") as writer:
        writer.write_batch(
            rows=SparseRunRows(
                run_index=np.empty(0, dtype=np.int64),
                public_row_id=np.empty(0, dtype=np.int64),
                values=np.empty(0, dtype=np.float64),
                value_column="acc",
            ),
            batch_index=0,
        )
    empty = acc_results / "empty.csv"
    empty.write_text("", encoding="utf-8")
    source_methods = acc_logs / "source_methods.csv"
    source_methods.write_text("source_component,source_name\nacc,demo\n", encoding="utf-8")
    acc_manifest = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=("asocc::inter_method_uncertainty",),
        artifacts={
            "scope_manifest": str(acc_logs / "scope_manifest.json"),
            "public_row_identity": str(acc_identity_path),
            "acc_runs": str(acc_runs_path),
            "summary_stats_runs": str(empty),
            "results_readme": str(empty),
            "source_methods": str(source_methods),
            "public_output": {"acc_runs": {"layout": "sparse_selected_rows"}},
        },
    )
    identity = pd.DataFrame({"public_row_id": [0], "year": [2005]})
    lca_input = LCAUncertaintyInput(
        identity=identity,
        fixed_values=pd.Series([2.0]).to_numpy(dtype=float),
        manifest=None,
        external_inputs=(),
        source_method_rows=pd.DataFrame(),
        active_sources=(),
        lca_type="io_lca",
    )
    summary_identity = pd.concat(
        [
            identity.assign(asr_metric="asr"),
            identity.assign(asr_metric="frequency_of_no_transgression"),
        ],
        ignore_index=True,
    )
    cumulative_identity = pd.DataFrame({"public_row_id": [0]})
    cumulative_summary_identity = pd.concat(
        [
            cumulative_identity.assign(asr_metric=ASR_CUMULATIVE_VALUE_METRIC),
            cumulative_identity.assign(
                asr_metric=ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC
            ),
        ],
        ignore_index=True,
    )
    plan = ASRUncertaintyPlan(
        identity=identity,
        summary_identity=summary_identity,
        summary_public_row_groups=(("0",),),
        cumulative_identity=cumulative_identity,
        cumulative_summary_identity=cumulative_summary_identity,
        cumulative_summary_public_row_groups=(("0",),),
        cumulative_public_row_groups=(("0",),),
        acc_positions=pd.Series([0]).to_numpy(dtype=int),
        lca_positions=pd.Series([0]).to_numpy(dtype=int),
        lca_unit_factors=pd.Series([1.0]).to_numpy(dtype=float),
        acc_manifest=acc_manifest,
        lca_input=lca_input,
        asr_run_layout="sparse_selected_rows",
        source_method_rows=pd.DataFrame(),
        active_sources=("acc::asocc::inter_method_uncertainty",),
    )
    paths = ASRUncertaintyRunPaths(
        run_root=tmp_path / "asr_run",
        public_row_identity=tmp_path / "asr_run" / "results" / "public_row_identity.csv",
        public_runs=tmp_path / "asr_run" / "results" / "asr_runs.csv",
        summary_stats_runs=tmp_path / "asr_run" / "results" / "summary_stats_runs.csv",
        cumulative_row_identity=tmp_path / "asr_run" / "results" / "cumulative_identity.csv",
        cumulative_runs=tmp_path / "asr_run" / "results" / "cumulative_asr_runs.csv",
        cumulative_summary_stats_runs=tmp_path
        / "asr_run"
        / "results"
        / "cumulative_summary_stats_runs.csv",
        results_readme=tmp_path / "asr_run" / "results" / "README.txt",
        source_methods=tmp_path / "asr_run" / "logs" / "source_methods.csv",
        sobol_indices=tmp_path / "asr_run" / "results" / "sobol" / "sobol_indices.csv",
        sobol_source_summary=tmp_path
        / "asr_run"
        / "results"
        / "sobol"
        / "sobol_source_summary.csv",
        sobol_readme=tmp_path / "asr_run" / "results" / "sobol" / "README_sobol.txt",
        scope_manifest=tmp_path / "asr_run" / "logs" / "scope_manifest.json",
    )
    runtime = normalize_uncertainty_request(
        family="asr",
        output_format="csv_compact",
        mc_parameters={
            "fixed": {"active": False},
            "convergence": {"active": True, "max_runs": 2, "stable_runs": 1},
        },
    )
    runtime = replace(runtime, batch_size=1)
    context = build_asr_manifest_context(
        base_args={"project_name": "demo"},
        runtime=runtime,
        plan=plan,
        sobol_status={"ran": False},
    )

    completed, convergence = write_asr_run_outputs(paths=paths, plan=plan, runtime=runtime)

    assert completed == 2
    assert convergence is not None
    assert convergence["reached"] is True
    assert pd.read_csv(paths.public_runs).empty
    cumulative_runs = pd.read_csv(paths.cumulative_runs)
    cumulative_summary = pd.read_csv(paths.cumulative_summary_stats_runs)
    assert cumulative_runs["run_index"].tolist() == [0, 1]
    assert bool(cumulative_runs["0"].isna().all())
    assert bool(cumulative_summary["mean"].isna().all())
    assert ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN in cumulative_summary.columns
    assert "component_inventory" not in context["deterministic_prerequisites"][0]


def test_asr_compact_cumulative_reuses_invariant_row_in_each_ssp_period(
    tmp_path: Path,
) -> None:
    acc_results = tmp_path / "acc_compact" / "results"
    acc_logs = tmp_path / "acc_compact" / "logs"
    acc_results.mkdir(parents=True)
    acc_logs.mkdir(parents=True)
    acc_identity_path = acc_results / "public_row_identity.csv"
    acc_runs_path = acc_results / "acc_runs.csv"
    identity = pd.DataFrame({"public_row_id": [0, 1, 2], "year": [2024, 2030, 2030]})
    write_uncertainty_table(path=acc_identity_path, output_format="csv_compact", frame=identity)
    with CompactRunMatrixWriter(path=acc_runs_path, output_format="csv_compact") as writer:
        writer.write_batch(
            run_indices=np.array([0, 1], dtype=np.int64),
            values=np.array(
                [
                    [10.0, 20.0, 40.0],
                    [10.0, 20.0, 40.0],
                ],
                dtype=np.float64,
            ),
            batch_index=0,
        )
    empty = acc_results / "empty.csv"
    empty.write_text("", encoding="utf-8")
    source_methods = acc_logs / "source_methods.csv"
    source_methods.write_text("source_component,source_name\nacc,demo\n", encoding="utf-8")
    acc_manifest = build_manifest(
        family="acc",
        mode="fixed",
        output_format="csv_compact",
        active_sources=(),
        artifacts={
            "scope_manifest": str(acc_logs / "scope_manifest.json"),
            "public_row_identity": str(acc_identity_path),
            "acc_runs": str(acc_runs_path),
            "summary_stats_runs": str(empty),
            "results_readme": str(empty),
            "source_methods": str(source_methods),
            "public_output": {"acc_runs": {"layout": "compact_run_matrix"}},
        },
    )
    lca_input = LCAUncertaintyInput(
        identity=identity,
        fixed_values=pd.Series([7.0, 20.0, 53.0]).to_numpy(dtype=float),
        manifest=None,
        external_inputs=(),
        source_method_rows=pd.DataFrame(),
        active_sources=(),
        lca_type="io_lca",
    )
    yearly_summary_identity = pd.concat(
        [
            identity.assign(
                asr_metric=ASR_VALUE_METRIC,
                **{ASR_SUMMARY_SCOPE_COLUMN: ASR_SUMMARY_SCOPE_PER_METHOD},
            ),
            identity.assign(
                asr_metric=ASR_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
                **{ASR_SUMMARY_SCOPE_COLUMN: ASR_SUMMARY_SCOPE_PER_METHOD},
            ),
        ],
        ignore_index=True,
    )
    cumulative_identity = pd.DataFrame(
        {
            "public_row_id": [0, 1],
            "asocc_ssp_scenario": ["SSP1", "SSP2"],
        }
    )
    cumulative_summary_identity = pd.concat(
        [
            cumulative_identity.assign(
                asr_metric=ASR_CUMULATIVE_VALUE_METRIC,
                **{ASR_SUMMARY_SCOPE_COLUMN: ASR_SUMMARY_SCOPE_PER_METHOD},
            ),
            pd.DataFrame(
                {
                    "public_row_id": [pd.NA],
                    "asocc_ssp_scenario": [pd.NA],
                    "asr_metric": [ASR_CUMULATIVE_VALUE_METRIC],
                    ASR_SUMMARY_SCOPE_COLUMN: [ASR_SUMMARY_SCOPE_INTER_METHOD],
                }
            ),
            cumulative_identity.assign(
                asr_metric=ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC,
                **{ASR_SUMMARY_SCOPE_COLUMN: ASR_SUMMARY_SCOPE_PER_METHOD},
            ),
            pd.DataFrame(
                {
                    "public_row_id": [pd.NA],
                    "asocc_ssp_scenario": [pd.NA],
                    "asr_metric": [ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC],
                    ASR_SUMMARY_SCOPE_COLUMN: [ASR_SUMMARY_SCOPE_INTER_METHOD],
                }
            ),
        ],
        ignore_index=True,
    )
    plan = ASRUncertaintyPlan(
        identity=identity,
        summary_identity=yearly_summary_identity,
        summary_public_row_groups=(("0",), ("1",), ("2",)),
        cumulative_identity=cumulative_identity,
        cumulative_summary_identity=cumulative_summary_identity,
        cumulative_summary_public_row_groups=(("0",), ("1",), ("0", "1")),
        cumulative_public_row_groups=(("0", "1"), ("0", "2")),
        acc_positions=pd.Series([0, 1, 2]).to_numpy(dtype=int),
        lca_positions=pd.Series([0, 1, 2]).to_numpy(dtype=int),
        lca_unit_factors=pd.Series([1.0, 1.0, 1.0]).to_numpy(dtype=float),
        acc_manifest=acc_manifest,
        lca_input=lca_input,
        asr_run_layout="compact_run_matrix",
        source_method_rows=pd.DataFrame(),
        active_sources=(),
    )
    paths = ASRUncertaintyRunPaths(
        run_root=tmp_path / "asr_run_compact",
        public_row_identity=tmp_path / "asr_run_compact" / "results" / "public_row_identity.csv",
        public_runs=tmp_path / "asr_run_compact" / "results" / "asr_runs.csv",
        summary_stats_runs=tmp_path / "asr_run_compact" / "results" / "summary_stats_runs.csv",
        cumulative_row_identity=tmp_path
        / "asr_run_compact"
        / "results"
        / "cumulative_identity.csv",
        cumulative_runs=tmp_path / "asr_run_compact" / "results" / "cumulative_asr_runs.csv",
        cumulative_summary_stats_runs=tmp_path
        / "asr_run_compact"
        / "results"
        / "cumulative_summary_stats_runs.csv",
        results_readme=tmp_path / "asr_run_compact" / "results" / "README.txt",
        source_methods=tmp_path / "asr_run_compact" / "logs" / "source_methods.csv",
        sobol_indices=tmp_path / "asr_run_compact" / "results" / "sobol" / "sobol_indices.csv",
        sobol_source_summary=tmp_path
        / "asr_run_compact"
        / "results"
        / "sobol"
        / "sobol_source_summary.csv",
        sobol_readme=tmp_path / "asr_run_compact" / "results" / "sobol" / "README_sobol.txt",
        scope_manifest=tmp_path / "asr_run_compact" / "logs" / "scope_manifest.json",
    )
    runtime = normalize_uncertainty_request(
        family="asr",
        output_format="csv_compact",
        mc_parameters={"fixed": {"active": True, "n_runs": 2}, "convergence": {"active": False}},
    )

    completed, convergence = write_asr_run_outputs(paths=paths, plan=plan, runtime=runtime)

    cumulative_runs = pd.read_csv(paths.cumulative_runs)
    cumulative_summary = pd.read_csv(paths.cumulative_summary_stats_runs)
    assert completed == 2
    assert convergence is None
    assert cumulative_runs.loc[:, ["0", "1"]].round(12).values.tolist() == [
        [0.9, 1.2],
        [0.9, 1.2],
    ]
    cumulative_asr = cumulative_summary.loc[
        cumulative_summary["asr_metric"].eq(ASR_CUMULATIVE_VALUE_METRIC)
    ]
    assert cumulative_asr["mean"].round(12).tolist() == [0.9, 1.2, 1.05]
    cumulative_frequency = cumulative_summary.loc[
        cumulative_summary["asr_metric"].eq(ASR_CUMULATIVE_FREQUENCY_OF_NO_TRANSGRESSION_METRIC)
    ]
    assert bool(cumulative_frequency["mean"].isna().all())
    assert cumulative_frequency[ASR_CUMULATIVE_FREQUENCY_VALUE_COLUMN].tolist() == [
        1.0,
        0.0,
        0.0,
    ]
    assert cumulative_summary[ASR_SUMMARY_SCOPE_COLUMN].unique().tolist() == [
        ASR_SUMMARY_SCOPE_PER_METHOD,
        ASR_SUMMARY_SCOPE_INTER_METHOD,
    ]


def test_asr_source_routing_rejects_invalid_public_config() -> None:
    routed = split_asr_uncertainty_config(
        {
            "asocc_uncertainty_sources": {"lcia_uncertainty": {"active": False}},
            "io_lca_uncertainty_sources": {
                "lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}}
            },
        }
    )
    assert "lcia_uncertainty" not in routed.acc_config
    assert routed.acc_config["projection_uncertainty"] == {}
    assert routed.acc_config["reference_year_uncertainty"] == {}
    assert routed.acc_config["inter_method_uncertainty"] == {}
    assert routed.acc_config["dynamic_ar6_cc_uncertainty"] == {}
    assert routed.lca_config == {"lcia_uncertainty": {"sector_cov_mapping": {"D": "Electricity"}}}

    with pytest.raises(ValueError):
        split_asr_uncertainty_config({"unknown_source": True})

    with pytest.raises(ValueError):
        split_asr_uncertainty_config({"io_lca_uncertainty_sources": "bad"})

    with pytest.raises(ValueError):
        split_asr_uncertainty_config(
            {"io_lca_uncertainty_sources": {"inter_method_uncertainty": {}}}
        )
