from pathlib import Path
import json

import pytest

from pyaesa import prepare_external_inputs
from pyaesa.external_inputs.asocc.deterministic.files import (
    load_external_deterministic_rows_with_source,
)
from pyaesa.external_inputs.asocc.monte_carlo.files import (
    materialize_external_monte_carlo_source,
    resolve_external_monte_carlo_source,
)
from pyaesa.external_inputs.asocc.schema.contracts import iter_external_method_selections
from pyaesa.external_inputs.lca.deterministic import load_external_lca_deterministic_rows
from pyaesa.external_inputs.lca.monte_carlo import (
    external_lca_values_for_runs,
    load_external_lca_monte_carlo_source,
)


def _assert_template_scaffold(report) -> None:
    assert report.external_asocc_root.exists()
    assert (report.external_asocc_root / "deterministic").exists()
    assert (report.external_asocc_root / "monte_carlo").exists()
    assert report.external_asocc_templates_dir.exists()
    assert report.external_lca_root.exists()
    assert report.external_lca_templates_dir.exists()
    asocc_readme = report.external_asocc_templates_dir / "README_external_asocc_templates.txt"
    lca_readme = report.external_lca_templates_dir / "README_external_lca_templates.txt"
    asocc_readme_text = asocc_readme.read_text(encoding="utf-8")
    lca_readme_text = lca_readme.read_text(encoding="utf-8")
    assert "AR(E)::UT(S)" in asocc_readme_text
    assert "dummy demonstration values" in asocc_readme_text
    assert "template__ef_3.1__ssp2.csv" in lca_readme_text
    assert "dummy demonstration values" in lca_readme_text

    assert (report.external_asocc_root / "deterministic" / "CO(S).csv").exists()
    assert (report.external_asocc_root / "deterministic" / "CO(S)__ssp2.csv").exists()
    assert (report.external_asocc_root / "deterministic" / "l1_AR(E)_l2_UT(S)__ef_3.1.csv").exists()
    assert (report.external_asocc_root / "monte_carlo" / "CO(S).csv").exists()
    assert (report.external_asocc_root / "monte_carlo" / "CO(S)" / "asocc_runs.csv").exists()
    assert (
        report.external_asocc_root / "monte_carlo" / "l1_AR(E)_l2_UT(S)__ef_3.1" / "asocc_runs.csv"
    ).exists()

    assert (report.external_lca_root / "deterministic" / "template__ef_3.1.csv").exists()
    assert (report.external_lca_root / "deterministic" / "template__ef_3.1__ssp2.csv").exists()
    assert (report.external_lca_root / "monte_carlo" / "template__ef_3.1.csv").exists()
    assert (report.external_lca_root / "monte_carlo" / "template__ef_3.1" / "lca_runs.csv").exists()


def test_prepare_external_inputs_project_scaffold(
    project_repo: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del project_repo
    report = prepare_external_inputs(project_name="external_inputs_public")
    assert str(report)
    assert report.metadata_path.exists()
    assert report.metadata_path == report.project_root / "prepare_external_inputs_log" / (
        "scope_manifest.json"
    )
    assert report.summary_log == report.project_root / "prepare_external_inputs_log" / "summary.log"
    assert report.summary_log.read_text(encoding="utf-8").strip()
    metadata = json.loads(report.metadata_path.read_text(encoding="utf-8"))
    assert metadata["function"] == "prepare_external_inputs"
    assert metadata["arguments"] == {"project_name": "external_inputs_public"}
    assert metadata["execution"]["status"] == "complete"
    assert metadata["artifacts"]["summary_log"] == str(report.summary_log)
    _assert_template_scaffold(report)
    output = capsys.readouterr().out
    assert "prepare_external_inputs" in output
    assert str(report.external_asocc_root) in output
    assert str(report.external_lca_root) in output
    assert "runnable examples" in output

    preserved_readme = report.external_asocc_templates_dir / "README_external_asocc_templates.txt"
    preserved_readme.write_text("local edits\n", encoding="utf-8")
    preserved_example = report.external_asocc_root / "deterministic" / "CO(S).csv"
    preserved_example.write_text("s_p,r_c,2019\nPaper,FR,1\n", encoding="utf-8")
    repeated_report = prepare_external_inputs(project_name="external_inputs_public")

    assert preserved_readme.read_text(encoding="utf-8") == "local edits\n"
    assert preserved_example.read_text(encoding="utf-8") == "s_p,r_c,2019\nPaper,FR,1\n"
    assert repeated_report.external_asocc_templates_dir == report.external_asocc_templates_dir
    user_file = report.external_lca_root / "deterministic" / "supplier__ef_3.1.csv"
    user_file.write_text("impact,impact_unit,2019\nGWP_100,kg CO2 eq,1\n", encoding="utf-8")

    second_report = prepare_external_inputs(project_name="external_inputs_public")

    assert second_report.external_asocc_templates_dir == report.external_asocc_templates_dir
    assert second_report.external_lca_templates_dir == report.external_lca_templates_dir
    assert user_file.read_text(encoding="utf-8").startswith("impact,impact_unit")


def test_prepare_external_inputs_runnable_example_shapes(project_repo: Path) -> None:
    del project_repo
    report = prepare_external_inputs(project_name="external_inputs_examples")

    ef_lca_base = report.external_lca_root / "deterministic" / "template__ef_3.1.csv"
    ef_lca_ssp = report.external_lca_root / "deterministic" / "template__ef_3.1__ssp2.csv"
    ef_lca_runs = report.external_lca_root / "monte_carlo" / "template__ef_3.1" / "lca_runs.csv"
    asocc_base = report.external_asocc_root / "deterministic" / "CO(S).csv"
    asocc_ssp = report.external_asocc_root / "deterministic" / "CO(S)__ssp2.csv"
    asocc_runs = report.external_asocc_root / "monte_carlo" / "CO(S)" / "asocc_runs.csv"

    assert ef_lca_base.read_text(encoding="utf-8").splitlines()[0].split(",") == [
        "s_p",
        "r_c",
        "impact",
        "impact_unit",
        "2019",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    ]
    assert "2030" in ef_lca_ssp.read_text(encoding="utf-8").splitlines()[0]
    assert "GWP_100" in ef_lca_base.read_text(encoding="utf-8")
    assert asocc_base.read_text(encoding="utf-8").startswith("s_p,r_c,2019")
    assert "2030" in asocc_ssp.read_text(encoding="utf-8").splitlines()[0]

    ef_lca_run_header = ef_lca_runs.read_text(encoding="utf-8").splitlines()[0].split(",")
    asocc_run_lines = asocc_runs.read_text(encoding="utf-8").splitlines()
    assert ef_lca_run_header[:2] == ["run_index", "0"]
    assert ef_lca_run_header[-1] == "191"
    assert len(asocc_run_lines) == 101
    assert asocc_run_lines[0].split(",")[-1] == "11"


def test_prepare_external_inputs_runnable_examples_load_through_contracts(
    project_repo: Path,
) -> None:
    del project_repo
    report = prepare_external_inputs(project_name="external_inputs_loadable_examples")
    years = list(range(2019, 2031))
    base_allocate_args = {
        "fu_code": "L2.c.b",
        "s_p": "Paper",
        "r_c": "FR",
        "ssp_scenario": "SSP2",
    }
    asocc_scenarios: dict[int, list[str | None]] = {
        year: [None] if year <= 2025 else ["SSP2"] for year in years
    }

    ef_lca_rows, ef_lca_paths = load_external_lca_deterministic_rows(
        proj_base=report.project_root,
        version_name="template",
        lcia_method="ef_3.1",
        years=years,
        ssp_scenario_options_by_year=None,
        base_allocate_args=base_allocate_args,
    )
    assert ef_lca_rows is not None
    assert ef_lca_rows.shape[0] == 192
    assert [path.name for path in ef_lca_paths] == [
        "template__ef_3.1.csv",
        "template__ef_3.1__ssp2.csv",
    ]

    ef_lca_monte_carlo = load_external_lca_monte_carlo_source(
        proj_base=report.project_root,
        version_name="template",
        lcia_method="ef_3.1",
        years=years,
        base_allocate_args=base_allocate_args,
    )
    assert ef_lca_monte_carlo is not None
    assert ef_lca_monte_carlo.identity.shape[0] == 192
    assert external_lca_values_for_runs(
        source=ef_lca_monte_carlo,
        run_indices=ef_lca_monte_carlo.run_indices,
    ).shape == (100, 192)

    co_selection = iter_external_method_selections(
        external_method={"one_step_methods": ["CO(S)"]},
        fu_code="L2.c.b",
    )[0]
    co_rows = load_external_deterministic_rows_with_source(
        proj_base=report.project_root,
        selection=co_selection,
        years=years,
        lcia_methods=None,
        ssp_scenario_options_by_year=asocc_scenarios,
    )
    assert co_rows is not None
    assert co_rows.rows.shape[0] == 12

    co_source = resolve_external_monte_carlo_source(
        proj_base=report.project_root,
        selection=co_selection,
        years=years,
        lcia_methods=None,
        ssp_scenario_options_by_year=asocc_scenarios,
    )
    assert co_source is not None
    co_monte_carlo = materialize_external_monte_carlo_source(source=co_source)
    assert co_monte_carlo.run_matrix.template.shape[0] == 12
    assert co_monte_carlo.run_matrix.values.shape == (100, 12)

    ar_ut_selection = iter_external_method_selections(
        external_method={"l1_l2_pairs": ["AR(E)::UT(S)"]},
        fu_code="L2.c.b",
    )[0]
    ar_ut_rows = load_external_deterministic_rows_with_source(
        proj_base=report.project_root,
        selection=ar_ut_selection,
        years=years,
        lcia_methods=["ef_3.1"],
        ssp_scenario_options_by_year=asocc_scenarios,
    )
    assert ar_ut_rows is not None
    assert ar_ut_rows.rows.shape[0] == 192

    ar_ut_source = resolve_external_monte_carlo_source(
        proj_base=report.project_root,
        selection=ar_ut_selection,
        years=years,
        lcia_methods=["ef_3.1"],
        ssp_scenario_options_by_year=asocc_scenarios,
    )
    assert ar_ut_source is not None
    ar_ut_monte_carlo = materialize_external_monte_carlo_source(source=ar_ut_source)
    assert ar_ut_monte_carlo.run_matrix.template.shape[0] == 192
    assert ar_ut_monte_carlo.run_matrix.values.shape == (100, 192)


def test_prepare_external_inputs_second_project_scaffold(
    project_repo: Path,
) -> None:
    del project_repo
    report = prepare_external_inputs(project_name="external_inputs_disagg")

    assert str(report)
    _assert_template_scaffold(report)


@pytest.mark.parametrize("project_name", [None, " "])
def test_prepare_external_inputs_rejects_invalid_project_name(project_name: str | None) -> None:
    with pytest.raises(
        ValueError,
    ):
        prepare_external_inputs(project_name=project_name)  # type: ignore[arg-type]
